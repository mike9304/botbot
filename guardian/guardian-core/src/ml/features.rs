//! EMBER-style feature extraction for ML classification.
//!
//! Extracts a high-dimensional feature vector from raw file bytes, inspired by the
//! EMBER (Endgame Malware BEnchmark for Research) feature set. The full PE feature
//! vector has ~2381 dimensions; non-PE files use a simpler ~626-dimension subset.
//!
//! Feature groups:
//! - Byte histogram (256): normalised byte-value frequency distribution
//! - Byte-entropy histogram (256): 2D histogram of (byte_value, local_entropy)
//! - String features (104): statistics about printable strings
//! - General file info (10): size, has_debug, virtual_size ratio, etc.
//! - PE header features (62): optional header fields, characteristics
//! - PE section features (255 max): per-section entropy, size, flags
//! - PE import features (1280 max): hashed import library+function names

use guardian_common::{FeatureVector, FileType};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Extract an EMBER-style feature vector from raw bytes.
pub fn extract_features(data: &[u8], file_type: FileType) -> FeatureVector {
    let mut numeric: Vec<f64> = Vec::with_capacity(2400);
    let mut categorical: Vec<String> = Vec::new();
    let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();

    // 1. Byte histogram (256 features)
    let byte_hist = byte_histogram(data);
    numeric.extend_from_slice(&byte_hist);

    // 2. Byte-entropy histogram (256 features)
    let be_hist = byte_entropy_histogram(data);
    numeric.extend_from_slice(&be_hist);

    // 3. String features (104 features)
    let (str_feats, str_cats) = string_features(data);
    numeric.extend_from_slice(&str_feats);
    categorical.extend(str_cats);

    // 4. General file info (10 features)
    let gen_info = general_file_info(data, file_type);
    numeric.extend_from_slice(&gen_info);

    metadata.insert(
        "feature_groups".to_string(),
        serde_json::json!({
            "byte_histogram": 256,
            "byte_entropy_histogram": 256,
            "string_features": 104,
            "general_info": 10,
        }),
    );

    // 5-7. PE-specific features (header, sections, imports)
    if file_type == FileType::PE {
        let (pe_header, pe_sections, pe_imports) = pe_features(data);
        let header_len = pe_header.len();
        let sections_len = pe_sections.len();
        let imports_len = pe_imports.len();

        numeric.extend(pe_header);
        numeric.extend(pe_sections);
        numeric.extend(pe_imports);

        metadata.insert(
            "pe_features".to_string(),
            serde_json::json!({
                "header": header_len,
                "sections": sections_len,
                "imports": imports_len,
            }),
        );
    }

    metadata.insert(
        "total_dimensions".to_string(),
        serde_json::json!(numeric.len()),
    );

    FeatureVector {
        numeric,
        categorical,
        metadata,
    }
}

// ---------------------------------------------------------------------------
// Byte histogram (256 features)
// ---------------------------------------------------------------------------

/// Normalised frequency of each byte value 0x00..0xFF.
fn byte_histogram(data: &[u8]) -> [f64; 256] {
    let mut counts = [0u64; 256];
    for &b in data {
        counts[b as usize] += 1;
    }
    let total = data.len().max(1) as f64;
    let mut hist = [0.0f64; 256];
    for i in 0..256 {
        hist[i] = counts[i] as f64 / total;
    }
    hist
}

// ---------------------------------------------------------------------------
// Byte-entropy histogram (256 features)
// ---------------------------------------------------------------------------

/// 2D histogram: for each byte, accumulate into an entropy bucket.
/// We use 16 entropy buckets (0..8 bits), then flatten to 256 features (16 * 16).
/// However, the canonical EMBER encoding uses 256 output features by hashing
/// (byte_value / 16, entropy_bucket) into a 16x16 grid and normalising.
fn byte_entropy_histogram(data: &[u8]) -> [f64; 256] {
    const WINDOW: usize = 2048;
    const ENTROPY_BUCKETS: usize = 16;
    const BYTE_BUCKETS: usize = 16;

    let mut grid = [[0u64; ENTROPY_BUCKETS]; BYTE_BUCKETS];

    if data.len() < WINDOW {
        // For very small files, compute single-window entropy.
        let ent = shannon_entropy(data);
        let ent_bucket = ((ent / 8.0) * (ENTROPY_BUCKETS - 1) as f64)
            .clamp(0.0, (ENTROPY_BUCKETS - 1) as f64) as usize;
        for &b in data {
            let byte_bucket = (b as usize) / 16;
            grid[byte_bucket][ent_bucket] += 1;
        }
    } else {
        // Sliding window entropy.
        let step = WINDOW / 4;
        let mut offset = 0;
        while offset + WINDOW <= data.len() {
            let window = &data[offset..offset + WINDOW];
            let ent = shannon_entropy(window);
            let ent_bucket = ((ent / 8.0) * (ENTROPY_BUCKETS - 1) as f64)
                .clamp(0.0, (ENTROPY_BUCKETS - 1) as f64) as usize;
            for &b in window {
                let byte_bucket = (b as usize) / 16;
                grid[byte_bucket][ent_bucket] += 1;
            }
            offset += step;
        }
    }

    let total: u64 = grid.iter().flat_map(|row| row.iter()).sum();
    let total = total.max(1) as f64;
    let mut out = [0.0f64; 256];
    for i in 0..BYTE_BUCKETS {
        for j in 0..ENTROPY_BUCKETS {
            out[i * ENTROPY_BUCKETS + j] = grid[i][j] as f64 / total;
        }
    }
    out
}

/// Shannon entropy of a byte slice in bits (0.0 .. 8.0).
pub fn shannon_entropy(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut counts = [0u64; 256];
    for &b in data {
        counts[b as usize] += 1;
    }
    let len = data.len() as f64;
    let mut entropy = 0.0f64;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / len;
            entropy -= p * p.log2();
        }
    }
    entropy
}

// ---------------------------------------------------------------------------
// String features (104 features)
// ---------------------------------------------------------------------------

/// Extract statistics about printable ASCII strings found in the binary.
/// Returns (104 numeric features, list of notable string categories).
fn string_features(data: &[u8]) -> ([f64; 104], Vec<String>) {
    let strings = extract_printable_strings(data, 5);
    let mut feats = [0.0f64; 104];
    let mut categories = Vec::new();

    // Basic counts
    let count = strings.len();
    let total_chars: usize = strings.iter().map(|s| s.len()).sum();

    feats[0] = count as f64;
    feats[1] = total_chars as f64;
    feats[2] = if count > 0 {
        total_chars as f64 / count as f64
    } else {
        0.0
    };

    // Length distribution (buckets: 5-10, 10-20, 20-50, 50-100, 100+)
    let mut len_buckets = [0u64; 5];
    for s in &strings {
        match s.len() {
            5..=9 => len_buckets[0] += 1,
            10..=19 => len_buckets[1] += 1,
            20..=49 => len_buckets[2] += 1,
            50..=99 => len_buckets[3] += 1,
            _ => len_buckets[4] += 1,
        }
    }
    for (i, &b) in len_buckets.iter().enumerate() {
        feats[3 + i] = b as f64;
    }

    // Character class ratios across all strings
    let mut n_alpha = 0u64;
    let mut n_digit = 0u64;
    let mut n_upper = 0u64;
    let mut n_lower = 0u64;
    let mut n_space = 0u64;
    let mut n_special = 0u64;

    for s in &strings {
        for ch in s.chars() {
            if ch.is_ascii_uppercase() {
                n_upper += 1;
                n_alpha += 1;
            } else if ch.is_ascii_lowercase() {
                n_lower += 1;
                n_alpha += 1;
            } else if ch.is_ascii_digit() {
                n_digit += 1;
            } else if ch.is_ascii_whitespace() {
                n_space += 1;
            } else {
                n_special += 1;
            }
        }
    }

    let total_c = total_chars.max(1) as f64;
    feats[8] = n_alpha as f64 / total_c;
    feats[9] = n_digit as f64 / total_c;
    feats[10] = n_upper as f64 / total_c;
    feats[11] = n_lower as f64 / total_c;
    feats[12] = n_space as f64 / total_c;
    feats[13] = n_special as f64 / total_c;

    // Keyword category hits — indicators of suspicious behaviour
    let keyword_groups: &[(&str, &[&str])] = &[
        ("url", &["http://", "https://", "ftp://", "www."]),
        (
            "registry",
            &[
                "HKEY_", "RegSetValue", "RegOpenKey", "RegCreateKey", "SOFTWARE\\",
            ],
        ),
        (
            "file_ops",
            &[
                "CreateFile",
                "DeleteFile",
                "WriteFile",
                "MoveFile",
                "CopyFile",
            ],
        ),
        (
            "process",
            &[
                "CreateProcess",
                "OpenProcess",
                "VirtualAlloc",
                "WriteProcessMemory",
                "CreateRemoteThread",
            ],
        ),
        (
            "crypto",
            &[
                "CryptEncrypt",
                "CryptDecrypt",
                "AES",
                "RSA",
                "encrypt",
                "decrypt",
            ],
        ),
        (
            "network",
            &[
                "WSAStartup",
                "connect",
                "socket",
                "send",
                "recv",
                "InternetOpen",
                "HttpSendRequest",
            ],
        ),
        (
            "privilege",
            &[
                "AdjustTokenPrivileges",
                "SeDebugPrivilege",
                "IsDebuggerPresent",
                "NtQueryInformationProcess",
            ],
        ),
        (
            "packing",
            &["UPX", "Themida", "VMProtect", "Enigma", "packed"],
        ),
        (
            "shell",
            &[
                "cmd.exe",
                "powershell",
                "bash",
                "/bin/sh",
                "wscript",
                "cscript",
            ],
        ),
        (
            "persistence",
            &[
                "CurrentVersion\\Run",
                "StartupFolder",
                "schtasks",
                "crontab",
                "systemd",
            ],
        ),
    ];

    for (i, (cat_name, keywords)) in keyword_groups.iter().enumerate() {
        let mut hits = 0u64;
        for s in &strings {
            let lower = s.to_ascii_lowercase();
            for kw in *keywords {
                if lower.contains(&kw.to_ascii_lowercase()) {
                    hits += 1;
                }
            }
        }
        feats[14 + i * 2] = hits as f64;
        feats[14 + i * 2 + 1] = if count > 0 {
            hits as f64 / count as f64
        } else {
            0.0
        };
        if hits > 0 {
            categories.push(cat_name.to_string());
        }
    }

    // Features 34..103: printable char frequency histogram (70 buckets).
    // We map printable ASCII (0x20..0x7E, 95 chars) into 70 buckets using modulo.
    let mut char_hist = [0u64; 70];
    for s in &strings {
        for &b in s.as_bytes() {
            if b >= 0x20 && b <= 0x7E {
                let bucket = (b - 0x20) as usize % 70;
                char_hist[bucket] += 1;
            }
        }
    }
    let char_total = char_hist.iter().sum::<u64>().max(1) as f64;
    for i in 0..70 {
        feats[34 + i] = char_hist[i] as f64 / char_total;
    }

    (feats, categories)
}

/// Extract printable ASCII strings of at least `min_len` bytes.
fn extract_printable_strings(data: &[u8], min_len: usize) -> Vec<String> {
    let mut strings = Vec::new();
    let mut current = Vec::new();

    for &b in data {
        if b.is_ascii_graphic() || b == b' ' {
            current.push(b);
        } else {
            if current.len() >= min_len {
                strings.push(String::from_utf8_lossy(&current).to_string());
            }
            current.clear();
        }
    }
    if current.len() >= min_len {
        strings.push(String::from_utf8_lossy(&current).to_string());
    }
    strings
}

// ---------------------------------------------------------------------------
// General file info (10 features)
// ---------------------------------------------------------------------------

/// 10 general file-level features.
fn general_file_info(data: &[u8], file_type: FileType) -> [f64; 10] {
    let mut feats = [0.0f64; 10];

    feats[0] = data.len() as f64;
    feats[1] = shannon_entropy(data);

    // File type one-hot encoding (7 types + unknown = 8, using indices 2..9)
    let type_idx = match file_type {
        FileType::PE => 0,
        FileType::ELF => 1,
        FileType::DEX => 2,
        FileType::MachO => 3,
        FileType::Script => 4,
        FileType::Document => 5,
        FileType::Archive => 6,
        FileType::Unknown => 7,
    };
    feats[2 + type_idx] = 1.0;

    feats
}

// ---------------------------------------------------------------------------
// PE-specific features
// ---------------------------------------------------------------------------

/// Extract PE-specific features. Returns (header_feats, section_feats, import_feats).
/// If parsing fails, returns empty vectors.
fn pe_features(data: &[u8]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    match goblin::pe::PE::parse(data) {
        Ok(pe) => {
            let header = pe_header_features(&pe, data);
            let sections = pe_section_features(&pe, data);
            let imports = pe_import_features(&pe);
            (header, sections, imports)
        }
        Err(_) => {
            // Return zero-padded feature vectors of expected sizes.
            (vec![0.0; 62], vec![0.0; 255], vec![0.0; 1280])
        }
    }
}

/// PE header features (62 dimensions).
fn pe_header_features(pe: &goblin::pe::PE, data: &[u8]) -> Vec<f64> {
    let mut feats = vec![0.0f64; 62];

    // COFF header fields
    if let Some(ref header) = pe.header.optional_header {
        let std = &header.standard_fields;
        let win = &header.windows_fields;

        feats[0] = std.magic as f64;
        feats[1] = std.major_linker_version as f64;
        feats[2] = std.minor_linker_version as f64;
        feats[3] = std.size_of_code as f64;
        feats[4] = std.size_of_initialized_data as f64;
        feats[5] = std.size_of_uninitialized_data as f64;
        feats[6] = std.address_of_entry_point as f64;
        feats[7] = std.base_of_code as f64;

        feats[8] = win.image_base as f64;
        feats[9] = win.section_alignment as f64;
        feats[10] = win.file_alignment as f64;
        feats[11] = win.major_operating_system_version as f64;
        feats[12] = win.minor_operating_system_version as f64;
        feats[13] = win.major_image_version as f64;
        feats[14] = win.minor_image_version as f64;
        feats[15] = win.major_subsystem_version as f64;
        feats[16] = win.minor_subsystem_version as f64;
        feats[17] = win.size_of_image as f64;
        feats[18] = win.size_of_headers as f64;
        feats[19] = win.check_sum as f64;
        feats[20] = win.subsystem as f64;
        feats[21] = win.dll_characteristics as f64;
        feats[22] = win.size_of_stack_reserve as f64;
        feats[23] = win.size_of_stack_commit as f64;
        feats[24] = win.size_of_heap_reserve as f64;
        feats[25] = win.size_of_heap_commit as f64;
        feats[26] = win.loader_flags as f64;
        feats[27] = win.number_of_rva_and_sizes as f64;
    }

    // COFF header characteristics
    let coff = &pe.header.coff_header;
    feats[28] = coff.machine as f64;
    feats[29] = coff.number_of_sections as f64;
    feats[30] = coff.time_date_stamp as f64;
    feats[31] = coff.pointer_to_symbol_table as f64;
    feats[32] = coff.number_of_symbol_table as f64;
    feats[33] = coff.size_of_optional_header as f64;
    feats[34] = coff.characteristics as f64;

    // Derived features
    feats[35] = pe.sections.len() as f64;
    feats[36] = if pe.is_64 { 1.0 } else { 0.0 };
    feats[37] = if pe.is_lib { 1.0 } else { 0.0 };

    // Has debug directory
    feats[38] = if pe.debug_data.is_some() { 1.0 } else { 0.0 };

    // Has TLS
    feats[39] = if pe.tls_data.is_some() { 1.0 } else { 0.0 };

    // Import / export counts
    let import_count = pe.imports.len();
    let export_count = pe.exports.len();
    feats[40] = import_count as f64;
    feats[41] = export_count as f64;

    // Unique library count
    let mut unique_libs = std::collections::HashSet::new();
    for imp in &pe.imports {
        unique_libs.insert(imp.dll.to_lowercase());
    }
    feats[42] = unique_libs.len() as f64;

    // Entry point anomaly: is EP outside any section?
    let ep = pe
        .header
        .optional_header
        .map(|h| h.standard_fields.address_of_entry_point as u64)
        .unwrap_or(0);
    let ep_in_section = pe.sections.iter().any(|s| {
        let start = s.virtual_address as u64;
        let end = start + s.virtual_size as u64;
        ep >= start && ep < end
    });
    feats[43] = if ep_in_section { 0.0 } else { 1.0 };

    // Virtual-to-raw size ratio
    let total_virtual: u64 = pe.sections.iter().map(|s| s.virtual_size as u64).sum();
    let total_raw: u64 = pe.sections.iter().map(|s| s.size_of_raw_data as u64).sum();
    feats[44] = if total_raw > 0 {
        total_virtual as f64 / total_raw as f64
    } else {
        0.0
    };

    // File size vs image size ratio
    let image_size = pe
        .header
        .optional_header
        .map(|h| h.windows_fields.size_of_image as f64)
        .unwrap_or(0.0);
    feats[45] = if image_size > 0.0 {
        data.len() as f64 / image_size
    } else {
        0.0
    };

    // Features 46..61 reserved for future PE characteristics bit decomposition.
    let chars = coff.characteristics;
    for bit in 0..16 {
        feats[46 + bit] = if chars & (1 << bit) != 0 { 1.0 } else { 0.0 };
    }

    feats
}

/// PE section features (up to 255 dimensions: 15 features * 17 sections max).
fn pe_section_features(pe: &goblin::pe::PE, data: &[u8]) -> Vec<f64> {
    const MAX_SECTIONS: usize = 17;
    const FEATURES_PER_SECTION: usize = 15;
    let mut feats = vec![0.0f64; MAX_SECTIONS * FEATURES_PER_SECTION];

    for (i, section) in pe.sections.iter().take(MAX_SECTIONS).enumerate() {
        let base = i * FEATURES_PER_SECTION;
        let raw_offset = section.pointer_to_raw_data as usize;
        let raw_size = section.size_of_raw_data as usize;

        feats[base] = section.virtual_size as f64;
        feats[base + 1] = section.virtual_address as f64;
        feats[base + 2] = raw_size as f64;
        feats[base + 3] = raw_offset as f64;
        feats[base + 4] = section.characteristics as f64;

        // Section entropy
        let section_data = if raw_offset + raw_size <= data.len() {
            &data[raw_offset..raw_offset + raw_size]
        } else if raw_offset < data.len() {
            &data[raw_offset..]
        } else {
            &[]
        };
        feats[base + 5] = shannon_entropy(section_data);

        // Size ratios
        feats[base + 6] = if section.size_of_raw_data > 0 {
            section.virtual_size as f64 / section.size_of_raw_data as f64
        } else {
            0.0
        };

        // Characteristic flags decomposition
        let ch = section.characteristics;
        feats[base + 7] = if ch & 0x00000020 != 0 { 1.0 } else { 0.0 }; // IMAGE_SCN_CNT_CODE
        feats[base + 8] = if ch & 0x00000040 != 0 { 1.0 } else { 0.0 }; // INITIALIZED_DATA
        feats[base + 9] = if ch & 0x00000080 != 0 { 1.0 } else { 0.0 }; // UNINITIALIZED_DATA
        feats[base + 10] = if ch & 0x20000000 != 0 { 1.0 } else { 0.0 }; // EXECUTE
        feats[base + 11] = if ch & 0x40000000 != 0 { 1.0 } else { 0.0 }; // READ
        feats[base + 12] = if ch & 0x80000000 != 0 { 1.0 } else { 0.0 }; // WRITE

        // Anomaly: writable + executable
        feats[base + 13] = if ch & 0x20000000 != 0 && ch & 0x80000000 != 0 {
            1.0
        } else {
            0.0
        };

        // Zero raw size but non-zero virtual size (common in packed files)
        feats[base + 14] = if section.size_of_raw_data == 0 && section.virtual_size > 0 {
            1.0
        } else {
            0.0
        };
    }

    feats
}

/// PE import features (up to 1280 dimensions).
/// Uses a hashing trick to map (library, function) pairs into a fixed-size vector.
fn pe_import_features(pe: &goblin::pe::PE) -> Vec<f64> {
    const IMPORT_DIM: usize = 1280;
    let mut feats = vec![0.0f64; IMPORT_DIM];

    for imp in &pe.imports {
        let key = format!("{}:{}", imp.dll.to_lowercase(), imp.name);
        let hash = simple_hash(&key) % IMPORT_DIM;
        feats[hash] += 1.0;
    }

    // Normalise
    let max_val = feats.iter().cloned().fold(0.0f64, f64::max);
    if max_val > 0.0 {
        for f in feats.iter_mut() {
            *f /= max_val;
        }
    }

    feats
}

/// Simple deterministic hash for feature hashing.
fn simple_hash(s: &str) -> usize {
    let mut h: u64 = 5381;
    for b in s.bytes() {
        h = h.wrapping_mul(33).wrapping_add(b as u64);
    }
    h as usize
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_histogram_uniform() {
        let data: Vec<u8> = (0..=255).collect();
        let hist = byte_histogram(&data);
        for &v in hist.iter() {
            assert!((v - 1.0 / 256.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_byte_histogram_single_value() {
        let data = vec![0x41u8; 100];
        let hist = byte_histogram(&data);
        assert!((hist[0x41] - 1.0).abs() < 1e-9);
        assert!((hist[0x00]).abs() < 1e-9);
    }

    #[test]
    fn test_shannon_entropy_empty() {
        assert_eq!(shannon_entropy(&[]), 0.0);
    }

    #[test]
    fn test_shannon_entropy_uniform() {
        let data: Vec<u8> = (0..=255).collect();
        let ent = shannon_entropy(&data);
        assert!((ent - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_shannon_entropy_single() {
        let data = vec![0xAA; 1000];
        let ent = shannon_entropy(&data);
        assert!(ent.abs() < 1e-9);
    }

    #[test]
    fn test_extract_printable_strings() {
        let data = b"hello\x00world\x00\x01\x02ab\x00longstring_here\x00";
        let strings = extract_printable_strings(data, 5);
        assert_eq!(strings, vec!["hello", "world", "longstring_here"]);
    }

    #[test]
    fn test_string_features_dimensions() {
        let data = b"This is a test string with some content\x00\x00more\x00";
        let (feats, _cats) = string_features(data);
        assert_eq!(feats.len(), 104);
    }

    #[test]
    fn test_general_file_info_pe_type() {
        let feats = general_file_info(&[0x4D, 0x5A, 0, 0], FileType::PE);
        assert_eq!(feats.len(), 10);
        // PE one-hot at index 2
        assert_eq!(feats[2], 1.0);
        assert_eq!(feats[3], 0.0);
    }

    #[test]
    fn test_extract_features_non_pe() {
        let data = vec![0x7F, 0x45, 0x4C, 0x46]; // ELF magic
        let fv = extract_features(&data, FileType::ELF);
        // 256 + 256 + 104 + 10 = 626 for non-PE
        assert_eq!(fv.numeric.len(), 626);
    }

    #[test]
    fn test_byte_entropy_histogram_dimensions() {
        let data = vec![0u8; 4096];
        let hist = byte_entropy_histogram(&data);
        assert_eq!(hist.len(), 256);
        let sum: f64 = hist.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_simple_hash_deterministic() {
        assert_eq!(simple_hash("test"), simple_hash("test"));
        assert_ne!(simple_hash("foo"), simple_hash("bar"));
    }

    #[test]
    fn test_pe_features_invalid_data() {
        // Non-PE data should return zero-padded vectors.
        let (header, sections, imports) = pe_features(b"not a PE file at all");
        assert_eq!(header.len(), 62);
        assert_eq!(sections.len(), 255);
        assert_eq!(imports.len(), 1280);
        assert!(header.iter().all(|&v| v == 0.0));
    }
}
