//! Structural anomaly detector for PE, ELF, and DEX files.
//!
//! Analyzes the structure of binary files to detect anomalies commonly seen in
//! malware, such as unusual section names, missing imports, entry point outside
//! the .text section, TLS callbacks, oversized overlays, and more.

use guardian_common::{Detection, DetectionEngine, FileType, Severity};
use std::collections::HashMap;

/// Results of structural analysis.
#[derive(Debug, Clone)]
pub struct StructuralResult {
    /// Total risk score from structural anomalies.
    pub score: u32,
    /// Individual anomaly findings.
    pub anomalies: Vec<StructuralAnomaly>,
}

/// A single structural anomaly detected in a binary.
#[derive(Debug, Clone)]
pub struct StructuralAnomaly {
    /// Short identifier for this anomaly type.
    pub id: &'static str,
    /// Human-readable description.
    pub description: String,
    /// Risk score contribution.
    pub score: u32,
    /// Severity level.
    pub severity: Severity,
}

/// Analyze structural anomalies based on file type.
pub fn analyze_structure(data: &[u8], file_type: FileType) -> StructuralResult {
    match file_type {
        FileType::PE => analyze_pe(data),
        FileType::ELF => analyze_elf(data),
        FileType::DEX => analyze_dex(data),
        _ => StructuralResult {
            score: 0,
            anomalies: Vec::new(),
        },
    }
}

/// Convert structural results into Detection objects.
pub fn to_detections(result: &StructuralResult) -> Vec<Detection> {
    result
        .anomalies
        .iter()
        .map(|a| {
            let mut metadata = HashMap::new();
            metadata.insert("anomaly_id".to_string(), a.id.to_string());
            metadata.insert("score".to_string(), a.score.to_string());

            Detection {
                engine: DetectionEngine::Heuristic,
                rule_name: format!("HEUR:Struct/{}", a.id),
                description: a.description.clone(),
                severity: a.severity,
                metadata,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// PE analysis
// ---------------------------------------------------------------------------

/// Analyze structural anomalies in a PE (Portable Executable) file.
fn analyze_pe(data: &[u8]) -> StructuralResult {
    let mut anomalies = Vec::new();
    let mut score = 0u32;

    // Need at least a DOS header + PE signature offset.
    if data.len() < 64 {
        return StructuralResult { score: 0, anomalies };
    }

    // Read PE signature offset from e_lfanew (offset 0x3C, 4 bytes LE).
    let e_lfanew = u32::from_le_bytes(
        data[0x3C..0x40].try_into().unwrap_or([0; 4]),
    ) as usize;

    // Validate PE signature.
    if e_lfanew + 4 > data.len() {
        return StructuralResult { score: 0, anomalies };
    }

    let pe_sig = &data[e_lfanew..e_lfanew + 4];
    if pe_sig != b"PE\0\0" {
        return StructuralResult { score: 0, anomalies };
    }

    // COFF header starts right after PE signature.
    let coff_offset = e_lfanew + 4;
    if coff_offset + 20 > data.len() {
        return StructuralResult { score: 0, anomalies };
    }

    let num_sections = u16::from_le_bytes(
        data[coff_offset + 2..coff_offset + 4]
            .try_into()
            .unwrap_or([0; 2]),
    ) as usize;

    let optional_header_size = u16::from_le_bytes(
        data[coff_offset + 16..coff_offset + 18]
            .try_into()
            .unwrap_or([0; 2]),
    ) as usize;

    // Optional header starts after COFF header.
    let optional_offset = coff_offset + 20;
    if optional_offset + optional_header_size > data.len() {
        return StructuralResult { score: 0, anomalies };
    }

    // Check if PE32 (0x10B) or PE32+ (0x20B).
    let magic = u16::from_le_bytes(
        data[optional_offset..optional_offset + 2]
            .try_into()
            .unwrap_or([0; 2]),
    );
    let is_pe32_plus = magic == 0x20B;

    // Entry point RVA.
    let entry_point_rva = if optional_offset + 20 <= data.len() {
        u32::from_le_bytes(
            data[optional_offset + 16..optional_offset + 20]
                .try_into()
                .unwrap_or([0; 4]),
        )
    } else {
        0
    };

    // Number of data directories.
    let data_dir_offset = if is_pe32_plus {
        optional_offset + 108
    } else {
        optional_offset + 92
    };
    let num_data_dirs = if data_dir_offset + 4 <= data.len() {
        u32::from_le_bytes(
            data[data_dir_offset..data_dir_offset + 4]
                .try_into()
                .unwrap_or([0; 4]),
        ) as usize
    } else {
        0
    };

    // Import directory (index 1) — check if imports exist.
    let import_dir_offset = data_dir_offset + 4 + 8; // skip NumberOfRvaAndSizes + Export entry
    if num_data_dirs >= 2 && import_dir_offset + 8 <= data.len() {
        let import_rva = u32::from_le_bytes(
            data[import_dir_offset..import_dir_offset + 4]
                .try_into()
                .unwrap_or([0; 4]),
        );
        let import_size = u32::from_le_bytes(
            data[import_dir_offset + 4..import_dir_offset + 8]
                .try_into()
                .unwrap_or([0; 4]),
        );
        if import_rva == 0 && import_size == 0 {
            let a = StructuralAnomaly {
                id: "PE_NO_IMPORTS",
                description: "PE file has no import directory — may be packed or use dynamic \
                              API resolution"
                    .to_string(),
                score: 5,
                severity: Severity::Medium,
            };
            score += a.score;
            anomalies.push(a);
        }
    }

    // TLS directory (index 9) — TLS callbacks can run before entry point.
    let tls_dir_offset = data_dir_offset + 4 + 9 * 8;
    if num_data_dirs >= 10 && tls_dir_offset + 8 <= data.len() {
        let tls_rva = u32::from_le_bytes(
            data[tls_dir_offset..tls_dir_offset + 4]
                .try_into()
                .unwrap_or([0; 4]),
        );
        let tls_size = u32::from_le_bytes(
            data[tls_dir_offset + 4..tls_dir_offset + 8]
                .try_into()
                .unwrap_or([0; 4]),
        );
        if tls_rva != 0 && tls_size != 0 {
            let a = StructuralAnomaly {
                id: "PE_TLS_CALLBACK",
                description: "PE file has TLS callbacks — code runs before the entry point, \
                              commonly used by packers and anti-debug"
                    .to_string(),
                score: 3,
                severity: Severity::Medium,
            };
            score += a.score;
            anomalies.push(a);
        }
    }

    // Parse section headers.
    let section_table_offset = optional_offset + optional_header_size;
    let mut text_section: Option<(u32, u32)> = None; // (VirtualAddress, VirtualSize)
    let mut has_suspicious_section_names = false;

    let suspicious_section_names: &[&[u8]] = &[
        b"UPX0\0\0\0\0",
        b"UPX1\0\0\0\0",
        b".ndata\0\0",
        b".aspack\0",
        b".adata\0\0",
        b".packed\0",
        b".petite\0",
        b".yP\0\0\0\0\0",
        b".themida",
    ];

    for i in 0..num_sections {
        let sec_offset = section_table_offset + i * 40;
        if sec_offset + 40 > data.len() {
            break;
        }

        let name = &data[sec_offset..sec_offset + 8];
        let virtual_size = u32::from_le_bytes(
            data[sec_offset + 8..sec_offset + 12]
                .try_into()
                .unwrap_or([0; 4]),
        );
        let virtual_address = u32::from_le_bytes(
            data[sec_offset + 12..sec_offset + 16]
                .try_into()
                .unwrap_or([0; 4]),
        );

        // Track .text section for entry point check.
        if name.starts_with(b".text") {
            text_section = Some((virtual_address, virtual_size));
        }

        // Check for suspicious packer section names.
        for &sus_name in suspicious_section_names {
            if name == &sus_name[..8.min(sus_name.len())] {
                has_suspicious_section_names = true;
                break;
            }
        }
    }

    // Anomaly: entry point outside .text section.
    if let Some((text_va, text_vs)) = text_section {
        if entry_point_rva != 0
            && (entry_point_rva < text_va || entry_point_rva >= text_va + text_vs)
        {
            let a = StructuralAnomaly {
                id: "PE_EP_OUTSIDE_TEXT",
                description: format!(
                    "Entry point RVA 0x{:X} is outside .text section (0x{:X}..0x{:X}) — \
                     may indicate packing or code injection",
                    entry_point_rva,
                    text_va,
                    text_va + text_vs
                ),
                score: 8,
                severity: Severity::High,
            };
            score += a.score;
            anomalies.push(a);
        }
    }

    // Anomaly: suspicious section names (packers).
    if has_suspicious_section_names {
        let a = StructuralAnomaly {
            id: "PE_PACKER_SECTIONS",
            description: "Section names associated with known packers (UPX, ASPack, Themida, \
                          etc.)"
                .to_string(),
            score: 3,
            severity: Severity::Low,
        };
        score += a.score;
        anomalies.push(a);
    }

    // Anomaly: large overlay (data appended after all sections).
    if num_sections > 0 {
        let last_sec_offset = section_table_offset + (num_sections - 1) * 40;
        if last_sec_offset + 24 <= data.len() {
            let raw_data_offset = u32::from_le_bytes(
                data[last_sec_offset + 20..last_sec_offset + 24]
                    .try_into()
                    .unwrap_or([0; 4]),
            ) as usize;
            let raw_data_size = u32::from_le_bytes(
                data[last_sec_offset + 16..last_sec_offset + 20]
                    .try_into()
                    .unwrap_or([0; 4]),
            ) as usize;

            let section_end = raw_data_offset.saturating_add(raw_data_size);
            if data.len() > section_end {
                let overlay_size = data.len() - section_end;
                let overlay_ratio =
                    overlay_size as f64 / data.len().max(1) as f64;

                if overlay_ratio > 0.25 {
                    let a = StructuralAnomaly {
                        id: "PE_LARGE_OVERLAY",
                        description: format!(
                            "Large overlay detected: {} bytes ({:.1}% of file) — \
                             may contain hidden payload",
                            overlay_size,
                            overlay_ratio * 100.0
                        ),
                        score: 2,
                        severity: Severity::Low,
                    };
                    score += a.score;
                    anomalies.push(a);
                }
            }
        }
    }

    StructuralResult { score, anomalies }
}

// ---------------------------------------------------------------------------
// ELF analysis
// ---------------------------------------------------------------------------

/// Analyze structural anomalies in an ELF binary.
fn analyze_elf(data: &[u8]) -> StructuralResult {
    let mut anomalies = Vec::new();
    let mut score = 0u32;

    // Minimum ELF header size.
    if data.len() < 64 {
        return StructuralResult { score: 0, anomalies };
    }

    // Verify ELF magic.
    if &data[0..4] != b"\x7fELF" {
        return StructuralResult { score: 0, anomalies };
    }

    let is_64bit = data[4] == 2;

    // Section header info.
    let (sh_offset, sh_entsize, sh_num, sh_strndx) = if is_64bit {
        if data.len() < 64 {
            return StructuralResult { score: 0, anomalies };
        }
        let sh_off = u64::from_le_bytes(
            data[0x28..0x30].try_into().unwrap_or([0; 8]),
        ) as usize;
        let sh_entsz = u16::from_le_bytes(
            data[0x3A..0x3C].try_into().unwrap_or([0; 2]),
        ) as usize;
        let sh_n = u16::from_le_bytes(
            data[0x3C..0x3E].try_into().unwrap_or([0; 2]),
        ) as usize;
        let sh_str = u16::from_le_bytes(
            data[0x3E..0x40].try_into().unwrap_or([0; 2]),
        ) as usize;
        (sh_off, sh_entsz, sh_n, sh_str)
    } else {
        if data.len() < 52 {
            return StructuralResult { score: 0, anomalies };
        }
        let sh_off = u32::from_le_bytes(
            data[0x20..0x24].try_into().unwrap_or([0; 4]),
        ) as usize;
        let sh_entsz = u16::from_le_bytes(
            data[0x2E..0x30].try_into().unwrap_or([0; 2]),
        ) as usize;
        let sh_n = u16::from_le_bytes(
            data[0x30..0x32].try_into().unwrap_or([0; 2]),
        ) as usize;
        let sh_str = u16::from_le_bytes(
            data[0x32..0x34].try_into().unwrap_or([0; 2]),
        ) as usize;
        (sh_off, sh_entsz, sh_n, sh_str)
    };

    // Check for stripped binary (no section headers or empty string table).
    let is_stripped = sh_num == 0 || sh_offset == 0;

    // Check for LD_PRELOAD reference in the data.
    let has_ld_preload = find_bytes(data, b"LD_PRELOAD");

    if has_ld_preload {
        let a = StructuralAnomaly {
            id: "ELF_LD_PRELOAD",
            description: "Binary references LD_PRELOAD — commonly used for library injection \
                          and rootkits"
                .to_string(),
            score: 7,
            severity: Severity::High,
        };
        score += a.score;
        anomalies.push(a);
    }

    // Check for stripped binary combined with high overall entropy hint.
    // We do a quick entropy estimate on the first 8KB.
    if is_stripped {
        let sample = &data[..data.len().min(8192)];
        let ent = crate::heuristic::entropy::shannon_entropy(sample);
        if ent > 7.0 {
            let a = StructuralAnomaly {
                id: "ELF_STRIPPED_HIGH_ENTROPY",
                description: format!(
                    "ELF binary is stripped and has high entropy ({:.2}) — \
                     likely packed or obfuscated",
                    ent
                ),
                score: 5,
                severity: Severity::Medium,
            };
            score += a.score;
            anomalies.push(a);
        }
    }

    // Parse section names for non-standard sections.
    let non_standard_sections = check_elf_section_names(
        data, sh_offset, sh_entsize, sh_num, sh_strndx, is_64bit,
    );
    if !non_standard_sections.is_empty() {
        let a = StructuralAnomaly {
            id: "ELF_NONSTANDARD_SECTIONS",
            description: format!(
                "Non-standard section names detected: {} — may indicate packing or \
                 custom loader",
                non_standard_sections.join(", ")
            ),
            score: 3,
            severity: Severity::Low,
        };
        score += a.score;
        anomalies.push(a);
    }

    // Check for ptrace anti-debug reference.
    if find_bytes(data, b"ptrace") && find_bytes(data, b"PTRACE_TRACEME") {
        let a = StructuralAnomaly {
            id: "ELF_ANTIDEBUG_PTRACE",
            description: "References PTRACE_TRACEME — anti-debugging technique".to_string(),
            score: 4,
            severity: Severity::Medium,
        };
        score += a.score;
        anomalies.push(a);
    }

    StructuralResult { score, anomalies }
}

/// Standard ELF section names that are expected.
const STANDARD_ELF_SECTIONS: &[&str] = &[
    "", ".text", ".data", ".bss", ".rodata", ".comment", ".note",
    ".symtab", ".strtab", ".shstrtab", ".dynsym", ".dynstr",
    ".dynamic", ".rel", ".rela", ".plt", ".got", ".got.plt",
    ".init", ".fini", ".init_array", ".fini_array", ".interp",
    ".hash", ".gnu.hash", ".gnu.version", ".gnu.version_r",
    ".note.gnu.build-id", ".note.ABI-tag", ".eh_frame",
    ".eh_frame_hdr", ".gcc_except_table", ".tbss", ".tdata",
    ".ctors", ".dtors", ".jcr", ".debug",
];

/// Check ELF section names for non-standard entries.
fn check_elf_section_names(
    data: &[u8],
    sh_offset: usize,
    sh_entsize: usize,
    sh_num: usize,
    sh_strndx: usize,
    is_64bit: bool,
) -> Vec<String> {
    let mut non_standard = Vec::new();

    if sh_entsize == 0 || sh_num == 0 || sh_strndx >= sh_num {
        return non_standard;
    }

    // Read string table section header to find strtab offset.
    let strtab_sh_offset = sh_offset + sh_strndx * sh_entsize;
    let strtab_file_offset = if is_64bit {
        if strtab_sh_offset + 64 > data.len() {
            return non_standard;
        }
        u64::from_le_bytes(
            data[strtab_sh_offset + 24..strtab_sh_offset + 32]
                .try_into()
                .unwrap_or([0; 8]),
        ) as usize
    } else {
        if strtab_sh_offset + 40 > data.len() {
            return non_standard;
        }
        u32::from_le_bytes(
            data[strtab_sh_offset + 16..strtab_sh_offset + 20]
                .try_into()
                .unwrap_or([0; 4]),
        ) as usize
    };

    // Iterate section headers and check names.
    for i in 0..sh_num {
        let sec_offset = sh_offset + i * sh_entsize;
        if sec_offset + 4 > data.len() {
            break;
        }
        let name_offset = u32::from_le_bytes(
            data[sec_offset..sec_offset + 4]
                .try_into()
                .unwrap_or([0; 4]),
        ) as usize;

        let str_start = strtab_file_offset + name_offset;
        if str_start >= data.len() {
            continue;
        }

        // Read null-terminated string.
        let name_end = data[str_start..]
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(0);
        if name_end == 0 {
            continue;
        }

        if let Ok(name) = std::str::from_utf8(&data[str_start..str_start + name_end]) {
            let is_standard = STANDARD_ELF_SECTIONS
                .iter()
                .any(|&s| name == s || name.starts_with(s));

            if !is_standard && !name.starts_with(".note") && !name.starts_with(".debug") {
                non_standard.push(name.to_string());
            }
        }
    }

    non_standard
}

// ---------------------------------------------------------------------------
// DEX analysis
// ---------------------------------------------------------------------------

/// Analyze structural anomalies in an Android DEX file.
fn analyze_dex(data: &[u8]) -> StructuralResult {
    let mut anomalies = Vec::new();
    let mut score = 0u32;

    // DEX header is 112 bytes.
    if data.len() < 112 {
        return StructuralResult { score: 0, anomalies };
    }

    // Verify DEX magic.
    if &data[0..4] != b"dex\n" {
        return StructuralResult { score: 0, anomalies };
    }

    // --- Check for dangerous Android permissions ---
    let dangerous_permissions: &[&[u8]] = &[
        b"android.permission.READ_SMS",
        b"android.permission.SEND_SMS",
        b"android.permission.RECEIVE_SMS",
        b"android.permission.READ_CONTACTS",
        b"android.permission.CAMERA",
        b"android.permission.RECORD_AUDIO",
        b"android.permission.ACCESS_FINE_LOCATION",
        b"android.permission.READ_PHONE_STATE",
        b"android.permission.CALL_PHONE",
        b"android.permission.WRITE_EXTERNAL_STORAGE",
        b"android.permission.READ_CALL_LOG",
        b"android.permission.WRITE_CALL_LOG",
        b"android.permission.INSTALL_PACKAGES",
        b"android.permission.SYSTEM_ALERT_WINDOW",
        b"android.permission.BIND_ACCESSIBILITY_SERVICE",
        b"android.permission.BIND_DEVICE_ADMIN",
        b"android.permission.REQUEST_INSTALL_PACKAGES",
    ];

    let mut perm_count = 0;
    for perm in dangerous_permissions {
        if find_bytes(data, perm) {
            perm_count += 1;
        }
    }

    if perm_count > 10 {
        let a = StructuralAnomaly {
            id: "DEX_EXCESSIVE_PERMISSIONS",
            description: format!(
                "DEX references {} dangerous permissions (>10) — may be spyware or \
                 over-privileged",
                perm_count
            ),
            score: 6,
            severity: Severity::High,
        };
        score += a.score;
        anomalies.push(a);
    } else if perm_count > 5 {
        let a = StructuralAnomaly {
            id: "DEX_MANY_PERMISSIONS",
            description: format!(
                "DEX references {} dangerous permissions — elevated concern",
                perm_count
            ),
            score: 3,
            severity: Severity::Medium,
        };
        score += a.score;
        anomalies.push(a);
    }

    // --- Check for obfuscated class/method names ---
    // Obfuscated names tend to be very short single-letter or use unusual
    // character patterns. We scan the string pool for indicators.
    let obfuscation_indicators: &[&[u8]] = &[
        b"\x00a\x00",  // Single-char class names in string pool.
        b"\x00b\x00",
        b"\x00c\x00",
        b"\x00aa\x00",
        b"\x00ab\x00",
    ];

    let mut obf_hits = 0;
    for &ind in obfuscation_indicators {
        if find_bytes(data, ind) {
            obf_hits += 1;
        }
    }

    if obf_hits >= 3 {
        let a = StructuralAnomaly {
            id: "DEX_OBFUSCATED_NAMES",
            description: "Class/method names appear heavily obfuscated (many single-letter \
                          identifiers)"
                .to_string(),
            score: 4,
            severity: Severity::Medium,
        };
        score += a.score;
        anomalies.push(a);
    }

    // --- Check for JNI native method loading ---
    if find_bytes(data, b"JNI_OnLoad") || find_bytes(data, b"RegisterNatives") {
        let a = StructuralAnomaly {
            id: "DEX_JNI_NATIVE",
            description: "DEX loads native JNI methods — native code may contain hidden \
                          malicious logic"
                .to_string(),
            score: 3,
            severity: Severity::Low,
        };
        score += a.score;
        anomalies.push(a);
    }

    // --- Check for dynamic class loading ---
    let dex_loader_strings: &[&[u8]] = &[
        b"DexClassLoader",
        b"PathClassLoader",
        b"InMemoryDexClassLoader",
        b"dalvik.system.DexClassLoader",
    ];

    let has_dynamic_loading = dex_loader_strings.iter().any(|s| find_bytes(data, s));

    if has_dynamic_loading {
        let a = StructuralAnomaly {
            id: "DEX_DYNAMIC_LOADING",
            description: "Uses DexClassLoader or similar for dynamic code loading — can load \
                          malicious payloads at runtime"
                .to_string(),
            score: 5,
            severity: Severity::Medium,
        };
        score += a.score;
        anomalies.push(a);
    }

    // --- Check for reflection usage ---
    if find_bytes(data, b"java.lang.reflect") || find_bytes(data, b"getDeclaredMethod") {
        let a = StructuralAnomaly {
            id: "DEX_REFLECTION",
            description: "Heavy use of Java reflection — can be used to invoke hidden APIs"
                .to_string(),
            score: 2,
            severity: Severity::Low,
        };
        score += a.score;
        anomalies.push(a);
    }

    // --- Check for crypto/data exfiltration indicators ---
    if find_bytes(data, b"javax.crypto") && find_bytes(data, b"HttpURLConnection") {
        let a = StructuralAnomaly {
            id: "DEX_CRYPTO_NETWORK",
            description: "Combines crypto operations with network connectivity — may indicate \
                          data exfiltration or ransomware"
                .to_string(),
            score: 4,
            severity: Severity::Medium,
        };
        score += a.score;
        anomalies.push(a);
    }

    StructuralResult { score, anomalies }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Simple substring search in binary data.
fn find_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() || haystack.len() < needle.len() {
        return false;
    }
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_bytes_present() {
        assert!(find_bytes(b"hello world", b"world"));
    }

    #[test]
    fn test_find_bytes_absent() {
        assert!(!find_bytes(b"hello world", b"xyz"));
    }

    #[test]
    fn test_find_bytes_empty_needle() {
        assert!(!find_bytes(b"hello", b""));
    }

    #[test]
    fn test_find_bytes_needle_larger() {
        assert!(!find_bytes(b"hi", b"hello"));
    }

    #[test]
    fn test_analyze_structure_unknown_type() {
        let result = analyze_structure(b"random data", FileType::Unknown);
        assert_eq!(result.score, 0);
        assert!(result.anomalies.is_empty());
    }

    #[test]
    fn test_analyze_pe_too_small() {
        let result = analyze_pe(&[0x4D, 0x5A, 0x00]);
        assert_eq!(result.score, 0);
    }

    #[test]
    fn test_analyze_pe_minimal_valid() {
        // Build a minimal PE with valid signature but no real sections.
        let mut pe = vec![0u8; 512];

        // DOS header: MZ magic.
        pe[0] = 0x4D;
        pe[1] = 0x5A;
        // e_lfanew at 0x3C → point to offset 0x80.
        pe[0x3C] = 0x80;

        // PE signature at 0x80.
        pe[0x80] = b'P';
        pe[0x81] = b'E';
        pe[0x82] = 0;
        pe[0x83] = 0;

        // COFF header at 0x84: 0 sections, small optional header.
        // NumberOfSections = 0.
        pe[0x86] = 0;
        pe[0x87] = 0;
        // SizeOfOptionalHeader = 96 (0x60).
        pe[0x94] = 0x60;
        pe[0x95] = 0;

        // Optional header at 0x98: PE32 magic.
        pe[0x98] = 0x0B;
        pe[0x99] = 0x01;

        let result = analyze_pe(&pe);
        // No sections → no entry-point check, no overlay, but might flag no imports.
        // With zero data directories, import check may not trigger.
        assert!(result.score <= 10);
    }

    #[test]
    fn test_analyze_pe_no_imports_detection() {
        // Build a PE where import directory is explicitly zeroed.
        let mut pe = vec![0u8; 600];

        // DOS header.
        pe[0] = 0x4D;
        pe[1] = 0x5A;
        pe[0x3C] = 0x80;

        // PE signature.
        pe[0x80] = b'P';
        pe[0x81] = b'E';

        // COFF: 1 section, optional header size = 0xF0 (240).
        pe[0x86] = 1;
        pe[0x87] = 0;
        pe[0x94] = 0xF0;
        pe[0x95] = 0x00;

        // Optional header: PE32 magic.
        let opt = 0x98;
        pe[opt] = 0x0B;
        pe[opt + 1] = 0x01;

        // NumberOfRvaAndSizes at offset 92 of optional header.
        let ndir_off = opt + 92;
        pe[ndir_off] = 16; // 16 data directories.

        // Import directory is at index 1 → offset = ndir_off + 4 + 8*1.
        // Leave it as 0,0 → no imports.
        let import_off = ndir_off + 4 + 8;
        pe[import_off] = 0;
        pe[import_off + 4] = 0;

        let result = analyze_pe(&pe);
        let has_no_imports = result.anomalies.iter().any(|a| a.id == "PE_NO_IMPORTS");
        assert!(has_no_imports, "Should detect PE with no imports");
    }

    #[test]
    fn test_analyze_elf_too_small() {
        let result = analyze_elf(b"\x7fELF");
        assert_eq!(result.score, 0);
    }

    #[test]
    fn test_analyze_elf_ld_preload() {
        let mut elf = vec![0u8; 256];
        elf[0] = 0x7F;
        elf[1] = b'E';
        elf[2] = b'L';
        elf[3] = b'F';
        elf[4] = 2; // 64-bit

        // Embed LD_PRELOAD string.
        let ld = b"LD_PRELOAD";
        elf[100..100 + ld.len()].copy_from_slice(ld);

        let result = analyze_elf(&elf);
        let has_ld = result.anomalies.iter().any(|a| a.id == "ELF_LD_PRELOAD");
        assert!(has_ld, "Should detect LD_PRELOAD reference");
        assert!(result.score >= 7);
    }

    #[test]
    fn test_analyze_dex_too_small() {
        let result = analyze_dex(b"dex\n035");
        assert_eq!(result.score, 0);
    }

    #[test]
    fn test_analyze_dex_dynamic_loading() {
        let mut dex = vec![0u8; 256];
        dex[0..4].copy_from_slice(b"dex\n");

        // Embed DexClassLoader string.
        let loader = b"DexClassLoader";
        dex[120..120 + loader.len()].copy_from_slice(loader);

        let result = analyze_dex(&dex);
        let has_loader = result
            .anomalies
            .iter()
            .any(|a| a.id == "DEX_DYNAMIC_LOADING");
        assert!(has_loader, "Should detect DexClassLoader");
        assert!(result.score >= 5);
    }

    #[test]
    fn test_analyze_dex_jni_native() {
        let mut dex = vec![0u8; 256];
        dex[0..4].copy_from_slice(b"dex\n");

        let jni = b"JNI_OnLoad";
        dex[120..120 + jni.len()].copy_from_slice(jni);

        let result = analyze_dex(&dex);
        let has_jni = result.anomalies.iter().any(|a| a.id == "DEX_JNI_NATIVE");
        assert!(has_jni, "Should detect JNI native methods");
    }

    #[test]
    fn test_analyze_dex_many_permissions() {
        let mut dex_data = vec![0u8; 2048];
        dex_data[0..4].copy_from_slice(b"dex\n");

        // Embed 12 dangerous permissions.
        let perms = [
            b"android.permission.READ_SMS" as &[u8],
            b"android.permission.SEND_SMS",
            b"android.permission.RECEIVE_SMS",
            b"android.permission.READ_CONTACTS",
            b"android.permission.CAMERA",
            b"android.permission.RECORD_AUDIO",
            b"android.permission.ACCESS_FINE_LOCATION",
            b"android.permission.READ_PHONE_STATE",
            b"android.permission.CALL_PHONE",
            b"android.permission.WRITE_EXTERNAL_STORAGE",
            b"android.permission.READ_CALL_LOG",
            b"android.permission.INSTALL_PACKAGES",
        ];
        let mut offset = 120;
        for perm in &perms {
            if offset + perm.len() + 1 < dex_data.len() {
                dex_data[offset..offset + perm.len()].copy_from_slice(perm);
                offset += perm.len() + 1; // +1 for null separator
            }
        }

        let result = analyze_dex(&dex_data);
        let has_excessive = result
            .anomalies
            .iter()
            .any(|a| a.id == "DEX_EXCESSIVE_PERMISSIONS");
        assert!(
            has_excessive,
            "Should detect excessive permissions; anomalies: {:?}",
            result.anomalies.iter().map(|a| a.id).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_to_detections_conversion() {
        let result = StructuralResult {
            score: 10,
            anomalies: vec![
                StructuralAnomaly {
                    id: "TEST_ANOMALY",
                    description: "Test anomaly".to_string(),
                    score: 5,
                    severity: Severity::Medium,
                },
                StructuralAnomaly {
                    id: "TEST_ANOMALY_2",
                    description: "Another test".to_string(),
                    score: 5,
                    severity: Severity::High,
                },
            ],
        };

        let detections = to_detections(&result);
        assert_eq!(detections.len(), 2);
        assert_eq!(detections[0].engine, DetectionEngine::Heuristic);
        assert_eq!(detections[0].rule_name, "HEUR:Struct/TEST_ANOMALY");
        assert_eq!(detections[1].severity, Severity::High);
    }

    #[test]
    fn test_analyze_structure_dispatches_pe() {
        let mut pe = vec![0u8; 128];
        pe[0] = 0x4D;
        pe[1] = 0x5A;
        let result = analyze_structure(&pe, FileType::PE);
        // At minimum it should not panic.
        let _ = result;
    }

    #[test]
    fn test_analyze_structure_dispatches_elf() {
        let mut elf = vec![0u8; 128];
        elf[0..4].copy_from_slice(b"\x7fELF");
        elf[4] = 2;
        let result = analyze_structure(&elf, FileType::ELF);
        let _ = result;
    }

    #[test]
    fn test_analyze_structure_dispatches_dex() {
        let mut dex = vec![0u8; 128];
        dex[0..4].copy_from_slice(b"dex\n");
        let result = analyze_structure(&dex, FileType::DEX);
        let _ = result;
    }
}
