//! Windows PE (Portable Executable) parser.
//!
//! Uses `goblin` to parse PE headers, sections, imports, and exports, then
//! flags suspicious API import combinations that are characteristic of common
//! malware families (process injection, ransomware, droppers, keyloggers).

use guardian_common::{FeatureVector, GuardianError, Result, Severity};
use std::collections::{HashMap, HashSet};

use super::{
    FileParser, ImportInfo, ParsedFile, SectionInfo, SuspiciousIndicator,
    indicators_to_metadata, shannon_entropy,
};

// ── Suspicious import combinations ──────────────────────────────────────────

/// A named combination of Win32 API imports that, when found together,
/// strongly suggest a specific malware technique.
struct SuspiciousCombo {
    name: &'static str,
    description: &'static str,
    severity: Severity,
    mitre_id: &'static str,
    /// All functions in this set must be present to trigger.
    functions: &'static [&'static str],
}

const SUSPICIOUS_COMBOS: &[SuspiciousCombo] = &[
    SuspiciousCombo {
        name: "ProcessInjection",
        description: "CreateRemoteThread + VirtualAllocEx + WriteProcessMemory \
                       indicates process injection (classic DLL injection / \
                       shellcode injection)",
        severity: Severity::Critical,
        mitre_id: "T1055",
        functions: &[
            "CreateRemoteThread",
            "VirtualAllocEx",
            "WriteProcessMemory",
        ],
    },
    SuspiciousCombo {
        name: "Ransomware",
        description: "CryptEncrypt + FindFirstFile + WriteFile suggests \
                       ransomware-style file enumeration and encryption",
        severity: Severity::Critical,
        mitre_id: "T1486",
        functions: &["CryptEncrypt", "FindFirstFile", "WriteFile"],
    },
    SuspiciousCombo {
        name: "Dropper",
        description: "URLDownloadToFile + ShellExecute indicates a dropper \
                       downloading and executing a second-stage payload",
        severity: Severity::High,
        mitre_id: "T1105",
        functions: &["URLDownloadToFile", "ShellExecute"],
    },
    SuspiciousCombo {
        name: "Keylogger",
        description: "SetWindowsHookEx + GetAsyncKeyState indicates keyboard \
                       input capture via hooks",
        severity: Severity::High,
        mitre_id: "T1056.001",
        functions: &["SetWindowsHookEx", "GetAsyncKeyState"],
    },
];

// ── PE information structures ───────────────────────────────────────────────

/// Structured information extracted from a PE binary.
#[derive(Debug, Clone)]
pub struct PeInfo {
    /// True if this is a 64-bit PE (PE32+).
    pub is_64bit: bool,
    /// Machine type (e.g. 0x14c = i386, 0x8664 = AMD64).
    pub machine: u16,
    /// Number of sections.
    pub number_of_sections: usize,
    /// Timestamp from COFF header.
    pub timestamp: u32,
    /// Address of entry point (RVA).
    pub entry_point: u64,
    /// Image base address.
    pub image_base: u64,
    /// Per-section details.
    pub sections: Vec<SectionInfo>,
    /// Imported DLLs and their functions.
    pub imports: Vec<ImportInfo>,
    /// Exported function names.
    pub exports: Vec<String>,
    /// True if a TLS directory (callbacks) is present.
    pub has_tls: bool,
    /// True if there is data beyond the last section (overlay).
    pub has_overlay: bool,
    /// Size of the overlay in bytes (0 if none).
    pub overlay_size: u64,
    /// Overall file entropy.
    pub file_entropy: f64,
    /// Suspicious patterns found.
    pub suspicious: Vec<SuspiciousIndicator>,
}

/// PE format parser.
pub struct PeParser;

impl FileParser for PeParser {
    fn parse(data: &[u8]) -> Result<ParsedFile> {
        let info = parse_pe(data)?;
        Ok(ParsedFile::PE(info))
    }

    fn extract_features(data: &[u8]) -> Result<FeatureVector> {
        let info = parse_pe(data)?;
        Ok(build_feature_vector(&info, data))
    }
}

// ── Core parsing logic ──────────────────────────────────────────────────────

fn parse_pe(data: &[u8]) -> Result<PeInfo> {
    use goblin::pe::PE;

    let pe = PE::parse(data).map_err(|e| {
        GuardianError::Parse(format!("PE parse failed: {e}"))
    })?;

    let is_64bit = pe.is_64;
    let machine = pe.header.coff_header.machine;
    let number_of_sections = pe.sections.len();
    let timestamp = pe.header.coff_header.time_date_stamp;

    let (entry_point, image_base) = if let Some(opt) = pe.header.optional_header {
        let ep = opt.standard_fields.address_of_entry_point as u64;
        let ib = opt.windows_fields.image_base;
        (ep, ib)
    } else {
        (0, 0)
    };

    // ── Sections ────────────────────────────────────────────────────────
    let sections: Vec<SectionInfo> = pe
        .sections
        .iter()
        .map(|sec| {
            let name = String::from_utf8_lossy(
                &sec.name[..sec
                    .name
                    .iter()
                    .position(|&b| b == 0)
                    .unwrap_or(sec.name.len())],
            )
            .to_string();

            let raw_offset = sec.pointer_to_raw_data as usize;
            let raw_size = sec.size_of_raw_data as usize;
            let section_data = if raw_offset < data.len() {
                let end = (raw_offset + raw_size).min(data.len());
                &data[raw_offset..end]
            } else {
                &[]
            };
            let entropy = shannon_entropy(section_data);

            let characteristics = sec.characteristics;
            // IMAGE_SCN_MEM_EXECUTE
            let is_executable = characteristics & 0x2000_0000 != 0;
            // IMAGE_SCN_MEM_WRITE
            let is_writable = characteristics & 0x8000_0000 != 0;

            SectionInfo {
                name,
                virtual_size: sec.virtual_size as u64,
                raw_size: sec.size_of_raw_data as u64,
                entropy,
                is_executable,
                is_writable,
            }
        })
        .collect();

    // ── Imports ─────────────────────────────────────────────────────────
    let imports: Vec<ImportInfo> = pe
        .imports
        .iter()
        .fold(HashMap::<String, Vec<String>>::new(), |mut acc, imp| {
            acc.entry(imp.dll.to_string())
                .or_default()
                .push(imp.name.to_string());
            acc
        })
        .into_iter()
        .map(|(library, functions)| ImportInfo { library, functions })
        .collect();

    // ── Exports ─────────────────────────────────────────────────────────
    let exports: Vec<String> = pe
        .exports
        .iter()
        .filter_map(|exp| exp.name.map(|n| n.to_string()))
        .collect();

    // ── TLS callbacks ───────────────────────────────────────────────────
    let has_tls = pe
        .header
        .optional_header
        .and_then(|opt| {
            opt.data_directories
                .get_tls_table()
                .map(|dd| dd.virtual_address != 0)
        })
        .unwrap_or(false);

    // ── Overlay detection ───────────────────────────────────────────────
    let last_section_end = pe
        .sections
        .iter()
        .map(|s| (s.pointer_to_raw_data as u64) + (s.size_of_raw_data as u64))
        .max()
        .unwrap_or(0);

    let has_overlay = last_section_end < data.len() as u64;
    let overlay_size = if has_overlay {
        data.len() as u64 - last_section_end
    } else {
        0
    };

    // ── File entropy ────────────────────────────────────────────────────
    let file_entropy = shannon_entropy(data);

    // ── Suspicious import combinations ──────────────────────────────────
    let all_imports: HashSet<&str> =
        pe.imports.iter().map(|i| i.name.as_ref()).collect();

    let mut suspicious: Vec<SuspiciousIndicator> = SUSPICIOUS_COMBOS
        .iter()
        .filter(|combo| {
            combo
                .functions
                .iter()
                .all(|func| all_imports.contains(func))
        })
        .map(|combo| SuspiciousIndicator {
            name: combo.name.to_string(),
            description: combo.description.to_string(),
            severity: combo.severity,
            mitre_id: Some(combo.mitre_id.to_string()),
        })
        .collect();

    // Flag sections that are both writable AND executable.
    let mut extra: Vec<SuspiciousIndicator> = sections
        .iter()
        .filter(|s| s.is_executable && s.is_writable)
        .map(|s| SuspiciousIndicator {
            name: "WritableExecutableSection".to_string(),
            description: format!(
                "Section '{}' is both writable and executable — \
                 common in packed / self-modifying malware",
                s.name
            ),
            severity: Severity::Medium,
            mitre_id: Some("T1027.002".to_string()),
        })
        .collect();

    // Flag very high entropy sections (likely packed / encrypted).
    extra.extend(sections.iter().filter(|s| s.entropy > 7.2).map(|s| {
        SuspiciousIndicator {
            name: "HighEntropySection".to_string(),
            description: format!(
                "Section '{}' has entropy {:.2} — likely packed or encrypted",
                s.name, s.entropy
            ),
            severity: Severity::Medium,
            mitre_id: Some("T1027".to_string()),
        }
    }));

    // Flag TLS callbacks (commonly abused for anti-debug).
    if has_tls {
        extra.push(SuspiciousIndicator {
            name: "TLSCallbacks".to_string(),
            description: "TLS callback directory present — may execute code \
                          before entry point (anti-debug / anti-sandbox)"
                .to_string(),
            severity: Severity::Low,
            mitre_id: Some("T1622".to_string()),
        });
    }

    // Flag large overlay (appended data, often used by packers / droppers).
    if overlay_size > 4096 {
        extra.push(SuspiciousIndicator {
            name: "LargeOverlay".to_string(),
            description: format!(
                "Overlay of {overlay_size} bytes detected — may contain \
                 appended payload / encrypted resource"
            ),
            severity: Severity::Low,
            mitre_id: Some("T1027".to_string()),
        });
    }

    suspicious.extend(extra);

    Ok(PeInfo {
        is_64bit,
        machine,
        number_of_sections,
        timestamp,
        entry_point,
        image_base,
        sections,
        imports,
        exports,
        has_tls,
        has_overlay,
        overlay_size,
        file_entropy,
        suspicious,
    })
}

// ── Feature vector construction ─────────────────────────────────────────────

fn build_feature_vector(info: &PeInfo, data: &[u8]) -> FeatureVector {
    let mut numeric: Vec<f64> = Vec::with_capacity(32);

    // Basic header features.
    numeric.push(if info.is_64bit { 1.0 } else { 0.0 });
    numeric.push(info.number_of_sections as f64);
    numeric.push(info.entry_point as f64);
    numeric.push(info.image_base as f64);
    numeric.push(info.file_entropy);
    numeric.push(data.len() as f64);

    // Section entropy stats.
    let entropies: Vec<f64> = info.sections.iter().map(|s| s.entropy).collect();
    numeric.push(
        entropies
            .iter()
            .copied()
            .reduce(f64::max)
            .unwrap_or(0.0),
    );
    numeric.push(
        entropies
            .iter()
            .copied()
            .reduce(f64::min)
            .unwrap_or(0.0),
    );
    let avg_entropy = if entropies.is_empty() {
        0.0
    } else {
        entropies.iter().sum::<f64>() / entropies.len() as f64
    };
    numeric.push(avg_entropy);

    // Import / export counts.
    let total_imports: usize = info.imports.iter().map(|i| i.functions.len()).sum();
    numeric.push(info.imports.len() as f64);
    numeric.push(total_imports as f64);
    numeric.push(info.exports.len() as f64);

    // Overlay / TLS.
    numeric.push(if info.has_overlay { 1.0 } else { 0.0 });
    numeric.push(info.overlay_size as f64);
    numeric.push(if info.has_tls { 1.0 } else { 0.0 });

    // W^X section count.
    let wx_count = info
        .sections
        .iter()
        .filter(|s| s.is_executable && s.is_writable)
        .count();
    numeric.push(wx_count as f64);

    // Suspicious indicator count.
    numeric.push(info.suspicious.len() as f64);

    // Categorical: section names, DLL names.
    let mut categorical: Vec<String> =
        info.sections.iter().map(|s| s.name.clone()).collect();
    categorical.extend(info.imports.iter().map(|i| i.library.clone()));

    // Metadata.
    let mut metadata = indicators_to_metadata(&info.suspicious);
    metadata.insert("machine".to_string(), serde_json::json!(info.machine));
    metadata.insert(
        "timestamp".to_string(),
        serde_json::json!(info.timestamp),
    );
    metadata.insert("is_64bit".to_string(), serde_json::json!(info.is_64bit));

    FeatureVector {
        numeric,
        categorical,
        metadata,
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid PE binary (x86, one tiny section).
    fn minimal_pe() -> Vec<u8> {
        let mut pe = vec![0u8; 512];
        // MZ signature
        pe[0] = 0x4D;
        pe[1] = 0x5A;
        // e_lfanew at offset 0x3C -> PE header starts at 0x80
        pe[0x3C] = 0x80;

        let pe_off = 0x80usize;
        // PE signature "PE\0\0"
        pe[pe_off] = b'P';
        pe[pe_off + 1] = b'E';
        pe[pe_off + 2] = 0;
        pe[pe_off + 3] = 0;

        // COFF header (20 bytes starting at pe_off+4)
        let coff = pe_off + 4;
        // Machine: i386 (0x14c)
        pe[coff] = 0x4C;
        pe[coff + 1] = 0x01;
        // NumberOfSections: 1
        pe[coff + 2] = 0x01;
        pe[coff + 3] = 0x00;
        // SizeOfOptionalHeader: 0xe0 (224)
        pe[coff + 16] = 0xE0;
        pe[coff + 17] = 0x00;
        // Characteristics: EXECUTABLE_IMAGE | 32BIT_MACHINE
        pe[coff + 18] = 0x02;
        pe[coff + 19] = 0x01;

        // Optional header (starts at coff + 20 = pe_off + 24)
        let opt = coff + 20;
        // Magic: PE32 (0x10b)
        pe[opt] = 0x0B;
        pe[opt + 1] = 0x01;
        // AddressOfEntryPoint at opt+16
        pe[opt + 16] = 0x00;
        pe[opt + 17] = 0x10; // 0x1000
        // ImageBase at opt+28
        pe[opt + 28] = 0x00;
        pe[opt + 29] = 0x00;
        pe[opt + 30] = 0x40;
        pe[opt + 31] = 0x00; // 0x00400000
        // SectionAlignment at opt+32
        pe[opt + 32] = 0x00;
        pe[opt + 33] = 0x10; // 0x1000
        // FileAlignment at opt+36
        pe[opt + 36] = 0x00;
        pe[opt + 37] = 0x02; // 0x200
        // SizeOfImage at opt+56
        pe[opt + 56] = 0x00;
        pe[opt + 57] = 0x20; // 0x2000
        // SizeOfHeaders at opt+60
        pe[opt + 60] = 0x00;
        pe[opt + 61] = 0x02; // 0x200
        // NumberOfRvaAndSizes at opt+116
        pe[opt + 116] = 0x10; // 16

        // Section header (starts at opt + 224 = pe_off + 248)
        let sec = opt + 224;
        // Name: ".text\0\0\0"
        pe[sec] = b'.';
        pe[sec + 1] = b't';
        pe[sec + 2] = b'e';
        pe[sec + 3] = b'x';
        pe[sec + 4] = b't';
        // VirtualSize at sec+8
        pe[sec + 8] = 0x00;
        pe[sec + 9] = 0x01; // 0x100
        // VirtualAddress at sec+12
        pe[sec + 12] = 0x00;
        pe[sec + 13] = 0x10; // 0x1000
        // SizeOfRawData at sec+16
        pe[sec + 16] = 0x00;
        pe[sec + 17] = 0x02; // 0x200
        // PointerToRawData at sec+20
        pe[sec + 20] = 0x00;
        pe[sec + 21] = 0x02; // 0x200
        // Characteristics at sec+36: MEM_EXECUTE | MEM_READ
        pe[sec + 36] = 0x00;
        pe[sec + 37] = 0x00;
        pe[sec + 38] = 0x00;
        pe[sec + 39] = 0x60; // 0x60000000

        pe
    }

    #[test]
    fn test_parse_minimal_pe() {
        let data = minimal_pe();
        let result = PeParser::parse(&data);
        assert!(result.is_ok(), "parse failed: {:?}", result.err());
        if let Ok(ParsedFile::PE(info)) = result {
            assert!(!info.is_64bit);
            assert_eq!(info.machine, 0x14C);
            assert_eq!(info.number_of_sections, 1);
            assert_eq!(info.sections[0].name, ".text");
        } else {
            panic!("expected PE variant");
        }
    }

    #[test]
    fn test_extract_features_minimal_pe() {
        let data = minimal_pe();
        let features = PeParser::extract_features(&data);
        assert!(features.is_ok());
        let fv = features.unwrap();
        assert!(!fv.numeric.is_empty());
        // First feature is is_64bit (0.0 for 32-bit).
        assert_eq!(fv.numeric[0], 0.0);
    }

    #[test]
    fn test_parse_invalid_pe() {
        // MZ header but garbage after that.
        let data = [0x4D, 0x5A, 0x00, 0x00];
        let result = PeParser::parse(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_suspicious_combo_process_injection() {
        let imports: HashSet<&str> = [
            "CreateRemoteThread",
            "VirtualAllocEx",
            "WriteProcessMemory",
        ]
        .into_iter()
        .collect();

        let matched: Vec<&SuspiciousCombo> = SUSPICIOUS_COMBOS
            .iter()
            .filter(|c| c.functions.iter().all(|f| imports.contains(f)))
            .collect();

        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].name, "ProcessInjection");
        assert_eq!(matched[0].severity, Severity::Critical);
    }

    #[test]
    fn test_suspicious_combo_ransomware() {
        let imports: HashSet<&str> =
            ["CryptEncrypt", "FindFirstFile", "WriteFile"]
                .into_iter()
                .collect();

        let matched: Vec<&SuspiciousCombo> = SUSPICIOUS_COMBOS
            .iter()
            .filter(|c| c.functions.iter().all(|f| imports.contains(f)))
            .collect();

        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].name, "Ransomware");
    }

    #[test]
    fn test_suspicious_combo_dropper() {
        let imports: HashSet<&str> =
            ["URLDownloadToFile", "ShellExecute"].into_iter().collect();

        let matched: Vec<&SuspiciousCombo> = SUSPICIOUS_COMBOS
            .iter()
            .filter(|c| c.functions.iter().all(|f| imports.contains(f)))
            .collect();

        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].name, "Dropper");
    }

    #[test]
    fn test_suspicious_combo_keylogger() {
        let imports: HashSet<&str> =
            ["SetWindowsHookEx", "GetAsyncKeyState"].into_iter().collect();

        let matched: Vec<&SuspiciousCombo> = SUSPICIOUS_COMBOS
            .iter()
            .filter(|c| c.functions.iter().all(|f| imports.contains(f)))
            .collect();

        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].name, "Keylogger");
    }

    #[test]
    fn test_suspicious_no_match() {
        let imports: HashSet<&str> =
            ["CreateFileW", "ReadFile"].into_iter().collect();

        let matched: Vec<&SuspiciousCombo> = SUSPICIOUS_COMBOS
            .iter()
            .filter(|c| c.functions.iter().all(|f| imports.contains(f)))
            .collect();

        assert!(matched.is_empty());
    }
}
