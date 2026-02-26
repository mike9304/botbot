//! Linux/BSD ELF (Executable and Linkable Format) parser.
//!
//! Uses `goblin` to parse ELF headers, sections, symbols, and the dynamic
//! section.  Flags suspicious strings and syscall patterns commonly seen in
//! Linux malware (LD_PRELOAD hooking, ptrace anti-debug, memfd_create
//! fileless execution, etc.).

use guardian_common::{FeatureVector, GuardianError, Result, Severity};

use super::{
    FileParser, ParsedFile, SectionInfo, SuspiciousIndicator, indicators_to_metadata,
    shannon_entropy,
};

// ── Suspicious strings to search for in symbol / string tables ──────────────

struct SuspiciousString {
    pattern: &'static str,
    name: &'static str,
    description: &'static str,
    severity: Severity,
    mitre_id: &'static str,
}

const SUSPICIOUS_STRINGS: &[SuspiciousString] = &[
    SuspiciousString {
        pattern: "LD_PRELOAD",
        name: "LdPreloadHijack",
        description: "Reference to LD_PRELOAD — used for shared-library \
                       hijacking / function hooking",
        severity: Severity::High,
        mitre_id: "T1574.006",
    },
    SuspiciousString {
        pattern: "ptrace",
        name: "PtraceAntiDebug",
        description: "Reference to ptrace — commonly used for anti-debug \
                       (PTRACE_TRACEME) or process injection",
        severity: Severity::Medium,
        mitre_id: "T1622",
    },
    SuspiciousString {
        pattern: "/proc/self/maps",
        name: "ProcSelfMaps",
        description: "Reads /proc/self/maps — may be enumerating loaded \
                       libraries or detecting debuggers / sandboxes",
        severity: Severity::Medium,
        mitre_id: "T1057",
    },
    SuspiciousString {
        pattern: "memfd_create",
        name: "MemfdFileless",
        description: "Uses memfd_create — fileless execution technique that \
                       runs code from anonymous memory",
        severity: Severity::High,
        mitre_id: "T1620",
    },
    SuspiciousString {
        pattern: "process_vm_writev",
        name: "ProcessVmWritev",
        description: "Uses process_vm_writev — direct cross-process memory \
                       write for injection without ptrace",
        severity: Severity::Critical,
        mitre_id: "T1055.009",
    },
];

// ── ELF information structures ──────────────────────────────────────────────

/// Structured information extracted from an ELF binary.
#[derive(Debug, Clone)]
pub struct ElfInfo {
    /// ELF class: 32 or 64.
    pub class: u8,
    /// ELF type (ET_EXEC, ET_DYN, ET_REL, etc.).
    pub elf_type: u16,
    /// Machine architecture (EM_X86_64, EM_ARM, etc.).
    pub machine: u16,
    /// Entry point address.
    pub entry_point: u64,
    /// Whether the binary is statically linked.
    pub is_static: bool,
    /// Whether the binary is position-independent (PIE / shared).
    pub is_pie: bool,
    /// Per-section details.
    pub sections: Vec<SectionInfo>,
    /// Dynamic library dependencies (NEEDED entries).
    pub libraries: Vec<String>,
    /// Dynamic symbols (imports).
    pub dynamic_symbols: Vec<String>,
    /// Exported symbols.
    pub exported_symbols: Vec<String>,
    /// Overall file entropy.
    pub file_entropy: f64,
    /// Suspicious indicators.
    pub suspicious: Vec<SuspiciousIndicator>,
}

/// ELF format parser.
pub struct ElfParser;

impl FileParser for ElfParser {
    fn parse(data: &[u8]) -> Result<ParsedFile> {
        let info = parse_elf(data)?;
        Ok(ParsedFile::ELF(info))
    }

    fn extract_features(data: &[u8]) -> Result<FeatureVector> {
        let info = parse_elf(data)?;
        Ok(build_feature_vector(&info, data))
    }
}

// ── Core parsing logic ──────────────────────────────────────────────────────

fn parse_elf(data: &[u8]) -> Result<ElfInfo> {
    use goblin::elf::Elf;

    let elf = Elf::parse(data).map_err(|e| {
        GuardianError::Parse(format!("ELF parse failed: {e}"))
    })?;

    let class = if elf.is_64 { 64 } else { 32 };
    let elf_type = elf.header.e_type;
    let machine = elf.header.e_machine;
    let entry_point = elf.header.e_entry;

    // Static = no INTERP program header and no dynamic section.
    let is_static = elf.interpreter.is_none() && elf.dynamic.is_none();

    // PIE = ET_DYN with an entry point.
    let is_pie = elf_type == goblin::elf::header::ET_DYN;

    // ── Sections ────────────────────────────────────────────────────────
    let sections: Vec<SectionInfo> = elf
        .section_headers
        .iter()
        .map(|sh| {
            let name = elf
                .shdr_strtab
                .get_at(sh.sh_name)
                .unwrap_or("")
                .to_string();

            let offset = sh.sh_offset as usize;
            let size = sh.sh_size as usize;
            let section_data =
                if sh.sh_type != goblin::elf::section_header::SHT_NOBITS
                    && offset < data.len()
                {
                    let end = (offset + size).min(data.len());
                    &data[offset..end]
                } else {
                    &[]
                };
            let entropy = shannon_entropy(section_data);

            let flags = sh.sh_flags as u64;
            let is_executable =
                flags & (goblin::elf::section_header::SHF_EXECINSTR as u64) != 0;
            let is_writable =
                flags & (goblin::elf::section_header::SHF_WRITE as u64) != 0;

            SectionInfo {
                name,
                virtual_size: sh.sh_size,
                raw_size: if sh.sh_type
                    == goblin::elf::section_header::SHT_NOBITS
                {
                    0
                } else {
                    sh.sh_size
                },
                entropy,
                is_executable,
                is_writable,
            }
        })
        .collect();

    // ── Dynamic libraries (DT_NEEDED) ───────────────────────────────────
    let libraries: Vec<String> =
        elf.libraries.iter().map(|l| l.to_string()).collect();

    // ── Symbols ─────────────────────────────────────────────────────────
    let dynamic_symbols: Vec<String> = elf
        .dynsyms
        .iter()
        .filter(|sym| sym.is_import())
        .filter_map(|sym| elf.dynstrtab.get_at(sym.st_name))
        .filter(|n| !n.is_empty())
        .map(|n| n.to_string())
        .collect();

    let exported_symbols: Vec<String> = elf
        .dynsyms
        .iter()
        .filter(|sym| !sym.is_import() && sym.st_value != 0)
        .filter_map(|sym| elf.dynstrtab.get_at(sym.st_name))
        .filter(|n| !n.is_empty())
        .map(|n| n.to_string())
        .collect();

    // ── File entropy ────────────────────────────────────────────────────
    let file_entropy = shannon_entropy(data);

    // ── Suspicious pattern scan ─────────────────────────────────────────
    // Collect all string material to search against.
    let mut all_strings = String::new();
    for sym_name in dynamic_symbols.iter().chain(exported_symbols.iter()) {
        all_strings.push_str(sym_name);
        all_strings.push('\n');
    }
    for lib in &libraries {
        all_strings.push_str(lib);
        all_strings.push('\n');
    }
    // Also scan printable sequences from the .rodata / .data sections.
    for sh in &elf.section_headers {
        let name = elf.shdr_strtab.get_at(sh.sh_name).unwrap_or("");
        if name == ".rodata" || name == ".data" {
            let offset = sh.sh_offset as usize;
            let size = sh.sh_size as usize;
            if offset < data.len() {
                let end = (offset + size).min(data.len());
                let section_data = &data[offset..end];
                // Extract ASCII printable runs of >= 4 chars.
                let mut run = String::new();
                for &b in section_data {
                    if b.is_ascii_graphic() || b == b' ' {
                        run.push(b as char);
                    } else {
                        if run.len() >= 4 {
                            all_strings.push_str(&run);
                            all_strings.push('\n');
                        }
                        run.clear();
                    }
                }
                if run.len() >= 4 {
                    all_strings.push_str(&run);
                    all_strings.push('\n');
                }
            }
        }
    }

    let mut suspicious: Vec<SuspiciousIndicator> = SUSPICIOUS_STRINGS
        .iter()
        .filter(|s| all_strings.contains(s.pattern))
        .map(|s| SuspiciousIndicator {
            name: s.name.to_string(),
            description: s.description.to_string(),
            severity: s.severity,
            mitre_id: Some(s.mitre_id.to_string()),
        })
        .collect();

    // Flag W^X sections.
    suspicious.extend(
        sections
            .iter()
            .filter(|s| s.is_executable && s.is_writable)
            .map(|s| SuspiciousIndicator {
                name: "WritableExecutableSection".to_string(),
                description: format!(
                    "Section '{}' is both writable and executable",
                    s.name
                ),
                severity: Severity::Medium,
                mitre_id: Some("T1027.002".to_string()),
            }),
    );

    // Flag high entropy sections.
    suspicious.extend(sections.iter().filter(|s| s.entropy > 7.2).map(|s| {
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

    // Static binaries are unusual for legitimate software on modern Linux.
    if is_static {
        suspicious.push(SuspiciousIndicator {
            name: "StaticBinary".to_string(),
            description: "Statically linked binary — unusual for normal \
                          software, common in droppers / implants"
                .to_string(),
            severity: Severity::Low,
            mitre_id: Some("T1027".to_string()),
        });
    }

    Ok(ElfInfo {
        class,
        elf_type,
        machine,
        entry_point,
        is_static,
        is_pie,
        sections,
        libraries,
        dynamic_symbols,
        exported_symbols,
        file_entropy,
        suspicious,
    })
}

// ── Feature vector construction ─────────────────────────────────────────────

fn build_feature_vector(info: &ElfInfo, data: &[u8]) -> FeatureVector {
    let mut numeric: Vec<f64> = Vec::with_capacity(24);

    // Basic header features.
    numeric.push(info.class as f64);
    numeric.push(info.elf_type as f64);
    numeric.push(info.machine as f64);
    numeric.push(info.entry_point as f64);
    numeric.push(if info.is_static { 1.0 } else { 0.0 });
    numeric.push(if info.is_pie { 1.0 } else { 0.0 });
    numeric.push(info.file_entropy);
    numeric.push(data.len() as f64);
    numeric.push(info.sections.len() as f64);

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

    // Symbol / library counts.
    numeric.push(info.libraries.len() as f64);
    numeric.push(info.dynamic_symbols.len() as f64);
    numeric.push(info.exported_symbols.len() as f64);

    // W^X section count.
    let wx_count = info
        .sections
        .iter()
        .filter(|s| s.is_executable && s.is_writable)
        .count();
    numeric.push(wx_count as f64);

    // Suspicious indicator count.
    numeric.push(info.suspicious.len() as f64);

    // Categorical: section names, library names.
    let mut categorical: Vec<String> =
        info.sections.iter().map(|s| s.name.clone()).collect();
    categorical.extend(info.libraries.iter().cloned());

    // Metadata.
    let mut metadata = indicators_to_metadata(&info.suspicious);
    metadata.insert("class".to_string(), serde_json::json!(info.class));
    metadata.insert(
        "elf_type".to_string(),
        serde_json::json!(info.elf_type),
    );
    metadata.insert("machine".to_string(), serde_json::json!(info.machine));
    metadata.insert(
        "is_static".to_string(),
        serde_json::json!(info.is_static),
    );
    metadata.insert("is_pie".to_string(), serde_json::json!(info.is_pie));

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

    /// Build a minimal valid 64-bit ELF binary.
    fn minimal_elf64() -> Vec<u8> {
        let mut elf = vec![0u8; 256];

        // ELF magic
        elf[0] = 0x7F;
        elf[1] = b'E';
        elf[2] = b'L';
        elf[3] = b'F';

        // EI_CLASS: ELFCLASS64
        elf[4] = 2;
        // EI_DATA: little-endian
        elf[5] = 1;
        // EI_VERSION
        elf[6] = 1;
        // EI_OSABI: ELFOSABI_NONE
        elf[7] = 0;

        // e_type: ET_EXEC (2) at offset 16
        elf[16] = 2;
        elf[17] = 0;
        // e_machine: EM_X86_64 (0x3E) at offset 18
        elf[18] = 0x3E;
        elf[19] = 0;
        // e_version at offset 20
        elf[20] = 1;
        // e_entry (8 bytes at offset 24)
        elf[24] = 0x00;
        elf[25] = 0x10;
        // e_phoff (8 bytes at offset 32) = 0 (no program headers)
        // e_shoff (8 bytes at offset 40) = 0 (no section headers)
        // e_ehsize (2 bytes at offset 52) = 64
        elf[52] = 64;
        // e_phentsize (2 bytes at offset 54) = 56
        elf[54] = 56;
        // e_shentsize (2 bytes at offset 58) = 64
        elf[58] = 64;

        elf
    }

    #[test]
    fn test_parse_minimal_elf() {
        let data = minimal_elf64();
        let result = ElfParser::parse(&data);
        assert!(result.is_ok(), "parse failed: {:?}", result.err());
        if let Ok(ParsedFile::ELF(info)) = result {
            assert_eq!(info.class, 64);
            assert_eq!(info.machine, 0x3E);
            assert!(info.is_static);
        } else {
            panic!("expected ELF variant");
        }
    }

    #[test]
    fn test_extract_features_minimal_elf() {
        let data = minimal_elf64();
        let features = ElfParser::extract_features(&data);
        assert!(features.is_ok());
        let fv = features.unwrap();
        // First feature is class (64.0).
        assert_eq!(fv.numeric[0], 64.0);
    }

    #[test]
    fn test_parse_invalid_elf() {
        // ELF magic but garbage.
        let data = [0x7F, 0x45, 0x4C, 0x46, 0x00, 0x00];
        let result = ElfParser::parse(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_suspicious_string_patterns_wellformed() {
        for s in SUSPICIOUS_STRINGS {
            assert!(!s.pattern.is_empty());
            assert!(!s.name.is_empty());
            assert!(!s.mitre_id.is_empty());
        }
    }

    #[test]
    fn test_suspicious_ld_preload_match() {
        let haystack = "some stuff\nLD_PRELOAD=/tmp/evil.so\nmore stuff";
        let matched: Vec<&SuspiciousString> = SUSPICIOUS_STRINGS
            .iter()
            .filter(|s| haystack.contains(s.pattern))
            .collect();
        assert!(matched.iter().any(|m| m.name == "LdPreloadHijack"));
    }

    #[test]
    fn test_suspicious_memfd_create_match() {
        let haystack = "call memfd_create then fexecve";
        let matched: Vec<&SuspiciousString> = SUSPICIOUS_STRINGS
            .iter()
            .filter(|s| haystack.contains(s.pattern))
            .collect();
        assert!(matched.iter().any(|m| m.name == "MemfdFileless"));
    }

    #[test]
    fn test_suspicious_no_match() {
        let haystack = "printf write read open close";
        let matched: Vec<&SuspiciousString> = SUSPICIOUS_STRINGS
            .iter()
            .filter(|s| haystack.contains(s.pattern))
            .collect();
        assert!(matched.is_empty());
    }
}
