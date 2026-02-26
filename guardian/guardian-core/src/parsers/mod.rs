//! File format parsers for Guardian Core.
//!
//! Handles PE, ELF, DEX, and archive formats. Each parser extracts structural
//! information and computes a [`FeatureVector`] for downstream heuristic and ML
//! engines.

pub mod archive;
pub mod dex;
pub mod elf;
pub mod pe;

use guardian_common::{FeatureVector, FileType, GuardianError, Result};
use std::collections::HashMap;

// ── Shared utility ──────────────────────────────────────────────────────────

/// Compute the Shannon entropy of a byte slice (bits per byte, 0.0 – 8.0).
pub fn shannon_entropy(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut counts = [0u64; 256];
    for &byte in data {
        counts[byte as usize] += 1;
    }
    let len = data.len() as f64;
    counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / len;
            -p * p.log2()
        })
        .sum()
}

// ── Parsed file representation ──────────────────────────────────────────────

/// Section metadata extracted from any binary format.
#[derive(Debug, Clone)]
pub struct SectionInfo {
    pub name: String,
    pub virtual_size: u64,
    pub raw_size: u64,
    pub entropy: f64,
    pub is_executable: bool,
    pub is_writable: bool,
}

/// An import entry (PE DLL import or ELF symbol).
#[derive(Debug, Clone)]
pub struct ImportInfo {
    pub library: String,
    pub functions: Vec<String>,
}

/// A suspicious pattern flagged during parsing.
#[derive(Debug, Clone)]
pub struct SuspiciousIndicator {
    pub name: String,
    pub description: String,
    pub severity: guardian_common::Severity,
    pub mitre_id: Option<String>,
}

/// Result of parsing a file into one of the supported formats.
#[derive(Debug)]
pub enum ParsedFile {
    /// Windows Portable Executable.
    PE(pe::PeInfo),
    /// Linux / BSD ELF binary.
    ELF(elf::ElfInfo),
    /// Android Dalvik Executable.
    DEX(dex::DexInfo),
    /// Archive containing inner files.
    Archive(archive::ArchiveContents),
}

// ── FileParser trait ────────────────────────────────────────────────────────

/// Trait implemented by each format-specific parser.
pub trait FileParser {
    /// Parse raw bytes into a structured representation.
    fn parse(data: &[u8]) -> Result<ParsedFile>
    where
        Self: Sized;

    /// Extract a normalised feature vector suitable for ML / heuristic scoring.
    fn extract_features(data: &[u8]) -> Result<FeatureVector>
    where
        Self: Sized;
}

// ── Dispatcher ──────────────────────────────────────────────────────────────

/// Auto-detect the file type from magic bytes and delegate to the correct
/// parser.
pub fn parse_file(data: &[u8]) -> Result<ParsedFile> {
    let file_type = FileType::from_magic(data);
    match file_type {
        FileType::PE => pe::PeParser::parse(data),
        FileType::ELF => elf::ElfParser::parse(data),
        FileType::DEX => dex::DexParser::parse(data),
        FileType::Archive => archive::ArchiveParser::parse(data),
        other => Err(GuardianError::UnsupportedFileType(format!("{:?}", other))),
    }
}

/// Auto-detect type and extract features.
pub fn extract_features(data: &[u8]) -> Result<FeatureVector> {
    let file_type = FileType::from_magic(data);
    match file_type {
        FileType::PE => pe::PeParser::extract_features(data),
        FileType::ELF => elf::ElfParser::extract_features(data),
        FileType::DEX => dex::DexParser::extract_features(data),
        FileType::Archive => archive::ArchiveParser::extract_features(data),
        other => Err(GuardianError::UnsupportedFileType(format!("{:?}", other))),
    }
}

// ── Helper: build metadata map from indicators ──────────────────────────────

fn indicators_to_metadata(
    indicators: &[SuspiciousIndicator],
) -> HashMap<String, serde_json::Value> {
    let mut map = HashMap::new();
    let names: Vec<String> = indicators.iter().map(|i| i.name.clone()).collect();
    map.insert(
        "suspicious_indicators".to_string(),
        serde_json::json!(names),
    );
    map.insert(
        "indicator_count".to_string(),
        serde_json::json!(indicators.len()),
    );
    map
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shannon_entropy_empty() {
        assert_eq!(shannon_entropy(&[]), 0.0);
    }

    #[test]
    fn test_shannon_entropy_uniform() {
        // All identical bytes → entropy = 0
        let data = vec![0xAAu8; 1024];
        assert_eq!(shannon_entropy(&data), 0.0);
    }

    #[test]
    fn test_shannon_entropy_two_values() {
        // Equal mix of 0x00 and 0xFF → entropy = 1.0 bit
        let mut data = vec![0x00u8; 512];
        data.extend(vec![0xFFu8; 512]);
        let e = shannon_entropy(&data);
        assert!((e - 1.0).abs() < 0.001, "expected ~1.0, got {e}");
    }

    #[test]
    fn test_shannon_entropy_random_high() {
        // Pseudo-uniform distribution → entropy close to 8.0
        let data: Vec<u8> = (0..=255).cycle().take(256 * 100).collect();
        let e = shannon_entropy(&data);
        assert!(e > 7.9, "expected high entropy, got {e}");
    }

    #[test]
    fn test_parse_file_unknown_type() {
        let data = [0x00, 0x01, 0x02, 0x03];
        let result = parse_file(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_features_unknown_type() {
        let data = [0x00, 0x01, 0x02, 0x03];
        let result = extract_features(&data);
        assert!(result.is_err());
    }
}
