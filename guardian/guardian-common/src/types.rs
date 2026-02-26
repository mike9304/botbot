//! Shared types used across all Guardian crates.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use uuid::Uuid;

/// Overall verdict for a scanned file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ScanVerdict {
    /// File is clean — no threats detected.
    Clean,
    /// File is suspicious — some indicators found but below malicious threshold.
    Suspicious(ThreatDetails),
    /// File is malicious — high-confidence threat detected.
    Malicious(ThreatDetails),
    /// Scan could not complete.
    Error(String),
}

/// Severity classification for detections.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

/// Details about a detected threat.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ThreatDetails {
    pub name: String,
    pub severity: Severity,
    pub family: Option<String>,
    pub description: String,
    pub detections: Vec<Detection>,
    pub mitre_ids: Vec<String>,
}

/// A single detection from one of the analysis layers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Detection {
    pub engine: DetectionEngine,
    pub rule_name: String,
    pub description: String,
    pub severity: Severity,
    pub metadata: HashMap<String, String>,
}

/// Which engine produced a detection.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DetectionEngine {
    Signature,
    Yara,
    Heuristic,
    MachineLearning,
    Behavioral,
    Network,
}

/// File type classification.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum FileType {
    PE,
    ELF,
    DEX,
    MachO,
    Script,
    Document,
    Archive,
    Unknown,
}

impl FileType {
    /// Detect file type from magic bytes.
    pub fn from_magic(data: &[u8]) -> Self {
        if data.len() < 4 {
            return FileType::Unknown;
        }
        match &data[..4] {
            // MZ header → PE
            [0x4D, 0x5A, ..] => FileType::PE,
            // ELF magic
            [0x7F, 0x45, 0x4C, 0x46] => FileType::ELF,
            // DEX magic "dex\n"
            [0x64, 0x65, 0x78, 0x0A] => FileType::DEX,
            // Mach-O
            [0xFE, 0xED, 0xFA, 0xCE]
            | [0xFE, 0xED, 0xFA, 0xCF]
            | [0xCE, 0xFA, 0xED, 0xFE]
            | [0xCF, 0xFA, 0xED, 0xFE] => FileType::MachO,
            // ZIP (PK\x03\x04) — could be archive or document
            [0x50, 0x4B, 0x03, 0x04] => FileType::Archive,
            // Gzip
            [0x1F, 0x8B, ..] => FileType::Archive,
            // 7z
            [0x37, 0x7A, 0xBC, 0xAF] => FileType::Archive,
            // Script shebangs
            [0x23, 0x21, ..] => FileType::Script,
            _ => FileType::Unknown,
        }
    }
}

/// Complete result from scanning a single file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanResult {
    pub id: Uuid,
    pub file_path: PathBuf,
    pub file_size: u64,
    pub file_type: FileType,
    pub sha256: String,
    pub verdict: ScanVerdict,
    pub detections: Vec<Detection>,
    pub scan_duration_ms: u64,
    pub timestamp: DateTime<Utc>,
}

/// Extracted features from a parsed file, used by heuristic and ML engines.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FeatureVector {
    /// Numeric features (entropy, counts, ratios, etc.)
    pub numeric: Vec<f64>,
    /// Categorical features (section names, import names, etc.)
    pub categorical: Vec<String>,
    /// Arbitrary metadata for reporting.
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Statistics for an engine.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EngineStats {
    pub total_scanned: u64,
    pub total_detections: u64,
    pub avg_scan_time_ms: f64,
    pub signatures_loaded: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_type_detection_pe() {
        let data = [0x4D, 0x5A, 0x90, 0x00, 0x03];
        assert_eq!(FileType::from_magic(&data), FileType::PE);
    }

    #[test]
    fn test_file_type_detection_elf() {
        let data = [0x7F, 0x45, 0x4C, 0x46, 0x02];
        assert_eq!(FileType::from_magic(&data), FileType::ELF);
    }

    #[test]
    fn test_file_type_detection_dex() {
        let data = [0x64, 0x65, 0x78, 0x0A, 0x30];
        assert_eq!(FileType::from_magic(&data), FileType::DEX);
    }

    #[test]
    fn test_file_type_detection_unknown() {
        let data = [0x00, 0x01, 0x02];
        assert_eq!(FileType::from_magic(&data), FileType::Unknown);
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Low < Severity::Medium);
        assert!(Severity::Medium < Severity::High);
        assert!(Severity::High < Severity::Critical);
    }

    #[test]
    fn test_scan_verdict_clean() {
        let verdict = ScanVerdict::Clean;
        let json = serde_json::to_string(&verdict).unwrap();
        assert_eq!(json, "\"Clean\"");
    }
}
