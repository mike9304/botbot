//! Configuration types for Guardian scanning and runtime.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Top-level configuration for a scan operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanConfig {
    /// Maximum file size to scan in bytes (default: 100MB).
    pub max_file_size: u64,
    /// File extensions to scan (empty = scan all).
    pub scan_extensions: Vec<String>,
    /// Number of threads for parallel scanning.
    pub thread_count: usize,
    /// Per-file scan timeout in milliseconds.
    pub timeout_ms: u64,
    /// Maximum archive recursion depth.
    pub max_archive_depth: u32,
    /// Maximum decompressed archive size in bytes.
    pub max_archive_size: u64,
    /// Archive compression ratio limit (bomb detection).
    pub max_compression_ratio: f64,
    /// Path to signature database.
    pub signature_db_path: PathBuf,
    /// Path to YARA rules directory.
    pub yara_rules_path: PathBuf,
    /// Path to ML model files.
    pub ml_model_path: PathBuf,
    /// Enable/disable individual engines.
    pub engines: EngineConfig,
    /// Heuristic thresholds.
    pub heuristic: HeuristicConfig,
}

impl Default for ScanConfig {
    fn default() -> Self {
        Self {
            max_file_size: 100 * 1024 * 1024, // 100MB
            scan_extensions: Vec::new(),
            thread_count: num_cpus(),
            timeout_ms: 30_000,
            max_archive_depth: 5,
            max_archive_size: 500 * 1024 * 1024, // 500MB
            max_compression_ratio: 100.0,
            signature_db_path: PathBuf::from("data/signatures.db"),
            yara_rules_path: PathBuf::from("data/yara_rules"),
            ml_model_path: PathBuf::from("data/models"),
            engines: EngineConfig::default(),
            heuristic: HeuristicConfig::default(),
        }
    }
}

/// Per-engine enable/disable toggles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub signature: bool,
    pub yara: bool,
    pub heuristic: bool,
    pub ml: bool,
    pub behavioral: bool,
    pub network: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            signature: true,
            yara: true,
            heuristic: true,
            ml: true,
            behavioral: false, // off by default; needs runtime setup
            network: false,    // off by default; needs runtime setup
        }
    }
}

/// Heuristic scoring thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeuristicConfig {
    /// Maximum combined score to be considered clean.
    pub clean_max: u32,
    /// Maximum combined score to be considered suspicious (above = malicious).
    pub suspicious_max: u32,
    /// Entropy threshold for "likely packed/encrypted".
    pub entropy_packed: f64,
    /// Entropy threshold for "suspicious".
    pub entropy_suspicious: f64,
}

impl Default for HeuristicConfig {
    fn default() -> Self {
        Self {
            clean_max: 15,
            suspicious_max: 30,
            entropy_packed: 7.5,
            entropy_suspicious: 7.0,
        }
    }
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = ScanConfig::default();
        assert_eq!(cfg.max_file_size, 100 * 1024 * 1024);
        assert_eq!(cfg.timeout_ms, 30_000);
        assert!(cfg.engines.signature);
        assert!(cfg.engines.yara);
    }

    #[test]
    fn test_heuristic_defaults() {
        let h = HeuristicConfig::default();
        assert_eq!(h.clean_max, 15);
        assert_eq!(h.suspicious_max, 30);
    }

    #[test]
    fn test_config_serialization() {
        let cfg = ScanConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let _: ScanConfig = serde_json::from_str(&json).unwrap();
    }
}
