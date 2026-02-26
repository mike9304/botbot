//! Guardian Core — shared detection engine for Home Guardian antivirus.
//!
//! This crate implements the 6-layer detection pipeline:
//! 1. Signature matching (Bloom filter + SHA-256 + Aho-Corasick)
//! 2. YARA-X rule scanning
//! 3. Heuristic analysis (entropy, strings, structural anomalies)
//! 4. ML classification (EMBER features + ONNX inference)
//! 5. Behavioral pattern matching (MITRE ATT&CK mapped)
//! 6. Network analysis (delegated to guardian-network crate)

pub mod behavior;
pub mod db;
pub mod heuristic;
pub mod ml;
pub mod parsers;
pub mod scanner;
pub mod signature;
pub mod yara;

use guardian_common::{
    Detection, DetectionEngine, FeatureVector, FileType, Result, ScanConfig, ScanResult,
    ScanVerdict, Severity, ThreatDetails,
};
use std::path::Path;
use tracing::{debug, info, warn};

/// Trait for any detection engine layer.
pub trait DetectionLayer: Send + Sync {
    /// Human-readable name of this engine.
    fn name(&self) -> &str;

    /// Which engine type this represents.
    fn engine_type(&self) -> DetectionEngine;

    /// Scan raw bytes and return detections.
    fn scan_bytes(&self, data: &[u8], file_type: FileType) -> Result<Vec<Detection>>;

    /// Scan a file on disk (default: read + scan_bytes).
    fn scan_file(&self, path: &Path) -> Result<Vec<Detection>> {
        let data = std::fs::read(path)?;
        let file_type = FileType::from_magic(&data);
        self.scan_bytes(&data, file_type)
    }
}

/// The unified scanner that orchestrates all detection layers.
pub struct GuardianScanner {
    config: ScanConfig,
    layers: Vec<Box<dyn DetectionLayer>>,
}

impl GuardianScanner {
    /// Create a new scanner with the given configuration.
    pub fn new(config: ScanConfig) -> Self {
        Self {
            config,
            layers: Vec::new(),
        }
    }

    /// Register a detection layer.
    pub fn add_layer(&mut self, layer: Box<dyn DetectionLayer>) {
        info!(engine = layer.name(), "Registered detection layer");
        self.layers.push(layer);
    }

    /// Scan a file through all registered layers.
    pub fn scan_file(&self, path: &Path) -> Result<ScanResult> {
        let start = std::time::Instant::now();
        let metadata = std::fs::metadata(path)?;

        if metadata.len() > self.config.max_file_size {
            return Err(guardian_common::GuardianError::FileTooLarge {
                size: metadata.len(),
                max: self.config.max_file_size,
            });
        }

        let data = std::fs::read(path)?;
        let file_type = FileType::from_magic(&data);
        let sha256 = compute_sha256(&data);

        debug!(
            path = %path.display(),
            size = metadata.len(),
            file_type = ?file_type,
            "Scanning file"
        );

        let mut all_detections = Vec::new();

        for layer in &self.layers {
            match layer.scan_bytes(&data, file_type) {
                Ok(detections) => {
                    if !detections.is_empty() {
                        debug!(
                            engine = layer.name(),
                            count = detections.len(),
                            "Detections found"
                        );
                    }
                    all_detections.extend(detections);
                }
                Err(e) => {
                    warn!(engine = layer.name(), error = %e, "Engine error during scan");
                }
            }
        }

        let verdict = determine_verdict(&all_detections);
        let duration = start.elapsed();

        Ok(ScanResult {
            id: uuid::Uuid::new_v4(),
            file_path: path.to_path_buf(),
            file_size: metadata.len(),
            file_type,
            sha256,
            verdict,
            detections: all_detections,
            scan_duration_ms: duration.as_millis() as u64,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Scan raw bytes (no file path).
    pub fn scan_bytes(&self, data: &[u8]) -> Result<Vec<Detection>> {
        let file_type = FileType::from_magic(data);
        let mut all_detections = Vec::new();
        for layer in &self.layers {
            match layer.scan_bytes(data, file_type) {
                Ok(d) => all_detections.extend(d),
                Err(e) => warn!(engine = layer.name(), error = %e, "Engine error"),
            }
        }
        Ok(all_detections)
    }
}

/// Compute SHA-256 hex digest.
pub fn compute_sha256(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

/// Determine the overall verdict from all detections.
fn determine_verdict(detections: &[Detection]) -> ScanVerdict {
    if detections.is_empty() {
        return ScanVerdict::Clean;
    }

    let max_severity = detections
        .iter()
        .map(|d| d.severity)
        .max()
        .unwrap_or(Severity::Low);

    let details = ThreatDetails {
        name: detections
            .first()
            .map(|d| d.rule_name.clone())
            .unwrap_or_default(),
        severity: max_severity,
        family: None,
        description: format!("{} detection(s) across engines", detections.len()),
        detections: detections.to_vec(),
        mitre_ids: detections
            .iter()
            .filter_map(|d| d.metadata.get("mitre_id").cloned())
            .collect(),
    };

    match max_severity {
        Severity::Critical | Severity::High => ScanVerdict::Malicious(details),
        Severity::Medium => ScanVerdict::Suspicious(details),
        Severity::Low => {
            if detections.len() >= 3 {
                ScanVerdict::Suspicious(details)
            } else {
                ScanVerdict::Clean
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_sha256() {
        let hash = compute_sha256(b"hello world");
        assert_eq!(
            hash,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[test]
    fn test_determine_verdict_clean() {
        let verdict = determine_verdict(&[]);
        assert_eq!(verdict, ScanVerdict::Clean);
    }

    #[test]
    fn test_determine_verdict_malicious() {
        let detections = vec![Detection {
            engine: DetectionEngine::Signature,
            rule_name: "Trojan.Generic".to_string(),
            description: "Known malware hash".to_string(),
            severity: Severity::Critical,
            metadata: std::collections::HashMap::new(),
        }];
        match determine_verdict(&detections) {
            ScanVerdict::Malicious(_) => {}
            other => panic!("Expected Malicious, got {:?}", other),
        }
    }

    #[test]
    fn test_scanner_no_layers() {
        let scanner = GuardianScanner::new(ScanConfig::default());
        let result = scanner.scan_bytes(b"test data").unwrap();
        assert!(result.is_empty());
    }
}
