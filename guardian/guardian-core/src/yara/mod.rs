//! YARA-based detection layer for Home Guardian.
//!
//! This module provides [`YaraEngine`], which implements [`DetectionLayer`]
//! using pattern-matching rules defined as [`YaraRule`] structs.
//!
//! ## Feature-gated `yara-x` support
//!
//! When the `yara-x` crate is available and the `yara_native` Cargo feature
//! is enabled, the engine delegates to the native YARA-X compiler/scanner for
//! maximum performance and full rule compatibility.  When compiled *without*
//! that feature (the default), a built-in fallback engine performs simple
//! multi-pattern matching — sufficient for the built-in rule set and for
//! straightforward custom rules.

pub mod rules;

use crate::DetectionLayer;
use guardian_common::{
    Detection, DetectionEngine, EngineStats, FileType, GuardianError, Result,
};
use rules::{builtin_rules, load_rules_from_dir, YaraRule};
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// YaraEngine
// ---------------------------------------------------------------------------

/// The YARA detection layer.
///
/// Maintains an ordered list of [`YaraRule`] values and scans incoming byte
/// buffers against every applicable rule.
pub struct YaraEngine {
    /// All loaded rules (read-heavy, rarely written).
    rules: RwLock<Vec<YaraRule>>,
    /// Running statistics.
    stats: ScanStats,
}

/// Internal bookkeeping counters.
struct ScanStats {
    scanned: AtomicU64,
    detections: AtomicU64,
    total_scan_us: AtomicU64, // microseconds
}

impl ScanStats {
    fn new() -> Self {
        Self {
            scanned: AtomicU64::new(0),
            detections: AtomicU64::new(0),
            total_scan_us: AtomicU64::new(0),
        }
    }
}

impl YaraEngine {
    // -----------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------

    /// Create a new engine pre-loaded with the built-in rule set.
    pub fn new() -> Self {
        let builtin = builtin_rules();
        info!(count = builtin.len(), "YaraEngine initialized with built-in rules");
        Self {
            rules: RwLock::new(builtin),
            stats: ScanStats::new(),
        }
    }

    /// Create an empty engine (no rules loaded).
    pub fn empty() -> Self {
        Self {
            rules: RwLock::new(Vec::new()),
            stats: ScanStats::new(),
        }
    }

    /// Create an engine, loading built-in rules plus any `.yar`/`.yara` files
    /// found in `rules_dir`.
    pub fn with_rules_dir(rules_dir: &Path) -> Result<Self> {
        let mut all_rules = builtin_rules();
        if rules_dir.is_dir() {
            match load_rules_from_dir(rules_dir) {
                Ok(extra) => {
                    info!(
                        count = extra.len(),
                        dir = %rules_dir.display(),
                        "Loaded custom YARA rules from directory"
                    );
                    all_rules.extend(extra);
                }
                Err(e) => {
                    warn!(
                        error = %e,
                        "Failed to load custom YARA rules — continuing with built-ins only"
                    );
                }
            }
        } else {
            debug!(
                dir = %rules_dir.display(),
                "YARA rules directory does not exist — using built-ins only"
            );
        }

        let count = all_rules.len();
        info!(count, "YaraEngine initialized");
        Ok(Self {
            rules: RwLock::new(all_rules),
            stats: ScanStats::new(),
        })
    }

    // -----------------------------------------------------------------
    // Rule management
    // -----------------------------------------------------------------

    /// Add a single rule at runtime.
    pub fn add_rule(&self, rule: YaraRule) {
        let mut rules = self.rules.write().expect("YaraEngine rules lock poisoned");
        info!(name = %rule.name, "Adding YARA rule");
        rules.push(rule);
    }

    /// Add multiple rules at once.
    pub fn add_rules(&self, new_rules: Vec<YaraRule>) {
        let mut rules = self.rules.write().expect("YaraEngine rules lock poisoned");
        info!(count = new_rules.len(), "Adding YARA rules");
        rules.extend(new_rules);
    }

    /// Remove a rule by name.  Returns `true` if a rule was removed.
    pub fn remove_rule(&self, name: &str) -> bool {
        let mut rules = self.rules.write().expect("YaraEngine rules lock poisoned");
        let before = rules.len();
        rules.retain(|r| r.name != name);
        let removed = rules.len() < before;
        if removed {
            info!(name, "Removed YARA rule");
        }
        removed
    }

    /// Total number of loaded rules.
    pub fn rule_count(&self) -> usize {
        self.rules
            .read()
            .expect("YaraEngine rules lock poisoned")
            .len()
    }

    /// Return names of all loaded rules.
    pub fn rule_names(&self) -> Vec<String> {
        self.rules
            .read()
            .expect("YaraEngine rules lock poisoned")
            .iter()
            .map(|r| r.name.clone())
            .collect()
    }

    // -----------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------

    /// Return aggregate engine statistics.
    pub fn get_stats(&self) -> EngineStats {
        let scanned = self.stats.scanned.load(Ordering::Relaxed);
        let detections = self.stats.detections.load(Ordering::Relaxed);
        let total_us = self.stats.total_scan_us.load(Ordering::Relaxed);
        let avg_ms = if scanned > 0 {
            (total_us as f64 / scanned as f64) / 1000.0
        } else {
            0.0
        };
        EngineStats {
            total_scanned: scanned,
            total_detections: detections,
            avg_scan_time_ms: avg_ms,
            signatures_loaded: self.rule_count() as u64,
        }
    }
}

// ---------------------------------------------------------------------------
// DetectionLayer implementation
// ---------------------------------------------------------------------------

impl DetectionLayer for YaraEngine {
    fn name(&self) -> &str {
        "YARA"
    }

    fn engine_type(&self) -> DetectionEngine {
        DetectionEngine::Yara
    }

    fn scan_bytes(&self, data: &[u8], file_type: FileType) -> Result<Vec<Detection>> {
        let start = std::time::Instant::now();

        let rules = self
            .rules
            .read()
            .map_err(|e| GuardianError::Yara(format!("rules lock poisoned: {e}")))?;

        let mut detections = Vec::new();

        for rule in rules.iter() {
            // Skip rules that do not apply to this file type.
            if !rule.applies_to(file_type) {
                continue;
            }

            if rule.matches(data) {
                let mut metadata: HashMap<String, String> = rule.metadata.clone();
                if let Some(ref mid) = rule.mitre_id {
                    metadata.insert("mitre_id".into(), mid.clone());
                }
                metadata.insert("tags".into(), rule.tags.join(", "));

                detections.push(Detection {
                    engine: DetectionEngine::Yara,
                    rule_name: rule.name.clone(),
                    description: rule.description.clone(),
                    severity: rule.severity,
                    metadata,
                });

                debug!(rule = %rule.name, severity = ?rule.severity, "YARA rule matched");
            }
        }

        // Update stats.
        let elapsed_us = start.elapsed().as_micros() as u64;
        self.stats.scanned.fetch_add(1, Ordering::Relaxed);
        self.stats
            .detections
            .fetch_add(detections.len() as u64, Ordering::Relaxed);
        self.stats
            .total_scan_us
            .fetch_add(elapsed_us, Ordering::Relaxed);

        Ok(detections)
    }
}

impl Default for YaraEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use guardian_common::Severity;

    #[test]
    fn test_new_engine_has_builtin_rules() {
        let engine = YaraEngine::new();
        assert!(engine.rule_count() >= 5);
    }

    #[test]
    fn test_empty_engine() {
        let engine = YaraEngine::empty();
        assert_eq!(engine.rule_count(), 0);
    }

    #[test]
    fn test_scan_eicar() {
        let engine = YaraEngine::new();
        let eicar =
            b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*";
        let detections = engine.scan_bytes(eicar, FileType::Unknown).unwrap();
        assert!(!detections.is_empty());
        assert!(detections.iter().any(|d| d.rule_name == "EICAR_test_file"));
    }

    #[test]
    fn test_scan_clean_file() {
        let engine = YaraEngine::new();
        let clean = b"This is a perfectly normal text file with nothing suspicious.";
        let detections = engine.scan_bytes(clean, FileType::Unknown).unwrap();
        assert!(detections.is_empty());
    }

    #[test]
    fn test_scan_respects_file_type_filter() {
        let engine = YaraEngine::new();
        // UPX rule only applies to PE files.  Construct data that would
        // match the UPX patterns but present it as an ELF.
        let mut data = vec![0u8; 16];
        data.extend_from_slice(b"UPX0");
        data.extend_from_slice(&[0u8; 16]);
        data.extend_from_slice(b"UPX1");
        data.extend_from_slice(&[0u8; 16]);
        data.extend_from_slice(b"UPX!");

        let detections = engine.scan_bytes(&data, FileType::ELF).unwrap();
        // UPX rule should NOT match because file_type is ELF.
        assert!(
            !detections.iter().any(|d| d.rule_name == "UPX_packed_PE"),
            "UPX rule should not match non-PE files"
        );
    }

    #[test]
    fn test_scan_crypto_miner() {
        let engine = YaraEngine::new();
        let miner_data = b"connecting to stratum+tcp://pool.example.com using xmrig v6";
        let detections = engine
            .scan_bytes(miner_data, FileType::Unknown)
            .unwrap();
        assert!(detections
            .iter()
            .any(|d| d.rule_name == "CryptoMiner_Indicators"));
    }

    #[test]
    fn test_detection_engine_type() {
        let engine = YaraEngine::new();
        assert_eq!(engine.engine_type(), DetectionEngine::Yara);
        assert_eq!(engine.name(), "YARA");
    }

    #[test]
    fn test_add_and_remove_rule() {
        let engine = YaraEngine::empty();
        assert_eq!(engine.rule_count(), 0);

        engine.add_rule(rules::YaraRule {
            name: "TestRule".into(),
            description: "test".into(),
            severity: Severity::Low,
            tags: vec![],
            mitre_id: None,
            applicable_types: vec![],
            patterns: vec![rules::BytePattern {
                id: "$a".into(),
                bytes: b"test".to_vec(),
                nocase: false,
            }],
            condition: rules::RuleCondition::AnyOf,
            metadata: HashMap::new(),
        });
        assert_eq!(engine.rule_count(), 1);

        assert!(engine.remove_rule("TestRule"));
        assert_eq!(engine.rule_count(), 0);

        // Removing a nonexistent rule returns false.
        assert!(!engine.remove_rule("DoesNotExist"));
    }

    #[test]
    fn test_rule_names() {
        let engine = YaraEngine::new();
        let names = engine.rule_names();
        assert!(names.contains(&"EICAR_test_file".to_string()));
        assert!(names.contains(&"UPX_packed_PE".to_string()));
    }

    #[test]
    fn test_stats_accumulate() {
        let engine = YaraEngine::new();
        let _ = engine.scan_bytes(b"nothing", FileType::Unknown).unwrap();
        let _ = engine.scan_bytes(b"nothing", FileType::Unknown).unwrap();
        let stats = engine.get_stats();
        assert_eq!(stats.total_scanned, 2);
    }

    #[test]
    fn test_detection_metadata_includes_mitre_id() {
        let engine = YaraEngine::new();
        let miner_data = b"stratum+tcp://pool.mine.org xmrig --donate=0";
        let detections = engine
            .scan_bytes(miner_data, FileType::Unknown)
            .unwrap();
        let miner_det = detections
            .iter()
            .find(|d| d.rule_name == "CryptoMiner_Indicators")
            .expect("miner rule should match");
        assert_eq!(miner_det.metadata.get("mitre_id").unwrap(), "T1496");
    }

    #[test]
    fn test_with_rules_dir_nonexistent() {
        // A nonexistent directory should still succeed with built-in rules.
        let engine =
            YaraEngine::with_rules_dir(Path::new("/nonexistent/path/yara_rules")).unwrap();
        assert!(engine.rule_count() >= 5);
    }

    #[test]
    fn test_with_rules_dir_empty() {
        let tmp = std::env::temp_dir().join("guardian_yara_test_empty");
        let _ = std::fs::create_dir_all(&tmp);
        let engine = YaraEngine::with_rules_dir(&tmp).unwrap();
        assert!(engine.rule_count() >= 5);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_with_rules_dir_custom_rule() {
        let tmp = std::env::temp_dir().join("guardian_yara_test_custom");
        let _ = std::fs::create_dir_all(&tmp);
        std::fs::write(
            tmp.join("custom.yar"),
            r#"
rule CustomTestRule {
    meta:
        description = "test custom rule"
        severity = "low"
    strings:
        $a = "CUSTOM_MARKER_12345"
    condition:
        any of them
}
"#,
        )
        .unwrap();

        let engine = YaraEngine::with_rules_dir(&tmp).unwrap();
        assert!(engine.rule_names().contains(&"CustomTestRule".to_string()));

        // Verify it actually matches.
        let detections = engine
            .scan_bytes(
                b"file contains CUSTOM_MARKER_12345 inside",
                FileType::Unknown,
            )
            .unwrap();
        assert!(detections.iter().any(|d| d.rule_name == "CustomTestRule"));

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
