//! Signature matching engine — 3-tier detection pipeline.
//!
//! This module implements the signature-based detection layer for Home Guardian.
//! It combines three complementary techniques in a tiered architecture:
//!
//! 1. **Bloom filter pre-screener** (`bloom`): O(k) probabilistic hash check.
//!    Rejects the vast majority of clean files without touching the database.
//!
//! 2. **SHA-256 hash database** (`hash_db`): SQLite-backed authoritative lookup.
//!    Confirms bloom filter hits against the full signature database with metadata.
//!
//! 3. **Aho-Corasick pattern matcher** (`pattern`): Multi-pattern byte search.
//!    Scans file contents for known-malicious byte sequences in a single pass
//!    over the first 512 KB.
//!
//! The [`SignatureEngine`] struct orchestrates all three tiers and implements
//! the [`DetectionLayer`](crate::DetectionLayer) trait for integration with
//! the unified scanning pipeline.

pub mod bloom;
pub mod hash_db;
pub mod pattern;

pub use bloom::BloomPreScreener;
pub use hash_db::{HashDatabase, SignatureRecord};
pub use pattern::{PatternEntry, PatternMatch, PatternMatcher};

use crate::DetectionLayer;
use guardian_common::{Detection, DetectionEngine, EngineStats, FileType, Result};
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{debug, info};

/// The unified signature matching engine combining all three tiers.
///
/// Scanning flow:
/// 1. Compute SHA-256 of input data.
/// 2. Query bloom filter — if negative, skip hash DB (definite non-match).
/// 3. If bloom filter says "maybe", query the SQLite hash database for
///    authoritative match with full threat metadata.
/// 4. Regardless of hash results, run Aho-Corasick pattern matching
///    against file byte content.
/// 5. Aggregate and return all detections.
pub struct SignatureEngine {
    /// Bloom filter for fast negative lookups.
    bloom: BloomPreScreener,
    /// SQLite hash database for confirmed matches.
    hash_db: HashDatabase,
    /// Aho-Corasick multi-pattern byte matcher.
    pattern_matcher: PatternMatcher,
    /// Running count of total files scanned.
    total_scanned: AtomicU64,
    /// Running count of total detections produced.
    total_detections: AtomicU64,
    /// Cumulative scan time in microseconds (for average calculation).
    total_scan_time_us: AtomicU64,
}

impl SignatureEngine {
    /// Create a new signature engine backed by a database file on disk.
    ///
    /// This opens (or creates) the SQLite database at `db_path`, initializes
    /// the bloom filter, and prepares the pattern matcher. The bloom filter
    /// is automatically populated from existing database entries.
    pub fn new(db_path: &Path) -> Result<Self> {
        info!(db_path = %db_path.display(), "Initializing signature engine");

        let hash_db = HashDatabase::open(db_path)?;
        let bloom = BloomPreScreener::new();
        let pattern_matcher = PatternMatcher::new();

        // Populate bloom filter from existing database entries
        let existing_hashes = hash_db.all_hashes()?;
        if !existing_hashes.is_empty() {
            let loaded = bloom.insert_hashes(&existing_hashes)?;
            info!(
                loaded = loaded,
                "Populated bloom filter from existing database"
            );
        }

        Ok(Self {
            bloom,
            hash_db,
            pattern_matcher,
            total_scanned: AtomicU64::new(0),
            total_detections: AtomicU64::new(0),
            total_scan_time_us: AtomicU64::new(0),
        })
    }

    /// Create a new signature engine with an in-memory database (for testing).
    pub fn new_in_memory() -> Result<Self> {
        info!("Initializing in-memory signature engine");

        let hash_db = HashDatabase::open_in_memory()?;
        let bloom = BloomPreScreener::with_capacity(10_000, 0.0001);
        let pattern_matcher = PatternMatcher::new();

        Ok(Self {
            bloom,
            hash_db,
            pattern_matcher,
            total_scanned: AtomicU64::new(0),
            total_detections: AtomicU64::new(0),
            total_scan_time_us: AtomicU64::new(0),
        })
    }

    /// Add a hash-based signature to both the database and bloom filter.
    pub fn add_hash_signature(&self, record: &SignatureRecord) -> Result<()> {
        self.hash_db.insert(record)?;
        self.bloom.insert_hash(&record.sha256)?;
        debug!(
            sha256 = %record.sha256,
            name = %record.name,
            "Added hash signature"
        );
        Ok(())
    }

    /// Batch-add hash signatures to both the database and bloom filter.
    ///
    /// Returns the number of signatures successfully loaded.
    pub fn load_hash_signatures(&self, records: &[SignatureRecord]) -> Result<usize> {
        let db_count = self.hash_db.batch_insert(records)?;

        let hashes: Vec<String> = records.iter().map(|r| r.sha256.clone()).collect();
        let bloom_count = self.bloom.insert_hashes(&hashes)?;

        info!(
            db_inserted = db_count,
            bloom_inserted = bloom_count,
            "Loaded hash signatures"
        );
        Ok(db_count)
    }

    /// Load byte patterns into the Aho-Corasick matcher.
    pub fn load_byte_patterns(&self, patterns: Vec<PatternEntry>) -> Result<()> {
        let count = patterns.len();
        self.pattern_matcher.load_patterns(patterns)?;
        info!(count = count, "Loaded byte patterns");
        Ok(())
    }

    /// Get a reference to the bloom filter.
    pub fn bloom_filter(&self) -> &BloomPreScreener {
        &self.bloom
    }

    /// Get a reference to the hash database.
    pub fn hash_database(&self) -> &HashDatabase {
        &self.hash_db
    }

    /// Get a reference to the pattern matcher.
    pub fn pattern_matcher(&self) -> &PatternMatcher {
        &self.pattern_matcher
    }

    /// Collect engine statistics.
    pub fn stats(&self) -> EngineStats {
        let scanned = self.total_scanned.load(Ordering::Relaxed);
        let detections = self.total_detections.load(Ordering::Relaxed);
        let total_us = self.total_scan_time_us.load(Ordering::Relaxed);

        let avg_ms = if scanned > 0 {
            (total_us as f64 / scanned as f64) / 1000.0
        } else {
            0.0
        };

        let sig_count = self.hash_db.count().unwrap_or(0)
            + self.pattern_matcher.pattern_count() as u64;

        EngineStats {
            total_scanned: scanned,
            total_detections: detections,
            avg_scan_time_ms: avg_ms,
            signatures_loaded: sig_count,
        }
    }

    /// Run the 3-tier scan pipeline on raw byte data.
    fn run_scan(&self, data: &[u8]) -> Result<Vec<Detection>> {
        let start = std::time::Instant::now();
        let mut detections = Vec::new();

        // ── Tier 1 + 2: Hash-based detection ───────────────────────────
        let (sha256, bloom_hit) = self.bloom.check_data(data)?;

        if bloom_hit {
            debug!(sha256 = %sha256, "Bloom filter hit — querying hash database");

            // Tier 2: Authoritative SQLite lookup
            if let Some(record) = self.hash_db.lookup(&sha256)? {
                info!(
                    sha256 = %sha256,
                    name = %record.name,
                    severity = ?record.severity,
                    "Hash signature match confirmed"
                );

                let mut metadata = HashMap::new();
                metadata.insert("sha256".to_string(), sha256.clone());
                metadata.insert("match_type".to_string(), "hash".to_string());
                if let Some(ref family) = record.family {
                    metadata.insert("family".to_string(), family.clone());
                }
                if let Some(ref first_seen) = record.first_seen {
                    metadata.insert("first_seen".to_string(), first_seen.clone());
                }

                detections.push(Detection {
                    engine: DetectionEngine::Signature,
                    rule_name: record.name.clone(),
                    description: format!(
                        "Known malicious file hash: {}",
                        &sha256[..16]
                    ),
                    severity: record.severity,
                    metadata,
                });
            } else {
                debug!(sha256 = %sha256, "Bloom filter false positive — not in database");
            }
        }

        // ── Tier 3: Byte pattern matching ──────────────────────────────
        let pattern_matches = self.pattern_matcher.scan(data)?;
        for pm in pattern_matches {
            let mut metadata = HashMap::new();
            metadata.insert("match_type".to_string(), "pattern".to_string());
            metadata.insert("offset".to_string(), pm.offset.to_string());
            metadata.insert("pattern_id".to_string(), pm.entry.id.clone());
            if let Some(ref family) = pm.entry.family {
                metadata.insert("family".to_string(), family.clone());
            }

            let description = pm
                .entry
                .description
                .clone()
                .unwrap_or_else(|| {
                    format!(
                        "Byte pattern '{}' matched at offset {}",
                        pm.entry.id, pm.offset
                    )
                });

            detections.push(Detection {
                engine: DetectionEngine::Signature,
                rule_name: pm.entry.name.clone(),
                description,
                severity: pm.entry.severity,
                metadata,
            });
        }

        // ── Update statistics ──────────────────────────────────────────
        let elapsed = start.elapsed();
        self.total_scanned.fetch_add(1, Ordering::Relaxed);
        self.total_detections
            .fetch_add(detections.len() as u64, Ordering::Relaxed);
        self.total_scan_time_us
            .fetch_add(elapsed.as_micros() as u64, Ordering::Relaxed);

        Ok(detections)
    }
}

impl DetectionLayer for SignatureEngine {
    fn name(&self) -> &str {
        "Signature Engine"
    }

    fn engine_type(&self) -> DetectionEngine {
        DetectionEngine::Signature
    }

    fn scan_bytes(&self, data: &[u8], _file_type: FileType) -> Result<Vec<Detection>> {
        self.run_scan(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use guardian_common::Severity;

    /// Helper: create a test engine with some preloaded signatures.
    fn test_engine() -> SignatureEngine {
        let engine = SignatureEngine::new_in_memory().unwrap();

        // Add some hash signatures
        let test_data = b"known_malware_binary_content";
        let hash = BloomPreScreener::compute_hash(test_data);
        engine
            .add_hash_signature(&SignatureRecord {
                sha256: hash,
                name: "Trojan.TestMalware".to_string(),
                severity: Severity::Critical,
                family: Some("TestFamily".to_string()),
                first_seen: Some("2025-01-01T00:00:00Z".to_string()),
            })
            .unwrap();

        // Add some byte patterns
        engine
            .load_byte_patterns(vec![
                PatternEntry {
                    id: "eicar".to_string(),
                    pattern: b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR".to_vec(),
                    name: "EICAR.TestFile".to_string(),
                    severity: Severity::Low,
                    family: None,
                    description: Some("EICAR test file signature".to_string()),
                },
                PatternEntry {
                    id: "suspicious_api".to_string(),
                    pattern: b"CreateRemoteThread".to_vec(),
                    name: "Suspicious.APICall".to_string(),
                    severity: Severity::Medium,
                    family: None,
                    description: Some("Suspicious API call pattern".to_string()),
                },
            ])
            .unwrap();

        engine
    }

    #[test]
    fn test_new_in_memory() {
        let engine = SignatureEngine::new_in_memory().unwrap();
        assert_eq!(engine.name(), "Signature Engine");
        assert_eq!(engine.engine_type(), DetectionEngine::Signature);
    }

    #[test]
    fn test_clean_file() {
        let engine = test_engine();
        let clean_data = b"this is a perfectly clean file with no threats";
        let detections = engine.scan_bytes(clean_data, FileType::Unknown).unwrap();
        assert!(detections.is_empty());
    }

    #[test]
    fn test_hash_match() {
        let engine = test_engine();
        let malware_data = b"known_malware_binary_content";
        let detections = engine.scan_bytes(malware_data, FileType::PE).unwrap();

        assert!(!detections.is_empty());
        let hash_det = detections
            .iter()
            .find(|d| d.metadata.get("match_type") == Some(&"hash".to_string()));
        assert!(hash_det.is_some());

        let det = hash_det.unwrap();
        assert_eq!(det.rule_name, "Trojan.TestMalware");
        assert_eq!(det.severity, Severity::Critical);
        assert_eq!(det.engine, DetectionEngine::Signature);
    }

    #[test]
    fn test_pattern_match() {
        let engine = test_engine();
        let data = b"binary code calls CreateRemoteThread and does bad things";
        let detections = engine.scan_bytes(data, FileType::PE).unwrap();

        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].rule_name, "Suspicious.APICall");
        assert_eq!(detections[0].severity, Severity::Medium);
        assert_eq!(
            detections[0].metadata.get("match_type"),
            Some(&"pattern".to_string())
        );
    }

    #[test]
    fn test_combined_hash_and_pattern() {
        let engine = SignatureEngine::new_in_memory().unwrap();

        // Create data that matches both hash AND contains a pattern
        let data = b"data_with_EVIL_pattern_inside";
        let hash = BloomPreScreener::compute_hash(data);

        engine
            .add_hash_signature(&SignatureRecord {
                sha256: hash,
                name: "Trojan.Combo".to_string(),
                severity: Severity::High,
                family: None,
                first_seen: None,
            })
            .unwrap();

        engine
            .load_byte_patterns(vec![PatternEntry {
                id: "evil".to_string(),
                pattern: b"EVIL".to_vec(),
                name: "Pattern.Evil".to_string(),
                severity: Severity::Medium,
                family: None,
                description: None,
            }])
            .unwrap();

        let detections = engine.scan_bytes(data, FileType::Unknown).unwrap();
        assert_eq!(detections.len(), 2);

        // One hash match and one pattern match
        let hash_matches: Vec<_> = detections
            .iter()
            .filter(|d| d.metadata.get("match_type") == Some(&"hash".to_string()))
            .collect();
        let pattern_matches: Vec<_> = detections
            .iter()
            .filter(|d| d.metadata.get("match_type") == Some(&"pattern".to_string()))
            .collect();
        assert_eq!(hash_matches.len(), 1);
        assert_eq!(pattern_matches.len(), 1);
    }

    #[test]
    fn test_detection_layer_trait() {
        let engine = test_engine();

        // Verify trait methods work correctly
        assert_eq!(engine.name(), "Signature Engine");
        assert_eq!(engine.engine_type(), DetectionEngine::Signature);

        // Can be used as a trait object
        let layer: &dyn DetectionLayer = &engine;
        let result = layer.scan_bytes(b"clean", FileType::Unknown).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_stats_initial() {
        let engine = SignatureEngine::new_in_memory().unwrap();
        let stats = engine.stats();
        assert_eq!(stats.total_scanned, 0);
        assert_eq!(stats.total_detections, 0);
        assert_eq!(stats.avg_scan_time_ms, 0.0);
    }

    #[test]
    fn test_stats_after_scans() {
        let engine = test_engine();

        // Scan a clean file
        engine.scan_bytes(b"clean_file", FileType::Unknown).unwrap();
        // Scan a file with a pattern match
        engine
            .scan_bytes(b"has CreateRemoteThread call", FileType::PE)
            .unwrap();

        let stats = engine.stats();
        assert_eq!(stats.total_scanned, 2);
        assert_eq!(stats.total_detections, 1);
        assert!(stats.avg_scan_time_ms >= 0.0);
    }

    #[test]
    fn test_load_hash_signatures_batch() {
        let engine = SignatureEngine::new_in_memory().unwrap();

        let records: Vec<SignatureRecord> = (0..50)
            .map(|i| SignatureRecord {
                sha256: format!("{:064x}", i),
                name: format!("Malware.Gen.{}", i),
                severity: Severity::High,
                family: Some("GenericFamily".to_string()),
                first_seen: None,
            })
            .collect();

        let loaded = engine.load_hash_signatures(&records).unwrap();
        assert_eq!(loaded, 50);

        let stats = engine.stats();
        assert!(stats.signatures_loaded >= 50);
    }

    #[test]
    fn test_bloom_filter_false_positive_handled() {
        // Even if the bloom filter says "maybe", the hash DB won't confirm
        // a match for data that's not actually in the database.
        let engine = SignatureEngine::new_in_memory().unwrap();

        // Add a hash to bloom filter via the engine
        let malware = b"actual_malware";
        let hash = BloomPreScreener::compute_hash(malware);
        engine
            .add_hash_signature(&SignatureRecord {
                sha256: hash,
                name: "Real.Malware".to_string(),
                severity: Severity::Critical,
                family: None,
                first_seen: None,
            })
            .unwrap();

        // Scan different data (bloom filter should say "no" for this)
        let clean = b"completely_different_clean_data";
        let detections = engine.scan_bytes(clean, FileType::Unknown).unwrap();
        assert!(detections.is_empty());
    }

    #[test]
    fn test_new_with_file_db() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_sigs.db");

        let engine = SignatureEngine::new(&db_path).unwrap();
        engine
            .add_hash_signature(&SignatureRecord {
                sha256: "a".repeat(64),
                name: "Test.Sig".to_string(),
                severity: Severity::Medium,
                family: None,
                first_seen: None,
            })
            .unwrap();

        let stats = engine.stats();
        assert!(stats.signatures_loaded >= 1);
    }

    #[test]
    fn test_file_db_persistence_and_bloom_reload() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("persist_sigs.db");

        let data = b"persist_test_malware_data";
        let hash = BloomPreScreener::compute_hash(data);

        // First engine instance: insert a signature
        {
            let engine = SignatureEngine::new(&db_path).unwrap();
            engine
                .add_hash_signature(&SignatureRecord {
                    sha256: hash.clone(),
                    name: "Persist.Test".to_string(),
                    severity: Severity::High,
                    family: None,
                    first_seen: None,
                })
                .unwrap();
        }

        // Second engine instance: bloom filter should be repopulated from DB
        {
            let engine = SignatureEngine::new(&db_path).unwrap();
            let detections = engine.scan_bytes(data, FileType::Unknown).unwrap();
            assert_eq!(detections.len(), 1);
            assert_eq!(detections[0].rule_name, "Persist.Test");
        }
    }

    #[test]
    fn test_scan_bytes_as_trait_object() {
        let engine = test_engine();
        let boxed: Box<dyn DetectionLayer> = Box::new(engine);

        let malware = b"known_malware_binary_content";
        let detections = boxed.scan_bytes(malware, FileType::PE).unwrap();
        assert!(!detections.is_empty());
    }
}
