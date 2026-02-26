//! Bloom filter pre-screener for O(k) hash lookup.
//!
//! This module provides a Bloom filter that acts as the first tier of the
//! signature matching pipeline. It enables fast rejection of files whose
//! SHA-256 hashes are definitely not in the malware database, avoiding
//! expensive SQLite lookups for the vast majority of clean files.
//!
//! Design targets:
//! - Capacity: 8 million signatures
//! - RAM footprint: ~120 MB
//! - False positive rate: < 0.01% (0.0001)

use guardian_common::{GuardianError, Result};
use sha2::{Digest, Sha256};
use std::sync::RwLock;
use tracing::{debug, info};

/// Target number of signatures the bloom filter is sized for.
const DEFAULT_CAPACITY: usize = 8_000_000;

/// Target false positive rate (0.01%).
const DEFAULT_FP_RATE: f64 = 0.0001;

/// Bloom filter pre-screener wrapping the `bloomfilter` crate.
///
/// This is thread-safe via an internal `RwLock`, allowing concurrent reads
/// during scans with exclusive access only needed when inserting new hashes.
pub struct BloomPreScreener {
    filter: RwLock<bloomfilter::Bloom<[u8]>>,
    item_count: RwLock<usize>,
    capacity: usize,
    fp_rate: f64,
}

impl BloomPreScreener {
    /// Create a new bloom filter with default capacity (8M) and FP rate (0.01%).
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY, DEFAULT_FP_RATE)
    }

    /// Create a bloom filter with custom capacity and false positive rate.
    pub fn with_capacity(capacity: usize, fp_rate: f64) -> Self {
        info!(
            capacity = capacity,
            fp_rate = fp_rate,
            "Initializing bloom filter pre-screener"
        );
        let filter = bloomfilter::Bloom::new_for_fp_rate(capacity, fp_rate);
        Self {
            filter: RwLock::new(filter),
            item_count: RwLock::new(0),
            capacity,
            fp_rate,
        }
    }

    /// Compute SHA-256 of the given data and return hex-encoded digest.
    pub fn compute_hash(data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        hex::encode(hasher.finalize())
    }

    /// Insert a known-malicious SHA-256 hash (hex string) into the filter.
    pub fn insert_hash(&self, sha256_hex: &str) -> Result<()> {
        let bytes = hex::decode(sha256_hex).map_err(|e| {
            GuardianError::Signature(format!("Invalid hex hash '{}': {}", sha256_hex, e))
        })?;
        let mut filter = self.filter.write().map_err(|e| {
            GuardianError::Signature(format!("Bloom filter lock poisoned: {}", e))
        })?;
        filter.set(&bytes);
        let mut count = self.item_count.write().map_err(|e| {
            GuardianError::Signature(format!("Item count lock poisoned: {}", e))
        })?;
        *count += 1;
        Ok(())
    }

    /// Bulk-insert multiple SHA-256 hashes into the filter.
    pub fn insert_hashes(&self, hashes: &[String]) -> Result<usize> {
        let mut filter = self.filter.write().map_err(|e| {
            GuardianError::Signature(format!("Bloom filter lock poisoned: {}", e))
        })?;
        let mut count = self.item_count.write().map_err(|e| {
            GuardianError::Signature(format!("Item count lock poisoned: {}", e))
        })?;

        let mut inserted = 0usize;
        for hash_hex in hashes {
            match hex::decode(hash_hex) {
                Ok(bytes) => {
                    filter.set(&bytes);
                    *count += 1;
                    inserted += 1;
                }
                Err(e) => {
                    debug!(hash = %hash_hex, error = %e, "Skipping invalid hash");
                }
            }
        }
        info!(inserted = inserted, total = *count, "Bulk inserted hashes");
        Ok(inserted)
    }

    /// Check if a SHA-256 hash *might* exist in the filter.
    ///
    /// Returns `true` if the hash might be present (possible false positive),
    /// or `false` if the hash is definitely NOT present (no false negatives).
    pub fn maybe_contains_hash(&self, sha256_hex: &str) -> Result<bool> {
        let bytes = hex::decode(sha256_hex).map_err(|e| {
            GuardianError::Signature(format!("Invalid hex hash '{}': {}", sha256_hex, e))
        })?;
        let filter = self.filter.read().map_err(|e| {
            GuardianError::Signature(format!("Bloom filter lock poisoned: {}", e))
        })?;
        Ok(filter.check(&bytes))
    }

    /// Check raw file data: compute SHA-256 and query the bloom filter.
    ///
    /// This is the primary entry point during scanning. Returns the hex hash
    /// and a boolean indicating whether a database lookup is needed.
    pub fn check_data(&self, data: &[u8]) -> Result<(String, bool)> {
        let hash = Self::compute_hash(data);
        let maybe_present = self.maybe_contains_hash(&hash)?;
        Ok((hash, maybe_present))
    }

    /// Current number of items inserted.
    pub fn item_count(&self) -> usize {
        *self.item_count.read().unwrap_or_else(|e| e.into_inner())
    }

    /// Configured capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Configured false positive rate.
    pub fn fp_rate(&self) -> f64 {
        self.fp_rate
    }

    /// Clear the bloom filter and reset the count.
    pub fn clear(&self) -> Result<()> {
        let mut filter = self.filter.write().map_err(|e| {
            GuardianError::Signature(format!("Bloom filter lock poisoned: {}", e))
        })?;
        *filter = bloomfilter::Bloom::new_for_fp_rate(self.capacity, self.fp_rate);
        let mut count = self.item_count.write().map_err(|e| {
            GuardianError::Signature(format!("Item count lock poisoned: {}", e))
        })?;
        *count = 0;
        Ok(())
    }
}

// Safety: BloomPreScreener uses RwLock internally, so it is safe to share.
unsafe impl Send for BloomPreScreener {}
unsafe impl Sync for BloomPreScreener {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_default() {
        let bloom = BloomPreScreener::new();
        assert_eq!(bloom.capacity(), DEFAULT_CAPACITY);
        assert_eq!(bloom.item_count(), 0);
    }

    #[test]
    fn test_insert_and_check() {
        let bloom = BloomPreScreener::with_capacity(1000, 0.0001);

        // SHA-256 of "malware_sample"
        let hash = BloomPreScreener::compute_hash(b"malware_sample");
        bloom.insert_hash(&hash).unwrap();

        assert_eq!(bloom.item_count(), 1);
        assert!(bloom.maybe_contains_hash(&hash).unwrap());
    }

    #[test]
    fn test_definitely_not_present() {
        let bloom = BloomPreScreener::with_capacity(1000, 0.0001);

        let known_hash = BloomPreScreener::compute_hash(b"known_malware");
        bloom.insert_hash(&known_hash).unwrap();

        // A completely different hash should (almost certainly) not be present
        let clean_hash = BloomPreScreener::compute_hash(b"definitely_clean_file");
        // With FP rate 0.01% and only 1 item, this should be false
        assert!(!bloom.maybe_contains_hash(&clean_hash).unwrap());
    }

    #[test]
    fn test_check_data() {
        let bloom = BloomPreScreener::with_capacity(1000, 0.0001);

        let data = b"suspicious_binary_content";
        let hash = BloomPreScreener::compute_hash(data);
        bloom.insert_hash(&hash).unwrap();

        let (returned_hash, maybe_present) = bloom.check_data(data).unwrap();
        assert_eq!(returned_hash, hash);
        assert!(maybe_present);
    }

    #[test]
    fn test_check_data_clean() {
        let bloom = BloomPreScreener::with_capacity(1000, 0.0001);
        let (_, maybe_present) = bloom.check_data(b"clean_file_content").unwrap();
        assert!(!maybe_present);
    }

    #[test]
    fn test_bulk_insert() {
        let bloom = BloomPreScreener::with_capacity(1000, 0.0001);

        let hashes: Vec<String> = (0..50)
            .map(|i| BloomPreScreener::compute_hash(format!("malware_{}", i).as_bytes()))
            .collect();

        let inserted = bloom.insert_hashes(&hashes).unwrap();
        assert_eq!(inserted, 50);
        assert_eq!(bloom.item_count(), 50);

        // Verify all are present
        for hash in &hashes {
            assert!(bloom.maybe_contains_hash(hash).unwrap());
        }
    }

    #[test]
    fn test_invalid_hex_hash() {
        let bloom = BloomPreScreener::with_capacity(100, 0.01);
        let result = bloom.insert_hash("not_valid_hex_zzz");
        assert!(result.is_err());
    }

    #[test]
    fn test_clear() {
        let bloom = BloomPreScreener::with_capacity(1000, 0.0001);
        let hash = BloomPreScreener::compute_hash(b"test");
        bloom.insert_hash(&hash).unwrap();
        assert_eq!(bloom.item_count(), 1);

        bloom.clear().unwrap();
        assert_eq!(bloom.item_count(), 0);
        // After clearing, the hash should no longer be present
        assert!(!bloom.maybe_contains_hash(&hash).unwrap());
    }

    #[test]
    fn test_compute_hash_known_value() {
        // SHA-256("hello world") is a well-known test vector
        let hash = BloomPreScreener::compute_hash(b"hello world");
        assert_eq!(
            hash,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[test]
    fn test_false_positive_rate_empirical() {
        // Insert 100 items, then test 10000 random items that were NOT inserted.
        // We expect < 0.01% false positives, so out of 10000 tests, we should
        // get very few (statistically ~1 on average, rarely more than 5).
        let bloom = BloomPreScreener::with_capacity(1000, 0.0001);

        // Insert 100 known hashes
        for i in 0..100 {
            let hash = BloomPreScreener::compute_hash(format!("malware_{}", i).as_bytes());
            bloom.insert_hash(&hash).unwrap();
        }

        // Test 10000 non-inserted items
        let mut false_positives = 0;
        for i in 0..10_000 {
            let hash =
                BloomPreScreener::compute_hash(format!("clean_file_{}", i + 100_000).as_bytes());
            if bloom.maybe_contains_hash(&hash).unwrap() {
                false_positives += 1;
            }
        }

        // With 0.01% FP rate, expect ~1 false positive out of 10000.
        // Allow up to 10 to account for statistical variance.
        assert!(
            false_positives <= 10,
            "Too many false positives: {} (expected < 10)",
            false_positives
        );
    }
}
