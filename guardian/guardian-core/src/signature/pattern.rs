//! Aho-Corasick multi-pattern byte matcher for signature scanning.
//!
//! This module provides the third tier of the signature matching pipeline.
//! It performs multi-pattern search against the first 512 KB of file data,
//! efficiently matching hundreds of thousands of byte patterns in a single
//! pass using the Aho-Corasick algorithm.
//!
//! Features:
//! - Scans only the first 512 KB of file data (configurable)
//! - Supports raw byte patterns and hex-encoded patterns
//! - Thread-safe: the compiled automaton is immutable after construction
//! - Rebuilds the automaton when patterns change

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use guardian_common::{GuardianError, Result, Severity};
use std::sync::RwLock;
use tracing::{debug, info};

/// Default maximum number of bytes to scan from the start of a file.
const DEFAULT_SCAN_LIMIT: usize = 512 * 1024; // 512 KB

/// Metadata associated with a byte pattern.
#[derive(Debug, Clone)]
pub struct PatternEntry {
    /// Unique identifier for this pattern.
    pub id: String,
    /// The raw byte pattern to match.
    pub pattern: Vec<u8>,
    /// Human-readable threat name.
    pub name: String,
    /// Severity of this pattern match.
    pub severity: Severity,
    /// Optional malware family.
    pub family: Option<String>,
    /// Optional description of what this pattern detects.
    pub description: Option<String>,
}

/// A match result from the pattern scanner.
#[derive(Debug, Clone)]
pub struct PatternMatch {
    /// The pattern entry that matched.
    pub entry: PatternEntry,
    /// Byte offset in the scanned data where the match occurred.
    pub offset: usize,
}

/// Aho-Corasick multi-pattern matcher.
///
/// Holds a compiled automaton plus the metadata for each pattern.
/// The automaton is rebuilt whenever patterns are modified.
pub struct PatternMatcher {
    /// Compiled Aho-Corasick automaton (None if no patterns loaded).
    automaton: RwLock<Option<AhoCorasick>>,
    /// Pattern metadata, indexed to align with automaton pattern IDs.
    entries: RwLock<Vec<PatternEntry>>,
    /// Maximum bytes to scan from start of file data.
    scan_limit: usize,
}

impl PatternMatcher {
    /// Create a new empty pattern matcher with default scan limit (512 KB).
    pub fn new() -> Self {
        Self::with_scan_limit(DEFAULT_SCAN_LIMIT)
    }

    /// Create a new pattern matcher with a custom scan limit.
    pub fn with_scan_limit(scan_limit: usize) -> Self {
        info!(scan_limit = scan_limit, "Initializing pattern matcher");
        Self {
            automaton: RwLock::new(None),
            entries: RwLock::new(Vec::new()),
            scan_limit,
        }
    }

    /// Add patterns and rebuild the automaton.
    ///
    /// This replaces all existing patterns with the provided entries.
    pub fn load_patterns(&self, patterns: Vec<PatternEntry>) -> Result<()> {
        let count = patterns.len();
        info!(count = count, "Loading byte patterns");

        if patterns.is_empty() {
            let mut automaton = self.automaton.write().map_err(|e| {
                GuardianError::Signature(format!("Pattern lock poisoned: {}", e))
            })?;
            let mut entries = self.entries.write().map_err(|e| {
                GuardianError::Signature(format!("Entries lock poisoned: {}", e))
            })?;
            *automaton = None;
            entries.clear();
            return Ok(());
        }

        // Extract raw byte patterns for the automaton builder
        let raw_patterns: Vec<&[u8]> = patterns.iter().map(|p| p.pattern.as_slice()).collect();

        let ac = AhoCorasickBuilder::new()
            .match_kind(MatchKind::LeftmostFirst)
            .build(&raw_patterns)
            .map_err(|e| {
                GuardianError::Signature(format!("Failed to build Aho-Corasick automaton: {}", e))
            })?;

        let mut automaton = self.automaton.write().map_err(|e| {
            GuardianError::Signature(format!("Pattern lock poisoned: {}", e))
        })?;
        let mut entries = self.entries.write().map_err(|e| {
            GuardianError::Signature(format!("Entries lock poisoned: {}", e))
        })?;

        *automaton = Some(ac);
        *entries = patterns;

        info!(loaded = count, "Pattern automaton rebuilt");
        Ok(())
    }

    /// Add a single pattern and rebuild the automaton.
    pub fn add_pattern(&self, entry: PatternEntry) -> Result<()> {
        let mut entries = self.entries.write().map_err(|e| {
            GuardianError::Signature(format!("Entries lock poisoned: {}", e))
        })?;
        entries.push(entry);

        let raw_patterns: Vec<&[u8]> = entries.iter().map(|p| p.pattern.as_slice()).collect();

        let ac = AhoCorasickBuilder::new()
            .match_kind(MatchKind::LeftmostFirst)
            .build(&raw_patterns)
            .map_err(|e| {
                GuardianError::Signature(format!("Failed to build Aho-Corasick automaton: {}", e))
            })?;

        let mut automaton = self.automaton.write().map_err(|e| {
            GuardianError::Signature(format!("Pattern lock poisoned: {}", e))
        })?;
        *automaton = Some(ac);

        debug!(total = entries.len(), "Pattern added, automaton rebuilt");
        Ok(())
    }

    /// Scan data for matching byte patterns.
    ///
    /// Only scans the first `scan_limit` bytes of the provided data.
    /// Returns all matches found.
    pub fn scan(&self, data: &[u8]) -> Result<Vec<PatternMatch>> {
        let automaton = self.automaton.read().map_err(|e| {
            GuardianError::Signature(format!("Pattern lock poisoned: {}", e))
        })?;
        let entries = self.entries.read().map_err(|e| {
            GuardianError::Signature(format!("Entries lock poisoned: {}", e))
        })?;

        let ac = match automaton.as_ref() {
            Some(ac) => ac,
            None => return Ok(Vec::new()), // No patterns loaded
        };

        // Limit scan to the configured window
        let scan_data = if data.len() > self.scan_limit {
            &data[..self.scan_limit]
        } else {
            data
        };

        let mut matches = Vec::new();
        for mat in ac.find_iter(scan_data) {
            let pattern_idx = mat.pattern().as_usize();
            if let Some(entry) = entries.get(pattern_idx) {
                matches.push(PatternMatch {
                    entry: entry.clone(),
                    offset: mat.start(),
                });
            }
        }

        if !matches.is_empty() {
            debug!(
                match_count = matches.len(),
                scan_size = scan_data.len(),
                "Pattern matches found"
            );
        }

        Ok(matches)
    }

    /// Number of patterns currently loaded.
    pub fn pattern_count(&self) -> usize {
        self.entries
            .read()
            .map(|e| e.len())
            .unwrap_or(0)
    }

    /// Configured scan limit in bytes.
    pub fn scan_limit(&self) -> usize {
        self.scan_limit
    }

    /// Parse a hex-encoded pattern string into raw bytes.
    ///
    /// Supports patterns like "4D5A9000" (contiguous hex) or
    /// "4D 5A 90 00" (space-separated hex bytes).
    pub fn parse_hex_pattern(hex_str: &str) -> Result<Vec<u8>> {
        let cleaned: String = hex_str.chars().filter(|c| !c.is_whitespace()).collect();
        hex::decode(&cleaned).map_err(|e| {
            GuardianError::Signature(format!("Invalid hex pattern '{}': {}", hex_str, e))
        })
    }
}

// Safety: PatternMatcher uses RwLock internally, providing thread safety.
unsafe impl Send for PatternMatcher {}
unsafe impl Sync for PatternMatcher {}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(id: &str, pattern: &[u8], name: &str, severity: Severity) -> PatternEntry {
        PatternEntry {
            id: id.to_string(),
            pattern: pattern.to_vec(),
            name: name.to_string(),
            severity,
            family: None,
            description: None,
        }
    }

    #[test]
    fn test_new_empty() {
        let matcher = PatternMatcher::new();
        assert_eq!(matcher.pattern_count(), 0);
        assert_eq!(matcher.scan_limit(), DEFAULT_SCAN_LIMIT);
    }

    #[test]
    fn test_scan_no_patterns() {
        let matcher = PatternMatcher::new();
        let matches = matcher.scan(b"some file data").unwrap();
        assert!(matches.is_empty());
    }

    #[test]
    fn test_single_pattern_match() {
        let matcher = PatternMatcher::new();
        let entry = make_entry("p1", b"MALWARE", "Test.Malware", Severity::High);
        matcher.load_patterns(vec![entry]).unwrap();

        let data = b"this file contains MALWARE signature in it";
        let matches = matcher.scan(data).unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].entry.name, "Test.Malware");
        assert_eq!(matches[0].offset, 19); // "MALWARE" starts at index 19
    }

    #[test]
    fn test_multiple_pattern_matches() {
        let matcher = PatternMatcher::new();
        let patterns = vec![
            make_entry("p1", b"AAAA", "Pattern.A", Severity::Low),
            make_entry("p2", b"BBBB", "Pattern.B", Severity::Medium),
            make_entry("p3", b"CCCC", "Pattern.C", Severity::High),
        ];
        matcher.load_patterns(patterns).unwrap();

        let data = b"xxAAAAxxBBBBxxCCCCxx";
        let matches = matcher.scan(data).unwrap();
        assert_eq!(matches.len(), 3);
    }

    #[test]
    fn test_no_match() {
        let matcher = PatternMatcher::new();
        let entry = make_entry("p1", b"DEADBEEF", "Pattern.Dead", Severity::Critical);
        matcher.load_patterns(vec![entry]).unwrap();

        let data = b"this file is completely clean and benign";
        let matches = matcher.scan(data).unwrap();
        assert!(matches.is_empty());
    }

    #[test]
    fn test_scan_limit() {
        let matcher = PatternMatcher::with_scan_limit(16);
        let entry = make_entry("p1", b"EVIL", "Pattern.Evil", Severity::High);
        matcher.load_patterns(vec![entry]).unwrap();

        // Pattern is at byte 20, beyond the 16-byte scan limit
        let data = b"0123456789ABCDEF----EVIL----";
        let matches = matcher.scan(data).unwrap();
        assert!(matches.is_empty());

        // Pattern within the limit
        let data2 = b"0123EVIL56789ABC";
        let matches2 = matcher.scan(data2).unwrap();
        assert_eq!(matches2.len(), 1);
    }

    #[test]
    fn test_binary_pattern() {
        let matcher = PatternMatcher::new();
        // Match PE header magic bytes
        let entry = make_entry(
            "pe_header",
            &[0x4D, 0x5A, 0x90, 0x00],
            "Suspicious.PE",
            Severity::Low,
        );
        matcher.load_patterns(vec![entry]).unwrap();

        let data: Vec<u8> = vec![0x00, 0x00, 0x4D, 0x5A, 0x90, 0x00, 0xFF, 0xFF];
        let matches = matcher.scan(&data).unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].offset, 2);
    }

    #[test]
    fn test_add_pattern_incremental() {
        let matcher = PatternMatcher::new();

        let e1 = make_entry("p1", b"AAA", "P.A", Severity::Low);
        matcher.add_pattern(e1).unwrap();
        assert_eq!(matcher.pattern_count(), 1);

        let e2 = make_entry("p2", b"BBB", "P.B", Severity::Medium);
        matcher.add_pattern(e2).unwrap();
        assert_eq!(matcher.pattern_count(), 2);

        let data = b"xxxAAAxxxBBBxxx";
        let matches = matcher.scan(data).unwrap();
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn test_load_patterns_replaces() {
        let matcher = PatternMatcher::new();

        let first = vec![make_entry("p1", b"OLD", "P.Old", Severity::Low)];
        matcher.load_patterns(first).unwrap();
        assert_eq!(matcher.pattern_count(), 1);

        let second = vec![
            make_entry("p2", b"NEW1", "P.New1", Severity::Medium),
            make_entry("p3", b"NEW2", "P.New2", Severity::High),
        ];
        matcher.load_patterns(second).unwrap();
        assert_eq!(matcher.pattern_count(), 2);

        // Old pattern should no longer match
        let data = b"xxOLDxxNEW1xxNEW2xx";
        let matches = matcher.scan(data).unwrap();
        assert_eq!(matches.len(), 2);
        assert!(matches.iter().all(|m| m.entry.name != "P.Old"));
    }

    #[test]
    fn test_parse_hex_pattern() {
        let bytes = PatternMatcher::parse_hex_pattern("4D5A9000").unwrap();
        assert_eq!(bytes, vec![0x4D, 0x5A, 0x90, 0x00]);
    }

    #[test]
    fn test_parse_hex_pattern_with_spaces() {
        let bytes = PatternMatcher::parse_hex_pattern("4D 5A 90 00").unwrap();
        assert_eq!(bytes, vec![0x4D, 0x5A, 0x90, 0x00]);
    }

    #[test]
    fn test_parse_hex_pattern_invalid() {
        let result = PatternMatcher::parse_hex_pattern("ZZZZ");
        assert!(result.is_err());
    }

    #[test]
    fn test_load_empty_patterns() {
        let matcher = PatternMatcher::new();
        let entry = make_entry("p1", b"AAA", "P.A", Severity::Low);
        matcher.load_patterns(vec![entry]).unwrap();
        assert_eq!(matcher.pattern_count(), 1);

        // Loading empty clears everything
        matcher.load_patterns(vec![]).unwrap();
        assert_eq!(matcher.pattern_count(), 0);

        let matches = matcher.scan(b"AAA").unwrap();
        assert!(matches.is_empty());
    }

    #[test]
    fn test_match_metadata() {
        let matcher = PatternMatcher::new();
        let entry = PatternEntry {
            id: "emotet_1".to_string(),
            pattern: b"\xDE\xAD\xBE\xEF".to_vec(),
            name: "Trojan.Emotet.Gen".to_string(),
            severity: Severity::Critical,
            family: Some("Emotet".to_string()),
            description: Some("Emotet dropper signature".to_string()),
        };
        matcher.load_patterns(vec![entry]).unwrap();

        let data: Vec<u8> = vec![0x00, 0xDE, 0xAD, 0xBE, 0xEF, 0x00];
        let matches = matcher.scan(&data).unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].entry.family, Some("Emotet".to_string()));
        assert_eq!(matches[0].entry.severity, Severity::Critical);
        assert_eq!(matches[0].offset, 1);
    }
}
