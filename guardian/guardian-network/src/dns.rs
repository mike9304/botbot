//! DNS anomaly detection.
//!
//! Detects Domain Generation Algorithm (DGA) domains, DNS tunneling attempts,
//! and fast-flux infrastructure through entropy analysis, heuristic scoring,
//! and IP resolution tracking.

use std::collections::{HashMap, HashSet};
use std::net::IpAddr;

use chrono::{DateTime, Utc};
use guardian_common::{Detection, DetectionEngine, Severity};
use serde::{Deserialize, Serialize};
use tracing::debug;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Shannon entropy threshold for a domain label to be considered DGA-suspicious.
const DGA_ENTROPY_THRESHOLD: f64 = 3.5;

/// Query name length above which DNS tunneling is suspected.
const TUNNEL_QUERY_LENGTH: usize = 100;

/// Number of distinct IPs that a single domain must resolve to (within the
/// tracking window) to be flagged as fast-flux.
const FAST_FLUX_IP_THRESHOLD: usize = 10;

/// Maximum consonant ratio for a "normal" domain label. DGA domains tend to
/// have very high consonant ratios.
const DGA_CONSONANT_RATIO: f64 = 0.7;

/// Maximum digit ratio in a label before it looks machine-generated.
const DGA_DIGIT_RATIO: f64 = 0.4;

/// TLDs that are commonly abused by DGA / malware.
const SUSPICIOUS_TLDS: &[&str] = &[
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "work", "click", "buzz",
    "surf", "cam", "icu", "monster", "pw", "cc",
];

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// DNS record types we care about for tunneling detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DnsRecordType {
    A,
    AAAA,
    CNAME,
    MX,
    TXT,
    NS,
    SRV,
    Other,
}

/// Represents a DNS query that we want to inspect.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DnsQuery {
    /// Fully-qualified domain name being queried.
    pub qname: String,
    /// Record type requested.
    pub record_type: DnsRecordType,
    /// When the query was observed.
    pub timestamp: DateTime<Utc>,
    /// Optional: resolved IPs (for fast-flux analysis).
    pub resolved_ips: Vec<IpAddr>,
}

/// Verdict produced by analysing a DNS query or domain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DnsVerdict {
    pub domain: String,
    pub suspicious: bool,
    pub reasons: Vec<String>,
    pub severity: Severity,
    pub entropy: f64,
    pub detection: Option<Detection>,
}

// ---------------------------------------------------------------------------
// DnsAnalyzer
// ---------------------------------------------------------------------------

/// Analyses DNS queries for anomalies.
#[derive(Debug)]
pub struct DnsAnalyzer {
    /// Per-domain resolution history for fast-flux detection.
    resolution_history: HashMap<String, HashSet<IpAddr>>,
    /// Per-domain TXT query count for tunneling heuristic.
    txt_query_counts: HashMap<String, usize>,
}

impl DnsAnalyzer {
    pub fn new() -> Self {
        Self {
            resolution_history: HashMap::new(),
            txt_query_counts: HashMap::new(),
        }
    }

    /// Analyse a single DNS query and return a verdict.
    ///
    /// This also updates internal state (resolution history, TXT counts) for
    /// stateful checks such as fast-flux and tunneling detection.
    pub fn analyze_query(&mut self, query: &DnsQuery) -> DnsVerdict {
        let domain = query.qname.to_lowercase();

        // Update resolution history.
        if !query.resolved_ips.is_empty() {
            let ips = self.resolution_history.entry(domain.clone()).or_default();
            for ip in &query.resolved_ips {
                ips.insert(*ip);
            }
        }

        // Track TXT queries.
        if query.record_type == DnsRecordType::TXT {
            *self.txt_query_counts.entry(domain.clone()).or_insert(0) += 1;
        }

        self.check_domain(&domain)
    }

    /// Check a domain name against all heuristics without recording any state.
    pub fn check_domain(&self, domain: &str) -> DnsVerdict {
        let domain = domain.to_lowercase();
        let mut reasons: Vec<String> = Vec::new();
        let mut max_severity = Severity::Low;

        // ---- entropy check on each label ----
        let entropy = domain_entropy(&domain);
        if entropy > DGA_ENTROPY_THRESHOLD {
            reasons.push(format!(
                "high Shannon entropy ({:.2} > {:.1})",
                entropy, DGA_ENTROPY_THRESHOLD
            ));
            max_severity = bump_severity(max_severity, Severity::Medium);
        }

        // ---- DGA heuristics ----
        if let Some(reason) = dga_heuristic(&domain) {
            reasons.push(reason);
            max_severity = bump_severity(max_severity, Severity::High);
        }

        // ---- tunnel detection: long query name ----
        if domain.len() > TUNNEL_QUERY_LENGTH {
            reasons.push(format!(
                "query name length {} exceeds {} (possible DNS tunneling)",
                domain.len(),
                TUNNEL_QUERY_LENGTH
            ));
            max_severity = bump_severity(max_severity, Severity::High);
        }

        // ---- tunnel detection: excessive TXT queries ----
        if let Some(&count) = self.txt_query_counts.get(&domain) {
            if count > 20 {
                reasons.push(format!(
                    "excessive TXT queries ({}) for domain (possible DNS tunneling)",
                    count
                ));
                max_severity = bump_severity(max_severity, Severity::High);
            }
        }

        // ---- fast-flux ----
        if let Some(ips) = self.resolution_history.get(&domain) {
            if ips.len() >= FAST_FLUX_IP_THRESHOLD {
                reasons.push(format!(
                    "domain resolved to {} distinct IPs (fast-flux indicator)",
                    ips.len()
                ));
                max_severity = bump_severity(max_severity, Severity::High);
            }
        }

        // ---- suspicious TLD ----
        if has_suspicious_tld(&domain) {
            reasons.push("uses commonly-abused TLD".to_string());
            max_severity = bump_severity(max_severity, Severity::Low);
        }

        let suspicious = !reasons.is_empty();

        let detection = if suspicious {
            Some(build_detection(&domain, &reasons, max_severity))
        } else {
            None
        };

        if suspicious {
            debug!(domain = %domain, reason_count = reasons.len(), "DNS anomaly detected");
        }

        DnsVerdict {
            domain: domain.to_string(),
            suspicious,
            reasons,
            severity: max_severity,
            entropy,
            detection,
        }
    }

    /// Return the number of unique IPs a domain has resolved to.
    pub fn resolution_count(&self, domain: &str) -> usize {
        self.resolution_history
            .get(&domain.to_lowercase())
            .map_or(0, |s| s.len())
    }

    /// Clear all internal state.
    pub fn reset(&mut self) {
        self.resolution_history.clear();
        self.txt_query_counts.clear();
    }
}

impl Default for DnsAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Pure helper functions
// ---------------------------------------------------------------------------

/// Compute the Shannon entropy of a domain's non-dot characters.
pub fn domain_entropy(domain: &str) -> f64 {
    shannon_entropy(&domain.replace('.', ""))
}

/// Shannon entropy of a byte string.
pub fn shannon_entropy(data: &str) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let mut freq = [0u64; 256];
    for &b in data.as_bytes() {
        freq[b as usize] += 1;
    }

    let len = data.len() as f64;
    freq.iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / len;
            -p * p.log2()
        })
        .sum()
}

/// Run DGA-detection heuristics on a domain and return a reason string if
/// suspicious.
fn dga_heuristic(domain: &str) -> Option<String> {
    // Work on the second-level label (the part before the TLD).
    let label = extract_second_level_label(domain)?;

    if label.len() < 4 {
        return None; // too short to judge
    }

    let chars: Vec<char> = label.chars().collect();
    let total = chars.len() as f64;

    let consonants = "bcdfghjklmnpqrstvwxyz";
    let consonant_count = chars.iter().filter(|c| consonants.contains(**c)).count() as f64;
    let digit_count = chars.iter().filter(|c| c.is_ascii_digit()).count() as f64;

    let consonant_ratio = consonant_count / total;
    let digit_ratio = digit_count / total;

    let mut reasons = Vec::new();

    if consonant_ratio > DGA_CONSONANT_RATIO {
        reasons.push(format!("consonant ratio {:.2}", consonant_ratio));
    }
    if digit_ratio > DGA_DIGIT_RATIO {
        reasons.push(format!("digit ratio {:.2}", digit_ratio));
    }

    if reasons.is_empty() {
        None
    } else {
        Some(format!("DGA heuristic triggered: {}", reasons.join(", ")))
    }
}

/// Extract the second-level label from a FQDN.
/// E.g. `"foo.example.com"` -> `"example"`, `"abc123xyz.tk"` -> `"abc123xyz"`.
fn extract_second_level_label(domain: &str) -> Option<String> {
    let parts: Vec<&str> = domain.trim_end_matches('.').split('.').collect();
    if parts.len() < 2 {
        return None;
    }
    Some(parts[parts.len() - 2].to_string())
}

/// Check whether the domain uses a commonly-abused TLD.
fn has_suspicious_tld(domain: &str) -> bool {
    let tld = domain
        .trim_end_matches('.')
        .rsplit('.')
        .next()
        .unwrap_or("");
    SUSPICIOUS_TLDS.contains(&tld)
}

fn bump_severity(current: Severity, candidate: Severity) -> Severity {
    if candidate > current {
        candidate
    } else {
        current
    }
}

fn build_detection(domain: &str, reasons: &[String], severity: Severity) -> Detection {
    let mut metadata = HashMap::new();
    metadata.insert("domain".to_string(), domain.to_string());

    Detection {
        engine: DetectionEngine::Network,
        rule_name: "dns_anomaly".to_string(),
        description: format!(
            "DNS anomaly for {}: {}",
            domain,
            reasons.join("; ")
        ),
        severity,
        metadata,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    #[test]
    fn test_shannon_entropy_uniform() {
        // All same character -> entropy 0.
        let e = shannon_entropy("aaaaaaa");
        assert!(e.abs() < 1e-9);
    }

    #[test]
    fn test_shannon_entropy_two_chars() {
        // Equal frequency of two chars -> entropy 1.0.
        let e = shannon_entropy("abababab");
        assert!((e - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_shannon_entropy_empty() {
        assert_eq!(shannon_entropy(""), 0.0);
    }

    #[test]
    fn test_domain_entropy_normal() {
        let e = domain_entropy("google.com");
        // "googlecom" has moderate entropy.
        assert!(e < DGA_ENTROPY_THRESHOLD);
    }

    #[test]
    fn test_domain_entropy_dga() {
        // High-entropy random-looking domain.
        let e = domain_entropy("x8k2q9m4j7b3v1n5.com");
        assert!(
            e > DGA_ENTROPY_THRESHOLD,
            "entropy {:.2} should exceed {:.1}",
            e,
            DGA_ENTROPY_THRESHOLD
        );
    }

    #[test]
    fn test_check_domain_clean() {
        let analyzer = DnsAnalyzer::new();
        let verdict = analyzer.check_domain("www.google.com");
        assert!(!verdict.suspicious);
        assert!(verdict.reasons.is_empty());
        assert!(verdict.detection.is_none());
    }

    #[test]
    fn test_check_domain_high_entropy() {
        let analyzer = DnsAnalyzer::new();
        let verdict = analyzer.check_domain("a1b2c3d4e5f6g7h8i9j0.xyz");
        assert!(verdict.suspicious);
        assert!(verdict.reasons.iter().any(|r| r.contains("entropy")
            || r.contains("DGA")
            || r.contains("abused TLD")));
    }

    #[test]
    fn test_check_domain_suspicious_tld() {
        let analyzer = DnsAnalyzer::new();
        let verdict = analyzer.check_domain("something.tk");
        assert!(verdict.suspicious);
        assert!(verdict.reasons.iter().any(|r| r.contains("abused TLD")));
    }

    #[test]
    fn test_long_query_tunnel_detection() {
        let analyzer = DnsAnalyzer::new();
        let long_domain = format!(
            "{}.{}.{}.example.com",
            "a".repeat(40),
            "b".repeat(40),
            "c".repeat(40),
        );
        let verdict = analyzer.check_domain(&long_domain);
        assert!(verdict.suspicious);
        assert!(verdict.reasons.iter().any(|r| r.contains("tunneling")));
    }

    #[test]
    fn test_excessive_txt_queries() {
        let mut analyzer = DnsAnalyzer::new();
        let domain = "tunnel.example.com";

        // Simulate 25 TXT queries.
        for _ in 0..25 {
            let query = DnsQuery {
                qname: domain.to_string(),
                record_type: DnsRecordType::TXT,
                timestamp: Utc::now(),
                resolved_ips: vec![],
            };
            analyzer.analyze_query(&query);
        }

        let verdict = analyzer.check_domain(domain);
        assert!(verdict.suspicious);
        assert!(
            verdict.reasons.iter().any(|r| r.contains("TXT queries")),
            "should flag excessive TXT queries, got: {:?}",
            verdict.reasons
        );
    }

    #[test]
    fn test_fast_flux_detection() {
        let mut analyzer = DnsAnalyzer::new();
        let domain = "fastflux.example.com";

        // Resolve to 15 distinct IPs.
        for i in 1..=15u8 {
            let query = DnsQuery {
                qname: domain.to_string(),
                record_type: DnsRecordType::A,
                timestamp: Utc::now(),
                resolved_ips: vec![IpAddr::V4(Ipv4Addr::new(10, 0, 0, i))],
            };
            analyzer.analyze_query(&query);
        }

        assert_eq!(analyzer.resolution_count(domain), 15);

        let verdict = analyzer.check_domain(domain);
        assert!(verdict.suspicious);
        assert!(verdict
            .reasons
            .iter()
            .any(|r| r.contains("fast-flux")));
    }

    #[test]
    fn test_dga_consonant_ratio() {
        let analyzer = DnsAnalyzer::new();
        // Domain with extremely high consonant ratio.
        let verdict = analyzer.check_domain("bdfghjklmnpq.com");
        assert!(verdict.suspicious);
        assert!(
            verdict.reasons.iter().any(|r| r.contains("DGA") || r.contains("consonant")),
            "should flag high consonant ratio, got: {:?}",
            verdict.reasons
        );
    }

    #[test]
    fn test_dga_digit_ratio() {
        let analyzer = DnsAnalyzer::new();
        let verdict = analyzer.check_domain("12345678ab.com");
        assert!(verdict.suspicious);
        assert!(
            verdict.reasons.iter().any(|r| r.contains("DGA") || r.contains("digit")),
            "should flag high digit ratio, got: {:?}",
            verdict.reasons
        );
    }

    #[test]
    fn test_extract_second_level_label() {
        assert_eq!(
            extract_second_level_label("www.example.com"),
            Some("example".to_string())
        );
        assert_eq!(
            extract_second_level_label("example.com"),
            Some("example".to_string())
        );
        assert_eq!(extract_second_level_label("localhost"), None);
    }

    #[test]
    fn test_analyze_query_records_state() {
        let mut analyzer = DnsAnalyzer::new();
        let query = DnsQuery {
            qname: "test.example.com".to_string(),
            record_type: DnsRecordType::A,
            timestamp: Utc::now(),
            resolved_ips: vec![IpAddr::V4(Ipv4Addr::new(1, 2, 3, 4))],
        };
        let verdict = analyzer.analyze_query(&query);
        assert!(!verdict.suspicious); // normal domain
        assert_eq!(analyzer.resolution_count("test.example.com"), 1);
    }

    #[test]
    fn test_reset_clears_state() {
        let mut analyzer = DnsAnalyzer::new();
        let query = DnsQuery {
            qname: "test.example.com".to_string(),
            record_type: DnsRecordType::TXT,
            timestamp: Utc::now(),
            resolved_ips: vec![IpAddr::V4(Ipv4Addr::new(1, 2, 3, 4))],
        };
        analyzer.analyze_query(&query);
        assert_eq!(analyzer.resolution_count("test.example.com"), 1);

        analyzer.reset();
        assert_eq!(analyzer.resolution_count("test.example.com"), 0);
    }
}
