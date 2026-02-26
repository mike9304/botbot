//! # guardian-network
//!
//! Network threat detection for Home Guardian. This crate provides three
//! complementary analysis engines:
//!
//! - **C2 beaconing detection** (`c2`) -- statistical analysis of outbound
//!   connection patterns to identify Command & Control communication.
//! - **DNS anomaly detection** (`dns`) -- entropy scoring, DGA heuristics,
//!   DNS tunneling detection, and fast-flux identification.
//! - **JA3/TLS fingerprinting** (`ja3`) -- hash-based identification of TLS
//!   client and server implementations against a malicious fingerprint database.
//!
//! All three are unified behind the [`NetworkAnalyzer`] facade which manages
//! shared state and produces [`Detection`](guardian_common::Detection) records
//! compatible with the rest of the Guardian pipeline.

pub mod c2;
pub mod dns;
pub mod ja3;

// Re-export the main user-facing types from each submodule.
pub use c2::{C2Alert, ConnectionMeta, ConnectionTracker, Destination};
pub use dns::{DnsAnalyzer, DnsQuery, DnsRecordType, DnsVerdict};
pub use ja3::{
    ClientHelloParams, Ja3Hash, ServerHelloParams, ThreatInfo, TlsFingerprinter,
};

use guardian_common::Detection;
use tracing::info;

// ---------------------------------------------------------------------------
// NetworkAnalyzer -- unified facade
// ---------------------------------------------------------------------------

/// High-level facade that owns the three network analysis engines and provides
/// convenience methods for feeding data and collecting detections.
#[derive(Debug)]
pub struct NetworkAnalyzer {
    /// C2 beaconing tracker.
    pub c2_tracker: ConnectionTracker,
    /// DNS anomaly analyser.
    pub dns_analyzer: DnsAnalyzer,
    /// TLS fingerprinter.
    pub tls_fingerprinter: TlsFingerprinter,
}

impl NetworkAnalyzer {
    /// Create a new `NetworkAnalyzer` with default settings.
    pub fn new() -> Self {
        info!("initialising NetworkAnalyzer");
        Self {
            c2_tracker: ConnectionTracker::new(),
            dns_analyzer: DnsAnalyzer::new(),
            tls_fingerprinter: TlsFingerprinter::new(),
        }
    }

    /// Create a `NetworkAnalyzer` with a custom C2 analysis window (seconds).
    pub fn with_c2_window(window_secs: i64) -> Self {
        Self {
            c2_tracker: ConnectionTracker::with_window(window_secs),
            dns_analyzer: DnsAnalyzer::new(),
            tls_fingerprinter: TlsFingerprinter::new(),
        }
    }

    // ---- C2 convenience wrappers ----

    /// Record an outbound connection for C2 analysis.
    pub fn record_connection(&mut self, meta: ConnectionMeta) {
        self.c2_tracker.record_connection(meta);
    }

    /// Run C2 beaconing analysis and return any alerts.
    pub fn check_c2_beaconing(&self) -> Vec<C2Alert> {
        self.c2_tracker.check_beaconing()
    }

    // ---- DNS convenience wrappers ----

    /// Analyse a DNS query.
    pub fn analyze_dns_query(&mut self, query: &DnsQuery) -> DnsVerdict {
        self.dns_analyzer.analyze_query(query)
    }

    /// Check a domain against DNS heuristics (stateless).
    pub fn check_domain(&self, domain: &str) -> DnsVerdict {
        self.dns_analyzer.check_domain(domain)
    }

    // ---- TLS convenience wrappers ----

    /// Compute and check a JA3 fingerprint from a ClientHello.
    pub fn check_tls_client(
        &self,
        params: &ClientHelloParams,
    ) -> (Ja3Hash, Option<ThreatInfo>) {
        self.tls_fingerprinter.fingerprint_and_check(params)
    }

    /// Compute and check a JA3S fingerprint from a ServerHello.
    pub fn check_tls_server(
        &self,
        params: &ServerHelloParams,
    ) -> (Ja3Hash, Option<ThreatInfo>) {
        self.tls_fingerprinter.fingerprint_and_check_server(params)
    }

    // ---- Aggregate ----

    /// Collect all current detections across all engines.
    ///
    /// This runs C2 beaconing checks (the DNS and TLS checks are typically
    /// run inline, but any cached state can contribute here).
    pub fn collect_detections(&self) -> Vec<Detection> {
        let mut detections = Vec::new();

        // C2 beaconing alerts.
        for alert in self.c2_tracker.check_beaconing() {
            detections.push(alert.detection);
        }

        detections
    }

    /// Prune old data from all stateful trackers.
    pub fn prune(&mut self) {
        self.c2_tracker.prune();
    }

    /// Reset all internal state.
    pub fn reset(&mut self) {
        self.c2_tracker = ConnectionTracker::new();
        self.dns_analyzer.reset();
        // TlsFingerprinter is stateless (lookup only), nothing to reset.
    }
}

impl Default for NetworkAnalyzer {
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
    use chrono::{Duration, Utc};
    use std::net::{IpAddr, Ipv4Addr};

    #[test]
    fn test_network_analyzer_creation() {
        let na = NetworkAnalyzer::new();
        assert_eq!(na.c2_tracker.destination_count(), 0);
        assert_eq!(na.c2_tracker.total_connections(), 0);
    }

    #[test]
    fn test_record_and_check_c2() {
        let mut na = NetworkAnalyzer::with_c2_window(7200);
        let ip = IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1));
        let base = Utc::now();

        for i in 0..8 {
            na.record_connection(ConnectionMeta {
                dest_ip: ip,
                dest_port: 4444,
                timestamp: base + Duration::seconds(i * 30),
                data_size: 64,
            });
        }

        let alerts = na.c2_tracker.check_beaconing_at(base + Duration::seconds(300));
        assert!(!alerts.is_empty());
    }

    #[test]
    fn test_dns_through_analyzer() {
        let mut na = NetworkAnalyzer::new();
        let query = DnsQuery {
            qname: "x8k2q9m4j7b3v1n5.tk".to_string(),
            record_type: DnsRecordType::A,
            timestamp: Utc::now(),
            resolved_ips: vec![],
        };
        let verdict = na.analyze_dns_query(&query);
        assert!(verdict.suspicious);
    }

    #[test]
    fn test_tls_through_analyzer() {
        let na = NetworkAnalyzer::new();
        let params = ClientHelloParams {
            tls_version: 771,
            cipher_suites: vec![0xc02c, 0xc02b],
            extensions: vec![0x0000, 0x0017],
            elliptic_curves: vec![0x001d],
            ec_point_formats: vec![0x00],
        };
        let (ja3, threat) = na.check_tls_client(&params);
        assert!(!ja3.hash.is_empty());
        // Normal params should not match known-bad.
        assert!(threat.is_none());
    }

    #[test]
    fn test_collect_detections_empty() {
        let na = NetworkAnalyzer::new();
        let dets = na.collect_detections();
        assert!(dets.is_empty());
    }

    #[test]
    fn test_collect_detections_with_c2() {
        let mut na = NetworkAnalyzer::with_c2_window(7200);
        let ip = IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1));
        let base = Utc::now();

        // Create regular beaconing to port 4444 (known C2 port).
        for i in 0..10 {
            na.record_connection(ConnectionMeta {
                dest_ip: ip,
                dest_port: 4444,
                timestamp: base + Duration::seconds(i * 60),
                data_size: 128,
            });
        }

        // Force check at a point where all connections are within the window.
        let alerts = na.c2_tracker.check_beaconing_at(base + Duration::seconds(700));
        assert!(!alerts.is_empty());
    }

    #[test]
    fn test_reset_clears_everything() {
        let mut na = NetworkAnalyzer::new();
        let ip = IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1));

        na.record_connection(ConnectionMeta {
            dest_ip: ip,
            dest_port: 80,
            timestamp: Utc::now(),
            data_size: 100,
        });

        let query = DnsQuery {
            qname: "example.com".to_string(),
            record_type: DnsRecordType::A,
            timestamp: Utc::now(),
            resolved_ips: vec![ip],
        };
        na.analyze_dns_query(&query);

        assert_eq!(na.c2_tracker.total_connections(), 1);
        assert_eq!(na.dns_analyzer.resolution_count("example.com"), 1);

        na.reset();

        assert_eq!(na.c2_tracker.total_connections(), 0);
        assert_eq!(na.dns_analyzer.resolution_count("example.com"), 0);
    }

    #[test]
    fn test_check_domain_stateless() {
        let na = NetworkAnalyzer::new();
        let verdict = na.check_domain("www.google.com");
        assert!(!verdict.suspicious);
    }

    #[test]
    fn test_check_tls_server() {
        let na = NetworkAnalyzer::new();
        let params = ServerHelloParams {
            tls_version: 771,
            cipher_suite: 0xc02f,
            extensions: vec![0xff01, 0x0000],
        };
        let (ja3s, threat) = na.check_tls_server(&params);
        assert!(!ja3s.hash.is_empty());
        assert!(threat.is_none());
    }
}
