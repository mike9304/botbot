//! C2 (Command & Control) beaconing detection.
//!
//! Tracks outbound connections and applies statistical analysis to detect
//! regular-interval beaconing patterns indicative of C2 communication.

use std::collections::HashMap;
use std::net::IpAddr;

use chrono::{DateTime, Duration, Utc};
use guardian_common::{Detection, DetectionEngine, Severity};
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Known ports commonly used by C2 frameworks.
const KNOWN_C2_PORTS: &[u16] = &[4444, 5555, 6666, 1337, 31337, 8443, 8080];

/// Coefficient-of-variation threshold: if stddev/mean < this value the
/// connection cadence is suspiciously regular.
const REGULARITY_THRESHOLD: f64 = 0.2;

/// Minimum number of connections to a single destination before we attempt
/// statistical beaconing analysis.
const MIN_CONNECTIONS_FOR_ANALYSIS: usize = 5;

/// Default analysis window: only consider connections from the last hour.
const ANALYSIS_WINDOW: i64 = 3600; // seconds

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Metadata for a single outbound connection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionMeta {
    pub dest_ip: IpAddr,
    pub dest_port: u16,
    pub timestamp: DateTime<Utc>,
    pub data_size: u64,
}

/// Destination key for grouping connections.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct Destination {
    pub ip: IpAddr,
    pub port: u16,
}

/// Alert raised when beaconing behaviour is detected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct C2Alert {
    pub destination: Destination,
    pub severity: Severity,
    pub description: String,
    pub interval_mean_secs: f64,
    pub interval_stddev_secs: f64,
    pub connection_count: usize,
    pub detection: Detection,
}

// ---------------------------------------------------------------------------
// ConnectionTracker
// ---------------------------------------------------------------------------

/// Tracks connections per destination and detects beaconing.
#[derive(Debug)]
pub struct ConnectionTracker {
    /// Connections grouped by destination (ip, port).
    connections: HashMap<Destination, Vec<ConnectionMeta>>,
    /// How far back in time to analyse (seconds).
    window_secs: i64,
}

impl ConnectionTracker {
    /// Create a new tracker with the default 1-hour window.
    pub fn new() -> Self {
        Self {
            connections: HashMap::new(),
            window_secs: ANALYSIS_WINDOW,
        }
    }

    /// Create a tracker with a custom analysis window (in seconds).
    pub fn with_window(window_secs: i64) -> Self {
        Self {
            connections: HashMap::new(),
            window_secs,
        }
    }

    /// Record an outbound connection.
    pub fn record_connection(&mut self, meta: ConnectionMeta) {
        let dest = Destination {
            ip: meta.dest_ip,
            port: meta.dest_port,
        };
        debug!(
            dest_ip = %meta.dest_ip,
            dest_port = meta.dest_port,
            data_size = meta.data_size,
            "recording outbound connection"
        );
        self.connections.entry(dest).or_default().push(meta);
    }

    /// Analyse recorded connections and return any C2 beaconing alerts.
    ///
    /// Only connections within the analysis window (relative to `now`) are
    /// considered. Each destination is evaluated independently.
    pub fn check_beaconing(&self) -> Vec<C2Alert> {
        self.check_beaconing_at(Utc::now())
    }

    /// Like [`check_beaconing`] but allows the caller to specify *now* for
    /// deterministic testing.
    pub fn check_beaconing_at(&self, now: DateTime<Utc>) -> Vec<C2Alert> {
        let mut alerts = Vec::new();
        let cutoff = now - Duration::seconds(self.window_secs);

        for (dest, conns) in &self.connections {
            // Filter to the analysis window and sort by time.
            let mut recent: Vec<&ConnectionMeta> = conns
                .iter()
                .filter(|c| c.timestamp >= cutoff)
                .collect();

            if recent.len() < MIN_CONNECTIONS_FOR_ANALYSIS {
                continue;
            }

            recent.sort_by_key(|c| c.timestamp);

            // ------ known C2 port check ------
            if KNOWN_C2_PORTS.contains(&dest.port) {
                let det = build_detection(
                    &format!(
                        "Connection to known C2 port {} on {}",
                        dest.port, dest.ip
                    ),
                    Severity::High,
                );
                alerts.push(C2Alert {
                    destination: dest.clone(),
                    severity: Severity::High,
                    description: det.description.clone(),
                    interval_mean_secs: 0.0,
                    interval_stddev_secs: 0.0,
                    connection_count: recent.len(),
                    detection: det,
                });
            }

            // ------ regularity analysis ------
            if let Some(alert) = self.analyse_intervals(dest, &recent) {
                alerts.push(alert);
            }
        }

        if !alerts.is_empty() {
            warn!(alert_count = alerts.len(), "C2 beaconing alerts generated");
        }

        alerts
    }

    /// Prune connections older than the analysis window.
    pub fn prune(&mut self) {
        self.prune_at(Utc::now());
    }

    /// Prune with an explicit *now*.
    pub fn prune_at(&mut self, now: DateTime<Utc>) {
        let cutoff = now - Duration::seconds(self.window_secs);
        for conns in self.connections.values_mut() {
            conns.retain(|c| c.timestamp >= cutoff);
        }
        self.connections.retain(|_, v| !v.is_empty());
    }

    /// Number of tracked destinations.
    pub fn destination_count(&self) -> usize {
        self.connections.len()
    }

    /// Total number of recorded connections across all destinations.
    pub fn total_connections(&self) -> usize {
        self.connections.values().map(|v| v.len()).sum()
    }

    // ----- internal helpers -----

    fn analyse_intervals(
        &self,
        dest: &Destination,
        recent: &[&ConnectionMeta],
    ) -> Option<C2Alert> {
        if recent.len() < 2 {
            return None;
        }

        // Compute inter-arrival intervals in seconds.
        let intervals: Vec<f64> = recent
            .windows(2)
            .map(|w| {
                (w[1].timestamp - w[0].timestamp)
                    .num_milliseconds() as f64
                    / 1000.0
            })
            .collect();

        let (mean, stddev) = mean_stddev(&intervals);

        // Guard against division by zero (all connections at the exact same
        // instant).
        if mean <= 0.0 {
            return None;
        }

        let cv = stddev / mean; // coefficient of variation

        if cv < REGULARITY_THRESHOLD {
            let severity = if cv < 0.05 {
                Severity::Critical
            } else if cv < 0.1 {
                Severity::High
            } else {
                Severity::Medium
            };

            let desc = format!(
                "Beaconing detected to {}:{} \u{2014} mean interval {:.1}s, \
                 stddev {:.2}s (CV={:.3}), {} connections",
                dest.ip,
                dest.port,
                mean,
                stddev,
                cv,
                recent.len(),
            );

            let det = build_detection(&desc, severity);

            return Some(C2Alert {
                destination: dest.clone(),
                severity,
                description: desc,
                interval_mean_secs: mean,
                interval_stddev_secs: stddev,
                connection_count: recent.len(),
                detection: det,
            });
        }

        None
    }
}

impl Default for ConnectionTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute mean and population standard deviation.
fn mean_stddev(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    (mean, variance.sqrt())
}

fn build_detection(description: &str, severity: Severity) -> Detection {
    Detection {
        engine: DetectionEngine::Network,
        rule_name: "c2_beaconing".to_string(),
        description: description.to_string(),
        severity,
        metadata: HashMap::new(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    fn make_conn(ip: IpAddr, port: u16, ts: DateTime<Utc>, size: u64) -> ConnectionMeta {
        ConnectionMeta {
            dest_ip: ip,
            dest_port: port,
            timestamp: ts,
            data_size: size,
        }
    }

    #[test]
    fn test_mean_stddev_basic() {
        let vals = vec![10.0, 10.0, 10.0, 10.0];
        let (m, s) = mean_stddev(&vals);
        assert!((m - 10.0).abs() < 1e-9);
        assert!(s.abs() < 1e-9);
    }

    #[test]
    fn test_mean_stddev_varied() {
        let vals = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let (m, _s) = mean_stddev(&vals);
        assert!((m - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_mean_stddev_empty() {
        let (m, s) = mean_stddev(&[]);
        assert_eq!(m, 0.0);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn test_regular_beaconing_detected() {
        let mut tracker = ConnectionTracker::with_window(7200);
        let ip = IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1));
        let base = Utc::now();

        // 10 connections exactly 60 seconds apart -> perfectly regular
        for i in 0..10 {
            tracker.record_connection(make_conn(
                ip,
                9999,
                base + Duration::seconds(i * 60),
                128,
            ));
        }

        let alerts = tracker.check_beaconing_at(base + Duration::seconds(600));
        assert!(
            !alerts.is_empty(),
            "should detect perfectly regular beaconing"
        );

        let alert = &alerts[0];
        assert!((alert.interval_mean_secs - 60.0).abs() < 1.0);
        assert!(alert.interval_stddev_secs < 1.0);
        assert!(alert.severity == Severity::Critical || alert.severity == Severity::High);
    }

    #[test]
    fn test_irregular_traffic_no_alert() {
        let mut tracker = ConnectionTracker::with_window(7200);
        let ip = IpAddr::V4(Ipv4Addr::new(10, 0, 0, 2));
        let base = Utc::now();

        // Random-ish intervals: 5, 120, 3, 200, 45, 90, 12, 300 seconds
        let offsets = [0, 5, 125, 128, 328, 373, 463, 475, 775];
        for &off in &offsets {
            tracker.record_connection(make_conn(
                ip,
                443,
                base + Duration::seconds(off),
                256,
            ));
        }

        let alerts = tracker.check_beaconing_at(base + Duration::seconds(800));
        // Filter to only regularity alerts (not port alerts).
        let regularity_alerts: Vec<_> = alerts
            .iter()
            .filter(|a| a.interval_mean_secs > 0.0)
            .collect();
        assert!(
            regularity_alerts.is_empty(),
            "irregular traffic should not trigger beaconing alert"
        );
    }

    #[test]
    fn test_known_c2_port_alert() {
        let mut tracker = ConnectionTracker::with_window(7200);
        let ip = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 100));
        let base = Utc::now();

        // Only 5 connections to port 4444 (known C2 port), irregular timing.
        let offsets = [0, 10, 55, 300, 900];
        for &off in &offsets {
            tracker.record_connection(make_conn(
                ip,
                4444,
                base + Duration::seconds(off),
                64,
            ));
        }

        let alerts = tracker.check_beaconing_at(base + Duration::seconds(1000));
        let port_alerts: Vec<_> = alerts
            .iter()
            .filter(|a| a.description.contains("known C2 port"))
            .collect();
        assert!(!port_alerts.is_empty(), "should flag known C2 port 4444");
        assert_eq!(port_alerts[0].severity, Severity::High);
    }

    #[test]
    fn test_too_few_connections_no_alert() {
        let mut tracker = ConnectionTracker::with_window(7200);
        let ip = IpAddr::V4(Ipv4Addr::new(10, 0, 0, 3));
        let base = Utc::now();

        // Only 3 connections -- below the threshold.
        for i in 0..3 {
            tracker.record_connection(make_conn(
                ip,
                9999,
                base + Duration::seconds(i * 60),
                128,
            ));
        }

        let alerts = tracker.check_beaconing_at(base + Duration::seconds(200));
        assert!(alerts.is_empty(), "too few connections should not alert");
    }

    #[test]
    fn test_prune_removes_old() {
        let mut tracker = ConnectionTracker::with_window(3600);
        let ip = IpAddr::V4(Ipv4Addr::new(10, 0, 0, 4));
        let now = Utc::now();

        // One old, one recent.
        tracker.record_connection(make_conn(
            ip,
            80,
            now - Duration::seconds(7200),
            100,
        ));
        tracker.record_connection(make_conn(ip, 80, now, 100));

        assert_eq!(tracker.total_connections(), 2);
        tracker.prune_at(now);
        assert_eq!(tracker.total_connections(), 1);
    }

    #[test]
    fn test_destination_count() {
        let mut tracker = ConnectionTracker::new();
        let now = Utc::now();

        tracker.record_connection(make_conn(
            IpAddr::V4(Ipv4Addr::new(1, 1, 1, 1)),
            443,
            now,
            64,
        ));
        tracker.record_connection(make_conn(
            IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8)),
            53,
            now,
            64,
        ));
        tracker.record_connection(make_conn(
            IpAddr::V4(Ipv4Addr::new(1, 1, 1, 1)),
            443,
            now,
            128,
        ));

        assert_eq!(tracker.destination_count(), 2);
        assert_eq!(tracker.total_connections(), 3);
    }

    #[test]
    fn test_beaconing_with_small_jitter() {
        let mut tracker = ConnectionTracker::with_window(7200);
        let ip = IpAddr::V4(Ipv4Addr::new(10, 0, 0, 5));
        let base = Utc::now();

        // 60-second intervals with very small jitter (+/- 1 second).
        let offsets = [0, 59, 121, 179, 241, 299, 361, 419, 480, 540];
        for &off in &offsets {
            tracker.record_connection(make_conn(
                ip,
                12345,
                base + Duration::seconds(off),
                100,
            ));
        }

        let alerts = tracker.check_beaconing_at(base + Duration::seconds(600));
        let regularity_alerts: Vec<_> = alerts
            .iter()
            .filter(|a| a.interval_mean_secs > 0.0)
            .collect();
        assert!(
            !regularity_alerts.is_empty(),
            "near-regular beaconing with small jitter should be detected"
        );
    }
}
