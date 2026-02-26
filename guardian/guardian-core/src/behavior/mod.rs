//! Behavioural pattern matching engine.
//!
//! Ingests runtime events (file operations, process creation, network activity,
//! etc.), stores them in a time-windowed ring buffer, and continuously matches
//! against a library of MITRE ATT&CK behavioural patterns.
//!
//! # Architecture
//!
//! ```text
//!  OS hooks / audit log
//!         │
//!         ▼
//!   BehaviorEngine::ingest()
//!         │
//!         ├──▶ RingBuffer (stores recent events)
//!         │
//!         └──▶ pattern matching (BehaviorPattern library)
//!                   │
//!                   ▼
//!              BehaviorAlert (with MITRE ATT&CK ID)
//! ```

pub mod patterns;
pub mod ring_buffer;

use chrono::{DateTime, Utc};
use guardian_common::{Detection, DetectionEngine, GuardianError, Result, Severity};
use patterns::{builtin_patterns, BehaviorPattern};
use ring_buffer::RingBuffer;
use std::collections::HashMap;
use std::sync::Mutex;
use tracing::{debug, info};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Classification of behavioural events observed at runtime.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BehaviorEventKind {
    /// A new file was created.
    FileCreate,
    /// An existing file was modified (write, truncate, rename).
    FileModify,
    /// A file was deleted.
    FileDelete,
    /// A new process was spawned.
    ProcessCreate,
    /// Code was injected into another process (WriteProcessMemory, ptrace, etc.).
    ProcessInject,
    /// An outbound network connection was established.
    NetworkConnect,
    /// A registry key or value was modified (Windows-specific).
    RegistryModify,
    /// Credentials were accessed (LSASS, SAM, /etc/shadow, keychain).
    CredentialAccess,
    /// Privilege escalation was attempted.
    PrivilegeEscalation,
}

/// Information about the process that generated an event.
#[derive(Debug, Clone)]
pub struct ProcessInfo {
    /// Process ID.
    pub pid: u32,
    /// Process name / executable basename.
    pub name: String,
    /// Full path to the executable.
    pub path: String,
    /// User ID that owns the process.
    pub uid: u32,
}

/// A complete behavioural event submitted to the engine.
#[derive(Debug, Clone)]
pub struct BehaviorEvent {
    /// What kind of event occurred.
    pub kind: BehaviorEventKind,
    /// When the event was observed.
    pub timestamp: DateTime<Utc>,
    /// The process that generated this event, if known.
    pub process: Option<ProcessInfo>,
    /// Free-form metadata (e.g. file path, destination IP, registry key).
    pub details: HashMap<String, String>,
}

/// An alert produced when a behavioural pattern matches.
#[derive(Debug, Clone)]
pub struct BehaviorAlert {
    /// MITRE ATT&CK technique ID.
    pub mitre_id: String,
    /// Human-readable technique name.
    pub name: String,
    /// Detailed description.
    pub description: String,
    /// Severity of this alert.
    pub severity: Severity,
    /// The event kinds that matched (as debug strings).
    pub matched_events: Vec<String>,
    /// When the last matching event occurred.
    pub timestamp: DateTime<Utc>,
    /// Process info from the matching events, if available.
    pub process_info: Option<ProcessInfo>,
}

impl BehaviorAlert {
    /// Convert this alert into a [`Detection`] suitable for the scanner pipeline.
    pub fn to_detection(&self) -> Detection {
        let mut metadata = HashMap::new();
        metadata.insert("mitre_id".to_string(), self.mitre_id.clone());
        metadata.insert(
            "matched_events".to_string(),
            format!("{:?}", self.matched_events),
        );
        if let Some(ref proc) = self.process_info {
            metadata.insert("process_name".to_string(), proc.name.clone());
            metadata.insert("process_pid".to_string(), proc.pid.to_string());
            metadata.insert("process_path".to_string(), proc.path.clone());
        }

        Detection {
            engine: DetectionEngine::Behavioral,
            rule_name: format!("BEHAVIOR/{}", self.mitre_id),
            description: self.description.clone(),
            severity: self.severity,
            metadata,
        }
    }
}

// ---------------------------------------------------------------------------
// BehaviorEngine
// ---------------------------------------------------------------------------

/// The main behavioural analysis engine.
///
/// Thread-safe via internal `Mutex`. Ingests events, stores them in a ring
/// buffer, and evaluates the event stream against MITRE ATT&CK patterns.
pub struct BehaviorEngine {
    /// Event ring buffer (guarded by mutex for thread safety).
    events: Mutex<RingBuffer<BehaviorEvent>>,
    /// Pattern library.
    patterns: Vec<BehaviorPattern>,
    /// Alerts already fired (keyed by mitre_id) to avoid duplicate alerts
    /// within the same time window.
    fired_alerts: Mutex<HashMap<String, DateTime<Utc>>>,
    /// Minimum interval between duplicate alerts for the same pattern (seconds).
    dedup_window_secs: i64,
}

impl BehaviorEngine {
    /// Create a new engine with the built-in pattern library.
    ///
    /// `buffer_capacity` controls how many events the ring buffer can hold.
    pub fn new(buffer_capacity: usize) -> Self {
        let patterns = builtin_patterns();
        info!(
            patterns = patterns.len(),
            buffer_capacity, "Behaviour engine initialised"
        );
        Self {
            events: Mutex::new(RingBuffer::new(buffer_capacity)),
            patterns,
            fired_alerts: Mutex::new(HashMap::new()),
            dedup_window_secs: 60,
        }
    }

    /// Create an engine with custom patterns (useful for testing).
    pub fn with_patterns(buffer_capacity: usize, patterns: Vec<BehaviorPattern>) -> Self {
        Self {
            events: Mutex::new(RingBuffer::new(buffer_capacity)),
            patterns,
            fired_alerts: Mutex::new(HashMap::new()),
            dedup_window_secs: 60,
        }
    }

    /// Ingest a single event and check for pattern matches.
    ///
    /// Returns any new alerts triggered by this event.
    pub fn ingest(&self, event: BehaviorEvent) -> Result<Vec<BehaviorAlert>> {
        {
            let mut buf = self
                .events
                .lock()
                .map_err(|e| GuardianError::Behavioral(format!("Lock poisoned: {e}")))?;
            buf.push_at(event.clone(), event.timestamp);
        }

        debug!(kind = ?event.kind, "Ingested behavioural event");
        self.evaluate()
    }

    /// Ingest multiple events at once.
    pub fn ingest_batch(&self, events: Vec<BehaviorEvent>) -> Result<Vec<BehaviorAlert>> {
        {
            let mut buf = self
                .events
                .lock()
                .map_err(|e| GuardianError::Behavioral(format!("Lock poisoned: {e}")))?;
            for event in &events {
                buf.push_at(event.clone(), event.timestamp);
            }
        }

        self.evaluate()
    }

    /// Evaluate all active events against the pattern library.
    pub fn evaluate(&self) -> Result<Vec<BehaviorAlert>> {
        let buf = self
            .events
            .lock()
            .map_err(|e| GuardianError::Behavioral(format!("Lock poisoned: {e}")))?;

        // Build the event tuple slice that patterns::BehaviorPattern::matches expects.
        let event_tuples: Vec<(BehaviorEventKind, DateTime<Utc>, Option<ProcessInfo>)> = buf
            .iter_active()
            .map(|te| {
                (
                    te.event.kind.clone(),
                    te.timestamp,
                    te.event.process.clone(),
                )
            })
            .collect();

        drop(buf); // Release the events lock before acquiring fired_alerts lock.

        let now = Utc::now();
        let mut fired = self
            .fired_alerts
            .lock()
            .map_err(|e| GuardianError::Behavioral(format!("Lock poisoned: {e}")))?;

        let mut alerts = Vec::new();

        for pattern in &self.patterns {
            if let Some(alert) = pattern.matches(&event_tuples) {
                // Dedup: skip if we already fired this pattern recently.
                let dominated = fired.get(&alert.mitre_id).map_or(false, |last| {
                    (now - *last).num_seconds() < self.dedup_window_secs
                });

                if !dominated {
                    info!(
                        mitre_id = %alert.mitre_id,
                        name = %alert.name,
                        severity = ?alert.severity,
                        "Behaviour pattern matched"
                    );
                    fired.insert(alert.mitre_id.clone(), now);
                    alerts.push(alert);
                }
            }
        }

        Ok(alerts)
    }

    /// Convert alerts to [`Detection`] objects for the scanner pipeline.
    pub fn alerts_to_detections(alerts: &[BehaviorAlert]) -> Vec<Detection> {
        alerts.iter().map(|a| a.to_detection()).collect()
    }

    /// Get the number of events currently in the buffer.
    pub fn event_count(&self) -> usize {
        self.events
            .lock()
            .map(|buf| buf.len())
            .unwrap_or(0)
    }

    /// Get the number of active (non-expired) events.
    pub fn active_event_count(&self) -> usize {
        self.events
            .lock()
            .map(|buf| buf.active_count())
            .unwrap_or(0)
    }

    /// Clear all events and reset alert deduplication state.
    pub fn reset(&self) {
        if let Ok(mut buf) = self.events.lock() {
            buf.clear();
        }
        if let Ok(mut fired) = self.fired_alerts.lock() {
            fired.clear();
        }
    }

    /// Return the list of loaded patterns.
    pub fn patterns(&self) -> &[BehaviorPattern] {
        &self.patterns
    }
}

// Safety: BehaviorEngine is Send + Sync via its Mutex-guarded fields.
unsafe impl Send for BehaviorEngine {}
unsafe impl Sync for BehaviorEngine {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    fn make_event(kind: BehaviorEventKind) -> BehaviorEvent {
        BehaviorEvent {
            kind,
            timestamp: Utc::now(),
            process: None,
            details: HashMap::new(),
        }
    }

    fn make_event_at(
        kind: BehaviorEventKind,
        ts: DateTime<Utc>,
        process_name: Option<&str>,
    ) -> BehaviorEvent {
        BehaviorEvent {
            kind,
            timestamp: ts,
            process: process_name.map(|name| ProcessInfo {
                pid: 1234,
                name: name.to_string(),
                path: format!("/usr/bin/{}", name),
                uid: 1000,
            }),
            details: HashMap::new(),
        }
    }

    #[test]
    fn test_engine_creation() {
        let engine = BehaviorEngine::new(1000);
        assert_eq!(engine.event_count(), 0);
        assert_eq!(engine.patterns().len(), 6);
    }

    #[test]
    fn test_ingest_single_event() {
        let engine = BehaviorEngine::new(100);
        let event = make_event(BehaviorEventKind::FileCreate);
        let alerts = engine.ingest(event).unwrap();
        // Single event should not trigger any multi-event pattern.
        assert!(alerts.is_empty());
        assert_eq!(engine.event_count(), 1);
    }

    #[test]
    fn test_process_injection_detection() {
        let engine = BehaviorEngine::new(100);
        let now = Utc::now();

        let e1 = make_event_at(BehaviorEventKind::ProcessCreate, now, None);
        let e2 = make_event_at(
            BehaviorEventKind::ProcessInject,
            now + Duration::seconds(5),
            None,
        );

        let _ = engine.ingest(e1).unwrap();
        let alerts = engine.ingest(e2).unwrap();

        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].mitre_id, "T1055");
        assert_eq!(alerts[0].severity, Severity::Critical);
    }

    #[test]
    fn test_ransomware_detection() {
        let engine = BehaviorEngine::new(100);
        let now = Utc::now();

        let events = vec![
            make_event_at(BehaviorEventKind::FileModify, now, None),
            make_event_at(
                BehaviorEventKind::FileCreate,
                now + Duration::seconds(10),
                None,
            ),
            make_event_at(
                BehaviorEventKind::FileDelete,
                now + Duration::seconds(20),
                None,
            ),
        ];

        let alerts = engine.ingest_batch(events).unwrap();
        assert!(alerts.iter().any(|a| a.mitre_id == "T1486"));
    }

    #[test]
    fn test_c2_beaconing_detection() {
        let engine = BehaviorEngine::new(100);
        let now = Utc::now();

        let events = vec![
            make_event_at(BehaviorEventKind::NetworkConnect, now, None),
            make_event_at(
                BehaviorEventKind::NetworkConnect,
                now + Duration::seconds(60),
                None,
            ),
            make_event_at(
                BehaviorEventKind::NetworkConnect,
                now + Duration::seconds(120),
                None,
            ),
        ];

        let alerts = engine.ingest_batch(events).unwrap();
        assert!(alerts.iter().any(|a| a.mitre_id == "T1071.001"));
    }

    #[test]
    fn test_alert_deduplication() {
        let engine = BehaviorEngine::new(100);
        let now = Utc::now();

        // Trigger T1055 twice in quick succession.
        let e1 = make_event_at(BehaviorEventKind::ProcessCreate, now, None);
        let e2 = make_event_at(
            BehaviorEventKind::ProcessInject,
            now + Duration::seconds(5),
            None,
        );
        let e3 = make_event_at(
            BehaviorEventKind::ProcessCreate,
            now + Duration::seconds(10),
            None,
        );
        let e4 = make_event_at(
            BehaviorEventKind::ProcessInject,
            now + Duration::seconds(15),
            None,
        );

        let _ = engine.ingest(e1).unwrap();
        let alerts1 = engine.ingest(e2).unwrap();
        let _ = engine.ingest(e3).unwrap();
        let alerts2 = engine.ingest(e4).unwrap();

        // First should trigger, second should be deduplicated.
        assert_eq!(alerts1.len(), 1);
        assert!(
            alerts2.is_empty() || !alerts2.iter().any(|a| a.mitre_id == "T1055"),
            "T1055 should be deduplicated"
        );
    }

    #[test]
    fn test_alert_to_detection() {
        let alert = BehaviorAlert {
            mitre_id: "T1055".to_string(),
            name: "Process Injection".to_string(),
            description: "Detected process injection".to_string(),
            severity: Severity::Critical,
            matched_events: vec!["ProcessCreate".to_string(), "ProcessInject".to_string()],
            timestamp: Utc::now(),
            process_info: Some(ProcessInfo {
                pid: 42,
                name: "malware.exe".to_string(),
                path: "/tmp/malware.exe".to_string(),
                uid: 1000,
            }),
        };

        let det = alert.to_detection();
        assert_eq!(det.engine, DetectionEngine::Behavioral);
        assert_eq!(det.rule_name, "BEHAVIOR/T1055");
        assert_eq!(det.severity, Severity::Critical);
        assert_eq!(det.metadata.get("mitre_id").unwrap(), "T1055");
        assert_eq!(det.metadata.get("process_pid").unwrap(), "42");
    }

    #[test]
    fn test_reset() {
        let engine = BehaviorEngine::new(100);
        let _ = engine
            .ingest(make_event(BehaviorEventKind::FileCreate))
            .unwrap();
        assert_eq!(engine.event_count(), 1);

        engine.reset();
        assert_eq!(engine.event_count(), 0);
    }

    #[test]
    fn test_alerts_to_detections() {
        let alerts = vec![BehaviorAlert {
            mitre_id: "T1486".to_string(),
            name: "Ransomware".to_string(),
            description: "Detected ransomware".to_string(),
            severity: Severity::Critical,
            matched_events: vec![],
            timestamp: Utc::now(),
            process_info: None,
        }];

        let detections = BehaviorEngine::alerts_to_detections(&alerts);
        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].rule_name, "BEHAVIOR/T1486");
    }

    #[test]
    fn test_engine_with_custom_patterns() {
        let patterns = vec![BehaviorPattern {
            mitre_id: "T9999".to_string(),
            name: "Custom Test Pattern".to_string(),
            description: "Test pattern".to_string(),
            severity: Severity::Low,
            sequence: vec![BehaviorEventKind::FileCreate],
            time_window: chrono::Duration::seconds(10),
            process_filters: vec![],
        }];

        let engine = BehaviorEngine::with_patterns(100, patterns);
        assert_eq!(engine.patterns().len(), 1);

        let alerts = engine
            .ingest(make_event(BehaviorEventKind::FileCreate))
            .unwrap();
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].mitre_id, "T9999");
    }

    #[test]
    fn test_behavior_engine_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<BehaviorEngine>();
    }

    #[test]
    fn test_powershell_pattern_with_process() {
        let engine = BehaviorEngine::new(100);
        let now = Utc::now();

        let e1 = make_event_at(
            BehaviorEventKind::ProcessCreate,
            now,
            Some("powershell.exe"),
        );
        let e2 = make_event_at(
            BehaviorEventKind::NetworkConnect,
            now + Duration::seconds(5),
            Some("powershell.exe"),
        );

        let _ = engine.ingest(e1).unwrap();
        let alerts = engine.ingest(e2).unwrap();

        assert!(alerts.iter().any(|a| a.mitre_id == "T1059.001"));
    }

    #[test]
    fn test_ingest_batch() {
        let engine = BehaviorEngine::new(100);

        let events = vec![
            make_event(BehaviorEventKind::FileCreate),
            make_event(BehaviorEventKind::FileModify),
            make_event(BehaviorEventKind::NetworkConnect),
        ];

        let _ = engine.ingest_batch(events).unwrap();
        assert_eq!(engine.event_count(), 3);
    }
}
