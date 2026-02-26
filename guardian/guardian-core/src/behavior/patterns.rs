//! MITRE ATT&CK behaviour pattern definitions and matching logic.
//!
//! Each [`BehaviorPattern`] describes a sequence of [`BehaviorEventKind`] events
//! that, when observed within a time window, constitute a known attack technique.
//! Patterns are identified by their MITRE ATT&CK technique ID and carry a
//! severity rating used to produce [`BehaviorAlert`]s.

use super::{BehaviorAlert, BehaviorEventKind, ProcessInfo};
use chrono::Duration;
use guardian_common::Severity;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// BehaviorPattern
// ---------------------------------------------------------------------------

/// A pattern that matches a sequence of behavioural events within a time window.
#[derive(Debug, Clone)]
pub struct BehaviorPattern {
    /// MITRE ATT&CK technique ID (e.g. "T1055").
    pub mitre_id: String,
    /// Human-readable name of the technique.
    pub name: String,
    /// Detailed description of what this pattern detects.
    pub description: String,
    /// Severity assigned when this pattern fires.
    pub severity: Severity,
    /// Ordered sequence of event kinds that must appear.
    pub sequence: Vec<BehaviorEventKind>,
    /// Maximum time window in which the entire sequence must occur.
    pub time_window: Duration,
    /// Optional: process name patterns that must match (lowercase substring).
    /// If empty, any process is accepted.
    pub process_filters: Vec<String>,
}

impl BehaviorPattern {
    /// Check whether a chronologically ordered slice of `(event_kind, timestamp, process_info)`
    /// tuples matches this pattern.
    ///
    /// Returns `Some(BehaviorAlert)` on the first successful match, `None` otherwise.
    pub fn matches(
        &self,
        events: &[(BehaviorEventKind, chrono::DateTime<chrono::Utc>, Option<ProcessInfo>)],
    ) -> Option<BehaviorAlert> {
        if self.sequence.is_empty() || events.is_empty() {
            return None;
        }

        let seq_len = self.sequence.len();

        // Sliding-window search: find the first occurrence of the full sequence
        // within the time constraint.
        let mut seq_idx = 0;
        let mut window_start = None;
        let mut matched_events = Vec::new();

        for (kind, ts, proc_info) in events {
            // Check process filter if applicable.
            if !self.process_filters.is_empty() {
                let matches_filter = proc_info.as_ref().map_or(false, |p| {
                    let pname = p.name.to_lowercase();
                    self.process_filters.iter().any(|f| pname.contains(f))
                });
                // For events that must come from a specific process, skip non-matching.
                // But we only enforce this on the first event in the sequence to avoid
                // being overly strict — child events may come from different processes.
                if seq_idx == 0 && !matches_filter {
                    continue;
                }
            }

            if *kind == self.sequence[seq_idx] {
                if seq_idx == 0 {
                    window_start = Some(*ts);
                    matched_events.clear();
                }
                matched_events.push(format!("{:?}", kind));
                seq_idx += 1;

                if seq_idx == seq_len {
                    // Check time window.
                    if let Some(start) = window_start {
                        if *ts - start <= self.time_window {
                            return Some(BehaviorAlert {
                                mitre_id: self.mitre_id.clone(),
                                name: self.name.clone(),
                                description: self.description.clone(),
                                severity: self.severity,
                                matched_events,
                                timestamp: *ts,
                                process_info: events
                                    .iter()
                                    .find_map(|(_, _, p)| p.clone()),
                            });
                        }
                    }
                    // Time window exceeded — reset and keep scanning.
                    seq_idx = 0;
                    matched_events.clear();
                    window_start = None;
                }
            }
        }

        None
    }
}

// ---------------------------------------------------------------------------
// Built-in pattern library
// ---------------------------------------------------------------------------

/// Return the full set of built-in MITRE ATT&CK behaviour patterns.
pub fn builtin_patterns() -> Vec<BehaviorPattern> {
    vec![
        pattern_t1055_process_injection(),
        pattern_t1486_ransomware(),
        pattern_t1059_001_powershell(),
        pattern_t1003_001_lsass_dump(),
        pattern_t1547_001_registry_persistence(),
        pattern_t1071_001_c2_beaconing(),
    ]
}

/// T1055 — Process Injection.
///
/// Detects the classic pattern: open a remote process, allocate memory in it,
/// write to that memory, then create a remote thread.
fn pattern_t1055_process_injection() -> BehaviorPattern {
    BehaviorPattern {
        mitre_id: "T1055".to_string(),
        name: "Process Injection".to_string(),
        description: "Detected process injection pattern: process creation followed by \
                       code injection into a remote process."
            .to_string(),
        severity: Severity::Critical,
        sequence: vec![
            BehaviorEventKind::ProcessCreate,
            BehaviorEventKind::ProcessInject,
        ],
        time_window: Duration::seconds(30),
        process_filters: vec![],
    }
}

/// T1486 — Data Encrypted for Impact (Ransomware).
///
/// Detects rapid file modification followed by file creation (ransom notes) and
/// file deletion (shadow copy removal pattern).
fn pattern_t1486_ransomware() -> BehaviorPattern {
    BehaviorPattern {
        mitre_id: "T1486".to_string(),
        name: "Ransomware Behavior".to_string(),
        description: "Detected ransomware-like behaviour: rapid file modification, \
                       creation of new files (possible ransom notes), and file deletion."
            .to_string(),
        severity: Severity::Critical,
        sequence: vec![
            BehaviorEventKind::FileModify,
            BehaviorEventKind::FileCreate,
            BehaviorEventKind::FileDelete,
        ],
        time_window: Duration::seconds(60),
        process_filters: vec![],
    }
}

/// T1059.001 — Command and Scripting Interpreter: PowerShell.
///
/// Detects PowerShell spawning followed by a network connection (download cradle).
fn pattern_t1059_001_powershell() -> BehaviorPattern {
    BehaviorPattern {
        mitre_id: "T1059.001".to_string(),
        name: "PowerShell Execution with Network Activity".to_string(),
        description: "Detected PowerShell process creation followed by a network connection, \
                       indicative of a download cradle or C2 stager."
            .to_string(),
        severity: Severity::High,
        sequence: vec![
            BehaviorEventKind::ProcessCreate,
            BehaviorEventKind::NetworkConnect,
        ],
        time_window: Duration::seconds(30),
        process_filters: vec!["powershell".to_string(), "pwsh".to_string()],
    }
}

/// T1003.001 — OS Credential Dumping: LSASS Memory.
///
/// Detects process creation targeting LSASS followed by credential access events.
fn pattern_t1003_001_lsass_dump() -> BehaviorPattern {
    BehaviorPattern {
        mitre_id: "T1003.001".to_string(),
        name: "LSASS Credential Dump".to_string(),
        description: "Detected process accessing LSASS memory for credential extraction."
            .to_string(),
        severity: Severity::Critical,
        sequence: vec![
            BehaviorEventKind::ProcessCreate,
            BehaviorEventKind::CredentialAccess,
        ],
        time_window: Duration::seconds(15),
        process_filters: vec![
            "mimikatz".to_string(),
            "procdump".to_string(),
            "lsass".to_string(),
        ],
    }
}

/// T1547.001 — Boot or Logon Autostart Execution: Registry Run Keys.
///
/// Detects registry modification (persistence) followed by file creation (dropped payload).
fn pattern_t1547_001_registry_persistence() -> BehaviorPattern {
    BehaviorPattern {
        mitre_id: "T1547.001".to_string(),
        name: "Registry Run Key Persistence".to_string(),
        description: "Detected registry modification to autostart keys followed by \
                       file creation, indicating persistence installation."
            .to_string(),
        severity: Severity::High,
        sequence: vec![
            BehaviorEventKind::RegistryModify,
            BehaviorEventKind::FileCreate,
        ],
        time_window: Duration::seconds(60),
        process_filters: vec![],
    }
}

/// T1071.001 — Application Layer Protocol: Web Protocols (C2 Beaconing).
///
/// Detects repeated network connections (beaconing pattern).
fn pattern_t1071_001_c2_beaconing() -> BehaviorPattern {
    BehaviorPattern {
        mitre_id: "T1071.001".to_string(),
        name: "C2 Beaconing".to_string(),
        description: "Detected repeated outbound network connections consistent with \
                       command-and-control beaconing behaviour."
            .to_string(),
        severity: Severity::High,
        sequence: vec![
            BehaviorEventKind::NetworkConnect,
            BehaviorEventKind::NetworkConnect,
            BehaviorEventKind::NetworkConnect,
        ],
        time_window: Duration::minutes(5),
        process_filters: vec![],
    }
}

/// Build a pattern name-to-MITRE-ID lookup table.
pub fn pattern_index() -> HashMap<String, String> {
    builtin_patterns()
        .into_iter()
        .map(|p| (p.name.clone(), p.mitre_id.clone()))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_events(
        kinds: &[BehaviorEventKind],
        spacing_secs: i64,
    ) -> Vec<(BehaviorEventKind, chrono::DateTime<chrono::Utc>, Option<ProcessInfo>)> {
        let base = Utc::now();
        kinds
            .iter()
            .enumerate()
            .map(|(i, k)| {
                (
                    k.clone(),
                    base + Duration::seconds(i as i64 * spacing_secs),
                    None,
                )
            })
            .collect()
    }

    fn make_events_with_process(
        kinds: &[BehaviorEventKind],
        spacing_secs: i64,
        process_name: &str,
    ) -> Vec<(BehaviorEventKind, chrono::DateTime<chrono::Utc>, Option<ProcessInfo>)> {
        let base = Utc::now();
        kinds
            .iter()
            .enumerate()
            .map(|(i, k)| {
                (
                    k.clone(),
                    base + Duration::seconds(i as i64 * spacing_secs),
                    Some(ProcessInfo {
                        pid: 1234,
                        name: process_name.to_string(),
                        path: format!("/usr/bin/{}", process_name),
                        uid: 1000,
                    }),
                )
            })
            .collect()
    }

    #[test]
    fn test_t1055_match() {
        let pattern = pattern_t1055_process_injection();
        let events = make_events(
            &[
                BehaviorEventKind::ProcessCreate,
                BehaviorEventKind::ProcessInject,
            ],
            5,
        );
        let alert = pattern.matches(&events);
        assert!(alert.is_some());
        let alert = alert.unwrap();
        assert_eq!(alert.mitre_id, "T1055");
        assert_eq!(alert.severity, Severity::Critical);
    }

    #[test]
    fn test_t1055_no_match_wrong_order() {
        let pattern = pattern_t1055_process_injection();
        let events = make_events(
            &[
                BehaviorEventKind::ProcessInject,
                BehaviorEventKind::ProcessCreate,
            ],
            5,
        );
        assert!(pattern.matches(&events).is_none());
    }

    #[test]
    fn test_t1486_ransomware_match() {
        let pattern = pattern_t1486_ransomware();
        let events = make_events(
            &[
                BehaviorEventKind::FileModify,
                BehaviorEventKind::FileCreate,
                BehaviorEventKind::FileDelete,
            ],
            10,
        );
        assert!(pattern.matches(&events).is_some());
    }

    #[test]
    fn test_t1486_timeout() {
        let pattern = pattern_t1486_ransomware();
        // Events spaced 30 seconds apart => total 60+ seconds, exceeding the window.
        let events = make_events(
            &[
                BehaviorEventKind::FileModify,
                BehaviorEventKind::FileCreate,
                BehaviorEventKind::FileDelete,
            ],
            31,
        );
        assert!(pattern.matches(&events).is_none());
    }

    #[test]
    fn test_t1059_001_powershell_match() {
        let pattern = pattern_t1059_001_powershell();
        let events = make_events_with_process(
            &[
                BehaviorEventKind::ProcessCreate,
                BehaviorEventKind::NetworkConnect,
            ],
            5,
            "powershell.exe",
        );
        assert!(pattern.matches(&events).is_some());
    }

    #[test]
    fn test_t1059_001_wrong_process() {
        let pattern = pattern_t1059_001_powershell();
        let events = make_events_with_process(
            &[
                BehaviorEventKind::ProcessCreate,
                BehaviorEventKind::NetworkConnect,
            ],
            5,
            "notepad.exe",
        );
        // Should not match because process filter requires powershell/pwsh.
        assert!(pattern.matches(&events).is_none());
    }

    #[test]
    fn test_t1003_001_lsass_dump_match() {
        let pattern = pattern_t1003_001_lsass_dump();
        let events = make_events_with_process(
            &[
                BehaviorEventKind::ProcessCreate,
                BehaviorEventKind::CredentialAccess,
            ],
            3,
            "mimikatz.exe",
        );
        assert!(pattern.matches(&events).is_some());
    }

    #[test]
    fn test_t1547_001_registry_persistence_match() {
        let pattern = pattern_t1547_001_registry_persistence();
        let events = make_events(
            &[
                BehaviorEventKind::RegistryModify,
                BehaviorEventKind::FileCreate,
            ],
            10,
        );
        assert!(pattern.matches(&events).is_some());
    }

    #[test]
    fn test_t1071_001_c2_beaconing_match() {
        let pattern = pattern_t1071_001_c2_beaconing();
        let events = make_events(
            &[
                BehaviorEventKind::NetworkConnect,
                BehaviorEventKind::NetworkConnect,
                BehaviorEventKind::NetworkConnect,
            ],
            60,
        );
        assert!(pattern.matches(&events).is_some());
    }

    #[test]
    fn test_empty_events() {
        let pattern = pattern_t1055_process_injection();
        assert!(pattern.matches(&[]).is_none());
    }

    #[test]
    fn test_partial_sequence_no_match() {
        let pattern = pattern_t1486_ransomware();
        let events = make_events(
            &[
                BehaviorEventKind::FileModify,
                BehaviorEventKind::FileCreate,
                // Missing FileDelete
            ],
            5,
        );
        assert!(pattern.matches(&events).is_none());
    }

    #[test]
    fn test_builtin_patterns_count() {
        let patterns = builtin_patterns();
        assert_eq!(patterns.len(), 6);
    }

    #[test]
    fn test_pattern_index() {
        let idx = pattern_index();
        assert_eq!(idx.get("Process Injection"), Some(&"T1055".to_string()));
        assert_eq!(idx.get("C2 Beaconing"), Some(&"T1071.001".to_string()));
    }

    #[test]
    fn test_sequence_with_noise() {
        // Pattern T1055 should match even with unrelated events in between,
        // because we do ordered subsequence matching.
        let pattern = pattern_t1055_process_injection();
        let events = make_events(
            &[
                BehaviorEventKind::FileCreate, // noise
                BehaviorEventKind::ProcessCreate,
                BehaviorEventKind::NetworkConnect, // noise
                BehaviorEventKind::ProcessInject,
            ],
            5,
        );
        assert!(pattern.matches(&events).is_some());
    }
}
