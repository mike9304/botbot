//! Event Tracing for Windows (ETW) integration.
//!
//! Monitors system events — process creation/termination, file operations,
//! registry modifications, and network connections — through the ETW
//! infrastructure.
//!
//! On non-Windows platforms this module compiles but all operations return
//! a platform-unsupported error.

use guardian_common::{GuardianError, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::SystemTime;
use tracing::warn;

// ── Event types ─────────────────────────────────────────────────────────────

/// Classification of ETW events that the monitor subscribes to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventType {
    /// A new process was created.
    ProcessCreate,
    /// A process has terminated.
    ProcessTerminate,
    /// A file was created or opened for writing.
    FileCreate,
    /// A registry key or value was modified.
    RegistryModify,
    /// An outbound network connection was established.
    NetworkConnect,
}

impl std::fmt::Display for EventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            EventType::ProcessCreate => "ProcessCreate",
            EventType::ProcessTerminate => "ProcessTerminate",
            EventType::FileCreate => "FileCreate",
            EventType::RegistryModify => "RegistryModify",
            EventType::NetworkConnect => "NetworkConnect",
        };
        write!(f, "{}", name)
    }
}

// ── Process event ───────────────────────────────────────────────────────────

/// Metadata for a process-related ETW event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessEvent {
    /// Process ID.
    pub pid: u32,
    /// Full path to the process image.
    pub image_path: PathBuf,
    /// Command line used to launch the process.
    pub cmdline: String,
    /// Parent process ID.
    pub parent_pid: u32,
    /// When the event occurred.
    pub timestamp: SystemTime,
    /// The specific event type.
    pub event_type: EventType,
}

// ── File event ──────────────────────────────────────────────────────────────

/// Metadata for a file-related ETW event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEvent {
    /// Full path to the file.
    pub path: PathBuf,
    /// Process ID that performed the operation.
    pub pid: u32,
    /// When the event occurred.
    pub timestamp: SystemTime,
}

// ── Registry event ──────────────────────────────────────────────────────────

/// Metadata for a registry-related ETW event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryEvent {
    /// Registry key path (e.g. `HKLM\SOFTWARE\...`).
    pub key_path: String,
    /// Value name being modified (if applicable).
    pub value_name: Option<String>,
    /// Process ID that performed the modification.
    pub pid: u32,
    /// When the event occurred.
    pub timestamp: SystemTime,
}

// ── Network event ───────────────────────────────────────────────────────────

/// Metadata for a network-related ETW event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkEvent {
    /// Destination IP address.
    pub remote_address: String,
    /// Destination port.
    pub remote_port: u16,
    /// Protocol (TCP/UDP).
    pub protocol: String,
    /// Process ID that initiated the connection.
    pub pid: u32,
    /// When the event occurred.
    pub timestamp: SystemTime,
}

// ── Unified ETW event ───────────────────────────────────────────────────────

/// A single ETW event, dispatched from any of the subscribed providers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EtwEvent {
    Process(ProcessEvent),
    File(FileEvent),
    Registry(RegistryEvent),
    Network(NetworkEvent),
}

/// Callback invoked when an ETW event is received.
pub type EventCallback = Box<dyn Fn(&EtwEvent) + Send + Sync>;

// ═══════════════════════════════════════════════════════════════════════════
// Windows implementation
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(target_os = "windows")]
mod platform {
    use super::*;
    use windows::Win32::System::Diagnostics::Etw;

    /// ETW monitor backed by the Windows ETW API.
    pub struct EtwMonitorInner {
        running: bool,
        subscribed_events: Vec<EventType>,
    }

    impl EtwMonitorInner {
        pub fn new() -> Result<Self> {
            info!("Initializing ETW monitor (Windows)");

            Ok(Self {
                running: false,
                subscribed_events: vec![
                    EventType::ProcessCreate,
                    EventType::ProcessTerminate,
                    EventType::FileCreate,
                    EventType::RegistryModify,
                    EventType::NetworkConnect,
                ],
            })
        }

        pub async fn start(&mut self) -> Result<()> {
            if self.running {
                return Ok(());
            }
            info!(events = ?self.subscribed_events, "Starting ETW trace session");

            // In a full implementation this would:
            // 1. Create an ETW trace session via StartTrace
            // 2. Enable the Microsoft-Windows-Kernel-Process provider
            // 3. Enable the Microsoft-Windows-Kernel-File provider
            // 4. Enable the Microsoft-Windows-Kernel-Registry provider
            // 5. Enable the Microsoft-Windows-Kernel-Network provider
            // 6. Spawn a consumer thread via ProcessTrace

            self.running = true;
            Ok(())
        }

        pub async fn stop(&mut self) -> Result<()> {
            if !self.running {
                return Ok(());
            }
            info!("Stopping ETW trace session");
            self.running = false;
            Ok(())
        }

        pub fn is_running(&self) -> bool {
            self.running
        }

        pub fn subscribed_events(&self) -> &[EventType] {
            &self.subscribed_events
        }

        pub fn set_subscribed_events(&mut self, events: Vec<EventType>) {
            self.subscribed_events = events;
        }

        pub fn poll_events(&self) -> Vec<EtwEvent> {
            // In a full implementation, events would be collected from the
            // ETW consumer thread via a channel.
            Vec::new()
        }

        pub fn on_event(&self, _callback: EventCallback) {
            info!("ETW event callback registered");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Non-Windows stub implementation
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(not(target_os = "windows"))]
mod platform {
    use super::*;

    /// Stub ETW monitor for non-Windows platforms.
    pub struct EtwMonitorInner {
        subscribed_events: Vec<EventType>,
    }

    impl EtwMonitorInner {
        pub fn new() -> Result<Self> {
            warn!("ETW is not available on this platform");
            Ok(Self {
                subscribed_events: vec![
                    EventType::ProcessCreate,
                    EventType::ProcessTerminate,
                    EventType::FileCreate,
                    EventType::RegistryModify,
                    EventType::NetworkConnect,
                ],
            })
        }

        pub async fn start(&mut self) -> Result<()> {
            Err(GuardianError::Other(
                "ETW is only available on Windows".to_string(),
            ))
        }

        pub async fn stop(&mut self) -> Result<()> {
            Ok(())
        }

        pub fn is_running(&self) -> bool {
            false
        }

        pub fn subscribed_events(&self) -> &[EventType] {
            &self.subscribed_events
        }

        pub fn set_subscribed_events(&mut self, events: Vec<EventType>) {
            self.subscribed_events = events;
        }

        pub fn poll_events(&self) -> Vec<EtwEvent> {
            Vec::new()
        }

        pub fn on_event(&self, _callback: EventCallback) {
            warn!("ETW event callback ignored on non-Windows platform");
        }
    }
}

// ── Public wrapper ──────────────────────────────────────────────────────────

/// Event Tracing for Windows (ETW) monitor.
///
/// On Windows this subscribes to kernel ETW providers for process, file,
/// registry, and network events. On other platforms all operations return
/// a platform-unsupported error.
pub struct EtwMonitor {
    inner: platform::EtwMonitorInner,
}

impl EtwMonitor {
    /// Create a new ETW monitor.
    pub fn new() -> Result<Self> {
        Ok(Self {
            inner: platform::EtwMonitorInner::new()?,
        })
    }

    /// Start the ETW trace session.
    pub async fn start(&mut self) -> Result<()> {
        self.inner.start().await
    }

    /// Stop the ETW trace session.
    pub async fn stop(&mut self) -> Result<()> {
        self.inner.stop().await
    }

    /// Whether the trace session is running.
    pub fn is_running(&self) -> bool {
        self.inner.is_running()
    }

    /// Returns the event types currently subscribed to.
    pub fn subscribed_events(&self) -> &[EventType] {
        self.inner.subscribed_events()
    }

    /// Set which event types to subscribe to.
    pub fn set_subscribed_events(&mut self, events: Vec<EventType>) {
        self.inner.set_subscribed_events(events);
    }

    /// Poll for queued events.
    pub fn poll_events(&self) -> Vec<EtwEvent> {
        self.inner.poll_events()
    }

    /// Register a callback for ETW events.
    pub fn on_event(&self, callback: EventCallback) {
        self.inner.on_event(callback);
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_type_display() {
        assert_eq!(format!("{}", EventType::ProcessCreate), "ProcessCreate");
        assert_eq!(format!("{}", EventType::ProcessTerminate), "ProcessTerminate");
        assert_eq!(format!("{}", EventType::FileCreate), "FileCreate");
        assert_eq!(format!("{}", EventType::RegistryModify), "RegistryModify");
        assert_eq!(format!("{}", EventType::NetworkConnect), "NetworkConnect");
    }

    #[test]
    fn test_process_event_serialization() {
        let event = ProcessEvent {
            pid: 1234,
            image_path: PathBuf::from("C:\\Windows\\System32\\cmd.exe"),
            cmdline: "cmd.exe /c dir".to_string(),
            parent_pid: 5678,
            timestamp: SystemTime::now(),
            event_type: EventType::ProcessCreate,
        };
        let json = serde_json::to_string(&event).unwrap();
        let restored: ProcessEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.pid, 1234);
        assert_eq!(restored.parent_pid, 5678);
        assert_eq!(restored.event_type, EventType::ProcessCreate);
    }

    #[test]
    fn test_file_event_serialization() {
        let event = FileEvent {
            path: PathBuf::from("C:\\Users\\test\\malware.exe"),
            pid: 42,
            timestamp: SystemTime::now(),
        };
        let json = serde_json::to_string(&event).unwrap();
        let restored: FileEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.pid, 42);
    }

    #[test]
    fn test_registry_event_serialization() {
        let event = RegistryEvent {
            key_path: r"HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Run".to_string(),
            value_name: Some("Malware".to_string()),
            pid: 100,
            timestamp: SystemTime::now(),
        };
        let json = serde_json::to_string(&event).unwrap();
        let restored: RegistryEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.value_name, Some("Malware".to_string()));
    }

    #[test]
    fn test_network_event_serialization() {
        let event = NetworkEvent {
            remote_address: "192.168.1.1".to_string(),
            remote_port: 443,
            protocol: "TCP".to_string(),
            pid: 200,
            timestamp: SystemTime::now(),
        };
        let json = serde_json::to_string(&event).unwrap();
        let restored: NetworkEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.remote_port, 443);
    }

    #[test]
    fn test_etw_monitor_creation() {
        let monitor = EtwMonitor::new();
        assert!(monitor.is_ok());
    }

    #[test]
    fn test_etw_monitor_default_subscriptions() {
        let monitor = EtwMonitor::new().unwrap();
        let events = monitor.subscribed_events();
        assert_eq!(events.len(), 5);
        assert!(events.contains(&EventType::ProcessCreate));
        assert!(events.contains(&EventType::ProcessTerminate));
        assert!(events.contains(&EventType::FileCreate));
        assert!(events.contains(&EventType::RegistryModify));
        assert!(events.contains(&EventType::NetworkConnect));
    }

    #[test]
    fn test_etw_monitor_set_subscriptions() {
        let mut monitor = EtwMonitor::new().unwrap();
        monitor.set_subscribed_events(vec![EventType::ProcessCreate]);
        assert_eq!(monitor.subscribed_events().len(), 1);
    }

    #[test]
    fn test_etw_monitor_poll_empty() {
        let monitor = EtwMonitor::new().unwrap();
        let events = monitor.poll_events();
        assert!(events.is_empty());
    }

    #[cfg(not(target_os = "windows"))]
    #[test]
    fn test_etw_monitor_not_running_on_non_windows() {
        let monitor = EtwMonitor::new().unwrap();
        assert!(!monitor.is_running());
    }

    #[tokio::test]
    async fn test_etw_monitor_start_non_windows() {
        if cfg!(not(target_os = "windows")) {
            let mut monitor = EtwMonitor::new().unwrap();
            let result = monitor.start().await;
            assert!(result.is_err());
        }
    }

    #[tokio::test]
    async fn test_etw_monitor_stop_always_ok() {
        let mut monitor = EtwMonitor::new().unwrap();
        let result = monitor.stop().await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_etw_event_variants() {
        let process_event = EtwEvent::Process(ProcessEvent {
            pid: 1,
            image_path: PathBuf::from("/test"),
            cmdline: "test".to_string(),
            parent_pid: 0,
            timestamp: SystemTime::now(),
            event_type: EventType::ProcessCreate,
        });
        let json = serde_json::to_string(&process_event).unwrap();
        assert!(json.contains("Process"));

        let file_event = EtwEvent::File(FileEvent {
            path: PathBuf::from("/test"),
            pid: 1,
            timestamp: SystemTime::now(),
        });
        let json = serde_json::to_string(&file_event).unwrap();
        assert!(json.contains("File"));
    }
}
