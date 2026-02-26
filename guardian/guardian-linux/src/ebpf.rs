//! eBPF-based syscall tracer for Linux.
//!
//! This module provides a placeholder for eBPF-powered system call tracing
//! using the Aya framework. When fully implemented (Phase 3), it will attach
//! eBPF programs to tracepoints and kprobes to monitor security-relevant
//! syscalls such as `execve`, `connect`, `memfd_create`, `ptrace`, and
//! `init_module`.
//!
//! The current implementation is a stub that logs a message indicating the
//! Aya runtime is required.

use guardian_common::{GuardianError, Result};
use serde::{Deserialize, Serialize};
use std::time::SystemTime;
use tracing::{info, warn};

// ── Syscall types ───────────────────────────────────────────────────────────

/// Classification of security-relevant system calls that the eBPF tracer
/// monitors (or will monitor once fully implemented).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SyscallType {
    /// `execve` / `execveat` — process execution.
    Execve,
    /// `connect` — outbound network connection.
    Connect,
    /// `open` / `openat` — file open.
    Open,
    /// `memfd_create` — anonymous in-memory file creation (fileless malware).
    MemfdCreate,
    /// `process_vm_writev` — cross-process memory write (code injection).
    ProcessVmWritev,
    /// `ptrace` — process tracing / debugging (debugger evasion, injection).
    Ptrace,
    /// `init_module` / `finit_module` — kernel module loading (rootkit).
    InitModule,
    /// `chmod` / `fchmod` — permission changes (privilege escalation).
    Chmod,
    /// `unlink` / `unlinkat` — file deletion (evidence destruction).
    Unlink,
    /// `socket` — socket creation.
    Socket,
}

impl std::fmt::Display for SyscallType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            SyscallType::Execve => "execve",
            SyscallType::Connect => "connect",
            SyscallType::Open => "open",
            SyscallType::MemfdCreate => "memfd_create",
            SyscallType::ProcessVmWritev => "process_vm_writev",
            SyscallType::Ptrace => "ptrace",
            SyscallType::InitModule => "init_module",
            SyscallType::Chmod => "chmod",
            SyscallType::Unlink => "unlink",
            SyscallType::Socket => "socket",
        };
        write!(f, "{}", name)
    }
}

// ── Syscall event ───────────────────────────────────────────────────────────

/// A single captured syscall event from the eBPF tracer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyscallEvent {
    /// When the syscall occurred.
    pub timestamp: SystemTime,
    /// Process ID.
    pub pid: u32,
    /// Thread ID.
    pub tid: u32,
    /// User ID.
    pub uid: u32,
    /// Process command name (up to 16 bytes, as reported by the kernel).
    pub comm: String,
    /// The type of syscall.
    pub syscall_type: SyscallType,
    /// Return value of the syscall (0 = success on most syscalls).
    pub return_value: i64,
}

impl SyscallEvent {
    /// Returns `true` if the syscall completed successfully.
    pub fn succeeded(&self) -> bool {
        self.return_value >= 0
    }

    /// Returns `true` if this event represents a potentially dangerous action.
    pub fn is_security_relevant(&self) -> bool {
        matches!(
            self.syscall_type,
            SyscallType::Execve
                | SyscallType::MemfdCreate
                | SyscallType::ProcessVmWritev
                | SyscallType::Ptrace
                | SyscallType::InitModule
        )
    }
}

// ── eBPF tracer callback ────────────────────────────────────────────────────

/// Callback function type invoked when a syscall event is captured.
pub type EventCallback = Box<dyn Fn(&SyscallEvent) + Send + Sync>;

// ── eBPF tracer ─────────────────────────────────────────────────────────────

/// eBPF-based system call tracer.
///
/// **Current status**: Stub implementation. The eBPF programs and Aya runtime
/// integration are planned for Phase 3. All methods log an informational
/// message and return successfully (or with an appropriate error) without
/// actually attaching any eBPF programs.
pub struct EbpfTracer {
    /// Whether the tracer has been "started" (stub: always false in practice).
    running: bool,

    /// Syscall types that should be monitored.
    monitored_syscalls: Vec<SyscallType>,
}

impl EbpfTracer {
    /// Create a new eBPF tracer.
    ///
    /// In the stub implementation this always succeeds.
    pub fn new() -> Result<Self> {
        info!("eBPF tracer created (stub — requires Aya runtime for Phase 3)");
        Ok(Self {
            running: false,
            monitored_syscalls: vec![
                SyscallType::Execve,
                SyscallType::Connect,
                SyscallType::Open,
                SyscallType::MemfdCreate,
                SyscallType::ProcessVmWritev,
                SyscallType::Ptrace,
                SyscallType::InitModule,
                SyscallType::Chmod,
                SyscallType::Unlink,
                SyscallType::Socket,
            ],
        })
    }

    /// Start the eBPF tracer.
    ///
    /// **Stub**: Logs a warning and returns an error indicating Aya is required.
    pub async fn start(&mut self) -> Result<()> {
        warn!("eBPF tracer requires Aya runtime — not starting (Phase 3 placeholder)");
        Err(GuardianError::Other(
            "eBPF tracer requires Aya runtime (planned for Phase 3)".to_string(),
        ))
    }

    /// Stop the eBPF tracer.
    ///
    /// **Stub**: Always succeeds since nothing is running.
    pub async fn stop(&mut self) -> Result<()> {
        self.running = false;
        info!("eBPF tracer stopped (stub)");
        Ok(())
    }

    /// Whether the tracer is currently running.
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// Returns the list of syscall types being monitored.
    pub fn monitored_syscalls(&self) -> &[SyscallType] {
        &self.monitored_syscalls
    }

    /// Set which syscall types to monitor.
    pub fn set_monitored_syscalls(&mut self, syscalls: Vec<SyscallType>) {
        self.monitored_syscalls = syscalls;
    }

    /// Register a callback for syscall events.
    ///
    /// **Stub**: Logs and discards the callback.
    pub fn on_event(&self, _callback: EventCallback) {
        warn!("eBPF tracer requires Aya runtime — event callback ignored");
    }

    /// Poll for events synchronously.
    ///
    /// **Stub**: Always returns an empty vector.
    pub fn poll_events(&self) -> Vec<SyscallEvent> {
        if !self.running {
            return Vec::new();
        }
        warn!("eBPF tracer requires Aya runtime — no events available");
        Vec::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syscall_type_display() {
        assert_eq!(format!("{}", SyscallType::Execve), "execve");
        assert_eq!(format!("{}", SyscallType::Connect), "connect");
        assert_eq!(format!("{}", SyscallType::MemfdCreate), "memfd_create");
        assert_eq!(format!("{}", SyscallType::ProcessVmWritev), "process_vm_writev");
        assert_eq!(format!("{}", SyscallType::Ptrace), "ptrace");
        assert_eq!(format!("{}", SyscallType::InitModule), "init_module");
        assert_eq!(format!("{}", SyscallType::Chmod), "chmod");
        assert_eq!(format!("{}", SyscallType::Unlink), "unlink");
        assert_eq!(format!("{}", SyscallType::Socket), "socket");
        assert_eq!(format!("{}", SyscallType::Open), "open");
    }

    #[test]
    fn test_syscall_event_succeeded() {
        let event = SyscallEvent {
            timestamp: SystemTime::now(),
            pid: 1234,
            tid: 1234,
            uid: 1000,
            comm: "test".to_string(),
            syscall_type: SyscallType::Execve,
            return_value: 0,
        };
        assert!(event.succeeded());

        let failed = SyscallEvent {
            return_value: -1,
            ..event.clone()
        };
        assert!(!failed.succeeded());
    }

    #[test]
    fn test_syscall_event_security_relevant() {
        let make_event = |st: SyscallType| SyscallEvent {
            timestamp: SystemTime::now(),
            pid: 1,
            tid: 1,
            uid: 0,
            comm: "test".to_string(),
            syscall_type: st,
            return_value: 0,
        };

        assert!(make_event(SyscallType::Execve).is_security_relevant());
        assert!(make_event(SyscallType::MemfdCreate).is_security_relevant());
        assert!(make_event(SyscallType::ProcessVmWritev).is_security_relevant());
        assert!(make_event(SyscallType::Ptrace).is_security_relevant());
        assert!(make_event(SyscallType::InitModule).is_security_relevant());

        // These are interesting but not in the "most dangerous" tier.
        assert!(!make_event(SyscallType::Open).is_security_relevant());
        assert!(!make_event(SyscallType::Connect).is_security_relevant());
        assert!(!make_event(SyscallType::Chmod).is_security_relevant());
        assert!(!make_event(SyscallType::Unlink).is_security_relevant());
        assert!(!make_event(SyscallType::Socket).is_security_relevant());
    }

    #[test]
    fn test_ebpf_tracer_creation() {
        let tracer = EbpfTracer::new().unwrap();
        assert!(!tracer.is_running());
        assert_eq!(tracer.monitored_syscalls().len(), 10);
    }

    #[test]
    fn test_ebpf_tracer_poll_when_stopped() {
        let tracer = EbpfTracer::new().unwrap();
        let events = tracer.poll_events();
        assert!(events.is_empty());
    }

    #[test]
    fn test_ebpf_tracer_set_monitored() {
        let mut tracer = EbpfTracer::new().unwrap();
        tracer.set_monitored_syscalls(vec![SyscallType::Execve, SyscallType::Connect]);
        assert_eq!(tracer.monitored_syscalls().len(), 2);
    }

    #[test]
    fn test_syscall_event_serialization() {
        let event = SyscallEvent {
            timestamp: SystemTime::now(),
            pid: 42,
            tid: 42,
            uid: 1000,
            comm: "test_proc".to_string(),
            syscall_type: SyscallType::Execve,
            return_value: 0,
        };
        let json = serde_json::to_string(&event).unwrap();
        let restored: SyscallEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.pid, 42);
        assert_eq!(restored.syscall_type, SyscallType::Execve);
    }

    #[tokio::test]
    async fn test_ebpf_tracer_start_returns_error() {
        let mut tracer = EbpfTracer::new().unwrap();
        // The stub always returns an error from start().
        let result = tracer.start().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_ebpf_tracer_stop_succeeds() {
        let mut tracer = EbpfTracer::new().unwrap();
        let result = tracer.stop().await;
        assert!(result.is_ok());
    }
}
