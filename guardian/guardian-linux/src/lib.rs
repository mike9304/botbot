//! Linux platform monitoring for Home Guardian antivirus.
//!
//! Provides real-time file-access monitoring via fanotify and syscall tracing
//! via eBPF. Both subsystems compile on all platforms but only have functional
//! implementations on Linux, using `#[cfg(target_os = "linux")]` guards.

pub mod ebpf;
pub mod fanotify;

use guardian_common::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;
use tracing::{error, info};

// ── Configuration ───────────────────────────────────────────────────────────

/// Configuration for the Linux real-time monitor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorConfig {
    /// Filesystem mount points to watch (e.g. `["/", "/home"]`).
    pub monitored_mounts: Vec<PathBuf>,

    /// Paths to exclude from scanning (e.g. `/proc`, `/sys`).
    pub excluded_paths: Vec<PathBuf>,

    /// Process names or PIDs to exclude (e.g. `["guardian", "systemd"]`).
    pub excluded_processes: Vec<String>,

    /// Maximum file size (bytes) to scan on access. Larger files are skipped.
    pub max_scan_size: u64,

    /// Timeout for responding ALLOW/DENY to a permission event.
    pub response_timeout: Duration,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            monitored_mounts: vec![PathBuf::from("/")],
            excluded_paths: vec![
                PathBuf::from("/proc"),
                PathBuf::from("/sys"),
                PathBuf::from("/dev"),
                PathBuf::from("/run"),
                PathBuf::from("/snap"),
            ],
            excluded_processes: vec![
                "guardian".to_string(),
                "systemd".to_string(),
                "journald".to_string(),
            ],
            max_scan_size: 50 * 1024 * 1024, // 50 MB
            response_timeout: Duration::from_secs(10),
        }
    }
}

// ── LinuxMonitor ────────────────────────────────────────────────────────────

/// Combined Linux monitor that manages both fanotify and eBPF subsystems.
pub struct LinuxMonitor {
    config: MonitorConfig,
    fanotify: fanotify::FanotifyMonitor,
    ebpf: ebpf::EbpfTracer,
}

impl LinuxMonitor {
    /// Create a new `LinuxMonitor` with the given configuration.
    pub fn new(config: MonitorConfig) -> Result<Self> {
        info!("Initializing Linux monitor");

        let fanotify = fanotify::FanotifyMonitor::new(&config)?;
        let ebpf = ebpf::EbpfTracer::new()?;

        Ok(Self {
            config,
            fanotify,
            ebpf,
        })
    }

    /// Start all monitoring subsystems.
    ///
    /// This spawns the fanotify event loop and (when available) the eBPF
    /// tracer on background tasks.
    pub async fn start(&mut self) -> Result<()> {
        info!(
            mounts = ?self.config.monitored_mounts,
            excluded = ?self.config.excluded_paths,
            "Starting Linux real-time monitor"
        );

        self.fanotify.start().await?;

        match self.ebpf.start().await {
            Ok(()) => info!("eBPF tracer started"),
            Err(e) => {
                // eBPF is optional — log the error but do not fail.
                error!(error = %e, "eBPF tracer failed to start (non-fatal)");
            }
        }

        Ok(())
    }

    /// Stop all monitoring subsystems.
    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping Linux real-time monitor");
        self.fanotify.stop().await?;
        self.ebpf.stop().await?;
        Ok(())
    }

    /// Returns a reference to the fanotify monitor.
    pub fn fanotify(&self) -> &fanotify::FanotifyMonitor {
        &self.fanotify
    }

    /// Returns a reference to the eBPF tracer.
    pub fn ebpf(&self) -> &ebpf::EbpfTracer {
        &self.ebpf
    }

    /// Returns the current configuration.
    pub fn config(&self) -> &MonitorConfig {
        &self.config
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = MonitorConfig::default();
        assert_eq!(cfg.monitored_mounts, vec![PathBuf::from("/")]);
        assert!(cfg.excluded_paths.contains(&PathBuf::from("/proc")));
        assert!(cfg.excluded_paths.contains(&PathBuf::from("/sys")));
        assert_eq!(cfg.max_scan_size, 50 * 1024 * 1024);
        assert_eq!(cfg.response_timeout, Duration::from_secs(10));
    }

    #[test]
    fn test_config_serialization() {
        let cfg = MonitorConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let restored: MonitorConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.monitored_mounts, cfg.monitored_mounts);
        assert_eq!(restored.max_scan_size, cfg.max_scan_size);
    }

    #[test]
    fn test_config_custom() {
        let cfg = MonitorConfig {
            monitored_mounts: vec![PathBuf::from("/home"), PathBuf::from("/tmp")],
            excluded_paths: vec![],
            excluded_processes: vec!["myapp".to_string()],
            max_scan_size: 10 * 1024 * 1024,
            response_timeout: Duration::from_secs(5),
        };
        assert_eq!(cfg.monitored_mounts.len(), 2);
        assert_eq!(cfg.excluded_processes, vec!["myapp"]);
    }
}
