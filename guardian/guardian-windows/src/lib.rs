//! Windows platform monitoring for Home Guardian antivirus.
//!
//! Provides integration with Windows Event Tracing (ETW) for process and
//! file event monitoring, and the Antimalware Scan Interface (AMSI) for
//! scanning in-memory buffers and scripts.
//!
//! Both subsystems compile on all platforms but only have functional
//! implementations on Windows, using `#[cfg(target_os = "windows")]` guards.

pub mod amsi;
pub mod etw;

use guardian_common::Result;
use tracing::{error, info};

// ── WindowsMonitor ──────────────────────────────────────────────────────────

/// Combined Windows monitor that manages both ETW and AMSI subsystems.
pub struct WindowsMonitor {
    etw: etw::EtwMonitor,
    amsi: amsi::AmsiScanner,
}

impl WindowsMonitor {
    /// Create a new `WindowsMonitor`, initializing ETW and AMSI.
    pub fn new() -> Result<Self> {
        info!("Initializing Windows monitor");

        let etw = etw::EtwMonitor::new()?;
        let amsi = amsi::AmsiScanner::new()?;

        Ok(Self { etw, amsi })
    }

    /// Start all monitoring subsystems.
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting Windows real-time monitor");

        self.etw.start().await?;

        match self.amsi.initialize() {
            Ok(()) => info!("AMSI scanner initialized"),
            Err(e) => {
                error!(error = %e, "AMSI initialization failed (non-fatal)");
            }
        }

        Ok(())
    }

    /// Stop all monitoring subsystems.
    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping Windows real-time monitor");
        self.etw.stop().await?;
        self.amsi.close();
        Ok(())
    }

    /// Returns a reference to the ETW monitor.
    pub fn etw(&self) -> &etw::EtwMonitor {
        &self.etw
    }

    /// Returns a mutable reference to the ETW monitor.
    pub fn etw_mut(&mut self) -> &mut etw::EtwMonitor {
        &mut self.etw
    }

    /// Returns a reference to the AMSI scanner.
    pub fn amsi(&self) -> &amsi::AmsiScanner {
        &self.amsi
    }

    /// Returns a mutable reference to the AMSI scanner.
    pub fn amsi_mut(&mut self) -> &mut amsi::AmsiScanner {
        &mut self.amsi
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_windows_monitor_creation() {
        // On non-Windows this should still succeed (stub implementations).
        let result = WindowsMonitor::new();
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_windows_monitor_start_stop() {
        let mut monitor = WindowsMonitor::new().unwrap();

        // On non-Windows, start returns a platform error from ETW.
        let start_result = monitor.start().await;
        if cfg!(not(target_os = "windows")) {
            assert!(start_result.is_err());
        }

        // Stop should always succeed.
        let stop_result = monitor.stop().await;
        assert!(stop_result.is_ok());
    }
}
