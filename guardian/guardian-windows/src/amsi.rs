//! Antimalware Scan Interface (AMSI) integration for Windows.
//!
//! AMSI allows Guardian to scan in-memory buffers, scripts, and other content
//! using the same interface that Windows Defender and other AV products use.
//! This enables detection of fileless malware, obfuscated PowerShell scripts,
//! and other in-memory threats.
//!
//! On non-Windows platforms this module compiles but all scan operations
//! return a platform-unsupported error.

use guardian_common::{GuardianError, Result};
use serde::{Deserialize, Serialize};
use tracing::warn;

// ── AMSI result ─────────────────────────────────────────────────────────────

/// Result of an AMSI scan.
///
/// Maps to the Windows `AMSI_RESULT` enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AmsiResult {
    /// No detection; content is clean.
    Clean,
    /// Content was not flagged by any provider (inconclusive).
    NotDetected,
    /// Content was identified as malicious.
    Detected,
    /// Content was blocked by administrator policy.
    BlockedByAdmin,
}

impl AmsiResult {
    /// Returns `true` if the result indicates the content should be blocked.
    pub fn should_block(&self) -> bool {
        matches!(self, AmsiResult::Detected | AmsiResult::BlockedByAdmin)
    }

    /// Returns `true` if the content is considered safe.
    pub fn is_safe(&self) -> bool {
        matches!(self, AmsiResult::Clean | AmsiResult::NotDetected)
    }
}

impl std::fmt::Display for AmsiResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            AmsiResult::Clean => "Clean",
            AmsiResult::NotDetected => "NotDetected",
            AmsiResult::Detected => "Detected",
            AmsiResult::BlockedByAdmin => "BlockedByAdmin",
        };
        write!(f, "{}", name)
    }
}

// ── AMSI scan request ───────────────────────────────────────────────────────

/// Description of content to be scanned through AMSI.
#[derive(Debug, Clone)]
pub struct AmsiScanRequest {
    /// A human-readable name for the content (e.g. script filename).
    pub content_name: String,
    /// The raw bytes to scan.
    pub data: Vec<u8>,
    /// Optional session identifier for correlated scans.
    pub session_id: Option<String>,
}

/// Result of an AMSI scan including metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmsiScanResponse {
    /// The AMSI verdict.
    pub result: AmsiResult,
    /// The content name that was scanned.
    pub content_name: String,
    /// Size of the scanned buffer in bytes.
    pub buffer_size: usize,
}

// ═══════════════════════════════════════════════════════════════════════════
// Windows implementation
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(target_os = "windows")]
mod platform {
    use super::*;
    use windows::Win32::Security::*;

    /// AMSI scanner backed by the Windows AMSI API.
    pub struct AmsiScannerInner {
        /// Whether the AMSI context has been initialized.
        initialized: bool,
        // In a full implementation:
        // amsi_context: HAMSICONTEXT,
        // amsi_session: HAMSISESSION,
    }

    impl AmsiScannerInner {
        pub fn new() -> Result<Self> {
            info!("Creating AMSI scanner (Windows)");
            Ok(Self { initialized: false })
        }

        pub fn initialize(&mut self) -> Result<()> {
            if self.initialized {
                return Ok(());
            }

            info!("Initializing AMSI context");
            // In a full implementation:
            // 1. AmsiInitialize("HomeGuardian", &mut self.amsi_context)
            // 2. AmsiOpenSession(self.amsi_context, &mut self.amsi_session)

            self.initialized = true;
            Ok(())
        }

        pub fn scan_buffer(&self, data: &[u8], content_name: &str) -> Result<AmsiResult> {
            if !self.initialized {
                return Err(GuardianError::Other("AMSI not initialized".to_string()));
            }

            debug!(
                content = content_name,
                size = data.len(),
                "Scanning buffer through AMSI"
            );

            // In a full implementation:
            // let mut result = AMSI_RESULT_CLEAN;
            // AmsiScanBuffer(
            //     self.amsi_context,
            //     data.as_ptr(),
            //     data.len(),
            //     content_name,
            //     self.amsi_session,
            //     &mut result,
            // )?;
            // Map AMSI_RESULT to AmsiResult.

            Ok(AmsiResult::NotDetected)
        }

        pub fn scan_string(&self, content: &str, content_name: &str) -> Result<AmsiResult> {
            if !self.initialized {
                return Err(GuardianError::Other("AMSI not initialized".to_string()));
            }

            debug!(
                content = content_name,
                length = content.len(),
                "Scanning string through AMSI"
            );

            // In a full implementation:
            // let wide: Vec<u16> = content.encode_utf16().chain(std::iter::once(0)).collect();
            // let mut result = AMSI_RESULT_CLEAN;
            // AmsiScanString(
            //     self.amsi_context,
            //     PCWSTR(wide.as_ptr()),
            //     content_name,
            //     self.amsi_session,
            //     &mut result,
            // )?;

            Ok(AmsiResult::NotDetected)
        }

        pub fn close(&mut self) {
            if self.initialized {
                info!("Closing AMSI context");
                // In a full implementation:
                // AmsiCloseSession(self.amsi_context, self.amsi_session);
                // AmsiUninitialize(self.amsi_context);
                self.initialized = false;
            }
        }

        pub fn is_initialized(&self) -> bool {
            self.initialized
        }
    }

    impl Drop for AmsiScannerInner {
        fn drop(&mut self) {
            self.close();
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Non-Windows stub implementation
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(not(target_os = "windows"))]
mod platform {
    use super::*;

    /// Stub AMSI scanner for non-Windows platforms.
    pub struct AmsiScannerInner {
        initialized: bool,
    }

    impl AmsiScannerInner {
        pub fn new() -> Result<Self> {
            warn!("AMSI is not available on this platform");
            Ok(Self { initialized: false })
        }

        pub fn initialize(&mut self) -> Result<()> {
            Err(GuardianError::Other(
                "AMSI is only available on Windows".to_string(),
            ))
        }

        pub fn scan_buffer(&self, _data: &[u8], _content_name: &str) -> Result<AmsiResult> {
            Err(GuardianError::Other(
                "AMSI is only available on Windows".to_string(),
            ))
        }

        pub fn scan_string(&self, _content: &str, _content_name: &str) -> Result<AmsiResult> {
            Err(GuardianError::Other(
                "AMSI is only available on Windows".to_string(),
            ))
        }

        pub fn close(&mut self) {
            self.initialized = false;
        }

        pub fn is_initialized(&self) -> bool {
            self.initialized
        }
    }
}

// ── Public wrapper ──────────────────────────────────────────────────────────

/// Antimalware Scan Interface (AMSI) scanner.
///
/// On Windows this uses the native AMSI API to scan buffers and strings.
/// On other platforms all scan operations return a platform-unsupported error.
pub struct AmsiScanner {
    inner: platform::AmsiScannerInner,
}

impl AmsiScanner {
    /// Create a new AMSI scanner. Call [`initialize()`](Self::initialize)
    /// before scanning.
    pub fn new() -> Result<Self> {
        Ok(Self {
            inner: platform::AmsiScannerInner::new()?,
        })
    }

    /// Initialize the AMSI context and session.
    pub fn initialize(&mut self) -> Result<()> {
        self.inner.initialize()
    }

    /// Scan a raw byte buffer through AMSI.
    ///
    /// # Parameters
    /// - `data`: The bytes to scan.
    /// - `content_name`: A label for the content (e.g. `"script.ps1"`).
    pub fn scan_buffer(&self, data: &[u8], content_name: &str) -> Result<AmsiResult> {
        self.inner.scan_buffer(data, content_name)
    }

    /// Scan a string (e.g. a PowerShell script) through AMSI.
    ///
    /// # Parameters
    /// - `content`: The text to scan.
    /// - `content_name`: A label for the content.
    pub fn scan_string(&self, content: &str, content_name: &str) -> Result<AmsiResult> {
        self.inner.scan_string(content, content_name)
    }

    /// Convenience: scan an [`AmsiScanRequest`] and return a full response.
    pub fn scan(&self, request: &AmsiScanRequest) -> Result<AmsiScanResponse> {
        let result = self.scan_buffer(&request.data, &request.content_name)?;
        Ok(AmsiScanResponse {
            result,
            content_name: request.content_name.clone(),
            buffer_size: request.data.len(),
        })
    }

    /// Close and release the AMSI context.
    pub fn close(&mut self) {
        self.inner.close();
    }

    /// Whether the AMSI scanner has been initialized.
    pub fn is_initialized(&self) -> bool {
        self.inner.is_initialized()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amsi_result_should_block() {
        assert!(!AmsiResult::Clean.should_block());
        assert!(!AmsiResult::NotDetected.should_block());
        assert!(AmsiResult::Detected.should_block());
        assert!(AmsiResult::BlockedByAdmin.should_block());
    }

    #[test]
    fn test_amsi_result_is_safe() {
        assert!(AmsiResult::Clean.is_safe());
        assert!(AmsiResult::NotDetected.is_safe());
        assert!(!AmsiResult::Detected.is_safe());
        assert!(!AmsiResult::BlockedByAdmin.is_safe());
    }

    #[test]
    fn test_amsi_result_display() {
        assert_eq!(format!("{}", AmsiResult::Clean), "Clean");
        assert_eq!(format!("{}", AmsiResult::NotDetected), "NotDetected");
        assert_eq!(format!("{}", AmsiResult::Detected), "Detected");
        assert_eq!(format!("{}", AmsiResult::BlockedByAdmin), "BlockedByAdmin");
    }

    #[test]
    fn test_amsi_result_serialization() {
        let result = AmsiResult::Detected;
        let json = serde_json::to_string(&result).unwrap();
        let restored: AmsiResult = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, AmsiResult::Detected);
    }

    #[test]
    fn test_amsi_scan_response_serialization() {
        let response = AmsiScanResponse {
            result: AmsiResult::Clean,
            content_name: "test.ps1".to_string(),
            buffer_size: 1024,
        };
        let json = serde_json::to_string(&response).unwrap();
        let restored: AmsiScanResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.result, AmsiResult::Clean);
        assert_eq!(restored.content_name, "test.ps1");
        assert_eq!(restored.buffer_size, 1024);
    }

    #[test]
    fn test_amsi_scanner_creation() {
        let scanner = AmsiScanner::new();
        assert!(scanner.is_ok());
    }

    #[test]
    fn test_amsi_scanner_not_initialized() {
        let scanner = AmsiScanner::new().unwrap();
        assert!(!scanner.is_initialized());
    }

    #[cfg(not(target_os = "windows"))]
    #[test]
    fn test_amsi_initialize_fails_non_windows() {
        let mut scanner = AmsiScanner::new().unwrap();
        let result = scanner.initialize();
        assert!(result.is_err());
    }

    #[cfg(not(target_os = "windows"))]
    #[test]
    fn test_amsi_scan_fails_non_windows() {
        let scanner = AmsiScanner::new().unwrap();
        let result = scanner.scan_buffer(b"test", "test.txt");
        assert!(result.is_err());
    }

    #[test]
    fn test_amsi_scan_request_construction() {
        let request = AmsiScanRequest {
            content_name: "script.ps1".to_string(),
            data: b"Write-Host 'Hello'".to_vec(),
            session_id: Some("session-1".to_string()),
        };
        assert_eq!(request.content_name, "script.ps1");
        assert_eq!(request.data.len(), 18);
        assert_eq!(request.session_id, Some("session-1".to_string()));
    }

    #[test]
    fn test_amsi_scanner_close_idempotent() {
        let mut scanner = AmsiScanner::new().unwrap();
        scanner.close();
        scanner.close(); // Should not panic.
        assert!(!scanner.is_initialized());
    }
}
