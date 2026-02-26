//! Android UniFFI/JNI bindings for Home Guardian antivirus.
//!
//! This crate provides a C-compatible FFI layer that can be loaded from
//! Android (Kotlin/Java) via JNI. All public functions use `#[no_mangle]`
//! and `extern "C"` for maximum interoperability.
//!
//! Return values are JSON-encoded strings for easy consumption on the
//! Kotlin/Java side. CString conversions handle null-terminated strings
//! for FFI safety.

use guardian_common::{ScanResult, ScanVerdict};
use guardian_core::GuardianScanner;
use serde::{Deserialize, Serialize};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::Path;
use tracing::{debug, error, info, warn};

// ── FFI helpers ─────────────────────────────────────────────────────────────

/// Convert a C string pointer to a Rust `&str`.
///
/// # Safety
/// The caller must ensure `ptr` is a valid, null-terminated C string.
unsafe fn cstr_to_str<'a>(ptr: *const c_char) -> Option<&'a str> {
    if ptr.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(ptr) }.to_str().ok()
}

/// Convert a Rust string to a C string pointer.
///
/// The caller is responsible for freeing the returned pointer via
/// [`guardian_free_string`].
fn str_to_cstring(s: &str) -> *mut c_char {
    match CString::new(s) {
        Ok(cs) => cs.into_raw(),
        Err(_) => {
            // The string contained a null byte; return an error JSON.
            let fallback = CString::new(r#"{"error":"string contains null byte"}"#)
                .expect("fallback is valid");
            fallback.into_raw()
        }
    }
}

/// Return an error JSON as a C string.
fn error_cstring(msg: &str) -> *mut c_char {
    let json = serde_json::json!({ "error": msg }).to_string();
    str_to_cstring(&json)
}

// ── FFI-safe scan result ────────────────────────────────────────────────────

/// Simplified scan result for FFI transport (serialized to JSON).
#[derive(Debug, Serialize, Deserialize)]
struct FfiScanResult {
    file_path: String,
    file_size: u64,
    file_type: String,
    sha256: String,
    verdict: String,
    threat_name: Option<String>,
    severity: Option<String>,
    detection_count: usize,
    scan_duration_ms: u64,
    timestamp: String,
}

impl From<&ScanResult> for FfiScanResult {
    fn from(r: &ScanResult) -> Self {
        let (verdict, threat_name, severity) = match &r.verdict {
            ScanVerdict::Clean => ("clean".to_string(), None, None),
            ScanVerdict::Suspicious(details) => (
                "suspicious".to_string(),
                Some(details.name.clone()),
                Some(format!("{:?}", details.severity)),
            ),
            ScanVerdict::Malicious(details) => (
                "malicious".to_string(),
                Some(details.name.clone()),
                Some(format!("{:?}", details.severity)),
            ),
            ScanVerdict::Error(msg) => ("error".to_string(), Some(msg.clone()), None),
        };

        Self {
            file_path: r.file_path.to_string_lossy().to_string(),
            file_size: r.file_size,
            file_type: format!("{:?}", r.file_type),
            sha256: r.sha256.clone(),
            verdict,
            threat_name,
            severity,
            detection_count: r.detections.len(),
            scan_duration_ms: r.scan_duration_ms,
            timestamp: r.timestamp.to_rfc3339(),
        }
    }
}

// ── Engine version info ─────────────────────────────────────────────────────

/// Version information returned by the engine.
#[derive(Debug, Serialize, Deserialize)]
struct EngineVersion {
    version: String,
    build: String,
    platform: String,
}

// ═══════════════════════════════════════════════════════════════════════════
// FFI entry points
// ═══════════════════════════════════════════════════════════════════════════

/// Scan a single file and return the result as a JSON string.
///
/// # Parameters
/// - `path`: Null-terminated C string with the file path to scan.
///
/// # Returns
/// A null-terminated JSON string containing the scan result. The caller must
/// free the returned string with [`guardian_free_string`].
///
/// # Safety
/// `path` must be a valid, null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn scan_file(path: *const c_char) -> *mut c_char {
    let path_str = match unsafe { cstr_to_str(path) } {
        Some(s) => s,
        None => return error_cstring("null or invalid path pointer"),
    };

    info!(path = path_str, "FFI: scan_file called");

    let file_path = Path::new(path_str);
    if !file_path.exists() {
        return error_cstring(&format!("file not found: {}", path_str));
    }

    let config = guardian_common::ScanConfig::default();
    let scanner = GuardianScanner::new(config);

    match scanner.scan_file(file_path) {
        Ok(result) => {
            let ffi_result = FfiScanResult::from(&result);
            match serde_json::to_string(&ffi_result) {
                Ok(json) => str_to_cstring(&json),
                Err(e) => error_cstring(&format!("JSON serialization failed: {}", e)),
            }
        }
        Err(e) => error_cstring(&format!("scan failed: {}", e)),
    }
}

/// Scan all files in a directory and return results as a JSON array string.
///
/// # Parameters
/// - `path`: Null-terminated C string with the directory path to scan.
///
/// # Returns
/// A null-terminated JSON array string. The caller must free the returned
/// string with [`guardian_free_string`].
///
/// # Safety
/// `path` must be a valid, null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn scan_directory(path: *const c_char) -> *mut c_char {
    let path_str = match unsafe { cstr_to_str(path) } {
        Some(s) => s,
        None => return error_cstring("null or invalid path pointer"),
    };

    info!(path = path_str, "FFI: scan_directory called");

    let dir_path = Path::new(path_str);
    if !dir_path.is_dir() {
        return error_cstring(&format!("not a directory: {}", path_str));
    }

    let config = guardian_common::ScanConfig::default();
    let scanner = GuardianScanner::new(config);

    match guardian_core::scanner::scan_directory(&scanner, dir_path, true) {
        Ok(results) => {
            let ffi_results: Vec<FfiScanResult> =
                results.iter().map(FfiScanResult::from).collect();
            match serde_json::to_string(&ffi_results) {
                Ok(json) => str_to_cstring(&json),
                Err(e) => error_cstring(&format!("JSON serialization failed: {}", e)),
            }
        }
        Err(e) => error_cstring(&format!("directory scan failed: {}", e)),
    }
}

/// Get the engine version as a JSON string.
///
/// # Returns
/// A null-terminated JSON string. The caller must free the returned string
/// with [`guardian_free_string`].
#[no_mangle]
pub extern "C" fn get_engine_version() -> *mut c_char {
    let version = EngineVersion {
        version: env!("CARGO_PKG_VERSION").to_string(),
        build: option_env!("GUARDIAN_BUILD_ID")
            .unwrap_or("dev")
            .to_string(),
        platform: std::env::consts::OS.to_string(),
    };

    match serde_json::to_string(&version) {
        Ok(json) => str_to_cstring(&json),
        Err(e) => error_cstring(&format!("JSON serialization failed: {}", e)),
    }
}

/// Get the number of loaded signatures.
///
/// # Returns
/// The count of loaded signatures (0 if no database is loaded).
#[no_mangle]
pub extern "C" fn get_signature_count() -> u64 {
    // Without a loaded database, return 0. In a full implementation this
    // would query the signature database.
    debug!("FFI: get_signature_count called");
    0
}

/// Update the signature database from a file.
///
/// # Parameters
/// - `db_path`: Null-terminated C string with the path to the new signature
///   database file.
///
/// # Returns
/// `true` if the update succeeded, `false` otherwise.
///
/// # Safety
/// `db_path` must be a valid, null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn update_signatures(db_path: *const c_char) -> bool {
    let path_str = match unsafe { cstr_to_str(db_path) } {
        Some(s) => s,
        None => {
            error!("FFI: update_signatures called with null/invalid path");
            return false;
        }
    };

    info!(path = path_str, "FFI: update_signatures called");

    let path = Path::new(path_str);
    if !path.exists() {
        warn!(path = path_str, "Signature database file not found");
        return false;
    }

    // In a full implementation this would:
    // 1. Validate the database file format and checksum
    // 2. Load signatures into the engine
    // 3. Update the Bloom filter and Aho-Corasick automaton
    // 4. Report the number of new/updated signatures

    info!(path = path_str, "Signature database update completed (stub)");
    true
}

/// Free a string previously returned by one of the FFI functions.
///
/// # Safety
/// `ptr` must be a pointer returned by one of the `scan_*`, `get_*`, or
/// similar functions in this crate, and must not have been freed already.
#[no_mangle]
pub unsafe extern "C" fn guardian_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(unsafe { CString::from_raw(ptr) });
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use guardian_common::{FileType, ScanVerdict, Severity};
    use std::ffi::CString;

    #[test]
    fn test_str_to_cstring_and_back() {
        let original = "hello, world";
        let c_ptr = str_to_cstring(original);
        assert!(!c_ptr.is_null());

        let recovered = unsafe { CStr::from_ptr(c_ptr) }.to_str().unwrap();
        assert_eq!(recovered, original);

        // Clean up.
        unsafe { guardian_free_string(c_ptr) };
    }

    #[test]
    fn test_str_to_cstring_with_null_byte() {
        let bad = "hello\0world";
        let c_ptr = str_to_cstring(bad);
        assert!(!c_ptr.is_null());

        let recovered = unsafe { CStr::from_ptr(c_ptr) }.to_str().unwrap();
        // Should return an error JSON since the string had a null byte.
        assert!(recovered.contains("error"));
        unsafe { guardian_free_string(c_ptr) };
    }

    #[test]
    fn test_error_cstring() {
        let c_ptr = error_cstring("test error");
        assert!(!c_ptr.is_null());

        let recovered = unsafe { CStr::from_ptr(c_ptr) }.to_str().unwrap();
        assert!(recovered.contains("test error"));
        assert!(recovered.contains("error"));
        unsafe { guardian_free_string(c_ptr) };
    }

    #[test]
    fn test_get_engine_version() {
        let c_ptr = get_engine_version();
        assert!(!c_ptr.is_null());

        let json = unsafe { CStr::from_ptr(c_ptr) }.to_str().unwrap();
        let version: EngineVersion = serde_json::from_str(json).unwrap();
        assert!(!version.version.is_empty());
        assert_eq!(version.platform, std::env::consts::OS);
        unsafe { guardian_free_string(c_ptr) };
    }

    #[test]
    fn test_get_signature_count() {
        let count = get_signature_count();
        // Stub always returns 0.
        assert_eq!(count, 0);
    }

    #[test]
    fn test_scan_file_null_path() {
        let c_ptr = unsafe { scan_file(std::ptr::null()) };
        assert!(!c_ptr.is_null());

        let json = unsafe { CStr::from_ptr(c_ptr) }.to_str().unwrap();
        assert!(json.contains("error"));
        assert!(json.contains("null"));
        unsafe { guardian_free_string(c_ptr) };
    }

    #[test]
    fn test_scan_file_nonexistent() {
        let path = CString::new("/nonexistent/file.bin").unwrap();
        let c_ptr = unsafe { scan_file(path.as_ptr()) };
        assert!(!c_ptr.is_null());

        let json = unsafe { CStr::from_ptr(c_ptr) }.to_str().unwrap();
        assert!(json.contains("error"));
        assert!(json.contains("not found"));
        unsafe { guardian_free_string(c_ptr) };
    }

    #[test]
    fn test_scan_directory_null_path() {
        let c_ptr = unsafe { scan_directory(std::ptr::null()) };
        assert!(!c_ptr.is_null());

        let json = unsafe { CStr::from_ptr(c_ptr) }.to_str().unwrap();
        assert!(json.contains("error"));
        unsafe { guardian_free_string(c_ptr) };
    }

    #[test]
    fn test_scan_directory_nonexistent() {
        let path = CString::new("/nonexistent/directory").unwrap();
        let c_ptr = unsafe { scan_directory(path.as_ptr()) };
        assert!(!c_ptr.is_null());

        let json = unsafe { CStr::from_ptr(c_ptr) }.to_str().unwrap();
        assert!(json.contains("error"));
        assert!(json.contains("not a directory"));
        unsafe { guardian_free_string(c_ptr) };
    }

    #[test]
    fn test_update_signatures_null_path() {
        let result = unsafe { update_signatures(std::ptr::null()) };
        assert!(!result);
    }

    #[test]
    fn test_update_signatures_nonexistent() {
        let path = CString::new("/nonexistent/signatures.db").unwrap();
        let result = unsafe { update_signatures(path.as_ptr()) };
        assert!(!result);
    }

    #[test]
    fn test_guardian_free_string_null() {
        // Should not panic.
        unsafe { guardian_free_string(std::ptr::null_mut()) };
    }

    #[test]
    fn test_ffi_scan_result_serialization() {
        let ffi = FfiScanResult {
            file_path: "/test/file.bin".to_string(),
            file_size: 1024,
            file_type: "ELF".to_string(),
            sha256: "abc123".to_string(),
            verdict: "clean".to_string(),
            threat_name: None,
            severity: None,
            detection_count: 0,
            scan_duration_ms: 15,
            timestamp: "2025-01-01T00:00:00Z".to_string(),
        };
        let json = serde_json::to_string(&ffi).unwrap();
        let restored: FfiScanResult = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.file_path, "/test/file.bin");
        assert_eq!(restored.verdict, "clean");
        assert_eq!(restored.detection_count, 0);
    }

    #[test]
    fn test_ffi_scan_result_from_scan_result() {
        let scan_result = ScanResult {
            id: uuid::Uuid::new_v4(),
            file_path: std::path::PathBuf::from("/test/malware.elf"),
            file_size: 2048,
            file_type: FileType::ELF,
            sha256: "deadbeef".to_string(),
            verdict: ScanVerdict::Malicious(guardian_common::ThreatDetails {
                name: "Trojan.Linux.Test".to_string(),
                severity: Severity::Critical,
                family: Some("Trojan".to_string()),
                description: "Test threat".to_string(),
                detections: vec![],
                mitre_ids: vec![],
            }),
            detections: vec![],
            scan_duration_ms: 42,
            timestamp: chrono::Utc::now(),
        };
        let ffi = FfiScanResult::from(&scan_result);
        assert_eq!(ffi.verdict, "malicious");
        assert_eq!(ffi.threat_name, Some("Trojan.Linux.Test".to_string()));
        assert_eq!(ffi.severity, Some("Critical".to_string()));
    }
}
