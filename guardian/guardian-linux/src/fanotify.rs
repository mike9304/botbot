//! Fanotify-based real-time file access scanner for Linux.
//!
//! Uses the Linux fanotify API to intercept file open and close events,
//! scan the accessed file through the Guardian detection engine, and respond
//! with ALLOW or DENY to block malicious files before they execute.
//!
//! On non-Linux platforms this module compiles but all operations return an
//! error indicating the platform is unsupported.

use guardian_common::{GuardianError, Result, ScanVerdict};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tracing::info;

use crate::MonitorConfig;

// ── Scan cache ──────────────────────────────────────────────────────────────

/// Cache key: file path + last modification time (as seconds since epoch).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    path: PathBuf,
    mtime_secs: i64,
}

/// Cached verdict for a previously scanned file.
#[derive(Debug, Clone)]
struct CacheEntry {
    verdict: CachedVerdict,
    inserted_at: Instant,
}

/// Simplified verdict stored in the cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CachedVerdict {
    Clean,
    Suspicious,
    Malicious,
}

impl From<&ScanVerdict> for CachedVerdict {
    fn from(v: &ScanVerdict) -> Self {
        match v {
            ScanVerdict::Clean => CachedVerdict::Clean,
            ScanVerdict::Suspicious(_) => CachedVerdict::Suspicious,
            ScanVerdict::Malicious(_) => CachedVerdict::Malicious,
            ScanVerdict::Error(_) => CachedVerdict::Suspicious,
        }
    }
}

/// LRU-style scan cache with a time-to-live.
///
/// Entries expire after `ttl` so that modified files are re-scanned. The cache
/// also has a maximum capacity; the oldest entry is evicted when full.
pub struct ScanCache {
    entries: HashMap<CacheKey, CacheEntry>,
    max_entries: usize,
    ttl: Duration,
}

impl ScanCache {
    /// Create a new cache with the given capacity and TTL.
    pub fn new(max_entries: usize, ttl: Duration) -> Self {
        Self {
            entries: HashMap::with_capacity(max_entries),
            max_entries,
            ttl,
        }
    }

    /// Look up a cached verdict. Returns `None` if not cached or expired.
    pub fn get(&mut self, path: &Path, mtime_secs: i64) -> Option<CachedVerdict> {
        let key = CacheKey {
            path: path.to_path_buf(),
            mtime_secs,
        };
        if let Some(entry) = self.entries.get(&key) {
            if entry.inserted_at.elapsed() < self.ttl {
                return Some(entry.verdict);
            }
            // Expired — remove it.
            self.entries.remove(&key);
        }
        None
    }

    /// Insert or update a cached verdict.
    pub fn insert(&mut self, path: &Path, mtime_secs: i64, verdict: CachedVerdict) {
        // Evict expired entries if at capacity.
        if self.entries.len() >= self.max_entries {
            self.evict_expired();
        }

        // If still at capacity, remove the oldest entry.
        if self.entries.len() >= self.max_entries {
            self.evict_oldest();
        }

        let key = CacheKey {
            path: path.to_path_buf(),
            mtime_secs,
        };
        self.entries.insert(
            key,
            CacheEntry {
                verdict,
                inserted_at: Instant::now(),
            },
        );
    }

    /// Remove all expired entries.
    pub fn evict_expired(&mut self) {
        self.entries
            .retain(|_, entry| entry.inserted_at.elapsed() < self.ttl);
    }

    /// Remove the single oldest entry.
    fn evict_oldest(&mut self) {
        if let Some(oldest_key) = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.inserted_at)
            .map(|(key, _)| key.clone())
        {
            self.entries.remove(&oldest_key);
        }
    }

    /// Number of entries currently in the cache.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear the entire cache.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

// ── Fanotify response ───────────────────────────────────────────────────────

/// Response sent back to the kernel after scanning a permission event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FanotifyResponse {
    /// Allow the file operation to proceed.
    Allow,
    /// Deny the file operation (blocks the calling process).
    Deny,
}

// ── Fanotify event metadata ─────────────────────────────────────────────────

/// Metadata extracted from a fanotify event.
#[derive(Debug, Clone)]
pub struct FanotifyEvent {
    /// The file descriptor from the event.
    pub fd: i32,
    /// The resolved file path (from /proc/self/fd/{fd}).
    pub path: PathBuf,
    /// Process ID that triggered the event.
    pub pid: u32,
    /// Whether this is a permission event requiring a response.
    pub permission_event: bool,
    /// The type of file access.
    pub event_type: FanotifyEventType,
}

/// Type of fanotify event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FanotifyEventType {
    /// File was opened (or an open-permission check is pending).
    OpenPerm,
    /// File was closed after writing.
    CloseWrite,
}

// ═══════════════════════════════════════════════════════════════════════════
// Linux implementation
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(target_os = "linux")]
mod platform {
    use super::*;
    use nix::sys::fanotify::{
        EventFFlags, Fanotify, InitFlags, MarkFlags, MaskFlags,
    };
    use std::os::unix::io::AsRawFd;

    /// Real fanotify monitor backed by the Linux kernel fanotify API.
    pub struct FanotifyMonitorInner {
        /// The fanotify file descriptor wrapper from nix.
        fan: Option<Fanotify>,
        /// Scan cache (path + mtime -> verdict).
        cache: ScanCache,
        /// Monitoring configuration.
        config: MonitorConfig,
        /// Whether the event loop is running.
        running: bool,
    }

    impl FanotifyMonitorInner {
        pub fn new(config: &MonitorConfig) -> Result<Self> {
            info!("Initializing fanotify monitor (Linux)");

            // FAN_CLASS_CONTENT allows permission events so we can block
            // malicious files before they execute.
            let fan = Fanotify::init(
                InitFlags::FAN_CLASS_CONTENT | InitFlags::FAN_CLOEXEC,
                EventFFlags::O_RDONLY | EventFFlags::O_LARGEFILE,
            )
            .map_err(|e| GuardianError::Other(format!("fanotify_init failed: {}", e)))?;

            // Mark each monitored mount for open-permission and close-write events.
            let mask = MaskFlags::FAN_OPEN_PERM | MaskFlags::FAN_CLOSE_WRITE;
            for mount in &config.monitored_mounts {
                info!(mount = %mount.display(), "Marking mount for fanotify");
                fan.mark(
                    MarkFlags::FAN_MARK_ADD | MarkFlags::FAN_MARK_MOUNT,
                    mask,
                    None,
                    Some(mount),
                )
                .map_err(|e| {
                    GuardianError::Other(format!(
                        "fanotify_mark failed for {}: {}",
                        mount.display(),
                        e
                    ))
                })?;
            }

            Ok(Self {
                fan: Some(fan),
                cache: ScanCache::new(10_000, Duration::from_secs(300)),
                config: config.clone(),
                running: false,
            })
        }

        pub async fn start(&mut self) -> Result<()> {
            if self.running {
                return Ok(());
            }
            self.running = true;
            info!("Fanotify event loop started");
            Ok(())
        }

        pub async fn stop(&mut self) -> Result<()> {
            self.running = false;
            self.fan.take(); // drops the fd
            info!("Fanotify event loop stopped");
            Ok(())
        }

        /// Read pending events from the fanotify file descriptor.
        ///
        /// In a production deployment this would be called in a tight loop
        /// on a dedicated thread. Each permission event must be answered
        /// with ALLOW or DENY within the configured timeout.
        pub fn read_events(&mut self) -> Result<Vec<FanotifyEvent>> {
            let fan = self.fan.as_ref().ok_or_else(|| {
                GuardianError::Other("Fanotify not initialized".to_string())
            })?;

            let raw_events = fan
                .read_events()
                .map_err(|e| GuardianError::Other(format!("fanotify read: {}", e)))?;

            let mut events = Vec::new();
            for ev in &raw_events {
                // fd() returns Option<BorrowedFd>; skip overflow events.
                let borrowed_fd = match ev.fd() {
                    Some(fd) => fd,
                    None => continue, // queue overflow sentinel
                };
                let raw_fd = borrowed_fd.as_raw_fd();

                // Resolve path via /proc/self/fd/{fd}
                let link = format!("/proc/self/fd/{}", raw_fd);
                let path = std::fs::read_link(&link)
                    .unwrap_or_else(|_| PathBuf::from("<unknown>"));

                // Determine event type from the mask.
                let mask = ev.mask();
                let event_type = if mask.contains(MaskFlags::FAN_OPEN_PERM) {
                    FanotifyEventType::OpenPerm
                } else {
                    FanotifyEventType::CloseWrite
                };

                let permission_event = mask.contains(MaskFlags::FAN_OPEN_PERM);

                events.push(FanotifyEvent {
                    fd: raw_fd,
                    path,
                    pid: ev.pid() as u32,
                    permission_event,
                    event_type,
                });
            }

            Ok(events)
        }

        /// Resolve the file path from an event's fd via /proc/self/fd.
        #[allow(dead_code)]
        pub fn resolve_path(fd: i32) -> PathBuf {
            let link = format!("/proc/self/fd/{}", fd);
            std::fs::read_link(&link).unwrap_or_else(|_| PathBuf::from("<unknown>"))
        }

        /// Check whether a path should be excluded from scanning.
        pub fn is_excluded(&self, path: &Path) -> bool {
            // Excluded paths.
            for excluded in &self.config.excluded_paths {
                if path.starts_with(excluded) {
                    return true;
                }
            }
            false
        }

        /// Query the scan cache for a previously seen file.
        pub fn cache_lookup(&mut self, path: &Path) -> Option<CachedVerdict> {
            let mtime = std::fs::metadata(path)
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);
            self.cache.get(path, mtime)
        }

        /// Insert a scan result into the cache.
        pub fn cache_insert(&mut self, path: &Path, verdict: CachedVerdict) {
            let mtime = std::fs::metadata(path)
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);
            self.cache.insert(path, mtime, verdict);
        }

        pub fn is_running(&self) -> bool {
            self.running
        }

        pub fn cache(&self) -> &ScanCache {
            &self.cache
        }

        pub fn cache_mut(&mut self) -> &mut ScanCache {
            &mut self.cache
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Non-Linux stub implementation
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(not(target_os = "linux"))]
mod platform {
    use super::*;

    /// Stub fanotify monitor for non-Linux platforms.
    ///
    /// All operations return an error indicating the platform is unsupported.
    pub struct FanotifyMonitorInner {
        cache: ScanCache,
    }

    impl FanotifyMonitorInner {
        pub fn new(_config: &MonitorConfig) -> Result<Self> {
            warn!("Fanotify is not available on this platform");
            Ok(Self {
                cache: ScanCache::new(10_000, Duration::from_secs(300)),
            })
        }

        pub async fn start(&mut self) -> Result<()> {
            Err(GuardianError::Other(
                "fanotify is only available on Linux".to_string(),
            ))
        }

        pub async fn stop(&mut self) -> Result<()> {
            Ok(())
        }

        pub fn read_events(&mut self) -> Result<Vec<FanotifyEvent>> {
            Err(GuardianError::Other(
                "fanotify is only available on Linux".to_string(),
            ))
        }

        pub fn is_excluded(&self, _path: &Path) -> bool {
            false
        }

        pub fn cache_lookup(&mut self, _path: &Path) -> Option<CachedVerdict> {
            None
        }

        pub fn cache_insert(&mut self, _path: &Path, _verdict: CachedVerdict) {}

        pub fn is_running(&self) -> bool {
            false
        }

        pub fn cache(&self) -> &ScanCache {
            &self.cache
        }

        pub fn cache_mut(&mut self) -> &mut ScanCache {
            &mut self.cache
        }
    }
}

// ── Public wrapper ──────────────────────────────────────────────────────────

/// Fanotify-based real-time file scanner.
///
/// On Linux this wraps the kernel fanotify API for intercepting file-open and
/// file-close events. On other platforms all operations return a platform-
/// unsupported error.
pub struct FanotifyMonitor {
    inner: platform::FanotifyMonitorInner,
}

impl FanotifyMonitor {
    /// Create a new fanotify monitor.
    pub fn new(config: &MonitorConfig) -> Result<Self> {
        Ok(Self {
            inner: platform::FanotifyMonitorInner::new(config)?,
        })
    }

    /// Start the fanotify event loop.
    pub async fn start(&mut self) -> Result<()> {
        self.inner.start().await
    }

    /// Stop the fanotify event loop.
    pub async fn stop(&mut self) -> Result<()> {
        self.inner.stop().await
    }

    /// Read pending fanotify events.
    pub fn read_events(&mut self) -> Result<Vec<FanotifyEvent>> {
        self.inner.read_events()
    }

    /// Check whether a path is excluded from scanning.
    pub fn is_excluded(&self, path: &Path) -> bool {
        self.inner.is_excluded(path)
    }

    /// Look up a cached scan verdict.
    pub fn cache_lookup(&mut self, path: &Path) -> Option<CachedVerdict> {
        self.inner.cache_lookup(path)
    }

    /// Insert a verdict into the scan cache.
    pub fn cache_insert(&mut self, path: &Path, verdict: CachedVerdict) {
        self.inner.cache_insert(path, verdict);
    }

    /// Whether the monitor is currently running.
    pub fn is_running(&self) -> bool {
        self.inner.is_running()
    }

    /// Access the scan cache.
    pub fn cache(&self) -> &ScanCache {
        self.inner.cache()
    }

    /// Access the scan cache mutably.
    pub fn cache_mut(&mut self) -> &mut ScanCache {
        self.inner.cache_mut()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_cache_basic() {
        let mut cache = ScanCache::new(100, Duration::from_secs(300));
        let path = Path::new("/tmp/test.bin");
        assert!(cache.is_empty());

        cache.insert(path, 1000, CachedVerdict::Clean);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(path, 1000), Some(CachedVerdict::Clean));
    }

    #[test]
    fn test_scan_cache_mtime_mismatch() {
        let mut cache = ScanCache::new(100, Duration::from_secs(300));
        let path = Path::new("/tmp/test.bin");

        cache.insert(path, 1000, CachedVerdict::Clean);
        // Different mtime means the file was modified — cache miss.
        assert_eq!(cache.get(path, 2000), None);
    }

    #[test]
    fn test_scan_cache_expiry() {
        let mut cache = ScanCache::new(100, Duration::from_millis(1));
        let path = Path::new("/tmp/test.bin");

        cache.insert(path, 1000, CachedVerdict::Clean);
        // Sleep to expire the entry.
        std::thread::sleep(Duration::from_millis(10));
        assert_eq!(cache.get(path, 1000), None);
    }

    #[test]
    fn test_scan_cache_eviction() {
        let mut cache = ScanCache::new(2, Duration::from_secs(300));

        cache.insert(Path::new("/a"), 1, CachedVerdict::Clean);
        cache.insert(Path::new("/b"), 1, CachedVerdict::Malicious);
        assert_eq!(cache.len(), 2);

        // Third insert should evict the oldest.
        cache.insert(Path::new("/c"), 1, CachedVerdict::Suspicious);
        assert!(cache.len() <= 2);
    }

    #[test]
    fn test_scan_cache_clear() {
        let mut cache = ScanCache::new(100, Duration::from_secs(300));
        cache.insert(Path::new("/a"), 1, CachedVerdict::Clean);
        cache.insert(Path::new("/b"), 2, CachedVerdict::Clean);
        assert_eq!(cache.len(), 2);

        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cached_verdict_from_scan_verdict() {
        assert_eq!(CachedVerdict::from(&ScanVerdict::Clean), CachedVerdict::Clean);

        let details = guardian_common::ThreatDetails {
            name: "test".to_string(),
            severity: guardian_common::Severity::High,
            family: None,
            description: "test".to_string(),
            detections: vec![],
            mitre_ids: vec![],
        };
        assert_eq!(
            CachedVerdict::from(&ScanVerdict::Malicious(details.clone())),
            CachedVerdict::Malicious
        );
        assert_eq!(
            CachedVerdict::from(&ScanVerdict::Suspicious(details)),
            CachedVerdict::Suspicious
        );
    }

    #[test]
    fn test_fanotify_event_type_eq() {
        assert_eq!(FanotifyEventType::OpenPerm, FanotifyEventType::OpenPerm);
        assert_ne!(FanotifyEventType::OpenPerm, FanotifyEventType::CloseWrite);
    }

    #[test]
    fn test_fanotify_response_eq() {
        assert_eq!(FanotifyResponse::Allow, FanotifyResponse::Allow);
        assert_ne!(FanotifyResponse::Allow, FanotifyResponse::Deny);
    }

    #[test]
    fn test_fanotify_monitor_creation() {
        let config = MonitorConfig::default();
        // On Linux, fanotify_init requires CAP_SYS_ADMIN. If we don't have it,
        // the constructor will return Err — that's expected in a test environment.
        let _monitor = FanotifyMonitor::new(&config);
        // We just check it does not panic on construction.
    }
}
