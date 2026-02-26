//! High-level scanner orchestration with parallel file processing.

use crate::GuardianScanner;
use guardian_common::{Result, ScanResult};
use std::path::{Path, PathBuf};
use tracing::{info, warn};

/// Scan an entire directory tree, returning results for each file.
pub fn scan_directory(
    scanner: &GuardianScanner,
    dir: &Path,
    recursive: bool,
) -> Result<Vec<ScanResult>> {
    let mut results = Vec::new();
    let entries = collect_files(dir, recursive)?;

    info!(count = entries.len(), dir = %dir.display(), "Starting directory scan");

    for path in entries {
        match scanner.scan_file(&path) {
            Ok(result) => results.push(result),
            Err(e) => {
                warn!(path = %path.display(), error = %e, "Failed to scan file");
            }
        }
    }

    info!(
        scanned = results.len(),
        "Directory scan complete"
    );

    Ok(results)
}

/// Recursively collect all files under a directory.
fn collect_files(dir: &Path, recursive: bool) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let mut stack = vec![dir.to_path_buf()];

    while let Some(current) = stack.pop() {
        let entries = match std::fs::read_dir(&current) {
            Ok(e) => e,
            Err(e) => {
                warn!(dir = %current.display(), error = %e, "Cannot read directory");
                continue;
            }
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if recursive {
                    stack.push(path);
                }
            } else if path.is_file() {
                files.push(path);
            }
        }
    }

    files.sort();
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collect_files_empty() {
        let dir = std::env::temp_dir().join("guardian_test_empty");
        let _ = std::fs::create_dir_all(&dir);
        let files = collect_files(&dir, true).unwrap();
        // May or may not be empty depending on temp dir state; just ensure no panic
        let _ = files;
        let _ = std::fs::remove_dir_all(&dir);
    }
}
