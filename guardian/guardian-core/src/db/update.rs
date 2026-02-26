//! Signature database update logic.
//!
//! Provides [`SignatureUpdate`] — a description of an incremental update
//! (added and removed hashes) — and the [`apply_update`] function that
//! transactionally applies the update to a [`SignatureDatabase`].

use super::{SignatureDatabase, SignatureRecord};
use guardian_common::{Result, Severity};
use tracing::{info, warn};

// ---------------------------------------------------------------------------
// SignatureUpdate
// ---------------------------------------------------------------------------

/// An incremental update to the signature database.
///
/// Contains lists of records to add and SHA-256 hashes to remove.
/// Use [`apply_update`] to apply an update transactionally.
#[derive(Debug, Clone, Default)]
pub struct SignatureUpdate {
    /// Records to insert (or upsert if already present).
    pub added: Vec<SignatureRecord>,
    /// SHA-256 hashes of signatures to remove.
    pub removed: Vec<String>,
    /// Human-readable version/tag for this update (e.g. `"2025-06-15-001"`).
    pub version: String,
    /// ISO-8601 timestamp when this update was produced.
    pub timestamp: String,
    /// Optional description (e.g. `"Weekly update #42"`).
    pub description: Option<String>,
}

/// Statistics returned after an update is applied.
#[derive(Debug, Clone, Default)]
pub struct UpdateStats {
    /// Number of signatures added/upserted.
    pub added: usize,
    /// Number of signatures removed.
    pub removed: usize,
    /// Total signature count *after* the update.
    pub total_after: u64,
    /// Version string recorded in the database.
    pub version: String,
}

// ---------------------------------------------------------------------------
// apply_update
// ---------------------------------------------------------------------------

/// Apply a [`SignatureUpdate`] to the given database.
///
/// The operation proceeds in two phases:
/// 1. Remove all hashes listed in `update.removed`.
/// 2. Insert all records listed in `update.added`.
///
/// Both phases are performed transactionally (via `insert_batch` /
/// `delete_hashes`).  Metadata (`last_update`, `last_update_version`) is
/// updated on success.
pub fn apply_update(db: &SignatureDatabase, update: &SignatureUpdate) -> Result<UpdateStats> {
    info!(
        version = %update.version,
        to_add = update.added.len(),
        to_remove = update.removed.len(),
        "Applying signature update"
    );

    // Phase 1: removals.
    let removed = if !update.removed.is_empty() {
        db.delete_hashes(&update.removed)?
    } else {
        0
    };

    // Phase 2: additions.
    let added = if !update.added.is_empty() {
        db.insert_batch(&update.added)?
    } else {
        0
    };

    // Update metadata.
    if !update.timestamp.is_empty() {
        db.set_meta("last_update", &update.timestamp)?;
    }
    if !update.version.is_empty() {
        db.set_meta("last_update_version", &update.version)?;
    }

    let total_after = db.count()?;

    let stats = UpdateStats {
        added,
        removed,
        total_after,
        version: update.version.clone(),
    };

    info!(
        added = stats.added,
        removed = stats.removed,
        total = stats.total_after,
        version = %stats.version,
        "Signature update applied"
    );

    Ok(stats)
}

// ---------------------------------------------------------------------------
// Helper: parse a simple line-based update payload
// ---------------------------------------------------------------------------

/// Parse a simple line-based update format into a [`SignatureUpdate`].
///
/// The format is:
///
/// ```text
/// # version: 2025-06-15-001
/// # timestamp: 2025-06-15T12:00:00Z
/// # description: Weekly update
/// + sha256_hex name severity family first_seen
/// - sha256_hex
/// ```
///
/// Lines starting with `#` are metadata headers.  Lines starting with `+`
/// are additions.  Lines starting with `-` are removals.  Blank lines and
/// unknown lines are ignored.
pub fn parse_update_text(text: &str) -> Result<SignatureUpdate> {
    let mut update = SignatureUpdate::default();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if line.starts_with("# version:") {
            update.version = line.trim_start_matches("# version:").trim().to_string();
        } else if line.starts_with("# timestamp:") {
            update.timestamp = line.trim_start_matches("# timestamp:").trim().to_string();
        } else if line.starts_with("# description:") {
            update.description =
                Some(line.trim_start_matches("# description:").trim().to_string());
        } else if line.starts_with('#') {
            // Other comment -- skip.
            continue;
        } else if let Some(rest) = line.strip_prefix("+ ") {
            let parts: Vec<&str> = rest.splitn(5, ' ').collect();
            if parts.len() < 5 {
                warn!(line, "Skipping malformed addition line (need 5 fields)");
                continue;
            }
            update.added.push(SignatureRecord {
                sha256: parts[0].to_string(),
                name: parts[1].to_string(),
                severity: parse_sev(parts[2]),
                family: if parts[3] == "-" {
                    None
                } else {
                    Some(parts[3].to_string())
                },
                first_seen: parts[4].to_string(),
            });
        } else if let Some(rest) = line.strip_prefix("- ") {
            let hash = rest.trim().to_string();
            if !hash.is_empty() {
                update.removed.push(hash);
            }
        }
        // Unknown lines are silently ignored.
    }

    Ok(update)
}

fn parse_sev(s: &str) -> Severity {
    match s.to_lowercase().as_str() {
        "low" => Severity::Low,
        "medium" | "med" => Severity::Medium,
        "high" => Severity::High,
        "critical" | "crit" => Severity::Critical,
        _ => Severity::Medium,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(i: u64) -> SignatureRecord {
        SignatureRecord {
            sha256: format!("{:064x}", i),
            name: format!("Malware.Update.{i}"),
            severity: Severity::High,
            family: Some("UpdateFamily".into()),
            first_seen: "2025-06-15T00:00:00Z".into(),
        }
    }

    #[test]
    fn test_apply_update_add_only() {
        let db = SignatureDatabase::open_in_memory().unwrap();
        let update = SignatureUpdate {
            added: (0..10).map(make_record).collect(),
            removed: vec![],
            version: "v1.0".into(),
            timestamp: "2025-06-15T12:00:00Z".into(),
            description: Some("Initial load".into()),
        };

        let stats = apply_update(&db, &update).unwrap();
        assert_eq!(stats.added, 10);
        assert_eq!(stats.removed, 0);
        assert_eq!(stats.total_after, 10);
        assert_eq!(stats.version, "v1.0");

        // Metadata should be updated.
        assert_eq!(
            db.get_meta("last_update").unwrap().as_deref(),
            Some("2025-06-15T12:00:00Z")
        );
        assert_eq!(
            db.get_meta("last_update_version").unwrap().as_deref(),
            Some("v1.0")
        );
    }

    #[test]
    fn test_apply_update_remove_only() {
        let db = SignatureDatabase::open_in_memory().unwrap();
        // Pre-populate.
        let initial: Vec<_> = (0..5).map(make_record).collect();
        db.insert_batch(&initial).unwrap();
        assert_eq!(db.count().unwrap(), 5);

        let update = SignatureUpdate {
            added: vec![],
            removed: vec![format!("{:064x}", 1), format!("{:064x}", 3)],
            version: "v1.1".into(),
            timestamp: "2025-06-16T00:00:00Z".into(),
            description: None,
        };

        let stats = apply_update(&db, &update).unwrap();
        assert_eq!(stats.added, 0);
        assert_eq!(stats.removed, 2);
        assert_eq!(stats.total_after, 3);
    }

    #[test]
    fn test_apply_update_mixed() {
        let db = SignatureDatabase::open_in_memory().unwrap();
        // Pre-populate with hashes 0..5.
        let initial: Vec<_> = (0..5).map(make_record).collect();
        db.insert_batch(&initial).unwrap();

        // Remove 0 and 4, add 10 and 11.
        let update = SignatureUpdate {
            added: vec![make_record(10), make_record(11)],
            removed: vec![format!("{:064x}", 0), format!("{:064x}", 4)],
            version: "v2.0".into(),
            timestamp: "2025-07-01T00:00:00Z".into(),
            description: Some("Mixed update".into()),
        };

        let stats = apply_update(&db, &update).unwrap();
        assert_eq!(stats.added, 2);
        assert_eq!(stats.removed, 2);
        assert_eq!(stats.total_after, 5); // 5 - 2 + 2

        // Removed hashes gone.
        assert!(db.lookup_hash(&format!("{:064x}", 0)).unwrap().is_none());
        assert!(db.lookup_hash(&format!("{:064x}", 4)).unwrap().is_none());

        // Added hashes present.
        assert!(db.lookup_hash(&format!("{:064x}", 10)).unwrap().is_some());
        assert!(db.lookup_hash(&format!("{:064x}", 11)).unwrap().is_some());

        // Surviving originals still present.
        assert!(db.lookup_hash(&format!("{:064x}", 1)).unwrap().is_some());
    }

    #[test]
    fn test_apply_empty_update() {
        let db = SignatureDatabase::open_in_memory().unwrap();
        let update = SignatureUpdate::default();
        let stats = apply_update(&db, &update).unwrap();
        assert_eq!(stats.added, 0);
        assert_eq!(stats.removed, 0);
        assert_eq!(stats.total_after, 0);
    }

    #[test]
    fn test_parse_update_text() {
        let text = r#"
# version: 2025-06-15-001
# timestamp: 2025-06-15T12:00:00Z
# description: Weekly update #42
+ aaaa000000000000000000000000000000000000000000000000000000000000 Trojan.Test high TrojanFamily 2025-06-15T00:00:00Z
+ bbbb000000000000000000000000000000000000000000000000000000000000 Worm.Generic medium - 2025-06-14T00:00:00Z
- cccc000000000000000000000000000000000000000000000000000000000000
- dddd000000000000000000000000000000000000000000000000000000000000
"#;
        let update = parse_update_text(text).unwrap();
        assert_eq!(update.version, "2025-06-15-001");
        assert_eq!(update.timestamp, "2025-06-15T12:00:00Z");
        assert_eq!(
            update.description.as_deref(),
            Some("Weekly update #42")
        );
        assert_eq!(update.added.len(), 2);
        assert_eq!(update.removed.len(), 2);

        let first = &update.added[0];
        assert_eq!(first.name, "Trojan.Test");
        assert_eq!(first.severity, Severity::High);
        assert_eq!(first.family.as_deref(), Some("TrojanFamily"));

        let second = &update.added[1];
        assert_eq!(second.name, "Worm.Generic");
        assert_eq!(second.severity, Severity::Medium);
        assert!(second.family.is_none());

        assert!(update.removed.contains(
            &"cccc000000000000000000000000000000000000000000000000000000000000".to_string()
        ));
    }

    #[test]
    fn test_parse_update_text_empty() {
        let update = parse_update_text("").unwrap();
        assert!(update.added.is_empty());
        assert!(update.removed.is_empty());
        assert!(update.version.is_empty());
    }

    #[test]
    fn test_parse_and_apply_roundtrip() {
        let text = r#"
# version: v3.0
# timestamp: 2025-07-01T00:00:00Z
+ aaaa000000000000000000000000000000000000000000000000000000000000 Trojan.Test critical EvilFamily 2025-07-01T00:00:00Z
"#;
        let update = parse_update_text(text).unwrap();
        let db = SignatureDatabase::open_in_memory().unwrap();
        let stats = apply_update(&db, &update).unwrap();

        assert_eq!(stats.added, 1);
        assert_eq!(stats.total_after, 1);

        let found = db
            .lookup_hash("aaaa000000000000000000000000000000000000000000000000000000000000")
            .unwrap()
            .unwrap();
        assert_eq!(found.name, "Trojan.Test");
        assert_eq!(found.severity, Severity::Critical);
        assert_eq!(found.family.as_deref(), Some("EvilFamily"));
    }

    #[test]
    fn test_update_stats_version_propagated() {
        let db = SignatureDatabase::open_in_memory().unwrap();
        let update = SignatureUpdate {
            version: "release-42".into(),
            timestamp: "2025-08-01T00:00:00Z".into(),
            ..Default::default()
        };
        let stats = apply_update(&db, &update).unwrap();
        assert_eq!(stats.version, "release-42");

        assert_eq!(
            db.get_meta("last_update_version").unwrap().as_deref(),
            Some("release-42")
        );
    }

    #[test]
    fn test_file_based_update() {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("update_test.db");
        let db = SignatureDatabase::open(&db_path).unwrap();

        // First update: add 5 records.
        let update1 = SignatureUpdate {
            added: (0..5).map(make_record).collect(),
            version: "v1".into(),
            timestamp: "2025-06-01T00:00:00Z".into(),
            ..Default::default()
        };
        let s1 = apply_update(&db, &update1).unwrap();
        assert_eq!(s1.total_after, 5);

        // Second update: remove 2, add 3.
        let update2 = SignatureUpdate {
            added: (100..103).map(make_record).collect(),
            removed: vec![format!("{:064x}", 0), format!("{:064x}", 2)],
            version: "v2".into(),
            timestamp: "2025-07-01T00:00:00Z".into(),
            ..Default::default()
        };
        let s2 = apply_update(&db, &update2).unwrap();
        assert_eq!(s2.total_after, 6); // 5 - 2 + 3
    }
}
