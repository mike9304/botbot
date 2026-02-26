//! Signature database management for Home Guardian.
//!
//! Provides [`SignatureDatabase`] — a SQLite-backed store of known-malware
//! SHA-256 hashes with associated metadata (name, severity, family).
//! The database uses WAL mode for concurrent read access and is designed
//! for fast hash lookups during scanning.

pub mod update;

use guardian_common::{GuardianError, Result, Severity};
use rusqlite::{params, Connection, OpenFlags};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tracing::{debug, info};

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A single signature record stored in the database.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignatureRecord {
    /// SHA-256 hash of the known-malware file.
    pub sha256: String,
    /// Human-readable name (e.g. `Trojan.GenericKD.12345`).
    pub name: String,
    /// Severity level.
    pub severity: Severity,
    /// Malware family (e.g. `Emotet`, `AgentTesla`).
    pub family: Option<String>,
    /// ISO-8601 date when this signature was first seen.
    pub first_seen: String,
}

/// Summary statistics for the database.
#[derive(Debug, Clone, Default)]
pub struct DbStats {
    pub total_signatures: u64,
    pub families: u64,
    pub db_size_bytes: u64,
}

// ---------------------------------------------------------------------------
// SignatureDatabase
// ---------------------------------------------------------------------------

/// Thread-safe SQLite signature database.
///
/// The underlying connection is wrapped in `Arc<Mutex<..>>` so that
/// `SignatureDatabase` can be shared across threads.  Reads are fast
/// because SQLite WAL mode allows concurrent readers.
pub struct SignatureDatabase {
    conn: Arc<Mutex<Connection>>,
    path: PathBuf,
}

impl SignatureDatabase {
    /// Open (or create) the signature database at `path`.
    ///
    /// Automatically creates the schema and enables WAL mode.
    pub fn open(path: &Path) -> Result<Self> {
        // Ensure the parent directory exists.
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                GuardianError::Database(format!(
                    "cannot create parent directory {}: {e}",
                    parent.display()
                ))
            })?;
        }

        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_CREATE
                | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .map_err(|e| GuardianError::Database(format!("cannot open {}: {e}", path.display())))?;

        let db = Self {
            conn: Arc::new(Mutex::new(conn)),
            path: path.to_path_buf(),
        };

        db.init()?;
        info!(path = %path.display(), "Signature database opened");
        Ok(db)
    }

    /// Open an in-memory database (useful for tests).
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()
            .map_err(|e| GuardianError::Database(format!("cannot open in-memory db: {e}")))?;

        let db = Self {
            conn: Arc::new(Mutex::new(conn)),
            path: PathBuf::from(":memory:"),
        };

        db.init()?;
        debug!("In-memory signature database opened");
        Ok(db)
    }

    /// Initialize the schema and pragmas.
    fn init(&self) -> Result<()> {
        let conn = self.conn.lock().map_err(|e| {
            GuardianError::Database(format!("database lock poisoned: {e}"))
        })?;

        // Enable WAL mode for concurrent reads.
        conn.execute_batch("PRAGMA journal_mode = WAL;")
            .map_err(|e| GuardianError::Database(format!("failed to set WAL mode: {e}")))?;

        // Performance pragmas.
        conn.execute_batch(
            "PRAGMA synchronous = NORMAL;
             PRAGMA cache_size = -8000;
             PRAGMA temp_store = MEMORY;",
        )
        .map_err(|e| GuardianError::Database(format!("failed to set pragmas: {e}")))?;

        // Create tables.
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS signatures (
                sha256     TEXT PRIMARY KEY NOT NULL,
                name       TEXT NOT NULL,
                severity   TEXT NOT NULL,
                family     TEXT,
                first_seen TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_signatures_family
                ON signatures(family);

            CREATE INDEX IF NOT EXISTS idx_signatures_severity
                ON signatures(severity);

            CREATE TABLE IF NOT EXISTS db_meta (
                key   TEXT PRIMARY KEY NOT NULL,
                value TEXT NOT NULL
            );",
        )
        .map_err(|e| GuardianError::Database(format!("schema creation failed: {e}")))?;

        // Seed metadata if missing.
        conn.execute(
            "INSERT OR IGNORE INTO db_meta(key, value) VALUES ('version', '1')",
            [],
        )
        .map_err(|e| GuardianError::Database(format!("metadata insert failed: {e}")))?;

        conn.execute(
            "INSERT OR IGNORE INTO db_meta(key, value) VALUES ('last_update', '')",
            [],
        )
        .map_err(|e| GuardianError::Database(format!("metadata insert failed: {e}")))?;

        Ok(())
    }

    // -----------------------------------------------------------------
    // Lookups
    // -----------------------------------------------------------------

    /// Look up a SHA-256 hash and return the matching record, if any.
    pub fn lookup_hash(&self, sha256: &str) -> Result<Option<SignatureRecord>> {
        let conn = self.conn.lock().map_err(|e| {
            GuardianError::Database(format!("database lock poisoned: {e}"))
        })?;

        let mut stmt = conn
            .prepare_cached(
                "SELECT sha256, name, severity, family, first_seen
                 FROM signatures WHERE sha256 = ?1",
            )
            .map_err(|e| GuardianError::Database(format!("prepare failed: {e}")))?;

        let result = stmt
            .query_row(params![sha256], |row| {
                Ok(SignatureRecord {
                    sha256: row.get(0)?,
                    name: row.get(1)?,
                    severity: parse_severity(&row.get::<_, String>(2)?),
                    family: row.get(3)?,
                    first_seen: row.get(4)?,
                })
            })
            .optional()
            .map_err(|e| GuardianError::Database(format!("query failed: {e}")))?;

        Ok(result)
    }

    // -----------------------------------------------------------------
    // Insertions
    // -----------------------------------------------------------------

    /// Insert a single signature record (upsert).
    pub fn insert(&self, record: &SignatureRecord) -> Result<()> {
        let conn = self.conn.lock().map_err(|e| {
            GuardianError::Database(format!("database lock poisoned: {e}"))
        })?;

        conn.execute(
            "INSERT OR REPLACE INTO signatures(sha256, name, severity, family, first_seen)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                record.sha256,
                record.name,
                severity_to_str(record.severity),
                record.family,
                record.first_seen,
            ],
        )
        .map_err(|e| GuardianError::Database(format!("insert failed: {e}")))?;

        Ok(())
    }

    /// Insert multiple records in a single transaction.
    pub fn insert_batch(&self, records: &[SignatureRecord]) -> Result<usize> {
        if records.is_empty() {
            return Ok(0);
        }

        let conn = self.conn.lock().map_err(|e| {
            GuardianError::Database(format!("database lock poisoned: {e}"))
        })?;

        let tx = conn
            .unchecked_transaction()
            .map_err(|e| GuardianError::Database(format!("begin transaction failed: {e}")))?;

        let mut inserted = 0usize;
        {
            let mut stmt = tx
                .prepare_cached(
                    "INSERT OR REPLACE INTO signatures(sha256, name, severity, family, first_seen)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                )
                .map_err(|e| GuardianError::Database(format!("prepare failed: {e}")))?;

            for rec in records {
                stmt.execute(params![
                    rec.sha256,
                    rec.name,
                    severity_to_str(rec.severity),
                    rec.family,
                    rec.first_seen,
                ])
                .map_err(|e| GuardianError::Database(format!("batch insert failed: {e}")))?;
                inserted += 1;
            }
        }

        tx.commit()
            .map_err(|e| GuardianError::Database(format!("commit failed: {e}")))?;

        debug!(count = inserted, "Batch insert complete");
        Ok(inserted)
    }

    // -----------------------------------------------------------------
    // Deletion
    // -----------------------------------------------------------------

    /// Remove signatures by SHA-256 hashes.  Returns how many were deleted.
    pub fn delete_hashes(&self, hashes: &[String]) -> Result<usize> {
        if hashes.is_empty() {
            return Ok(0);
        }

        let conn = self.conn.lock().map_err(|e| {
            GuardianError::Database(format!("database lock poisoned: {e}"))
        })?;

        let tx = conn
            .unchecked_transaction()
            .map_err(|e| GuardianError::Database(format!("begin transaction failed: {e}")))?;

        let mut deleted = 0usize;
        {
            let mut stmt = tx
                .prepare_cached("DELETE FROM signatures WHERE sha256 = ?1")
                .map_err(|e| GuardianError::Database(format!("prepare failed: {e}")))?;

            for hash in hashes {
                let n = stmt
                    .execute(params![hash])
                    .map_err(|e| GuardianError::Database(format!("delete failed: {e}")))?;
                deleted += n;
            }
        }

        tx.commit()
            .map_err(|e| GuardianError::Database(format!("commit failed: {e}")))?;

        debug!(count = deleted, "Batch delete complete");
        Ok(deleted)
    }

    // -----------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------

    /// Total number of signatures in the database.
    pub fn count(&self) -> Result<u64> {
        let conn = self.conn.lock().map_err(|e| {
            GuardianError::Database(format!("database lock poisoned: {e}"))
        })?;

        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM signatures", [], |row| row.get(0))
            .map_err(|e| GuardianError::Database(format!("count query failed: {e}")))?;

        Ok(count as u64)
    }

    /// Aggregate statistics for the database.
    pub fn get_stats(&self) -> Result<DbStats> {
        let conn = self.conn.lock().map_err(|e| {
            GuardianError::Database(format!("database lock poisoned: {e}"))
        })?;

        let total: i64 = conn
            .query_row("SELECT COUNT(*) FROM signatures", [], |row| row.get(0))
            .map_err(|e| GuardianError::Database(format!("count query failed: {e}")))?;

        let families: i64 = conn
            .query_row(
                "SELECT COUNT(DISTINCT family) FROM signatures WHERE family IS NOT NULL",
                [],
                |row| row.get(0),
            )
            .map_err(|e| GuardianError::Database(format!("families query failed: {e}")))?;

        let db_size = if self.path.exists() {
            std::fs::metadata(&self.path)
                .map(|m| m.len())
                .unwrap_or(0)
        } else {
            0
        };

        Ok(DbStats {
            total_signatures: total as u64,
            families: families as u64,
            db_size_bytes: db_size,
        })
    }

    // -----------------------------------------------------------------
    // Metadata helpers
    // -----------------------------------------------------------------

    /// Read a metadata value from the `db_meta` table.
    pub fn get_meta(&self, key: &str) -> Result<Option<String>> {
        let conn = self.conn.lock().map_err(|e| {
            GuardianError::Database(format!("database lock poisoned: {e}"))
        })?;

        let result = conn
            .query_row(
                "SELECT value FROM db_meta WHERE key = ?1",
                params![key],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| GuardianError::Database(format!("meta query failed: {e}")))?;

        Ok(result)
    }

    /// Set a metadata value in the `db_meta` table.
    pub fn set_meta(&self, key: &str, value: &str) -> Result<()> {
        let conn = self.conn.lock().map_err(|e| {
            GuardianError::Database(format!("database lock poisoned: {e}"))
        })?;

        conn.execute(
            "INSERT OR REPLACE INTO db_meta(key, value) VALUES (?1, ?2)",
            params![key, value],
        )
        .map_err(|e| GuardianError::Database(format!("meta update failed: {e}")))?;

        Ok(())
    }

    /// Return the filesystem path of this database.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

// ---------------------------------------------------------------------------
// rusqlite extension: Optional helper
// ---------------------------------------------------------------------------

/// Extension trait to provide `.optional()` on `rusqlite::Result` (mirrors
/// the pattern from the `rusqlite` crate itself).
trait OptionalExt<T> {
    fn optional(self) -> std::result::Result<Option<T>, rusqlite::Error>;
}

impl<T> OptionalExt<T> for std::result::Result<T, rusqlite::Error> {
    fn optional(self) -> std::result::Result<Option<T>, rusqlite::Error> {
        match self {
            Ok(v) => Ok(Some(v)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e),
        }
    }
}

// ---------------------------------------------------------------------------
// Severity ↔ string helpers
// ---------------------------------------------------------------------------

fn severity_to_str(s: Severity) -> &'static str {
    match s {
        Severity::Low => "low",
        Severity::Medium => "medium",
        Severity::High => "high",
        Severity::Critical => "critical",
    }
}

fn parse_severity(s: &str) -> Severity {
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

    fn sample_record(hash: &str) -> SignatureRecord {
        SignatureRecord {
            sha256: hash.into(),
            name: format!("Malware.Test.{}", &hash[..8]),
            severity: Severity::High,
            family: Some("TestFamily".into()),
            first_seen: "2025-01-15T00:00:00Z".into(),
        }
    }

    #[test]
    fn test_open_in_memory() {
        let db = SignatureDatabase::open_in_memory().unwrap();
        assert_eq!(db.count().unwrap(), 0);
    }

    #[test]
    fn test_insert_and_lookup() {
        let db = SignatureDatabase::open_in_memory().unwrap();
        let rec = sample_record("aabbccdd11223344aabbccdd11223344aabbccdd11223344aabbccdd11223344");
        db.insert(&rec).unwrap();

        let found = db.lookup_hash(&rec.sha256).unwrap();
        assert!(found.is_some());
        let found = found.unwrap();
        assert_eq!(found.name, rec.name);
        assert_eq!(found.severity, Severity::High);
        assert_eq!(found.family.as_deref(), Some("TestFamily"));
    }

    #[test]
    fn test_lookup_missing_hash() {
        let db = SignatureDatabase::open_in_memory().unwrap();
        let found = db.lookup_hash("0000000000000000000000000000000000000000000000000000000000000000").unwrap();
        assert!(found.is_none());
    }

    #[test]
    fn test_insert_batch() {
        let db = SignatureDatabase::open_in_memory().unwrap();
        let records: Vec<SignatureRecord> = (0..100)
            .map(|i| SignatureRecord {
                sha256: format!("{:064x}", i),
                name: format!("Malware.Batch.{i}"),
                severity: Severity::Medium,
                family: Some("BatchFamily".into()),
                first_seen: "2025-06-01T00:00:00Z".into(),
            })
            .collect();

        let inserted = db.insert_batch(&records).unwrap();
        assert_eq!(inserted, 100);
        assert_eq!(db.count().unwrap(), 100);

        // Verify a specific record.
        let found = db.lookup_hash(&format!("{:064x}", 42)).unwrap().unwrap();
        assert_eq!(found.name, "Malware.Batch.42");
    }

    #[test]
    fn test_insert_batch_empty() {
        let db = SignatureDatabase::open_in_memory().unwrap();
        let inserted = db.insert_batch(&[]).unwrap();
        assert_eq!(inserted, 0);
    }

    #[test]
    fn test_upsert_overwrites() {
        let db = SignatureDatabase::open_in_memory().unwrap();
        let hash = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
        let rec1 = SignatureRecord {
            sha256: hash.into(),
            name: "Original.Name".into(),
            severity: Severity::Low,
            family: None,
            first_seen: "2025-01-01T00:00:00Z".into(),
        };
        db.insert(&rec1).unwrap();

        let rec2 = SignatureRecord {
            sha256: hash.into(),
            name: "Updated.Name".into(),
            severity: Severity::Critical,
            family: Some("NewFamily".into()),
            first_seen: "2025-06-01T00:00:00Z".into(),
        };
        db.insert(&rec2).unwrap();

        assert_eq!(db.count().unwrap(), 1);
        let found = db.lookup_hash(hash).unwrap().unwrap();
        assert_eq!(found.name, "Updated.Name");
        assert_eq!(found.severity, Severity::Critical);
    }

    #[test]
    fn test_delete_hashes() {
        let db = SignatureDatabase::open_in_memory().unwrap();
        let records: Vec<SignatureRecord> = (0..5)
            .map(|i| SignatureRecord {
                sha256: format!("{:064x}", i),
                name: format!("Malware.Del.{i}"),
                severity: Severity::Medium,
                family: None,
                first_seen: "2025-01-01T00:00:00Z".into(),
            })
            .collect();
        db.insert_batch(&records).unwrap();
        assert_eq!(db.count().unwrap(), 5);

        let to_delete = vec![format!("{:064x}", 1), format!("{:064x}", 3)];
        let deleted = db.delete_hashes(&to_delete).unwrap();
        assert_eq!(deleted, 2);
        assert_eq!(db.count().unwrap(), 3);

        // Deleted hashes should not be found.
        assert!(db.lookup_hash(&format!("{:064x}", 1)).unwrap().is_none());
        assert!(db.lookup_hash(&format!("{:064x}", 3)).unwrap().is_none());

        // Non-deleted hashes should still exist.
        assert!(db.lookup_hash(&format!("{:064x}", 0)).unwrap().is_some());
    }

    #[test]
    fn test_delete_empty_list() {
        let db = SignatureDatabase::open_in_memory().unwrap();
        let deleted = db.delete_hashes(&[]).unwrap();
        assert_eq!(deleted, 0);
    }

    #[test]
    fn test_get_stats() {
        let db = SignatureDatabase::open_in_memory().unwrap();
        let records = vec![
            SignatureRecord {
                sha256: format!("{:064x}", 1),
                name: "A".into(),
                severity: Severity::High,
                family: Some("FamilyA".into()),
                first_seen: "2025-01-01T00:00:00Z".into(),
            },
            SignatureRecord {
                sha256: format!("{:064x}", 2),
                name: "B".into(),
                severity: Severity::Low,
                family: Some("FamilyB".into()),
                first_seen: "2025-01-01T00:00:00Z".into(),
            },
            SignatureRecord {
                sha256: format!("{:064x}", 3),
                name: "C".into(),
                severity: Severity::Medium,
                family: Some("FamilyA".into()),
                first_seen: "2025-01-01T00:00:00Z".into(),
            },
        ];
        db.insert_batch(&records).unwrap();

        let stats = db.get_stats().unwrap();
        assert_eq!(stats.total_signatures, 3);
        assert_eq!(stats.families, 2); // FamilyA and FamilyB
    }

    #[test]
    fn test_metadata() {
        let db = SignatureDatabase::open_in_memory().unwrap();

        // Default metadata.
        let version = db.get_meta("version").unwrap();
        assert_eq!(version.as_deref(), Some("1"));

        // Set custom metadata.
        db.set_meta("last_update", "2025-06-15T12:00:00Z").unwrap();
        let lu = db.get_meta("last_update").unwrap();
        assert_eq!(lu.as_deref(), Some("2025-06-15T12:00:00Z"));

        // Missing key.
        let missing = db.get_meta("nonexistent").unwrap();
        assert!(missing.is_none());
    }

    #[test]
    fn test_open_file_based() {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("test_sigs.db");

        {
            let db = SignatureDatabase::open(&db_path).unwrap();
            let rec = sample_record("1111111111111111111111111111111111111111111111111111111111111111");
            db.insert(&rec).unwrap();
            assert_eq!(db.count().unwrap(), 1);
        }

        // Re-open and verify persistence.
        {
            let db = SignatureDatabase::open(&db_path).unwrap();
            assert_eq!(db.count().unwrap(), 1);
            let found = db
                .lookup_hash("1111111111111111111111111111111111111111111111111111111111111111")
                .unwrap();
            assert!(found.is_some());
        }
    }

    #[test]
    fn test_severity_roundtrip() {
        for sev in [Severity::Low, Severity::Medium, Severity::High, Severity::Critical] {
            let s = severity_to_str(sev);
            let parsed = parse_severity(s);
            assert_eq!(parsed, sev);
        }
    }
}
