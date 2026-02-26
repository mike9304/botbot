//! SQLite-backed SHA-256 hash database for confirmed signature lookups.
//!
//! This module provides the second tier of the signature matching pipeline.
//! After the bloom filter indicates a possible match, this module performs
//! an authoritative lookup against a SQLite database of known-malicious
//! file hashes.
//!
//! Features:
//! - WAL journal mode for concurrent read access during scans
//! - Batch insert for efficient database population
//! - Full signature metadata: name, severity, family, first_seen

use guardian_common::{GuardianError, Result, Severity};
use rusqlite::{params, Connection, OpenFlags};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tracing::{debug, info, warn};

/// A single signature record from the hash database.
#[derive(Debug, Clone)]
pub struct SignatureRecord {
    /// SHA-256 hash (hex-encoded, lowercase).
    pub sha256: String,
    /// Threat name (e.g., "Trojan.GenericKD.12345").
    pub name: String,
    /// Severity classification.
    pub severity: Severity,
    /// Malware family (e.g., "Emotet", "WannaCry").
    pub family: Option<String>,
    /// ISO 8601 date when the signature was first catalogued.
    pub first_seen: Option<String>,
}

/// SQLite hash database for authoritative signature lookups.
///
/// Thread-safety is provided by a `Mutex` around the SQLite connection,
/// which is required because `rusqlite::Connection` is not `Sync`.
pub struct HashDatabase {
    conn: Mutex<Connection>,
    db_path: PathBuf,
}

impl HashDatabase {
    /// Open (or create) a hash database at the given path.
    ///
    /// Initializes the schema and enables WAL mode for concurrent reads.
    pub fn open(path: &Path) -> Result<Self> {
        info!(path = %path.display(), "Opening signature hash database");

        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_CREATE
                | OpenFlags::SQLITE_OPEN_FULL_MUTEX,
        )
        .map_err(|e| GuardianError::Signature(format!("Failed to open hash DB: {}", e)))?;

        // Enable WAL mode for concurrent reads
        conn.execute_batch("PRAGMA journal_mode=WAL;")
            .map_err(|e| GuardianError::Signature(format!("Failed to set WAL mode: {}", e)))?;

        // Performance pragmas
        conn.execute_batch(
            "PRAGMA synchronous=NORMAL;
             PRAGMA cache_size=-64000;
             PRAGMA temp_store=MEMORY;",
        )
        .map_err(|e| GuardianError::Signature(format!("Failed to set pragmas: {}", e)))?;

        let db = Self {
            conn: Mutex::new(conn),
            db_path: path.to_path_buf(),
        };
        db.initialize_schema()?;

        Ok(db)
    }

    /// Create an in-memory database (useful for testing).
    pub fn open_in_memory() -> Result<Self> {
        info!("Opening in-memory signature hash database");

        let conn = Connection::open_in_memory()
            .map_err(|e| GuardianError::Signature(format!("Failed to open in-memory DB: {}", e)))?;

        let db = Self {
            conn: Mutex::new(conn),
            db_path: PathBuf::from(":memory:"),
        };
        db.initialize_schema()?;

        Ok(db)
    }

    /// Create the signatures table if it does not exist.
    fn initialize_schema(&self) -> Result<()> {
        let conn = self.conn.lock().map_err(|e| {
            GuardianError::Signature(format!("DB lock poisoned: {}", e))
        })?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS signatures (
                sha256     TEXT PRIMARY KEY NOT NULL,
                name       TEXT NOT NULL,
                severity   TEXT NOT NULL,
                family     TEXT,
                first_seen TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_signatures_family
                ON signatures(family);
            CREATE INDEX IF NOT EXISTS idx_signatures_severity
                ON signatures(severity);",
        )
        .map_err(|e| GuardianError::Signature(format!("Failed to create schema: {}", e)))?;

        debug!("Signature database schema initialized");
        Ok(())
    }

    /// Look up a SHA-256 hash in the database.
    ///
    /// Returns `Some(SignatureRecord)` if found, `None` otherwise.
    pub fn lookup(&self, sha256_hex: &str) -> Result<Option<SignatureRecord>> {
        let conn = self.conn.lock().map_err(|e| {
            GuardianError::Signature(format!("DB lock poisoned: {}", e))
        })?;

        let mut stmt = conn
            .prepare_cached(
                "SELECT sha256, name, severity, family, first_seen
                 FROM signatures WHERE sha256 = ?1",
            )
            .map_err(|e| GuardianError::Signature(format!("Prepare failed: {}", e)))?;

        let result = stmt
            .query_row(params![sha256_hex.to_lowercase()], |row| {
                Ok(SignatureRecord {
                    sha256: row.get(0)?,
                    name: row.get(1)?,
                    severity: parse_severity(&row.get::<_, String>(2)?),
                    family: row.get(3)?,
                    first_seen: row.get(4)?,
                })
            })
            .optional()
            .map_err(|e| GuardianError::Signature(format!("Query failed: {}", e)))?;

        Ok(result)
    }

    /// Insert a single signature record into the database.
    pub fn insert(&self, record: &SignatureRecord) -> Result<()> {
        let conn = self.conn.lock().map_err(|e| {
            GuardianError::Signature(format!("DB lock poisoned: {}", e))
        })?;

        conn.execute(
            "INSERT OR REPLACE INTO signatures (sha256, name, severity, family, first_seen)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                record.sha256.to_lowercase(),
                record.name,
                severity_to_string(record.severity),
                record.family,
                record.first_seen,
            ],
        )
        .map_err(|e| GuardianError::Signature(format!("Insert failed: {}", e)))?;

        Ok(())
    }

    /// Batch-insert multiple signature records within a single transaction.
    ///
    /// Returns the number of records successfully inserted.
    pub fn batch_insert(&self, records: &[SignatureRecord]) -> Result<usize> {
        let conn = self.conn.lock().map_err(|e| {
            GuardianError::Signature(format!("DB lock poisoned: {}", e))
        })?;

        let tx = conn
            .unchecked_transaction()
            .map_err(|e| GuardianError::Signature(format!("Transaction begin failed: {}", e)))?;

        let mut count = 0usize;
        {
            let mut stmt = tx
                .prepare_cached(
                    "INSERT OR REPLACE INTO signatures (sha256, name, severity, family, first_seen)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                )
                .map_err(|e| GuardianError::Signature(format!("Prepare failed: {}", e)))?;

            for record in records {
                match stmt.execute(params![
                    record.sha256.to_lowercase(),
                    record.name,
                    severity_to_string(record.severity),
                    record.family,
                    record.first_seen,
                ]) {
                    Ok(_) => count += 1,
                    Err(e) => {
                        warn!(
                            sha256 = %record.sha256,
                            error = %e,
                            "Failed to insert signature"
                        );
                    }
                }
            }
        }

        tx.commit()
            .map_err(|e| GuardianError::Signature(format!("Transaction commit failed: {}", e)))?;

        info!(inserted = count, total = records.len(), "Batch insert complete");
        Ok(count)
    }

    /// Count the total number of signatures in the database.
    pub fn count(&self) -> Result<u64> {
        let conn = self.conn.lock().map_err(|e| {
            GuardianError::Signature(format!("DB lock poisoned: {}", e))
        })?;

        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM signatures", [], |row| row.get(0))
            .map_err(|e| GuardianError::Signature(format!("Count query failed: {}", e)))?;

        Ok(count as u64)
    }

    /// Retrieve all SHA-256 hashes in the database (for bloom filter population).
    pub fn all_hashes(&self) -> Result<Vec<String>> {
        let conn = self.conn.lock().map_err(|e| {
            GuardianError::Signature(format!("DB lock poisoned: {}", e))
        })?;

        let mut stmt = conn
            .prepare("SELECT sha256 FROM signatures")
            .map_err(|e| GuardianError::Signature(format!("Prepare failed: {}", e)))?;

        let hashes = stmt
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(|e| GuardianError::Signature(format!("Query failed: {}", e)))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(hashes)
    }

    /// Path to the database file.
    pub fn db_path(&self) -> &Path {
        &self.db_path
    }
}

/// Convert a `Severity` to its string representation for storage.
fn severity_to_string(severity: Severity) -> &'static str {
    match severity {
        Severity::Low => "low",
        Severity::Medium => "medium",
        Severity::High => "high",
        Severity::Critical => "critical",
    }
}

/// Parse a severity string from the database back into a `Severity` enum.
fn parse_severity(s: &str) -> Severity {
    match s.to_lowercase().as_str() {
        "critical" => Severity::Critical,
        "high" => Severity::High,
        "medium" => Severity::Medium,
        _ => Severity::Low,
    }
}

/// Extension trait for `rusqlite::OptionalExtension` functionality.
trait OptionalExt<T> {
    fn optional(self) -> std::result::Result<Option<T>, rusqlite::Error>;
}

impl<T> OptionalExt<T> for std::result::Result<T, rusqlite::Error> {
    fn optional(self) -> std::result::Result<Option<T>, rusqlite::Error> {
        match self {
            Ok(val) => Ok(Some(val)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_record(sha256: &str, name: &str, severity: Severity) -> SignatureRecord {
        SignatureRecord {
            sha256: sha256.to_string(),
            name: name.to_string(),
            severity,
            family: Some("TestFamily".to_string()),
            first_seen: Some("2025-01-15T00:00:00Z".to_string()),
        }
    }

    #[test]
    fn test_open_in_memory() {
        let db = HashDatabase::open_in_memory().unwrap();
        assert_eq!(db.count().unwrap(), 0);
    }

    #[test]
    fn test_insert_and_lookup() {
        let db = HashDatabase::open_in_memory().unwrap();
        let hash = "a" .repeat(64); // 64 hex chars
        let record = sample_record(&hash, "Trojan.Test", Severity::High);

        db.insert(&record).unwrap();

        let found = db.lookup(&hash).unwrap();
        assert!(found.is_some());
        let found = found.unwrap();
        assert_eq!(found.name, "Trojan.Test");
        assert_eq!(found.severity, Severity::High);
        assert_eq!(found.family, Some("TestFamily".to_string()));
    }

    #[test]
    fn test_lookup_not_found() {
        let db = HashDatabase::open_in_memory().unwrap();
        let result = db.lookup("deadbeef").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_batch_insert() {
        let db = HashDatabase::open_in_memory().unwrap();

        let records: Vec<SignatureRecord> = (0..100)
            .map(|i| {
                let hash = format!("{:064x}", i);
                sample_record(&hash, &format!("Malware.Gen.{}", i), Severity::Medium)
            })
            .collect();

        let inserted = db.batch_insert(&records).unwrap();
        assert_eq!(inserted, 100);
        assert_eq!(db.count().unwrap(), 100);
    }

    #[test]
    fn test_batch_insert_upsert() {
        let db = HashDatabase::open_in_memory().unwrap();
        let hash = "b".repeat(64);

        let record1 = sample_record(&hash, "Trojan.OldName", Severity::Low);
        db.insert(&record1).unwrap();

        // Upsert with new name and severity
        let record2 = sample_record(&hash, "Trojan.NewName", Severity::Critical);
        db.insert(&record2).unwrap();

        let found = db.lookup(&hash).unwrap().unwrap();
        assert_eq!(found.name, "Trojan.NewName");
        assert_eq!(found.severity, Severity::Critical);

        // Still only one record
        assert_eq!(db.count().unwrap(), 1);
    }

    #[test]
    fn test_all_hashes() {
        let db = HashDatabase::open_in_memory().unwrap();

        let records: Vec<SignatureRecord> = (0..5)
            .map(|i| {
                let hash = format!("{:064x}", i);
                sample_record(&hash, &format!("Mal.{}", i), Severity::High)
            })
            .collect();

        db.batch_insert(&records).unwrap();

        let hashes = db.all_hashes().unwrap();
        assert_eq!(hashes.len(), 5);
    }

    #[test]
    fn test_severity_roundtrip() {
        for severity in [Severity::Low, Severity::Medium, Severity::High, Severity::Critical] {
            let s = severity_to_string(severity);
            let parsed = parse_severity(s);
            assert_eq!(parsed, severity);
        }
    }

    #[test]
    fn test_case_insensitive_lookup() {
        let db = HashDatabase::open_in_memory().unwrap();
        let hash_upper = "AABBCCDD".to_string() + &"00".repeat(28);
        let record = sample_record(&hash_upper, "Trojan.CaseTest", Severity::Medium);
        db.insert(&record).unwrap();

        // Lookup with lowercase should find it (stored as lowercase)
        let hash_lower = hash_upper.to_lowercase();
        let found = db.lookup(&hash_lower).unwrap();
        assert!(found.is_some());
    }

    #[test]
    fn test_open_file_database() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_signatures.db");

        {
            let db = HashDatabase::open(&db_path).unwrap();
            let hash = "c".repeat(64);
            db.insert(&sample_record(&hash, "Trojan.File", Severity::High))
                .unwrap();
            assert_eq!(db.count().unwrap(), 1);
        }

        // Re-open and verify persistence
        {
            let db = HashDatabase::open(&db_path).unwrap();
            let hash = "c".repeat(64);
            let found = db.lookup(&hash).unwrap();
            assert!(found.is_some());
            assert_eq!(found.unwrap().name, "Trojan.File");
        }
    }
}
