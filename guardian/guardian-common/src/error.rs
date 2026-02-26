//! Error types for the Guardian system.

use thiserror::Error;

/// Unified result type for Guardian.
pub type Result<T> = std::result::Result<T, GuardianError>;

/// Top-level error enum covering all Guardian subsystems.
#[derive(Debug, Error)]
pub enum GuardianError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Database error: {0}")]
    Database(String),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Signature engine error: {0}")]
    Signature(String),

    #[error("YARA error: {0}")]
    Yara(String),

    #[error("Heuristic error: {0}")]
    Heuristic(String),

    #[error("ML inference error: {0}")]
    Ml(String),

    #[error("Behavioral analysis error: {0}")]
    Behavioral(String),

    #[error("Network analysis error: {0}")]
    Network(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Scan timeout after {0}ms")]
    Timeout(u64),

    #[error("File too large: {size} bytes (max: {max})")]
    FileTooLarge { size: u64, max: u64 },

    #[error("Unsupported file type: {0}")]
    UnsupportedFileType(String),

    #[error("Archive bomb detected: compression ratio {ratio:.1}:1 exceeds limit")]
    ArchiveBomb { ratio: f64 },

    #[error("{0}")]
    Other(String),
}
