//! Shared types, errors, and configuration for Home Guardian.

pub mod config;
pub mod error;
pub mod types;

pub use config::ScanConfig;
pub use error::{GuardianError, Result};
pub use types::*;
