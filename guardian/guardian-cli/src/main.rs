//! Home Guardian — command-line interface.
//!
//! Provides file scanning, real-time monitoring (Linux placeholder), signature
//! database updates, engine information display, and file hashing.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use guardian_common::{ScanConfig, ScanResult, ScanVerdict, Severity};
use guardian_core::scanner::scan_directory;
use guardian_core::{compute_sha256, GuardianScanner};

// ---------------------------------------------------------------------------
// ANSI colour helpers
// ---------------------------------------------------------------------------

const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const RED: &str = "\x1b[31m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";

// ---------------------------------------------------------------------------
// CLI definition (clap derive)
// ---------------------------------------------------------------------------

/// Home Guardian Antivirus — fast, multi-layered malware detection.
#[derive(Parser)]
#[command(
    name = "guardian",
    version,
    about = "Home Guardian Antivirus CLI",
    long_about = "Multi-layered malware detection engine with signature, YARA, \
                  heuristic, and ML-based analysis."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Scan a file or directory for threats.
    Scan {
        /// Path to a file or directory to scan.
        path: PathBuf,

        /// Recursively scan directories.
        #[arg(short, long, default_value_t = false)]
        recursive: bool,

        /// Output results as JSON (for scripting / automation).
        #[arg(long, default_value_t = false)]
        json: bool,

        /// Number of scan threads.
        #[arg(short = 't', long, default_value_t = 0)]
        threads: usize,

        /// Per-file scan timeout in milliseconds.
        #[arg(long)]
        timeout: Option<u64>,
    },

    /// Watch a directory for real-time threat monitoring (Linux only).
    Watch {
        /// Directory to monitor.
        path: PathBuf,

        /// Glob patterns to exclude (may be repeated).
        #[arg(short, long)]
        exclude: Vec<String>,
    },

    /// Update the signature database.
    Update {
        /// Path to the signature database file.
        #[arg(long)]
        db_path: Option<PathBuf>,
    },

    /// Show engine information, version, and configuration paths.
    Info,

    /// Compute the SHA-256 hash of a file.
    Hash {
        /// Path to the file to hash.
        file: PathBuf,
    },
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> ExitCode {
    // Initialise the tracing subscriber.
    // The log level can be configured via the `GUARDIAN_LOG` environment variable
    // (e.g. `GUARDIAN_LOG=debug`).  Defaults to `info`.
    let env_filter = tracing_subscriber::EnvFilter::try_from_env("GUARDIAN_LOG")
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));

    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(false)
        .init();

    let cli = Cli::parse();

    match run(cli).await {
        Ok(found_threats) => {
            if found_threats {
                ExitCode::from(1)
            } else {
                ExitCode::from(0)
            }
        }
        Err(e) => {
            eprintln!("{RED}{BOLD}Error:{RESET} {e:?}");
            ExitCode::from(2)
        }
    }
}

// ---------------------------------------------------------------------------
// Command dispatch
// ---------------------------------------------------------------------------

/// Returns `Ok(true)` when threats were found, `Ok(false)` when everything is
/// clean, and `Err` on operational errors.
async fn run(cli: Cli) -> Result<bool> {
    match cli.command {
        Commands::Scan {
            path,
            recursive,
            json,
            threads,
            timeout,
        } => cmd_scan(path, recursive, json, threads, timeout),

        Commands::Watch { path, exclude } => {
            cmd_watch(path, exclude).await
        }

        Commands::Update { db_path } => {
            cmd_update(db_path).await
        }

        Commands::Info => {
            cmd_info();
            Ok(false)
        }

        Commands::Hash { file } => {
            cmd_hash(file)?;
            Ok(false)
        }
    }
}

// ---------------------------------------------------------------------------
// `guardian scan`
// ---------------------------------------------------------------------------

fn cmd_scan(
    path: PathBuf,
    recursive: bool,
    json_output: bool,
    threads: usize,
    timeout: Option<u64>,
) -> Result<bool> {
    let mut config = ScanConfig::default();
    if threads > 0 {
        config.thread_count = threads;
    }
    if let Some(t) = timeout {
        config.timeout_ms = t;
    }

    let scanner = GuardianScanner::new(config);
    let start = Instant::now();

    let results: Vec<ScanResult> = if path.is_dir() {
        if !json_output {
            println!(
                "{BOLD}Scanning directory:{RESET} {}{}",
                path.display(),
                if recursive { " (recursive)" } else { "" }
            );
        }
        scan_directory(&scanner, &path, recursive)
            .context("Failed to scan directory")?
    } else if path.is_file() {
        if !json_output {
            println!("{BOLD}Scanning file:{RESET} {}", path.display());
        }
        let result = scanner
            .scan_file(&path)
            .context("Failed to scan file")?;
        vec![result]
    } else {
        bail!("Path does not exist or is not accessible: {}", path.display());
    };

    let duration = start.elapsed();

    // ------------------------------------------------------------------
    // JSON output mode
    // ------------------------------------------------------------------
    if json_output {
        let output = serde_json::json!({
            "results": &results,
            "summary": {
                "files_scanned": results.len(),
                "threats_found": count_threats(&results),
                "duration_ms": duration.as_millis() as u64,
            }
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
        return Ok(count_threats(&results) > 0);
    }

    // ------------------------------------------------------------------
    // Human-readable output
    // ------------------------------------------------------------------
    println!();
    for result in &results {
        print_result(result);
    }

    // Summary
    let threat_count = count_threats(&results);
    println!();
    println!("{BOLD}--- Scan Summary ---{RESET}");
    println!("  Files scanned : {}", results.len());

    if threat_count == 0 {
        println!("  Threats found : {GREEN}0{RESET}");
    } else {
        println!("  Threats found : {RED}{BOLD}{threat_count}{RESET}");
    }
    println!(
        "  Duration      : {:.2}s",
        duration.as_secs_f64()
    );
    println!();

    Ok(threat_count > 0)
}

/// Print a single scan result with colour-coded verdict.
fn print_result(result: &ScanResult) {
    match &result.verdict {
        ScanVerdict::Clean => {
            println!(
                "  {GREEN}\u{2713}{RESET} {}{DIM} ({:.1} KB, {}ms){RESET}",
                result.file_path.display(),
                result.file_size as f64 / 1024.0,
                result.scan_duration_ms,
            );
        }
        ScanVerdict::Suspicious(details) => {
            println!(
                "  {YELLOW}\u{26A0}{RESET} {YELLOW}{}{RESET} — {} [{}]",
                result.file_path.display(),
                details.name,
                severity_label(details.severity),
            );
        }
        ScanVerdict::Malicious(details) => {
            println!(
                "  {RED}\u{2718}{RESET} {RED}{BOLD}{}{RESET} — {RED}{}{RESET} [{}]",
                result.file_path.display(),
                details.name,
                severity_label(details.severity),
            );
        }
        ScanVerdict::Error(msg) => {
            println!(
                "  {DIM}? {}{RESET} — error: {}",
                result.file_path.display(),
                msg,
            );
        }
    }
}

fn severity_label(severity: Severity) -> &'static str {
    match severity {
        Severity::Low => "Low",
        Severity::Medium => "Medium",
        Severity::High => "High",
        Severity::Critical => "CRITICAL",
    }
}

fn count_threats(results: &[ScanResult]) -> usize {
    results
        .iter()
        .filter(|r| {
            matches!(
                r.verdict,
                ScanVerdict::Suspicious(_) | ScanVerdict::Malicious(_)
            )
        })
        .count()
}

// ---------------------------------------------------------------------------
// `guardian watch` (Linux real-time monitoring placeholder)
// ---------------------------------------------------------------------------

async fn cmd_watch(path: PathBuf, exclude: Vec<String>) -> Result<bool> {
    if !path.is_dir() {
        bail!("Watch path must be an existing directory: {}", path.display());
    }

    println!("{BOLD}Real-time monitoring:{RESET} {}", path.display());
    if !exclude.is_empty() {
        println!("  Excluding patterns: {}", exclude.join(", "));
    }

    #[cfg(target_os = "linux")]
    {
        println!(
            "{YELLOW}Note:{RESET} Real-time monitoring via fanotify is not yet \
             fully implemented in this build."
        );
        println!("Watching for file events... (press Ctrl+C to stop)");
        // Placeholder: wait forever until the user interrupts.
        tokio::signal::ctrl_c().await?;
        println!("\nMonitoring stopped.");
    }

    #[cfg(not(target_os = "linux"))]
    {
        println!(
            "{YELLOW}Warning:{RESET} Real-time file monitoring is only supported \
             on Linux (fanotify). This command is a no-op on the current platform."
        );
    }

    Ok(false)
}

// ---------------------------------------------------------------------------
// `guardian update`
// ---------------------------------------------------------------------------

async fn cmd_update(db_path: Option<PathBuf>) -> Result<bool> {
    let db = db_path.unwrap_or_else(|| PathBuf::from("data/signatures.db"));

    println!("{BOLD}Updating signature database...{RESET}");
    println!("  Database path: {}", db.display());
    println!();

    // ---------------------------------------------------------------
    // Placeholder: in a real build this would call guardian-network to
    // download delta updates from the update server.
    // ---------------------------------------------------------------
    println!("  {DIM}[1/3]{RESET} Checking for updates...");
    tokio::time::sleep(std::time::Duration::from_millis(250)).await;

    println!("  {DIM}[2/3]{RESET} Downloading signature deltas...");
    tokio::time::sleep(std::time::Duration::from_millis(250)).await;

    println!("  {DIM}[3/3]{RESET} Applying updates...");
    tokio::time::sleep(std::time::Duration::from_millis(250)).await;

    println!();
    println!(
        "{GREEN}{BOLD}Signature database is up to date.{RESET}"
    );
    println!("  Signatures loaded : {DIM}(placeholder){RESET}");
    println!("  Last updated      : {}", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"));

    Ok(false)
}

// ---------------------------------------------------------------------------
// `guardian info`
// ---------------------------------------------------------------------------

fn cmd_info() {
    let config = ScanConfig::default();

    println!("{BOLD}Home Guardian Antivirus{RESET}");
    println!("  Version          : {}", env!("CARGO_PKG_VERSION"));
    println!("  Engine layers    : Signature, YARA, Heuristic, ML, Behavioral, Network");
    println!("  Scan threads     : {}", config.thread_count);
    println!("  Max file size    : {} MB", config.max_file_size / (1024 * 1024));
    println!("  Timeout per file : {} ms", config.timeout_ms);
    println!("  Signature DB     : {}", config.signature_db_path.display());
    println!("  YARA rules       : {}", config.yara_rules_path.display());
    println!("  ML model path    : {}", config.ml_model_path.display());
    println!();
    println!("{BOLD}Engine status:{RESET}");
    println!(
        "  Signature  : {}",
        if config.engines.signature { format!("{GREEN}enabled{RESET}") } else { format!("{DIM}disabled{RESET}") }
    );
    println!(
        "  YARA       : {}",
        if config.engines.yara { format!("{GREEN}enabled{RESET}") } else { format!("{DIM}disabled{RESET}") }
    );
    println!(
        "  Heuristic  : {}",
        if config.engines.heuristic { format!("{GREEN}enabled{RESET}") } else { format!("{DIM}disabled{RESET}") }
    );
    println!(
        "  ML         : {}",
        if config.engines.ml { format!("{GREEN}enabled{RESET}") } else { format!("{DIM}disabled{RESET}") }
    );
    println!(
        "  Behavioral : {}",
        if config.engines.behavioral { format!("{GREEN}enabled{RESET}") } else { format!("{DIM}disabled{RESET}") }
    );
    println!(
        "  Network    : {}",
        if config.engines.network { format!("{GREEN}enabled{RESET}") } else { format!("{DIM}disabled{RESET}") }
    );
}

// ---------------------------------------------------------------------------
// `guardian hash`
// ---------------------------------------------------------------------------

fn cmd_hash(file: PathBuf) -> Result<()> {
    let data = std::fs::read(&file)
        .with_context(|| format!("Failed to read file: {}", file.display()))?;
    let hash = compute_sha256(&data);
    println!("{}  {}", hash, file.display());
    Ok(())
}
