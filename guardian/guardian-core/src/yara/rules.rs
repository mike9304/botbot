//! YARA rule definitions and rule-file parser.
//!
//! Provides [`YaraRule`] — a self-contained pattern-matching rule — plus
//! a simple parser that can ingest a subset of the YARA text-file format
//! (`rule NAME { strings: ... condition: ... }`).  Built-in rules cover
//! the EICAR test file, UPX-packed PEs, PowerShell encoded commands,
//! Android permission abuse, and cryptocurrency miner indicators.

use guardian_common::{FileType, Severity};
use std::collections::HashMap;
use std::path::Path;

// ---------------------------------------------------------------------------
// Pattern types
// ---------------------------------------------------------------------------

/// A single byte-level pattern that a rule looks for inside a file.
#[derive(Debug, Clone)]
pub struct BytePattern {
    /// Human-readable identifier (e.g. `$eicar`).
    pub id: String,
    /// The raw bytes to search for.  Hex strings from YARA files are decoded
    /// into this representation.
    pub bytes: Vec<u8>,
    /// If `true`, the match is case-insensitive (ASCII only).
    pub nocase: bool,
}

/// How the patterns inside a rule combine to produce a match.
#[derive(Debug, Clone)]
pub enum RuleCondition {
    /// *Any one* of the patterns is enough.
    AnyOf,
    /// *All* patterns must be present.
    AllOf,
    /// At least `n` of the patterns must match.
    AtLeast(usize),
}

// ---------------------------------------------------------------------------
// YaraRule
// ---------------------------------------------------------------------------

/// A single YARA-style rule with metadata, patterns, and a condition.
#[derive(Debug, Clone)]
pub struct YaraRule {
    /// Rule name (e.g. `EICAR_test_file`).
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Severity assigned when this rule matches.
    pub severity: Severity,
    /// Free-form tags (e.g. `["test", "eicar"]`).
    pub tags: Vec<String>,
    /// Optional MITRE ATT&CK technique ID.
    pub mitre_id: Option<String>,
    /// File types this rule applies to.  Empty means *all* types.
    pub applicable_types: Vec<FileType>,
    /// The byte patterns to search for.
    pub patterns: Vec<BytePattern>,
    /// How the patterns combine.
    pub condition: RuleCondition,
    /// Arbitrary key-value metadata.
    pub metadata: HashMap<String, String>,
}

impl YaraRule {
    /// Check whether this rule should run against the given file type.
    pub fn applies_to(&self, ft: FileType) -> bool {
        self.applicable_types.is_empty() || self.applicable_types.contains(&ft)
    }

    /// Scan `data` against this rule, returning `true` on a match.
    pub fn matches(&self, data: &[u8]) -> bool {
        let hit_count = self
            .patterns
            .iter()
            .filter(|p| pattern_found(data, p))
            .count();

        match &self.condition {
            RuleCondition::AnyOf => hit_count > 0,
            RuleCondition::AllOf => hit_count == self.patterns.len(),
            RuleCondition::AtLeast(n) => hit_count >= *n,
        }
    }
}

/// Brute-force substring search for a single [`BytePattern`].
fn pattern_found(haystack: &[u8], pattern: &BytePattern) -> bool {
    if pattern.bytes.is_empty() || pattern.bytes.len() > haystack.len() {
        return false;
    }

    if pattern.nocase {
        let needle: Vec<u8> = pattern.bytes.iter().map(|b| b.to_ascii_lowercase()).collect();
        haystack
            .windows(needle.len())
            .any(|window| {
                window
                    .iter()
                    .zip(needle.iter())
                    .all(|(a, b)| a.to_ascii_lowercase() == *b)
            })
    } else {
        haystack.windows(pattern.bytes.len()).any(|w| w == pattern.bytes.as_slice())
    }
}

// ---------------------------------------------------------------------------
// Built-in rules
// ---------------------------------------------------------------------------

/// Return the full set of built-in rules that ship with Guardian.
pub fn builtin_rules() -> Vec<YaraRule> {
    vec![
        eicar_rule(),
        upx_packed_pe_rule(),
        powershell_encoded_command_rule(),
        android_permission_abuse_rule(),
        crypto_miner_rule(),
    ]
}

/// EICAR anti-malware test file.
fn eicar_rule() -> YaraRule {
    YaraRule {
        name: "EICAR_test_file".into(),
        description: "Matches the EICAR anti-malware test string".into(),
        severity: Severity::High,
        tags: vec!["test".into(), "eicar".into()],
        mitre_id: None,
        applicable_types: vec![], // all types
        patterns: vec![BytePattern {
            id: "$eicar".into(),
            bytes: b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"
                .to_vec(),
            nocase: false,
        }],
        condition: RuleCondition::AnyOf,
        metadata: HashMap::new(),
    }
}

/// Detect UPX-packed PE executables.
fn upx_packed_pe_rule() -> YaraRule {
    YaraRule {
        name: "UPX_packed_PE".into(),
        description: "Detects UPX-packed Portable Executable files".into(),
        severity: Severity::Medium,
        tags: vec!["packer".into(), "upx".into(), "pe".into()],
        mitre_id: Some("T1027.002".into()), // Obfuscated Files – Software Packing
        applicable_types: vec![FileType::PE],
        patterns: vec![
            BytePattern {
                id: "$upx0".into(),
                bytes: b"UPX0".to_vec(),
                nocase: false,
            },
            BytePattern {
                id: "$upx1".into(),
                bytes: b"UPX1".to_vec(),
                nocase: false,
            },
            BytePattern {
                id: "$upx_sig".into(),
                bytes: b"UPX!".to_vec(),
                nocase: false,
            },
        ],
        // UPX typically has all three section markers.
        condition: RuleCondition::AtLeast(2),
        metadata: {
            let mut m = HashMap::new();
            m.insert("author".into(), "Guardian".into());
            m.insert("reference".into(), "https://upx.github.io/".into());
            m
        },
    }
}

/// Suspicious PowerShell encoded commands.
fn powershell_encoded_command_rule() -> YaraRule {
    YaraRule {
        name: "Suspicious_PowerShell_EncodedCommand".into(),
        description: "Detects PowerShell invocations using encoded/hidden commands".into(),
        severity: Severity::High,
        tags: vec!["powershell".into(), "script".into(), "obfuscation".into()],
        mitre_id: Some("T1059.001".into()), // Command and Scripting Interpreter: PowerShell
        applicable_types: vec![FileType::Script, FileType::PE, FileType::Unknown],
        patterns: vec![
            BytePattern {
                id: "$enc_cmd".into(),
                bytes: b"-EncodedCommand".to_vec(),
                nocase: true,
            },
            BytePattern {
                id: "$enc_short".into(),
                bytes: b"-enc ".to_vec(),
                nocase: true,
            },
            BytePattern {
                id: "$hidden".into(),
                bytes: b"-WindowStyle Hidden".to_vec(),
                nocase: true,
            },
            BytePattern {
                id: "$bypass".into(),
                bytes: b"-ExecutionPolicy Bypass".to_vec(),
                nocase: true,
            },
            BytePattern {
                id: "$noprofile".into(),
                bytes: b"-NoProfile".to_vec(),
                nocase: true,
            },
        ],
        condition: RuleCondition::AtLeast(2),
        metadata: HashMap::new(),
    }
}

/// Android DEX files requesting dangerous permissions.
fn android_permission_abuse_rule() -> YaraRule {
    YaraRule {
        name: "Android_Permission_Abuse".into(),
        description: "DEX file referencing excessive dangerous permissions".into(),
        severity: Severity::Medium,
        tags: vec!["android".into(), "dex".into(), "permissions".into()],
        mitre_id: Some("T1404".into()), // Exploitation for Privilege Escalation (Mobile)
        applicable_types: vec![FileType::DEX],
        patterns: vec![
            BytePattern {
                id: "$sms".into(),
                bytes: b"android.permission.SEND_SMS".to_vec(),
                nocase: false,
            },
            BytePattern {
                id: "$contacts".into(),
                bytes: b"android.permission.READ_CONTACTS".to_vec(),
                nocase: false,
            },
            BytePattern {
                id: "$camera".into(),
                bytes: b"android.permission.CAMERA".to_vec(),
                nocase: false,
            },
            BytePattern {
                id: "$record".into(),
                bytes: b"android.permission.RECORD_AUDIO".to_vec(),
                nocase: false,
            },
            BytePattern {
                id: "$location".into(),
                bytes: b"android.permission.ACCESS_FINE_LOCATION".to_vec(),
                nocase: false,
            },
            BytePattern {
                id: "$call_log".into(),
                bytes: b"android.permission.READ_CALL_LOG".to_vec(),
                nocase: false,
            },
        ],
        // Four or more dangerous permissions is suspicious.
        condition: RuleCondition::AtLeast(4),
        metadata: HashMap::new(),
    }
}

/// Cryptocurrency miner indicators.
fn crypto_miner_rule() -> YaraRule {
    YaraRule {
        name: "CryptoMiner_Indicators".into(),
        description: "Detects indicators of cryptocurrency mining software".into(),
        severity: Severity::High,
        tags: vec!["miner".into(), "cryptominer".into()],
        mitre_id: Some("T1496".into()), // Resource Hijacking
        applicable_types: vec![], // all types
        patterns: vec![
            BytePattern {
                id: "$stratum".into(),
                bytes: b"stratum+tcp://".to_vec(),
                nocase: true,
            },
            BytePattern {
                id: "$stratum_ssl".into(),
                bytes: b"stratum+ssl://".to_vec(),
                nocase: true,
            },
            BytePattern {
                id: "$xmrig".into(),
                bytes: b"xmrig".to_vec(),
                nocase: true,
            },
            BytePattern {
                id: "$monero".into(),
                bytes: b"cryptonight".to_vec(),
                nocase: true,
            },
            BytePattern {
                id: "$pool".into(),
                bytes: b"mining.pool".to_vec(),
                nocase: true,
            },
            BytePattern {
                id: "$wallet_pattern".into(),
                // Typical XMR address prefix
                bytes: b"4{44}".to_vec(),
                nocase: false,
            },
        ],
        condition: RuleCondition::AtLeast(2),
        metadata: {
            let mut m = HashMap::new();
            m.insert("author".into(), "Guardian".into());
            m
        },
    }
}

// ---------------------------------------------------------------------------
// Simple YARA-format text parser
// ---------------------------------------------------------------------------

/// Errors that can occur during rule parsing.
#[derive(Debug)]
pub struct RuleParseError {
    pub line: usize,
    pub message: String,
}

impl std::fmt::Display for RuleParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

/// Parse a YARA rule file and return zero or more [`YaraRule`] values.
///
/// Supports a simplified subset of YARA syntax:
///
/// ```text
/// rule RuleName : tag1 tag2 {
///     meta:
///         description = "..."
///         severity = "high"
///         mitre_id = "T1059"
///     strings:
///         $name = "text pattern"
///         $hex  = { 4D 5A 90 }
///     condition:
///         any of them
/// }
/// ```
pub fn parse_rule_file(source: &str) -> std::result::Result<Vec<YaraRule>, RuleParseError> {
    let mut rules = Vec::new();
    let mut chars = source.char_indices().peekable();
    let mut line_no: usize = 1;

    // Count newlines up to a byte offset.
    let line_at = |src: &str, pos: usize| -> usize {
        1 + src[..pos].bytes().filter(|&b| b == b'\n').count()
    };

    while let Some(&(pos, _)) = chars.peek() {
        // Skip whitespace.
        skip_ws(&mut chars, &mut line_no);
        if chars.peek().is_none() {
            break;
        }

        // Look for `rule` keyword.
        let word = take_word(&mut chars);
        if word.is_empty() {
            // Skip stray characters.
            chars.next();
            continue;
        }
        if word == "import" || word == "include" {
            // Skip the rest of the line.
            skip_line(&mut chars, &mut line_no);
            continue;
        }
        if word != "rule" {
            // Skip unknown top-level token.
            continue;
        }

        skip_ws(&mut chars, &mut line_no);
        let name = take_word(&mut chars);
        if name.is_empty() {
            return Err(RuleParseError {
                line: line_at(source, pos),
                message: "expected rule name after `rule`".into(),
            });
        }

        skip_ws(&mut chars, &mut line_no);

        // Optional tags after `:`.
        let mut tags = Vec::new();
        if chars.peek().map(|&(_, c)| c) == Some(':') {
            chars.next(); // consume ':'
            skip_ws(&mut chars, &mut line_no);
            loop {
                let tag = take_word(&mut chars);
                if tag.is_empty() {
                    break;
                }
                tags.push(tag);
                skip_ws(&mut chars, &mut line_no);
            }
        }

        // Expect `{`.
        skip_ws(&mut chars, &mut line_no);
        match chars.peek() {
            Some(&(_, '{')) => {
                chars.next();
            }
            _ => {
                return Err(RuleParseError {
                    line: line_at(source, pos),
                    message: format!("expected '{{' after rule `{name}`"),
                });
            }
        }

        // Parse body sections until `}`.
        let mut description = String::new();
        let mut severity = Severity::Medium;
        let mut mitre_id: Option<String> = None;
        let mut metadata = HashMap::new();
        let mut patterns: Vec<BytePattern> = Vec::new();
        let mut condition = RuleCondition::AnyOf;

        enum Section {
            None,
            Meta,
            Strings,
            Condition,
        }
        let mut section = Section::None;

        loop {
            skip_ws(&mut chars, &mut line_no);
            match chars.peek() {
                None => {
                    return Err(RuleParseError {
                        line: line_at(source, pos),
                        message: format!("unexpected EOF inside rule `{name}`"),
                    });
                }
                Some(&(_, '}')) => {
                    chars.next();
                    break;
                }
                _ => {}
            }

            let kw = take_word(&mut chars);
            skip_ws(&mut chars, &mut line_no);

            // Section headers.
            if kw == "meta" || kw == "meta:" {
                consume_colon(&mut chars);
                section = Section::Meta;
                continue;
            }
            if kw == "strings" || kw == "strings:" {
                consume_colon(&mut chars);
                section = Section::Strings;
                continue;
            }
            if kw == "condition" || kw == "condition:" {
                consume_colon(&mut chars);
                section = Section::Condition;
                // Read until `}` — capture the condition text.
                let cond_text = take_until_brace(&mut chars, &mut line_no);
                condition = parse_condition(&cond_text);
                continue;
            }

            match section {
                Section::Meta => {
                    // key = "value"
                    let key = kw;
                    skip_ws(&mut chars, &mut line_no);
                    consume_eq(&mut chars);
                    skip_ws(&mut chars, &mut line_no);
                    let value = take_quoted(&mut chars);
                    match key.as_str() {
                        "description" => description = value.clone(),
                        "severity" => severity = parse_severity(&value),
                        "mitre_id" => mitre_id = Some(value.clone()),
                        _ => {}
                    }
                    metadata.insert(key, value);
                }
                Section::Strings => {
                    // $id = "text" or $id = { hex bytes }
                    let id = kw; // includes leading `$`
                    skip_ws(&mut chars, &mut line_no);
                    consume_eq(&mut chars);
                    skip_ws(&mut chars, &mut line_no);

                    let mut nocase = false;
                    let bytes = match chars.peek() {
                        Some(&(_, '"')) => {
                            let text = take_quoted(&mut chars);
                            // Check for modifiers after the string.
                            skip_ws(&mut chars, &mut line_no);
                            let modifier = peek_word(&chars);
                            if modifier == "nocase" {
                                let _ = take_word(&mut chars);
                                nocase = true;
                            }
                            text.into_bytes()
                        }
                        Some(&(_, '{')) => {
                            chars.next();
                            let hex = take_until_char(&mut chars, '}');
                            parse_hex_string(&hex)
                        }
                        _ => Vec::new(),
                    };
                    if !bytes.is_empty() {
                        patterns.push(BytePattern { id, bytes, nocase });
                    }
                }
                Section::Condition | Section::None => {
                    // Already handled or ignored.
                }
            }
        }

        rules.push(YaraRule {
            name,
            description,
            severity,
            tags,
            mitre_id,
            applicable_types: vec![],
            patterns,
            condition,
            metadata,
        });
    }

    Ok(rules)
}

/// Load rules from a YARA file on disk.
pub fn load_rules_from_file(path: &Path) -> std::result::Result<Vec<YaraRule>, RuleParseError> {
    let source = std::fs::read_to_string(path).map_err(|e| RuleParseError {
        line: 0,
        message: format!("cannot read {}: {e}", path.display()),
    })?;
    parse_rule_file(&source)
}

/// Load all `.yar` / `.yara` files from a directory (non-recursive).
pub fn load_rules_from_dir(dir: &Path) -> std::result::Result<Vec<YaraRule>, RuleParseError> {
    let entries = std::fs::read_dir(dir).map_err(|e| RuleParseError {
        line: 0,
        message: format!("cannot read directory {}: {e}", dir.display()),
    })?;

    let mut all_rules = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        if ext == "yar" || ext == "yara" {
            match load_rules_from_file(&path) {
                Ok(rules) => all_rules.extend(rules),
                Err(e) => {
                    tracing::warn!(
                        path = %path.display(),
                        error = %e,
                        "Failed to parse YARA rule file — skipping"
                    );
                }
            }
        }
    }
    Ok(all_rules)
}

// ---------------------------------------------------------------------------
// Parser helpers
// ---------------------------------------------------------------------------

fn skip_ws(
    chars: &mut std::iter::Peekable<std::str::CharIndices<'_>>,
    line_no: &mut usize,
) {
    while let Some(&(_, c)) = chars.peek() {
        if c == '\n' {
            *line_no += 1;
            chars.next();
        } else if c.is_whitespace() {
            chars.next();
        } else if c == '/' {
            // Peek ahead for `//` line comment.
            let saved = chars.clone();
            chars.next();
            if chars.peek().map(|&(_, c2)| c2) == Some('/') {
                // Skip to end of line.
                skip_line(chars, line_no);
            } else {
                // Not a comment — put back.
                *chars = saved;
                break;
            }
        } else {
            break;
        }
    }
}

fn skip_line(
    chars: &mut std::iter::Peekable<std::str::CharIndices<'_>>,
    line_no: &mut usize,
) {
    for (_, c) in chars.by_ref() {
        if c == '\n' {
            *line_no += 1;
            return;
        }
    }
}

fn take_word(chars: &mut std::iter::Peekable<std::str::CharIndices<'_>>) -> String {
    let mut word = String::new();
    while let Some(&(_, c)) = chars.peek() {
        if c.is_alphanumeric() || c == '_' || c == '$' || c == '.' || c == ':' {
            word.push(c);
            chars.next();
        } else {
            break;
        }
    }
    // Strip trailing colon (section headers like `meta:` become `meta`).
    if word.ends_with(':') && word.len() > 1 {
        word.pop();
    }
    word
}

fn peek_word(chars: &std::iter::Peekable<std::str::CharIndices<'_>>) -> String {
    let mut cloned = chars.clone();
    take_word(&mut cloned)
}

fn take_quoted(chars: &mut std::iter::Peekable<std::str::CharIndices<'_>>) -> String {
    let mut result = String::new();
    if chars.peek().map(|&(_, c)| c) != Some('"') {
        return result;
    }
    chars.next(); // opening quote
    let mut escaped = false;
    for (_, c) in chars.by_ref() {
        if escaped {
            match c {
                'n' => result.push('\n'),
                't' => result.push('\t'),
                '\\' => result.push('\\'),
                '"' => result.push('"'),
                _ => {
                    result.push('\\');
                    result.push(c);
                }
            }
            escaped = false;
        } else if c == '\\' {
            escaped = true;
        } else if c == '"' {
            break;
        } else {
            result.push(c);
        }
    }
    result
}

fn consume_colon(chars: &mut std::iter::Peekable<std::str::CharIndices<'_>>) {
    if chars.peek().map(|&(_, c)| c) == Some(':') {
        chars.next();
    }
}

fn consume_eq(chars: &mut std::iter::Peekable<std::str::CharIndices<'_>>) {
    if chars.peek().map(|&(_, c)| c) == Some('=') {
        chars.next();
    }
}

fn take_until_brace(
    chars: &mut std::iter::Peekable<std::str::CharIndices<'_>>,
    line_no: &mut usize,
) -> String {
    let mut text = String::new();
    while let Some(&(_, c)) = chars.peek() {
        if c == '}' {
            break;
        }
        if c == '\n' {
            *line_no += 1;
        }
        text.push(c);
        chars.next();
    }
    text
}

fn take_until_char(
    chars: &mut std::iter::Peekable<std::str::CharIndices<'_>>,
    stop: char,
) -> String {
    let mut text = String::new();
    for (_, c) in chars.by_ref() {
        if c == stop {
            break;
        }
        text.push(c);
    }
    text
}

fn parse_hex_string(hex: &str) -> Vec<u8> {
    let cleaned: String = hex.chars().filter(|c| c.is_ascii_hexdigit()).collect();
    (0..cleaned.len() / 2)
        .filter_map(|i| u8::from_str_radix(&cleaned[i * 2..i * 2 + 2], 16).ok())
        .collect()
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

fn parse_condition(text: &str) -> RuleCondition {
    let trimmed = text.trim().to_lowercase();

    if trimmed.contains("all of") {
        return RuleCondition::AllOf;
    }
    if trimmed.contains("any of") {
        return RuleCondition::AnyOf;
    }
    // `N of them` or `N of ($...)`.
    if let Some(n_str) = trimmed.split_whitespace().next() {
        if let Ok(n) = n_str.parse::<usize>() {
            return RuleCondition::AtLeast(n);
        }
    }
    // Default to any-of.
    RuleCondition::AnyOf
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_rules_load() {
        let rules = builtin_rules();
        assert_eq!(rules.len(), 5);
        assert_eq!(rules[0].name, "EICAR_test_file");
    }

    #[test]
    fn test_eicar_match() {
        let rules = builtin_rules();
        let eicar = rules.iter().find(|r| r.name == "EICAR_test_file").unwrap();
        let data =
            b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*";
        assert!(eicar.matches(data));
        assert!(!eicar.matches(b"This is not EICAR"));
    }

    #[test]
    fn test_upx_match() {
        let rules = builtin_rules();
        let upx = rules.iter().find(|r| r.name == "UPX_packed_PE").unwrap();
        // Contains UPX0, UPX1, and UPX! markers.
        let mut data = Vec::new();
        data.extend_from_slice(b"MZ");
        data.extend_from_slice(&[0u8; 100]);
        data.extend_from_slice(b"UPX0");
        data.extend_from_slice(&[0u8; 50]);
        data.extend_from_slice(b"UPX1");
        data.extend_from_slice(&[0u8; 50]);
        data.extend_from_slice(b"UPX!");
        assert!(upx.matches(&data));

        // Only one marker is not enough.
        assert!(!upx.matches(b"MZ...UPX0...only_one"));
    }

    #[test]
    fn test_powershell_match() {
        let rules = builtin_rules();
        let ps = rules
            .iter()
            .find(|r| r.name == "Suspicious_PowerShell_EncodedCommand")
            .unwrap();
        let data = b"powershell.exe -encodedcommand SQBFAF... -windowstyle hidden";
        assert!(ps.matches(data));
        assert!(!ps.matches(b"echo hello"));
    }

    #[test]
    fn test_crypto_miner_match() {
        let rules = builtin_rules();
        let miner = rules
            .iter()
            .find(|r| r.name == "CryptoMiner_Indicators")
            .unwrap();
        let data = b"config: stratum+tcp://pool.example.com xmrig --threads=4";
        assert!(miner.matches(data));
    }

    #[test]
    fn test_applies_to() {
        let rules = builtin_rules();
        let upx = rules.iter().find(|r| r.name == "UPX_packed_PE").unwrap();
        assert!(upx.applies_to(FileType::PE));
        assert!(!upx.applies_to(FileType::ELF));

        let eicar = rules.iter().find(|r| r.name == "EICAR_test_file").unwrap();
        assert!(eicar.applies_to(FileType::Unknown)); // empty list = all types
    }

    #[test]
    fn test_nocase_matching() {
        let pattern = BytePattern {
            id: "$test".into(),
            bytes: b"hello".to_vec(),
            nocase: true,
        };
        assert!(pattern_found(b"say HELLO world", &pattern));
        assert!(pattern_found(b"HeLLo", &pattern));
        assert!(!pattern_found(b"goodbye", &pattern));
    }

    #[test]
    fn test_parse_simple_rule() {
        let source = r#"
rule TestRule : tag1 tag2 {
    meta:
        description = "A test rule"
        severity = "high"
        mitre_id = "T1234"
    strings:
        $a = "malware_string"
        $b = { 4D 5A 90 00 }
    condition:
        any of them
}
"#;
        let rules = parse_rule_file(source).unwrap();
        assert_eq!(rules.len(), 1);
        let r = &rules[0];
        assert_eq!(r.name, "TestRule");
        assert_eq!(r.tags, vec!["tag1", "tag2"]);
        assert_eq!(r.description, "A test rule");
        assert_eq!(r.severity, Severity::High);
        assert_eq!(r.mitre_id.as_deref(), Some("T1234"));
        assert_eq!(r.patterns.len(), 2);
        assert_eq!(r.patterns[0].id, "$a");
        assert_eq!(r.patterns[0].bytes, b"malware_string");
        assert_eq!(r.patterns[1].id, "$b");
        assert_eq!(r.patterns[1].bytes, vec![0x4D, 0x5A, 0x90, 0x00]);
    }

    #[test]
    fn test_parse_multiple_rules() {
        let source = r#"
rule Rule1 {
    strings:
        $a = "first"
    condition:
        any of them
}

rule Rule2 {
    strings:
        $b = "second"
    condition:
        all of them
}
"#;
        let rules = parse_rule_file(source).unwrap();
        assert_eq!(rules.len(), 2);
        assert_eq!(rules[0].name, "Rule1");
        assert_eq!(rules[1].name, "Rule2");
    }

    #[test]
    fn test_parse_condition_variants() {
        assert!(matches!(parse_condition("any of them"), RuleCondition::AnyOf));
        assert!(matches!(
            parse_condition("all of them"),
            RuleCondition::AllOf
        ));
        assert!(matches!(
            parse_condition("3 of them"),
            RuleCondition::AtLeast(3)
        ));
    }

    #[test]
    fn test_parse_hex_string() {
        assert_eq!(parse_hex_string("4D 5A 90 00"), vec![0x4D, 0x5A, 0x90, 0x00]);
        assert_eq!(parse_hex_string("  FF  EE  "), vec![0xFF, 0xEE]);
    }

    #[test]
    fn test_parse_severity_variants() {
        assert_eq!(parse_severity("low"), Severity::Low);
        assert_eq!(parse_severity("HIGH"), Severity::High);
        assert_eq!(parse_severity("Critical"), Severity::Critical);
        assert_eq!(parse_severity("unknown"), Severity::Medium);
    }
}
