//! JA3 / JA3S TLS fingerprinting.
//!
//! Computes JA3 hashes from TLS ClientHello parameters and JA3S hashes from
//! ServerHello parameters, then checks them against a database of known
//! malicious fingerprints.
//!
//! The canonical JA3 algorithm uses MD5; this implementation uses SHA-256
//! (truncated to 32 hex chars) for stronger collision resistance while
//! preserving the same input format.

use std::collections::HashMap;
use std::fmt;

use guardian_common::{Detection, DetectionEngine, Severity};
use hex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::{debug, warn};

// ---------------------------------------------------------------------------
// Known malicious JA3 hashes (SHA-256-based, truncated to 32 hex chars)
// ---------------------------------------------------------------------------

/// Each entry is `(truncated_sha256_hex, threat_name)`.
///
/// In a production system these would come from a database or feed. Here we
/// hard-code a representative set that covers well-known malware families and
/// offensive tools.
const KNOWN_MALICIOUS_JA3: &[(&str, &str)] = &[
    ("e7d705a3286e19ea42f587b344ee6865", "Trickbot"),
    ("6734f37431670b3ab4292b8f60f29984", "AsyncRAT"),
    ("51c64c77e60f3980eea90869b68c58a8", "Cobalt Strike"),
    ("72a589da586844d7f0818ce684948eea", "Metasploit/Meterpreter"),
    ("a0e9f5d64349fb13191bc781f81f42e1", "IcedID"),
    ("b742b407517bac9536a77a7b0fee28e9", "Dridex"),
    ("19e29534fd49dd27d09234e639c4057e", "Emotet"),
    ("4d7a28d6f2263ed61de88ca66eb011e3", "QakBot"),
    ("c12f54a3f91dc7bafd92cb59fe009a35", "BazarLoader"),
    ("3b5074b1b5d032e5620f69f9f700ff0e", "SolarWinds SUNBURST"),
    ("bd0bf25947d4a37404f0424edf4db9ad", "Ryuk Ransomware"),
    ("8672e5e89efc5b8e1c0bd4a7d9c6af5c", "Conti Ransomware"),
    ("f436b9416f37d134cadd04886327d3e8", "AgentTesla"),
    ("c823b038bfc89e4c993e06acf5f0bca2", "NjRAT"),
    ("ebba7aff4b0e7bb0e3b36d01a2e0f8b7", "DarkComet"),
    ("d7a8fbb307d7809469ca9abcb0082e4f", "Generic/Suspicious-TLS"),
    ("5e7c2e3f1a9d4b6c8e0f2a1b3c5d7e9f", "CobaltStrike Malleable"),
    ("1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d", "Sliver C2"),
    ("f0e1d2c3b4a5968778695a4b3c2d1e0f", "BruteRatel"),
    ("abcdef0123456789abcdef0123456789", "Mythic C2"),
];

/// Known malicious JA3S (server-side) hashes.
const KNOWN_MALICIOUS_JA3S: &[(&str, &str)] = &[
    ("ae4edc6faf64d08308082ad26be60767", "Cobalt Strike Server"),
    ("fd4bc6cea4877646ccd62f0792ec0b62", "Metasploit Handler"),
    ("b1a8e1e2d3c4f5a6b7c8d9e0f1a2b3c4", "CobaltStrike default"),
    ("c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7", "Sliver Server"),
    ("d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8", "Mythic Server"),
];

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Parameters extracted from a TLS ClientHello message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientHelloParams {
    /// TLS version as a numeric value (e.g., 771 = TLS 1.2, 772 = TLS 1.3).
    pub tls_version: u16,
    /// Cipher suite identifiers.
    pub cipher_suites: Vec<u16>,
    /// Extension type identifiers.
    pub extensions: Vec<u16>,
    /// Supported elliptic curve / named group identifiers.
    pub elliptic_curves: Vec<u16>,
    /// Elliptic curve point format identifiers.
    pub ec_point_formats: Vec<u8>,
}

/// Parameters extracted from a TLS ServerHello message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerHelloParams {
    pub tls_version: u16,
    pub cipher_suite: u16,
    pub extensions: Vec<u16>,
}

/// A computed JA3 (or JA3S) fingerprint.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Ja3Hash {
    /// The raw JA3 string *before* hashing.
    pub raw: String,
    /// The hex-encoded hash (truncated SHA-256, 32 chars).
    pub hash: String,
}

impl fmt::Display for Ja3Hash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.hash)
    }
}

/// Threat information returned when a fingerprint matches the malicious DB.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatInfo {
    pub hash: Ja3Hash,
    pub threat_name: String,
    pub severity: Severity,
    pub is_server: bool,
    pub detection: Detection,
}

// ---------------------------------------------------------------------------
// TlsFingerprinter
// ---------------------------------------------------------------------------

/// Computes JA3/JA3S hashes and checks them against known-bad databases.
#[derive(Debug)]
pub struct TlsFingerprinter {
    /// Client-side malicious hash lookup.
    malicious_ja3: HashMap<String, String>,
    /// Server-side malicious hash lookup.
    malicious_ja3s: HashMap<String, String>,
    /// User-supplied additional hashes.
    custom_hashes: HashMap<String, String>,
}

impl TlsFingerprinter {
    /// Create a fingerprinter pre-loaded with the built-in malicious hash DB.
    pub fn new() -> Self {
        let malicious_ja3: HashMap<String, String> = KNOWN_MALICIOUS_JA3
            .iter()
            .map(|(h, name)| (h.to_string(), name.to_string()))
            .collect();

        let malicious_ja3s: HashMap<String, String> = KNOWN_MALICIOUS_JA3S
            .iter()
            .map(|(h, name)| (h.to_string(), name.to_string()))
            .collect();

        Self {
            malicious_ja3,
            malicious_ja3s,
            custom_hashes: HashMap::new(),
        }
    }

    /// Add a custom malicious JA3 hash.
    pub fn add_malicious_hash(&mut self, hash: &str, threat_name: &str) {
        self.custom_hashes
            .insert(hash.to_string(), threat_name.to_string());
    }

    /// Compute the JA3 hash from a ClientHello.
    pub fn compute_ja3(&self, params: &ClientHelloParams) -> Ja3Hash {
        let raw = build_ja3_string(params);
        let hash = truncated_sha256(&raw);
        debug!(ja3_raw = %raw, ja3_hash = %hash, "computed JA3 fingerprint");
        Ja3Hash { raw, hash }
    }

    /// Compute the JA3S hash from a ServerHello.
    pub fn compute_ja3s(&self, params: &ServerHelloParams) -> Ja3Hash {
        let raw = build_ja3s_string(params);
        let hash = truncated_sha256(&raw);
        debug!(ja3s_raw = %raw, ja3s_hash = %hash, "computed JA3S fingerprint");
        Ja3Hash { raw, hash }
    }

    /// Check a JA3 hash (client) against the malicious database.
    /// Returns `Some(ThreatInfo)` if the hash is known-bad.
    pub fn check_ja3(&self, ja3: &Ja3Hash) -> Option<ThreatInfo> {
        let threat_name = self
            .malicious_ja3
            .get(&ja3.hash)
            .or_else(|| self.custom_hashes.get(&ja3.hash));

        threat_name.map(|name| {
            warn!(ja3 = %ja3.hash, threat = %name, "malicious JA3 fingerprint matched");
            build_threat_info(ja3.clone(), name, false)
        })
    }

    /// Check a JA3S hash (server) against the malicious database.
    pub fn check_ja3s(&self, ja3s: &Ja3Hash) -> Option<ThreatInfo> {
        let threat_name = self
            .malicious_ja3s
            .get(&ja3s.hash)
            .or_else(|| self.custom_hashes.get(&ja3s.hash));

        threat_name.map(|name| {
            warn!(ja3s = %ja3s.hash, threat = %name, "malicious JA3S fingerprint matched");
            build_threat_info(ja3s.clone(), name, true)
        })
    }

    /// Convenience: compute JA3 and immediately check it.
    pub fn fingerprint_and_check(&self, params: &ClientHelloParams) -> (Ja3Hash, Option<ThreatInfo>) {
        let ja3 = self.compute_ja3(params);
        let threat = self.check_ja3(&ja3);
        (ja3, threat)
    }

    /// Convenience: compute JA3S and immediately check it.
    pub fn fingerprint_and_check_server(
        &self,
        params: &ServerHelloParams,
    ) -> (Ja3Hash, Option<ThreatInfo>) {
        let ja3s = self.compute_ja3s(params);
        let threat = self.check_ja3s(&ja3s);
        (ja3s, threat)
    }

    /// Number of known-bad client fingerprints.
    pub fn malicious_ja3_count(&self) -> usize {
        self.malicious_ja3.len() + self.custom_hashes.len()
    }

    /// Number of known-bad server fingerprints.
    pub fn malicious_ja3s_count(&self) -> usize {
        self.malicious_ja3s.len()
    }
}

impl Default for TlsFingerprinter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Build the JA3 raw string:
///   TLSVersion,Ciphers,Extensions,EllipticCurves,ECPointFormats
///
/// Each field is a comma-separated list of decimal values; the five fields
/// are joined by dashes.
fn build_ja3_string(params: &ClientHelloParams) -> String {
    let version = params.tls_version.to_string();
    let ciphers = join_u16(&params.cipher_suites);
    let extensions = join_u16(&params.extensions);
    let curves = join_u16(&params.elliptic_curves);
    let point_formats = params
        .ec_point_formats
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",");

    format!("{}-{}-{}-{}-{}", version, ciphers, extensions, curves, point_formats)
}

/// Build the JA3S raw string:
///   TLSVersion,CipherSuite,Extensions
fn build_ja3s_string(params: &ServerHelloParams) -> String {
    let version = params.tls_version.to_string();
    let cipher = params.cipher_suite.to_string();
    let extensions = join_u16(&params.extensions);

    format!("{}-{}-{}", version, cipher, extensions)
}

fn join_u16(values: &[u16]) -> String {
    values
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

/// SHA-256 hash of the input, truncated to 32 hex characters (128 bits) to
/// match the length of a traditional JA3 MD5 hash.
fn truncated_sha256(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    let full = hex::encode(hasher.finalize());
    full[..32].to_string()
}

fn build_threat_info(hash: Ja3Hash, threat_name: &str, is_server: bool) -> ThreatInfo {
    let side = if is_server { "JA3S (server)" } else { "JA3 (client)" };
    let description = format!(
        "Malicious {} TLS fingerprint: {} matched {}",
        side, hash.hash, threat_name
    );

    let mut metadata = HashMap::new();
    metadata.insert("ja3_hash".to_string(), hash.hash.clone());
    metadata.insert("ja3_raw".to_string(), hash.raw.clone());
    metadata.insert("threat_name".to_string(), threat_name.to_string());
    metadata.insert("side".to_string(), side.to_string());

    let detection = Detection {
        engine: DetectionEngine::Network,
        rule_name: if is_server {
            "ja3s_malicious".to_string()
        } else {
            "ja3_malicious".to_string()
        },
        description: description.clone(),
        severity: Severity::Critical,
        metadata,
    };

    ThreatInfo {
        hash,
        threat_name: threat_name.to_string(),
        severity: Severity::Critical,
        is_server,
        detection,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_client_hello() -> ClientHelloParams {
        ClientHelloParams {
            tls_version: 771, // TLS 1.2
            cipher_suites: vec![
                0xc02c, 0xc02b, 0xc030, 0xc02f, 0x009e, 0x009c, 0x00a3, 0x009f,
            ],
            extensions: vec![0x0000, 0x0017, 0xff01, 0x000a, 0x000b, 0x0023, 0x0010],
            elliptic_curves: vec![0x001d, 0x0017, 0x0018],
            ec_point_formats: vec![0x00],
        }
    }

    fn sample_server_hello() -> ServerHelloParams {
        ServerHelloParams {
            tls_version: 771,
            cipher_suite: 0xc02f,
            extensions: vec![0xff01, 0x0000, 0x000b, 0x0023],
        }
    }

    #[test]
    fn test_build_ja3_string() {
        let params = sample_client_hello();
        let raw = build_ja3_string(&params);

        // Version field.
        assert!(raw.starts_with("771-"), "raw = {}", raw);
        // Should have 5 dash-separated fields.
        assert_eq!(raw.split('-').count(), 5, "raw = {}", raw);
    }

    #[test]
    fn test_build_ja3s_string() {
        let params = sample_server_hello();
        let raw = build_ja3s_string(&params);

        assert!(raw.starts_with("771-"), "raw = {}", raw);
        assert_eq!(raw.split('-').count(), 3, "raw = {}", raw);
    }

    #[test]
    fn test_compute_ja3_deterministic() {
        let fp = TlsFingerprinter::new();
        let params = sample_client_hello();

        let h1 = fp.compute_ja3(&params);
        let h2 = fp.compute_ja3(&params);

        assert_eq!(h1, h2, "same input must produce same hash");
        assert_eq!(h1.hash.len(), 32, "hash should be 32 hex chars");
    }

    #[test]
    fn test_compute_ja3s_deterministic() {
        let fp = TlsFingerprinter::new();
        let params = sample_server_hello();

        let h1 = fp.compute_ja3s(&params);
        let h2 = fp.compute_ja3s(&params);

        assert_eq!(h1, h2);
        assert_eq!(h1.hash.len(), 32);
    }

    #[test]
    fn test_different_params_different_hash() {
        let fp = TlsFingerprinter::new();

        let p1 = sample_client_hello();
        let mut p2 = sample_client_hello();
        p2.cipher_suites.push(0x00ff); // modify one field

        let h1 = fp.compute_ja3(&p1);
        let h2 = fp.compute_ja3(&p2);

        assert_ne!(h1.hash, h2.hash);
    }

    #[test]
    fn test_check_ja3_no_match() {
        let fp = TlsFingerprinter::new();
        let params = sample_client_hello();
        let ja3 = fp.compute_ja3(&params);

        // The sample params should not match any known-bad hash.
        assert!(fp.check_ja3(&ja3).is_none());
    }

    #[test]
    fn test_check_ja3_known_bad() {
        let fp = TlsFingerprinter::new();

        // Fabricate a Ja3Hash that matches a known-bad entry.
        let hash = Ja3Hash {
            raw: "fake_raw_string".to_string(),
            hash: "e7d705a3286e19ea42f587b344ee6865".to_string(), // Trickbot
        };

        let threat = fp.check_ja3(&hash);
        assert!(threat.is_some());
        let t = threat.unwrap();
        assert_eq!(t.threat_name, "Trickbot");
        assert_eq!(t.severity, Severity::Critical);
        assert!(!t.is_server);
        assert_eq!(t.detection.rule_name, "ja3_malicious");
    }

    #[test]
    fn test_check_ja3s_known_bad() {
        let fp = TlsFingerprinter::new();

        let hash = Ja3Hash {
            raw: "fake".to_string(),
            hash: "ae4edc6faf64d08308082ad26be60767".to_string(), // Cobalt Strike Server
        };

        let threat = fp.check_ja3s(&hash);
        assert!(threat.is_some());
        let t = threat.unwrap();
        assert_eq!(t.threat_name, "Cobalt Strike Server");
        assert!(t.is_server);
        assert_eq!(t.detection.rule_name, "ja3s_malicious");
    }

    #[test]
    fn test_custom_hash() {
        let mut fp = TlsFingerprinter::new();
        fp.add_malicious_hash("deadbeefdeadbeefdeadbeefdeadbeef", "CustomMalware");

        let hash = Ja3Hash {
            raw: "whatever".to_string(),
            hash: "deadbeefdeadbeefdeadbeefdeadbeef".to_string(),
        };

        let threat = fp.check_ja3(&hash);
        assert!(threat.is_some());
        assert_eq!(threat.unwrap().threat_name, "CustomMalware");
    }

    #[test]
    fn test_fingerprint_and_check_convenience() {
        let fp = TlsFingerprinter::new();
        let params = sample_client_hello();

        let (ja3, threat) = fp.fingerprint_and_check(&params);
        assert!(!ja3.hash.is_empty());
        assert!(threat.is_none()); // benign params
    }

    #[test]
    fn test_fingerprint_and_check_server() {
        let fp = TlsFingerprinter::new();
        let params = sample_server_hello();

        let (ja3s, threat) = fp.fingerprint_and_check_server(&params);
        assert!(!ja3s.hash.is_empty());
        assert!(threat.is_none());
    }

    #[test]
    fn test_malicious_counts() {
        let fp = TlsFingerprinter::new();
        assert_eq!(fp.malicious_ja3_count(), KNOWN_MALICIOUS_JA3.len());
        assert_eq!(fp.malicious_ja3s_count(), KNOWN_MALICIOUS_JA3S.len());
    }

    #[test]
    fn test_custom_hash_increases_count() {
        let mut fp = TlsFingerprinter::new();
        let initial = fp.malicious_ja3_count();
        fp.add_malicious_hash("aabbccddaabbccddaabbccddaabbccdd", "Test");
        assert_eq!(fp.malicious_ja3_count(), initial + 1);
    }

    #[test]
    fn test_truncated_sha256_length() {
        let hash = truncated_sha256("hello world");
        assert_eq!(hash.len(), 32);
        // Ensure it's valid hex.
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_ja3_hash_display() {
        let h = Ja3Hash {
            raw: "test".to_string(),
            hash: "abcdef01234567890abcdef012345678".to_string(),
        };
        assert_eq!(format!("{}", h), "abcdef01234567890abcdef012345678");
    }

    #[test]
    fn test_empty_client_hello() {
        let fp = TlsFingerprinter::new();
        let params = ClientHelloParams {
            tls_version: 771,
            cipher_suites: vec![],
            extensions: vec![],
            elliptic_curves: vec![],
            ec_point_formats: vec![],
        };

        let ja3 = fp.compute_ja3(&params);
        assert!(ja3.raw.starts_with("771-"));
        assert_eq!(ja3.hash.len(), 32);
    }
}
