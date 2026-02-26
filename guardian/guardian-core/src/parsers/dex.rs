//! Android DEX (Dalvik Executable) parser.
//!
//! Custom parser that validates the DEX header, extracts the string table,
//! and scans for sensitive Android API usage and permission indicators.

use guardian_common::{FeatureVector, GuardianError, Result, Severity};

use super::{
    FileParser, ParsedFile, SuspiciousIndicator, indicators_to_metadata, shannon_entropy,
};

// ── DEX constants ───────────────────────────────────────────────────────────

/// Accepted DEX magic prefixes: "dex\n035\0", "dex\n037\0", "dex\n038\0",
/// "dex\n039\0", "dex\n041\0".
const DEX_MAGIC_PREFIX: &[u8; 4] = b"dex\n";
const DEX_HEADER_SIZE: usize = 0x70; // 112 bytes

// ── Sensitive API patterns ──────────────────────────────────────────────────

struct SensitiveApi {
    pattern: &'static str,
    name: &'static str,
    description: &'static str,
    severity: Severity,
    mitre_id: &'static str,
}

const SENSITIVE_APIS: &[SensitiveApi] = &[
    // SMS abuse
    SensitiveApi {
        pattern: "SmsManager",
        name: "SmsAccess",
        description: "References SmsManager — may send premium SMS or \
                       exfiltrate data via text messages",
        severity: Severity::High,
        mitre_id: "T1582.001",
    },
    SensitiveApi {
        pattern: "sendTextMessage",
        name: "SmsSend",
        description: "Calls sendTextMessage — may send SMS without user \
                       consent",
        severity: Severity::High,
        mitre_id: "T1582.001",
    },
    // Package / install abuse
    SensitiveApi {
        pattern: "PackageManager",
        name: "PackageManagerAccess",
        description: "References PackageManager — may enumerate installed \
                       apps or install packages silently",
        severity: Severity::Medium,
        mitre_id: "T1418",
    },
    SensitiveApi {
        pattern: "getInstalledPackages",
        name: "AppEnumeration",
        description: "Calls getInstalledPackages — enumerating installed \
                       applications for reconnaissance",
        severity: Severity::Medium,
        mitre_id: "T1418",
    },
    // Crypto operations
    SensitiveApi {
        pattern: "javax/crypto",
        name: "CryptoUsage",
        description: "References javax.crypto — may encrypt user data \
                       (ransomware) or C2 communications",
        severity: Severity::Medium,
        mitre_id: "T1573",
    },
    SensitiveApi {
        pattern: "SecretKeySpec",
        name: "SymmetricKeyUsage",
        description: "Uses SecretKeySpec — symmetric encryption key setup, \
                       possible ransomware or covert channel",
        severity: Severity::Medium,
        mitre_id: "T1573",
    },
    // Network access
    SensitiveApi {
        pattern: "HttpURLConnection",
        name: "NetworkHttp",
        description: "Uses HttpURLConnection — network communication that \
                       may contact C2 server",
        severity: Severity::Low,
        mitre_id: "T1071.001",
    },
    SensitiveApi {
        pattern: "OkHttpClient",
        name: "NetworkOkHttp",
        description: "Uses OkHttpClient — common HTTP library, may be used \
                       for C2 communication",
        severity: Severity::Low,
        mitre_id: "T1071.001",
    },
    // Reflection / dynamic loading
    SensitiveApi {
        pattern: "DexClassLoader",
        name: "DynamicDexLoading",
        description: "Uses DexClassLoader — dynamically loads DEX code at \
                       runtime, common in malware to evade detection",
        severity: Severity::High,
        mitre_id: "T1406",
    },
    SensitiveApi {
        pattern: "java/lang/reflect",
        name: "ReflectionUsage",
        description: "Uses Java reflection — may invoke hidden APIs or \
                       bypass security checks",
        severity: Severity::Medium,
        mitre_id: "T1620",
    },
    // Device info / fingerprinting
    SensitiveApi {
        pattern: "TelephonyManager",
        name: "TelephonyAccess",
        description: "References TelephonyManager — may collect IMEI, phone \
                       number, or SIM info for fingerprinting",
        severity: Severity::High,
        mitre_id: "T1422",
    },
    SensitiveApi {
        pattern: "getDeviceId",
        name: "DeviceIdAccess",
        description: "Calls getDeviceId — collecting device identifier for \
                       tracking / fingerprinting",
        severity: Severity::High,
        mitre_id: "T1422",
    },
    // Accessibility abuse
    SensitiveApi {
        pattern: "AccessibilityService",
        name: "AccessibilityAbuse",
        description: "References AccessibilityService — may overlay UI, \
                       keylog, or perform actions on behalf of user",
        severity: Severity::Critical,
        mitre_id: "T1629.001",
    },
    // Root detection evasion / su
    SensitiveApi {
        pattern: "/system/bin/su",
        name: "RootAccess",
        description: "References /system/bin/su — may attempt to gain root \
                       or detect rooted devices",
        severity: Severity::High,
        mitre_id: "T1404",
    },
];

/// Android permission strings that indicate sensitive capabilities.
const SENSITIVE_PERMISSIONS: &[&str] = &[
    "SEND_SMS",
    "READ_SMS",
    "RECEIVE_SMS",
    "READ_CONTACTS",
    "READ_PHONE_STATE",
    "CAMERA",
    "RECORD_AUDIO",
    "ACCESS_FINE_LOCATION",
    "READ_EXTERNAL_STORAGE",
    "WRITE_EXTERNAL_STORAGE",
    "INSTALL_PACKAGES",
    "SYSTEM_ALERT_WINDOW",
    "BIND_ACCESSIBILITY_SERVICE",
    "BIND_DEVICE_ADMIN",
    "READ_CALL_LOG",
    "PROCESS_OUTGOING_CALLS",
];

// ── DEX information structures ──────────────────────────────────────────────

/// Structured information extracted from a DEX binary.
#[derive(Debug, Clone)]
pub struct DexInfo {
    /// DEX version string (e.g. "035", "039").
    pub version: String,
    /// File size recorded in header.
    pub file_size: u32,
    /// SHA-1 signature from header (hex).
    pub signature: String,
    /// Number of string IDs.
    pub string_ids_count: u32,
    /// Number of type IDs.
    pub type_ids_count: u32,
    /// Number of method IDs.
    pub method_ids_count: u32,
    /// Number of class definitions.
    pub class_defs_count: u32,
    /// Extracted strings from the string table.
    pub strings: Vec<String>,
    /// Sensitive APIs detected in the string table.
    pub sensitive_apis: Vec<String>,
    /// Sensitive permissions found in strings.
    pub permissions: Vec<String>,
    /// Overall file entropy.
    pub file_entropy: f64,
    /// Suspicious indicators.
    pub suspicious: Vec<SuspiciousIndicator>,
}

/// DEX format parser.
pub struct DexParser;

impl FileParser for DexParser {
    fn parse(data: &[u8]) -> Result<ParsedFile> {
        let info = parse_dex(data)?;
        Ok(ParsedFile::DEX(info))
    }

    fn extract_features(data: &[u8]) -> Result<FeatureVector> {
        let info = parse_dex(data)?;
        Ok(build_feature_vector(&info, data))
    }
}

// ── Helper: read little-endian u32 ──────────────────────────────────────────

fn read_u32_le(data: &[u8], offset: usize) -> std::result::Result<u32, GuardianError> {
    if offset + 4 > data.len() {
        return Err(GuardianError::Parse(format!(
            "DEX: out of bounds reading u32 at offset {offset}"
        )));
    }
    Ok(u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]))
}

// ── ULEB128 decoder ─────────────────────────────────────────────────────────

/// Decode an unsigned LEB128 value, returning (value, bytes_consumed).
fn decode_uleb128(data: &[u8]) -> (u32, usize) {
    let mut result: u32 = 0;
    let mut shift = 0u32;
    let mut consumed = 0usize;
    for &byte in data.iter().take(5) {
        consumed += 1;
        result |= ((byte & 0x7F) as u32) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    (result, consumed)
}

// ── Core parsing logic ──────────────────────────────────────────────────────

fn parse_dex(data: &[u8]) -> Result<DexInfo> {
    // ── Validate magic ──────────────────────────────────────────────────
    if data.len() < DEX_HEADER_SIZE {
        return Err(GuardianError::Parse(
            "DEX: file too small for header".to_string(),
        ));
    }

    if &data[0..4] != DEX_MAGIC_PREFIX {
        return Err(GuardianError::Parse(
            "DEX: invalid magic bytes".to_string(),
        ));
    }

    // Version bytes at offset 4..7 (e.g. "035\0").
    let version = String::from_utf8_lossy(&data[4..7]).to_string();

    // Verify the null terminator at offset 7.
    if data[7] != 0 {
        return Err(GuardianError::Parse(
            "DEX: missing null terminator after version".to_string(),
        ));
    }

    // ── Parse header fields ─────────────────────────────────────────────
    // Signature (SHA-1) at offset 12..32 (20 bytes).
    let signature = hex::encode(&data[12..32]);

    let file_size = read_u32_le(data, 32)?;

    // Validate header size field (should be 0x70).
    let header_size = read_u32_le(data, 36)?;
    if header_size != DEX_HEADER_SIZE as u32 {
        return Err(GuardianError::Parse(format!(
            "DEX: unexpected header size {header_size:#x}, expected {DEX_HEADER_SIZE:#x}"
        )));
    }

    let string_ids_count = read_u32_le(data, 56)?;
    let string_ids_off = read_u32_le(data, 60)? as usize;
    let type_ids_count = read_u32_le(data, 64)?;
    let method_ids_count = read_u32_le(data, 88)?;
    let class_defs_count = read_u32_le(data, 96)?;

    // ── Extract string table ────────────────────────────────────────────
    let mut strings = Vec::with_capacity(string_ids_count.min(100_000) as usize);

    for i in 0..(string_ids_count.min(100_000) as usize) {
        let id_offset = string_ids_off + i * 4;
        if id_offset + 4 > data.len() {
            break;
        }
        let string_data_off = read_u32_le(data, id_offset)? as usize;
        if string_data_off >= data.len() {
            continue;
        }

        // The string_data_item starts with a ULEB128 size, then MUTF-8 data.
        let remaining = &data[string_data_off..];
        let (utf16_len, consumed) = decode_uleb128(remaining);
        if utf16_len > 0x10000 {
            // Sanity limit.
            continue;
        }

        let str_start = consumed;
        // Find null terminator.
        let str_bytes = &remaining[str_start..];
        let null_pos = str_bytes
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(str_bytes.len().min(4096));

        let s = String::from_utf8_lossy(&str_bytes[..null_pos]).to_string();
        if !s.is_empty() {
            strings.push(s);
        }
    }

    // ── Detect sensitive APIs ───────────────────────────────────────────
    let all_text: String = strings.join("\n");

    let sensitive_apis: Vec<String> = SENSITIVE_APIS
        .iter()
        .filter(|api| all_text.contains(api.pattern))
        .map(|api| api.pattern.to_string())
        .collect();

    let suspicious: Vec<SuspiciousIndicator> = SENSITIVE_APIS
        .iter()
        .filter(|api| all_text.contains(api.pattern))
        .map(|api| SuspiciousIndicator {
            name: api.name.to_string(),
            description: api.description.to_string(),
            severity: api.severity,
            mitre_id: Some(api.mitre_id.to_string()),
        })
        .collect();

    // ── Detect permissions ──────────────────────────────────────────────
    let permissions: Vec<String> = SENSITIVE_PERMISSIONS
        .iter()
        .filter(|perm| all_text.contains(*perm))
        .map(|perm| perm.to_string())
        .collect();

    // ── File entropy ────────────────────────────────────────────────────
    let file_entropy = shannon_entropy(data);

    Ok(DexInfo {
        version,
        file_size,
        signature,
        string_ids_count,
        type_ids_count,
        method_ids_count,
        class_defs_count,
        strings,
        sensitive_apis,
        permissions,
        file_entropy,
        suspicious,
    })
}

// ── Feature vector construction ─────────────────────────────────────────────

fn build_feature_vector(info: &DexInfo, data: &[u8]) -> FeatureVector {
    let mut numeric: Vec<f64> = Vec::with_capacity(16);

    numeric.push(info.file_size as f64);
    numeric.push(info.string_ids_count as f64);
    numeric.push(info.type_ids_count as f64);
    numeric.push(info.method_ids_count as f64);
    numeric.push(info.class_defs_count as f64);
    numeric.push(info.strings.len() as f64);
    numeric.push(info.sensitive_apis.len() as f64);
    numeric.push(info.permissions.len() as f64);
    numeric.push(info.file_entropy);
    numeric.push(data.len() as f64);

    // Ratio of sensitive APIs to total strings.
    let api_ratio = if info.strings.is_empty() {
        0.0
    } else {
        info.sensitive_apis.len() as f64 / info.strings.len() as f64
    };
    numeric.push(api_ratio);

    // Categorical: sensitive APIs, permissions.
    let mut categorical: Vec<String> = info.sensitive_apis.clone();
    categorical.extend(info.permissions.iter().cloned());

    // Metadata.
    let mut metadata = indicators_to_metadata(&info.suspicious);
    metadata.insert(
        "version".to_string(),
        serde_json::json!(info.version),
    );
    metadata.insert(
        "permissions".to_string(),
        serde_json::json!(info.permissions),
    );
    metadata.insert(
        "sensitive_apis".to_string(),
        serde_json::json!(info.sensitive_apis),
    );

    FeatureVector {
        numeric,
        categorical,
        metadata,
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid DEX 035 file with a small string table.
    fn minimal_dex(extra_strings: &[&str]) -> Vec<u8> {
        // We build a bare-minimum DEX file:
        //   - Header (0x70 bytes)
        //   - String ID table
        //   - String data items

        let num_strings = extra_strings.len() as u32;

        // Pre-compute string data items.
        let mut string_data: Vec<Vec<u8>> = Vec::new();
        for s in extra_strings {
            let mut item = Vec::new();
            // ULEB128-encode the UTF-16 length (for ASCII, same as byte len).
            let len = s.len() as u32;
            if len < 0x80 {
                item.push(len as u8);
            } else {
                item.push((len & 0x7F) as u8 | 0x80);
                item.push((len >> 7) as u8);
            }
            item.extend_from_slice(s.as_bytes());
            item.push(0); // null terminator
            string_data.push(item);
        }

        let string_ids_off = DEX_HEADER_SIZE;
        let string_ids_size = (num_strings * 4) as usize;
        let string_data_start = string_ids_off + string_ids_size;

        // Compute total file size.
        let total_string_data: usize = string_data.iter().map(|sd| sd.len()).sum();
        let file_size = string_data_start + total_string_data;

        let mut buf = vec![0u8; file_size];

        // ── Header ──────────────────────────────────────────────────────
        // Magic: "dex\n035\0"
        buf[0..4].copy_from_slice(b"dex\n");
        buf[4..8].copy_from_slice(b"035\0");

        // Checksum at 8..12 — skip (not validated here).
        // Signature at 12..32 — skip (not validated here).

        // file_size at 32
        buf[32..36].copy_from_slice(&(file_size as u32).to_le_bytes());
        // header_size at 36
        buf[36..40].copy_from_slice(&0x70u32.to_le_bytes());
        // endian_tag at 40 (little-endian = 0x12345678)
        buf[40..44].copy_from_slice(&0x12345678u32.to_le_bytes());

        // string_ids_size at 56
        buf[56..60].copy_from_slice(&num_strings.to_le_bytes());
        // string_ids_off at 60
        buf[60..64].copy_from_slice(&(string_ids_off as u32).to_le_bytes());

        // ── String ID table ─────────────────────────────────────────────
        let mut data_offset = string_data_start;
        for i in 0..(num_strings as usize) {
            let id_off = string_ids_off + i * 4;
            buf[id_off..id_off + 4]
                .copy_from_slice(&(data_offset as u32).to_le_bytes());
            data_offset += string_data[i].len();
        }

        // ── String data items ───────────────────────────────────────────
        let mut offset = string_data_start;
        for sd in &string_data {
            buf[offset..offset + sd.len()].copy_from_slice(sd);
            offset += sd.len();
        }

        buf
    }

    #[test]
    fn test_parse_minimal_dex() {
        let data = minimal_dex(&["Hello", "World"]);
        let result = DexParser::parse(&data);
        assert!(result.is_ok(), "parse failed: {:?}", result.err());
        if let Ok(ParsedFile::DEX(info)) = result {
            assert_eq!(info.version, "035");
            assert_eq!(info.string_ids_count, 2);
            assert!(info.strings.contains(&"Hello".to_string()));
            assert!(info.strings.contains(&"World".to_string()));
        } else {
            panic!("expected DEX variant");
        }
    }

    #[test]
    fn test_dex_sensitive_api_detection() {
        let data = minimal_dex(&[
            "Landroid/telephony/SmsManager;",
            "sendTextMessage",
            "Ljavax/crypto/Cipher;",
            "normalString",
        ]);
        let result = parse_dex(&data).unwrap();
        assert!(result.sensitive_apis.iter().any(|a| a == "SmsManager"));
        assert!(result
            .sensitive_apis
            .iter()
            .any(|a| a == "sendTextMessage"));
        assert!(result
            .sensitive_apis
            .iter()
            .any(|a| a == "javax/crypto"));
        assert!(!result
            .sensitive_apis
            .iter()
            .any(|a| a == "normalString"));
    }

    #[test]
    fn test_dex_permission_detection() {
        let data = minimal_dex(&[
            "android.permission.SEND_SMS",
            "android.permission.READ_CONTACTS",
            "android.permission.INTERNET",
        ]);
        let result = parse_dex(&data).unwrap();
        assert!(result.permissions.contains(&"SEND_SMS".to_string()));
        assert!(result.permissions.contains(&"READ_CONTACTS".to_string()));
        // INTERNET is not in our sensitive list.
        assert!(!result.permissions.contains(&"INTERNET".to_string()));
    }

    #[test]
    fn test_dex_invalid_magic() {
        let data = vec![0x00u8; 0x70];
        let result = DexParser::parse(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_dex_too_small() {
        let data = b"dex\n035\0";
        let result = DexParser::parse(data);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_features_dex() {
        let data = minimal_dex(&["SmsManager", "test"]);
        let features = DexParser::extract_features(&data);
        assert!(features.is_ok());
        let fv = features.unwrap();
        assert!(!fv.numeric.is_empty());
        // Sensitive API count should be at least 1.
        assert!(fv.numeric[6] >= 1.0);
    }

    #[test]
    fn test_uleb128_decode() {
        // Single byte: 0x05 = 5.
        assert_eq!(decode_uleb128(&[0x05]), (5, 1));
        // Two bytes: 0x80 0x01 = 128.
        assert_eq!(decode_uleb128(&[0x80, 0x01]), (128, 2));
        // 0xE5 0x8E 0x26 = 624485.
        assert_eq!(decode_uleb128(&[0xE5, 0x8E, 0x26]), (624485, 3));
    }

    #[test]
    fn test_sensitive_apis_wellformed() {
        for api in SENSITIVE_APIS {
            assert!(!api.pattern.is_empty());
            assert!(!api.name.is_empty());
            assert!(!api.mitre_id.is_empty());
        }
    }

    #[test]
    fn test_dex_accessibility_service_critical() {
        let data = minimal_dex(&["AccessibilityService"]);
        let result = parse_dex(&data).unwrap();
        assert!(result
            .suspicious
            .iter()
            .any(|s| s.name == "AccessibilityAbuse"
                && s.severity == Severity::Critical));
    }
}
