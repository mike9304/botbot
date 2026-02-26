//! Heuristic analysis engine for Home Guardian.
//!
//! Combines three analysis sub-engines — entropy calculation, suspicious string
//! scanning, and structural anomaly detection — into a single [`HeuristicEngine`]
//! that implements the [`DetectionLayer`](crate::DetectionLayer) trait.
//!
//! The engine produces a combined score from all sub-engines and classifies the
//! result as clean, suspicious, or malicious based on configurable thresholds
//! from [`HeuristicConfig`].

pub mod entropy;
pub mod strings;
pub mod structural;

use crate::DetectionLayer;
use guardian_common::{
    config::HeuristicConfig,
    Detection, DetectionEngine, FileType, GuardianError, Result, Severity,
};
use std::collections::HashMap;
use tracing::{debug, trace};

use entropy::EntropyClassification;
use strings::StringScanner;

/// Combined heuristic analysis engine.
///
/// Orchestrates entropy analysis, suspicious string scanning, and structural
/// anomaly detection. The combined score determines the severity level:
///
/// - `score <= clean_max` (default 15) → no detections
/// - `clean_max < score <= suspicious_max` (default 30) → `Severity::Medium`
/// - `score > suspicious_max` → `Severity::High`
pub struct HeuristicEngine {
    /// Configuration thresholds.
    config: HeuristicConfig,
    /// Pre-compiled string scanner (Aho-Corasick automaton).
    string_scanner: StringScanner,
}

impl HeuristicEngine {
    /// Create a new heuristic engine with the given configuration.
    pub fn new(config: HeuristicConfig) -> Self {
        debug!(
            clean_max = config.clean_max,
            suspicious_max = config.suspicious_max,
            entropy_packed = config.entropy_packed,
            entropy_suspicious = config.entropy_suspicious,
            "Initializing heuristic engine"
        );

        Self {
            config,
            string_scanner: StringScanner::new(),
        }
    }

    /// Create a heuristic engine with default thresholds.
    pub fn with_defaults() -> Self {
        Self::new(HeuristicConfig::default())
    }

    /// Run all heuristic sub-engines and return a detailed report.
    pub fn analyze(
        &self,
        data: &[u8],
        file_type: FileType,
    ) -> Result<HeuristicReport> {
        if data.is_empty() {
            return Err(GuardianError::Heuristic(
                "Cannot analyze empty data".to_string(),
            ));
        }

        // 1. Entropy analysis.
        let entropy_result = entropy::analyze_entropy(data, &self.config);
        trace!(
            overall_entropy = entropy_result.overall,
            entropy_score = entropy_result.score,
            classification = ?entropy_result.classification,
            "Entropy analysis complete"
        );

        // 2. String scanning.
        let string_result = self.string_scanner.scan(data);
        trace!(
            string_score = string_result.total_score,
            pattern_matches = string_result.matches.len(),
            "String scan complete"
        );

        // 3. Structural analysis.
        let structural_result = structural::analyze_structure(data, file_type);
        trace!(
            structural_score = structural_result.score,
            anomalies = structural_result.anomalies.len(),
            "Structural analysis complete"
        );

        // Combined score.
        let combined_score = entropy_result
            .score
            .saturating_add(string_result.total_score)
            .saturating_add(structural_result.score);

        debug!(
            entropy = entropy_result.score,
            strings = string_result.total_score,
            structural = structural_result.score,
            combined = combined_score,
            clean_max = self.config.clean_max,
            suspicious_max = self.config.suspicious_max,
            "Heuristic scoring complete"
        );

        Ok(HeuristicReport {
            combined_score,
            entropy_score: entropy_result.score,
            string_score: string_result.total_score,
            structural_score: structural_result.score,
            overall_entropy: entropy_result.overall,
            entropy_classification: entropy_result.classification,
            string_matches: string_result.matches.len(),
            structural_anomalies: structural_result.anomalies.len(),
            entropy_result,
            string_result,
            structural_result,
            config: self.config.clone(),
        })
    }
}

impl DetectionLayer for HeuristicEngine {
    fn name(&self) -> &str {
        "Heuristic"
    }

    fn engine_type(&self) -> DetectionEngine {
        DetectionEngine::Heuristic
    }

    fn scan_bytes(&self, data: &[u8], file_type: FileType) -> Result<Vec<Detection>> {
        let report = self.analyze(data, file_type)?;

        // If the combined score is below the clean threshold, return no detections.
        if report.combined_score <= self.config.clean_max {
            return Ok(Vec::new());
        }

        let mut detections = Vec::new();

        // Determine the overall severity for the combined detection.
        let overall_severity = if report.combined_score > self.config.suspicious_max {
            Severity::High
        } else {
            Severity::Medium
        };

        // Add the combined heuristic detection.
        let mut metadata = HashMap::new();
        metadata.insert(
            "combined_score".to_string(),
            report.combined_score.to_string(),
        );
        metadata.insert(
            "entropy_score".to_string(),
            report.entropy_score.to_string(),
        );
        metadata.insert(
            "string_score".to_string(),
            report.string_score.to_string(),
        );
        metadata.insert(
            "structural_score".to_string(),
            report.structural_score.to_string(),
        );
        metadata.insert(
            "overall_entropy".to_string(),
            format!("{:.3}", report.overall_entropy),
        );

        detections.push(Detection {
            engine: DetectionEngine::Heuristic,
            rule_name: "HEUR:Combined/Score".to_string(),
            description: format!(
                "Heuristic score {} (entropy={}, strings={}, structural={}) exceeds threshold {}",
                report.combined_score,
                report.entropy_score,
                report.string_score,
                report.structural_score,
                self.config.clean_max,
            ),
            severity: overall_severity,
            metadata,
        });

        // Add individual string detections for detail.
        detections.extend(self.string_scanner.to_detections(&report.string_result));

        // Add structural anomaly detections.
        detections.extend(structural::to_detections(&report.structural_result));

        // Add entropy detection if entropy itself was notable.
        if report.entropy_score >= 4 {
            let mut ent_meta = HashMap::new();
            ent_meta.insert(
                "entropy".to_string(),
                format!("{:.3}", report.overall_entropy),
            );
            ent_meta.insert(
                "classification".to_string(),
                format!("{:?}", report.entropy_classification),
            );

            detections.push(Detection {
                engine: DetectionEngine::Heuristic,
                rule_name: "HEUR:Entropy/High".to_string(),
                description: format!(
                    "File entropy {:.3} classified as {:?}",
                    report.overall_entropy, report.entropy_classification,
                ),
                severity: if report.entropy_classification
                    == EntropyClassification::PackedOrEncrypted
                {
                    Severity::Medium
                } else {
                    Severity::Low
                },
                metadata: ent_meta,
            });
        }

        Ok(detections)
    }
}

/// Detailed report from heuristic analysis.
#[derive(Debug, Clone)]
pub struct HeuristicReport {
    /// Combined score from all sub-engines.
    pub combined_score: u32,
    /// Score from entropy analysis (0-10).
    pub entropy_score: u32,
    /// Score from string scanning.
    pub string_score: u32,
    /// Score from structural analysis.
    pub structural_score: u32,
    /// Overall Shannon entropy of the file.
    pub overall_entropy: f64,
    /// Entropy classification.
    pub entropy_classification: EntropyClassification,
    /// Number of suspicious string matches.
    pub string_matches: usize,
    /// Number of structural anomalies.
    pub structural_anomalies: usize,

    /// Full entropy result.
    pub entropy_result: entropy::EntropyResult,
    /// Full string scan result.
    pub string_result: strings::StringScanResult,
    /// Full structural result.
    pub structural_result: structural::StructuralResult,

    /// Configuration used for this analysis.
    pub config: HeuristicConfig,
}

impl HeuristicReport {
    /// Returns `true` if the combined score indicates the file is clean.
    pub fn is_clean(&self) -> bool {
        self.combined_score <= self.config.clean_max
    }

    /// Returns `true` if the combined score indicates the file is suspicious.
    pub fn is_suspicious(&self) -> bool {
        self.combined_score > self.config.clean_max
            && self.combined_score <= self.config.suspicious_max
    }

    /// Returns `true` if the combined score indicates the file is malicious.
    pub fn is_malicious(&self) -> bool {
        self.combined_score > self.config.suspicious_max
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> HeuristicEngine {
        HeuristicEngine::with_defaults()
    }

    #[test]
    fn test_engine_creation() {
        let e = engine();
        assert_eq!(e.name(), "Heuristic");
        assert_eq!(e.engine_type(), DetectionEngine::Heuristic);
    }

    #[test]
    fn test_scan_clean_file() {
        let e = engine();
        let data = b"This is a perfectly normal text file with nothing suspicious.";
        let detections = e.scan_bytes(data, FileType::Unknown).unwrap();
        assert!(
            detections.is_empty(),
            "Clean file should produce no detections, got {} detections",
            detections.len()
        );
    }

    #[test]
    fn test_scan_empty_file_error() {
        let e = engine();
        let result = e.scan_bytes(b"", FileType::Unknown);
        assert!(result.is_err());
    }

    #[test]
    fn test_analyze_clean_file() {
        let e = engine();
        let data = b"Normal file content with no suspicious patterns.";
        let report = e.analyze(data, FileType::Unknown).unwrap();

        assert!(report.is_clean());
        assert!(!report.is_suspicious());
        assert!(!report.is_malicious());
        assert_eq!(report.string_matches, 0);
        assert_eq!(report.structural_anomalies, 0);
    }

    #[test]
    fn test_analyze_suspicious_strings() {
        let e = engine();
        let data = b"CreateRemoteThread VirtualAllocEx WriteProcessMemory \
                     IsDebuggerPresent mimikatz sekurlsa";
        let report = e.analyze(data, FileType::Unknown).unwrap();

        assert!(
            report.combined_score > 15,
            "Score {} should exceed clean_max 15",
            report.combined_score
        );
        assert!(report.string_score > 0);
        assert!(report.string_matches >= 5);
    }

    #[test]
    fn test_scan_bytes_produces_detections_for_malicious() {
        let e = engine();
        // Load up enough suspicious strings to exceed suspicious_max (30).
        let data = b"CreateRemoteThread VirtualAllocEx WriteProcessMemory \
                     NtUnmapViewOfSection mimikatz sekurlsa \
                     -EncodedCommand -nop -w hidden nc -e /bin/sh \
                     /etc/shadow IsDebuggerPresent CheckRemoteDebuggerPresent \
                     Invoke-Mimikatz";
        let detections = e.scan_bytes(data, FileType::Unknown).unwrap();

        assert!(
            !detections.is_empty(),
            "Should produce detections for clearly malicious content"
        );

        // Should have the combined detection.
        let has_combined = detections
            .iter()
            .any(|d| d.rule_name == "HEUR:Combined/Score");
        assert!(has_combined, "Should include combined score detection");

        // Should have individual string detections.
        let string_detections: Vec<_> = detections
            .iter()
            .filter(|d| d.rule_name.starts_with("HEUR:String/"))
            .collect();
        assert!(
            !string_detections.is_empty(),
            "Should include individual string detections"
        );
    }

    #[test]
    fn test_scan_bytes_with_pe_structure() {
        let e = engine();

        // Build a PE with suspicious strings embedded.
        let mut pe = vec![0u8; 600];
        pe[0] = 0x4D;
        pe[1] = 0x5A;
        pe[0x3C] = 0x80;

        pe[0x80] = b'P';
        pe[0x81] = b'E';

        // Embed enough suspicious content.
        let suspicious = b"CreateRemoteThread VirtualAllocEx mimikatz sekurlsa \
                           -EncodedCommand -nop -w hidden";
        let start = 0x100;
        pe[start..start + suspicious.len()].copy_from_slice(suspicious);

        let detections = e.scan_bytes(&pe, FileType::PE).unwrap();

        // Should have at least some detections from string scanning.
        assert!(
            !detections.is_empty(),
            "PE with suspicious strings should produce detections"
        );
    }

    #[test]
    fn test_report_classification_thresholds() {
        let e = engine();

        // Score just at clean_max.
        let data = b"Some normal data with just a curl command: curl http://example.com";
        let report = e.analyze(data, FileType::Unknown).unwrap();
        // curl has weight 4, so score = 4 which is under 15.
        assert!(report.is_clean(), "Score {} should be clean", report.combined_score);
    }

    #[test]
    fn test_custom_config_thresholds() {
        // Very strict config: anything above 5 is suspicious, above 10 is malicious.
        let config = HeuristicConfig {
            clean_max: 5,
            suspicious_max: 10,
            entropy_packed: 7.5,
            entropy_suspicious: 7.0,
        };
        let e = HeuristicEngine::new(config);

        // This should exceed even the strict threshold.
        let data = b"CreateRemoteThread IsDebuggerPresent";
        let detections = e.scan_bytes(data, FileType::Unknown).unwrap();
        assert!(
            !detections.is_empty(),
            "Strict thresholds should flag suspicious API calls"
        );
    }

    #[test]
    fn test_all_detection_engines_are_heuristic() {
        let e = engine();
        let data = b"CreateRemoteThread VirtualAllocEx WriteProcessMemory \
                     mimikatz sekurlsa -EncodedCommand -nop -w hidden";
        let detections = e.scan_bytes(data, FileType::Unknown).unwrap();

        for d in &detections {
            assert_eq!(
                d.engine,
                DetectionEngine::Heuristic,
                "All detections must be from Heuristic engine"
            );
        }
    }

    #[test]
    fn test_entropy_detection_included_for_high_entropy() {
        let e = engine();

        // Create data with both high entropy and suspicious strings.
        let mut data: Vec<u8> = (0..=255).cycle().take(8192).collect();
        // Embed strings to push past threshold.
        let suspicious = b"mimikatz sekurlsa CreateRemoteThread";
        data[0..suspicious.len()].copy_from_slice(suspicious);

        let detections = e.scan_bytes(&data, FileType::Unknown).unwrap();

        if !detections.is_empty() {
            // If we have detections, check that entropy detection might be present.
            let has_entropy_det = detections
                .iter()
                .any(|d| d.rule_name.starts_with("HEUR:Entropy/"));

            // The high-entropy data should contribute to scoring.
            let report = e.analyze(&data, FileType::Unknown).unwrap();
            if report.entropy_score >= 4 && report.combined_score > e.config.clean_max {
                assert!(
                    has_entropy_det,
                    "Should include entropy detection when score >= 4"
                );
            }
        }
    }

    #[test]
    fn test_heuristic_report_scores_add_up() {
        let e = engine();
        let data = b"CreateRemoteThread VirtualAllocEx mimikatz";
        let report = e.analyze(data, FileType::Unknown).unwrap();

        let expected_combined =
            report.entropy_score + report.string_score + report.structural_score;
        assert_eq!(
            report.combined_score, expected_combined,
            "Combined score should equal sum of sub-scores"
        );
    }

    #[test]
    fn test_scan_file_default_implementation() {
        // Test that the trait object can be constructed.
        let e = engine();
        let _: &dyn DetectionLayer = &e;
    }
}
