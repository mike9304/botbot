//! Machine Learning classification engine.
//!
//! This module provides EMBER-style feature extraction and ONNX-based inference
//! (with a deterministic fallback) to classify files as benign, suspicious, or
//! malicious. It implements [`DetectionLayer`] so the ML engine plugs directly
//! into the Guardian scanner pipeline.

pub mod classifier;
pub mod features;

use crate::DetectionLayer;
use classifier::{Classifier, PredictionLabel};
use features::extract_features;
use guardian_common::{Detection, DetectionEngine, FileType, GuardianError, Result, Severity};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::{debug, info};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Thresholds for the ML engine.
#[derive(Debug, Clone)]
pub struct MlConfig {
    /// Score at or above which a file is considered suspicious.
    pub suspicious_threshold: f64,
    /// Score at or above which a file is considered malicious.
    pub malicious_threshold: f64,
    /// Minimum confidence to emit a detection at all.
    pub min_confidence: f64,
    /// Path to the ONNX model file.
    pub model_path: PathBuf,
}

impl Default for MlConfig {
    fn default() -> Self {
        Self {
            suspicious_threshold: 0.4,
            malicious_threshold: 0.7,
            min_confidence: 0.3,
            model_path: PathBuf::from("data/models/guardian_ml.onnx"),
        }
    }
}

// ---------------------------------------------------------------------------
// MlClassifier — implements DetectionLayer
// ---------------------------------------------------------------------------

/// ML-based detection layer.
///
/// Thread-safe: the inner [`Classifier`] uses an `Arc<Session>` for ONNX and
/// the fallback path is stateless.
pub struct MlClassifier {
    config: MlConfig,
    classifier: Classifier,
}

impl MlClassifier {
    /// Create a new ML classifier. Attempts to load the ONNX model; falls back
    /// to threshold-based heuristics if loading fails.
    pub fn new(config: MlConfig) -> Self {
        let classifier = if config.model_path.exists() {
            Classifier::new(&config.model_path)
        } else {
            Classifier::fallback()
        };
        info!(
            onnx = classifier.is_onnx(),
            model_path = %config.model_path.display(),
            "ML classifier initialised"
        );
        Self { config, classifier }
    }

    /// Create a classifier with the default config, attempting to load a model
    /// from the given path.
    pub fn from_model_path(path: &Path) -> Self {
        let config = MlConfig {
            model_path: path.to_path_buf(),
            ..Default::default()
        };
        Self::new(config)
    }

    /// Create a fallback-only classifier (useful for testing and when no model
    /// is shipped).
    pub fn fallback() -> Self {
        Self {
            config: MlConfig::default(),
            classifier: Classifier::fallback(),
        }
    }

    /// Whether the ONNX backend is active.
    pub fn is_onnx(&self) -> bool {
        self.classifier.is_onnx()
    }
}

impl DetectionLayer for MlClassifier {
    fn name(&self) -> &str {
        "ML Classifier"
    }

    fn engine_type(&self) -> DetectionEngine {
        DetectionEngine::MachineLearning
    }

    fn scan_bytes(&self, data: &[u8], file_type: FileType) -> Result<Vec<Detection>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        // 1. Feature extraction
        let features = extract_features(data, file_type);
        debug!(
            dimensions = features.numeric.len(),
            file_type = ?file_type,
            "Extracted ML features"
        );

        // 2. Inference
        let prediction = self
            .classifier
            .predict(&features)
            .map_err(|e| GuardianError::Ml(format!("Prediction failed: {e}")))?;

        debug!(
            score = prediction.score,
            label = %prediction.label,
            confidence = prediction.confidence,
            "ML prediction"
        );

        // 3. Thresholding
        if prediction.confidence < self.config.min_confidence {
            return Ok(Vec::new());
        }

        let mut detections = Vec::new();

        match prediction.label {
            PredictionLabel::Malicious if prediction.score >= self.config.malicious_threshold => {
                let mut metadata = HashMap::new();
                metadata.insert("ml_score".to_string(), format!("{:.4}", prediction.score));
                metadata.insert(
                    "ml_confidence".to_string(),
                    format!("{:.4}", prediction.confidence),
                );
                metadata.insert("ml_backend".to_string(), backend_name(&self.classifier));
                metadata.insert(
                    "top_features".to_string(),
                    format!("{:?}", prediction.top_features),
                );

                detections.push(Detection {
                    engine: DetectionEngine::MachineLearning,
                    rule_name: "ML/Malicious".to_string(),
                    description: format!(
                        "ML classifier detected malicious content (score: {:.2}%, confidence: {:.2}%)",
                        prediction.score * 100.0,
                        prediction.confidence * 100.0,
                    ),
                    severity: Severity::High,
                    metadata,
                });
            }
            PredictionLabel::Suspicious
                if prediction.score >= self.config.suspicious_threshold =>
            {
                let mut metadata = HashMap::new();
                metadata.insert("ml_score".to_string(), format!("{:.4}", prediction.score));
                metadata.insert(
                    "ml_confidence".to_string(),
                    format!("{:.4}", prediction.confidence),
                );
                metadata.insert("ml_backend".to_string(), backend_name(&self.classifier));

                detections.push(Detection {
                    engine: DetectionEngine::MachineLearning,
                    rule_name: "ML/Suspicious".to_string(),
                    description: format!(
                        "ML classifier flagged suspicious content (score: {:.2}%, confidence: {:.2}%)",
                        prediction.score * 100.0,
                        prediction.confidence * 100.0,
                    ),
                    severity: Severity::Medium,
                    metadata,
                });
            }
            _ => {
                // Benign or below threshold — no detection.
            }
        }

        Ok(detections)
    }
}

fn backend_name(classifier: &Classifier) -> String {
    if classifier.is_onnx() {
        "onnx".to_string()
    } else {
        "fallback".to_string()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ml_classifier_name() {
        let clf = MlClassifier::fallback();
        assert_eq!(clf.name(), "ML Classifier");
        assert_eq!(clf.engine_type(), DetectionEngine::MachineLearning);
    }

    #[test]
    fn test_ml_classifier_empty_data() {
        let clf = MlClassifier::fallback();
        let result = clf.scan_bytes(&[], FileType::Unknown).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_ml_classifier_benign_data() {
        let clf = MlClassifier::fallback();
        let data = b"Hello, this is a perfectly normal text file with nothing suspicious.";
        let result = clf.scan_bytes(data, FileType::Unknown).unwrap();
        // Benign data should produce no detections
        assert!(result.is_empty());
    }

    #[test]
    fn test_ml_classifier_scan_file_nonexistent() {
        let clf = MlClassifier::fallback();
        let result = clf.scan_file(Path::new("/nonexistent/file.exe"));
        assert!(result.is_err());
    }

    #[test]
    fn test_ml_config_default() {
        let cfg = MlConfig::default();
        assert!((cfg.suspicious_threshold - 0.4).abs() < f64::EPSILON);
        assert!((cfg.malicious_threshold - 0.7).abs() < f64::EPSILON);
        assert!((cfg.min_confidence - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ml_classifier_from_model_path() {
        let clf = MlClassifier::from_model_path(Path::new("/nonexistent/model.onnx"));
        assert!(!clf.is_onnx());
    }

    #[test]
    fn test_ml_classifier_detection_layer_trait() {
        // Verify MlClassifier can be used as a trait object.
        let clf = MlClassifier::fallback();
        let layer: &dyn DetectionLayer = &clf;
        assert_eq!(layer.name(), "ML Classifier");

        let result = layer.scan_bytes(b"test data", FileType::Unknown).unwrap();
        assert!(result.is_empty()); // benign
    }

    #[test]
    fn test_backend_name() {
        let clf = Classifier::fallback();
        assert_eq!(backend_name(&clf), "fallback");
    }

    #[test]
    fn test_ml_classifier_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MlClassifier>();
    }
}
