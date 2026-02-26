//! ONNX-based ML classifier with graceful fallback.
//!
//! Wraps the `ort` crate for ONNX Runtime inference. When the ONNX model cannot be
//! loaded (missing file, unsupported platform, dynamic library unavailable) the
//! classifier transparently falls back to a deterministic threshold-based heuristic
//! that operates on the same [`FeatureVector`].

use guardian_common::{FeatureVector, GuardianError, Result};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use std::sync::Mutex;
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Prediction produced by the classifier.
#[derive(Debug, Clone)]
pub struct Prediction {
    /// Maliciousness score in [0.0, 1.0].
    pub score: f64,
    /// Classification label.
    pub label: PredictionLabel,
    /// Confidence of the prediction in [0.0, 1.0].
    pub confidence: f64,
    /// Indices of the top contributing features (by absolute value).
    pub top_features: Vec<usize>,
}

/// Predicted classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictionLabel {
    Benign,
    Suspicious,
    Malicious,
}

impl std::fmt::Display for PredictionLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Benign => write!(f, "benign"),
            Self::Suspicious => write!(f, "suspicious"),
            Self::Malicious => write!(f, "malicious"),
        }
    }
}

// ---------------------------------------------------------------------------
// Classifier
// ---------------------------------------------------------------------------

/// ML classifier that uses ONNX inference or falls back to thresholds.
pub struct Classifier {
    inner: ClassifierBackend,
}

// Manual Send+Sync: Session is behind a Mutex, Fallback is stateless.
unsafe impl Send for Classifier {}
unsafe impl Sync for Classifier {}

enum ClassifierBackend {
    /// ONNX Runtime session (thread-safe via Mutex since `run()` requires `&mut self`).
    #[allow(dead_code)]
    Onnx(Mutex<Session>),
    /// Deterministic fallback when ONNX is unavailable.
    Fallback,
}

impl Classifier {
    /// Try to load an ONNX model from `path`.
    ///
    /// If loading fails for any reason, the classifier silently falls back to the
    /// threshold-based heuristic and logs a warning.
    pub fn new(model_path: &Path) -> Self {
        match Self::try_load_onnx(model_path) {
            Ok(session) => {
                info!(path = %model_path.display(), "ONNX model loaded successfully");
                Self {
                    inner: ClassifierBackend::Onnx(Mutex::new(session)),
                }
            }
            Err(e) => {
                warn!(
                    path = %model_path.display(),
                    error = %e,
                    "Failed to load ONNX model, using fallback classifier"
                );
                Self {
                    inner: ClassifierBackend::Fallback,
                }
            }
        }
    }

    /// Create a fallback-only classifier (no model loading attempted).
    pub fn fallback() -> Self {
        info!("Initialising fallback ML classifier (no ONNX model)");
        Self {
            inner: ClassifierBackend::Fallback,
        }
    }

    /// Run inference on a feature vector.
    pub fn predict(&self, features: &FeatureVector) -> Result<Prediction> {
        match &self.inner {
            ClassifierBackend::Onnx(session_mutex) => {
                self.predict_onnx(session_mutex, features)
            }
            ClassifierBackend::Fallback => Ok(self.predict_fallback(features)),
        }
    }

    /// Whether this classifier is using the ONNX backend.
    pub fn is_onnx(&self) -> bool {
        matches!(self.inner, ClassifierBackend::Onnx(_))
    }

    // -----------------------------------------------------------------------
    // ONNX inference
    // -----------------------------------------------------------------------

    fn try_load_onnx(
        model_path: &Path,
    ) -> std::result::Result<Session, Box<dyn std::error::Error>> {
        // ort with `load-dynamic` may panic if the ONNX Runtime shared library
        // is not found. Catch that so we fall back gracefully.
        let path = model_path.to_path_buf();
        let result = std::panic::catch_unwind(move || {
            let session = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(1)?
                .commit_from_file(&path)?;
            Ok(session)
        });
        match result {
            Ok(inner) => inner,
            Err(_) => Err("ONNX Runtime library not available (panic during init)".into()),
        }
    }

    fn predict_onnx(
        &self,
        session_mutex: &Mutex<Session>,
        features: &FeatureVector,
    ) -> Result<Prediction> {
        let input_data: Vec<f32> = features.numeric.iter().map(|&v| v as f32).collect();
        let dim = input_data.len();

        // Create input tensor with shape [1, dim] using the (shape, data) tuple API.
        let input_tensor = Tensor::from_array((
            vec![1i64, dim as i64],
            input_data.into_boxed_slice(),
        ))
        .map_err(|e| GuardianError::Ml(format!("Failed to create ONNX tensor: {e}")))?;

        let mut session = session_mutex
            .lock()
            .map_err(|e| GuardianError::Ml(format!("Session lock poisoned: {e}")))?;

        let outputs = session
            .run(ort::inputs![input_tensor])
            .map_err(|e| GuardianError::Ml(format!("ONNX inference failed: {e}")))?;

        // Extract output -- expect a single float output (maliciousness probability).
        if outputs.len() == 0 {
            return Err(GuardianError::Ml("No output from ONNX model".into()));
        }

        let output = &outputs[0];
        let score = extract_score_from_value(output).unwrap_or(0.5);

        let label = score_to_label(score);
        let confidence = if score > 0.5 { score } else { 1.0 - score };
        let top_features = top_feature_indices(&features.numeric, 10);

        Ok(Prediction {
            score,
            label,
            confidence,
            top_features,
        })
    }

    // -----------------------------------------------------------------------
    // Fallback classifier
    // -----------------------------------------------------------------------

    /// Deterministic heuristic classifier based on basic feature statistics.
    fn predict_fallback(&self, features: &FeatureVector) -> Prediction {
        debug!("Running fallback ML classifier");
        let numeric = &features.numeric;
        let mut score = 0.0f64;
        let mut signals = 0u32;

        // --- Signal 1: Overall entropy (feature index 617 in standard layout) ---
        // Index layout: 0..256 = byte_hist, 256..512 = byte_ent_hist, 512..616 = str, 616..626 = gen
        // General info starts at 512+104 = 616; index 617 = overall entropy.
        let entropy_idx = 256 + 256 + 104 + 1; // 617
        if let Some(&entropy) = numeric.get(entropy_idx) {
            if entropy > 7.5 {
                score += 0.3; // Likely packed / encrypted
                signals += 1;
            } else if entropy > 7.0 {
                score += 0.15;
                signals += 1;
            }
        }

        // --- Signal 2: Suspicious string categories ---
        // String features start at index 512. Keyword hits at indices 512+14..512+34.
        let str_base = 256 + 256; // 512
        // Process-related keyword count (group index 3, offset 14 + 3*2 = 20)
        if let Some(&process_hits) = numeric.get(str_base + 20) {
            if process_hits > 3.0 {
                score += 0.15;
                signals += 1;
            }
        }
        // Crypto keyword count at str_base + 14 + 8
        if let Some(&crypto_hits) = numeric.get(str_base + 22) {
            if crypto_hits > 2.0 {
                score += 0.10;
                signals += 1;
            }
        }
        // Shell keyword count at str_base + 14 + 16
        if let Some(&shell_hits) = numeric.get(str_base + 30) {
            if shell_hits > 1.0 {
                score += 0.10;
                signals += 1;
            }
        }
        // Persistence keywords at str_base + 14 + 18
        if let Some(&persist_hits) = numeric.get(str_base + 32) {
            if persist_hits > 0.0 {
                score += 0.15;
                signals += 1;
            }
        }

        // --- Signal 3: Byte histogram anomalies ---
        // High proportion of null bytes (index 0)
        if let Some(&null_ratio) = numeric.first() {
            if null_ratio > 0.5 {
                score += 0.05;
                signals += 1;
            }
        }

        // --- Signal 4: PE-specific features (if present) ---
        // PE features start after general info. For PE files, total non-PE = 626.
        let pe_base = 626;
        // Entry point outside sections (index pe_base + 43)
        if let Some(&ep_anomaly) = numeric.get(pe_base + 43) {
            if ep_anomaly > 0.5 {
                score += 0.20;
                signals += 1;
            }
        }

        // Writable + executable section anomaly in first section (pe_base + 62 + 13)
        let section_base = pe_base + 62;
        if let Some(&wx) = numeric.get(section_base + 13) {
            if wx > 0.5 {
                score += 0.15;
                signals += 1;
            }
        }

        score = score.clamp(0.0, 1.0);
        let confidence = if signals > 0 {
            (0.3 + 0.1 * signals as f64).min(0.85)
        } else {
            0.75 // Confident it's benign when no signals fire
        };

        let label = score_to_label(score);
        let top_features = top_feature_indices(numeric, 10);

        Prediction {
            score,
            label,
            confidence,
            top_features,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn score_to_label(score: f64) -> PredictionLabel {
    if score >= 0.7 {
        PredictionLabel::Malicious
    } else if score >= 0.4 {
        PredictionLabel::Suspicious
    } else {
        PredictionLabel::Benign
    }
}

/// Return indices of the top-N features by absolute value.
fn top_feature_indices(features: &[f64], n: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f64)> = features
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v.abs()))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.into_iter().take(n).map(|(i, _)| i).collect()
}

/// Try to extract a single f64 score from an ONNX DynValue.
fn extract_score_from_value(value: &ort::value::DynValue) -> Option<f64> {
    // Try f32 tensor first (most common for ML models).
    // try_extract_tensor returns (&Shape, &[T]) -- data is in .1
    if let Ok(tensor) = value.try_extract_tensor::<f32>() {
        return tensor.1.iter().next().map(|&v| v as f64);
    }
    // Then try f64.
    if let Ok(tensor) = value.try_extract_tensor::<f64>() {
        return tensor.1.iter().next().copied();
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use guardian_common::FeatureVector;
    use std::collections::HashMap;

    fn make_feature_vector(n: usize) -> FeatureVector {
        FeatureVector {
            numeric: vec![0.0; n],
            categorical: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_fallback_classifier_benign() {
        let clf = Classifier::fallback();
        assert!(!clf.is_onnx());

        let fv = make_feature_vector(626);
        let pred = clf.predict(&fv).unwrap();
        assert_eq!(pred.label, PredictionLabel::Benign);
        assert!(pred.score < 0.4);
    }

    #[test]
    fn test_fallback_classifier_high_entropy() {
        let clf = Classifier::fallback();
        let mut fv = make_feature_vector(626);
        // Set overall entropy (index 617) to 7.8
        fv.numeric[617] = 7.8;
        let pred = clf.predict(&fv).unwrap();
        assert!(pred.score > 0.0);
    }

    #[test]
    fn test_fallback_classifier_pe_suspicious() {
        let clf = Classifier::fallback();
        // Create a PE-sized feature vector with anomalies
        let mut fv = make_feature_vector(2381);
        // High entropy
        fv.numeric[617] = 7.6;
        // Entry point anomaly
        fv.numeric[626 + 43] = 1.0;
        // Writable+executable section
        fv.numeric[626 + 62 + 13] = 1.0;
        // Persistence keywords
        fv.numeric[512 + 32] = 3.0;

        let pred = clf.predict(&fv).unwrap();
        assert!(pred.score >= 0.4);
        assert!(
            pred.label == PredictionLabel::Suspicious
                || pred.label == PredictionLabel::Malicious
        );
    }

    #[test]
    fn test_score_to_label() {
        assert_eq!(score_to_label(0.0), PredictionLabel::Benign);
        assert_eq!(score_to_label(0.39), PredictionLabel::Benign);
        assert_eq!(score_to_label(0.5), PredictionLabel::Suspicious);
        assert_eq!(score_to_label(0.7), PredictionLabel::Malicious);
        assert_eq!(score_to_label(1.0), PredictionLabel::Malicious);
    }

    #[test]
    fn test_top_feature_indices() {
        let features = vec![0.1, 0.5, 0.3, 0.9, 0.0];
        let top = top_feature_indices(&features, 3);
        assert_eq!(top[0], 3); // 0.9
        assert_eq!(top[1], 1); // 0.5
        assert_eq!(top[2], 2); // 0.3
    }

    #[test]
    fn test_prediction_label_display() {
        assert_eq!(format!("{}", PredictionLabel::Benign), "benign");
        assert_eq!(format!("{}", PredictionLabel::Suspicious), "suspicious");
        assert_eq!(format!("{}", PredictionLabel::Malicious), "malicious");
    }

    #[test]
    fn test_classifier_from_nonexistent_model() {
        let clf = Classifier::new(Path::new("/nonexistent/model.onnx"));
        // Should gracefully fall back
        assert!(!clf.is_onnx());
        let fv = make_feature_vector(626);
        let pred = clf.predict(&fv).unwrap();
        assert_eq!(pred.label, PredictionLabel::Benign);
    }
}
