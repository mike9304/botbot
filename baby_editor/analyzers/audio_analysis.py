"""Audio analyzers: silence, laughter, voice/transcription.

Implements DEL-2 (silence), ADD-2 (baby laughter), ADD-3 (human voice).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SilenceMetrics:
    """Raw metrics from silence analysis."""
    rms_db: float                     # average RMS in dB
    silent_window_pct: float          # % of 100ms windows below threshold
    per_window_db: list[float] = field(default_factory=list)


@dataclass
class LaughterMetrics:
    """Raw metrics from laughter detection."""
    laugh_confidence: float           # max confidence for laughter
    laugh_class: str                  # matched YAMNet class name
    laugh_windows: int                # number of windows with laughter
    total_windows: int
    cry_detected: bool = False        # for cry→laugh transition tracking


@dataclass
class VoiceMetrics:
    """Raw metrics from voice/speech analysis."""
    voice_activity_pct: float         # % of segment with detected voice
    transcript: str                   # transcribed text
    has_exclamation: bool
    has_baby_name: bool
    is_babble: bool
    whisper_confidence: float         # average segment confidence


class SilenceAnalyzer:
    """DEL-2: Silence detection via RMS energy.

    Computes RMS volume in dB across 100ms windows and checks
    what percentage falls below the silence threshold.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        sil_cfg = config.get("delete_rules", {}).get("silence", {})
        self.threshold_db = sil_cfg.get("threshold_db", -45)
        self.near_silence_pct = sil_cfg.get("near_silence_pct", 0.6)
        self.rms_window_ms = sil_cfg.get("rms_window_ms", 100)

    def analyze(self, audio: np.ndarray, sr: int) -> SilenceMetrics:
        """Analyze silence from an audio segment.

        Args:
            audio: Audio samples for this segment (mono, float).
            sr: Sample rate.

        Returns:
            SilenceMetrics with RMS dB and silence percentage.
        """
        if len(audio) == 0:
            return SilenceMetrics(rms_db=-100.0, silent_window_pct=1.0)

        window_samples = int(sr * self.rms_window_ms / 1000)
        if window_samples <= 0:
            window_samples = sr // 10

        per_window_db = []
        silent_count = 0
        total_windows = 0

        for start in range(0, len(audio), window_samples):
            end = min(start + window_samples, len(audio))
            window = audio[start:end]
            if len(window) == 0:
                continue

            rms = float(np.sqrt(np.mean(window ** 2)))
            db = 20 * np.log10(rms + 1e-10)
            per_window_db.append(round(db, 1))
            total_windows += 1

            if db < self.threshold_db:
                silent_count += 1

        if total_windows == 0:
            return SilenceMetrics(rms_db=-100.0, silent_window_pct=1.0)

        avg_db = float(np.mean(per_window_db))
        silent_pct = silent_count / total_windows

        return SilenceMetrics(
            rms_db=round(avg_db, 1),
            silent_window_pct=round(silent_pct, 3),
            per_window_db=per_window_db,
        )


class LaughterAnalyzer:
    """ADD-2: Baby laughter detection via YAMNet audio classification.

    Uses TensorFlow Hub YAMNet model to classify audio segments,
    looking for laughter-related classes.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        laugh_cfg = config.get("addition_rules", {}).get("laugh", {})
        self.min_confidence = laugh_cfg.get("min_confidence", 0.7)
        self.yamnet_window = laugh_cfg.get("yamnet_window", 0.96)
        self.yamnet_hop = laugh_cfg.get("yamnet_hop", 0.48)
        self.laughter_classes = set(laugh_cfg.get("laughter_classes", [
            "Laughter", "Baby laughter", "Giggle",
        ]))
        self.crying_classes = set(laugh_cfg.get("crying_classes", [
            "Crying, sobbing", "Baby cry, infant cry",
        ]))
        self._model = None
        self._class_names = None

    def _load_model(self) -> bool:
        """Lazy-load YAMNet model on first use."""
        if self._model is not None:
            return True
        try:
            import tensorflow_hub as hub
            import tensorflow as tf
            self._model = hub.load("https://tfhub.dev/google/yamnet/1")
            # Load class names
            class_map_path = self._model.class_map_path().numpy().decode("utf-8")
            with open(class_map_path) as f:
                lines = f.readlines()
            # CSV: index, mid, display_name
            self._class_names = [
                line.strip().split(",")[-1].strip('"')
                for line in lines[1:]  # skip header
            ]
            logger.info(f"YAMNet loaded: {len(self._class_names)} classes")
            return True
        except ImportError:
            logger.warning(
                "tensorflow/tensorflow_hub not installed. "
                "Laughter detection will be disabled."
            )
            return False
        except Exception as e:
            logger.error(f"Failed to load YAMNet: {e}")
            return False

    def analyze(self, audio: np.ndarray, sr: int) -> LaughterMetrics:
        """Analyze audio for baby laughter.

        Args:
            audio: Audio samples (mono, float, 16kHz).
            sr: Sample rate.

        Returns:
            LaughterMetrics with laughter confidence and class info.
        """
        if len(audio) == 0 or not self._load_model():
            return LaughterMetrics(
                laugh_confidence=0.0, laugh_class="",
                laugh_windows=0, total_windows=0,
            )

        try:
            import tensorflow as tf

            # YAMNet expects float32 waveform at 16kHz
            waveform = tf.cast(audio, tf.float32)
            scores, embeddings, spectrogram = self._model(waveform)
            scores_np = scores.numpy()

            max_laugh_conf = 0.0
            max_laugh_class = ""
            laugh_windows = 0
            cry_detected = False

            for window_scores in scores_np:
                top_idx = int(np.argmax(window_scores))
                top_class = self._class_names[top_idx] if top_idx < len(self._class_names) else ""
                top_score = float(window_scores[top_idx])

                # Check all laughter classes
                for cls_name in self.laughter_classes:
                    try:
                        cls_idx = self._class_names.index(cls_name)
                        cls_score = float(window_scores[cls_idx])
                        if cls_score > max_laugh_conf:
                            max_laugh_conf = cls_score
                            max_laugh_class = cls_name
                        if cls_score > self.min_confidence:
                            laugh_windows += 1
                    except (ValueError, IndexError):
                        continue

                # Check crying classes for cry→laugh transition
                for cls_name in self.crying_classes:
                    try:
                        cls_idx = self._class_names.index(cls_name)
                        if float(window_scores[cls_idx]) > 0.5:
                            cry_detected = True
                    except (ValueError, IndexError):
                        continue

            return LaughterMetrics(
                laugh_confidence=round(max_laugh_conf, 3),
                laugh_class=max_laugh_class,
                laugh_windows=laugh_windows,
                total_windows=len(scores_np),
                cry_detected=cry_detected,
            )

        except Exception as e:
            logger.error(f"YAMNet inference failed: {e}")
            return LaughterMetrics(
                laugh_confidence=0.0, laugh_class="",
                laugh_windows=0, total_windows=0,
            )


class VoiceAnalyzer:
    """ADD-3: Voice detection and transcription via Whisper.

    Uses OpenAI Whisper (local GPU) for voice activity detection,
    speech-to-text, and keyword detection.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        voice_cfg = config.get("addition_rules", {}).get("voice", {})
        self.model_name = voice_cfg.get("whisper_model", "tiny")
        self.language = voice_cfg.get("whisper_language", "ko")
        self.exclamation_keywords = set(voice_cfg.get("exclamation_keywords", []))
        self.baby_names = set(voice_cfg.get("baby_names", []))
        self.babble_threshold = voice_cfg.get("babble_confidence_threshold", 0.3)
        self._model = None

    def _load_model(self) -> bool:
        """Lazy-load Whisper model on first use."""
        if self._model is not None:
            return True
        try:
            import whisper
            logger.info(f"Loading Whisper model '{self.model_name}'...")
            self._model = whisper.load_model(self.model_name)
            logger.info("Whisper model loaded")
            return True
        except ImportError:
            logger.warning(
                "openai-whisper not installed. "
                "Voice analysis will be disabled."
            )
            return False
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
            return False

    def analyze(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> VoiceMetrics:
        """Analyze audio segment for voice activity and transcription.

        Args:
            audio: Audio samples for this segment (mono, float, 16kHz).
            sr: Sample rate.

        Returns:
            VoiceMetrics with transcription and keyword detection.
        """
        if len(audio) == 0 or not self._load_model():
            return VoiceMetrics(
                voice_activity_pct=0.0, transcript="",
                has_exclamation=False, has_baby_name=False,
                is_babble=False, whisper_confidence=0.0,
            )

        try:
            import whisper

            # Pad or trim to 30 seconds (Whisper requirement)
            padded = whisper.pad_or_trim(audio.astype(np.float32))

            # Create mel spectrogram
            mel = whisper.log_mel_spectrogram(padded).to(self._model.device)

            # Detect language and decode
            options = whisper.DecodingOptions(
                language=self.language,
                without_timestamps=False,
            )
            result = whisper.decode(self._model, mel, options)

            transcript = result.text.strip()
            avg_logprob = float(result.avg_logprob) if hasattr(result, "avg_logprob") else 0.0
            confidence = np.exp(avg_logprob)  # convert log-prob to probability
            no_speech_prob = float(result.no_speech_prob) if hasattr(result, "no_speech_prob") else 1.0

            voice_activity_pct = 1.0 - no_speech_prob

            # Keyword detection
            has_exclamation = any(kw in transcript for kw in self.exclamation_keywords)
            has_baby_name = any(name in transcript for name in self.baby_names) if self.baby_names else False

            # Babble detection: low confidence + some voice activity
            is_babble = (
                voice_activity_pct > 0.3
                and confidence < self.babble_threshold
                and len(transcript) < 10
            )

            return VoiceMetrics(
                voice_activity_pct=round(voice_activity_pct, 3),
                transcript=transcript,
                has_exclamation=has_exclamation,
                has_baby_name=has_baby_name,
                is_babble=is_babble,
                whisper_confidence=round(float(confidence), 3),
            )

        except Exception as e:
            logger.error(f"Whisper inference failed: {e}")
            return VoiceMetrics(
                voice_activity_pct=0.0, transcript="",
                has_exclamation=False, has_baby_name=False,
                is_babble=False, whisper_confidence=0.0,
            )
