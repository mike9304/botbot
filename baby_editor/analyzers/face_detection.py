"""Face detection and gaze estimation analyzers.

Implements ADD-1 (face detected) and ADD-5 (camera gaze / eye contact).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FaceMetrics:
    """Raw metrics from face detection analysis."""
    face_count: int                   # max faces detected in any single frame
    max_confidence: float             # highest detection confidence
    max_size_pct: float               # largest face bbox as % of frame area
    is_frontal: bool                  # frontal face detected (high confidence proxy)
    is_closeup: bool                  # face > 10% of frame area
    has_multiple: bool                # 2+ faces in single frame
    has_twins: bool                   # 2 baby-sized faces detected
    per_frame_counts: list[int] = field(default_factory=list)


@dataclass
class GazeMetrics:
    """Raw metrics from gaze/eye contact analysis."""
    looking_at_camera: bool
    gaze_confidence: float            # proxy confidence for eye contact
    is_smiling: bool                  # placeholder for Sprint 2


class FaceAnalyzer:
    """ADD-1: Face detection using MediaPipe.

    Detects faces in sampled frames and classifies them by:
    - Confidence level
    - Size relative to frame (closeup detection)
    - Count (multiple faces, twins)
    - Orientation (frontal via confidence proxy)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        face_cfg = config.get("addition_rules", {}).get("face", {})
        self.min_confidence = face_cfg.get("min_confidence", 0.7)
        self.frontal_confidence = face_cfg.get("frontal_confidence", 0.85)
        self.closeup_pct = face_cfg.get("closeup_pct", 0.10)
        self.model_selection = face_cfg.get("model_selection", 1)
        self._detector = None

    def _init_detector(self) -> bool:
        """Initialize MediaPipe face detector on first use."""
        if self._detector is not None:
            return True
        try:
            import mediapipe as mp
            self._detector = mp.solutions.face_detection.FaceDetection(
                model_selection=self.model_selection,
                min_detection_confidence=self.min_confidence,
            )
            logger.info("MediaPipe FaceDetection initialized")
            return True
        except ImportError:
            logger.warning(
                "mediapipe not installed. Face detection will be disabled."
            )
            return False
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")
            return False

    def analyze(self, frames: list[np.ndarray]) -> FaceMetrics:
        """Analyze frames for face detection.

        Args:
            frames: List of BGR frames sampled from a 2s segment.

        Returns:
            FaceMetrics with face count, confidence, size, and classification.
        """
        if not frames or not self._init_detector():
            return FaceMetrics(
                face_count=0, max_confidence=0.0, max_size_pct=0.0,
                is_frontal=False, is_closeup=False,
                has_multiple=False, has_twins=False,
            )

        max_face_count = 0
        max_confidence = 0.0
        max_size_pct = 0.0
        per_frame_counts = []
        any_frontal = False
        any_closeup = False
        any_multiple = False
        any_twins = False

        for frame in frames:
            h, w = frame.shape[:2]
            frame_area = h * w

            # MediaPipe expects RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._detector.process(rgb)

            if not results.detections:
                per_frame_counts.append(0)
                continue

            detections = results.detections
            face_count = len(detections)
            per_frame_counts.append(face_count)
            max_face_count = max(max_face_count, face_count)

            if face_count >= 2:
                any_multiple = True

            face_sizes = []
            for det in detections:
                confidence = det.score[0]
                max_confidence = max(max_confidence, confidence)

                # Bounding box size
                bbox = det.location_data.relative_bounding_box
                face_w = bbox.width * w
                face_h = bbox.height * h
                face_area_pct = (face_w * face_h) / frame_area
                max_size_pct = max(max_size_pct, face_area_pct)
                face_sizes.append(face_area_pct)

                if confidence > self.frontal_confidence:
                    any_frontal = True
                if face_area_pct > self.closeup_pct:
                    any_closeup = True

            # Twin detection heuristic: 2 faces of similar size (both smallish)
            if face_count >= 2:
                sorted_sizes = sorted(face_sizes, reverse=True)
                if len(sorted_sizes) >= 2:
                    size_ratio = sorted_sizes[1] / sorted_sizes[0] if sorted_sizes[0] > 0 else 0
                    # Both faces similar size AND both relatively small (baby-sized)
                    if size_ratio > 0.5 and sorted_sizes[0] < 0.25:
                        any_twins = True

        return FaceMetrics(
            face_count=max_face_count,
            max_confidence=round(max_confidence, 3),
            max_size_pct=round(max_size_pct * 100, 1),  # convert to percentage
            is_frontal=any_frontal,
            is_closeup=any_closeup,
            has_multiple=any_multiple,
            has_twins=any_twins,
            per_frame_counts=per_frame_counts,
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._detector is not None:
            self._detector.close()
            self._detector = None


class GazeAnalyzer:
    """ADD-5: Camera gaze / eye contact detection.

    Sprint 1: Simplified proxy — uses high face detection confidence
    combined with frontal face as an indicator of eye contact.

    Sprint 2: Will use MediaPipe Face Mesh iris landmarks for precise
    gaze angle estimation.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        gaze_cfg = config.get("addition_rules", {}).get("gaze", {})
        self.enabled = gaze_cfg.get("enabled", False)
        self.proxy_confidence = gaze_cfg.get("proxy_confidence", 0.9)

    def analyze(
        self,
        frames: list[np.ndarray],
        face_metrics: FaceMetrics,
    ) -> GazeMetrics:
        """Estimate gaze direction from face detection results.

        Sprint 1 uses a simplified proxy: if the face is frontal and
        detection confidence is very high, we assume eye contact.

        Args:
            frames: Sampled frames (unused in Sprint 1).
            face_metrics: Results from FaceAnalyzer.

        Returns:
            GazeMetrics with eye contact estimation.
        """
        if not self.enabled:
            # Sprint 1: proxy-based estimation
            looking = (
                face_metrics.is_frontal
                and face_metrics.max_confidence > self.proxy_confidence
            )
            return GazeMetrics(
                looking_at_camera=looking,
                gaze_confidence=face_metrics.max_confidence if looking else 0.0,
                is_smiling=False,
            )

        # Sprint 2: full Face Mesh gaze estimation (placeholder)
        return GazeMetrics(
            looking_at_camera=False,
            gaze_confidence=0.0,
            is_smiling=False,
        )
