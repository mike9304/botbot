"""Motion detection analyzer.

Implements ADD-4 (subject movement detection with camera pan rejection).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MotionMetrics:
    """Raw metrics from motion analysis."""
    movement_pct: float               # % of pixels that changed between frames
    is_camera_pan: bool               # True if motion is uniform (camera, not subject)
    flow_uniformity: float            # how uniform the optical flow is (0-1)
    per_pair_pct: list[float]         # per-frame-pair movement percentages


class MotionAnalyzer:
    """ADD-4: Subject movement detection via frame differencing.

    Detects subject movement by computing pixel changes between
    consecutive frames, while filtering out camera pans by checking
    optical flow uniformity.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        motion_cfg = config.get("addition_rules", {}).get("movement", {})
        self.min_change_pct = motion_cfg.get("min_change_pct", 5)
        self.active_change_pct = motion_cfg.get("active_change_pct", 15)
        self.pixel_diff_threshold = motion_cfg.get("pixel_diff_threshold", 25)
        self.uniform_flow_threshold = motion_cfg.get("uniform_flow_threshold", 0.7)

    def analyze(self, frames: list[np.ndarray]) -> MotionMetrics:
        """Analyze subject movement from sampled frames.

        Uses frame differencing to detect pixel changes, then checks
        optical flow uniformity to distinguish subject movement from
        camera pans.

        Args:
            frames: List of BGR frames from a 2s segment.

        Returns:
            MotionMetrics with movement percentage and pan detection.
        """
        if len(frames) < 2:
            return MotionMetrics(
                movement_pct=0.0, is_camera_pan=False,
                flow_uniformity=0.0, per_pair_pct=[],
            )

        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        per_pair_pct = []
        flow_uniformities = []

        for i in range(len(grays) - 1):
            # Frame differencing
            diff = cv2.absdiff(grays[i], grays[i + 1])
            _, thresh = cv2.threshold(diff, self.pixel_diff_threshold, 255, cv2.THRESH_BINARY)
            non_zero = cv2.countNonZero(thresh)
            total_pixels = grays[i].shape[0] * grays[i].shape[1]
            pct = (non_zero / total_pixels) * 100
            per_pair_pct.append(round(pct, 1))

            # Optical flow uniformity check (camera pan detection)
            flow = cv2.calcOpticalFlowFarneback(
                grays[i], grays[i + 1], None,
                pyr_scale=0.5, levels=2, winsize=15,
                iterations=2, poly_n=5, poly_sigma=1.1,
                flags=0,
            )
            uniformity = self._flow_uniformity(flow)
            flow_uniformities.append(uniformity)

        if not per_pair_pct:
            return MotionMetrics(
                movement_pct=0.0, is_camera_pan=False,
                flow_uniformity=0.0, per_pair_pct=[],
            )

        avg_movement = float(np.mean(per_pair_pct))
        avg_uniformity = float(np.mean(flow_uniformities))
        is_pan = avg_uniformity > self.uniform_flow_threshold

        return MotionMetrics(
            movement_pct=round(avg_movement, 1),
            is_camera_pan=is_pan,
            flow_uniformity=round(avg_uniformity, 3),
            per_pair_pct=per_pair_pct,
        )

    @staticmethod
    def _flow_uniformity(flow: np.ndarray) -> float:
        """Measure how uniform optical flow vectors are.

        Uniformity close to 1.0 means all pixels move in the same
        direction (camera pan). Low uniformity means varied motion
        (subject movement).

        Computed as: magnitude of mean flow vector / mean of flow magnitudes.
        """
        fx, fy = flow[..., 0], flow[..., 1]
        magnitudes = np.sqrt(fx ** 2 + fy ** 2)
        mean_mag = float(np.mean(magnitudes))

        if mean_mag < 0.5:
            # Very little motion overall — not a pan
            return 0.0

        mean_fx = float(np.mean(fx))
        mean_fy = float(np.mean(fy))
        mean_vector_mag = np.sqrt(mean_fx ** 2 + mean_fy ** 2)

        uniformity = mean_vector_mag / (mean_mag + 1e-6)
        return min(1.0, uniformity)
