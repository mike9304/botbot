"""Video quality analyzers: shake, exposure, focus, obstruction.

Implements DEL-1 (camera shake), DEL-3 (exposure anomaly),
DEL-4 (out of focus), DEL-5 (lens obstruction).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ShakeMetrics:
    """Raw metrics from camera shake analysis."""
    mean_flow: float                  # mean optical flow magnitude (px/frame)
    exceeded_pct: float               # % of sampled pairs exceeding threshold
    raw_magnitudes: list[float]       # per-pair magnitudes


@dataclass
class ExposureMetrics:
    """Raw metrics from exposure analysis."""
    top5_pct: float                   # % of pixels in top 5% luminance
    bot5_pct: float                   # % of pixels in bottom 5% luminance
    per_frame_top5: list[float]
    per_frame_bot5: list[float]


@dataclass
class FocusMetrics:
    """Raw metrics from focus analysis."""
    laplacian_var: float              # minimum Laplacian variance
    laplacian_std: float              # std across sampled frames
    per_frame_var: list[float]


@dataclass
class ObstructionMetrics:
    """Raw metrics from lens obstruction analysis."""
    frame_std: float                  # minimum frame std across samples
    mean_brightness: float            # mean brightness of worst frame
    per_frame_std: list[float]
    per_frame_brightness: list[float]


class ShakeAnalyzer:
    """DEL-1: Camera shake detection via optical flow.

    Uses Farneback dense optical flow to measure frame-to-frame motion.
    High mean flow magnitude sustained across a segment indicates shake.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        shake_cfg = config.get("delete_rules", {}).get("shake", {})
        of_cfg = shake_cfg.get("optical_flow", {})
        self.severe_threshold = shake_cfg.get("severe_threshold", 12)
        self.mild_threshold = shake_cfg.get("mild_threshold", 5)
        self.sustained_pct = shake_cfg.get("sustained_pct", 0.8)
        self.frame_step = of_cfg.get("frame_step", 3)
        self.pyr_scale = of_cfg.get("pyr_scale", 0.5)
        self.levels = of_cfg.get("levels", 3)
        self.winsize = of_cfg.get("winsize", 15)
        self.iterations = of_cfg.get("iterations", 3)
        self.poly_n = of_cfg.get("poly_n", 5)
        self.poly_sigma = of_cfg.get("poly_sigma", 1.2)

    def analyze(self, frames: list[np.ndarray]) -> ShakeMetrics:
        """Analyze camera shake from a list of frames.

        Computes dense optical flow between consecutive sampled frames
        and returns the mean magnitude of motion vectors.
        """
        if len(frames) < 2:
            return ShakeMetrics(mean_flow=0.0, exceeded_pct=0.0, raw_magnitudes=[])

        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        magnitudes = []

        for i in range(len(grays) - 1):
            flow = cv2.calcOpticalFlowFarneback(
                grays[i], grays[i + 1],
                None,
                pyr_scale=self.pyr_scale,
                levels=self.levels,
                winsize=self.winsize,
                iterations=self.iterations,
                poly_n=self.poly_n,
                poly_sigma=self.poly_sigma,
                flags=0,
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            magnitudes.append(float(np.mean(mag)))

        if not magnitudes:
            return ShakeMetrics(mean_flow=0.0, exceeded_pct=0.0, raw_magnitudes=[])

        mean_flow = float(np.mean(magnitudes))
        exceeded = sum(1 for m in magnitudes if m > self.mild_threshold)
        exceeded_pct = exceeded / len(magnitudes)

        return ShakeMetrics(
            mean_flow=round(mean_flow, 2),
            exceeded_pct=round(exceeded_pct, 3),
            raw_magnitudes=[round(m, 2) for m in magnitudes],
        )


class ExposureAnalyzer:
    """DEL-3: Exposure anomaly detection via histogram analysis.

    Checks if too many pixels are concentrated in the extreme
    top or bottom 5% of the luminance range.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        exp_cfg = config.get("delete_rules", {}).get("exposure", {})
        self.overexpose_pct = exp_cfg.get("overexpose_pct", 0.60)
        self.underexpose_pct = exp_cfg.get("underexpose_pct", 0.70)
        self.warn_overexpose_pct = exp_cfg.get("warn_overexpose_pct", 0.40)
        self.warn_underexpose_pct = exp_cfg.get("warn_underexpose_pct", 0.50)
        self.top_bin_start = exp_cfg.get("top_bin_start", 243)
        self.bot_bin_end = exp_cfg.get("bot_bin_end", 12)

    def analyze(self, frames: list[np.ndarray]) -> ExposureMetrics:
        """Analyze exposure from sampled frames.

        Computes histogram and checks pixel concentration in extreme ranges.
        """
        per_frame_top5 = []
        per_frame_bot5 = []

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            total_pixels = gray.shape[0] * gray.shape[1]

            top5 = float(np.sum(hist[self.top_bin_start:256])) / total_pixels
            bot5 = float(np.sum(hist[0:self.bot_bin_end + 1])) / total_pixels

            per_frame_top5.append(round(top5, 4))
            per_frame_bot5.append(round(bot5, 4))

        avg_top5 = float(np.mean(per_frame_top5)) if per_frame_top5 else 0.0
        avg_bot5 = float(np.mean(per_frame_bot5)) if per_frame_bot5 else 0.0

        return ExposureMetrics(
            top5_pct=round(avg_top5, 4),
            bot5_pct=round(avg_bot5, 4),
            per_frame_top5=per_frame_top5,
            per_frame_bot5=per_frame_bot5,
        )


class FocusAnalyzer:
    """DEL-4: Focus quality detection via Laplacian variance.

    Uses the variance of the Laplacian operator as a measure of
    image sharpness. Low variance indicates a blurry/out-of-focus image.
    Analyzes center 60% crop to ignore bokeh at edges.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        focus_cfg = config.get("delete_rules", {}).get("focus", {})
        self.blur_threshold = focus_cfg.get("blur_threshold", 50)
        self.soft_threshold = focus_cfg.get("soft_threshold", 100)
        self.hunting_std = focus_cfg.get("hunting_std", 30)
        self.center_crop_ratio = focus_cfg.get("center_crop_ratio", 0.6)

    def _center_crop(self, frame: np.ndarray) -> np.ndarray:
        """Crop center portion of frame to ignore edge bokeh."""
        h, w = frame.shape[:2]
        margin_h = int(h * (1 - self.center_crop_ratio) / 2)
        margin_w = int(w * (1 - self.center_crop_ratio) / 2)
        return frame[margin_h:h - margin_h, margin_w:w - margin_w]

    def analyze(self, frames: list[np.ndarray]) -> FocusMetrics:
        """Analyze focus quality from sampled frames.

        Returns minimum Laplacian variance and checks for focus hunting.
        """
        per_frame_var = []

        for frame in frames:
            cropped = self._center_crop(frame)
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = float(laplacian.var())
            per_frame_var.append(round(variance, 1))

        if not per_frame_var:
            return FocusMetrics(laplacian_var=0.0, laplacian_std=0.0, per_frame_var=[])

        return FocusMetrics(
            laplacian_var=round(min(per_frame_var), 1),
            laplacian_std=round(float(np.std(per_frame_var)), 1),
            per_frame_var=per_frame_var,
        )


class ObstructionAnalyzer:
    """DEL-5: Lens obstruction detection via frame statistics.

    Detects lens cap, pocket filming, or finger over lens by
    checking if the frame has extremely low standard deviation
    (uniform color = something blocking the lens).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        obs_cfg = config.get("delete_rules", {}).get("obstruction", {})
        self.std_hard = obs_cfg.get("std_threshold_hard", 5)
        self.std_soft = obs_cfg.get("std_threshold_soft", 10)
        self.dark_brightness = obs_cfg.get("dark_brightness", 30)
        self.bright_brightness = obs_cfg.get("bright_brightness", 220)

    def analyze(self, frames: list[np.ndarray]) -> ObstructionMetrics:
        """Analyze lens obstruction from sampled frames.

        Both frames must fail for the score to apply (avoids false
        positives from fade transitions).
        """
        per_frame_std = []
        per_frame_brightness = []

        for frame in frames:
            std = float(np.std(frame))
            brightness = float(np.mean(frame))
            per_frame_std.append(round(std, 2))
            per_frame_brightness.append(round(brightness, 2))

        if not per_frame_std:
            return ObstructionMetrics(
                frame_std=999.0, mean_brightness=128.0,
                per_frame_std=[], per_frame_brightness=[],
            )

        # Use minimum std (worst case) — but both frames must fail
        min_std = min(per_frame_std)
        # Brightness of the frame with minimum std
        min_idx = per_frame_std.index(min_std)
        brightness_at_min = per_frame_brightness[min_idx]

        return ObstructionMetrics(
            frame_std=round(min_std, 2),
            mean_brightness=round(brightness_at_min, 2),
            per_frame_std=per_frame_std,
            per_frame_brightness=per_frame_brightness,
        )
