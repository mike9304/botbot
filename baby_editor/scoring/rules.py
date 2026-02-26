"""Scoring rules: apply thresholds to raw metrics and produce scores/tags/reasons.

All thresholds are read from config so they can be tuned without code changes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from baby_editor.analyzers.audio_analysis import (
    LaughterMetrics,
    SilenceMetrics,
    VoiceMetrics,
)
from baby_editor.analyzers.face_detection import FaceMetrics, GazeMetrics
from baby_editor.analyzers.motion_detection import MotionMetrics
from baby_editor.analyzers.video_quality import (
    ExposureMetrics,
    FocusMetrics,
    ObstructionMetrics,
    ShakeMetrics,
)

logger = logging.getLogger(__name__)


@dataclass
class SegmentScore:
    """Scored result for a single 2-second segment."""
    scores: dict[str, int] = field(default_factory=dict)
    total_score: int = 0
    tags: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


class SegmentScorer:
    """Applies all 10 scoring rules (5 delete + 5 addition) to raw metrics.

    Each rule method receives the corresponding metrics dataclass and
    returns (score, tags, reasons).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.del_cfg = config.get("delete_rules", {})
        self.add_cfg = config.get("addition_rules", {})

    def score_segment(
        self,
        shake: ShakeMetrics,
        silence: SilenceMetrics,
        exposure: ExposureMetrics,
        focus: FocusMetrics,
        obstruction: ObstructionMetrics,
        face: FaceMetrics,
        laughter: LaughterMetrics,
        voice: VoiceMetrics,
        motion: MotionMetrics,
        gaze: GazeMetrics,
    ) -> SegmentScore:
        """Apply all scoring rules to a segment's raw metrics.

        Returns a SegmentScore with individual and total scores.
        """
        result = SegmentScore()

        # Delete rules
        self._score_shake(shake, result)
        self._score_silence(silence, result)
        self._score_exposure(exposure, result)
        self._score_focus(focus, result)
        self._score_obstruction(obstruction, result)

        # Addition rules
        self._score_face(face, result)
        self._score_laughter(laughter, result)
        self._score_voice(voice, result)
        self._score_movement(motion, voice, result)
        self._score_gaze(gaze, result)

        result.total_score = sum(result.scores.values())
        return result

    # ─── DELETE RULES ───────────────────────────────────

    def _score_shake(self, m: ShakeMetrics, r: SegmentScore) -> None:
        """DEL-1: Camera shake scoring."""
        cfg = self.del_cfg.get("shake", {})
        severe = cfg.get("severe_threshold", 12)
        mild = cfg.get("mild_threshold", 5)
        sustained = cfg.get("sustained_pct", 0.8)

        if m.mean_flow > severe and m.exceeded_pct >= sustained:
            r.scores["del_shake"] = -10
            r.tags.append("shake_severe")
            r.reasons.append(
                f"DEL-1: Mean flow={m.mean_flow} px/frame (> {severe}), "
                f"{m.exceeded_pct:.0%} sustained → severe shake"
            )
        elif m.mean_flow > mild and m.exceeded_pct >= sustained:
            r.scores["del_shake"] = -3
            r.tags.append("shake_mild")
            r.reasons.append(
                f"DEL-1: Mean flow={m.mean_flow} px/frame ({mild}-{severe}), "
                f"{m.exceeded_pct:.0%} sustained → mild shake"
            )
        else:
            r.scores["del_shake"] = 0

    def _score_silence(self, m: SilenceMetrics, r: SegmentScore) -> None:
        """DEL-2: Silence scoring.

        Note: consecutive silence across segments is handled by the
        aggregator as a post-processing step.
        """
        cfg = self.del_cfg.get("silence", {})
        near_pct = cfg.get("near_silence_pct", 0.6)

        if m.silent_window_pct >= 1.0:
            r.scores["del_silence"] = -8
            r.tags.append("silence")
            r.reasons.append(
                f"DEL-2: RMS={m.rms_db:.1f}dB, 100% windows silent → full silence"
            )
        elif m.silent_window_pct >= near_pct:
            r.scores["del_silence"] = -3
            r.tags.append("near_silence")
            r.reasons.append(
                f"DEL-2: RMS={m.rms_db:.1f}dB, {m.silent_window_pct:.0%} windows "
                f"silent → near silence"
            )
        else:
            r.scores["del_silence"] = 0

    def _score_exposure(self, m: ExposureMetrics, r: SegmentScore) -> None:
        """DEL-3: Exposure anomaly scoring."""
        cfg = self.del_cfg.get("exposure", {})
        over_pct = cfg.get("overexpose_pct", 0.60)
        under_pct = cfg.get("underexpose_pct", 0.70)
        warn_over = cfg.get("warn_overexpose_pct", 0.40)
        warn_under = cfg.get("warn_underexpose_pct", 0.50)

        if m.top5_pct > over_pct:
            r.scores["del_exposure"] = -8
            r.tags.append("overexposed")
            r.reasons.append(
                f"DEL-3: Top 5% luminance has {m.top5_pct:.0%} of pixels → overexposed"
            )
        elif m.bot5_pct > under_pct:
            r.scores["del_exposure"] = -9
            r.tags.append("underexposed")
            r.reasons.append(
                f"DEL-3: Bottom 5% luminance has {m.bot5_pct:.0%} of pixels → underexposed"
            )
        elif m.top5_pct > warn_over or m.bot5_pct > warn_under:
            r.scores["del_exposure"] = -3
            r.tags.append("exposure_warn")
            r.reasons.append(
                f"DEL-3: Top5={m.top5_pct:.0%}, Bot5={m.bot5_pct:.0%} → exposure warning"
            )
        else:
            r.scores["del_exposure"] = 0

    def _score_focus(self, m: FocusMetrics, r: SegmentScore) -> None:
        """DEL-4: Out-of-focus scoring."""
        cfg = self.del_cfg.get("focus", {})
        blur = cfg.get("blur_threshold", 50)
        soft = cfg.get("soft_threshold", 100)
        hunting_std = cfg.get("hunting_std", 30)

        # Focus hunting takes priority (oscillating focus)
        if m.laplacian_std > hunting_std and len(m.per_frame_var) >= 2:
            r.scores["del_focus"] = -5
            r.tags.append("focus_hunting")
            r.reasons.append(
                f"DEL-4: Laplacian var std={m.laplacian_std:.1f} (> {hunting_std}) → focus hunting"
            )
        elif m.laplacian_var < blur:
            r.scores["del_focus"] = -9
            r.tags.append("out_of_focus")
            r.reasons.append(
                f"DEL-4: Laplacian variance={m.laplacian_var:.1f} (< {blur}) → out of focus"
            )
        elif m.laplacian_var < soft:
            r.scores["del_focus"] = -3
            r.tags.append("soft_focus")
            r.reasons.append(
                f"DEL-4: Laplacian variance={m.laplacian_var:.1f} ({blur}-{soft}) → soft focus"
            )
        else:
            r.scores["del_focus"] = 0

    def _score_obstruction(self, m: ObstructionMetrics, r: SegmentScore) -> None:
        """DEL-5: Lens obstruction scoring."""
        cfg = self.del_cfg.get("obstruction", {})
        std_hard = cfg.get("std_threshold_hard", 5)
        std_soft = cfg.get("std_threshold_soft", 10)
        dark = cfg.get("dark_brightness", 30)
        bright = cfg.get("bright_brightness", 220)

        # Both frames must fail for hard obstruction
        all_blocked = all(s < std_hard for s in m.per_frame_std) if m.per_frame_std else False

        if all_blocked and m.mean_brightness < dark:
            r.scores["del_obstruction"] = -10
            r.tags.append("lens_blocked_dark")
            r.reasons.append(
                f"DEL-5: Frame std={m.frame_std:.1f} (< {std_hard}), "
                f"brightness={m.mean_brightness:.0f} → lens blocked (dark)"
            )
        elif all_blocked and m.mean_brightness > bright:
            r.scores["del_obstruction"] = -10
            r.tags.append("lens_blocked_bright")
            r.reasons.append(
                f"DEL-5: Frame std={m.frame_std:.1f} (< {std_hard}), "
                f"brightness={m.mean_brightness:.0f} → lens blocked (bright)"
            )
        elif m.frame_std < std_soft:
            r.scores["del_obstruction"] = -7
            r.tags.append("lens_obstructed")
            r.reasons.append(
                f"DEL-5: Frame std={m.frame_std:.1f} (< {std_soft}) → lens obstructed"
            )
        else:
            r.scores["del_obstruction"] = 0

    # ─── ADDITION RULES ────────────────────────────────

    def _score_face(self, m: FaceMetrics, r: SegmentScore) -> None:
        """ADD-1: Face detection scoring."""
        score = 0
        if m.face_count == 0 or m.max_confidence < self.add_cfg.get("face", {}).get("min_confidence", 0.7):
            r.scores["add_face"] = 0
            return

        # Base: face detected
        score += 3
        r.tags.append("face")
        reasons = [f"ADD-1: Face detected (conf={m.max_confidence:.2f})"]

        # Frontal bonus
        if m.is_frontal:
            score += 2
            r.tags.append("face_frontal")
            reasons[0] += ", frontal"

        # Closeup bonus
        if m.is_closeup:
            score += 2
            r.tags.append("face_closeup")
            reasons[0] += f", closeup ({m.max_size_pct:.1f}% of frame)"

        # Multiple faces
        if m.has_multiple:
            score += 3
            r.tags.append("face_multiple")
            reasons.append(f"ADD-1: Multiple faces detected ({m.face_count})")

        # Twins together
        if m.has_twins:
            score += 5
            r.tags.append("twins_together")
            reasons.append("ADD-1: Two baby-sized faces detected → twins together")

        r.scores["add_face"] = score
        r.reasons.extend(reasons)

    def _score_laughter(self, m: LaughterMetrics, r: SegmentScore) -> None:
        """ADD-2: Laughter scoring."""
        if m.laugh_confidence < self.add_cfg.get("laugh", {}).get("min_confidence", 0.7):
            r.scores["add_laugh"] = 0
            return

        long_dur = self.add_cfg.get("laugh", {}).get("long_duration", 3.0)
        yamnet_hop = self.add_cfg.get("laugh", {}).get("yamnet_hop", 0.48)

        # Estimate laughter duration from window count
        laugh_duration = m.laugh_windows * yamnet_hop

        if laugh_duration >= long_dur:
            r.scores["add_laugh"] = 8
            r.tags.append("laugh_long")
            r.reasons.append(
                f"ADD-2: Sustained laughter ({laugh_duration:.1f}s, "
                f"conf={m.laugh_confidence:.2f})"
            )
        elif "Giggle" in m.laugh_class:
            r.scores["add_laugh"] = 7
            r.tags.append("giggle")
            r.reasons.append(
                f"ADD-2: Giggle detected (conf={m.laugh_confidence:.2f})"
            )
        else:
            r.scores["add_laugh"] = 5
            r.tags.append("laugh")
            r.reasons.append(
                f"ADD-2: Baby laughter (conf={m.laugh_confidence:.2f})"
            )

    def _score_voice(self, m: VoiceMetrics, r: SegmentScore) -> None:
        """ADD-3: Voice/speech scoring."""
        score = 0

        if m.voice_activity_pct < 0.1:
            r.scores["add_voice"] = 0
            return

        # Base voice
        score += 2
        r.tags.append("voice")
        reason_parts = [f"ADD-3: Voice activity {m.voice_activity_pct:.0%}"]

        if m.has_exclamation:
            score += 4
            r.tags.append("exclamation")
            reason_parts.append("exclamation detected")

        if m.has_baby_name:
            score += 3
            r.tags.append("name_called")
            reason_parts.append("baby name called")

        if m.is_babble:
            score += 3
            r.tags.append("babble")
            reason_parts.append("baby babbling")

        r.scores["add_voice"] = score
        r.reasons.append(", ".join(reason_parts))

    def _score_movement(
        self, m: MotionMetrics, v: VoiceMetrics, r: SegmentScore,
    ) -> None:
        """ADD-4: Movement scoring with camera pan rejection."""
        active_pct = self.add_cfg.get("movement", {}).get("active_change_pct", 15)
        min_pct = self.add_cfg.get("movement", {}).get("min_change_pct", 5)

        # If it's a camera pan, not subject movement → ignore
        if m.is_camera_pan and m.movement_pct > min_pct:
            r.scores["add_movement"] = 0
            r.tags.append("camera_pan")
            r.reasons.append(
                f"ADD-4: Movement {m.movement_pct:.1f}% but uniform flow "
                f"(uniformity={m.flow_uniformity:.2f}) → camera pan, not subject"
            )
            return

        if m.movement_pct >= active_pct:
            r.scores["add_movement"] = 3
            r.tags.append("active_movement")
            r.reasons.append(
                f"ADD-4: Active movement {m.movement_pct:.1f}% pixel change"
            )
        elif m.movement_pct >= min_pct:
            r.scores["add_movement"] = 2
            r.tags.append("movement")
            r.reasons.append(
                f"ADD-4: Movement {m.movement_pct:.1f}% pixel change"
            )
        elif m.movement_pct < min_pct and v.voice_activity_pct < 0.1:
            r.scores["add_movement"] = -1
            r.tags.append("static_boring")
            r.reasons.append(
                f"ADD-4: No movement ({m.movement_pct:.1f}%) and no voice → static"
            )
        else:
            r.scores["add_movement"] = 0

    def _score_gaze(self, m: GazeMetrics, r: SegmentScore) -> None:
        """ADD-5: Camera gaze / eye contact scoring."""
        if not m.looking_at_camera:
            r.scores["add_gaze"] = 0
            return

        score = 3
        r.tags.append("eye_contact")
        reason = f"ADD-5: Eye contact detected (conf={m.gaze_confidence:.2f})"

        # Smile bonus (Sprint 2)
        if m.is_smiling:
            score += 5
            r.tags.append("eye_contact_smile")
            reason += " + smiling"

        r.scores["add_gaze"] = score
        r.reasons.append(reason)
