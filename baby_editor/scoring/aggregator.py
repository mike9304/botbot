"""Score aggregator: combines all analyzer outputs into the final score_matrix.

Handles cross-segment rules (consecutive silence, cry-to-laugh transitions)
and builds the output JSON structure.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
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
from baby_editor.scoring.rules import SegmentScore

logger = logging.getLogger(__name__)


@dataclass
class SegmentResult:
    """Full result for a single 2-second segment."""
    index: int
    start: float
    end: float
    scores: dict[str, int]
    total_score: int
    action: str                  # "delete" | "select" | "neutral"
    tags: list[str]
    reasons: list[str]
    metrics: dict[str, Any]


@dataclass
class VideoResult:
    """Full result for a single video file."""
    filename: str
    path: str
    duration: float
    resolution: str
    fps: float
    created_at: str
    segments: list[SegmentResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


class ScoreAggregator:
    """Aggregates per-segment results into the final score_matrix structure.

    Handles:
    - Action classification (delete/select/neutral) based on thresholds
    - Cross-segment consecutive silence penalty
    - Cry-to-laugh transition bonus
    - Video and daily summaries
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.delete_threshold = config.get("delete_threshold", -7)
        self.selection_baseline = config.get("selection_baseline", 3)
        self.segment_duration = config.get("segment_duration", 2.0)
        self.consecutive_silence_sec = (
            config.get("delete_rules", {})
            .get("silence", {})
            .get("consecutive_seconds", 5)
        )

    def build_segment_result(
        self,
        index: int,
        start: float,
        end: float,
        score: SegmentScore,
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
    ) -> SegmentResult:
        """Build a SegmentResult from scores and raw metrics."""
        action = self._classify_action(score.total_score)

        metrics = {
            "optical_flow_mean": shake.mean_flow,
            "rms_db": silence.rms_db,
            "histogram_top5_pct": exposure.top5_pct,
            "histogram_bot5_pct": exposure.bot5_pct,
            "laplacian_var": focus.laplacian_var,
            "frame_std": obstruction.frame_std,
            "face_count": face.face_count,
            "face_max_confidence": face.max_confidence,
            "face_max_size_pct": face.max_size_pct,
            "laugh_confidence": laughter.laugh_confidence,
            "voice_activity_pct": voice.voice_activity_pct,
            "movement_pct": motion.movement_pct,
            "transcript": voice.transcript,
        }

        return SegmentResult(
            index=index,
            start=start,
            end=end,
            scores=score.scores,
            total_score=score.total_score,
            action=action,
            tags=score.tags,
            reasons=score.reasons,
            metrics=metrics,
        )

    def apply_cross_segment_rules(
        self, segments: list[SegmentResult],
    ) -> list[SegmentResult]:
        """Apply rules that require cross-segment context.

        1. Consecutive silence: if >=5 seconds of consecutive silence,
           mark ALL those segments as -8.
        2. Cry-to-laugh transition: bonus for segments where crying
           transitions to laughing within 5 seconds.
        """
        self._apply_consecutive_silence(segments)
        self._apply_cry_to_laugh(segments)
        return segments

    def _apply_consecutive_silence(self, segments: list[SegmentResult]) -> None:
        """Mark all segments in a consecutive silence run as -8."""
        min_segments = int(self.consecutive_silence_sec / self.segment_duration)
        run_start = None
        run_length = 0

        for i, seg in enumerate(segments):
            is_silent = "silence" in seg.tags or "near_silence" in seg.tags
            if is_silent:
                if run_start is None:
                    run_start = i
                run_length += 1
            else:
                if run_length >= min_segments and run_start is not None:
                    # Upgrade all segments in this run to full silence penalty
                    for j in range(run_start, run_start + run_length):
                        if segments[j].scores.get("del_silence", 0) > -8:
                            segments[j].scores["del_silence"] = -8
                            if "silence" not in segments[j].tags:
                                segments[j].tags.append("silence")
                            if "near_silence" in segments[j].tags:
                                segments[j].tags.remove("near_silence")
                            segments[j].reasons.append(
                                f"DEL-2: Part of {run_length * self.segment_duration:.0f}s "
                                f"consecutive silence run → upgraded to -8"
                            )
                            # Recalculate total score
                            segments[j].total_score = sum(segments[j].scores.values())
                            segments[j].action = self._classify_action(segments[j].total_score)
                run_start = None
                run_length = 0

        # Handle run at end of video
        if run_length >= min_segments and run_start is not None:
            for j in range(run_start, run_start + run_length):
                if segments[j].scores.get("del_silence", 0) > -8:
                    segments[j].scores["del_silence"] = -8
                    if "silence" not in segments[j].tags:
                        segments[j].tags.append("silence")
                    segments[j].total_score = sum(segments[j].scores.values())
                    segments[j].action = self._classify_action(segments[j].total_score)

    def _apply_cry_to_laugh(self, segments: list[SegmentResult]) -> None:
        """Detect cry→laugh transitions across adjacent segments."""
        window = int(5.0 / self.segment_duration)  # 5 seconds in segments
        for i in range(len(segments)):
            if "laugh" in segments[i].tags or "laugh_long" in segments[i].tags:
                # Look back within window for crying
                for j in range(max(0, i - window), i):
                    if segments[j].metrics.get("_cry_detected", False):
                        segments[i].scores["add_laugh"] = max(
                            segments[i].scores.get("add_laugh", 0), 6,
                        )
                        if "cry_to_laugh" not in segments[i].tags:
                            segments[i].tags.append("cry_to_laugh")
                            segments[i].reasons.append(
                                "ADD-2: Transition from crying to laughing detected"
                            )
                        segments[i].total_score = sum(segments[i].scores.values())
                        segments[i].action = self._classify_action(segments[i].total_score)
                        break

    def build_video_summary(self, segments: list[SegmentResult]) -> dict[str, Any]:
        """Build summary statistics for a video."""
        if not segments:
            return {}

        deleted = sum(1 for s in segments if s.action == "delete")
        selected = sum(1 for s in segments if s.action == "select")
        neutral = sum(1 for s in segments if s.action == "neutral")
        scores = [s.total_score for s in segments]

        face_time = sum(
            self.segment_duration
            for s in segments
            if s.metrics.get("face_count", 0) > 0
        )
        laugh_time = sum(
            self.segment_duration
            for s in segments
            if any(t in s.tags for t in ("laugh", "laugh_long", "giggle"))
        )
        voice_time = sum(
            self.segment_duration
            for s in segments
            if s.metrics.get("voice_activity_pct", 0) > 0.1
        )

        return {
            "total_segments": len(segments),
            "deleted_segments": deleted,
            "selected_segments": selected,
            "neutral_segments": neutral,
            "top_score": max(scores) if scores else 0,
            "avg_score": round(sum(scores) / len(scores), 1) if scores else 0,
            "total_face_time": round(face_time, 1),
            "total_laugh_time": round(laugh_time, 1),
            "total_voice_time": round(voice_time, 1),
        }

    def build_daily_summary(
        self, video_results: list[VideoResult],
    ) -> dict[str, Any]:
        """Build daily summary across all videos."""
        total_videos = len(video_results)
        total_duration = sum(v.duration for v in video_results)
        all_segments = [s for v in video_results for s in v.segments]
        total_segments = len(all_segments)

        deleted = sum(1 for s in all_segments if s.action == "delete")
        selected = sum(1 for s in all_segments if s.action == "select")

        # Find highlight segments (top scoring across all videos)
        highlights = []
        for video in video_results:
            for seg in video.segments:
                if seg.total_score >= 10:
                    top_tag = seg.tags[0] if seg.tags else ""
                    highlights.append({
                        "video": video.filename,
                        "start": seg.start,
                        "end": seg.end,
                        "score": seg.total_score,
                        "top_tag": top_tag,
                    })
        highlights.sort(key=lambda x: x["score"], reverse=True)
        highlights = highlights[:20]  # top 20

        return {
            "total_videos": total_videos,
            "total_duration": round(total_duration, 1),
            "total_segments": total_segments,
            "deleted_pct": round(deleted / total_segments * 100, 1) if total_segments else 0,
            "selected_pct": round(selected / total_segments * 100, 1) if total_segments else 0,
            "highlight_segments": highlights,
        }

    def build_score_matrix(
        self, video_results: list[VideoResult],
    ) -> dict[str, Any]:
        """Build the complete score_matrix.json structure."""
        return {
            "version": "1.0",
            "analyzed_at": datetime.now().isoformat(timespec="seconds"),
            "config": {
                "segment_duration": self.segment_duration,
                "delete_threshold": self.delete_threshold,
                "selection_baseline": self.selection_baseline,
            },
            "videos": [self._serialize_video(v) for v in video_results],
            "daily_summary": self.build_daily_summary(video_results),
        }

    def _classify_action(self, total_score: int) -> str:
        """Classify segment action based on total score."""
        if total_score <= self.delete_threshold:
            return "delete"
        elif total_score >= self.selection_baseline:
            return "select"
        return "neutral"

    @staticmethod
    def _serialize_video(v: VideoResult) -> dict[str, Any]:
        """Serialize a VideoResult to a JSON-compatible dict."""
        return {
            "filename": v.filename,
            "path": v.path,
            "duration": v.duration,
            "resolution": v.resolution,
            "fps": v.fps,
            "created_at": v.created_at,
            "segments": [
                {
                    "index": s.index,
                    "start": s.start,
                    "end": s.end,
                    "scores": s.scores,
                    "total_score": s.total_score,
                    "action": s.action,
                    "tags": s.tags,
                    "reasons": s.reasons,
                    "metrics": {
                        k: v for k, v in s.metrics.items()
                        if not k.startswith("_")
                    },
                }
                for s in v.segments
            ],
            "summary": v.summary,
        }
