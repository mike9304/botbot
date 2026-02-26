"""Baby Video Auto-Editor — Phase 1: Score Matrix Generator.

Analyzes raw video files and produces a scoring matrix (JSON) for each
2-second segment. The matrix is consumed by Phase 2 (LLM editing decisions).

Usage:
    python -m baby_editor.analyze --input /videos/2025-10-15/
    python -m baby_editor.analyze --input /videos/2025-10-15/ --output results/
    python -m baby_editor.analyze --input video.mp4 --config custom.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

from baby_editor.analyzers.audio_analysis import (
    LaughterAnalyzer,
    LaughterMetrics,
    SilenceAnalyzer,
    SilenceMetrics,
    VoiceAnalyzer,
    VoiceMetrics,
)
from baby_editor.analyzers.face_detection import (
    FaceAnalyzer,
    FaceMetrics,
    GazeAnalyzer,
    GazeMetrics,
)
from baby_editor.analyzers.motion_detection import MotionAnalyzer, MotionMetrics
from baby_editor.analyzers.video_quality import (
    ExposureAnalyzer,
    ExposureMetrics,
    FocusAnalyzer,
    FocusMetrics,
    ObstructionAnalyzer,
    ObstructionMetrics,
    ShakeAnalyzer,
    ShakeMetrics,
)
from baby_editor.scoring.aggregator import ScoreAggregator, SegmentResult, VideoResult
from baby_editor.scoring.rules import SegmentScorer
from baby_editor.utils.audio_io import extract_audio, load_audio, slice_audio
from baby_editor.utils.gpu_utils import check_cuda
from baby_editor.utils.video_io import (
    VideoMeta,
    find_video_files,
    get_video_meta,
    iter_segments,
)

logger = logging.getLogger("baby_editor")


def load_config(config_path: Optional[str] = None) -> dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = str(Path(__file__).parent / "config.yaml")

    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {}

    with open(path) as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config or {}


def setup_logging(config: dict[str, Any]) -> None:
    """Configure logging from config."""
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    log_file = log_cfg.get("file", "analyze.log")

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="a", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


class AnalysisPipeline:
    """Main analysis pipeline: orchestrates all analyzers and scoring.

    Processes videos sequentially, segments within each video sequentially,
    and produces the final score_matrix.json output.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.segment_duration = config.get("segment_duration", 2.0)
        self.frame_sample_rate = config.get("frame_sample_rate", 3)
        self.audio_sr = config.get("audio", {}).get("sample_rate", 16000)

        # Initialize analyzers
        self.shake_analyzer = ShakeAnalyzer(config)
        self.exposure_analyzer = ExposureAnalyzer(config)
        self.focus_analyzer = FocusAnalyzer(config)
        self.obstruction_analyzer = ObstructionAnalyzer(config)
        self.silence_analyzer = SilenceAnalyzer(config)
        self.laughter_analyzer = LaughterAnalyzer(config)
        self.voice_analyzer = VoiceAnalyzer(config)
        self.face_analyzer = FaceAnalyzer(config)
        self.gaze_analyzer = GazeAnalyzer(config)
        self.motion_analyzer = MotionAnalyzer(config)

        # Scoring
        self.scorer = SegmentScorer(config)
        self.aggregator = ScoreAggregator(config)

    def analyze_video(self, video_path: str) -> Optional[VideoResult]:
        """Analyze a single video file.

        Returns VideoResult or None if the video cannot be processed.
        """
        t0 = time.time()
        logger.info(f"Analyzing: {video_path}")

        # 1. Get metadata
        meta = get_video_meta(video_path)
        if meta is None:
            logger.error(f"Skipping corrupted/unreadable video: {video_path}")
            return None

        logger.info(
            f"  Duration: {meta.duration}s, Resolution: {meta.resolution}, "
            f"FPS: {meta.fps}"
        )

        video_result = VideoResult(
            filename=meta.filename,
            path=meta.path,
            duration=meta.duration,
            resolution=meta.resolution,
            fps=meta.fps,
            created_at=meta.created_at,
        )

        # 2. Extract audio
        audio_data = None
        audio_path = extract_audio(video_path, sample_rate=self.audio_sr)
        if audio_path:
            audio_data, _ = load_audio(audio_path, sample_rate=self.audio_sr)
            if audio_data is not None:
                logger.info(f"  Audio: {len(audio_data) / self.audio_sr:.1f}s loaded")
            else:
                logger.warning(f"  Audio: failed to load, skipping audio rules")
        else:
            logger.info(f"  Audio: no audio track, skipping audio rules")

        # 3. Process each segment
        segment_results: list[SegmentResult] = []

        for segment in iter_segments(
            video_path,
            segment_duration=self.segment_duration,
            sample_rate=self.frame_sample_rate,
        ):
            try:
                result = self._analyze_segment(
                    segment.index,
                    segment.start,
                    segment.end,
                    segment.frames,
                    audio_data,
                )
                segment_results.append(result)
            except Exception as e:
                logger.error(
                    f"  Segment {segment.index} ({segment.start}-{segment.end}s) "
                    f"failed: {e}"
                )
                continue

        # 4. Cross-segment rules
        segment_results = self.aggregator.apply_cross_segment_rules(segment_results)

        # 5. Build summary
        video_result.segments = segment_results
        video_result.summary = self.aggregator.build_video_summary(segment_results)

        elapsed = time.time() - t0
        total_seg = len(segment_results)
        logger.info(
            f"  Completed: {total_seg} segments in {elapsed:.1f}s "
            f"({meta.duration / elapsed:.1f}x realtime)"
        )
        logger.info(
            f"  Summary: {video_result.summary.get('deleted_segments', 0)} deleted, "
            f"{video_result.summary.get('selected_segments', 0)} selected, "
            f"{video_result.summary.get('neutral_segments', 0)} neutral"
        )

        return video_result

    def _analyze_segment(
        self,
        index: int,
        start: float,
        end: float,
        frames: list[np.ndarray],
        audio_data: Optional[np.ndarray],
    ) -> SegmentResult:
        """Run all analyzers on a single segment and score it."""
        # ─── Video analyzers ───
        shake = self.shake_analyzer.analyze(frames)
        exposure = self.exposure_analyzer.analyze(frames)
        focus = self.focus_analyzer.analyze(frames)
        obstruction = self.obstruction_analyzer.analyze(frames)
        face = self.face_analyzer.analyze(frames)
        gaze = self.gaze_analyzer.analyze(frames, face)
        motion = self.motion_analyzer.analyze(frames)

        # ─── Audio analyzers ───
        if audio_data is not None and len(audio_data) > 0:
            audio_slice = slice_audio(audio_data, self.audio_sr, start, end)
            silence = self.silence_analyzer.analyze(audio_slice, self.audio_sr)
            laughter = self.laughter_analyzer.analyze(audio_slice, self.audio_sr)
            voice = self.voice_analyzer.analyze(audio_slice, self.audio_sr)
        else:
            silence = SilenceMetrics(rms_db=-100.0, silent_window_pct=1.0)
            laughter = LaughterMetrics(
                laugh_confidence=0.0, laugh_class="",
                laugh_windows=0, total_windows=0,
            )
            voice = VoiceMetrics(
                voice_activity_pct=0.0, transcript="",
                has_exclamation=False, has_baby_name=False,
                is_babble=False, whisper_confidence=0.0,
            )

        # ─── Score ───
        score = self.scorer.score_segment(
            shake, silence, exposure, focus, obstruction,
            face, laughter, voice, motion, gaze,
        )

        # Build result
        result = self.aggregator.build_segment_result(
            index, start, end, score,
            shake, silence, exposure, focus, obstruction,
            face, laughter, voice, motion, gaze,
        )

        # Store cry detection flag for cross-segment cry→laugh rule
        result.metrics["_cry_detected"] = laughter.cry_detected

        return result

    def analyze_folder(self, folder_path: str) -> dict[str, Any]:
        """Analyze all video files in a folder.

        Returns the complete score_matrix dict.
        """
        video_files = find_video_files(folder_path)
        if not video_files:
            logger.error(f"No video files found in {folder_path}")
            return self.aggregator.build_score_matrix([])

        video_results: list[VideoResult] = []
        for vf in video_files:
            result = self.analyze_video(vf)
            if result is not None:
                video_results.append(result)

        return self.aggregator.build_score_matrix(video_results)

    def analyze_single(self, video_path: str) -> dict[str, Any]:
        """Analyze a single video file.

        Returns the complete score_matrix dict.
        """
        result = self.analyze_video(video_path)
        results = [result] if result else []
        return self.aggregator.build_score_matrix(results)

    def close(self) -> None:
        """Release resources held by analyzers."""
        self.face_analyzer.close()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Baby Video Auto-Editor — Phase 1 Score Matrix Generator",
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to video file or folder of videos",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory for score_matrix.json (default: baby_editor/output/)",
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to config.yaml (default: baby_editor/config.yaml)",
    )
    args = parser.parse_args()

    # Load config and setup logging
    config = load_config(args.config)
    setup_logging(config)
    logger.info("=" * 60)
    logger.info("Baby Video Auto-Editor — Phase 1")
    logger.info("=" * 60)

    # Check GPU
    has_gpu = check_cuda()
    if has_gpu:
        logger.info("GPU acceleration: ENABLED")
    else:
        logger.info("GPU acceleration: DISABLED (CPU mode)")

    # Determine input type
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {args.input}")
        sys.exit(1)

    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run pipeline
    pipeline = AnalysisPipeline(config)
    t_start = time.time()

    try:
        if input_path.is_dir():
            score_matrix = pipeline.analyze_folder(str(input_path))
        elif input_path.is_file():
            score_matrix = pipeline.analyze_single(str(input_path))
        else:
            logger.error(f"Input is neither a file nor directory: {args.input}")
            sys.exit(1)
    finally:
        pipeline.close()

    # Write output
    output_file = output_dir / "score_matrix.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(score_matrix, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t_start
    n_videos = score_matrix.get("daily_summary", {}).get("total_videos", 0)
    n_segments = score_matrix.get("daily_summary", {}).get("total_segments", 0)

    logger.info("=" * 60)
    logger.info(f"Analysis complete: {n_videos} videos, {n_segments} segments")
    logger.info(f"Total time: {elapsed:.1f}s")
    logger.info(f"Output: {output_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
