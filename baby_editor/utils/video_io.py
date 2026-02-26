"""Video I/O utilities: reading, metadata extraction, frame sampling."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VideoMeta:
    """Metadata extracted from a video file."""
    filename: str
    path: str
    duration: float
    width: int
    height: int
    fps: float
    total_frames: int
    created_at: str

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"


@dataclass
class Segment:
    """A 2-second segment of video frames."""
    index: int
    start: float
    end: float
    frames: list[np.ndarray]
    frame_indices: list[int]


def get_video_meta(video_path: str) -> Optional[VideoMeta]:
    """Extract metadata from a video file.

    Returns None if the file cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"Cannot open video: {video_path}")
        return None

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0.0

        # Try to get creation time from file metadata
        path = Path(video_path)
        try:
            stat = path.stat()
            created_at = datetime.fromtimestamp(stat.st_mtime).isoformat()
        except OSError:
            created_at = datetime.now().isoformat()

        return VideoMeta(
            filename=path.name,
            path=str(path.resolve()),
            duration=round(duration, 1),
            width=width,
            height=height,
            fps=round(fps, 2),
            total_frames=total_frames,
            created_at=created_at,
        )
    finally:
        cap.release()


def iter_segments(
    video_path: str,
    segment_duration: float = 2.0,
    sample_rate: int = 3,
) -> Iterator[Segment]:
    """Iterate over video segments, yielding sampled frames per segment.

    Args:
        video_path: Path to video file.
        segment_duration: Duration of each segment in seconds.
        sample_rate: Number of frames to sample per segment.

    Yields:
        Segment objects with sampled frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video for segment iteration: {video_path}")
        return

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total_frames <= 0:
            logger.error(f"Invalid video properties: fps={fps}, frames={total_frames}")
            return

        frames_per_segment = int(fps * segment_duration)
        num_segments = int(total_frames / frames_per_segment)

        for seg_idx in range(num_segments):
            start_frame = seg_idx * frames_per_segment
            end_frame = start_frame + frames_per_segment

            # Calculate which frames to sample (evenly spaced)
            if sample_rate >= frames_per_segment:
                sample_indices = list(range(start_frame, end_frame))
            else:
                step = frames_per_segment / sample_rate
                sample_indices = [
                    int(start_frame + i * step)
                    for i in range(sample_rate)
                ]

            frames = []
            frame_indices = []
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    frames.append(frame)
                    frame_indices.append(idx)
                else:
                    logger.debug(f"Failed to read frame {idx} in {video_path}")

            if not frames:
                logger.warning(
                    f"No frames read for segment {seg_idx} "
                    f"(frames {start_frame}-{end_frame})"
                )
                continue

            yield Segment(
                index=seg_idx,
                start=round(seg_idx * segment_duration, 1),
                end=round((seg_idx + 1) * segment_duration, 1),
                frames=frames,
                frame_indices=frame_indices,
            )
    finally:
        cap.release()


def read_frames_range(
    video_path: str,
    start_frame: int,
    end_frame: int,
    step: int = 1,
) -> list[np.ndarray]:
    """Read a range of frames from a video file.

    Args:
        video_path: Path to video file.
        start_frame: First frame index (inclusive).
        end_frame: Last frame index (exclusive).
        step: Read every Nth frame.

    Returns:
        List of BGR frames as numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    frames = []
    try:
        for idx in range(start_frame, end_frame, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)
    finally:
        cap.release()
    return frames


def find_video_files(folder_path: str) -> list[str]:
    """Find all video files (MP4/MOV) in a folder.

    Returns sorted list of absolute paths.
    """
    extensions = {".mp4", ".mov", ".MP4", ".MOV"}
    folder = Path(folder_path)
    if not folder.is_dir():
        logger.error(f"Not a directory: {folder_path}")
        return []

    videos = [
        str(f.resolve())
        for f in folder.iterdir()
        if f.is_file() and f.suffix in extensions
    ]
    videos.sort()
    logger.info(f"Found {len(videos)} video files in {folder_path}")
    return videos
