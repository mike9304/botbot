"""Audio I/O utilities: extraction from video, loading, slicing."""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def extract_audio(
    video_path: str,
    output_path: Optional[str] = None,
    sample_rate: int = 16000,
    mono: bool = True,
) -> Optional[str]:
    """Extract audio from video file using ffmpeg.

    Args:
        video_path: Path to source video.
        output_path: Where to save WAV. If None, uses a temp file.
        sample_rate: Target sample rate in Hz.
        mono: Convert to mono if True.

    Returns:
        Path to extracted WAV file, or None on failure.
    """
    if output_path is None:
        stem = Path(video_path).stem
        output_path = str(
            Path(tempfile.gettempdir()) / f"{stem}_audio.wav"
        )

    channels = "1" if mono else "2"
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",                          # no video
        "-acodec", "pcm_s16le",         # 16-bit PCM
        "-ar", str(sample_rate),        # sample rate
        "-ac", channels,                # channels
        output_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            # Check if video simply has no audio track
            if "does not contain any stream" in result.stderr:
                logger.info(f"No audio track in {video_path}")
                return None
            logger.warning(
                f"ffmpeg audio extraction failed for {video_path}: "
                f"{result.stderr[:200]}"
            )
            return None
        logger.debug(f"Audio extracted: {output_path}")
        return output_path
    except FileNotFoundError:
        logger.error("ffmpeg not found. Install ffmpeg to enable audio analysis.")
        return None
    except subprocess.TimeoutExpired:
        logger.error(f"ffmpeg timed out extracting audio from {video_path}")
        return None


def load_audio(
    audio_path: str,
    sample_rate: int = 16000,
) -> tuple[Optional[np.ndarray], int]:
    """Load audio file as numpy array using librosa.

    Args:
        audio_path: Path to WAV file.
        sample_rate: Target sample rate.

    Returns:
        Tuple of (audio_array, sample_rate). audio_array is None on failure.
    """
    try:
        import librosa
        audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
        return audio, sr
    except Exception as e:
        logger.error(f"Failed to load audio {audio_path}: {e}")
        return None, sample_rate


def slice_audio(
    audio: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
) -> np.ndarray:
    """Slice audio array by time range.

    Args:
        audio: Full audio array.
        sr: Sample rate.
        start_sec: Start time in seconds.
        end_sec: End time in seconds.

    Returns:
        Sliced audio array.
    """
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    # Clamp to audio length
    start_sample = max(0, start_sample)
    end_sample = min(len(audio), end_sample)
    return audio[start_sample:end_sample]


def has_audio_track(video_path: str) -> bool:
    """Check if a video file contains an audio track using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_type",
        "-of", "csv=p=0",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return "audio" in result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
