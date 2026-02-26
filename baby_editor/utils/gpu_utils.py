"""GPU utilities: CUDA availability checks and memory management."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

logger = logging.getLogger(__name__)


def check_cuda() -> bool:
    """Check if CUDA is available via OpenCV or PyTorch."""
    cuda_available = False

    # Check OpenCV CUDA
    try:
        import cv2
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            logger.info(f"OpenCV CUDA: {cv2.cuda.getCudaEnabledDeviceCount()} device(s)")
            cuda_available = True
        else:
            logger.info("OpenCV CUDA: not available")
    except Exception:
        logger.info("OpenCV CUDA: not compiled with CUDA support")

    # Check PyTorch CUDA (used by Whisper)
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            mem_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
            logger.info(f"PyTorch CUDA: {device_name} ({mem_gb:.1f} GB)")
            cuda_available = True
        else:
            logger.info("PyTorch CUDA: not available")
    except ImportError:
        logger.info("PyTorch: not installed")

    return cuda_available


def get_gpu_memory_mb() -> float:
    """Return available GPU memory in MB, or 0 if unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            return free / (1024 ** 2)
    except (ImportError, RuntimeError):
        pass
    return 0.0


@contextmanager
def gpu_fallback(analyzer_name: str) -> Generator[str, None, None]:
    """Context manager that catches GPU OOM and falls back to CPU.

    Yields the device string ("cuda" or "cpu").

    Usage:
        with gpu_fallback("FaceAnalyzer") as device:
            model.to(device)
            result = model(input.to(device))
    """
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except ImportError:
        pass

    try:
        yield device
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            logger.warning(
                f"{analyzer_name}: GPU OOM, falling back to CPU. "
                f"Error: {e}"
            )
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            yield "cpu"
        else:
            raise
