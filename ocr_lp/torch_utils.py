"""Small helpers around optional PyTorch imports."""

from __future__ import annotations

import os
import warnings


def _thread_count() -> int:
    try:
        return max(1, int(os.environ.get("TORCH_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS") or "1"))
    except ValueError:
        return 1


def require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required for model training/inference. Install the "
            "dependencies from requirements.txt in an environment with torch."
        ) from exc
    threads = _thread_count()
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        pass
    return torch


def pick_device(prefer: str = "auto") -> str:
    torch = require_torch()
    if prefer != "auto":
        return prefer
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="CUDA initialization:*")
        return "cuda" if torch.cuda.is_available() else "cpu"
