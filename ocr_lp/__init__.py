"""OCR license-plate utilities package."""

from __future__ import annotations

import os


def _configure_native_threads() -> None:
    """Keep native math libraries modest in constrained environments."""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "1")


_configure_native_threads()

__version__ = "0.1"
