from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    script = Path(__file__).resolve().parent / "pipeline" / "infer_full_vehicle.py"
    if not script.exists():
        raise SystemExit(f"Cannot find pipeline inference script: {script}")

    # Run the real inference script from inside pipeline/ so its default model
    # paths resolve to pipeline/best.pth, pipeline/best_layout.pt, and
    # pipeline/best_recognition.pt.
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
