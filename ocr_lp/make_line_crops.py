"""Create recognizer line crops and a line-level training manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from PIL import Image

from .splitter import resize_line_crop, save_debug_split, split_two_line_image


LINE_FIELDNAMES = [
    "line_image_path",
    "image_path",
    "source",
    "split",
    "layout",
    "line_index",
    "line_role",
    "line_label",
    "plate_label",
    "split_confidence",
    "low_confidence_split",
]


def read_metadata(path: str) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def resolve_image_path(data_root: Path, row: Dict[str, str]) -> Optional[Path]:
    image_path = Path(row["image_path"])
    candidates = [
        data_root / image_path,
        data_root / row.get("source", "") / image_path.name,
        data_root / image_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def safe_base_name(row: Dict[str, str]) -> str:
    stem = Path(row["image_path"]).stem
    source = row.get("source", "image")
    return f"{source}_{stem}".replace("/", "_")


def write_line_manifest(rows: Iterable[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LINE_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in LINE_FIELDNAMES})


def _line_row(
    row: Dict[str, str],
    file_name: str,
    line_index: int,
    line_role: str,
    line_label: str,
    split_confidence: str = "",
    low_confidence_split: str = "",
) -> Dict[str, str]:
    return {
        "line_image_path": file_name,
        "image_path": row["image_path"],
        "source": row.get("source", ""),
        "split": row.get("split", "train"),
        "layout": row.get("layout", "one_line"),
        "line_index": str(line_index),
        "line_role": line_role,
        "line_label": line_label,
        "plate_label": row.get("canonical_label", ""),
        "split_confidence": split_confidence,
        "low_confidence_split": low_confidence_split,
    }


def make_line_crops(
    metadata_csv: str,
    data_root: str,
    out_dir: str,
    manifest_name: str = "line_metadata.csv",
    debug_dir: str = "outputs/debug_splits",
    line_height: int = 48,
    max_width: int = 320,
) -> int:
    metadata = read_metadata(metadata_csv)
    data_root_path = Path(data_root)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    debug_path = Path(debug_dir) if debug_dir else None
    line_rows: List[Dict[str, str]] = []

    for row in metadata:
        image_path = resolve_image_path(data_root_path, row)
        if image_path is None:
            continue
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            base = safe_base_name(row)
            layout = row.get("layout", "one_line")
            if layout == "two_line":
                result = split_two_line_image(image)
                top = resize_line_crop(result.top, height=line_height, max_width=max_width)
                bottom = resize_line_crop(result.bottom, height=line_height, max_width=max_width)
                top_name = f"{base}_top.jpg"
                bottom_name = f"{base}_bottom.jpg"
                top.save(out_path / top_name)
                bottom.save(out_path / bottom_name)
                save_debug_split(
                    result,
                    None if debug_path is None else debug_path / f"{base}_split.jpg",
                )
                confidence = f"{result.confidence:.4f}"
                low_conf = "1" if result.low_confidence else "0"
                line_rows.append(
                    _line_row(
                        row,
                        top_name,
                        0,
                        "top",
                        row.get("top_label", ""),
                        confidence,
                        low_conf,
                    )
                )
                line_rows.append(
                    _line_row(
                        row,
                        bottom_name,
                        1,
                        "bottom",
                        row.get("bottom_label", ""),
                        confidence,
                        low_conf,
                    )
                )
            else:
                line = resize_line_crop(image, height=line_height, max_width=max_width)
                line_name = f"{base}_line0.jpg"
                line.save(out_path / line_name)
                line_rows.append(
                    _line_row(row, line_name, 0, "full", row.get("canonical_label", ""))
                )

    write_line_manifest(line_rows, out_path / manifest_name)
    return len(line_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create OCR line crops from metadata.")
    parser.add_argument("--metadata", default="data/metadata.csv")
    parser.add_argument("--data-root", default="archive")
    parser.add_argument("--out", default="data/line_crops")
    parser.add_argument("--manifest-name", default="line_metadata.csv")
    parser.add_argument("--debug-out", default="outputs/debug_splits")
    parser.add_argument("--line-height", type=int, default=48)
    parser.add_argument("--max-width", type=int, default=320)
    args = parser.parse_args()

    n = make_line_crops(
        args.metadata,
        args.data_root,
        args.out,
        manifest_name=args.manifest_name,
        debug_dir=args.debug_out,
        line_height=args.line_height,
        max_width=args.max_width,
    )
    print(f"Created {n} line crops -> {args.out}")


if __name__ == "__main__":
    main()
