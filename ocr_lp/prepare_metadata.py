"""Prepare the canonical metadata table used by training and evaluation."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

from .label_utils import (
    canonicalize_label,
    source_family,
    split_key_from_name,
    split_two_line_canonical,
)


PLAN_COLUMNS = [
    "image_path",
    "source",
    "raw_label",
    "canonical_label",
    "layout",
    "top_label",
    "bottom_label",
    "type",
    "split_key",
]
EXTRA_COLUMNS = ["split", "source_family", "width", "height"]
FIELDNAMES = PLAN_COLUMNS + EXTRA_COLUMNS


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def find_image_path(data_root: Path, source: str, name: str) -> Optional[Path]:
    name_path = Path(name)
    candidates = []
    if name_path.parts and name_path.parts[0] in {"cropped", "generated"}:
        candidates.append(data_root / name_path)
    candidates.extend(
        [
            data_root / source / name_path.name,
            data_root / "cropped" / name_path.name,
            data_root / "generated" / name_path.name,
            data_root / name_path,
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def image_geometry(path: Optional[Path]) -> Tuple[Optional[int], Optional[int]]:
    if path is None:
        return None, None
    try:
        with Image.open(path) as im:
            return im.size
    except OSError:
        return None, None


def derive_layout(width: Optional[int], height: Optional[int], threshold: float = 2.6) -> str:
    if not width or not height:
        return "one_line"
    return "two_line" if width / float(height) <= threshold else "one_line"


def _relative_image_path(data_root: Path, image_path: Optional[Path], fallback_name: str) -> str:
    if image_path is None:
        return fallback_name
    try:
        return image_path.relative_to(data_root).as_posix()
    except ValueError:
        return image_path.as_posix()


def _rows_from_label_file(
    data_root: Path,
    csv_path: Path,
    source: str,
    layout_threshold: float,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for row in read_csv_rows(csv_path):
        name = (row.get("Name") or row.get("name") or "").strip()
        if not name:
            continue
        raw_label = (row.get("Label") or row.get("label") or "").strip()
        typ = (row.get("Type") or row.get("type") or "").strip()
        img_path = find_image_path(data_root, source, name)
        width, height = image_geometry(img_path)
        layout = derive_layout(width, height, threshold=layout_threshold)
        canonical = canonicalize_label(raw_label)
        top_label, bottom_label = canonical, ""
        if layout == "two_line":
            top_label, bottom_label = split_two_line_canonical(raw_label)
        rel_path = _relative_image_path(data_root, img_path, name)
        rows.append(
            {
                "image_path": rel_path,
                "source": source,
                "raw_label": raw_label,
                "canonical_label": canonical,
                "layout": layout,
                "top_label": top_label,
                "bottom_label": bottom_label,
                "type": typ,
                "split_key": split_key_from_name(name),
                "split": "",
                "source_family": source_family(source, name),
                "width": "" if width is None else str(width),
                "height": "" if height is None else str(height),
            }
        )
    return rows


def _split_counts(n: int) -> Tuple[int, int, int]:
    if n <= 1:
        return n, 0, 0
    if n == 2:
        return 1, 1, 0
    n_val = max(1, round(n * 0.1))
    n_test = max(1, round(n * 0.1))
    while n - n_val - n_test < 1:
        if n_val >= n_test and n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1
        else:
            break
    return n - n_val - n_test, n_val, n_test


def assign_splits(rows: Sequence[Dict[str, str]], seed: int = 1337) -> List[Dict[str, str]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["split_key"]].append(row)

    strata: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for split_key, group_rows in grouped.items():
        first = group_rows[0]
        strata[(first["layout"], first["source_family"])].append(split_key)

    rng = random.Random(seed)
    split_by_key: Dict[str, str] = {}
    for keys in strata.values():
        keys = sorted(keys)
        rng.shuffle(keys)
        n_train, n_val, n_test = _split_counts(len(keys))
        test_keys = set(keys[:n_test])
        val_keys = set(keys[n_test : n_test + n_val])
        for key in keys:
            if key in test_keys:
                split_by_key[key] = "test"
            elif key in val_keys:
                split_by_key[key] = "val"
            else:
                split_by_key[key] = "train"

    out: List[Dict[str, str]] = []
    for row in rows:
        row = dict(row)
        row["split"] = split_by_key[row["split_key"]]
        out.append(row)
    return out


def write_metadata(rows: Iterable[Dict[str, str]], out_csv: str) -> None:
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDNAMES})


def build_metadata(
    data_root: str,
    out_csv: str,
    seed: int = 1337,
    layout_threshold: float = 2.6,
) -> List[Dict[str, str]]:
    data_root_path = Path(data_root)
    labels_dir = data_root_path / "labels"
    label_files = [
        ("cropped", labels_dir / "crop_labels.csv"),
        ("generated", labels_dir / "gen_labels.csv"),
    ]

    rows: List[Dict[str, str]] = []
    for source, path in label_files:
        if path.exists():
            rows.extend(_rows_from_label_file(data_root_path, path, source, layout_threshold))

    rows = assign_splits(rows, seed=seed)
    write_metadata(rows, out_csv)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OCR metadata from archive labels.")
    parser.add_argument("--data-root", default="archive", help="Path to archive folder")
    parser.add_argument("--out", default="data/metadata.csv", help="Output metadata CSV")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--layout-threshold", type=float, default=2.6)
    args = parser.parse_args()

    rows = build_metadata(
        args.data_root,
        args.out,
        seed=args.seed,
        layout_threshold=args.layout_threshold,
    )
    print(f"Wrote metadata rows={len(rows)} -> {args.out}")


if __name__ == "__main__":
    main()
