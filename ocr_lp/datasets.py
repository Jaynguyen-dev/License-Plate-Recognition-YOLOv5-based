"""Dataset and tensor utilities for the OCR pipeline."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .constants import LAYOUT_TO_IDX
from .ctc import encode_label
from .torch_utils import require_torch


def read_csv(path: str) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def filter_rows(rows: Sequence[Dict[str, str]], split: Optional[str] = None) -> List[Dict[str, str]]:
    if split is None:
        return list(rows)
    return [row for row in rows if row.get("split", "train") == split]


def resolve_path(root: Path, path_value: str, fallback_parent: Optional[Path] = None) -> Path:
    path = Path(path_value)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    if fallback_parent is not None:
        candidates.append(fallback_parent / path)
    candidates.append(root / path)
    candidates.append(root / path.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def pil_to_tensor(image: Image.Image, channels: int = 1):
    torch = require_torch()
    if channels == 1:
        arr = np.asarray(image.convert("L"), dtype=np.float32)[None, :, :] / 255.0
    else:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy(arr)


def _perspective_coeffs(dst_points, src_points):
    matrix = []
    vector = []
    for (x, y), (u, v) in zip(dst_points, src_points):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        vector.extend([u, v])
    coeffs, *_ = np.linalg.lstsq(np.asarray(matrix, dtype=float), np.asarray(vector, dtype=float), rcond=None)
    return tuple(float(c) for c in coeffs)


def _perspective_jitter(image: Image.Image, max_frac: float = 0.04) -> Image.Image:
    width, height = image.size
    dx = width * max_frac
    dy = height * max_frac
    src = [(0, 0), (width, 0), (width, height), (0, height)]
    dst = [
        (random.uniform(0, dx), random.uniform(0, dy)),
        (width - random.uniform(0, dx), random.uniform(0, dy)),
        (width - random.uniform(0, dx), height - random.uniform(0, dy)),
        (random.uniform(0, dx), height - random.uniform(0, dy)),
    ]
    coeffs = _perspective_coeffs(dst, src)
    return image.transform(
        image.size,
        Image.Transform.PERSPECTIVE,
        coeffs,
        resample=Image.Resampling.BICUBIC,
        fillcolor=(255, 255, 255),
    )


def augment_layout_image(image: Image.Image) -> Image.Image:
    if random.random() < 0.7:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.75, 1.25))
    if random.random() < 0.7:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.75, 1.35))
    if random.random() < 0.25:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.9)))
    if random.random() < 0.5:
        image = image.rotate(random.uniform(-4.0, 4.0), resample=Image.Resampling.BICUBIC, fillcolor=(255, 255, 255))
    if random.random() < 0.35:
        image = _perspective_jitter(image)
    return image


class LayoutDataset:
    def __init__(
        self,
        metadata_csv: str,
        data_root: str = "archive",
        split: Optional[str] = "train",
        image_size: Sequence[int] = (192, 96),
        augment: bool = False,
    ):
        self.metadata_csv = metadata_csv
        self.metadata_parent = Path(metadata_csv).parent
        self.data_root = Path(data_root)
        self.rows = filter_rows(read_csv(metadata_csv), split=split)
        self.image_size = tuple(image_size)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        torch = require_torch()
        row = self.rows[idx]
        image_path = resolve_path(self.data_root, row["image_path"], self.metadata_parent)
        with Image.open(image_path) as image:
            image = image.convert("RGB").resize(self.image_size, Image.Resampling.BILINEAR)
            if self.augment:
                image = augment_layout_image(image)
            tensor = pil_to_tensor(image, channels=3)
        label = torch.tensor(LAYOUT_TO_IDX[row.get("layout", "one_line")], dtype=torch.long)
        return tensor, label


class LineCropDataset:
    def __init__(
        self,
        line_metadata_csv: str,
        split: Optional[str] = "train",
        image_height: int = 48,
        max_width: int = 320,
    ):
        self.line_metadata_csv = line_metadata_csv
        self.manifest_parent = Path(line_metadata_csv).parent
        self.rows = filter_rows(read_csv(line_metadata_csv), split=split)
        self.image_height = image_height
        self.max_width = max_width

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.rows[idx]
        path = resolve_path(self.manifest_parent, row["line_image_path"], self.manifest_parent)
        with Image.open(path) as image:
            image = image.convert("L")
            if image.height != self.image_height:
                width = max(1, int(round(image.width * (self.image_height / float(image.height)))))
                image = image.resize((min(width, self.max_width), self.image_height), Image.Resampling.BILINEAR)
            elif image.width > self.max_width:
                image = image.resize((self.max_width, self.image_height), Image.Resampling.BILINEAR)
            tensor = pil_to_tensor(image, channels=1)
        return {
            "image": tensor,
            "width": tensor.shape[-1],
            "label": row.get("line_label", ""),
            "target": encode_label(row.get("line_label", "")),
            "row": row,
        }


def line_collate(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    torch = require_torch()
    max_width = max(int(item["width"]) for item in batch)
    height = int(batch[0]["image"].shape[-2])
    images = torch.ones((len(batch), 1, height, max_width), dtype=torch.float32)
    widths = []
    targets = []
    target_lengths = []
    labels = []
    rows = []
    for idx, item in enumerate(batch):
        image = item["image"]
        width = int(item["width"])
        images[idx, :, :, :width] = image
        widths.append(width)
        target = list(item["target"])
        targets.extend(target)
        target_lengths.append(len(target))
        labels.append(item["label"])
        rows.append(item["row"])
    return {
        "images": images,
        "widths": torch.tensor(widths, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
        "labels": labels,
        "rows": rows,
    }


def class_weights(rows: Iterable[Dict[str, str]]):
    torch = require_torch()
    counts = [0, 0]
    for row in rows:
        counts[LAYOUT_TO_IDX[row.get("layout", "one_line")]] += 1
    total = sum(counts)
    weights = [0.0, 0.0]
    for idx, count in enumerate(counts):
        weights[idx] = total / max(1, count)
    return torch.tensor(weights, dtype=torch.float32)
