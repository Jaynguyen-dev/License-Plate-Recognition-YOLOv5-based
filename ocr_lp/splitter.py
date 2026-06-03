"""Heuristic splitting for two-line license plates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class LineSplitResult:
    top: Image.Image
    bottom: Image.Image
    split_y: int
    confidence: float
    low_confidence: bool
    resized: Image.Image


def resize_to_height(image: Image.Image, height: int = 96) -> Image.Image:
    width, old_height = image.size
    if old_height <= 0:
        return image
    new_width = max(1, int(round(width * (height / float(old_height)))))
    return image.resize((new_width, height), Image.Resampling.LANCZOS)


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values.astype(float)
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values.astype(float), kernel, mode="same")


def foreground_mask(gray: np.ndarray) -> np.ndarray:
    gray_f = gray.astype(float)
    gy = np.zeros_like(gray_f)
    gx = np.zeros_like(gray_f)
    gy[1:, :] = np.abs(np.diff(gray_f, axis=0))
    gx[:, 1:] = np.abs(np.diff(gray_f, axis=1))
    edge = gx + gy
    edge_threshold = max(15.0, float(np.percentile(edge, 85)))
    dark_threshold = min(220.0, float(np.percentile(gray_f, 35)) + 10.0)
    return (edge >= edge_threshold) | (gray_f <= dark_threshold)


def trim_empty_rows(image: Image.Image, pad: int = 2) -> Image.Image:
    gray = np.asarray(image.convert("L"))
    mask = foreground_mask(gray)
    projection = mask.mean(axis=1)
    rows = np.where(projection > 0.02)[0]
    if rows.size == 0:
        return image
    top = max(0, int(rows[0]) - pad)
    bottom = min(image.height, int(rows[-1]) + 1 + pad)
    if bottom <= top:
        return image
    return image.crop((0, top, image.width, bottom))


def trim_border_margins(image: Image.Image, pad: int = 2) -> Tuple[Image.Image, Tuple[int, int]]:
    gray = np.asarray(image.convert("L"))
    mask = foreground_mask(gray)
    rows = np.where(mask.mean(axis=1) > 0.02)[0]
    if rows.size == 0:
        return image, (0, image.height)
    top = max(0, int(rows[0]) - pad)
    bottom = min(image.height, int(rows[-1]) + 1 + pad)
    if bottom <= top:
        return image, (0, image.height)
    return image.crop((0, top, image.width, bottom)), (top, bottom)


def find_split_y(gray: np.ndarray, min_band: float = 0.35, max_band: float = 0.65) -> Tuple[int, float]:
    mask = foreground_mask(gray)
    projection = _smooth(mask.mean(axis=1), window=max(3, gray.shape[0] // 24))
    height = gray.shape[0]
    lo = int(height * min_band)
    hi = int(height * max_band)
    if lo >= hi:
        return height // 2, 0.0

    band = projection[lo:hi]
    if band.size == 0:
        return height // 2, 0.0

    idx = int(np.argmin(band)) + lo
    valley = float(projection[idx])
    shoulder_top = float(np.max(projection[max(0, lo // 2) : idx + 1])) if idx > 0 else 0.0
    shoulder_bottom = float(np.max(projection[idx: min(height, hi + (height - hi) // 2)]))
    shoulder = max(shoulder_top, shoulder_bottom, 1e-6)
    confidence = max(0.0, min(1.0, (shoulder - valley) / shoulder))
    return idx, confidence


def split_two_line_image(
    image: Image.Image,
    resize_height: int = 96,
    confidence_threshold: float = 0.12,
    padding: int = 2,
) -> LineSplitResult:
    resized = resize_to_height(image.convert("RGB"), resize_height)
    trimmed, (trim_top, _) = trim_border_margins(resized, pad=padding)
    gray = np.asarray(trimmed.convert("L"))
    split_y_trimmed, confidence = find_split_y(gray)
    low_confidence = confidence < confidence_threshold
    if low_confidence:
        split_y_trimmed = trimmed.height // 2

    top = trimmed.crop((0, 0, trimmed.width, split_y_trimmed))
    bottom = trimmed.crop((0, split_y_trimmed, trimmed.width, trimmed.height))
    top = trim_empty_rows(top, pad=padding)
    bottom = trim_empty_rows(bottom, pad=padding)
    split_y_resized = trim_top + split_y_trimmed
    return LineSplitResult(
        top=top,
        bottom=bottom,
        split_y=split_y_resized,
        confidence=confidence,
        low_confidence=low_confidence,
        resized=resized,
    )


def resize_line_crop(image: Image.Image, height: int = 48, max_width: int = 320) -> Image.Image:
    image = image.convert("L")
    if image.height <= 0:
        return Image.new("L", (1, height), color=255)
    width = max(1, int(round(image.width * (height / float(image.height)))))
    width = min(width, max_width)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def save_debug_split(result: LineSplitResult, out_path: Optional[Path]) -> None:
    if out_path is None:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    debug = result.resized.copy()
    draw = ImageDraw.Draw(debug)
    color = (255, 0, 0) if result.low_confidence else (0, 170, 0)
    draw.line((0, result.split_y, debug.width, result.split_y), fill=color, width=2)
    debug.save(out_path)
