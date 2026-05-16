"""
Crop detections from batch_inference.py results for OCR.

Default usage:
    python crop_from_inference_results.py \
        --detections inference_outputs/detections.json \
        --images-root LP_detection/images/val
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop OCR-ready license plate images from detections.json."
    )
    parser.add_argument(
        "--detections",
        default=Path("inference_outputs/detections.json"),
        type=Path,
        help="Path to detections.json produced by batch_inference.py.",
    )
    parser.add_argument(
        "--images-root",
        required=True,
        type=Path,
        help="Root folder used as --input for batch_inference.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("inference_outputs/crops"),
        type=Path,
        help="Folder where cropped detection images will be saved.",
    )
    parser.add_argument(
        "--index-file",
        default=Path("inference_outputs/crops.csv"),
        type=Path,
        help="CSV file mapping crops back to source images and boxes.",
    )
    parser.add_argument(
        "--padding",
        default=0,
        type=int,
        help="Extra pixels to include around each detected box.",
    )
    parser.add_argument(
        "--min-score",
        default=None,
        type=float,
        help="Optional score threshold applied while cropping.",
    )
    parser.add_argument(
        "--image-format",
        default="jpg",
        choices=["jpg", "png"],
        help="Output crop image format.",
    )
    return parser.parse_args()


def load_detections(detections_path: Path) -> List[Dict]:
    detections_path = detections_path.expanduser().resolve()
    if not detections_path.is_file():
        raise FileNotFoundError(f"Detections file not found: {detections_path}")

    with detections_path.open("r", encoding="utf-8") as f:
        results = json.load(f)

    if not isinstance(results, list):
        raise ValueError("Detections JSON must contain a list of image results.")

    return results


def clamp_box(box: Sequence[float], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    left = max(0, min(width, int(round(x1))))
    top = max(0, min(height, int(round(y1))))
    right = max(0, min(width, int(round(x2))))
    bottom = max(0, min(height, int(round(y2))))
    return left, top, right, bottom


def expand_box(
    box: Sequence[float],
    padding: int,
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    padded_box = (x1 - padding, y1 - padding, x2 + padding, y2 + padding)
    return clamp_box(padded_box, image_width, image_height)


def resolve_source_image(images_root: Path, image_name: str) -> Path:
    image_path = Path(image_name)
    if image_path.is_absolute():
        return image_path
    return images_root / image_path


def crop_detections_for_image(
    result: Dict,
    images_root: Path,
    output_dir: Path,
    padding: int,
    min_score: float = None,
    image_format: str = "jpg",
) -> List[Dict]:
    source_rel = Path(result["image"])
    source_path = resolve_source_image(images_root, result["image"])
    if not source_path.is_file():
        raise FileNotFoundError(f"Source image not found: {source_path}")

    crop_dir = output_dir / source_rel.with_suffix("")
    crop_dir.mkdir(parents=True, exist_ok=True)

    crop_records = []
    with Image.open(source_path) as original_image:
        image = original_image.convert("RGB")
        width, height = image.size

        for det_idx, detection in enumerate(result.get("detections", []), start=1):
            score = float(detection["score"])
            if min_score is not None and score < min_score:
                continue

            crop_box = expand_box(detection["bbox_xyxy"], padding, width, height)
            if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
                continue

            crop_name = f"det_{det_idx:03d}.{image_format}"
            crop_path = crop_dir / crop_name
            crop = image.crop(crop_box)
            if image_format == "jpg":
                crop.save(crop_path, quality=95)
            else:
                crop.save(crop_path)

            crop_records.append(
                {
                    "crop_image": str(crop_path).replace("\\", "/"),
                    "source_image": result["image"],
                    "detection_index": det_idx,
                    "label": detection["label"],
                    "score": score,
                    "x1": crop_box[0],
                    "y1": crop_box[1],
                    "x2": crop_box[2],
                    "y2": crop_box[3],
                    "width": crop_box[2] - crop_box[0],
                    "height": crop_box[3] - crop_box[1],
                }
            )

    return crop_records


def write_crop_index(crop_records: Sequence[Dict], index_file: Path) -> None:
    index_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "crop_image",
        "source_image",
        "detection_index",
        "label",
        "score",
        "x1",
        "y1",
        "x2",
        "y2",
        "width",
        "height",
    ]

    with index_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in crop_records:
            row = dict(record)
            row["score"] = f"{record['score']:.6f}"
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    if args.padding < 0:
        raise ValueError("--padding must be 0 or greater.")

    images_root = args.images_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_detections(args.detections)
    crop_records = []
    for result in results:
        crop_records.extend(
            crop_detections_for_image(
                result,
                images_root=images_root,
                output_dir=output_dir,
                padding=args.padding,
                min_score=args.min_score,
                image_format=args.image_format,
            )
        )

    write_crop_index(crop_records, args.index_file)
    print(f"Saved {len(crop_records)} crop(s) to: {output_dir}")
    print(f"Crop index: {args.index_file}")


if __name__ == "__main__":
    main()
