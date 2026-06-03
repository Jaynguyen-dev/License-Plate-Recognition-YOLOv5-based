"""
Prepare the License Plate dataset for Faster R-CNN training.

This script converts YOLO-format annotations (class x_center y_center width height)
to COCO-format JSON annotations required by torchvision's Faster R-CNN.

YOLO format (normalized):
    class_id  x_center  y_center  width  height

COCO format (absolute pixels):
    {
        "images": [...],
        "annotations": [...],
        "categories": [...]
    }

Each annotation has: bbox [x_min, y_min, width, height], area, category_id, image_id, etc.
"""

import os
import json
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import defaultdict


def yolo_to_coco_bbox(x_center, y_center, w, h, img_width, img_height):
    """
    Convert YOLO normalized (x_center, y_center, w, h) to
    COCO absolute (x_min, y_min, width, height).
    """
    abs_w = w * img_width
    abs_h = h * img_height
    x_min = (x_center * img_width) - (abs_w / 2)
    y_min = (y_center * img_height) - (abs_h / 2)

    # Clamp to image boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    abs_w = min(abs_w, img_width - x_min)
    abs_h = min(abs_h, img_height - y_min)

    return [round(x_min, 2), round(y_min, 2), round(abs_w, 2), round(abs_h, 2)]


def parse_yolo_label(label_path):
    """Parse a YOLO label file and return list of (class_id, x_c, y_c, w, h)."""
    annotations = []
    if not os.path.exists(label_path):
        return annotations

    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            x_c, y_c, w, h = map(float, parts[1:5])
            annotations.append((class_id, x_c, y_c, w, h))
    return annotations


def convert_split(images_dir, labels_dir, output_json_path, category_mapping):
    """
    Convert an entire split (train or val) from YOLO to COCO format.

    Args:
        images_dir: Path to images folder (e.g., images/train/)
        labels_dir: Path to labels folder (e.g., labels/train/)
        output_json_path: Where to save the resulting COCO JSON
        category_mapping: Dict mapping YOLO class_id -> {"id": int, "name": str}
    """
    coco = {
        "images": [],
        "annotations": [],
        "categories": list(category_mapping.values())
    }

    image_id = 0
    annotation_id = 0
    skipped_images = 0
    empty_label_count = 0
    stats = defaultdict(int)

    # Ensure provided directories exist
    images_dir = str(images_dir)
    labels_dir = str(labels_dir)
    if not os.path.isdir(images_dir):
        print(f"ERROR: images_dir does not exist: {images_dir}")
        return coco
    if not os.path.isdir(labels_dir):
        print(f"WARNING: labels_dir does not exist (no annotations will be found): {labels_dir}")

    # Gather all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if Path(f).suffix.lower() in image_extensions
    ])

    print(f"\nProcessing {len(image_files)} images from: {images_dir}")
    print(f"Labels from: {labels_dir}")
    print(f"Output: {output_json_path}")

    for img_filename in tqdm(image_files, desc="Converting"):
        img_path = os.path.join(images_dir, img_filename)

        # Get image dimensions
        try:
            with Image.open(img_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"  WARNING: Could not open {img_filename}: {e}")
            skipped_images += 1
            continue

        # Add image entry
        image_info = {
            "id": image_id,
            "file_name": img_filename,
            "width": img_width,
            "height": img_height
        }
        coco["images"].append(image_info)

        # Parse corresponding label file
        label_filename = Path(img_filename).stem + ".txt"
        label_path = os.path.join(labels_dir, label_filename)
        yolo_annotations = parse_yolo_label(label_path)

        if not yolo_annotations:
            empty_label_count += 1
            # Still include the image (negative example) - Faster R-CNN can use these
            image_id += 1
            continue

        for class_id, x_c, y_c, w, h in yolo_annotations:
            # Map YOLO class_id to COCO category_id (Faster R-CNN reserves 0 for background)
            # So YOLO class 0 -> COCO category 1
            coco_category_id = class_id + 1

            bbox = yolo_to_coco_bbox(x_c, y_c, w, h, img_width, img_height)
            area = bbox[2] * bbox[3]  # width * height

            # Skip degenerate boxes
            if bbox[2] < 1 or bbox[3] < 1:
                stats["degenerate_boxes"] += 1
                continue

            annotation_info = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": coco_category_id,
                "bbox": bbox,
                "area": round(area, 2),
                "iscrowd": 0
            }
            coco["annotations"].append(annotation_info)
            stats[f"class_{class_id}"] += 1
            annotation_id += 1

        image_id += 1

    # Save COCO JSON
    out_path = Path(output_json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(coco, f, indent=2)

    # Print summary
    print(f"\n{'='*50}")
    print(f"Conversion Summary:")
    print(f"  Total images processed: {image_id}")
    print(f"  Images skipped (errors): {skipped_images}")
    print(f"  Images with no annotations: {empty_label_count}")
    print(f"  Total annotations: {annotation_id}")
    for key, val in sorted(stats.items()):
        print(f"  {key}: {val}")
    print(f"  Output saved to: {output_json_path}")
    print(f"{'='*50}")

    return coco


def validate_coco_json(json_path, images_dir):
    """Validate the generated COCO JSON by spot-checking a few entries."""
    if not os.path.exists(json_path):
        print(f"Validation skipped: file not found: {json_path}")
        return

    with open(json_path, 'r') as f:
        coco = json.load(f)

    print(f"\nValidation of {json_path}:")
    print(f"  Categories: {coco['categories']}")
    print(f"  Number of images: {len(coco['images'])}")
    print(f"  Number of annotations: {len(coco['annotations'])}")

    # Check a sample annotation
    if coco['annotations']:
        sample = coco['annotations'][0]
        img_info = next(i for i in coco['images'] if i['id'] == sample['image_id'])
        print(f"\n  Sample annotation:")
        print(f"    Image: {img_info['file_name']} ({img_info['width']}x{img_info['height']})")
        print(f"    Bbox (x, y, w, h): {sample['bbox']}")
        print(f"    Category ID: {sample['category_id']}")
        print(f"    Area: {sample['area']}")

        # Sanity check: bbox within image bounds
        x, y, w, h = sample['bbox']
        assert x >= 0 and y >= 0, "Bbox has negative coordinates!"
        assert x + w <= img_info['width'] + 1, "Bbox exceeds image width!"
        assert y + h <= img_info['height'] + 1, "Bbox exceeds image height!"
        print(f"    [OK] Bbox within image bounds")


def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLO dataset to COCO format for Faster R-CNN training"
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="LP_detection",
        help="Root directory of the YOLO dataset (contains images/ and labels/)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for COCO JSON files (defaults to dataset_root/annotations)"
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    # If the provided dataset_root doesn't directly contain images/, try a nested folder
    if not (dataset_root / "images").exists():
        nested = dataset_root / dataset_root.name
        if nested.exists() and (nested / "images").exists():
            print(f"Detected nested dataset layout, switching dataset root to: {nested}")
            dataset_root = nested

    # Resolve output dir: default to <dataset_root>/annotations unless user overrides
    output_dir = Path(args.output_dir) if args.output_dir else (dataset_root / "annotations")

    # Category mapping: YOLO class 0 -> COCO category 1 (background = 0 in Faster R-CNN)
    # Adjust names/IDs if you have more classes
    category_mapping = {
        0: {"id": 1, "name": "license_plate", "supercategory": "vehicle"}
    }

    # Convert train split
    print("=" * 60)
    print("CONVERTING TRAIN SPLIT")
    print("=" * 60)
    convert_split(
        images_dir=str(dataset_root / "images" / "train"),
        labels_dir=str(dataset_root / "labels" / "train"),
        output_json_path=str(output_dir / "instances_train.json"),
        category_mapping=category_mapping
    )

    # Convert val split
    print("\n" + "=" * 60)
    print("CONVERTING VALIDATION SPLIT")
    print("=" * 60)
    convert_split(
        images_dir=str(dataset_root / "images" / "val"),
        labels_dir=str(dataset_root / "labels" / "val"),
        output_json_path=str(output_dir / "instances_val.json"),
        category_mapping=category_mapping
    )

    # Validate both
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)
    validate_coco_json(
        str(output_dir / "instances_train.json"),
        str(dataset_root / "images" / "train")
    )
    validate_coco_json(
        str(output_dir / "instances_val.json"),
        str(dataset_root / "images" / "val")
    )


if __name__ == "__main__":
    main()
