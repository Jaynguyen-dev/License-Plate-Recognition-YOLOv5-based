"""Sample conversion and visualization for YOLO -> COCO annotations.

Creates a small COCO JSON containing `--sample-count` images from a split
and saves overlay images with bounding boxes so you can inspect annotations
before running a full conversion / training.

Usage examples:
  python3 sample_test_conversion.py --dataset-root LP_detection --split val --sample-count 20

"""
import argparse
import json
import os
import random
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont


def yolo_to_coco_bbox(x_center, y_center, w, h, img_width, img_height):
    abs_w = w * img_width
    abs_h = h * img_height
    x_min = (x_center * img_width) - (abs_w / 2)
    y_min = (y_center * img_height) - (abs_h / 2)
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    abs_w = min(abs_w, img_width - x_min)
    abs_h = min(abs_h, img_height - y_min)
    return [round(x_min, 2), round(y_min, 2), round(abs_w, 2), round(abs_h, 2)]


def parse_yolo_label(label_path):
    anns = []
    if not os.path.exists(label_path):
        return anns
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            x_c, y_c, w, h = map(float, parts[1:5])
            anns.append((cls, x_c, y_c, w, h))
    return anns


def draw_annotations(image, annotations, img_size, category_names=None):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for ann in annotations:
        bbox = ann['bbox']
        x, y, w, h = bbox
        x2 = x + w
        y2 = y + h
        draw.rectangle([x, y, x2, y2], outline='red', width=2)
        label = category_names.get(ann['category_id'], str(ann['category_id'])) if category_names else str(ann['category_id'])
        text_pos = (x + 3, y + 3)
        if font:
            draw.text(text_pos, label, fill='yellow', font=font)
        else:
            draw.text(text_pos, label, fill='yellow')


def main():
    parser = argparse.ArgumentParser(description='Sample and visualize YOLO -> COCO conversion')
    parser.add_argument('--dataset-root', type=str, default='LP_detection')
    parser.add_argument('--split', type=str, choices=['train', 'val'], default='val')
    parser.add_argument('--sample-count', type=int, default=10)
    parser.add_argument('--out-dir', type=str, default=None, help='Where to save overlays and small COCO JSON')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    # handle nested layout
    if not (dataset_root / 'images').exists():
        nested = dataset_root / dataset_root.name
        if nested.exists() and (nested / 'images').exists():
            dataset_root = nested

    images_dir = dataset_root / 'images' / args.split
    labels_dir = dataset_root / 'labels' / args.split
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = dataset_root / 'samples'

    if not images_dir.exists():
        print(f'ERROR: images directory does not exist: {images_dir}')
        return

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = sorted([f for f in os.listdir(images_dir) if Path(f).suffix.lower() in image_extensions])
    if not image_files:
        print(f'No images found in: {images_dir}')
        return

    random.seed(args.seed)
    samples = random.sample(image_files, min(args.sample_count, len(image_files)))

    out_dir = out_dir / args.split
    overlays_dir = out_dir / 'overlays'
    overlays_dir.mkdir(parents=True, exist_ok=True)

    coco = {'images': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'license_plate', 'supercategory': 'vehicle'}]}
    ann_id = 0
    img_id = 0

    category_names = {1: 'license_plate'}

    for fname in samples:
        img_path = images_dir / fname
        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
                overlay = img.convert('RGB')
        except Exception as e:
            print(f'Could not open image {img_path}: {e}')
            continue

        label_path = labels_dir / (Path(fname).stem + '.txt')
        yolo_anns = parse_yolo_label(str(label_path))

        anns_for_image = []
        for cls, x_c, y_c, w, h in yolo_anns:
            coco_cat_id = cls + 1
            bbox = yolo_to_coco_bbox(x_c, y_c, w, h, img_w, img_h)
            area = bbox[2] * bbox[3]
            if bbox[2] < 1 or bbox[3] < 1:
                continue
            ann = {
                'id': ann_id,
                'image_id': img_id,
                'category_id': coco_cat_id,
                'bbox': bbox,
                'area': round(area, 2),
                'iscrowd': 0
            }
            coco['annotations'].append(ann)
            anns_for_image.append(ann)
            ann_id += 1

        image_info = {'id': img_id, 'file_name': fname, 'width': img_w, 'height': img_h}
        coco['images'].append(image_info)

        if anns_for_image:
            draw_annotations(overlay, anns_for_image, (img_w, img_h), category_names)

        save_path = overlays_dir / fname
        overlay.save(save_path)
        print(f'Saved overlay: {save_path}')

        img_id += 1

    out_json = out_dir / f'sample_instances_{args.split}.json'
    with open(out_json, 'w') as f:
        json.dump(coco, f, indent=2)

    print('\nSummary:')
    print(f'  Samples processed: {img_id}')
    print(f'  Annotations saved: {ann_id}')
    print(f'  Overlays: {overlays_dir}')
    print(f'  Small COCO JSON: {out_json}')


if __name__ == '__main__':
    main()
