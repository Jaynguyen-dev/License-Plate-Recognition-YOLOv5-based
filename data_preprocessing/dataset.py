"""
PyTorch Dataset for License Plate detection using Faster R-CNN.

Loads images and COCO-format annotations, applies transforms,
and returns data in the format expected by torchvision's Faster R-CNN.

Expected format per sample:
    image: Tensor[C, H, W]  (float, 0-1 range)
    target: dict with keys:
        - boxes: Tensor[N, 4] in (x_min, y_min, x_max, y_max) format
        - labels: Tensor[N] (int64, class IDs starting from 1)
        - image_id: Tensor[1]
        - area: Tensor[N]
        - iscrowd: Tensor[N]
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from collections import defaultdict


class LicensePlateDataset(Dataset):
    """
    Dataset for License Plate detection in COCO format.

    Args:
        images_dir: Path to the directory containing images.
        annotation_file: Path to the COCO-format JSON annotation file.
        transforms: Optional transform pipeline (should handle both image and target).
    """

    def __init__(self, images_dir, annotation_file, transforms=None):
        self.images_dir = images_dir
        self.transforms = transforms

        # Load COCO annotations
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        self.images_info = {img['id']: img for img in coco_data['images']}
        self.categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

        # Group annotations by image_id
        self.img_annotations = defaultdict(list)
        for ann in coco_data['annotations']:
            self.img_annotations[ann['image_id']].append(ann)

        # Create ordered list of image IDs (include images even if no annotations)
        self.image_ids = sorted(self.images_info.keys())

        print(f"Loaded dataset: {len(self.image_ids)} images, "
              f"{len(coco_data['annotations'])} annotations, "
              f"{len(self.categories)} categories: {self.categories}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.images_info[image_id]

        # Load image
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")

        # Build target
        annotations = self.img_annotations.get(image_id, [])

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in annotations:
            x, y, w, h = ann['bbox']
            # Convert COCO (x, y, w, h) -> (x_min, y_min, x_max, y_max)
            x_min = x
            y_min = y
            x_max = x + w
            y_max = y + h

            # Skip degenerate boxes
            if w < 1 or h < 1:
                continue

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(ann['category_id'])
            areas.append(ann['area'])
            iscrowd.append(ann['iscrowd'])

        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        else:
            # Empty annotations (negative sample)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id]),
            "area": areas,
            "iscrowd": iscrowd,
        }

        # Apply transforms
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            # Default: just convert to tensor
            image = F.to_image(image)
            image = F.to_dtype(image, torch.float32, scale=True)

        return image, target


def get_transforms(train=False):
    """
    Build transform pipeline for Faster R-CNN.

    For training: random horizontal flip + photometric augmentation.
    For validation: just convert to tensor.

    Note: Faster R-CNN internally handles resizing (min_size=800, max_size=1333),
    so we do NOT resize here.
    """
    transforms = []

    if train:
        # Random horizontal flip (also flips boxes)
        transforms.append(T.RandomHorizontalFlip(p=0.5))

        # Photometric distortions (these don't affect bounding boxes)
        transforms.append(T.RandomPhotometricDistort(p=0.5))

    # Convert PIL to tensor and scale to [0, 1]
    transforms.append(T.ToImage())
    transforms.append(T.ToDtype(torch.float32, scale=True))

    return T.Compose(transforms)


def collate_fn(batch):
    """
    Custom collate function for Faster R-CNN.

    Images have different sizes, so we can't stack them into a single tensor.
    Instead, return a list of images and a list of targets.
    """
    return tuple(zip(*batch))


if __name__ == "__main__":
    # Quick test
    dataset = LicensePlateDataset(
        images_dir="LP_detection/images/train",
        annotation_file="LP_detection/annotations/instances_train.json",
        transforms=get_transforms(train=True)
    )

    print(f"\nDataset size: {len(dataset)}")

    # Test first sample
    image, target = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    print(f"Boxes: {target['boxes']}")
    print(f"Labels: {target['labels']}")
    print(f"Area: {target['area']}")

    # Test collate
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    images, targets = next(iter(loader))
    print(f"\nBatch: {len(images)} images")
    for i, (img, tgt) in enumerate(zip(images, targets)):
        print(f"  Image {i}: {img.shape}, {len(tgt['boxes'])} boxes")
