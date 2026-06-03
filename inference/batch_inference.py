"""
Batch inference for the Faster R-CNN license plate detector.

Default usage:
    python batch_inference.py --input LP_detection/images/val --checkpoint best.pth

Then crop detections for OCR with:
    python crop_from_inference_results.py --images-root LP_detection/images/val

This script expects the checkpoint to match train.py's Faster R-CNN ResNet18-FPN
model with 2 classes: background + license_plate.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as TF

from train import get_model

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is listed in requirements.
    tqdm = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch inference with a Faster R-CNN license plate detector."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Image file or directory of images to process.",
    )
    parser.add_argument(
        "--checkpoint",
        default=Path("best.pth"),
        type=Path,
        help="Path to the .pth checkpoint. Defaults to best.pth.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("inference_outputs"),
        type=Path,
        help="Directory for annotated images and detection files.",
    )
    parser.add_argument(
        "--backbone",
        default="resnet18",
        choices=["resnet18", "resnet50"],
        help="Backbone used to build the Faster R-CNN model.",
    )
    parser.add_argument(
        "--num-classes",
        default=2,
        type=int,
        help="Number of classes including background.",
    )
    parser.add_argument(
        "--class-names",
        default=["license_plate"],
        nargs="+",
        help="Class names for labels 1..N. Defaults to license_plate.",
    )
    parser.add_argument(
        "--score-threshold",
        default=0.5,
        type=float,
        help="Minimum confidence score to keep a detection.",
    )
    parser.add_argument(
        "--batch-size",
        default=4,
        type=int,
        help="Number of images per forward pass.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for inference. Defaults to auto.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for images when --input is a directory.",
    )
    parser.add_argument(
        "--max-detections",
        default=None,
        type=int,
        help="Keep at most this many detections per image after thresholding.",
    )
    parser.add_argument(
        "--no-save-annotated",
        action="store_false",
        dest="save_annotated",
        help="Do not write annotated preview images.",
    )
    parser.set_defaults(save_annotated=True)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device


def build_label_map(class_names: Sequence[str]) -> Dict[int, str]:
    return {idx + 1: name for idx, name in enumerate(class_names)}


def find_images(input_path: Path, recursive: bool) -> Tuple[List[Path], Path]:
    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Input file is not a supported image: {input_path}")
        return [input_path], input_path.parent

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    pattern = "**/*" if recursive else "*"
    image_paths = [
        path
        for path in input_path.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    image_paths.sort()

    if not image_paths:
        mode = "recursively " if recursive else ""
        raise FileNotFoundError(f"No supported images found {mode}under: {input_path}")

    return image_paths, input_path


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> Dict:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    metadata = {}
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            metadata = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            metadata = {k: v for k, v in checkpoint.items() if k != "state_dict"}
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a model state dict.")

    state_dict = strip_module_prefix(state_dict)
    model.load_state_dict(state_dict)
    return metadata


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict

    if all(key.startswith("module.") for key in state_dict):
        return {key[len("module.") :]: value for key, value in state_dict.items()}

    return state_dict


def batched(items: Sequence[Path], batch_size: int) -> Iterable[Sequence[Path]]:
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def load_image(image_path: Path) -> Tuple[Image.Image, torch.Tensor]:
    image = Image.open(image_path).convert("RGB")
    tensor = TF.to_tensor(image)
    return image, tensor


def filter_detections(
    output: Dict[str, torch.Tensor],
    score_threshold: float,
    max_detections: int = None,
) -> List[Dict]:
    boxes = output["boxes"].detach().cpu()
    scores = output["scores"].detach().cpu()
    labels = output["labels"].detach().cpu()

    keep = scores >= score_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    if max_detections is not None:
        boxes = boxes[:max_detections]
        scores = scores[:max_detections]
        labels = labels[:max_detections]

    detections = []
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [float(value) for value in box.tolist()]
        detections.append(
            {
                "label": int(label.item()),
                "score": float(score.item()),
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
            }
        )
    return detections


def draw_detections(
    image: Image.Image,
    detections: Sequence[Dict],
    label_map: Dict[int, str],
) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        label = label_map.get(detection["label"], f"class_{detection['label']}")
        caption = f"{label} {detection['score']:.2f}"

        draw.rectangle((x1, y1, x2, y2), outline=(255, 36, 36), width=3)
        text_bbox = draw.textbbox((x1, y1), caption, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_y = max(0, y1 - text_h - 4)

        draw.rectangle(
            (x1, text_y, x1 + text_w + 6, text_y + text_h + 4),
            fill=(255, 36, 36),
        )
        draw.text((x1 + 3, text_y + 2), caption, fill=(255, 255, 255), font=font)

    return annotated


def relative_output_path(image_path: Path, input_root: Path) -> Path:
    try:
        return image_path.relative_to(input_root)
    except ValueError:
        return Path(image_path.name)


def write_json(results: Sequence[Dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def write_csv(results: Sequence[Dict], output_path: Path, label_map: Dict[int, str]) -> None:
    fieldnames = [
        "image",
        "label",
        "class_name",
        "score",
        "x1",
        "y1",
        "x2",
        "y2",
        "width",
        "height",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for detection in result["detections"]:
                x1, y1, x2, y2 = detection["bbox_xyxy"]
                writer.writerow(
                    {
                        "image": result["image"],
                        "label": detection["label"],
                        "class_name": label_map.get(
                            detection["label"], f"class_{detection['label']}"
                        ),
                        "score": f"{detection['score']:.6f}",
                        "x1": f"{x1:.2f}",
                        "y1": f"{y1:.2f}",
                        "x2": f"{x2:.2f}",
                        "y2": f"{y2:.2f}",
                        "width": f"{x2 - x1:.2f}",
                        "height": f"{y2 - y1:.2f}",
                    }
                )


def progress_bar(items: Iterable, total: int, desc: str):
    if tqdm is None:
        return items
    return tqdm(items, total=total, desc=desc)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    label_map = build_label_map(args.class_names)

    image_paths, input_root = find_images(args.input, recursive=args.recursive)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Found {len(image_paths)} image(s).")
    print(f"Loading model: Faster R-CNN {args.backbone}")

    model = get_model(args.num_classes, backbone_name=args.backbone, pretrained=False)
    model.to(device)
    metadata = load_checkpoint(model, args.checkpoint, device)
    model.eval()

    if metadata:
        summary_keys = ["epoch", "train_loss", "val_loss"]
        summary = {key: metadata[key] for key in summary_keys if key in metadata}
        if summary:
            print(f"Checkpoint metadata: {summary}")

    results = []
    total_batches = (len(image_paths) + args.batch_size - 1) // args.batch_size
    batch_iter = batched(image_paths, args.batch_size)

    with torch.inference_mode():
        for batch_paths in progress_bar(batch_iter, total_batches, "Infer"):
            loaded = [load_image(path) for path in batch_paths]
            images = [image for image, _ in loaded]
            tensors = [tensor.to(device) for _, tensor in loaded]
            outputs = model(tensors)

            for image_path, image, output in zip(batch_paths, images, outputs):
                detections = filter_detections(
                    output,
                    score_threshold=args.score_threshold,
                    max_detections=args.max_detections,
                )

                rel_path = relative_output_path(image_path, input_root)
                result = {
                    "image": str(rel_path).replace("\\", "/"),
                    "width": image.width,
                    "height": image.height,
                    "detections": detections,
                }
                results.append(result)

                if args.save_annotated:
                    annotated = draw_detections(image, detections, label_map)
                    annotated_path = args.output_dir / "annotated" / rel_path
                    annotated_path.parent.mkdir(parents=True, exist_ok=True)
                    annotated.save(annotated_path)

    json_path = args.output_dir / "detections.json"
    csv_path = args.output_dir / "detections.csv"
    write_json(results, json_path)
    write_csv(results, csv_path, label_map)

    num_detections = sum(len(result["detections"]) for result in results)
    print(f"Done. Kept {num_detections} detection(s) over {len(results)} image(s).")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    if args.save_annotated:
        print(f"Annotated images: {args.output_dir / 'annotated'}")
    print("Crop for OCR with:")
    print(
        f"python crop_from_inference_results.py --detections {json_path} "
        f"--images-root {input_root}"
    )


if __name__ == "__main__":
    main()
