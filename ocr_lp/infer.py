"""Run layout detection, optional line splitting, OCR, and post-processing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from .constants import IDX_TO_LAYOUT
from .ctc import greedy_decode
from .datasets import pil_to_tensor, read_csv
from .models import CRNN, LayoutCNN
from .postprocess import PlateValidator
from .splitter import resize_line_crop, split_two_line_image
from .torch_utils import pick_device, require_torch


def load_validator(metadata_csv: str) -> PlateValidator:
    path = Path(metadata_csv)
    if not path.exists():
        return PlateValidator([])
    rows = read_csv(str(path))
    return PlateValidator.from_labels(row.get("canonical_label", "") for row in rows)


def predict_layout(image: Image.Image, checkpoint: str, device: str) -> str:
    torch = require_torch()
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        width, height = image.size
        return "two_line" if height and width / float(height) <= 2.6 else "one_line"
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model = LayoutCNN().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    size = ckpt.get("config", {}).get("image_size", [192, 96])
    with torch.no_grad():
        tensor = pil_to_tensor(image.convert("RGB").resize(tuple(size), Image.Resampling.BILINEAR), channels=3)
        pred = int(model(tensor.unsqueeze(0).to(device)).argmax(dim=1).item())
    return IDX_TO_LAYOUT[pred]


def recognize_line(model, image: Image.Image, device: str) -> str:
    torch = require_torch()
    line = resize_line_crop(image)
    tensor = pil_to_tensor(line, channels=1).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
    return greedy_decode(logits)[0]


def infer_image(
    image_path: str,
    recognizer_checkpoint: str = "outputs/crnn/best.pt",
    layout: str = "auto",
    layout_checkpoint: str = "outputs/layout/best.pt",
    metadata: str = "data/metadata.csv",
) -> dict:
    torch = require_torch()
    device = pick_device("auto")
    ckpt = torch.load(recognizer_checkpoint, map_location=device)
    recognizer = CRNN().to(device)
    recognizer.load_state_dict(ckpt["model_state"])
    recognizer.eval()
    validator = load_validator(metadata)

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        resolved_layout = predict_layout(image, layout_checkpoint, device) if layout == "auto" else layout
        if resolved_layout == "two_line":
            split = split_two_line_image(image)
            line_images = [split.top, split.bottom]
            split_confidence = split.confidence
            low_split = split.low_confidence
        else:
            line_images = [image]
            split_confidence = None
            low_split = False
        line_predictions = [recognize_line(recognizer, line, device) for line in line_images]
        raw = "".join(line_predictions)
        post = validator.correct(raw)
    return {
        "image": image_path,
        "layout": resolved_layout,
        "line_predictions": line_predictions,
        "raw_prediction": raw,
        "prediction": post.text,
        "low_confidence": post.low_confidence or low_split,
        "corrected": post.corrected,
        "split_confidence": split_confidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OCR inference on a plate image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--layout", default="auto", choices=["auto", "one_line", "two_line"])
    parser.add_argument("--checkpoint", default="outputs/crnn/best.pt")
    parser.add_argument("--layout-checkpoint", default="outputs/layout/best.pt")
    parser.add_argument("--metadata", default="data/metadata.csv")
    args = parser.parse_args()
    result = infer_image(
        args.image,
        recognizer_checkpoint=args.checkpoint,
        layout=args.layout,
        layout_checkpoint=args.layout_checkpoint,
        metadata=args.metadata,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
