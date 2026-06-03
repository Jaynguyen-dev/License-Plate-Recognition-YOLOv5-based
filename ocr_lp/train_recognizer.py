"""Train the CRNN recognizer with CTC loss."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .constants import BLANK_INDEX, IDX_TO_CHAR
from .ctc import greedy_decode
from .datasets import LineCropDataset, line_collate
from .metrics import exact_match_rate
from .models import CRNN
from .torch_utils import pick_device, require_torch


DEFAULT_CONFIG = {
    "line_metadata": "data/line_crops/line_metadata.csv",
    "output_dir": "outputs/crnn",
    "batch_size": 32,
    "epochs": 20,
    "lr": 0.0003,
    "num_workers": 0,
    "image_height": 48,
    "max_width": 320,
    "device": "auto",
}


def load_config(path: str) -> dict:
    config = dict(DEFAULT_CONFIG)
    if path:
        with Path(path).open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        config.update(loaded)
    return config


def evaluate(model, loader, device: str) -> float:
    torch = require_torch()
    model.eval()
    pairs = []
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            logits = model(images)
            preds = greedy_decode(logits)
            pairs.extend(zip(preds, batch["labels"]))
    return exact_match_rate(pairs)


def train(config: dict) -> Path:
    torch = require_torch()
    from torch.utils.data import DataLoader

    line_metadata = Path(config["line_metadata"])
    if not line_metadata.exists():
        raise RuntimeError("Line metadata not found. Run make_line_crops before training the recognizer.")

    device = pick_device(config.get("device", "auto"))
    train_ds = LineCropDataset(
        str(line_metadata),
        split="train",
        image_height=int(config["image_height"]),
        max_width=int(config["max_width"]),
    )
    val_ds = LineCropDataset(
        str(line_metadata),
        split="val",
        image_height=int(config["image_height"]),
        max_width=int(config["max_width"]),
    )
    if len(train_ds) == 0:
        raise RuntimeError("No recognizer training rows found.")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        collate_fn=line_collate,
        num_workers=int(config.get("num_workers", 0)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        collate_fn=line_collate,
        num_workers=int(config.get("num_workers", 0)),
    )

    model = CRNN(num_classes=len(IDX_TO_CHAR)).to(device)
    criterion = torch.nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]))

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    best_acc = -1.0

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        running = 0.0
        seen = 0
        for batch in train_loader:
            images = batch["images"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            input_lengths = model.feature_lengths(batch["widths"].to(device))
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * images.shape[0]
            seen += images.shape[0]
        val_exact = evaluate(model, val_loader, device) if len(val_ds) else 0.0
        train_loss = running / max(1, seen)
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_line_exact={val_exact:.4f}")
        if val_exact >= best_acc:
            best_acc = val_exact
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "idx_to_char": IDX_TO_CHAR,
                    "val_line_exact": best_acc,
                },
                best_path,
            )
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CRNN recognizer.")
    parser.add_argument("--config", default="configs/crnn.yaml")
    args = parser.parse_args()
    path = train(load_config(args.config))
    print(f"Saved best recognizer checkpoint -> {path}")


if __name__ == "__main__":
    main()
