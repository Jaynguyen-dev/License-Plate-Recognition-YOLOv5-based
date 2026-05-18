"""Train the one-line/two-line layout classifier."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

from .constants import IDX_TO_LAYOUT
from .datasets import LayoutDataset, class_weights
from .models import LayoutCNN
from .torch_utils import pick_device, require_torch


DEFAULT_CONFIG = {
    "metadata": "data/metadata.csv",
    "data_root": "archive",
    "output_dir": "outputs/layout",
    "report": "outputs/reports/layout_metrics.csv",
    "image_size": [192, 96],
    "batch_size": 64,
    "epochs": 10,
    "lr": 0.001,
    "num_workers": 0,
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
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            pred = logits.argmax(dim=1)
            correct += int((pred == labels).sum().item())
            total += int(labels.numel())
    return correct / max(1, total)


def write_layout_report(path: str, rows) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)


def train(config: dict) -> Path:
    torch = require_torch()
    from torch.utils.data import DataLoader

    device = pick_device(config.get("device", "auto"))
    train_ds = LayoutDataset(
        config["metadata"],
        data_root=config["data_root"],
        split="train",
        image_size=config["image_size"],
        augment=True,
    )
    val_ds = LayoutDataset(
        config["metadata"],
        data_root=config["data_root"],
        split="val",
        image_size=config["image_size"],
        augment=False,
    )
    if len(train_ds) == 0:
        raise RuntimeError("No training rows found. Run prepare_metadata first.")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config.get("num_workers", 0)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
    )

    model = LayoutCNN().to(device)
    weights = class_weights(train_ds.rows).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]))

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    best_acc = -1.0

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        running = 0.0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * int(labels.numel())
            seen += int(labels.numel())
        val_acc = evaluate(model, val_loader, device) if len(val_ds) else 0.0
        train_loss = running / max(1, seen)
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "idx_to_layout": IDX_TO_LAYOUT,
                    "val_acc": best_acc,
                },
                best_path,
            )

    write_layout_report(config.get("report", "outputs/reports/layout_metrics.csv"), [{"split": "val", "accuracy": f"{best_acc:.6f}"}])
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train layout classifier.")
    parser.add_argument("--config", default="configs/layout.yaml")
    args = parser.parse_args()
    path = train(load_config(args.config))
    print(f"Saved best layout checkpoint -> {path}")


if __name__ == "__main__":
    main()
