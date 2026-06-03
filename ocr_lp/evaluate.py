"""Evaluate CRNN OCR checkpoints on a metadata split."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from .ctc import greedy_decode
from .datasets import LineCropDataset, line_collate, read_csv
from .metrics import summarize_pairs
from .models import CRNN
from .postprocess import PlateValidator
from .torch_utils import pick_device, require_torch


def _write_report(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scope", "layout", "count", "exact_match", "cer"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summary_row(scope: str, layout: str, pairs: List[Tuple[str, str]]) -> Dict[str, str]:
    metrics = summarize_pairs(pairs)
    return {
        "scope": scope,
        "layout": layout,
        "count": str(int(metrics["count"])),
        "exact_match": f"{metrics['exact_match']:.6f}",
        "cer": f"{metrics['cer']:.6f}",
    }


def evaluate_checkpoint(checkpoint: str, split: str = "test", report: str = "outputs/reports/ocr_metrics.csv") -> Path:
    torch = require_torch()
    from torch.utils.data import DataLoader

    device = pick_device("auto")
    ckpt = torch.load(checkpoint, map_location=device)
    config = ckpt.get("config", {})
    line_metadata = config.get("line_metadata", "data/line_crops/line_metadata.csv")
    rows_all = read_csv(line_metadata)
    validator = PlateValidator.from_labels(row.get("plate_label", "") for row in rows_all)

    dataset = LineCropDataset(
        line_metadata,
        split=split,
        image_height=int(config.get("image_height", 48)),
        max_width=int(config.get("max_width", 320)),
    )
    loader = DataLoader(dataset, batch_size=int(config.get("batch_size", 32)), shuffle=False, collate_fn=line_collate)
    model = CRNN()
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    line_pairs: List[Tuple[str, str]] = []
    by_plate: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["images"].to(device))
            preds = greedy_decode(logits)
            for pred, label, row in zip(preds, batch["labels"], batch["rows"]):
                line_pairs.append((pred, label))
                enriched = dict(row)
                enriched["prediction"] = pred
                by_plate[row["image_path"]].append(enriched)

    full_raw_pairs: List[Tuple[str, str]] = []
    full_post_pairs: List[Tuple[str, str]] = []
    layout_pairs: Dict[str, Dict[str, List[Tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for rows in by_plate.values():
        rows = sorted(rows, key=lambda item: int(item.get("line_index", "0")))
        raw_pred = "".join(row["prediction"] for row in rows)
        target = rows[0].get("plate_label", "")
        layout = rows[0].get("layout", "")
        post = validator.correct(raw_pred).text
        full_raw_pairs.append((raw_pred, target))
        full_post_pairs.append((post, target))
        layout_pairs["full_plate_raw"][layout].append((raw_pred, target))
        layout_pairs["full_plate_post"][layout].append((post, target))

    report_rows = [
        _summary_row("line", "all", line_pairs),
        _summary_row("full_plate_raw", "all", full_raw_pairs),
        _summary_row("full_plate_post", "all", full_post_pairs),
    ]
    for scope in ["full_plate_raw", "full_plate_post"]:
        for layout, pairs in sorted(layout_pairs[scope].items()):
            report_rows.append(_summary_row(scope, layout, pairs))

    report_path = Path(report)
    _write_report(report_path, report_rows)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OCR checkpoint.")
    parser.add_argument("--checkpoint", default="outputs/crnn/best.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--report", default="outputs/reports/ocr_metrics.csv")
    args = parser.parse_args()
    report = evaluate_checkpoint(args.checkpoint, split=args.split, report=args.report)
    print(f"Wrote OCR metrics -> {report}")


if __name__ == "__main__":
    main()
