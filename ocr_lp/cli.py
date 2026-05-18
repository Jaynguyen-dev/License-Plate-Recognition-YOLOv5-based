"""CLI dispatcher for OCR license-plate utilities."""

from __future__ import annotations

import argparse

from . import make_line_crops, prepare_metadata


def main() -> None:
    parser = argparse.ArgumentParser(prog="ocr_lp")
    sub = parser.add_subparsers(dest="cmd")

    pm = sub.add_parser("prepare_metadata")
    pm.add_argument("--data-root", default="archive")
    pm.add_argument("--out", default="data/metadata.csv")
    pm.add_argument("--seed", type=int, default=1337)
    pm.add_argument("--layout-threshold", type=float, default=2.6)

    mc = sub.add_parser("make_line_crops")
    mc.add_argument("--metadata", default="data/metadata.csv")
    mc.add_argument("--data-root", default="archive")
    mc.add_argument("--out", default="data/line_crops")
    mc.add_argument("--manifest-name", default="line_metadata.csv")
    mc.add_argument("--debug-out", default="outputs/debug_splits")
    mc.add_argument("--line-height", type=int, default=48)
    mc.add_argument("--max-width", type=int, default=320)

    tl = sub.add_parser("train_layout")
    tl.add_argument("--config", default="configs/layout.yaml")

    tr = sub.add_parser("train_recognizer")
    tr.add_argument("--config", default="configs/crnn.yaml")

    ev = sub.add_parser("evaluate")
    ev.add_argument("--checkpoint", default="outputs/crnn/best.pt")
    ev.add_argument("--split", default="test")
    ev.add_argument("--report", default="outputs/reports/ocr_metrics.csv")

    inf = sub.add_parser("infer")
    inf.add_argument("--image", required=True)
    inf.add_argument("--layout", default="auto", choices=["auto", "one_line", "two_line"])
    inf.add_argument("--checkpoint", default="outputs/crnn/best.pt")
    inf.add_argument("--layout-checkpoint", default="outputs/layout/best.pt")
    inf.add_argument("--metadata", default="data/metadata.csv")

    args = parser.parse_args()
    if args.cmd == "prepare_metadata":
        rows = prepare_metadata.build_metadata(
            args.data_root,
            args.out,
            seed=args.seed,
            layout_threshold=args.layout_threshold,
        )
        print(f"Wrote metadata rows={len(rows)} -> {args.out}")
    elif args.cmd == "make_line_crops":
        n = make_line_crops.make_line_crops(
            args.metadata,
            args.data_root,
            args.out,
            manifest_name=args.manifest_name,
            debug_dir=args.debug_out,
            line_height=args.line_height,
            max_width=args.max_width,
        )
        print(f"Created {n} line crops -> {args.out}")
    elif args.cmd == "train_layout":
        from .train_layout import load_config, train

        print(f"Saved best layout checkpoint -> {train(load_config(args.config))}")
    elif args.cmd == "train_recognizer":
        from .train_recognizer import load_config, train

        print(f"Saved best recognizer checkpoint -> {train(load_config(args.config))}")
    elif args.cmd == "evaluate":
        from .evaluate import evaluate_checkpoint

        print(f"Wrote OCR metrics -> {evaluate_checkpoint(args.checkpoint, args.split, args.report)}")
    elif args.cmd == "infer":
        from .infer import infer_image
        import json

        print(
            json.dumps(
                infer_image(
                    args.image,
                    recognizer_checkpoint=args.checkpoint,
                    layout=args.layout,
                    layout_checkpoint=args.layout_checkpoint,
                    metadata=args.metadata,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
