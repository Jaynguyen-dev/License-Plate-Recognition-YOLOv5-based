"""
Comprehensive Model Evaluation Script for License Plate Detection.

Evaluates a trained Faster R-CNN model and computes detailed metrics:
- Precision, Recall
- Intersection over Union (IoU)
- mAP@0.5 (mean Average Precision at IoU=0.5)
- mAP@0.5:0.95 (mean Average Precision across IoU thresholds)

Usage:
    python evaluate_model.py --checkpoint checkpoints_resnet18/best.pth
    python evaluate_model.py --checkpoint checkpoints_resnet18/best.pth --visualize
"""

import os
import json
import torch
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict
import warnings

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from data_preprocessing.dataset import LicensePlateDataset, get_transforms, collate_fn
from model.train import get_model


def compute_iou(box1, box2):
    """
    Compute IoU (Intersection over Union) between two boxes.
    
    Args:
        box1, box2: boxes in (x1, y1, x2, y2) format
    
    Returns:
        IoU value (0-1)
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection area
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    
    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def match_detections_to_gt(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Match predicted boxes to ground truth boxes using IoU.
    
    Args:
        pred_boxes: predicted boxes [N, 4]
        gt_boxes: ground truth boxes [M, 4]
        iou_threshold: IoU threshold for matching
    
    Returns:
        tuple: (true_positives, false_positives, matched_ious)
    """
    if len(pred_boxes) == 0:
        return 0, 0, []
    
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), []
    
    gt_matched = np.zeros(len(gt_boxes), dtype=bool)
    matched_ious = []
    tp = 0
    fp = 0
    
    for pred_box in pred_boxes:
        max_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_matched[gt_idx]:
                continue
            
            iou = compute_iou(pred_box, gt_box)
            if iou > max_iou:
                max_iou = iou
                best_gt_idx = gt_idx
        
        if max_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            gt_matched[best_gt_idx] = True
            matched_ious.append(max_iou)
        else:
            fp += 1
            matched_ious.append(max_iou)
    
    return tp, fp, matched_ious


@torch.no_grad()
def evaluate_model(model, data_loader, device, annotation_file, iou_threshold=0.5):
    """
    Comprehensive evaluation of the model.
    
    Returns:
        dict with evaluation metrics
    """
    model.eval()
    coco_gt = COCO(annotation_file)
    
    results = []
    all_tp = 0
    all_fp = 0
    all_fn = 0
    all_ious = []
    images_evaluated = 0
    
    print("\nEvaluating on validation set...")
    
    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        outputs = model(images)
        
        for target, output in zip(targets, outputs):
            images_evaluated += 1
            image_id = target["image_id"].item()
            gt_boxes = target["boxes"].cpu().numpy()  # [N, 4]
            
            pred_boxes = output["boxes"].cpu().numpy()
            pred_scores = output["scores"].cpu().numpy()
            pred_labels = output["labels"].cpu().numpy()
            
            # Match detections to GT
            tp, fp, ious = match_detections_to_gt(pred_boxes, gt_boxes, iou_threshold)
            all_tp += tp
            all_fp += fp
            all_fn += len(gt_boxes) - tp
            all_ious.extend(ious)
            
            # Prepare COCO format results
            for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                x1, y1, x2, y2 = box
                results.append({
                    "image_id": int(image_id),
                    "category_id": int(label),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score)
                })
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {images_evaluated} images...")
    
    # Compute precision and recall
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    avg_iou = np.mean(all_ious) if all_ious else 0.0
    
    # Compute mAP using COCO evaluation
    mAP50 = 0.0
    mAP50_95 = 0.0
    
    if results:
        coco_dt = coco_gt.loadRes(results)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coco_eval.summarize()
        
        # COCO stats indices:
        # [0]: mAP@[0.5:0.95]
        # [1]: mAP@0.5
        # [2]: mAP@0.75
        mAP50_95 = coco_eval.stats[0]
        mAP50 = coco_eval.stats[1]
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'avg_iou': avg_iou,
        'mAP50': mAP50,
        'mAP50_95': mAP50_95,
        'total_images': images_evaluated,
        'total_detections': len(results),
        'true_positives': all_tp,
        'false_positives': all_fp,
        'false_negatives': all_fn
    }
    
    return metrics


def print_evaluation_results(metrics, checkpoint_path):
    """Print evaluation results in a formatted table."""
    print("\n" + "="*70)
    print(f"EVALUATION RESULTS - {Path(checkpoint_path).name}")
    print("="*70)
    
    print(f"\n{'Metric':<30} {'Value':<15} {'Interpretation':<25}")
    print("-" * 70)
    
    print(f"{'Precision':<30} {metrics['precision']:>6.4f} {'':<25}")
    print(f"{'Recall':<30} {metrics['recall']:>6.4f} {'':<25}")
    print(f"{'Average IoU':<30} {metrics['avg_iou']:>6.4f} {'':<25}")
    print(f"{'mAP@0.5':<30} {metrics['mAP50']:>6.4f} {'':<25}")
    print(f"{'mAP@0.5:0.95':<30} {metrics['mAP50_95']:>6.4f} {'':<25}")
    
    print("\n" + "-" * 70)
    print(f"{'Detection Statistics':<30} {'':<15} {'':<25}")
    print("-" * 70)
    print(f"{'Images Evaluated':<30} {metrics['total_images']:>6d} {'':<25}")
    print(f"{'Total Detections':<30} {metrics['total_detections']:>6d} {'':<25}")
    print(f"{'True Positives':<30} {metrics['true_positives']:>6d} {'':<25}")
    print(f"{'False Positives':<30} {metrics['false_positives']:>6d} {'':<25}")
    print(f"{'False Negatives':<30} {metrics['false_negatives']:>6d} {'':<25}")
    
    # F1 Score
    if (metrics['precision'] + metrics['recall']) > 0:
        f1_score = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
    else:
        f1_score = 0.0
    
    print("\n" + "-" * 70)
    print(f"{'F1 Score':<30} {f1_score:>6.4f} {'':<25}")
    print("="*70 + "\n")
    
    return f1_score


def visualize_results(metrics, output_file=None):
    """Create visualization of evaluation metrics with improved clarity.

    Saves PNG (and SVG) with higher DPI and clearer labels/annotations.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

        # Common style settings
        title_kw = dict(fontsize=13, fontweight='bold')
        label_kw = dict(fontsize=11, fontweight='bold')
        value_fmt = '{:.3f}'

        # 1. Detection Metrics Bar Chart (normalized scores)
        ax = axes[0, 0]
        metrics_names = ['Precision', 'Recall', 'Avg IoU']
        metrics_values = [metrics.get('precision', 0.0), metrics.get('recall', 0.0), metrics.get('avg_iou', 0.0)]
        colors_bar = ['#2b83ba', '#f39b4a', '#57a773']
        bars = ax.bar(metrics_names, metrics_values, color=colors_bar, alpha=0.9, edgecolor='none')
        ax.set_ylabel('Score', **label_kw)
        ax.set_title('Detection Performance Metrics', pad=12, **title_kw)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.25, axis='y')
        y_max = ax.get_ylim()[1]
        for bar, val in zip(bars, metrics_values):
            label_y = min(y_max * 0.92, val + 0.03)
            ax.text(bar.get_x() + bar.get_width() / 2, label_y,
                    value_fmt.format(val), ha='center', va='bottom', fontsize=10)

        # 2. mAP Metrics
        ax = axes[0, 1]
        map_names = ['mAP@0.5', 'mAP@0.5:0.95']
        map_values = [metrics.get('mAP50', 0.0), metrics.get('mAP50_95', 0.0)]
        colors_map = ['#d62728', '#9467bd']
        bars = ax.bar(map_names, map_values, color=colors_map, alpha=0.9)
        ax.set_ylabel('mAP Score', **label_kw)
        ax.set_title('Mean Average Precision', pad=12, **title_kw)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.25, axis='y')
        y_max_map = ax.get_ylim()[1]
        for bar, val in zip(bars, map_values):
            label_y = min(y_max_map * 0.92, val + 0.03)
            ax.text(bar.get_x() + bar.get_width() / 2, label_y,
                    value_fmt.format(val), ha='center', va='bottom', fontsize=10)

        # Annotate best mAP if available, placing annotation within bounds
        try:
            if map_values and any([v is not None for v in map_values]):
                best_idx = int(np.nanargmax(map_values))
                best_val = map_values[best_idx]
                ann_y = min(y_max_map * 0.92, best_val + 0.05)
                # x position in data coordinates: center of corresponding bar
                bar = bars[best_idx]
                bar_x = bar.get_x() + bar.get_width() / 2
                ax.annotate(f'Best: {value_fmt.format(best_val)}', xy=(bar_x, best_val),
                            xytext=(bar_x, ann_y), ha='center', fontsize=9,
                            arrowprops=dict(arrowstyle='->', color='black'))
        except Exception:
            pass

        # 3. Detection Statistics (styled textbox)
        ax = axes[1, 0]
        ax.axis('off')
        f1_val = 2 * (metrics.get('precision', 0.0) * metrics.get('recall', 0.0)) / (metrics.get('precision', 0.0) + metrics.get('recall', 0.0)) if (metrics.get('precision', 0.0) + metrics.get('recall', 0.0)) > 0 else 0.0
        stats_lines = [
            f"Total Images: {metrics.get('total_images', 'N/A')}",
            f"Total Detections: {metrics.get('total_detections', 0)}",
            f"True Positives: {metrics.get('true_positives', 0)}",
            f"False Positives: {metrics.get('false_positives', 0)}",
            f"False Negatives: {metrics.get('false_negatives', 0)}",
            f"F1 Score: {f1_val:.4f}"
        ]
        stats_text = '\n'.join(stats_lines)
        ax.text(0.01, 0.99, stats_text, fontsize=11, family='monospace', verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#fff7e6', edgecolor='#e6cfa6'))

        # 4. Detection Outcomes (counts) with percentage annotations
        ax = axes[1, 1]
        categories = ['TP', 'FP', 'FN']
        counts = [metrics.get('true_positives', 0), metrics.get('false_positives', 0), metrics.get('false_negatives', 0)]
        colors_cm = ['#2ca02c', '#d62728', '#ff7f0e']
        bars = ax.bar(categories, counts, color=colors_cm, alpha=0.95)
        ax.set_ylabel('Count', **label_kw)
        ax.set_title('Detection Outcomes', pad=12, **title_kw)
        ax.grid(True, alpha=0.2, axis='y')

        total = sum(counts) if sum(counts) > 0 else 1
        y_max_out = ax.get_ylim()[1]
        for bar, val in zip(bars, counts):
            pct = 100.0 * val / total
            label_y = min(y_max_out * 0.92, val + max(1, total * 0.01))
            ax.text(bar.get_x() + bar.get_width() / 2, label_y,
                    f'{int(val)} ({pct:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Finalize and save
        out_dir = '.'
        out_name = output_file if output_file else 'evaluation_results'
        if not out_name.lower().endswith('.png'):
            out_name_png = out_name + '.png'
        else:
            out_name_png = out_name
        out_name_svg = out_name_png.rsplit('.', 1)[0] + '.svg'

        dpi = 200
        plt.savefig(out_name_png, dpi=dpi, bbox_inches='tight')
        try:
            plt.savefig(out_name_svg, dpi=dpi, bbox_inches='tight')
        except Exception:
            pass
        print(f"Visualization saved to: {out_name_png} (and SVG)")

        plt.close(fig)

    except ImportError:
        print("matplotlib not available, skipping visualization")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained License Plate Detection model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_resnet18/best.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--dataset-root", type=str, default="LP_detection",
                        help="Root directory of the dataset")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold for matching detections to GT")
    parser.add_argument("--visualize", action="store_true",
                        help="Create visualization of evaluation results")
    parser.add_argument("--save-results", type=str, default=None,
                        help="Save evaluation results to JSON file")
    
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Paths
    checkpoint_path = args.checkpoint
    val_images = os.path.join(args.dataset_root, "images", "val")
    val_ann = os.path.join(args.dataset_root, "annotations", "instances_val.json")
    
    # Check checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load validation dataset
    print("Loading validation dataset...")
    val_dataset = LicensePlateDataset(
        images_dir=val_images,
        annotation_file=val_ann,
        transforms=get_transforms(train=False)
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == "cuda" else False,
    )
    print(f"Validation dataset size: {len(val_dataset)}\n")
    
    # Load model
    print("Building model...")
    model = get_model(num_classes=2, backbone_name='resnet18', pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print("Model loaded successfully")
    
    # Evaluate
    print("\n" + "="*70)
    print("STARTING EVALUATION")
    print("="*70)
    
    metrics = evaluate_model(model, val_loader, device, val_ann, iou_threshold=args.iou_threshold)
    f1 = print_evaluation_results(metrics, checkpoint_path)
    
    # Save results if requested
    if args.save_results:
        results_dict = {
            'checkpoint': checkpoint_path,
            'metrics': metrics,
            'f1_score': f1
        }
        with open(args.save_results, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"Results saved to: {args.save_results}")
    
    # Visualize if requested
    if args.visualize:
        output_file = 'evaluation_results.png'
        visualize_results(metrics, output_file)


if __name__ == "__main__":
    main()
