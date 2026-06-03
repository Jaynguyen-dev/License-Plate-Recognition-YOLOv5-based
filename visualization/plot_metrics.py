"""
Plot training metrics and evaluation results over epochs.

This script loads model checkpoints, computes metrics on the validation set,
and generates comprehensive visualization charts.

Metrics visualized:
- Training Loss
- Validation Loss
- Precision
- Recall
- IoU (Intersection over Union)
- mAP@0.5 (mean Average Precision at IoU=0.5)
- mAP@0.5:0.95 (mean Average Precision across IoU thresholds)
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import argparse

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from data_preprocessing.dataset import LicensePlateDataset, get_transforms, collate_fn
from model.train import get_model


def load_checkpoints_in_order(checkpoint_dir):
    """Load all checkpoints sorted by epoch."""
    checkpoints = {}
    checkpoint_files = sorted(Path(checkpoint_dir).glob("*.pth"))
    
    for checkpoint_file in checkpoint_files:
        name = checkpoint_file.stem
        # Parse epoch from filename (epoch_5.pth, epoch_10.pth, etc.)
        if name.startswith("epoch_"):
            try:
                epoch = int(name.split("_")[1])
                checkpoints[epoch] = str(checkpoint_file)
            except ValueError:
                continue
        elif name == "best":
            checkpoints['best'] = str(checkpoint_file)
        elif name == "latest":
            checkpoints['latest'] = str(checkpoint_file)
    
    return checkpoints


def compute_metrics_for_checkpoint(model, data_loader, device, annotation_file):
    """
    Compute metrics including mAP, Precision, Recall, and IoU.
    
    Returns a dict with:
    - mAP50: mAP@0.5
    - mAP50_95: mAP@0.5:0.95
    - precision: Average precision across detections
    - recall: Average recall across detections
    - avg_iou: Average IoU across detections
    """
    model.eval()
    coco_gt = COCO(annotation_file)
    
    results = []
    ious_list = []
    precisions = []
    recalls = []
    tp_total = 0
    fp_total = 0
    fn_total = 0
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            
            outputs = model(images)
            
            for target, output in zip(targets, outputs):
                image_id = target["image_id"].item()
                gt_boxes = target["boxes"].cpu().numpy()  # [N, 4]
                
                pred_boxes = output["boxes"].cpu().numpy()
                pred_scores = output["scores"].cpu().numpy()
                pred_labels = output["labels"].cpu().numpy()
                
                # Append prediction entries for COCO evaluation
                for pred_box, score in zip(pred_boxes, pred_scores):
                    results.append({
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [float(pred_box[0]), float(pred_box[1]), 
                                   float(pred_box[2] - pred_box[0]), float(pred_box[3] - pred_box[1])],
                        "score": float(score)
                    })

                # Compute TP/FP/FN via greedy matching at IoU threshold 0.5
                matched_preds = set()
                iou_threshold = 0.5

                if len(gt_boxes) > 0:
                    # For each GT, find best unmatched prediction
                    for gt_box in gt_boxes:
                        best_iou = 0.0
                        best_pred_idx = -1
                        for pred_idx, pred_box in enumerate(pred_boxes):
                            if pred_idx in matched_preds:
                                continue
                            iou = compute_iou(pred_box, gt_box)
                            if iou > best_iou:
                                best_iou = iou
                                best_pred_idx = pred_idx

                        if best_iou >= iou_threshold and best_pred_idx >= 0:
                            tp_total += 1
                            matched_preds.add(best_pred_idx)
                            ious_list.append(best_iou)
                        else:
                            fn_total += 1
                else:
                    # No ground truth boxes: all predictions are false positives
                    pass

                # Any unmatched predictions are false positives
                for pred_idx in range(len(pred_boxes)):
                    if pred_idx not in matched_preds:
                        fp_total += 1
    
    # Compute mAP using pycocotools
    mAP50 = 0.0
    mAP50_95 = 0.0
    
    if results:
        coco_dt = coco_gt.loadRes(results)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # coco_eval.stats:
        # [0]: mAP@0.5:0.95
        # [1]: mAP@0.5
        # [2]: mAP@0.75
        # [3]: mAP (small)
        # [4]: mAP (medium)
        # [5]: mAP (large)
        mAP50_95 = coco_eval.stats[0]
        mAP50 = coco_eval.stats[1]
    
    avg_iou = np.mean(ious_list) if ious_list else 0.0

    # Compute precision and recall from aggregate counts
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0

    total_detections = len(results)
    
    return {
        'mAP50': mAP50,
        'mAP50_95': mAP50_95,
        'precision': precision,
        'recall': recall,
        'avg_iou': avg_iou,
        'tp': tp_total,
        'fp': fp_total,
        'fn': fn_total,
        'total_detections': total_detections
    }


def compute_iou(box1, box2):
    """Compute IoU between two boxes in (x1, y1, x2, y2) format."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def plot_metrics(epochs, train_losses, val_losses, map50_scores, map50_95_scores, 
                 precisions, recalls, ious, output_dir="metrics_plots",
                 tp_counts=None, fp_counts=None, fn_counts=None, total_images=None, total_detections_list=None):
    """Create comprehensive visualization plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = {
        'train': '#1f77b4',
        'val': '#ff7f0e',
        'metric': '#2ca02c'
    }
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Training and Validation Loss
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(epochs, train_losses, marker='o', label='Train Loss', color=colors['train'], linewidth=2, markersize=6)
    ax1.plot(epochs, val_losses, marker='s', label='Val Loss', color=colors['val'], linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax1.set_title('Training vs Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. mAP@0.5
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(epochs, map50_scores, marker='D', color='#d62728', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('mAP@0.5', fontsize=11, fontweight='bold')
    ax2.set_title('Mean Average Precision @ IoU=0.5', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # 3. mAP@0.5:0.95
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(epochs, map50_95_scores, marker='^', color='#9467bd', linewidth=2, markersize=6)
    ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax3.set_ylabel('mAP@0.5:0.95', fontsize=11, fontweight='bold')
    ax3.set_title('Mean Average Precision @ IoU=0.5:0.95', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # 4. Precision
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(epochs, precisions, marker='o', color='#1f77b4', linewidth=2, markersize=6)
    ax4.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Precision', fontsize=11, fontweight='bold')
    ax4.set_title('Precision Over Epochs', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    # 5. Recall
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(epochs, recalls, marker='s', color='#ff7f0e', linewidth=2, markersize=6)
    ax5.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Recall', fontsize=11, fontweight='bold')
    ax5.set_title('Recall Over Epochs', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 1])
    
    # 6. Average IoU
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(epochs, ious, marker='D', color='#2ca02c', linewidth=2, markersize=6)
    ax6.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Average IoU', fontsize=11, fontweight='bold')
    ax6.set_title('Average IoU Over Epochs', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([0, 1])
    
    # Annotate best mAP points (if available)
    try:
        if len(map50_scores) > 0:
            best_idx = int(np.nanargmax(map50_scores))
            best_epoch = epochs[best_idx]
            best_val = map50_scores[best_idx]
            ax2.annotate(f"Best: {best_val:.3f}\n(Epoch {best_epoch})",
                         xy=(best_epoch, best_val), xytext=(best_epoch, min(1.0, best_val + 0.08)),
                         arrowprops=dict(arrowstyle='->', color='#d62728'))
        if len(map50_95_scores) > 0:
            best_idx2 = int(np.nanargmax(map50_95_scores))
            best_epoch2 = epochs[best_idx2]
            best_val2 = map50_95_scores[best_idx2]
            ax3.annotate(f"Best: {best_val2:.3f}\n(Epoch {best_epoch2})",
                         xy=(best_epoch2, best_val2), xytext=(best_epoch2, min(1.0, best_val2 + 0.08)),
                         arrowprops=dict(arrowstyle='->', color='#9467bd'))
    except Exception:
        pass

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'metrics_overview.png')
    # Add a detection outcomes inset showing TP/FP/FN for the latest epoch (if provided)
    try:
        if tp_counts is not None and fp_counts is not None and fn_counts is not None and len(tp_counts) > 0:
            last_idx = -1
            last_tp = tp_counts[last_idx]
            last_fp = fp_counts[last_idx]
            last_fn = fn_counts[last_idx]

            ax_det = fig.add_axes([0.62, 0.12, 0.32, 0.28])
            bars = ax_det.bar(['TP', 'FP', 'FN'], [last_tp, last_fp, last_fn], color=['#2ca02c', '#d62728', '#8c564b'], alpha=0.85)
            ax_det.set_ylabel('Count', fontsize=10, fontweight='bold')
            ax_det.set_title('Detection Outcomes (latest epoch)', fontsize=11, fontweight='bold')
            ax_det.grid(False)
            for bar in bars:
                h = bar.get_height()
                ax_det.text(bar.get_x() + bar.get_width()/2, h + max(1, h*0.02), f"{int(h)}", ha='center', va='bottom', fontweight='bold')

            # Stats textbox
            total_dets = total_detections_list[last_idx] if total_detections_list is not None and len(total_detections_list) > 0 else (last_tp + last_fp)
            tot_imgs = total_images if total_images is not None else 'N/A'
            f1 = 2*last_tp / (2*last_tp + last_fp + last_fn) if (2*last_tp + last_fp + last_fn) > 0 else 0.0
            stats = (
                f"Detection Statistics:\n\n"
                f"Total Images: {tot_imgs}\n"
                f"Total Detections: {int(total_dets)}\n"
                f"True Positives: {int(last_tp)}\n"
                f"False Positives: {int(last_fp)}\n"
                f"False Negatives: {int(last_fn)}\n\n"
                f"F1 Score: {f1:.4f}"
            )
            ax_det.text(-0.05, -0.55, stats, fontsize=9, va='top', ha='left', bbox=dict(boxstyle='round', facecolor='#f7f1e1', edgecolor='#d3c4a3'))
    except Exception:
        pass

    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    # Individual detailed plots
    
    # Loss details
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_losses, marker='o', label='Train Loss', linewidth=2.5, markersize=7, color=colors['train'])
    ax.plot(epochs, val_losses, marker='s', label='Val Loss', linewidth=2.5, markersize=7, color=colors['val'])
    ax.fill_between(epochs, train_losses, alpha=0.2, color=colors['train'])
    ax.fill_between(epochs, val_losses, alpha=0.2, color=colors['val'])
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training vs Validation Loss Over Epochs', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_detailed.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'loss_detailed.png')}")
    
    # mAP details
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, map50_scores, marker='D', label='mAP@0.5', linewidth=2.5, markersize=7, color='#d62728')
    ax.plot(epochs, map50_95_scores, marker='^', label='mAP@0.5:0.95', linewidth=2.5, markersize=7, color='#9467bd')
    ax.fill_between(epochs, map50_scores, alpha=0.2, color='#d62728')
    ax.fill_between(epochs, map50_95_scores, alpha=0.2, color='#9467bd')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('mAP Score', fontsize=12, fontweight='bold')
    ax.set_title('Mean Average Precision Over Epochs', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'map_detailed.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'map_detailed.png')}")
    
    # Detection metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, precisions, marker='o', label='Precision', linewidth=2.5, markersize=7, color='#1f77b4')
    ax.plot(epochs, recalls, marker='s', label='Recall', linewidth=2.5, markersize=7, color='#ff7f0e')
    ax.plot(epochs, ious, marker='D', label='Avg IoU', linewidth=2.5, markersize=7, color='#2ca02c')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Detection Metrics: Precision, Recall, and IoU', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detection_metrics_detailed.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'detection_metrics_detailed.png')}")
    
    plt.close('all')
    print(f"\nAll plots saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics from checkpoints")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_resnet18",
                        help="Directory containing model checkpoints")
    parser.add_argument("--dataset-root", type=str, default="LP_detection",
                        help="Root directory of the dataset")
    parser.add_argument("--output-dir", type=str, default="metrics_plots",
                        help="Directory to save plots")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: only plot loss from checkpoints, skip metric computation")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Paths
    val_images = os.path.join(args.dataset_root, "images", "val")
    val_ann = os.path.join(args.dataset_root, "annotations", "instances_val.json")
    
    # Load checkpoints
    checkpoints = load_checkpoints_in_order(args.checkpoint_dir)
    print(f"Found {len(checkpoints)} checkpoints:")
    for epoch in sorted([e for e in checkpoints.keys() if isinstance(e, int)]):
        print(f"  Epoch {epoch}: {checkpoints[epoch]}")
    print()
    
    # Prepare validation dataset
    if not args.quick:
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
    
    # Extract metrics from checkpoints
    epochs = []
    train_losses = []
    val_losses = []
    map50_scores = []
    map50_95_scores = []
    precisions = []
    recalls = []
    ious = []
    tp_counts = []
    fp_counts = []
    fn_counts = []
    total_detections_list = []
    
    sorted_epochs = sorted([e for e in checkpoints.keys() if isinstance(e, int)])
    
    for i, epoch in enumerate(sorted_epochs):
        checkpoint_path = checkpoints[epoch]
        print(f"[{i+1}/{len(sorted_epochs)}] Processing epoch {epoch}...")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract losses from checkpoint
        train_loss = checkpoint.get('train_loss', 0.0)
        val_loss = checkpoint.get('val_loss', 0.0)
        
        epochs.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"  Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}")
        
        # Compute metrics if not in quick mode
        if not args.quick:
            # Load model and compute metrics
            model = get_model(num_classes=2, backbone_name='resnet18', pretrained=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            
            try:
                metrics = compute_metrics_for_checkpoint(model, val_loader, device, val_ann)
                map50_scores.append(metrics['mAP50'])
                map50_95_scores.append(metrics['mAP50_95'])
                precisions.append(metrics['precision'])
                recalls.append(metrics['recall'])
                ious.append(metrics['avg_iou'])
                tp_counts.append(metrics.get('tp', 0))
                fp_counts.append(metrics.get('fp', 0))
                fn_counts.append(metrics.get('fn', 0))
                total_detections_list.append(metrics.get('total_detections', 0))
                
                print(f"  mAP@0.5: {metrics['mAP50']:.4f}  |  mAP@0.5:0.95: {metrics['mAP50_95']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}  |  Avg IoU: {metrics['avg_iou']:.4f}")
            except Exception as e:
                print(f"  Error computing metrics: {e}")
                map50_scores.append(0.0)
                map50_95_scores.append(0.0)
                precisions.append(0.0)
                recalls.append(0.0)
                ious.append(0.0)
            
            del model
    
    print("\n" + "="*60)
    print("Generating plots...")
    print("="*60)
    
    if args.quick:
        # Only plot losses
        os.makedirs(args.output_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, train_losses, marker='o', label='Train Loss', linewidth=2.5, markersize=7)
        ax.plot(epochs, val_losses, marker='s', label='Val Loss', linewidth=2.5, markersize=7)
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax.set_title('Training vs Validation Loss Over Epochs', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'loss_only.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: {os.path.join(args.output_dir, 'loss_only.png')}")
    else:
        plot_metrics(epochs, train_losses, val_losses, map50_scores, map50_95_scores,
                precisions, recalls, ious, args.output_dir,
                tp_counts=tp_counts, fp_counts=fp_counts, fn_counts=fn_counts,
                total_images=len(val_dataset) if not args.quick else None,
                total_detections_list=total_detections_list)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
