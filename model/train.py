"""
Faster R-CNN Training Script for License Plate Detection.

Model: Faster R-CNN with ResNet50-FPN backbone (pretrained on COCO).
Fine-tuned for single-class (license plate) detection.

Usage:
    python train.py --dataset-root LP_detection --epochs 30 --batch-size 4 --lr 0.005
"""

import os
import time
import argparse
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN
from torch.utils.data import DataLoader

from data_preprocessing.dataset import LicensePlateDataset, get_transforms, collate_fn


def get_model(num_classes, backbone_name='resnet50', pretrained=True):
    """
    Build Faster R-CNN with ResNet50-FPN v2 backbone.

    Uses COCO-pretrained weights and replaces the classification head
    with a new one for our number of classes.

    Args:
        num_classes: Number of classes INCLUDING background.
                     For license plate: 2 (background + license_plate).
        pretrained: Whether to use pretrained backbone weights.

    Returns:
        Faster R-CNN model ready for fine-tuning.
    """
    if backbone_name == 'resnet50':
        if pretrained:
            # Load model with COCO-pretrained weights (better than just ImageNet backbone)
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        else:
            model = fasterrcnn_resnet50_fpn_v2(weights=None)

        # Replace the box predictor head for our number of classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        return model

    elif backbone_name == 'resnet18':
        # Build a ResNet18 backbone with an FPN on top. `resnet_fpn_backbone`
        # constructs the feature extractor compatible with torchvision's
        # detection heads.
        backbone = resnet_fpn_backbone('resnet18', pretrained=pretrained, trainable_layers=3)

        # Create a Faster R-CNN model using the custom backbone. Passing
        # `num_classes` makes the constructor create an appropriate predictor.
        model = FasterRCNN(backbone, num_classes=num_classes)

        return model

    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50):
    """Train for one epoch, printing loss every `print_freq` batches."""
    model.train()
    running_loss = 0.0
    total_batches = len(data_loader)

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass - model returns losses in training mode
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Check for NaN
        if not torch.isfinite(losses):
            print(f"WARNING: Non-finite loss ({losses.item()}), skipping batch {batch_idx}")
            continue

        # Backward pass
        optimizer.zero_grad()
        losses.backward()

        # Gradient clipping to stabilize training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += losses.item()

        if (batch_idx + 1) % print_freq == 0 or batch_idx == 0:
            avg_loss = running_loss / (batch_idx + 1)
            loss_str = "  ".join(f"{k}: {v.item():.4f}" for k, v in loss_dict.items())
            print(f"  Epoch [{epoch}] Batch [{batch_idx+1}/{total_batches}]  "
                  f"Loss: {losses.item():.4f}  Avg: {avg_loss:.4f}  ({loss_str})")

    return running_loss / total_batches


@torch.no_grad()
def evaluate(model, data_loader, device):
    """
    Run evaluation and compute average loss.

    For proper mAP evaluation, use pycocotools (see evaluate_map function below).
    """
    model.train()  # Keep in train mode to get losses
    total_loss = 0.0
    num_batches = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


@torch.no_grad()
def evaluate_map(model, data_loader, device, annotation_file):
    """
    Evaluate using COCO mAP metrics.

    Args:
        model: The trained model.
        data_loader: Validation data loader.
        device: Device to run on.
        annotation_file: Path to the COCO-format annotation JSON.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    model.eval()
    coco_gt = COCO(annotation_file)

    results = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]

        outputs = model(images)

        for target, output in zip(targets, outputs):
            image_id = target["image_id"].item()

            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                results.append({
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score)
                })

    if not results:
        print("No detections to evaluate!")
        return 0.0

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats[0]  # mAP@[0.5:0.95]


def main():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN for License Plate Detection")
    parser.add_argument("--dataset-root", type=str, default="LP_detection",
                        help="Root directory of the dataset")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="Initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=0.0005,
                        help="Weight decay (L2 regularization)")
    parser.add_argument("--lr-step-size", type=int, default=10,
                        help="StepLR: decay LR every N epochs")
    parser.add_argument("--lr-gamma", type=float, default=0.1,
                        help="StepLR: multiply LR by this factor")
    parser.add_argument("--backbone", type=str, choices=['resnet50','resnet18'], default='resnet50',
                            help="Backbone network to use (resnet50 or resnet18)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run evaluation, no training")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Don't use pretrained COCO weights")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Paths
    train_images = os.path.join(args.dataset_root, "images", "train")
    val_images = os.path.join(args.dataset_root, "images", "val")
    train_ann = os.path.join(args.dataset_root, "annotations", "instances_train.json")
    val_ann = os.path.join(args.dataset_root, "annotations", "instances_val.json")

    # Datasets
    print("\n--- Loading Datasets ---")
    train_dataset = LicensePlateDataset(
        images_dir=train_images,
        annotation_file=train_ann,
        transforms=get_transforms(train=True)
    )
    val_dataset = LicensePlateDataset(
        images_dir=val_images,
        annotation_file=val_ann,
        transforms=get_transforms(train=False)
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == "cuda" else False,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Model: 2 classes = background + license_plate
    num_classes = 2
    model = get_model(num_classes, backbone_name=args.backbone, pretrained=not args.no_pretrained)
    model.to(device)

    print(f"\nModel: Faster R-CNN")
    print(f"  Number of classes: {num_classes} (background + license_plate)")
    print(f"  Backbone: {args.backbone.upper()}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Optimizer & Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
    )

    start_epoch = 0

    # Resume from checkpoint
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"  Resumed at epoch {start_epoch}")

    # Eval-only mode
    if args.eval_only:
        print("\n--- Evaluation Only ---")
        map_score = evaluate_map(model, val_loader, device, val_ann)
        print(f"\nmAP@[0.5:0.95]: {map_score:.4f}")
        return

    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float('inf')

    print(f"\n{'='*60}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  LR schedule: StepLR(step={args.lr_step_size}, gamma={args.lr_gamma})")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n--- Epoch {epoch+1}/{args.epochs} (LR: {current_lr:.6f}) ---")

        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch + 1)

        # Validate
        val_loss = evaluate(model, val_loader, device)

        # Step LR scheduler
        lr_scheduler.step()

        epoch_time = time.time() - epoch_start
        print(f"\n  Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}  |  "
              f"Time: {epoch_time:.1f}s")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }

        # Save latest
        torch.save(checkpoint, os.path.join(args.output_dir, "latest.pth"))

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(args.output_dir, "best.pth"))
            print(f"  * New best model saved (val_loss: {val_loss:.4f})")

        # Save every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(checkpoint, os.path.join(args.output_dir, f"epoch_{epoch+1}.pth"))

    # Final mAP evaluation
    print(f"\n{'='*60}")
    print("Final Evaluation (loading best model)")
    print(f"{'='*60}")
    best_ckpt = torch.load(os.path.join(args.output_dir, "best.pth"), map_location=device)
    model.load_state_dict(best_ckpt['model_state_dict'])
    map_score = evaluate_map(model, val_loader, device, val_ann)
    print(f"\nFinal mAP@[0.5:0.95]: {map_score:.4f}")


if __name__ == "__main__":
    main()
