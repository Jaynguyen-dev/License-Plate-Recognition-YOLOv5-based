"""
Simple script to extract metrics from checkpoints and create visualizations.
"""

import os
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Get checkpoint directory
checkpoint_dir = "checkpoints_resnet18"
output_dir = "metrics_plots"
os.makedirs(output_dir, exist_ok=True)

# Load checkpoints
checkpoints_info = {}
checkpoint_files = sorted(Path(checkpoint_dir).glob("epoch_*.pth"))

print("Loading checkpoints...")
for checkpoint_file in checkpoint_files:
    name = checkpoint_file.stem
    if name.startswith("epoch_"):
        try:
            epoch = int(name.split("_")[1])
            checkpoint = torch.load(str(checkpoint_file), map_location='cpu')
            train_loss = checkpoint.get('train_loss', 0.0)
            val_loss = checkpoint.get('val_loss', 0.0)
            checkpoints_info[epoch] = {'train_loss': train_loss, 'val_loss': val_loss}
            print(f"  Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        except Exception as e:
            print(f"  Error loading {checkpoint_file}: {e}")

# Sort by epoch
sorted_epochs = sorted(checkpoints_info.keys())
epochs = sorted_epochs
train_losses = [checkpoints_info[e]['train_loss'] for e in epochs]
val_losses = [checkpoints_info[e]['val_loss'] for e in epochs]

print(f"\nTotal epochs with data: {len(epochs)}")

# Create plots
print("\nGenerating plots...")

# Plot 1: Loss curves
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(epochs, train_losses, marker='o', label='Train Loss', linewidth=2.5, markersize=8, color='#1f77b4')
ax.plot(epochs, val_losses, marker='s', label='Validation Loss', linewidth=2.5, markersize=8, color='#ff7f0e')
ax.fill_between(epochs, train_losses, alpha=0.15, color='#1f77b4')
ax.fill_between(epochs, val_losses, alpha=0.15, color='#ff7f0e')
ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax.set_ylabel('Loss', fontsize=13, fontweight='bold')
ax.set_title('Training vs Validation Loss Over Epochs', fontsize=15, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='best', framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xticks(epochs)
plt.tight_layout()
loss_file = os.path.join(output_dir, '01_loss_curve.png')
plt.savefig(loss_file, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {loss_file}")
plt.close()

# Plot 2: Train loss detail
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(epochs, train_losses, marker='o', linewidth=2.5, markersize=8, color='#1f77b4')
ax.fill_between(epochs, train_losses, alpha=0.3, color='#1f77b4')
ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax.set_ylabel('Training Loss', fontsize=13, fontweight='bold')
ax.set_title('Training Loss Over Epochs', fontsize=15, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xticks(epochs)
plt.tight_layout()
train_file = os.path.join(output_dir, '02_training_loss.png')
plt.savefig(train_file, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {train_file}")
plt.close()

# Plot 3: Validation loss detail
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(epochs, val_losses, marker='s', linewidth=2.5, markersize=8, color='#ff7f0e')
ax.fill_between(epochs, val_losses, alpha=0.3, color='#ff7f0e')
ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax.set_ylabel('Validation Loss', fontsize=13, fontweight='bold')
ax.set_title('Validation Loss Over Epochs', fontsize=15, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xticks(epochs)
plt.tight_layout()
val_file = os.path.join(output_dir, '03_validation_loss.png')
plt.savefig(val_file, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {val_file}")
plt.close()

# Plot 4: Loss difference (val - train)
loss_diff = [v - t for v, t in zip(val_losses, train_losses)]
fig, ax = plt.subplots(figsize=(12, 6))
colors = ['#d62728' if diff > 0 else '#2ca02c' for diff in loss_diff]
ax.bar(epochs, loss_diff, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax.set_ylabel('Loss Difference (Val - Train)', fontsize=13, fontweight='bold')
ax.set_title('Validation vs Training Loss Difference (Overfitting Indicator)', fontsize=15, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')
ax.set_xticks(epochs)
plt.tight_layout()
diff_file = os.path.join(output_dir, '04_loss_difference.png')
plt.savefig(diff_file, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {diff_file}")
plt.close()

# Summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Number of epochs: {len(epochs)}")
print(f"Epochs: {epochs[0]} - {epochs[-1]}")
print(f"\nTraining Loss:")
print(f"  Min: {min(train_losses):.4f} (epoch {epochs[train_losses.index(min(train_losses))]})")
print(f"  Max: {max(train_losses):.4f} (epoch {epochs[train_losses.index(max(train_losses))]})")
print(f"  Final: {train_losses[-1]:.4f}")
print(f"\nValidation Loss:")
print(f"  Min: {min(val_losses):.4f} (epoch {epochs[val_losses.index(min(val_losses))]})")
print(f"  Max: {max(val_losses):.4f} (epoch {epochs[val_losses.index(max(val_losses))]})")
print(f"  Final: {val_losses[-1]:.4f}")

print(f"\nAll plots saved to: {output_dir}/")
print("="*60)
