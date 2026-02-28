"""
ScanSure AI v2 — train.py
Training pipeline for LightUNet on BraTS 2023 dataset.

Usage:
    python train.py --data_dir /path/to/BraTS2023 --epochs 20

Saves best model weights to: checkpoints/best_model.pth
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# ── Allow imports from v1 or same folder ──────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from unet_model import LightUNet
from utils import BCEDiceLoss, get_device
from brats_dataset import BraTSDataset


# ──────────────────────────────────────────────────────────────
# Argument Parser
# ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="ScanSure AI v2 — BraTS Training")
    parser.add_argument("--data_dir",    type=str,   required=True,  help="Path to BraTS2023 root folder")
    parser.add_argument("--epochs",      type=int,   default=20,     help="Number of training epochs")
    parser.add_argument("--batch_size",  type=int,   default=8,      help="Batch size")
    parser.add_argument("--lr",          type=float, default=1e-3,   help="Learning rate")
    parser.add_argument("--img_size",    type=int,   default=256,    help="Slice resize dimension")
    parser.add_argument("--max_patients",type=int,   default=None,   help="Cap patients (for quick tests)")
    parser.add_argument("--val_split",   type=float, default=0.2,    help="Validation fraction")
    parser.add_argument("--save_dir",    type=str,   default="checkpoints", help="Where to save weights")

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────
# Metric: Dice Score
# ──────────────────────────────────────────────────────────────

def dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum()
    return (2 * intersection / (pred_bin.sum() + target.sum() + 1e-8)).item()


# ──────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_dice = 0.0

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss  = criterion(preds, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice += dice_score(preds.detach(), masks)

        if (batch_idx + 1) % 10 == 0:
            print(f"    Batch [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {loss.item():.4f}")

    return total_loss / len(loader), total_dice / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks  = masks.to(device)
        preds  = model(images)
        loss   = criterion(preds, masks)
        total_loss += loss.item()
        total_dice += dice_score(preds, masks)

    return total_loss / len(loader), total_dice / len(loader)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = get_device()

    # ── Dataset ───────────────────────────────────────────────
    print("\n[ScanSure AI v2] Loading BraTS dataset…")
    full_dataset = BraTSDataset(
        root_dir=args.data_dir,

        img_size=args.img_size,
        max_patients=args.max_patients,
    )

    if len(full_dataset) == 0:
        print("❌  No slices found. Check your BraTS folder structure.")
        sys.exit(1)

    val_size   = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"  Train slices : {train_size}")
    print(f"  Val   slices : {val_size}")

    # ── Model ─────────────────────────────────────────────────
    model     = LightUNet(in_channels=1, dropout_p=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)

    # ── Checkpoint dir ────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_ckpt     = os.path.join(args.save_dir, "best_model.pth")

    # ── Training loop ─────────────────────────────────────────
    print(f"\n[ScanSure AI v2] Starting training for {args.epochs} epochs…\n")

    history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}

    for epoch in range(1, args.epochs + 1):
        print(f"─── Epoch {epoch}/{args.epochs} ───────────────────────────")

        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_dice   = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)

        print(f"  Train → Loss: {train_loss:.4f}  Dice: {train_dice:.4f}")
        print(f"  Val   → Loss: {val_loss:.4f}  Dice: {val_dice:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":      epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":   val_loss,
                "val_dice":   val_dice,
                "args":       vars(args),
            }, best_ckpt)
            print(f"  ✅  Saved best model → {best_ckpt}  (val_loss={val_loss:.4f})")

        print()

    print(f"[ScanSure AI v2] Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best weights saved at: {best_ckpt}")

    # ── Save training history ─────────────────────────────────
    history_path = os.path.join(args.save_dir, "training_history.pt")
    torch.save(history, history_path)
    print(f"Training history saved at: {history_path}")


if __name__ == "__main__":
    main()
