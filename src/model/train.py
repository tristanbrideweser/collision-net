"""
train.py — Training loop for the PointNet collision detector.

Features
--------
- Weighted BCE loss to handle class imbalance (collision rates vary 5-60%)
- Feature-transform regularisation (PointNet paper, λ=0.001)
- LR scheduling: cosine annealing with warm restarts
- Gradient clipping
- Best-val-loss checkpointing + final save
- TensorBoard-compatible metric logging (plain text fallback if not installed)
- Early stopping

Location: src/model/train.py

Usage:
    python train.py --data-dir ../data/scenes --epochs 100 --batch-size 64
    python train.py --data-dir ../data/scenes --resume checkpoints/best.pt
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.cuda.amp import GradScaler, autocast
from torch.amp import GradScaler, autocast

# Allow running from src/model/ or src/
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR    = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model.model import (
    PointNetCollisionDetector,
    feature_transform_regulariser,
    count_parameters,
)
from model.dataset import get_dataloaders


# ──────────────────────────────────────────────
# Metrics helpers
# ──────────────────────────────────────────────

def binary_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = (logits > 0).long()
    return (preds == labels).float().mean().item()


@torch.no_grad()
def evaluate(model, loader, criterion, device, reg_lambda=0.001):
    """Return (loss, accuracy, precision, recall, f1) on a dataloader."""
    model.eval()
    total_loss = 0.0
    tp = fp = fn = tn = 0

    for batch in loader:
        pc     = batch["point_cloud"].to(device)   # (B, N, 3)
        cfg    = batch["config"].to(device)        # (B, 7)
        labels = batch["label"].to(device)         # (B,)

        logits, trans = model(pc, cfg)
        loss = criterion(logits, labels.float())
        reg  = feature_transform_regulariser(trans)
        total_loss += (loss + reg_lambda * reg).item() * len(labels)

        preds = (logits > 0).long()
        tp += ((preds == 1) & (labels == 1)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()

    n       = tp + fp + fn + tn
    acc     = (tp + tn) / n if n > 0 else 0.0
    prec    = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec     = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1      = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    avg_loss = total_loss / n if n > 0 else 0.0

    return avg_loss, acc, prec, rec, f1


# ──────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────

def save_checkpoint(path: str, model, optimizer, scheduler, epoch: int,
                    best_val_loss: float, cfg: dict):
    """Save a full resumable checkpoint."""
    torch.save({
        "epoch":         epoch,
        "model_state":   model.state_dict(),
        "optim_state":   optimizer.state_dict(),
        "sched_state":   scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "model_cfg":     cfg,
    }, path)


def save_model_only(path: str, model, model_cfg: dict):
    """Save just the model weights + config for inference."""
    torch.save({
        "model_state": model.state_dict(),
        "model_cfg":   model_cfg,
    }, path)
    print(f"  → Saved inference model: {path}")


def load_checkpoint(path: str, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer  and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])
    if scheduler  and "sched_state" in ckpt:
        scheduler.load_state_dict(ckpt["sched_state"])
    return ckpt.get("epoch", 0), ckpt.get("best_val_loss", float("inf"))


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion,
                    scaler, device, reg_lambda=0.001, grad_clip=1.0):
    model.train()
    total_loss  = 0.0
    total_acc   = 0.0
    n_batches   = 0

    for batch in loader:
        pc     = batch["point_cloud"].to(device, non_blocking=True)
        cfg    = batch["config"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast(enabled=(scaler is not None)):
            logits, trans = model(pc, cfg)
            loss = criterion(logits, labels.float())
            reg  = feature_transform_regulariser(trans)
            total = loss + reg_lambda * reg

        if scaler is not None:
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += total.item()
        total_acc  += binary_accuracy(logits.detach(), labels)
        n_batches  += 1

    return total_loss / n_batches, total_acc / n_batches


def build_criterion(train_loader, device):
    """Compute pos_weight from training label distribution for weighted BCE."""
    all_labels = train_loader.dataset.labels
    n_pos   = int(all_labels.sum())
    n_neg   = len(all_labels) - n_pos
    if n_pos == 0:
        pos_weight = torch.tensor(1.0, device=device)
    else:
        pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32, device=device)
    print(f"  pos_weight = {pos_weight.item():.2f}  "
          f"({n_pos} pos / {n_neg} neg in train split)")
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def main(args):
    # ── Paths ──────────────────────────────────
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path  = str(ckpt_dir / "best.pt")
    final_model_path = str(ckpt_dir / "collision_model.pt")
    log_path        = str(ckpt_dir / "train_log.jsonl")

    # ── Device ────────────────────────────────
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────
    print("Loading datasets …")
    data_dir = os.path.normpath(
        os.path.join(SCRIPT_DIR, args.data_dir)
        if not os.path.isabs(args.data_dir) else args.data_dir
    )
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        n_points=args.n_points,
        num_workers=args.num_workers,
        augment_train=True,
        normalize_pc=True,
    )

    # ── Model ─────────────────────────────────
    model_cfg = dict(
        pc_feat_dim=1024,
        cfg_hidden_dim=256,
        fuse_hidden_dim=512,
        dropout=args.dropout,
        use_input_tnet=True,
        use_feat_tnet=True,
    )
    model = PointNetCollisionDetector(**model_cfg).to(device)
    print(f"Parameters: {count_parameters(model):,}")

    # ── Optimiser ─────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, args.epochs // 3), T_mult=1, eta_min=1e-6
    )

    # Mixed precision (CUDA only)
    scaler = GradScaler() if device.type == "cuda" else None

    # ── Loss ──────────────────────────────────
    criterion = build_criterion(train_loader, device)

    # ── Resume ────────────────────────────────
    start_epoch    = 0
    best_val_loss  = float("inf")
    patience_count = 0

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        start_epoch, best_val_loss = load_checkpoint(
            args.resume, model, optimizer, scheduler
        )
        start_epoch += 1

    # ── Training loop ─────────────────────────
    print(f"\nTraining for {args.epochs} epochs …\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion,
            scaler, device, reg_lambda=args.reg_lambda
        )
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(
            model, val_loader, criterion, device, reg_lambda=args.reg_lambda
        )
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"val_f1={val_f1:.4f}  lr={lr_now:.2e}  "
            f"[{elapsed:.1f}s]"
        )

        # Log to JSONL
        with open(log_path, "a") as f:
            json.dump({
                "epoch": epoch + 1,
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc,
                "val_prec": val_prec, "val_rec": val_rec, "val_f1": val_f1,
                "lr": lr_now,
            }, f)
            f.write("\n")

        # Best checkpoint
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            save_checkpoint(best_ckpt_path, model, optimizer, scheduler,
                            epoch, best_val_loss, model_cfg)
            print(f"  ✓ New best val loss: {best_val_loss:.4f}")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\nEarly stopping: no improvement for {args.patience} epochs.")
                break

    # ── Final evaluation ──────────────────────
    print("\nLoading best checkpoint for final evaluation …")
    load_checkpoint(best_ckpt_path, model)

    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(
        model, test_loader, criterion, device
    )
    print(
        f"\nTest set results:"
        f"\n  loss      = {test_loss:.4f}"
        f"\n  accuracy  = {test_acc:.4f}"
        f"\n  precision = {test_prec:.4f}"
        f"\n  recall    = {test_rec:.4f}"
        f"\n  F1        = {test_f1:.4f}"
    )

    # ── Save inference model ──────────────────
    save_model_only(final_model_path, model, model_cfg)
    print(f"\nAll done.  Inference model saved to: {final_model_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train PointNet collision detector")
    p.add_argument("--data-dir",        default="../data/scenes",   help="Path to scenes directory (absolute or relative to this script)")
    p.add_argument("--epochs",          type=int,   default=100)
    p.add_argument("--batch-size",      type=int,   default=64)
    p.add_argument("--n-points",        type=int,   default=2048)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--dropout",         type=float, default=0.3)
    p.add_argument("--reg-lambda",      type=float, default=0.001,  help="Feature-transform regularisation weight")
    p.add_argument("--patience",        type=int,   default=15,     help="Early-stopping patience (epochs)")
    p.add_argument("--num-workers",     type=int,   default=4)
    p.add_argument("--checkpoint-dir",  default="checkpoints")
    p.add_argument("--resume",          default=None,               help="Path to checkpoint to resume from")
    args = p.parse_args()

    main(args)
