import os
import shutil
import argparse
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ddp + scheduling imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, SequentialLR

from model.collisionnet import CollisionNet
from model.dataset import CollisionDataset

def feature_transform_reg_loss(trans_feat):
    """Enforces orthogonality on the PointNet feature transform matrix."""
    if trans_feat is None:
        return 0.0
    B, K, _ = trans_feat.size()
    I = torch.eye(K, device=trans_feat.device).unsqueeze(0)
    A_At = torch.bmm(trans_feat, trans_feat.transpose(2, 1))
    return torch.mean(torch.norm(A_At - I, dim=(1, 2)))

@torch.no_grad()
def evaluate_metrics(model, loader, criterion, device, reg_weight=0.001):
    """Calculates research-grade metrics for the final report."""
    model.eval()
    tp, fp, fn, tn = 0, 0, 0, 0
    total_loss = 0.0

    for batch in loader:
        robot_pts = batch["robot_keypoints"].to(device)
        scene_pts = batch["point_cloud"].to(device)
        labels = batch["label"].to(device)

        with torch.amp.autocast('cuda'):
            logits, _, trans_feat = model(robot_pts, scene_pts)
            loss = criterion(logits, labels) + (reg_weight * feature_transform_reg_loss(trans_feat))
        
        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).float()
        
        tp += ((preds == 1) & (labels == 1)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()

    n = tp + fp + fn + tn
    acc = (tp + tn) / n if n > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    
    return total_loss / len(loader), acc, prec, rec, f1

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./data/scenes")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128) 
    parser.add_argument("--lr", type=float, default=1e-4) 
    parser.add_argument("--reg-weight", type=float, default=0.001)
    parser.add_argument("--log-dir", type=str, default="./runs/collisionnet_final")
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    # ddp init
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    save_path = os.path.abspath(args.save_dir)
    log_file = os.path.join(save_path, "train_log.jsonl")
    
    if global_rank == 0:
        print(f"--- Firing up Merged Trainer (DDP + AMP + Metrics) ---")
        os.makedirs(save_path, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    # data
    train_ds = CollisionDataset(args.data_dir, split="train", augment=True)
    val_ds = CollisionDataset(args.data_dir, split="val", augment=False)
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.num_workers, pin_memory=True)

    # model + optim
    model = CollisionNet(embed_dim=1024, num_heads=8).to(device)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 5-epoch warmup + cosine annealing
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=5)
    cosine = CosineAnnealingWarmRestarts(optimizer, T_0=max(1, (args.epochs - 5) // 3))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])

    scaler = torch.amp.GradScaler('cuda')
    start_epoch, best_val_loss = 1, float('inf')
    latest_ckpt = os.path.join(save_path, "collisionnet_latest.pth")

    # auto-resume
    if os.path.exists(latest_ckpt):
        checkpoint = torch.load(latest_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch, best_val_loss = checkpoint['epoch'] + 1, checkpoint['val_loss']

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank])

    # training loop
    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch}", mininterval=60, ascii=True) if global_rank == 0 else train_loader
        
        for batch in pbar:
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits, _, trans = model(batch["robot_keypoints"].to(device), batch["point_cloud"].to(device))
                loss = criterion(logits, batch["label"].to(device)) + (args.reg_weight * feature_transform_reg_loss(trans))
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        # eval + logging
        v_loss, v_acc, v_prec, v_rec, v_f1 = evaluate_metrics(model, val_loader, criterion, device)
        scheduler.step()

        if global_rank == 0:
            print(f"Epoch {epoch} | Val Loss: {v_loss:.4f} | Acc: {v_acc*100:.2f}% | F1: {v_f1:.4f}")
            
            # TensorBoard
            writer.add_scalar("Loss/Val", v_loss, epoch)
            writer.add_scalar("Metrics/F1", v_f1, epoch)
            writer.add_scalar("Metrics/Recall", v_rec, epoch)

            # JSONL Log
            with open(log_file, "a") as f:
                f.write(json.dumps({"epoch": epoch, "loss": v_loss, "acc": v_acc, "f1": v_f1, "lr": optimizer.param_groups[0]['lr']}) + "\n")

            # Checkpointing
            ckpt_data = {
                'epoch': epoch, 'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': v_loss
            }
            torch.save(ckpt_data, latest_ckpt)
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                shutil.copyfile(latest_ckpt, os.path.join(save_path, "collisionnet_best.pth"))

    if global_rank == 0: writer.close()
    dist.destroy_process_group()

if __name__ == "__main__":
    train()