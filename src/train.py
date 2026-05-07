import os
import shutil
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# --- DDP Imports ---
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from model.collisionnet import CollisionNet
from model.dataset import CollisionDataset

def feature_transform_reg_loss(trans_feat):
    """Enforces orthogonality on the PointNet feature transform matrix."""
    if trans_feat is None:
        return 0.0
    B, K, _ = trans_feat.size()
    I = torch.eye(K, device=trans_feat.device).unsqueeze(0)
    A_At = torch.bmm(trans_feat, trans_feat.transpose(2, 1))
    loss = torch.mean(torch.norm(A_At - I, dim=(1, 2)))
    return loss

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./data/scenes")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128) 
    parser.add_argument("--lr", type=float, default=1e-4) # Lowered for stability
    parser.add_argument("--reg-weight", type=float, default=0.001)
    parser.add_argument("--log-dir", type=str, default="./runs/collisionnet_ddp")
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    # 1. DDP Initialization
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = dist.get_world_size()
    
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    # Ensure paths are absolute and within the repo
    save_path = os.path.abspath(args.save_dir)
    log_path = os.path.abspath(args.log_dir)
    
    if global_rank == 0:
        print(f"--- Firing up DDP with Modern AMP (Torch 2.11+) ---")
        os.makedirs(save_path, exist_ok=True)
        writer = SummaryWriter(log_path)
    else:
        writer = None

    # 2. Datasets & Loaders
    train_ds = CollisionDataset(args.data_dir, split="train", augment=True)
    val_ds = CollisionDataset(args.data_dir, split="val", augment=False)

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler, 
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, sampler=val_sampler, 
        num_workers=args.num_workers, pin_memory=True
    )

    # 3. Model, Loss, Optimizer
    model = CollisionNet(embed_dim=1024, num_heads=8).to(device)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # --- MODERN AMP SCALER ---
    scaler = torch.amp.GradScaler('cuda')

    # 4. Auto-Resume Logic
    start_epoch = 1
    best_val_loss = float('inf')
    latest_ckpt_path = os.path.join(save_path, "collisionnet_latest.pth")

    if os.path.exists(latest_ckpt_path):
        if global_rank == 0: print(f"[INFO] Resuming from {latest_ckpt_path}...")
        checkpoint = torch.load(latest_ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['val_loss']

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 5. Training Loop
    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        total_train_loss, train_correct, train_total = 0.0, 0, 0
        
        # tqdm refresh set to 60s for cluster log friendliness
        pbar = tqdm(train_loader, desc=f"Ep {epoch} [Train]", mininterval=60, ascii=True) if global_rank == 0 else train_loader
        
        for i, batch in enumerate(pbar):
            robot_pts = batch["robot_keypoints"].to(device)
            scene_pts = batch["point_cloud"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            # --- MODERN AMP FORWARD ---
            with torch.amp.autocast('cuda'):
                logits, _, trans_feat = model(robot_pts, scene_pts)
                bce_loss = criterion(logits, labels)
                reg_loss = feature_transform_reg_loss(trans_feat)
                loss = bce_loss + (args.reg_weight * reg_loss)
            
            # --- STABILIZED BACKWARD ---
            scaler.scale(loss).backward()
            
            # Unscale before clipping to ensure math is in FP32 range
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            # Metrics
            total_train_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            # Safety Checkpoint (every 500 batches)
            if global_rank == 0 and i % 500 == 0:
                if i > 0: pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'val_loss': best_val_loss,
                }, latest_ckpt_path)

        # 6. Validation Loop
        model.eval()
        total_val_loss, val_correct, val_total = 0.0, 0, 0
        val_pbar = tqdm(val_loader, desc=f"Ep {epoch} [Val]", mininterval=60, ascii=True) if global_rank == 0 else val_loader
        
        with torch.no_grad():
            for batch in val_pbar:
                robot_pts = batch["robot_keypoints"].to(device)
                scene_pts = batch["point_cloud"].to(device)
                labels = batch["label"].to(device)

                with torch.amp.autocast('cuda'):
                    logits, _, trans_feat = model(robot_pts, scene_pts)
                    val_loss = criterion(logits, labels) + (args.reg_weight * feature_transform_reg_loss(trans_feat))
                
                total_val_loss += val_loss.item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        if global_rank == 0:
            print(f"\nEpoch {epoch} | Val Loss: {avg_val_loss:.4f} | Acc: {(val_correct/val_total)*100:.2f}%")
            writer.add_scalar("Loss/Val", avg_val_loss, epoch)
            
            # Save Latest
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_loss': avg_val_loss,
            }, latest_ckpt_path)

            # Save Best
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                shutil.copyfile(latest_ckpt_path, os.path.join(save_path, "collisionnet_best.pth"))

    if global_rank == 0: writer.close()
    dist.destroy_process_group()

if __name__ == "__main__":
    train()