# src/train.py
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
    parser.add_argument("--data-dir", type=str, default="../data/scenes")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128) # Note: This is PER GPU
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reg-weight", type=float, default=0.001)
    parser.add_argument("--log-dir", type=str, default="../runs/collisionnet_ddp")
    parser.add_argument("--save-dir", type=str, default="../checkpoints")
    args = parser.parse_args()

    # 1. DDP Initialization
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = dist.get_world_size()
    
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    if global_rank == 0:
        print(f"--- Firing up DDP across {world_size} GPUs ---")
        os.makedirs(args.save_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    # 2. Datasets & Distributed Samplers
    train_ds = CollisionDataset(args.data_dir, split="train", augment=True)
    val_ds = CollisionDataset(args.data_dir, split="val", augment=False)

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler, 
        num_workers=8, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, sampler=val_sampler, 
        num_workers=8, pin_memory=True
    )

    # 3. Model, Loss, Optimizer Setup
    model = CollisionNet(embed_dim=1024, num_heads=8).to(device)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # 4. Auto-Resume Logic (Survive the 4-hour limit)
    start_epoch = 1
    best_val_loss = float('inf')
    latest_ckpt_path = os.path.join(args.save_dir, "collisionnet_latest.pth")

    if os.path.exists(latest_ckpt_path):
        if global_rank == 0:
            print(f"\n[INFO] Found interrupted run. Resuming from {latest_ckpt_path}...\n")
        
        # Load weights into the specific GPU's memory
        checkpoint = torch.load(latest_ckpt_path, map_location=device, weights_only=True)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']

    # 5. DDP Wrapper & SyncBatchNorm
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 6. The Training Loop
    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        
        total_train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]") if global_rank == 0 else train_loader
        
        for batch in pbar:
            robot_pts = batch["robot_keypoints"].to(device)
            scene_pts = batch["point_cloud"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits, _, trans_feat = model(robot_pts, scene_pts)
            
            bce_loss = criterion(logits, labels)
            reg_loss = feature_transform_reg_loss(trans_feat)
            loss = bce_loss + (args.reg_weight * reg_loss)
            
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            if global_rank == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = train_correct / train_total

        # 7. The Validation Loop
        model.eval()
        total_val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]") if global_rank == 0 else val_loader
        
        with torch.no_grad():
            for batch in val_pbar:
                robot_pts = batch["robot_keypoints"].to(device)
                scene_pts = batch["point_cloud"].to(device)
                labels = batch["label"].to(device)

                logits, _, trans_feat = model(robot_pts, scene_pts)
                
                bce_loss = criterion(logits, labels)
                reg_loss = feature_transform_reg_loss(trans_feat)
                loss = bce_loss + (args.reg_weight * reg_loss)
                
                total_val_loss += loss.item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        scheduler.step(avg_val_loss)

        # 8. Logging and Saving (Rank 0 ONLY)
        if global_rank == 0:
            print(f"\nEpoch {epoch} Summary:")
            print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
            print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc*100:.2f}%\n")

            writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            writer.add_scalar("Loss/Val", avg_val_loss, epoch)
            writer.add_scalar("Acc/Train", train_acc, epoch)
            writer.add_scalar("Acc/Val", val_acc, epoch)

            # ALWAYS save the latest state so Slurm can pick it up if killed
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': best_val_loss,
            }, latest_ckpt_path)

            # ONLY save the best model if it actually improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_save_path = os.path.join(args.save_dir, "collisionnet_best.pth")
                shutil.copyfile(latest_ckpt_path, best_save_path)
                print(f"*** New personal best! Saved to {best_save_path} ***\n")

    if global_rank == 0:
        writer.close()
        print("Distributed Training Complete!")
        
    dist.destroy_process_group()

if __name__ == "__main__":
    train()