# src/eval.py

import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from model.collisionnet import CollisionNet
from model.dataset import CollisionDataset
from sklearn.metrics import classification_report, confusion_matrix
import argparse

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="../data/scenes")
    parser.add_argument("--checkpoint", type=str, default="../checkpoints/collisionnet_best.pth")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=1000, help="limit samples for local eval")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    test_ds = CollisionDataset(args.data_dir, split="test", augment=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    
    if args.limit > 0:
        indices = np.arange(args.limit)
        test_ds.scene_ids = test_ds.scene_ids[indices]
        test_ds.configs = test_ds.configs[indices]
        test_ds.labels = test_ds.labels[indices]
        print(f"🚀 Running Quick-Check on {args.limit} samples...")

    # 2. Load Model
    model = CollisionNet(embed_dim=1024, num_heads=8).to(device)
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint {args.checkpoint} not found!")
        return

    # Load checkpoint - weights_only=True is faster and more secure
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds = []
    all_labels = []

    print(f"--- Evaluating {args.checkpoint} on Test Split ---")
    
    # Use no_grad AND autocast for the evaluation sprint
    with torch.no_grad():
        for batch in test_loader:
            robot_pts = batch["robot_keypoints"].to(device)
            scene_pts = batch["point_cloud"].to(device)
            labels = batch["label"].to(device)

            # Use autocast to match training precision
            with torch.cuda.amp.autocast():
                logits, _, _ = model(robot_pts, scene_pts)
                # Sigmoid and thresholding
                preds = (torch.sigmoid(logits) > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 3. Final Report
    print("\n" + "="*30)
    print("FINAL TEST PERFORMANCE")
    print("="*30)
    print(classification_report(all_labels, all_preds, target_names=["No Collision", "Collision"]))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    evaluate()