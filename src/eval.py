import torch
import numpy as np
from torch.utils.data import DataLoader
from model.collisionnet import CollisionNet
from model.dataset import CollisionDataset
from sklearn.metrics import classification_report, confusion_matrix
import argparse

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/scenes")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/collisionnet_best.pth")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    test_ds = CollisionDataset(args.data_dir, split="test", augment=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=4)

    # 2. Load Model
    model = CollisionNet(embed_dim=1024, num_heads=8).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds = []
    all_labels = []

    print(f"--- Evaluating {args.checkpoint} on Test Split ---")
    with torch.no_grad():
        for batch in test_loader:
            robot_pts = batch["robot_keypoints"].to(device)
            scene_pts = batch["point_cloud"].to(device)
            labels = batch["label"].to(device)

            logits, _, _ = model(robot_pts, scene_pts)
            preds = (torch.sigmoid(logits) > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 3. Metrics
    print("\n[Final Metrics]")
    print(classification_report(all_labels, all_preds, target_names=["No Collision", "Collision"]))
    
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    evaluate()