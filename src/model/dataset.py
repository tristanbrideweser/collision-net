# src/model/dataset.py
"""
dataset.py — PyTorch Dataset for Cartesian Cross-Attention CollisionNet.

Loads the output from generate_dataset.py and prepares batches for training.
Each sample consists of:
    - Obstacle point cloud (N, 3)
    - Robot Cartesian keypoints (9, 3)
    - Binary collision label (1,)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CollisionDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        n_points: int = 2048,
        augment: bool = False,
        normalize_pc: bool = True,
    ):
        self.data_dir = data_dir
        self.split = split
        self.n_points = n_points
        self.augment = augment and (split == "train")
        self.normalize_pc = normalize_pc
        
        # Load split data
        split_file = os.path.join(data_dir, f"{split}.npz")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        data = np.load(split_file)
        self.scene_ids = data["scene_ids"]
        self.configs = data["configs"] # Now expected to be (Samples, 9, 3)
        self.labels = data["labels"]
        
        # Cache for loaded scene point clouds
        self._scene_cache = {}
        self._preload_scenes()
        
        print(f"Loaded {split} split: {len(self)} samples, "
              f"{len(self._scene_cache)} unique scenes, "
              f"{self.labels.sum()}/{len(self.labels)} collisions "
              f"({self.labels.mean()*100:.1f}%)")
    
    def _preload_scenes(self):
        unique_scenes = np.unique(self.scene_ids)
        scenes_subdir = os.path.join(self.data_dir, "scenes")
        for scene_id in unique_scenes:
            scene_file = os.path.join(scenes_subdir, f"scene_{scene_id:04d}.npz")
            if os.path.exists(scene_file):
                self._scene_cache[scene_id] = np.load(scene_file)["point_cloud"]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        scene_id = self.scene_ids[idx]
        robot_keypoints = self.configs[idx].copy() # (9, 3)
        label = self.labels[idx]
        
        if scene_id in self._scene_cache:
            point_cloud = self._scene_cache[scene_id].copy()
        else:
            scene_file = os.path.join(self.data_dir, "scenes", f"scene_{scene_id:04d}.npz")
            point_cloud = np.load(scene_file)["point_cloud"].copy()
        
        # 1. Resample scene
        point_cloud = self._resample(point_cloud)
        
        # 2. Unified Spatial Normalization
        if self.normalize_pc:
            point_cloud, robot_keypoints = self._normalize_scene(point_cloud, robot_keypoints)
        
        # 3. Unified Data Augmentation
        if self.augment:
            point_cloud, robot_keypoints = self._augment_scene(point_cloud, robot_keypoints)
        
        return {
            "point_cloud": torch.from_numpy(point_cloud).float(),       # (N, 3)
            "robot_keypoints": torch.from_numpy(robot_keypoints).float(), # (9, 3)
            "label": torch.tensor([label], dtype=torch.float32),        # (1,) for BCEWithLogitsLoss
            "scene_id": scene_id,
        }
    
    def _resample(self, points: np.ndarray) -> np.ndarray:
        n = len(points)
        if n == 0:
            return np.zeros((self.n_points, 3), dtype=np.float32)
        if n >= self.n_points:
            indices = np.random.choice(n, self.n_points, replace=False)
        else:
            indices = np.random.choice(n, self.n_points, replace=True)
        return points[indices]
    
    def _normalize_scene(self, points: np.ndarray, keypoints: np.ndarray):
        """Center and scale BOTH point cloud and robot to unit sphere."""
        centroid = points.mean(axis=0)
        
        # Shift both
        points = points - centroid
        keypoints = keypoints - centroid
        
        # Scale both
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 0:
            points = points / max_dist
            keypoints = keypoints / max_dist
            
        return points, keypoints
    
    def _augment_scene(self, points: np.ndarray, keypoints: np.ndarray):
        """Apply random augmentations to both geometries to preserve spatial relations."""
        # Random rotation around z-axis
        if np.random.rand() > 0.5:
            theta = np.random.uniform(0, 2 * np.pi)
            rot = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta),  np.cos(theta), 0],
                [0, 0, 1]
            ])
            points = points @ rot.T
            keypoints = keypoints @ rot.T
        
        # Random scale
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            points = points * scale
            keypoints = keypoints * scale
            
        # Random jitter (Apply ONLY to scene to simulate depth sensor noise)
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.01, points.shape)
            points = points + noise
        
        return points.astype(np.float32), keypoints.astype(np.float32)

def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    n_points: int = 2048,
    num_workers: int = 4,
    augment_train: bool = True,
    normalize_pc: bool = True,
):
    train_ds = CollisionDataset(
        data_dir, split="train", n_points=n_points,
        augment=augment_train, normalize_pc=normalize_pc
    )
    val_ds = CollisionDataset(
        data_dir, split="val", n_points=n_points,
        augment=False, normalize_pc=normalize_pc
    )
    test_ds = CollisionDataset(
        data_dir, split="test", n_points=n_points,
        augment=False, normalize_pc=normalize_pc
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../scenes")
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(script_dir, args.data_dir))
    
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=32,
        num_workers=0,
    )
    
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  point_cloud:     {batch['point_cloud'].shape}")
    print(f"  robot_keypoints: {batch['robot_keypoints'].shape}")
    print(f"  label:           {batch['label'].shape}")