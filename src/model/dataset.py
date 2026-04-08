"""
dataset.py — PyTorch Dataset for collision detection with PointNet.

Loads the output from generate_dataset.py and prepares batches for training.
Each sample consists of:
    - Obstacle point cloud (N, 3) from scene file
    - Robot configuration (7,) normalized to [-1, 1]
    - Binary collision label

Location: src/model/dataset.py

Usage:
    from dataset import CollisionDataset, get_dataloaders
    
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir="../data/scenes",
        batch_size=64,
        num_workers=4,
    )
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CollisionDataset(Dataset):
    """Dataset for collision detection training.
    
    Loads pre-generated data from generate_dataset.py output.
    
    Args:
        data_dir: Path to scenes directory containing train/val/test.npz and scenes/
        split: One of 'train', 'val', 'test'
        n_points: Number of points to sample from point cloud (default: 2048)
        augment: Whether to apply data augmentation (only for training)
        normalize_pc: Whether to normalize point clouds to unit sphere
    """
    
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
        self.configs = data["configs"]
        self.labels = data["labels"]
        
        # Cache for loaded scene point clouds
        self._scene_cache = {}
        
        # Preload all unique scenes (memory permitting)
        self._preload_scenes()
        
        print(f"Loaded {split} split: {len(self)} samples, "
              f"{len(self._scene_cache)} unique scenes, "
              f"{self.labels.sum()}/{len(self.labels)} collisions "
              f"({self.labels.mean()*100:.1f}%)")
    
    def _preload_scenes(self):
        """Preload all scene point clouds into memory."""
        unique_scenes = np.unique(self.scene_ids)
        scenes_subdir = os.path.join(self.data_dir, "scenes")
        
        for scene_id in unique_scenes:
            scene_file = os.path.join(scenes_subdir, f"scene_{scene_id:04d}.npz")
            if os.path.exists(scene_file):
                scene_data = np.load(scene_file)
                self._scene_cache[scene_id] = scene_data["point_cloud"]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        scene_id = self.scene_ids[idx]
        config = self.configs[idx].copy()
        label = self.labels[idx]
        
        # Get point cloud from cache
        if scene_id in self._scene_cache:
            point_cloud = self._scene_cache[scene_id].copy()
        else:
            # Fallback: load from disk
            scene_file = os.path.join(self.data_dir, "scenes", f"scene_{scene_id:04d}.npz")
            point_cloud = np.load(scene_file)["point_cloud"].copy()
        
        # Resample to fixed number of points
        point_cloud = self._resample(point_cloud)
        
        # Normalize point cloud
        if self.normalize_pc:
            point_cloud = self._normalize_point_cloud(point_cloud)
        
        # Data augmentation
        if self.augment:
            point_cloud = self._augment_point_cloud(point_cloud)
        
        return {
            "point_cloud": torch.from_numpy(point_cloud).float(),  # (N, 3)
            "config": torch.from_numpy(config).float(),            # (7,)
            "label": torch.tensor(label, dtype=torch.long),        # scalar
            "scene_id": scene_id,
        }
    
    def _resample(self, points: np.ndarray) -> np.ndarray:
        """Resample point cloud to fixed size."""
        n = len(points)
        if n == 0:
            return np.zeros((self.n_points, 3), dtype=np.float32)
        
        if n >= self.n_points:
            indices = np.random.choice(n, self.n_points, replace=False)
        else:
            indices = np.random.choice(n, self.n_points, replace=True)
        
        return points[indices]
    
    def _normalize_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """Center and scale point cloud to unit sphere."""
        centroid = points.mean(axis=0)
        points = points - centroid
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 0:
            points = points / max_dist
        return points
    
    def _augment_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """Apply random augmentations to point cloud."""
        # Random rotation around z-axis
        if np.random.rand() > 0.5:
            theta = np.random.uniform(0, 2 * np.pi)
            rot = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta),  np.cos(theta), 0],
                [0, 0, 1]
            ])
            points = points @ rot.T
        
        # Random jitter
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.01, points.shape)
            points = points + noise
        
        # Random scaling
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            points = points * scale
        
        return points.astype(np.float32)


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    n_points: int = 2048,
    num_workers: int = 4,
    augment_train: bool = True,
    normalize_pc: bool = True,
):
    """Create train, val, test dataloaders.
    
    Args:
        data_dir: Path to scenes directory
        batch_size: Batch size for all loaders
        n_points: Points per point cloud
        num_workers: DataLoader workers
        augment_train: Whether to augment training data
        normalize_pc: Whether to normalize point clouds
        
    Returns:
        train_loader, val_loader, test_loader
    """
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


def collate_fn(batch):
    """Custom collate function (optional, for variable-size handling)."""
    return {
        "point_cloud": torch.stack([b["point_cloud"] for b in batch]),
        "config": torch.stack([b["config"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "scene_id": [b["scene_id"] for b in batch],
    }


if __name__ == "__main__":
    # Quick test
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../data/scenes")
    args = parser.parse_args()
    
    # Resolve path relative to script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(script_dir, args.data_dir))
    
    print(f"Testing dataset from: {data_dir}")
    
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=32,
        num_workers=0,  # 0 for debugging
    )
    
    # Check a batch
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  point_cloud: {batch['point_cloud'].shape}")  # (B, N, 3)
    print(f"  config:      {batch['config'].shape}")       # (B, 7)
    print(f"  label:       {batch['label'].shape}")        # (B,)
    print(f"  labels:      {batch['label'].tolist()[:10]}...")