"""
nn_collision_detector.py — Cartesian-Aware Inference Wrapper for PointNet CollisionNet.

This version performs real-time Forward Kinematics (FK) to transform 7-DOF joint
configurations into 9 Cartesian keypoints before performing inference.
"""

import os
import sys
import numpy as np
import torch
import pybullet as p
from typing import List, Union
from pathlib import Path

# Ensure model imports work correctly
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from model.collisionnet import CollisionNet
from planner.collision_detector import JOINT_LIMITS

# ──────────────────────────────────────────────
# Constants & Joint Limits
# ──────────────────────────────────────────────

JOINT_LOWER = np.array([lo for lo, _ in JOINT_LIMITS], dtype=np.float32)
JOINT_UPPER = np.array([hi for _, hi in JOINT_LIMITS], dtype=np.float32)
JOINT_RANGE = JOINT_UPPER - JOINT_LOWER

# ──────────────────────────────────────────────
# Neural Collision Checker Class
# ──────────────────────────────────────────────

class NeuralCollisionChecker:
    """
    PointNet-based collision checker using Cartesian keypoints and PyBullet FK.
    """
    def __init__(
        self,
        scene_point_cloud: np.ndarray,
        panda_id: int,
        model_path: str,
        threshold: float = 0.5,
        device: str = None
    ):
        """
        Args:
            scene_point_cloud: Raw (N, 3) point cloud from the environment.
            panda_id: The PyBullet body ID for the Franka Panda.
            model_path: Path to the .pth or .pt checkpoint.
            threshold: Probability threshold for collision (default 0.5).
            device: 'cpu' or 'cuda'. Auto-selected if None.
        """
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.threshold = threshold
        self.panda_id = panda_id
        
        # 1. Load Model with Flexible Metadata Handling
        ckpt = torch.load(model_path, map_location="cpu")
        
        # Expert Model A defaults (1024 dim, 8 heads)
        model_cfg = ckpt.get("model_cfg", {"embed_dim": 1024, "num_heads": 8})
        self.model = CollisionNet(**model_cfg).to(self.device)
        
        # Handle different state_dict keys (model_state vs model_state_dict)
        state_key = "model_state" if "model_state" in ckpt else "model_state_dict"
        self.model.load_state_dict(ckpt[state_key])
        self.model.eval()
        
        print(f"[NeuralCollisionChecker] Loaded model from {model_path}")
        print(f"  Device: {self.device} | Threshold: {self.threshold}")

        # 2. Process and Cache Scene Metadata
        self.update_scene(scene_point_cloud)

    def _prepare_point_cloud(self, raw_pc: np.ndarray) -> torch.Tensor:
        """Centres and scales point cloud to unit sphere per training protocol."""
        pts = raw_pc - self.centroid
        self.max_dist = np.max(np.linalg.norm(pts, axis=1))
        if self.max_dist > 0:
            pts = pts / self.max_dist
        
        # Subsample to exactly 2048 points to match PointNet input
        if len(pts) > 2048:
            idx = np.random.choice(len(pts), 2048, replace=False)
            pts = pts[idx]
        elif len(pts) < 2048:
            idx = np.random.choice(len(pts), 2048, replace=True)
            pts = pts[idx]
            
        return torch.from_numpy(pts).float().unsqueeze(0).to(self.device)

    def _get_cartesian_keypoints(self, config: np.ndarray) -> np.ndarray:
        """Calculates 9 Cartesian link positions via PyBullet Forward Kinematics."""
        # Update internal PyBullet state to compute FK
        for i in range(7):
            p.resetJointState(self.panda_id, i, config[i])
        
        keypoints = []
        # Extract world positions for links 0-8 (Base through Hand)
        for i in range(9):
            state = p.getLinkState(self.panda_id, i)
            keypoints.append(state[0]) # worldLinkFramePosition
            
        return np.array(keypoints) # Resulting shape: (9, 3)

    @torch.no_grad()
    def in_collision(self, config: Union[List[float], np.ndarray]) -> bool:
        """Checks a single 7-DOF configuration for collision."""
        config_arr = np.asarray(config, dtype=np.float32)
        
        # 1. Transform: 7 Joint Angles -> 9 Cartesian (X,Y,Z) Keypoints
        robot_pts = self._get_cartesian_keypoints(config_arr)
        
        # 2. Normalize using cached scene metadata (Centroid & Max Distance)
        robot_pts = (robot_pts - self.centroid) / self.max_dist
        robot_t = torch.from_numpy(robot_pts).float().unsqueeze(0).to(self.device)
        
        # 3. Inference Pass (Order: Robot Keypoints, Scene Point Cloud)
        logits, _, _ = self.model(robot_t, self._pc_tensor)
        prob = torch.sigmoid(logits).item()
        
        return prob > self.threshold

    @torch.no_grad()
    def batch_in_collision(self, configs: np.ndarray) -> np.ndarray:
        """
        Batch check configurations. 
        Note: Currently sequential due to PyBullet FK requirements.
        """
        return np.array([self.in_collision(q) for q in configs])

    def update_scene(self, scene_point_cloud: np.ndarray):
        """Updates the internal scene representation when the robot enters a new environment."""
        self.centroid = scene_point_cloud.mean(axis=0)
        self._pc_tensor = self._prepare_point_cloud(scene_point_cloud)
        print(f"[NeuralCollisionChecker] Scene updated. Point cloud centroid: {self.centroid}")

# ──────────────────────────────────────────────
# Standalone Smoke Test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import pybullet_data
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    panda_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
    
    fake_pc = np.random.uniform(-1, 1, (2048, 3))
    
    # Replace with your actual model path for a real test
    print("Initializing checker (Headless)...")
    try:
        checker = NeuralCollisionChecker(
            scene_point_cloud=fake_pc,
            panda_id=panda_id,
            model_path="model_a_expert_95pct.pth"
        )
        test_config = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        print(f"Collision result: {checker.in_collision(test_config)}")
    except Exception as e:
        print(f"Checker initialization failed (expected if model file is missing): {e}")
    finally:
        p.disconnect()