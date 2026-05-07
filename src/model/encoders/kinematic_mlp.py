# src/model/encoders/kinematic_mlp.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.encoders.utils import SharedMLP

class CartesianConfigEncoder(nn.Module):
    """
    """
    def __init__(self, out_dim=1024):
        super().__init__()

        self.mlp = nn.Sequential(
            SharedMLP(3, 64),
            SharedMLP(64, 128),
            SharedMLP(128, out_dim),
        )

    def forward(self, keypoints):
        # permute for 1D conv
        x = keypoints.transpose(1, 2)  # (B, 3, K)
        
        # apply shared MLPs
        x = self.mlp(x) # (B, out_dim, K)

        # transpose back to standard seq format
        x = x.transpose(1, 2)       # (B, K, out_dim)

        return x