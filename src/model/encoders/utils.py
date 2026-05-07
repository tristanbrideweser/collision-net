# src/model/encoders/utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedMLP(nn.Module):
    """1-D convolution acting as a shared MLP across all points.
    Equivalent to applying the same MLP independently to every point.
    """
    def __init__(self, in_channels: int, out_channels: int, bn: bool = True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=not bn)
        self.bn   = nn.BatchNorm1d(out_channels) if bn else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class TNet(nn.Module):
    """Input / feature transform: predict a k×k alignment matrix.

    Predicts a residual on top of the identity so that training is stable.
    """
    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k
        self.shared = nn.Sequential(
            SharedMLP(k, 64),
            SharedMLP(64, 128),
            SharedMLP(128, 1024),
        )
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512,  256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256,  k * k),
        )
        # Initialise output to zero so the transform starts as identity
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, k, N)
        feat = self.shared(x)                           # (B, 1024, N)
        feat = feat.max(dim=2)[0]                       # (B, 1024)
        mat  = self.mlp(feat).view(-1, self.k, self.k) # (B, k, k)
        eye  = torch.eye(self.k, device=x.device).unsqueeze(0)
        return mat + eye                                # residual on identity