"""
model.py — PointNet-based collision detector for Franka Panda.

Architecture:
    1. PointNet encoder: shared MLP on each point → max-pool → global feature (1024-d)
    2. Config encoder: MLP on 7-DOF joint config (normalised to [-1,1])
    3. Fusion MLP: concatenate both features → binary collision logit

Location: src/model/model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────

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


# ──────────────────────────────────────────────
# PointNet encoder
# ──────────────────────────────────────────────

class PointNetEncoder(nn.Module):
    """Encode an (N,3) point cloud to a fixed-size global feature vector.

    Args:
        out_dim:        Dimensionality of the global feature (default 1024).
        use_input_tnet: Whether to apply the 3×3 input transform (default True).
        use_feat_tnet:  Whether to apply the 64×64 feature transform (default True).
    """
    def __init__(
        self,
        out_dim: int = 1024,
        use_input_tnet: bool = True,
        use_feat_tnet:  bool = True,
    ):
        super().__init__()
        self.out_dim        = out_dim
        self.use_input_tnet = use_input_tnet
        self.use_feat_tnet  = use_feat_tnet

        if use_input_tnet:
            self.input_tnet = TNet(k=3)
        if use_feat_tnet:
            self.feat_tnet  = TNet(k=64)

        # Shared MLPs: lift 3-D points to 1024-D descriptors
        self.mlp1 = nn.Sequential(SharedMLP(3,    64),
                                   SharedMLP(64,   64))
        self.mlp2 = nn.Sequential(SharedMLP(64,   128),
                                   SharedMLP(128,  out_dim))

    def forward(self, xyz: torch.Tensor):
        """
        Args:
            xyz: (B, N, 3)  point cloud
        Returns:
            global_feat: (B, out_dim)
            trans_feat:  (B, 64, 64) or None — needed for regularisation loss
        """
        B, N, _ = xyz.shape
        x = xyz.transpose(1, 2)             # (B, 3, N)

        # Input transform
        if self.use_input_tnet:
            t_in = self.input_tnet(x)       # (B, 3, 3)
            x    = torch.bmm(t_in, x)       # align points

        # First MLP block → local features
        x = self.mlp1(x)                    # (B, 64, N)

        # Feature transform
        trans_feat = None
        if self.use_feat_tnet:
            t_feat     = self.feat_tnet(x)  # (B, 64, 64)
            x          = torch.bmm(t_feat, x)
            trans_feat = t_feat

        # Second MLP block → high-dim per-point features
        x = self.mlp2(x)                    # (B, out_dim, N)

        # Symmetric aggregation: global max-pool
        global_feat = x.max(dim=2)[0]       # (B, out_dim)

        return global_feat, trans_feat


# ──────────────────────────────────────────────
# Full collision detection model
# ──────────────────────────────────────────────

class PointNetCollisionDetector(nn.Module):
    """PointNet collision detector for a 7-DOF manipulator.

    Inputs
    ------
    point_cloud : (B, N, 3)   obstacle surface points (normalised to unit sphere)
    config      : (B, 7)      joint angles normalised to [-1, 1]

    Output
    ------
    logit       : (B,)        raw (un-sigmoided) collision score
                              positive  → collision,  negative → free
    """

    def __init__(
        self,
        pc_feat_dim:   int   = 1024,
        cfg_hidden_dim: int  = 256,
        fuse_hidden_dim: int = 512,
        dropout:        float = 0.3,
        use_input_tnet: bool  = True,
        use_feat_tnet:  bool  = True,
    ):
        super().__init__()
        self.encoder = PointNetEncoder(
            out_dim=pc_feat_dim,
            use_input_tnet=use_input_tnet,
            use_feat_tnet=use_feat_tnet,
        )

        # Config encoder: 7 → cfg_hidden_dim
        self.config_encoder = nn.Sequential(
            nn.Linear(7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, cfg_hidden_dim),
            nn.BatchNorm1d(cfg_hidden_dim),
            nn.ReLU(),
        )

        fuse_in = pc_feat_dim + cfg_hidden_dim

        # Fusion classifier: concatenated features → logit
        self.classifier = nn.Sequential(
            nn.Linear(fuse_in,         fuse_hidden_dim), nn.BatchNorm1d(fuse_hidden_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fuse_hidden_dim, fuse_hidden_dim // 2), nn.BatchNorm1d(fuse_hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fuse_hidden_dim // 2, 1),
        )

    def forward(self, point_cloud: torch.Tensor, config: torch.Tensor):
        pc_feat,   trans_feat = self.encoder(point_cloud)    # (B, pc_feat_dim)
        cfg_feat              = self.config_encoder(config)  # (B, cfg_hidden_dim)
        fused                 = torch.cat([pc_feat, cfg_feat], dim=1)
        logit                 = self.classifier(fused).squeeze(1)  # (B,)
        return logit, trans_feat


# ──────────────────────────────────────────────
# Regularisation loss (PointNet paper)
# ──────────────────────────────────────────────

def feature_transform_regulariser(trans: torch.Tensor) -> torch.Tensor:
    """Penalise deviation of the 64×64 feature transform from orthogonality.

    Loss = ||I - A·Aᵀ||²_F  (mean over batch)
    """
    if trans is None:
        return torch.tensor(0.0)
    B, k, _ = trans.shape
    I   = torch.eye(k, device=trans.device).unsqueeze(0).expand(B, -1, -1)
    AAt = torch.bmm(trans, trans.transpose(1, 2))
    return F.mse_loss(AAt, I)


# ──────────────────────────────────────────────
# Convenience: model info
# ──────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    B, N = 4, 2048
    pc  = torch.randn(B, N, 3)
    cfg = torch.randn(B, 7)

    model = PointNetCollisionDetector()
    logit, tf = model(pc, cfg)

    print(f"Output shape : {logit.shape}")        # (4,)
    print(f"Params       : {count_parameters(model):,}")
    reg = feature_transform_regulariser(tf)
    print(f"Reg loss     : {reg.item():.4f}")
