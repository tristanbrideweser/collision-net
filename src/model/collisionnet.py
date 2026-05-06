# src/model/collisionnet.py
import torch 
import torch.nn as nn
import torch.nn.functional as F

from encoders import pointnet as PointNetEncoder

class CollisionNet(nn.Module):
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

    model = CollisionNet()
    logit, tf = model(pc, cfg)

    print(f"Output shape : {logit.shape}")        # (4,)
    print(f"Params       : {count_parameters(model):,}")
    reg = feature_transform_regulariser(tf)
    print(f"Reg loss     : {reg.item():.4f}")
