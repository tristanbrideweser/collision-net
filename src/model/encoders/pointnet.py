# src/model/encoders/pointnet.py
import torch
import torch.nn as nn

# Since utils.py is in the same directory, this relative import works perfectly
from .utils import SharedMLP, TNet 
    
class PointNetEncoder(nn.Module):
    """Encode an (N,3) point cloud to dense (N, out_dim) features for attention.

    Args:
        out_dim:        Dimensionality of the per-point feature (default 1024).
        use_input_tnet: Whether to apply the 3x3 input transform (default True).
        use_feat_tnet:  Whether to apply the 64x64 feature transform (default True).
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
            xyz: (B, N, 3) point cloud
        Returns:
            dense_feat: (B, N, out_dim) <-- CRITICAL: No max pooling!
            trans_feat: (B, 64, 64) or None for regularization loss
        """
        B, N, _ = xyz.shape
        x = xyz.transpose(1, 2)             # (B, 3, N)

        # Input transform
        if self.use_input_tnet:
            t_in = self.input_tnet(x)       # (B, 3, 3)
            x    = torch.bmm(t_in, x)       # align points

        # First MLP block -> local features
        x = self.mlp1(x)                    # (B, 64, N)

        # Feature transform
        trans_feat = None
        if self.use_feat_tnet:
            t_feat     = self.feat_tnet(x)  # (B, 64, 64)
            x          = torch.bmm(t_feat, x)
            trans_feat = t_feat

        # Second MLP block -> high-dim per-point features
        x = self.mlp2(x)                    # (B, out_dim, N)

        # CRITICAL CHANGE: Skip the global max-pool!
        # Transpose back to sequence format for the attention module
        dense_feat = x.transpose(1, 2)      # (B, N, out_dim)

        return dense_feat, trans_feat