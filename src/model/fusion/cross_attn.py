# src/model/fusion/attn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SpatialCrossAttention(nn.Module):
    """
    Computes Cross-Attention between Robot Keypoints (Queries) 
    and Scene Points (Keys/Values) without causal masking.
    """
    def __init__(self, embed_dim=1024, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, robot_queries, scene_keys, scene_values):
        """
        Args:
            robot_queries: (B, K, D) features from the Kinematic MLP
            scene_keys:    (B, N, D) features from the PointNet
            scene_values:  (B, N, D) features from the PointNet
            
        Returns:
            out: (B, K, D) context-aware robot features
            attn_weights: (B, H, K, N) the raw heatmaps for visualization
        """
        B, K_pts, D = robot_queries.size()
        _, N_pts, _ = scene_keys.size()

        # 1. Project and split into attention heads
        # Shape becomes: (B, Num_Heads, Seq_Len, Head_Dim)
        Q = self.q_proj(robot_queries).view(B, K_pts, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(scene_keys).view(B, N_pts, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(scene_values).view(B, N_pts, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Compute spatial attention scores (Dot product of Q and K)
        # Shape: (B, Num_Heads, K_pts, N_pts)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # CRITICAL: No causal masking! Every robot point sees every scene point.
        attn_weights = F.softmax(scores, dim=-1)

        # 3. Apply attention weights to the Values
        out = torch.matmul(attn_weights, V) # (B, Num_Heads, K_pts, Head_Dim)
        
        # 4. Concatenate heads back together and project
        out = out.transpose(1, 2).contiguous().view(B, K_pts, D)
        out = self.out_proj(out)

        return out, attn_weights