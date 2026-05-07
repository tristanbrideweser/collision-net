# src/model/collisionnet.py
import torch
import torch.nn as nn

# Import our three Lego blocks
from src.model.encoders.kinematic_mlp import CartesianConfigEncoder
from src.model.encoders.pointnet import PointNetEncoder
from src.model.fusion.cross_attn import SpatialCrossAttention

class CollisionNet(nn.Module):
    """
    The master model. Fuses kinematic keypoints and scene point clouds 
    to predict collision probabilities.
    """
    def __init__(self, embed_dim=1024, num_heads=8):
        super().__init__()
        
        # 1. The Encoders
        self.robot_encoder = CartesianConfigEncoder(out_dim=embed_dim)
        self.scene_encoder = PointNetEncoder(out_dim=embed_dim)
        
        # 2. The Fusion Bridge
        self.cross_attn = SpatialCrossAttention(embed_dim=embed_dim, num_heads=num_heads)
        
        # 3. The Collision Classifier (applied per-keypoint)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1) # Outputs a single logit per keypoint
        )

    def forward(self, robot_keypoints, scene_points):
        """
        Args:
            robot_keypoints: (B, K, 3) 
            scene_points:    (B, N, 3)
        Returns:
            logits:       (B, 1) raw collision score (pass through Sigmoid later)
            attn_weights: (B, H, K, N) for the heatmap visualization
            trans_feat:   (B, 64, 64) for PointNet regularization loss
        """
        B, K, _ = robot_keypoints.shape
        
        # 1. Encode
        robot_feat = self.robot_encoder(robot_keypoints)            # (B, K, D)
        scene_feat, trans_feat = self.scene_encoder(scene_points)   # (B, N, D)
        
        # 2. Fuse
        fused_feat, attn_weights = self.cross_attn(
            robot_queries=robot_feat, 
            scene_keys=scene_feat, 
            scene_values=scene_feat
        ) # fused_feat is (B, K, D)
        
        # 3. Classify each keypoint independently
        # Reshape to (B * K, D) for the Linear layers
        flat_feat = fused_feat.view(-1, fused_feat.size(-1))
        kp_logits = self.classifier(flat_feat)
        
        # Reshape back to (B, K, 1)
        kp_logits = kp_logits.view(B, K, 1)
        
        # 4. Global Max Pool over the Keypoint dimension
        # If any keypoint is in collision, the max logit will be high.
        global_logits = kp_logits.max(dim=1)[0] # (B, 1)
        
        return global_logits, attn_weights, trans_feat