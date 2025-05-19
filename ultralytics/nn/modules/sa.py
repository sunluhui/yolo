import torch
import torch.nn as nn


class LightSABlock(nn.Module):
    """Lightweight Self-Attention for shallow features"""

    def __init__(self, in_channels, reduction_ratio=8):
        super(LightSABlock, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)

        # Spatial attention (simplified SA)
        sa = self.spatial_attention(x)

        # Combine attentions
        return x * ca * sa
