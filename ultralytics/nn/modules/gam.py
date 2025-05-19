import torch
import torch.nn as nn


class GAMAttention(nn.Module):
    """Global Attention Mechanism (GAM) module"""

    def __init__(self, in_channels, reduction=4, kernel_size=7):
        super(GAMAttention, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        b, c, h, w = x.size()
        channel_avg = x.view(b, c, -1).mean(-1)  # [b,c]
        channel_max = x.view(b, c, -1).max(-1)[0]  # [b,c]

        channel_att = self.channel_attention(channel_avg + channel_max)  # [b,c]
        channel_att = torch.sigmoid(channel_att).view(b, c, 1, 1)  # [b,c,1,1]
        x_channel = x * channel_att  # [b,c,h,w]

        # Spatial attention
        spatial_avg = torch.mean(x_channel, dim=1, keepdim=True)  # [b,1,h,w]
        spatial_max, _ = torch.max(x_channel, dim=1, keepdim=True)  # [b,1,h,w]
        spatial_cat = torch.cat([spatial_avg, spatial_max], dim=1)  # [b,2,h,w]

        spatial_att = self.spatial_attention(spatial_cat)  # [b,1,h,w]
        x_spatial = x_channel * spatial_att  # [b,c,h,w]

        return x_spatial
