import torch
import torch.nn as nn
import math


class ECAAttention(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class DWR(nn.Module):
    def __init__(self, c1, c2, dilation_rates=[1, 2, 3], reduction=16):
        super().__init__()
        if isinstance(dilation_rates, int):
            dilation_rates = [dilation_rates]

        self.convs = nn.ModuleList()
        for rate in dilation_rates:
            self.convs.append(
                nn.Sequential(
                    # 深度可分离卷积
                    nn.Conv2d(c1, c1, 3, padding=rate, dilation=rate, groups=c1, bias=False),
                    nn.Conv2d(c1, c2, 1, bias=False),
                    nn.BatchNorm2d(c2),
                    nn.SiLU()
                )
            )

        # 改进的特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(c2 * len(dilation_rates), c2 * 2, 1),
            nn.GELU(),
            nn.Conv2d(c2 * 2, c2, 1)
        )

        # 双重注意力机制
        self.channel_attention = ECAAttention(c2)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(c2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

        self.shortcut = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()

    def forward(self, x):
        features = [conv(x) for conv in self.convs]
        fused = self.fusion(torch.cat(features, dim=1))

        # 注意力机制应用
        channel_att = self.channel_attention(fused)
        spatial_att = self.spatial_attention(fused)

        return self.shortcut(x) + fused * channel_att * spatial_att