import torch
import torch.nn as nn


class DWR(nn.Module):
    def __init__(self, c1, c2, dilation_rates=[1, 3, 5], reduction=16, lightweight=False):
        super().__init__()
        if isinstance(dilation_rates, int):
            dilation_rates = [dilation_rates]
        if lightweight:
            dilation_rates = dilation_rates[:2]  # 轻量化模式下减少 dilation_rates

        # 确保 reduction 是一个整数
        if isinstance(reduction, list):
            reduction = reduction[0]

        self.convs = nn.ModuleList()
        for rate in dilation_rates:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(c1, c2, 3, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(c2),
                    nn.SiLU()
                )
            )

        # 特征融合前的注意力
        self.scale_attention = nn.Sequential(
            nn.Conv2d(len(dilation_rates), len(dilation_rates), 1),
            nn.Softmax(dim=1)
        )

        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_attention = nn.Sequential(
            nn.Conv2d(c2 * 2, c2 // reduction, 1),
            nn.SiLU(),
            nn.Conv2d(c2 // reduction, c2, 1),
            nn.Sigmoid()
        )

        # 特征融合
        self.fusion = nn.Conv2d(c2 * len(dilation_rates), c2, 1)
        self.shortcut = nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2)
        ) if c1 != c2 else nn.Identity()

    def forward(self, x):
        features = [conv(x) for conv in self.convs]

        # 为每个尺度的特征分配注意力权重
        scale_weights = self.scale_attention(torch.cat([f.mean(dim=1, keepdim=True) for f in features], dim=1))
        features = [f * scale_weights[:, i:i + 1, :, :] for i, f in enumerate(features)]

        fused = self.fusion(torch.cat(features, dim=1))

        # 结合全局平均池化和全局最大池化
        avg_out = self.avg_pool(fused)
        max_out = self.max_pool(fused)
        attention = self.channel_attention(torch.cat([avg_out, max_out], dim=1))

        return self.shortcut(x) + fused * attention