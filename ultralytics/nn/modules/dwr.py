import torch
import torch.nn as nn


class DWR(nn.Module):
    def __init__(self, c1, c2, dilation_rates=[1, 3, 5], reduction=16):  # 保持默认参数
        super().__init__()
        if isinstance(dilation_rates, int):  # 防御性编程
            dilation_rates = [dilation_rates]
        self.convs = nn.ModuleList()
        for rate in dilation_rates:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(c1, c2, 3, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(c2),
                    nn.SiLU()
                )
            )

        # 通道注意力
        reduction = 16
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, c2 // reduction, 1),
            nn.SiLU(),
            nn.Conv2d(c2 // reduction, c2, 1),
            nn.Sigmoid()
        )

        # 特征融合
        self.fusion = nn.Conv2d(c2 * len(dilation_rates), c2, 1)
        self.shortcut = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()

    def forward(self, x):
        features = [conv(x) for conv in self.convs]
        fused = self.fusion(torch.cat(features, dim=1))
        attention = self.channel_attention(fused)
        return self.shortcut(x) + fused * attention