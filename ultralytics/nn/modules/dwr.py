import torch
import torch.nn as nn


class DWR(nn.Module):
    """Dense With Residual模块"""

    def __init__(self, in_channels, growth_rate=32, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        current_channels = in_channels

        # 构建密集连接层
        for _ in range(num_layers):
            layer = nn.Sequential(
                nn.Conv2d(current_channels, growth_rate, 3, padding=1, bias=False),
                nn.BatchNorm2d(growth_rate),
                nn.SiLU(inplace=True)
            )
            self.layers.append(layer)
            current_channels += growth_rate

        # 残差路径的1x1卷积
        self.res_conv = nn.Sequential(
            nn.Conv2d(in_channels, current_channels, 1, bias=False),
            nn.BatchNorm2d(current_channels)
        )

        # 最终融合卷积
        self.fusion = nn.Sequential(
            nn.Conv2d(current_channels * 2, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )

    def forward(self, x):
        identity = x
        features = [x]

        # 密集连接前向传播
        for layer in self.layers:
            new_feat = layer(torch.cat(features, dim=1))
            features.append(new_feat)

        # 特征拼接
        dense_out = torch.cat(features[1:], dim=1)

        # 残差路径
        res_out = self.res_conv(identity)

        # 融合并返回
        fused = self.fusion(torch.cat([dense_out, res_out], dim=1))
        return fused + identity  # 残差连接