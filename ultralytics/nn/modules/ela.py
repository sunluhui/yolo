import torch
import torch.nn as nn
import torch.nn.functional as F


class ELA(nn.Module):
    """
    Efficient Local Attention (ELA) 模块
    论文: https://arxiv.org/abs/2108.02457
    """

    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(ELA, self).__init__()
        self.kernel_size = kernel_size
        self.reduction = reduction

        # 通道压缩
        self.channel_compress = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True)
        )

        # 空间注意力生成
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        # 通道扩展
        self.channel_expand = nn.Sequential(
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x

        # 通道压缩
        x_compressed = self.channel_compress(x)

        # 生成空间注意力图
        avg_out = torch.mean(x_compressed, dim=1, keepdim=True)
        max_out, _ = torch.max(x_compressed, dim=1, keepdim=True)
        spatial_att = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))

        # 应用空间注意力
        x_attended = x_compressed * spatial_att

        # 通道扩展
        x_expanded = self.channel_expand(x_attended)

        # 残差连接
        return identity * self.sigmoid(x_expanded)