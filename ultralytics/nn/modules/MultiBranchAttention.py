import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiBranchAttention(nn.Module):
    """多分支注意力机制模块"""

    def __init__(self, in_channels, reduction_ratio=16, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.branches = nn.ModuleList()

        # 分支1: 通道注意力
        self.branches.append(ChannelAttention(in_channels, reduction_ratio))

        # 分支2: 空间注意力
        self.branches.append(SpatialAttention())

        # 分支3: 局部上下文注意力 (不同感受野)
        #for k in kernel_sizes:
            #self.branches.append(LocalContextAttention(in_channels, k))

        # 分支4: 全局上下文注意力
        #self.branches.append(GlobalContextAttention(in_channels))

        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * len(self.branches), in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )

        # 门控机制
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, len(self.branches), 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]

        # 门控加权
        gate_weights = self.gate(x)  # [B, num_branches, 1, 1]
        weighted_outputs = []
        for i, out in enumerate(branch_outputs):
            weight = gate_weights[:, i:i + 1]  # [B, 1, 1, 1]
            weighted_outputs.append(out * weight)

        # 特征融合
        fused = torch.cat(weighted_outputs, dim=1)
        return self.fusion(fused) + x  # 残差连接


class ChannelAttention(nn.Module):
    """通道注意力分支"""

    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return x * out.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """空间注意力分支"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        att = self.conv(combined)
        return x * self.sigmoid(att)


class LocalContextAttention(nn.Module):
    """局部上下文注意力分支"""

    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size,
                              padding=padding, groups=in_channels, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class GlobalContextAttention(nn.Module):
    """全局上下文注意力分支"""

    def __init__(self, in_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        att = self.fc(self.pool(x).view(b, c))
        return x * att.view(b, c, 1, 1)