import torch
import torch.nn as nn
import torch.nn.functional as F


class VSSBlock_YOLO(nn.Module):
    """简化的 Mamba 块，使用纯 PyTorch 实现"""

    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * expand),
            nn.GELU(),
            nn.Linear(dim * expand, dim)
        )

        # 使用卷积替代 Mamba 的选择性扫描
        self.conv1d = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=d_conv,
            padding=d_conv // 2,
            groups=dim  # 深度可分离卷积
        )

        self.norm = nn.LayerNorm(dim)
        self.drop_path = nn.Dropout(0.1) if 0.1 > 0. else nn.Identity()

    def forward(self, x):
        residual = x

        # 处理 4D 输入 [B, C, H, W]
        if len(x.shape) == 4:
            B, C, H, W = x.shape
            x = x.view(B, C, -1).transpose(1, 2)  # [B, L, C]

        x = self.norm(x)

        # 应用 1D 卷积
        x_conv = x.transpose(1, 2)  # [B, C, L]
        x_conv = self.conv1d(x_conv)
        x_conv = x_conv.transpose(1, 2)  # [B, L, C]

        # MLP
        x_mlp = self.mlp(x)

        # 合并
        x = x_conv + x_mlp

        # 恢复形状
        if len(residual.shape) == 4:
            x = x.transpose(1, 2).view(B, C, H, W)

        return residual + self.drop_path(x)


class XSSBlock(nn.Module):
    """简化的 XSS 块"""

    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim)
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class SimpleStem(nn.Module):
    def __init__(self, in_ch=3, out_ch=128, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=2, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class VisionClueMerge(nn.Module):
    def __init__(self, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# 移除所有 SelectiveScan 相关的复杂代码
# 只保留简单的模块实现