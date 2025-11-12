import torch
import torch.nn as nn
import torch.nn.functional as F


class VSSBlock_YOLO(nn.Module):
    """修正参数数量"""

    def __init__(self, c1, c2):  # 只接受两个参数
        super().__init__()
        dim = c2  # 使用第二个参数作为维度
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

        self.conv1d = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            padding=1,
            groups=dim
        )

        self.norm = nn.LayerNorm(dim)
        self.drop_path = nn.Dropout(0.1) if 0.1 > 0. else nn.Identity()

    def forward(self, x):
        residual = x

        if len(x.shape) == 4:
            B, C, H, W = x.shape
            x = x.view(B, C, -1).transpose(1, 2)

        x = self.norm(x)
        x_conv = x.transpose(1, 2)
        x_conv = self.conv1d(x_conv)
        x_conv = x_conv.transpose(1, 2)
        x_mlp = self.mlp(x)
        x = x_conv + x_mlp

        if len(residual.shape) == 4:
            x = x.transpose(1, 2).view(B, C, H, W)

        return residual + self.drop_path(x)


class XSSBlock(nn.Module):
    """修正参数数量"""

    def __init__(self, c1, c2):  # 只接受两个参数
        super().__init__()
        dim = c2
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


class SimpleStem(nn.Module):
    """修正参数数量"""

    def __init__(self, c1, c2):  # 只接受两个参数
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class VisionClueMerge(nn.Module):
    """修正参数数量"""

    def __init__(self, c1, c2):  # 只接受两个参数
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))