import torch
import torch.nn as nn
import torch.nn.functional as F


class VSSBlock_YOLO(nn.Module):
    def __init__(self, c1, c2):  # c1: 输入通道, c2: 输出通道
        super().__init__()
        self.dim = c2

        # 通道适配器 - 如果输入输出通道不同，需要调整
        if c1 != c2:
            self.channel_adapter = nn.Conv2d(c1, c2, 1)
        else:
            self.channel_adapter = nn.Identity()

        self.mlp = nn.Sequential(
            nn.Linear(c2, c2 * 2),
            nn.GELU(),
            nn.Linear(c2 * 2, c2)
        )

        self.conv1d = nn.Conv1d(
            in_channels=c2,
            out_channels=c2,
            kernel_size=3,
            padding=1,
            groups=c2
        )

        self.norm = nn.LayerNorm(c2)
        self.drop_path = nn.Dropout(0.1) if 0.1 > 0. else nn.Identity()

    def forward(self, x):
        # 先进行通道适配
        x = self.channel_adapter(x)
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
    def __init__(self, c1, c2):  # c1: 输入通道, c2: 输出通道
        super().__init__()

        # 通道适配器
        if c1 != c2:
            self.channel_adapter = nn.Conv2d(c1, c2, 1)
        else:
            self.channel_adapter = nn.Identity()

        self.block = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
            nn.Conv2d(c2, c2, 3, padding=1),
            nn.BatchNorm2d(c2)
        )

    def forward(self, x):
        # 先进行通道适配
        x = self.channel_adapter(x)
        return x + self.block(x)


class SimpleStem(nn.Module):
    def __init__(self, c1, c2):  # c1: 输入通道, c2: 输出通道
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class VisionClueMerge(nn.Module):
    def __init__(self, c1, c2):  # c1: 输入通道, c2: 输出通道
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))