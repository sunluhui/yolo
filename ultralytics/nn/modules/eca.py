import torch
import torch.nn as nn
import math


class ECABlock(nn.Module):
    """Efficient Channel Attention module
    Args:
        channel: Number of input channels
        k_size: Optional, kernel size for 1D convolution. If None, will be adaptively determined
    """

    def __init__(self, channel, k_size=None):
        super(ECABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 自适应确定卷积核大小
        if k_size is None:
            t = int(abs((math.log(channel, 2) + 1) / 2))
            k_size = t if t % 2 else t + 1

        # 确保k_size是正奇数
        k_size = max(3, k_size)  # 最小为3
        k_size = k_size if k_size % 2 else k_size + 1  # 确保为奇数

        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.channel = channel  # 保存通道数用于验证

    def forward(self, x):
        b, c, h, w = x.size()

        # 验证输入通道数
        if c != self.channel:
            raise ValueError(f"Input channels {c} don't match initialized channels {self.channel}")

        # 1. Global average pooling [b,c,h,w] -> [b,c,1,1]
        y = self.avg_pool(x)

        # 2. 1D convolution [b,c,1,1] -> [b,c,1,1]
        # [b,c,1,1] -> [b,1,c] -> conv -> [b,1,c] -> [b,c,1,1]
        y = y.view(b, 1, c)  # [b,1,c]
        y = self.conv(y)  # [b,1,c]
        y = y.view(b, c, 1, 1)  # [b,c,1,1]

        # 3. Sigmoid activation and scaling
        y = self.sigmoid(y)

        return x * y.expand_as(x)