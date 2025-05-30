import torch
from torch import nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv


class BiFPN_Block(nn.Module):
    """Simplified BiFPN Block"""

    def __init__(self, c2, num_inputs=2, eps=1e-4):
        super().__init__()
        self.eps = eps
        self.num_inputs = num_inputs
        self.out_channels = c2
        self.w = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32), requires_grad=True)
        self.conv = Conv(c2, c2, k=3, s=1, p=1, g=1, act=True)

    def forward(self, x):
        # 确保输入数量正确
        if len(x) != self.num_inputs:
            # 自动调整输入数量
            x = x[:self.num_inputs]

        # 加权融合
        weights = F.relu(self.w)
        weights = weights / (weights.sum() + self.eps)
        fused = sum(weights[i] * x[i] for i in range(len(x)))

        # 特征转换
        return self.conv(fused)