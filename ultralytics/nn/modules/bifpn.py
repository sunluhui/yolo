import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv


class BiFPN_Concat2(nn.Module):
    """BiFPN Block with channel alignment"""

    def __init__(self, c1, c2, num_inputs=2, eps=1e-4):
        super().__init__()
        self.eps = eps
        self.num_inputs = num_inputs
        self.out_channels = c2

        # 为每个输入创建独立的通道对齐卷积
        self.align_convs = nn.ModuleList()
        for _ in range(num_inputs):
            self.align_convs.append(Conv(c1, c2, k=1, s=1, act=False))

        # 可学习权重
        self.w = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32), requires_grad=True)

        # 特征转换层
        self.conv = Conv(c2, c2, k=3, s=1, p=1, g=1, act=True)

    def forward(self, x):
        # 对每个输入进行通道对齐
        aligned = []
        for i in range(self.num_inputs):
            aligned.append(self.align_convs[i](x[i]))

        # 计算归一化权重
        weights = F.relu(self.w)
        weights = weights / (weights.sum() + self.eps)

        # 加权特征融合
        fused = sum(weights[i] * aligned[i] for i in range(self.num_inputs))

        # 特征转换
        return self.conv(fused)




