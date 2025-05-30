import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv


class BiFPN_Concat2(nn.Module):
    def __init__(self, c1, c2, num_inputs=2, eps=1e-4):
        super().__init__()
        self.eps = eps
        self.num_inputs = num_inputs

        # 可学习权重参数
        self.w = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32), requires_grad=True)

        # 特征转换层
        self.conv = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        # 输入x是特征图列表
        weights = F.relu(self.w)
        weights = weights / (weights.sum() + self.eps)

        # 加权特征融合
        fused = sum(weights[i] * x[i] for i in range(self.num_inputs))

        # 特征转换
        return self.act(self.bn(self.conv(fused)))





