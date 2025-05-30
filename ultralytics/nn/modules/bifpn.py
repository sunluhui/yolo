import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv


class BiFPN_Concat2(nn.Module):
    """Robust BiFPN Block with Automatic Channel Alignment"""

    def __init__(self, c2, num_inputs=2, eps=1e-4):
        """
        Args:
            c2 (int): 输出通道数
            num_inputs (int): 输入特征图数量
            eps (float): 防止除零的小常数
        """
        super().__init__()
        self.eps = eps
        self.num_inputs = num_inputs
        self.out_channels = c2

        # 为每个输入创建通道对齐层
        self.align_convs = nn.ModuleList([
            Conv(c2, c2, k=1, s=1, act=False) for _ in range(num_inputs)
        ])

        # 可学习权重
        self.w = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32), requires_grad=True)

        # 特征转换层
        self.conv = Conv(c2, c2, k=3, s=1, p=1, g=1, act=True)

    def forward(self, x):
        """
        处理输入特征图，确保通道数一致
        """
        # 对齐所有输入特征图到统一通道数
        aligned = [self.align_convs[i](x[i]) for i in range(self.num_inputs)]

        # 计算归一化权重
        weights = F.relu(self.w)
        weights = weights / (weights.sum() + self.eps)

        # 加权特征融合
        fused = sum(weights[i] * aligned[i] for i in range(self.num_inputs))

        # 特征转换
        return self.conv(fused)