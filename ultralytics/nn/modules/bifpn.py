import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv


class BiFPN_Concat2(nn.Module):
    """Bidirectional Feature Pyramid Network Block with Weighted Feature Fusion"""

    def __init__(self, c2, num_inputs=2, eps=1e-4):
        """
        Args:
            c2 (int): 输出通道数
            num_inputs (int): 输入特征图数量 (通常为2)
            eps (float): 防止除零的小常数
        """
        super().__init__()
        self.eps = eps
        self.num_inputs = num_inputs
        self.out_channels = c2

        # 可学习权重参数 (每个输入一个权重)
        self.w = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32), requires_grad=True)

        # 特征转换层
        self.conv = Conv(c2, c2, k=3, s=1, p=1, g=1, act=True)

    def forward(self, x):
        """
        Args:
            x (list of torch.Tensor): 输入特征图列表

        Returns:
            (torch.Tensor): 融合并转换后的特征图
        """
        # 确保所有输入特征图通道数一致
        if not all(x[i].size(1) == x[0].size(1) for i in range(1, self.num_inputs)):
            # 自动对齐通道数
            target_channels = x[0].size(1)
            aligned = [x[0]]
            for i in range(1, self.num_inputs):
                if x[i].size(1) != target_channels:
                    # 使用1x1卷积对齐通道
                    align_conv = Conv(x[i].size(1), target_channels, k=1).to(x[i].device)
                    aligned.append(align_conv(x[i]))
                else:
                    aligned.append(x[i])
            x = aligned

        # 计算归一化权重
        weights = F.relu(self.w)
        weights = weights / (weights.sum() + self.eps)

        # 加权特征融合
        fused = sum(weights[i] * x[i] for i in range(self.num_inputs))

        # 特征转换
        return self.conv(fused)


