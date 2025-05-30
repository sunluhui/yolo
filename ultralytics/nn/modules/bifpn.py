import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv


class BiFPN_Concat2(nn.Module):
    """BiFPN Block with automatic channel alignment"""

    def __init__(self, channels, c2, num_inputs=2, eps=1e-4):
        """
        channels: 输出通道数 (统一后的通道数)
        c2: 模块输出通道数
        num_inputs: 输入特征图数量
        """
        super().__init__()
        self.eps = eps
        self.num_inputs = num_inputs
        self.out_channels = c2

        # 创建自适应通道对齐层
        self.align_convs = nn.ModuleList()
        for _ in range(num_inputs):
            # 使用自适应卷积，不预设输入通道数
            self.align_convs.append(
                nn.Sequential(
                    nn.Conv2d(1, channels, kernel_size=1),  # 占位符，实际输入通道将在forward中确定
                    nn.BatchNorm2d(channels),
                    nn.SiLU()
                )
            )

        # 可学习权重
        self.w = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32), requires_grad=True)

        # 特征转换层
        self.conv = Conv(channels, c2, k=3, s=1, p=1, g=1, act=True)

        # 标记第一次运行
        self.first_run = True

    def forward(self, x):
        # 第一次运行时动态创建对齐卷积
        if self.first_run:
            self.first_run = False
            self._create_align_convs(x)

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

    def _create_align_convs(self, x):
        """动态创建对齐卷积层，基于实际输入通道数"""
        for i in range(self.num_inputs):
            in_channels = x[i].size(1)

            # 创建正确的卷积层
            align_conv = nn.Sequential(
                nn.Conv2d(in_channels, self.out_channels, kernel_size=1),
                nn.BatchNorm2d(self.out_channels),
                nn.SiLU()
            )

            # 替换占位符
            self.align_convs[i] = align_conv.to(x[i].device)

            # 打印调试信息
            print(f"Created align_conv for input {i}: {in_channels} -> {self.out_channels} channels")






