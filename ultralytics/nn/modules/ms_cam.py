import torch
import torch.nn as nn


class MS_CAM(nn.Module):
    """
    多尺度通道注意力机制 (MS-CAM)
    论文: https://arxiv.org/abs/2105.02477
    """

    def __init__(self, channels, reduction=4):
        super(MS_CAM, self).__init__()
        mid_channels = channels // reduction

        # 局部特征分支 (1x1卷积)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        # 全局特征分支 (全局平均池化)
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 并行处理两个分支
        local_feat = self.local_att(x)
        global_feat = self.global_att(x)

        # 特征融合
        feat_sum = local_feat + global_feat
        attention = self.sigmoid(feat_sum)

        # 应用注意力权重
        return x * attention.expand_as(x)