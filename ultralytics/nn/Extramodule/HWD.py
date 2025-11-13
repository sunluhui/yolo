# custom_modules.py
# 这个文件用于存放为复现论文而自定义的网络模块。
import torch
import torch.nn as nn
import torch.nn.functional as F


class HWD(nn.Module):
    """
    Haar Wavelet Downsampling module as described in the paper.
    This module replaces a standard strided convolution for downsampling.
    """

    def __init__(self, c1, c2, k=1, s=2, p=None, g=1, act=True):
        super(HWD, self).__init__()
        # 接收的c1是Haar变换前（上一层）的通道数
        # Haar变换后通道数会变为4倍
        self.c_ = int(c1 * 4)
        # 定义后续的卷积层，输入通道为c_，输出通道为c2
        self.conv = nn.Conv2d(self.c_, c2, k, 1, autopad(k), groups=g, bias=False)  # 步长在这里设为1，因为Haar变换已经降采样
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.s = s  # 保存步长参数，用于验证

    def forward(self, x):
        # 执行Haar小波变换
        # 假设输入 x 的形状是 (B, C, H, W)
        B, C, H, W = x.shape

        # 将输入分解为4个子带
        # (B, C, H/2, W/2)
        x_ll = x[:, :, 0::2, 0::2]  # 左上角子带
        x_lh = x[:, :, 0::2, 1::2]  # 右上角子带
        x_hl = x[:, :, 1::2, 0::2]  # 左下角子带
        x_hh = x[:, :, 1::2, 1::2]  # 右下角子带

        # 在通道维度上拼接
        # (B, 4*C, H/2, W/2)
        x = torch.cat((x_ll, x_lh, x_hl, x_hh), 1)

        # 通过卷积层进行处理
        return self.act(self.bn(self.conv(x)))


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p