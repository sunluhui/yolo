import torch
import torch.nn as nn


class SimAM(torch.nn.Module):
    def __init__(self, channels, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.channels = channels  # 添加通道参数记录
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activation(y)
