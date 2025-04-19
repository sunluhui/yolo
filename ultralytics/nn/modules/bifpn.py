import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv


class BiFPN_Concat2(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat2, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1]]
        return torch.cat(x, self.d)


# 三个分支concat操作
# ultralytics/nn/modules/bifpn.py
class BiFPN_Concat3(nn.Module):
    def __init__(self, c1, c2, c3, dimension=1):  # 显式接收三个输入通道参数
        super().__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.epsilon = 0.0001

        # 通道对齐卷积（可选）
        self.conv1 = Conv(c1, c1, 1) if c1 != c2 else nn.Identity()
        self.conv2 = Conv(c2, c2, 1) if c2 != c3 else nn.Identity()

    def forward(self, x):
        x0 = self.conv1(x[0])
        x1 = self.conv2(x[1])
        x2 = x[2]

        w = self.w / (torch.sum(self.w, dim=0) + self.epsilon)
        return torch.cat([w[0] * x0, w[1] * x1, w[2] * x2], self.d)


