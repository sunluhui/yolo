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
# ultralytics/nn/modules/bifpn.py
class BiFPN_Concat3(nn.Module):
    def __init__(self, c1, c2, c3, dimension=1):
        super().__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.epsilon = 0.0001

        # 添加尺寸对齐模块
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')  # 用于小尺寸→大尺寸
        self.downsample = Conv(c3, c3, 3, 2)  # 用于大尺寸→小尺寸

    def forward(self, x):
        # 输入特征图尺寸验证
        assert x[0].shape[-2:] == x[1].shape[-2:], "前两个特征图尺寸必须一致"

        # 对第三个特征图进行下采样
        x2_resized = self.downsample(x[2])  # 假设x[2]是最大的特征图

        # 加权融合
        w = self.w / (torch.sum(self.w, dim=0) + self.epsilon)
        return torch.cat([
            w[0] * x[0],
            w[1] * x[1],
            w[2] * x2_resized  # 使用调整后的特征图
        ], self.d)



