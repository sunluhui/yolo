import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__() # 调用父类初始化方法
        # 初始化权重参数，形状为normalized_shape，初始值为1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        # 初始化偏置参数，形状为normalized_shape，初始值为0
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps # 小常量，防止除以零
        self.data_format = data_format # 数据格式，支持"channels_last"和"channels_first"
        # 如果数据格式不是"channels_last"或"channels_first"，抛出NotImplementedError异常
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,) # 归一化形状

    def forward(self, x):
        # 如果数据格式是"channels_last"，使用F.layer_norm函数进行层归一化
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # 如果数据格式是"channels_first"，手动实现层归一化
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True) # 计算均值
            s = (x - u).pow(2).mean(1, keepdim=True) # 计算方差
            x = (x - u) / torch.sqrt(s + self.eps) # 标准化
            x = self.weight[:, None, None] * x + self.bias[:, None, None] # 应用权重和偏置
            return x


# Multi-Scale Large Kernel Attention (MLKA)（多尺度大核注意力）
class MLKA(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        # 如果特征数量不能被3整除，则抛出ValueError异常
        if n_feats % 3 != 0:
            raise ValueError("n_feats must be divisible by 3 for MLKA.")
        i_feats = 2 * n_feats # 计算中间特征的数量

        self.norm = LayerNorm(n_feats, data_format='channels_first')

        self.proj_first = nn.Sequential(nn.Conv2d(n_feats, i_feats, 1, 1, 0))

        # 定义LKA3模块
        self.LKA3 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 3, 1, 1, groups=n_feats // 3), # 深度可分离卷积
            nn.Conv2d(n_feats // 3, n_feats // 3, 5, stride=1, padding=(5 // 2) * 2, groups=n_feats // 3, dilation=2), # 空洞卷积
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0)) # 1x1卷积
        self.X3 = nn.Conv2d(n_feats // 3, n_feats // 3, 3, 1, 1, groups=n_feats // 3) # 3x3深度可分离卷积

        # 定义LKA5模块
        self.LKA5 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 5, 1, 5 // 2, groups=n_feats // 3), # 深度可分离卷积
            nn.Conv2d(n_feats // 3, n_feats // 3, 7, stride=1, padding=(7 // 2) * 3, groups=n_feats // 3, dilation=3), # 空洞卷积
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0)) # 1x1卷积
        self.X5 = nn.Conv2d(n_feats // 3, n_feats // 3, 5, 1, 5 // 2, groups=n_feats // 3) # 5x5深度可分离卷积

        # 定义X7模块
        self.X7 = nn.Conv2d(n_feats // 3, n_feats // 3, 7, 1, 7 // 2, groups=n_feats // 3) # 7x7深度可分离卷积
        # 定义LKA7模块
        self.LKA7 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 7, 1, 7 // 2, groups=n_feats // 3), # 深度可分离卷积
            nn.Conv2d(n_feats // 3, n_feats // 3, 9, stride=1, padding=(9 // 2) * 4, groups=n_feats // 3, dilation=4), # 空洞卷积
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0)) # 1x1卷积

        # 定义最后一个投影层
        self.proj_last = nn.Sequential(nn.Conv2d(n_feats, n_feats, 1, 1, 0))

        # 定义缩放参数【可学习参数】
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

    def forward(self, x):
        # x：torch.Size([1, 3, 128, 128])
        shortcut = x.clone() # 保存输入作为残差连接
        x = self.norm(x) # 应用层归一化
        x = self.proj_first(x) # 第一个投影层 torch.Size([1, 6, 128, 128])

        # 分割张量成两个部分 torch.Size([1, 3, 128, 128]) torch.Size([1, 3, 128, 128])
        a, x = torch.chunk(x, 2, dim=1)

        # 分割a为三个部分
        a_1, a_2, a_3 = torch.chunk(a, 3, dim=1) # torch.Size([1, 1, 128, 128])
        # 应用LKA3, LKA5, LKA7模块并组合结果
        # [核心：大核卷积结果×深度可分离结果 实现加权注意力]
        a = torch.cat([self.LKA3(a_1) * self.X3(a_1),
                       self.LKA5(a_2) * self.X5(a_2),
                       self.LKA7(a_3) * self.X7(a_3)],dim=1) # torch.Size([1, 3, 128, 128])

        # 最后一个投影层，并应用缩放和残差连接
        """
            x * a : torch.Size([1, 3, 128, 128])
            self.proj_last(x * a): torch.Size([1, 3, 128, 128])
            self.scale : torch.Size([1, 3, 1, 1])
            shortcut :torch.Size([1, 3, 128, 128])
        """
        x = self.proj_last(x * a) * self.scale + shortcut
        return x