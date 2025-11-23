import torch
from torch import nn
from timm.layers import DropPath
import torch.nn.functional as F
from timm.layers import trunc_normal_


# https://arxiv.org/pdf/2407.09379
# FANET: FEATURE AMPLIFICATION NETWORK FOR SEMANTIC SEGMENTATION IN CLUTTERED BACKGROUND
# 自定义的LayerNorm层，支持"channels_last"和"channels_first"两种数据格式
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))  # 可学习的缩放参数，初始化为1
        self.bias = nn.Parameter(torch.zeros(normalized_shape))  # 可学习的偏移参数，初始化为0
        self.eps = eps  # 防止除零的微小值
        self.data_format = data_format  # 数据格式
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError  # 如果数据格式不支持则抛出异常
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            # 使用F.layer_norm进行归一化
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # 手动实现归一化
            u = x.mean(1, keepdim=True)  # 计算均值
            s = (x - u).pow(2).mean(1, keepdim=True)  # 计算方差
            x = (x - u) / torch.sqrt(s + self.eps)  # 标准化
            x = self.weight[:, None, None] * x + self.bias[:, None, None]  # 应用缩放和偏移
            return x


# 特征细化模块，用于特征增强和提取
class FeatureRefinementModule(nn.Module):
    def __init__(self, in_dim=128, out_dim=128, down_kernel=5, down_stride=4):
        super().__init__()

        # 深度可分离卷积层，用于低频和高频特征提取
        self.lconv = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, groups=in_dim)
        self.hconv = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, groups=in_dim)
        self.norm1 = LayerNorm(in_dim, eps=1e-6, data_format="channels_first")  # 用于低频特征的归一化
        self.norm2 = LayerNorm(in_dim, eps=1e-6, data_format="channels_first")  # 用于高频特征的归一化
        self.act = nn.GELU()  # 激活函数

        # 下采样层，用于减少特征图的尺寸 低频特征通常代表图像中的平滑或全局信息，它们可以通过对输入特征进行下采样、滤波和归一化等操作来提取

        self.down = nn.Conv2d(in_dim, in_dim, kernel_size=down_kernel, stride=down_stride, padding=down_kernel // 2,
                              groups=in_dim)
        self.proj = nn.Conv2d(in_dim * 2, out_dim, kernel_size=1, stride=1, padding=0)  # 投影层，用于合并低频和高频特征

        self.apply(self._init_weights)  # 初始化权重

    def _init_weights(self, m):
        # 初始化权重的方法
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)  # 使用截断正态分布初始化
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # 偏置初始化为0

        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, C, H, W = x.shape
        # 首先对输入特征进行下采样（通过卷积操作）。下采样减少了特征图的空间分辨率，保留了图像的粗略结构，这有助于捕捉低频信息。
        dx = self.down(x)  # 下采样特征
        # 将下采样的特征图恢复到原始尺寸，近似地保留了原图的低频信息。
        udx = F.interpolate(dx, size=(H, W), mode='bilinear', align_corners=False)  # 恢复特征图尺寸
        # 利用低频特征（udx）与输入特征（x）的乘积进行卷积，进一步提取低频信息。norm1和act操作帮助稳定训练并增强非线性特征。
        lx = self.norm1(self.lconv(self.act(x * udx)))  # 低频特征细化
        hx = self.norm2(self.hconv(self.act(x - udx)))  # 高频特征细化

        out = self.act(self.proj(torch.cat([lx, hx], dim=1)))  # 合并并投影特征

        return out


# 自适应特征增强模块
class AFE(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()

        # 深度可分离卷积，用于初步特征提取
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim // 2, 1, padding=0)  # 将通道数减半
        self.proj2 = nn.Conv2d(dim, dim, 1, padding=0)  # 恢复通道数

        # 上下文卷积层，用于提取全局上下文信息
        self.ctx_conv = nn.Conv2d(dim // 2, dim // 2, kernel_size=7, padding=3, groups=4)

        # 归一化层
        self.norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(dim // 2, eps=1e-6, data_format="channels_first")
        self.norm3 = LayerNorm(dim // 2, eps=1e-6, data_format="channels_first")

        # 特征细化模块
        self.enhance = FeatureRefinementModule(in_dim=dim // 2, out_dim=dim // 2, down_kernel=3, down_stride=2)

        self.act = nn.GELU()  # 激活函数

    def forward(self, x):
        B, C, H, W = x.shape

        x = x + self.norm1(self.act(self.dwconv(x)))  # 初步特征增强
        x = self.norm2(self.act(self.proj1(x)))  # 通道数减半后归一化

        ctx = self.norm3(self.act(self.ctx_conv(x)))  # 提取全局上下文信息

        enh_x = self.enhance(x)  # 细化特征
        x = self.act(self.proj2(torch.cat([ctx, enh_x], dim=1)))  # 合并上下文和细化特征

        return x


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Bottleneck_AFE(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.cv3 = AFE(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))


class C2f_AFE(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            Bottleneck_AFE(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))