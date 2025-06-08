# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ultralytics.utils.torch_utils import fuse_conv_and_bn
from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock
__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "C3k2",
    "C2fPSA",
    "C2PSA",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
    "TorchVision",
    "CoordAtt",
    "SPPFCSPC",
)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF_DC(nn.Module):
    """SPPF with Dilated Convolutions for small objects"""

    def __init__(self, c1, c2=None, k=5, dilation_rates=[1, 2, 3, 5]):
        super().__init__()
        c2 = c2 or c1
        hidden_c = c1 // 2  # 减少通道数以降低计算量

        # 多尺度空洞卷积分支
        self.branches = nn.ModuleList()
        for rate in dilation_rates:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(c1, hidden_c, kernel_size=k,
                              padding=rate * (k - 1) // 2, dilation=rate, bias=False),
                    nn.BatchNorm2d(hidden_c),
                    nn.SiLU()
                )
            )

        # 原始最大池化分支
        self.maxpool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(c1 + hidden_c * len(dilation_rates), c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

    def forward(self, x):
        x0 = x
        # 原始SPPF分支
        x1 = self.maxpool(x)
        x2 = self.maxpool(x1)
        x3 = self.maxpool(x2)
        sppf_out = torch.cat([x, x1, x2, x3], 1)

        # 多尺度空洞卷积分支
        dc_outs = [branch(x) for branch in self.branches]

        # 融合特征
        all_features = torch.cat([sppf_out] + dc_outs, dim=1)
        return self.fusion(all_features)


class SPPF_Att(nn.Module):
    """SPPF with Attention Mechanism"""

    def __init__(self, c1, c2=None, k=5, reduction_ratio=8):
        super().__init__()
        c2 = c2 or c1

        # 原始SPPF分支
        self.maxpool1 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(4 * c1, max(4 * c1 // reduction_ratio, 4), 1),
            nn.SiLU(),
            nn.Conv2d(max(4 * c1 // reduction_ratio, 4), 4 * c1, 1),
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # 输出转换
        self.conv = nn.Conv2d(4 * c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        # 原始SPPF处理
        x1 = self.maxpool1(x)
        x2 = self.maxpool2(x1)
        x3 = self.maxpool3(x2)
        sppf_out = torch.cat([x, x1, x2, x3], 1)

        # 通道注意力
        channel_att = self.channel_att(sppf_out)
        channel_out = sppf_out * channel_att

        # 空间注意力
        spatial_avg = torch.mean(channel_out, dim=1, keepdim=True)
        spatial_max, _ = torch.max(channel_out, dim=1, keepdim=True)
        spatial_concat = torch.cat([spatial_avg, spatial_max], dim=1)
        spatial_att = self.spatial_att(spatial_concat)
        att_out = channel_out * spatial_att

        # 输出
        return self.act(self.bn(self.conv(att_out)))


class SPPF_GAP(nn.Module):
    """SPPF with Global Adaptive Pooling"""

    def __init__(self, c1, c2=None, k=5):
        super().__init__()
        c2 = c2 or c1

        # 原始SPPF分支
        self.maxpool1 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

        # 全局上下文分支
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c1 // 4, 1),
            nn.SiLU(),
            nn.Conv2d(c1 // 4, c1, 1),
            nn.Sigmoid()
        )

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(4 * c1 + c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

    def forward(self, x):
        # 原始SPPF处理
        x1 = self.maxpool1(x)
        x2 = self.maxpool2(x1)
        x3 = self.maxpool3(x2)
        sppf_out = torch.cat([x, x1, x2, x3], 1)

        # 全局上下文
        global_att = self.global_branch(x)
        global_feat = x * global_att

        # 特征融合
        all_features = torch.cat([sppf_out, global_feat], dim=1)
        return self.fusion(all_features)


class SPPFA(nn.Module):
    """SPPFA: Spatial Pyramid Pooling with Feature Aggregation for Small Object Detection"""

    def __init__(self, c1, c2=None, k=3, e=0.75, dilation_rates=[1, 2, 3, 5], use_attention=True):
        """
        Args:
            c1: 输入通道数
            c2: 输出通道数 (默认 c1)
            k: 基础卷积核尺寸 (默认 3)
            e: 扩展率 (控制中间通道数)
            dilation_rates: 空洞卷积的扩张率 (多尺度特征提取)
            use_attention: 是否使用注意力机制
        """
        super().__init__()
        c2 = c2 or c1  # 输出通道数默认等于输入通道数
        self.c1 = c1
        self.c2 = c2
        hidden_channels = int(c1 * e)  # 中间通道数
        self.dilation_rates = dilation_rates
        self.use_attention = use_attention

        # 多尺度空洞卷积分支
        self.dilated_convs = nn.ModuleList()
        for rate in dilation_rates:
            padding = rate * (k - 1) // 2  # 保持输出尺寸不变
            self.dilated_convs.append(
                nn.Sequential(
                    nn.Conv2d(c1, hidden_channels, kernel_size=k,
                              padding=padding, dilation=rate, bias=False),
                    nn.BatchNorm2d(hidden_channels),
                    nn.SiLU()
                )
            )

        # 全局上下文分支
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, hidden_channels, 1),
            nn.SiLU()
        )

        # 修复1: 正确计算融合层的输入通道数
        # 分支数量 = 空洞卷积分支数 + 1 (全局分支)
        num_branches = len(dilation_rates) + 1
        # 每个分支输出 hidden_channels 个通道
        fuse_in_channels = hidden_channels * num_branches

        # 特征融合层 (修复2: 使用计算好的通道数)
        self.fusion = nn.Sequential(
            nn.Conv2d(fuse_in_channels, hidden_channels, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, c2, 1),
            nn.BatchNorm2d(c2)
        )

        # 注意力机制 (可选)
        if use_attention:
            self.attention = nn.Sequential(
                nn.Conv2d(c2, c2 // 4, 1),
                nn.SiLU(),
                nn.Conv2d(c2 // 4, c2, 1),
                nn.Sigmoid()
            )

        # 残差连接
        self.residual = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, 1)

    def forward(self, x):
        # 原始输入特征
        identity = x

        # 多尺度特征提取
        features = []
        for conv in self.dilated_convs:
            features.append(conv(x))

        # 全局上下文
        global_feat = self.global_branch(x)
        global_feat = F.interpolate(global_feat, size=x.shape[2:], mode='nearest')
        features.append(global_feat)

        # 特征拼接
        x_cat = torch.cat(features, dim=1)

        # 特征融合
        x_out = self.fusion(x_cat)

        # 应用注意力机制
        if self.use_attention:
            attn = self.attention(x_out)
            x_out = x_out * attn

        # 残差连接
        return self.residual(identity) + x_out


class FocalModulation(nn.Module):
    def __init__(self, in_channels, kernel_size=3, reduction=2, focal_level=4,
                 focal_window=5, *dilation_args, use_ca=True):
        super().__init__()

        # 更健壮的处理 dilation_rates 参数
        dilation_rates = []

        # 处理所有 dilation 参数
        for arg in dilation_args:
            if isinstance(arg, (list, tuple)):
                # 展平嵌套列表
                for item in arg:
                    if isinstance(item, (list, tuple)):
                        dilation_rates.extend(item)
                    else:
                        dilation_rates.append(item)
            else:
                dilation_rates.append(arg)

        # 如果没有提供参数，使用默认值
        if not dilation_rates:
            dilation_rates = [1, 2, 4]

        # 确保所有元素都是整数
        try:
            dilation_rates = [int(d) for d in dilation_rates]
        except (TypeError, ValueError):
            # 如果转换失败，使用默认值
            print(f"Warning: Invalid dilation_rates {dilation_rates}, using default [1,2,4]")
            dilation_rates = [1, 2, 4]

        # 确保有足够的dilation_rates
        if len(dilation_rates) < focal_level:
            # 重复列表直到满足长度要求
            dilation_rates = dilation_rates * (focal_level // len(dilation_rates) + 1)
            dilation_rates = dilation_rates[:focal_level]

        self.in_channels = in_channels
        self.reduction = reduction
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.dilation_rates = dilation_rates
        self.use_ca = use_ca

        # 投影层
        self.projector = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.GELU(),
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3,
                      padding=1, groups=in_channels),
            nn.GELU()
        )

        # 多尺度上下文聚合
        self.aggregators = nn.ModuleList()
        for k in range(focal_level):
            # 获取当前层级的dilation值
            dilation_val = self.dilation_rates[k]

            # 计算当前层级的卷积核大小
            kernel_size_val = self.focal_window + 2 * k

            # 正确的padding计算
            padding_val = dilation_val * (kernel_size_val - 1) // 2
            padding_val = int(padding_val)  # 确保是整数

            # 添加聚合层
            self.aggregators.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels,
                              kernel_size=kernel_size_val,
                              padding=padding_val,
                              dilation=dilation_val,
                              groups=in_channels,
                              bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.GELU()
                )
            )

        # 门控机制
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 坐标注意力
        if use_ca:
            self.ca = CoordinateAttention(in_channels)

        # 调制器
        reduced_channels = max(in_channels // reduction, 32)
        self.modulator = nn.Sequential(
            nn.Conv2d(in_channels * focal_level, reduced_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1),
            nn.LayerNorm([in_channels, 1, 1])
        )

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        # 保存残差连接
        residual = x

        # 特征投影
        proj = self.projector(x)
        query, context = proj.chunk(2, dim=1)

        # 多尺度上下文聚合
        context_layers = []
        for agg in self.aggregators:
            ctx = agg(context)
            if self.use_ca:
                ctx = self.ca(ctx)  # 应用坐标注意力
            context_layers.append(ctx)

        # 拼接多尺度特征
        context_all = torch.cat(context_layers, dim=1)

        # 调制特征
        modulated = self.modulator(context_all) * query

        # 空间门控
        gate = self.gate(x)
        modulated = modulated * gate

        # 输出投影 + 残差连接
        output = self.output_proj(modulated) + residual
        return output


class CoordinateAttention(nn.Module):
    """坐标注意力机制 - 特别适合小目标定位"""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        reduced_channels = max(in_channels // reduction, 8)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(reduced_channels)
        self.act = nn.ReLU()

        self.conv_h = nn.Conv2d(reduced_channels, in_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(reduced_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # 高度方向注意力
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # 特征融合
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 分离高度和宽度注意力
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # 生成注意力图
        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))

        return identity * a_h * a_w


class SmallObjectAmplifier(nn.Module):
    """针对小目标的特征增强模块"""

    def __init__(self, in_channels, scale_factors=[2, 4]):
        super().__init__()
        self.scale_factors = scale_factors
        self.conv_layers = nn.ModuleList()

        # 创建多尺度卷积
        for factor in scale_factors:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
                    nn.BatchNorm2d(in_channels // 4),
                    nn.SiLU()
                )
            )

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels // 4 * len(scale_factors), in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        amplified_features = []
        B, C, H, W = x.shape

        for i, factor in enumerate(self.scale_factors):
            # 上采样
            resized = F.interpolate(x, scale_factor=factor, mode='bilinear', align_corners=False)

            # 特征提取
            features = self.conv_layers[i](resized)

            # 下采样回原尺寸
            features = F.interpolate(features, size=(H, W), mode='bilinear', align_corners=False)
            amplified_features.append(features)

        # 融合放大特征
        fused = self.fusion(torch.cat(amplified_features, dim=1))

        # 增强小目标特征
        return x * (1 + fused)


class AdaptiveSPPF(nn.Module):
    def __init__(self, in_channels, out_channels, scales=[1, 2, 4, 8], use_soa=True):
        super().__init__()
        self.scales = scales
        self.branches = nn.ModuleDict()
        self.use_soa = use_soa

        # 主分支（恒等映射）
        self.branches['identity'] = nn.Identity()

        # 多尺度自适应池化分支
        for scale in scales:
            self.branches[f'pool_{scale}'] = nn.Sequential(
                nn.AdaptiveMaxPool2d(output_size=(None, None)),  # 动态内核占位
                nn.Conv2d(in_channels, out_channels // len(scales), 1, bias=False)
            )

        # 小目标增强模块
        if self.use_soa:
            self.soa = SmallObjectAmplifier(in_channels)

        self.fusion_conv = nn.Conv2d(
            in_channels + out_channels // len(scales) * len(scales),
            out_channels, 1
        )

    def _dynamic_pool_params(self, x, scale):
        """动态计算池化参数"""
        H, W = x.shape[2:]
        kernel_size = max(1, int(min(H, W) * scale / 32))  # 基于特征图尺寸的缩放因子
        stride = max(1, kernel_size // 2)  # 自动步长策略
        return kernel_size, stride

    def forward(self, x):
        # 小目标增强
        if self.use_soa:
            x = self.soa(x)

        outputs = [self.branches['identity'](x)]

        for scale in self.scales:
            # 动态生成池化层
            kernel, stride = self._dynamic_pool_params(x, scale)
            pool_layer = nn.MaxPool2d(kernel, stride, kernel // 2)

            # 执行池化并处理
            pooled = pool_layer(x)
            processed = self.branches[f'pool_{scale}'](pooled)
            outputs.append(nn.functional.interpolate(
                processed, size=x.shape[2:], mode='nearest'
            ))

        # 多尺度特征融合
        return self.fusion_conv(torch.cat(outputs, dim=1))


class MultiModalFusion(nn.Module):
    """融合可见光与红外特征的小目标增强模块"""

    def __init__(self, c1, c2):
        super().__init__()
        self.vis_conv = nn.Sequential(
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )
        self.ir_conv = nn.Sequential(
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )
        self.fusion_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2 * 2, c2 // 8, 1),
            nn.ReLU(),
            nn.Conv2d(c2 // 8, 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, vis_feat, ir_feat=None):
        # 单模态时直接返回可见光特征
        if ir_feat is None:
            return self.vis_conv(vis_feat)

        vis_out = self.vis_conv(vis_feat)
        ir_out = self.ir_conv(ir_feat)

        # 特征拼接
        fused = torch.cat([vis_out, ir_out], dim=1)

        # 注意力权重
        att_weights = self.fusion_att(fused)
        v_weight, i_weight = att_weights[:, 0:1], att_weights[:, 1:2]

        # 加权融合
        return v_weight * vis_out + i_weight * ir_out


class DynamicRFAtt(nn.Module):
    """自适应选择最佳感受野的注意力机制"""

    def __init__(self, in_channels, kernels=[1, 3, 5, 7]):
        super().__init__()
        self.branches = nn.ModuleList()
        self.kernels = kernels

        # 创建不同感受野的分支
        for k in kernels:
            padding = k // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, k, padding=padding, groups=in_channels),
                    nn.BatchNorm2d(in_channels),
                    nn.SiLU()
                )
            )

        # 动态选择器
        self.selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 32),
            nn.ReLU(),
            nn.Linear(32, len(kernels)),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # 并行计算各分支
        branch_outputs = [branch(x) for branch in self.branches]

        # 生成选择权重
        weights = self.selector(x)  # [B, num_kernels]

        # 加权融合
        out = torch.zeros_like(x)
        for i in range(len(self.kernels)):
            weight = weights[:, i].view(-1, 1, 1, 1)
            out += weight * branch_outputs[i]

        return out


class AdvancedCA_RFA_EnhancedSPPF(nn.Module):
    """终极改进版SPPF模块 - 专为无人机小目标检测优化"""

    def __init__(self, c1, c2, k=5, multimodal=False):
        super().__init__()
        self.multimodal = multimodal
        c_ = c1 * 2 // 3  # 减少通道压缩

        # 多模态输入处理
        self.modal_fusion = MultiModalFusion(c1, c_) if multimodal else None

        # 输入卷积 - 使用可变形卷积
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, groups=c1),  # 深度卷积
            nn.Conv2d(c1, c_, kernel_size=1),  # 点卷积
            nn.BatchNorm2d(c_),
            nn.SiLU()
        )

        # 多尺度池化分支
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)

        # 频域增强坐标注意力
        self.ca0 = EnhancedCoordAtt(c_)
        self.ca1 = EnhancedCoordAtt(c_)
        self.ca2 = EnhancedCoordAtt(c_)
        self.ca3 = EnhancedCoordAtt(c_)

        # 输出卷积
        self.cv2 = nn.Sequential(
            nn.Conv2d(c_ * 4, c2, 1),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

        # 动态感受野注意力
        self.rfa = DynamicRFAtt(c2, kernels=[1, 3, 5, 7])

        # 残差连接
        self.residual = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1, groups=c1),
            nn.Conv2d(c1, c2, 1),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        ) if c1 != c2 else nn.Identity()

        # 小目标特征放大
        self.object_amp = SmallObjectAmplifier(c2, scale_factors=[2, 4])

        # 自适应参数
        self.alpha = nn.Parameter(torch.tensor(0.6))
        self.beta = nn.Parameter(torch.tensor(0.4))

    def forward(self, x, ir_x=None):
        # 多模态融合
        if self.multimodal and ir_x is not None:
            x = self.modal_fusion(x, ir_x)
        elif self.multimodal:
            x = self.modal_fusion(x)

        identity = self.residual(x)

        # 第一阶段：通道调整
        x = self.cv1(x)

        # 多尺度池化分支
        y0 = self.ca0(x)  # 原始特征 + CA
        y1 = self.ca1(self.pool1(x))
        y2 = self.ca2(self.pool2(x))
        y3 = self.ca3(self.pool3(x))

        # 特征拼接
        x = torch.cat([y0, y1, y2, y3], dim=1)

        # 输出卷积
        x = self.cv2(x)

        # 感受野注意力
        x = self.rfa(x)

        # 小目标特征放大
        x = self.object_amp(x)

        # 残差连接 + 自适应融合
        return self.beta * identity + self.alpha * x


class EnhancedCoordAtt(nn.Module):
    """增强版坐标注意力机制 - 改进小目标特征提取"""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # 使用分组卷积减少参数
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mid_channels = max(8, in_channels // reduction)

        # 添加深度可分离卷积增强特征提取
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.Hardswish(inplace=True)
        )

        # 使用不同卷积核增强特征
        self.conv_h = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1)
        )
        self.conv_w = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1)
        )

        # 添加通道注意力作为补充
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x

        # 高度方向注意力
        h = self.pool_h(x)
        h = self.conv1(h)
        h = self.conv_h(h)
        h_att = self.sigmoid(h)

        # 宽度方向注意力
        w = self.pool_w(x)
        w = self.conv1(w)
        w = self.conv_w(w)
        w_att = self.sigmoid(w)

        # 通道注意力
        c_att = self.channel_att(x)

        # 融合空间和通道注意力
        att = h_att * w_att * c_att

        # 应用注意力权重
        return identity * att + x  # 残差连接


class RFAtt(nn.Module):
    """感受野注意力机制 (Receptive Field Attention)"""

    def __init__(self, in_channels, kernels=[3, 5, 7], reduction=16):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernels:
            padding = k // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=k, padding=padding, groups=in_channels),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True)
                ))

            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(in_channels * len(kernels), in_channels // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels // reduction, in_channels * len(kernels)),
                nn.Sigmoid()
            )

    def forward(self, x):
        batch_size, _, H, W = x.shape
        outputs = []

        # 并行多分支卷积
        for branch in self.branches:
            outputs.append(branch(x))

        # 拼接多尺度特征
        u = torch.cat(outputs, dim=1)

        # 通道注意力
        s = self.avg_pool(u).view(batch_size, -1)
        z = self.fc(s).view(batch_size, -1, 1, 1)

        # 特征加权融合
        att_map = u * z

        # 分割加权后的特征
        split_att = torch.split(att_map, x.size(1), dim=1)

        # 特征融合
        out = sum(split_att) / len(split_att)
        return out


class CA_RFA_EnhancedSPPF(nn.Module):
    """增强版SPPF模块 - 专为小目标检测优化"""

    def __init__(self, c1, c2, k=5, bins=100):
        super().__init__()
        # 保留更多细节信息
        c_ = c1 * 2 // 3  # 减少通道压缩

        # 输入卷积 - 使用深度可分离卷积保留细节
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, groups=c1),  # 深度卷积
            nn.Conv2d(c1, c_, kernel_size=1),  # 点卷积
            nn.BatchNorm2d(c_),
            nn.SiLU()
        )

        # 多尺度池化分支 - 使用不同大小的池化核
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)

        # 坐标注意力模块 (每个分支后) - 使用增强版
        self.ca0 = EnhancedCoordAtt(c_)  # 原始特征
        self.ca1 = EnhancedCoordAtt(c_)  # 第一层池化
        self.ca2 = EnhancedCoordAtt(c_)  # 第二层池化
        self.ca3 = EnhancedCoordAtt(c_)  # 第三层池化

        # 输出卷积 - 使用残差连接
        self.cv2 = nn.Sequential(
            nn.Conv2d(c_ * 4, c2, 1, 1),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

        # 感受野注意力模块 - 使用增强版
        self.rfa = RFAtt(c2, kernels=[1, 3, 5])

        # 残差连接 - 使用卷积保留更多信息
        self.residual = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1, groups=c1),  # 深度卷积
            nn.Conv2d(c1, c2, 1),  # 点卷积
            nn.BatchNorm2d(c2),
            nn.SiLU()
        ) if c1 != c2 else nn.Identity()

        # 自适应参数 - 使用多个参数
        self.alpha = nn.Parameter(torch.tensor(0.7))  # 注意力融合系数
        self.beta = nn.Parameter(torch.tensor(0.3))  # 残差融合系数

        # 小目标增强模块 - 高分辨率特征保留
        self.detail_enhance = nn.Sequential(
            nn.Conv2d(c1, c2 // 4, 3, padding=1),
            nn.BatchNorm2d(c2 // 4),
            nn.SiLU(),
            nn.Conv2d(c2 // 4, c2, 1),
            nn.Sigmoid()  # 生成注意力图
        )

    def forward(self, x):
        # 原始输入保留
        identity = self.residual(x)

        # 提取小目标细节特征
        detail_att = self.detail_enhance(x)

        # 第一阶段：通道调整
        x = self.cv1(x)

        # 多尺度池化分支
        y0 = self.ca0(x)  # 原始特征 + CA
        y1 = self.ca1(self.pool1(x))  # 第一层池化 + CA
        y2 = self.ca2(self.pool2(x))  # 第二层池化 + CA
        y3 = self.ca3(self.pool3(x))  # 第三层池化 + CA

        # 确保所有特征图尺寸一致
        target_size = y0.shape[2:]
        y1 = self._resize_if_needed(y1, target_size)
        y2 = self._resize_if_needed(y2, target_size)
        y3 = self._resize_if_needed(y3, target_size)

        # 特征拼接
        x = torch.cat([y0, y1, y2, y3], dim=1)

        # 输出卷积
        x = self.cv2(x)

        # 应用感受野注意力
        x = self.rfa(x)

        # 应用小目标细节增强
        x = x * (1 + detail_att)  # 增强小目标特征

        # 残差连接 + 自适应融合
        return self.beta * identity + self.alpha * x

    def _resize_if_needed(self, tensor, target_size):
        """如果需要，调整张量尺寸到目标尺寸"""
        if tensor.shape[2:] != target_size:
            return F.interpolate(tensor, size=target_size, mode='bilinear', align_corners=False)
        return tensor


class CoordAtt(nn.Module):
    """坐标注意力机制 (Coordinate Attention)"""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mid_channels = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.Hardswish(inplace=True)

        self.conv_h = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x

        # 高度方向的注意力
        h = self.pool_h(x)
        h = self.conv1(h)
        h = self.bn1(h)
        h = self.act(h)
        h = self.conv_h(h)
        h_att = self.sigmoid(h)

        # 宽度方向的注意力
        w = self.pool_w(x)
        w = self.conv1(w)
        w = self.bn1(w)
        w = self.act(w)
        w = self.conv_w(w)
        w_att = self.sigmoid(w)

        # 融合空间注意力
        att = h_att * w_att

        # 应用注意力权重
        return identity * att


class RFAtt(nn.Module):
    """感受野注意力机制 (Receptive Field Attention)"""

    def __init__(self, in_channels, kernels=[3, 5, 7], reduction=16):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernels:
            padding = k // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=k, padding=padding, groups=in_channels),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True)
                ))

            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(in_channels * len(kernels), in_channels // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels // reduction, in_channels * len(kernels)),
                nn.Sigmoid()
            )

    def forward(self, x):
        batch_size, _, H, W = x.shape
        outputs = []

        # 并行多分支卷积
        for branch in self.branches:
            outputs.append(branch(x))

        # 拼接多尺度特征
        u = torch.cat(outputs, dim=1)

        # 通道注意力
        s = self.avg_pool(u).view(batch_size, -1)
        z = self.fc(s).view(batch_size, -1, 1, 1)

        # 特征加权融合
        att_map = u * z

        # 分割加权后的特征
        split_att = torch.split(att_map, x.size(1), dim=1)

        # 特征融合
        out = sum(split_att) / len(split_att)
        return out


class CA_RFA_SPPF(nn.Module):
    """集成CA和RFA的改进SPPF模块"""

    def __init__(self, c1, c2, k=5, bins=100):
        super().__init__()
        # 通道调整
        c_ = c1 // 2  # 减少通道数

        # 输入卷积
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, c_, 1, 1),
            nn.BatchNorm2d(c_),
            nn.SiLU()
        )

        # 池化分支
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

        # 坐标注意力模块 (每个分支后)
        self.ca1 = CoordAtt(c_)
        self.ca2 = CoordAtt(c_)
        self.ca3 = CoordAtt(c_)

        # 输出卷积
        self.cv2 = nn.Sequential(
            nn.Conv2d(c_ * 4, c2, 1, 1),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

        # 感受野注意力模块 (拼接后)
        self.rfa = RFAtt(c2, kernels=[3, 5, 7])

        # 残差连接
        self.residual = nn.Conv2d(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

        # 自适应参数
        self.alpha = nn.Parameter(torch.tensor(0.7))  # 注意力融合系数

    def forward(self, x):
        # 原始输入保留
        identity = self.residual(x)

        # 第一阶段：通道减少
        x = self.cv1(x)

        # 多尺度池化分支
        y0 = x  # 原始特征
        y1 = self.m(x)  # 第一层池化
        y2 = self.m(y1)  # 第二层池化
        y3 = self.m(y2)  # 第三层池化

        # 在池化分支后应用坐标注意力
        y1 = self.ca1(y1)
        y2 = self.ca2(y2)
        y3 = self.ca3(y3)

        # 特征拼接
        x = torch.cat([y0, y1, y2, y3], dim=1)

        # 输出卷积
        x = self.cv2(x)

        # 应用感受野注意力
        x = self.rfa(x)

        # 残差连接 + 自适应融合
        return identity + self.alpha * x


class DynamicConv2d(nn.Module):
    """CVPR 2024 动态卷积实现 - 修复版本"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=True, K=4, reduction=16):
        super().__init__()
        self.K = K  # 卷积核数量
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # 生成多个卷积核
        self.weight = nn.Parameter(
            torch.randn(K, out_channels, in_channels // groups, kernel_size, kernel_size),
            requires_grad=True
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_channels), requires_grad=True)
        else:
            self.bias = None

        # 注意力机制生成核权重
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, K, 1),
            nn.Softmax(dim=1))

        # 初始化参数
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        B, C, H, W = x.shape

        # 生成注意力权重 (B, K, 1, 1)
        attn_weights = self.attention(x).view(B, self.K, 1, 1)

        # 加权融合卷积核 (关键修复部分)
        combined_weight = torch.einsum('bki,oihw->bokihw', attn_weights, self.weight)

        # 分组卷积参数校验
        out_channels = self.cv2[0].out_channels
        groups = self.cv2[0].groups
        assert combined_weight.shape[1] == out_channels, "通道维度不匹配"
        assert combined_weight.shape[2] == C // groups, "输入通道分组错误"

        # 分组卷积实现
        x = x.view(1, B * C, H, W)  # 合并批量维度
        output = F.conv2d(
            x,
            weight=combined_weight.view(B * out_channels, C // groups, *self.kernel_size),
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=B * groups  # 修正分组参数
        )

        # 恢复输出形状
        return output.view(B, out_channels, output.size(2), output.size(3))

class AdaptivePooling(nn.Module):
    """自适应池化层 - 动态调整感受野 - 修复版本"""

    def __init__(self, channels, max_kernel=7, min_kernel=3):
        super().__init__()
        self.max_kernel = max_kernel
        self.min_kernel = min_kernel
        self.pool_types = ['max', 'avg']

        # 池化层选择器
        self.selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 2, 1),
            nn.Softmax(dim=1)
        )

        # 卷积池化层
        self.conv_pool = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)

    def forward(self, x):
        B, C, H, W = x.shape

        # 生成注意力权重 (B, K, 1, 1)
        attn_weights = self.attention(x).view(B, self.K, 1, 1)  # 确保是4维张量

        # 修复einsum维度匹配问题
        combined_weight = torch.einsum(
            'bki... , oihw -> bokihw...',  # 修正后的方程
            attn_weights,
            self.weight
        )

        # 动态调整权重形状
        out_channels = self.cv2[0].out_channels
        groups = self.cv2[0].groups

        # 验证形状兼容性
        assert combined_weight.dim() == 4, "combined_weight 必须是4维张量"
        assert combined_weight.shape[1] == out_channels, "通道维度不匹配"
        assert combined_weight.shape[2] == C // groups, "输入通道分组错误"

        # 分组卷积实现
        x = x.view(1, B * C, H, W)  # 合并批量维度
        output = F.conv2d(
            x,
            weight=combined_weight.view(B * out_channels, C // groups, *self.kernel_size),
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=B * groups
        )

        return output.view(B, out_channels, output.size(2), output.size(3))


class DynamicSPPF(nn.Module):
    """动态卷积增强的SPPF模块 - 修复版本"""

    def __init__(self, c1, c2, k=5, bins=100):
        super().__init__()
        # 使用动态卷积替代标准卷积
        self.cv1 = nn.Sequential(
            DynamicConv2d(c1, c1 // 2, kernel_size=1, stride=1, padding=0, K=4),
            nn.BatchNorm2d(c1 // 2),
            nn.SiLU()
        )

        # 自适应池化层
        self.pool = AdaptivePooling(c1 // 2)

        # 动态输出卷积
        self.cv2 = nn.Sequential(
            DynamicConv2d(c1 // 2 * 2, c2, kernel_size=1, stride=1, padding=0, K=4),  # 修正输入通道数
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

        # 残差连接
        self.residual = nn.Conv2d(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

        # 自适应融合参数
        self.alpha = nn.Parameter(torch.tensor(0.6))

    def forward(self, x):
        identity = self.residual(x)

        # 动态卷积减少通道
        x = self.cv1(x)

        # 多尺度特征提取
        y0 = x
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)

        # 特征拼接 (原始特征 + 最深特征)
        x = torch.cat([y0, y3], dim=1)  # 通道维度拼接

        # 动态输出卷积
        x = self.cv2(x)

        # 残差连接
        return identity + self.alpha * x


class SPPF_MultiScale(nn.Module):
    """多尺度SPPF模块 - 使用不同大小的池化核增强特征提取能力"""

    def __init__(self, c1, c2, pool_sizes=(3, 5, 7)):
        """
        Args:
            c1: 输入通道数
            c2: 输出通道数
            pool_sizes: 池化核尺寸列表 (默认: 3, 5, 7)
        """
        super().__init__()
        c_ = c1 // 2  # 隐藏层通道数

        # 1x1卷积降维
        self.cv1 = Conv(c1, c_, 1, 1)

        # 多尺度池化层
        self.pool_layers = nn.ModuleList([
            nn.MaxPool2d(kernel_size=size, stride=1, padding=size // 2)
            for size in pool_sizes
        ])

        # 1x1卷积升维
        self.cv2 = Conv(c_ * (len(pool_sizes) + 1), c2, 1, 1)

    def forward(self, x):
        """前向传播: 应用多尺度池化并拼接特征"""
        x = self.cv1(x)  # 降维
        features = [x]  # 原始特征

        # 应用不同尺度的池化
        for pool in self.pool_layers:
            features.append(pool(x))

        # 拼接所有特征并升维
        return self.cv2(torch.cat(features, dim=1))


class DroneSPPF(nn.Module):
    """Drone-Optimized SPPF for UAV Small Object Detection"""

    def __init__(self, c1, c2, k=3, dilation_rates=[1, 2, 3], use_attention=True):
        """
        Args:
            c1: 输入通道数
            c2: 输出通道数
            k: 基础池化尺寸 (推荐3-5)
            dilation_rates: 空洞卷积扩张率
            use_attention: 是否使用无人机专用注意力
        """
        super().__init__()
        # 降维层 (保留更多细节)
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, c1 // 2, 1, bias=False),
            nn.BatchNorm2d(c1 // 2),
            nn.SiLU(),
            nn.Conv2d(c1 // 2, c1 // 2, 3, padding=1, bias=False),  # 增加感受野
            nn.BatchNorm2d(c1 // 2),
            nn.SiLU()
        )

        # 多尺度池化分支 (不同尺寸的池化核)
        self.pool_layers = nn.ModuleList([
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
            for ks in [k, k + 2, k + 4]  # 3种不同尺寸
        ])

        # 多尺度空洞卷积分支 (增强小目标特征)
        self.dilated_convs = nn.ModuleList()
        for rate in dilation_rates:
            self.dilated_convs.append(
                nn.Sequential(
                    nn.Conv2d(c1 // 2, c1 // 4, 3,
                              padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(c1 // 4),
                    nn.SiLU()
                )
            )

        # 特征融合层
        fuse_in = (c1 // 2) * 4 + (c1 // 4) * len(dilation_rates)
        self.cv2 = nn.Sequential(
            nn.Conv2d(fuse_in, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

        # 无人机专用注意力机制
        if use_attention:
            self.attention = nn.Sequential(
                # 通道注意力
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c2, max(c2 // 16, 4), 1),
                nn.SiLU(),
                nn.Conv2d(max(c2 // 16, 4), c2, 1),
                nn.Sigmoid(),
                # 空间注意力 (简化)
                nn.Conv2d(c2, 1, kernel_size=3, padding=1),
                nn.Sigmoid()
            )
        else:
            self.attention = None

    def forward(self, x):
        # 第一步: 特征提取与降维
        x = self.cv1(x)

        # 多尺度池化分支
        pool_features = [x]
        for pool in self.pool_layers:
            pool_features.append(pool(pool_features[-1]))

        # 多尺度空洞卷积分支
        dilated_features = [conv(x) for conv in self.dilated_convs]

        # 特征拼接 (池化特征 + 空洞卷积特征)
        all_features = torch.cat(pool_features + dilated_features, dim=1)

        # 特征融合
        out = self.cv2(all_features)

        # 应用无人机专用注意力
        if self.attention is not None:
            attn = self.attention(out)
            out = out * attn

        return out


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes a CSP Bottleneck with 2 convolutions and optional shortcut connection."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc ** 0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initializes C2f module with attention mechanism for enhanced feature extraction and processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k ** 2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc ** 0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcuts, groups and expansion."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1, c2, c3, c4):
        """Initializes ELAN1 layer with specified channel sizes."""
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1, c2):
        """Initializes AConv module with convolution layers."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed) -> None:
        """Initializes RepVGGDW with depthwise separable convolutional layers for efficient processing."""
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Performs a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        Performs a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuses the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes the custom model with optional shortcut, scaling factor, and RepVGGDW layer."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initializes the module with specified parameters for channel, shortcut, local key, groups, and expansion."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = forward(input_tensor)
    """

    def __init__(self, c1, c2, e=0.5):
        """Initializes the PSA module with input/output channels and attention mechanism for feature extraction."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """Executes forward pass in PSA module, applying attention and feed-forward layers to the input tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2fPSA module, a variant of C2f with PSA blocks for enhanced feature extraction."""
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1, c2, k, s):
        """Initializes the SCDown module with specified input/output channels, kernel size, and stride."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        """Applies convolution and downsampling to the input tensor in the SCDown module."""
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """
    TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and customize the model by truncating or unwrapping layers.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.

    Args:
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): If True, unwraps the model to a sequential containing all but the last `truncate` layers. Default is True.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.
    """

    def __init__(self, model, weights="DEFAULT", unwrap=True, truncate=2, split=False):
        """Load the model and weights from torchvision."""
        import torchvision  # scope for faster 'import ultralytics'

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())
            if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x):
        """Forward pass through the model."""
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y


class AAttn(nn.Module):
    """
    Area-attention module with the requirement of flash attention.

    Attributes:
        dim (int): Number of hidden channels;
        num_heads (int): Number of heads into which the attention mechanism is divided;
        area (int, optional): Number of areas the feature map is divided. Defaults to 1.

    Methods:
        forward: Performs a forward process of input tensor and outputs a tensor after the execution of the area attention mechanism.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules import AAttn
        >>> model = AAttn(dim=64, num_heads=2, area=4)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)

    Notes:
        recommend that dim//num_heads be a multiple of 32 or 64.

    """

    def __init__(self, dim, num_heads, area=1):
        """Initializes the area-attention module, a simple yet efficient attention module for YOLO."""
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qk = Conv(dim, all_head_dim * 2, 1, act=False)
        self.v = Conv(dim, all_head_dim, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)

        self.pe = Conv(all_head_dim, dim, 5, 1, 2, g=dim, act=False)

    def forward(self, x):
        """Processes the input tensor 'x' through the area-attention"""
        B, C, H, W = x.shape
        N = H * W

        qk = self.qk(x).flatten(2).transpose(1, 2)
        v = self.v(x)
        pp = self.pe(v)
        v = v.flatten(2).transpose(1, 2)

        if self.area > 1:
            qk = qk.reshape(B * self.area, N // self.area, C * 2)
            v = v.reshape(B * self.area, N // self.area, C)
            B, N, _ = qk.shape
        q, k = qk.split([C, C], dim=2)
        try:
            import flash_attn  # 检查是否安装了flash_attn库
            USE_FLASH_ATTN = True
        except ImportError:
            USE_FLASH_ATTN = False
        if x.is_cuda and USE_FLASH_ATTN:
            q = q.view(B, N, self.num_heads, self.head_dim)
            k = k.view(B, N, self.num_heads, self.head_dim)
            v = v.view(B, N, self.num_heads, self.head_dim)

            x = flash_attn_func(
                q.contiguous().half(),
                k.contiguous().half(),
                v.contiguous().half()
            ).to(q.dtype)
        else:
            q = q.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
            k = k.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
            v = v.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)

            attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
            max_attn = attn.max(dim=-1, keepdim=True).values
            exp_attn = torch.exp(attn - max_attn)
            attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
            x = (v @ attn.transpose(-2, -1))

            x = x.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return self.proj(x + pp)


class ABlock(nn.Module):
    """
    ABlock class implementing a Area-Attention block with effective feature extraction.

    This class encapsulates the functionality for applying multi-head attention with feature map are dividing into areas
    and feed-forward neural network layers.

    Attributes:
        dim (int): Number of hidden channels;
        num_heads (int): Number of heads into which the attention mechanism is divided;
        mlp_ratio (float, optional): MLP expansion ratio (or MLP hidden dimension ratio). Defaults to 1.2;
        area (int, optional): Number of areas the feature map is divided.  Defaults to 1.

    Methods:
        forward: Performs a forward pass through the ABlock, applying area-attention and feed-forward layers.

    Examples:
        Create a ABlock and perform a forward pass
        >>> model = ABlock(dim=64, num_heads=2, mlp_ratio=1.2, area=4)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)

    Notes:
        recommend that dim//num_heads be a multiple of 32 or 64.
    """

    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        """Initializes the ABlock with area-attention and feed-forward layers for faster feature extraction."""
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights using a truncated normal distribution."""
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Executes a forward pass through ABlock, applying area-attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class A2C2f(nn.Module):
    """
    A2C2f module with residual enhanced feature extraction using ABlock blocks with area-attention. Also known as R-ELAN

    This class extends the C2f module by incorporating ABlock blocks for fast attention mechanisms and feature extraction.

    Attributes:
        c1 (int): Number of input channels;
        c2 (int): Number of output channels;
        n (int, optional): Number of 2xABlock modules to stack. Defaults to 1;
        a2 (bool, optional): Whether use area-attention. Defaults to True;
        area (int, optional): Number of areas the feature map is divided. Defaults to 1;
        residual (bool, optional): Whether use the residual (with layer scale). Defaults to False;
        mlp_ratio (float, optional): MLP expansion ratio (or MLP hidden dimension ratio). Defaults to 1.2;
        e (float, optional): Expansion ratio for R-ELAN modules. Defaults to 0.5;
        g (int, optional): Number of groups for grouped convolution. Defaults to 1;
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to True;

    Methods:
        forward: Performs a forward pass through the A2C2f module.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules import A2C2f
        >>> model = A2C2f(c1=64, c2=64, n=2, a2=True, area=4, residual=True, e=0.5)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        # num_heads = c_ // 64 if c_ // 64 >= 2 else c_ // 32
        num_heads = c_ // 32

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)  # optional act=FReLU(c2)

        init_values = 0.01  # or smaller
        self.gamma = nn.Parameter(init_values * torch.ones((c2)), requires_grad=True) if a2 and residual else None

        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, num_heads, mlp_ratio, area) for _ in range(2))) if a2 else C3k(c_, c_, 2,
                                                                                                      shortcut, g) for _
            in range(n)
        )

    def forward(self, x):
        """Forward pass through R-ELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        if self.gamma is not None:
            return x + self.gamma.view(1, -1, 1, 1) * self.cv2(torch.cat(y, 1))
        return self.cv2(torch.cat(y, 1))
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x): return self.relu(x + 3) / 6


class CoordAtt(nn.Module):
    def __init__(self, channels, reduction=32):
        super().__init__()
        # 水平/垂直方向的特征池化
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 输出形状 [H,1]
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 输出形状 [1,W]

        # 中间层通道数计算（至少保留8通道）
        mid_channels = max(channels // reduction, 8)

        # 特征转换层
        self.conv1 = nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = h_sigmoid()

        # 注意力生成层
        self.conv_h = nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        identity = x  # 保留原始特征

        # 步骤1：坐标信息嵌入
        n, c, h, w = x.size()
        x_h = self.pool_h(x)  # 水平池化 → [n,c,h,1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # 垂直池化 → [n,c,w,1]

        # 步骤2：生成注意力图
        y = torch.cat([x_h, x_w], dim=2)  # 拼接 → [n,c,h+w,1]
        y = self.conv1(y)  # 1x1卷积压缩通道
        y = self.bn1(y)  # 批量归一化
        y = self.act(y)  # 激活函数

        # 分离水平和垂直注意力
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # 恢复垂直维度

        # 步骤3：特征重校准
        att_h = self.conv_h(x_h).sigmoid()  # 水平注意力图
        att_w = self.conv_w(x_w).sigmoid()  # 垂直注意力图

        return identity * att_w * att_h  # 应用注意力


class Conv(nn.Module):
    # Standard convolution with batch normalization and activation
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


def autopad(k, p=None):
    # Calculate padding based on kernel size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class SPPFCSPC(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) with CSP (Cross Stage Partial Networks)
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=5):
        super().__init__()
        hidden_channels = int(2 * c2 * e)  # Hidden channels
        self.conv1 = Conv(c1, hidden_channels, 1, 1)
        self.conv2 = Conv(c1, hidden_channels, 1, 1)
        self.conv3 = Conv(hidden_channels, hidden_channels, 3, 1)
        self.conv4 = Conv(hidden_channels, hidden_channels, 1, 1)
        self.max_pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.conv5 = Conv(4 * hidden_channels, hidden_channels, 1, 1)
        self.conv6 = Conv(hidden_channels, hidden_channels, 3, 1)
        self.conv7 = Conv(2 * hidden_channels, c2, 1, 1)

    def forward(self, x):
        x1 = self.conv4(self.conv3(self.conv1(x)))  # First branch
        x2 = self.max_pool(x1)
        x3 = self.max_pool(x2)
        x4 = self.max_pool(x3)
        y1 = self.conv6(self.conv5(torch.cat((x1, x2, x3, x4), 1)))  # Concatenate and process
        y2 = self.conv2(x)  # Second branch
        return self.conv7(torch.cat((y1, y2), dim=1))  # Final concatenation and output


class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))