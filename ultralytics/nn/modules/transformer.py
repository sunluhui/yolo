# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Transformer modules."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.init import constant_, xavier_uniform_

from .conv import Conv
from .utils import _get_clones, inverse_sigmoid, multi_scale_deformable_attn_pytorch

__all__ = (
    "TransformerEncoderLayer",
    "TransformerLayer",
    "TransformerBlock",
    "MLPBlock",
    "LayerNorm2d",
    "AIFI",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "MSDeformAttn",
    "MLP",
    "SBSAtt",
    "FrequencyAttention",
    "ChannelAggregationFFN",
    "CrossModelAtt",
    "EdgeGaussianAggregation",
    "DSAM",
    "FCM",
    "RCSSC",
    "KernelSelectiveFusionAttention",
)


class TransformerEncoderLayer(nn.Module):
    """Defines a single layer of the transformer encoder."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
        """Initialize the TransformerEncoderLayer with specified parameters."""
        super().__init__()
        from ...utils.torch_utils import TORCH_1_9

        if not TORCH_1_9:
            raise ModuleNotFoundError(
                "TransformerEncoderLayer() requires torch>=1.9 to use nn.MultiheadAttention(batch_first=True)."
            )
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = act
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor, pos=None):
        """Add position embeddings to the tensor if provided."""
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with post-normalization."""
        q = k = self.with_pos_embed(src, pos)
        src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with pre-normalization."""
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.ma(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
        return src + self.dropout2(src2)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Forward propagates the input through the encoder module."""
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class AIFI(TransformerEncoderLayer):
    """Defines the AIFI transformer layer."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
        """Initialize the AIFI instance with specified parameters."""
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)

    def forward(self, x):
        """Forward pass for the AIFI transformer layer."""
        c, h, w = x.shape[1:]
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c)
        # Flatten [B, C, H, W] to [B, HxW, C]
        x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed.to(device=x.device, dtype=x.dtype))
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        """Builds 2D sine-cosine position embedding."""
        assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]


class TransformerLayer(nn.Module):
    """Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)."""

    def __init__(self, c, num_heads):
        """Initializes a self-attention mechanism using linear transformations and multi-head attention."""
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        """Apply a transformer block to the input x and return the output."""
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        return self.fc2(self.fc1(x)) + x


class TransformerBlock(nn.Module):
    """Vision Transformer https://arxiv.org/abs/2010.11929."""

    def __init__(self, c1, c2, num_heads, num_layers):
        """Initialize a Transformer module with position embedding and specified number of heads and layers."""
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        """Forward propagates the input through the bottleneck module."""
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class MLPBlock(nn.Module):
    """Implements a single block of a multi-layer perceptron."""

    def __init__(self, embedding_dim, mlp_dim, act=nn.GELU):
        """Initialize the MLPBlock with specified embedding dimension, MLP dimension, and activation function."""
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the MLPBlock."""
        return self.lin2(self.act(self.lin1(x)))


class MLP(nn.Module):
    """Implements a simple multi-layer perceptron (also called FFN)."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act=nn.ReLU, sigmoid=False):
        """Initialize the MLP with specified input, hidden, output dimensions and number of layers."""
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.sigmoid = sigmoid
        self.act = act()

    def forward(self, x):
        """Forward pass for the entire MLP."""
        for i, layer in enumerate(self.layers):
            x = getattr(self, "act", nn.ReLU())(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x.sigmoid() if getattr(self, "sigmoid", False) else x


class LayerNorm2d(nn.Module):
    """
    2D Layer Normalization module inspired by Detectron2 and ConvNeXt implementations.

    Original implementations in
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py
    and
    https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py.
    """

    def __init__(self, num_channels, eps=1e-6):
        """Initialize LayerNorm2d with the given parameters."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        """Perform forward pass for 2D layer normalization."""
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MSDeformAttn(nn.Module):
    """
    Multiscale Deformable Attention Module based on Deformable-DETR and PaddleDetection implementations.

    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
    """

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """Initialize MSDeformAttn with the given parameters."""
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, but got {d_model} and {n_heads}")
        _d_per_head = d_model // n_heads
        # Better to set _d_per_head to a power of 2 which is more efficient in a CUDA implementation
        assert _d_per_head * n_heads == d_model, "`d_model` must be divisible by `n_heads`"

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        """Reset module parameters."""
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(self, query, refer_bbox, value, value_shapes, value_mask=None):
        """
        Perform forward pass for multiscale deformable attention.

        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py

        Args:
            query (torch.Tensor): [bs, query_length, C]
            refer_bbox (torch.Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (torch.Tensor): [bs, value_length, C]
            value_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, len_q = query.shape[:2]
        len_v = value.shape[1]
        assert sum(s[0] * s[1] for s in value_shapes) == len_v

        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
        value = value.view(bs, len_v, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(bs, len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(bs, len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(bs, len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        num_points = refer_bbox.shape[-1]
        if num_points == 2:
            offset_normalizer = torch.as_tensor(value_shapes, dtype=query.dtype, device=query.device).flip(-1)
            add = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations = refer_bbox[:, :, None, :, None, :] + add
        elif num_points == 4:
            add = sampling_offsets / self.n_points * refer_bbox[:, :, None, :, None, 2:] * 0.5
            sampling_locations = refer_bbox[:, :, None, :, None, :2] + add
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {num_points}.")
        output = multi_scale_deformable_attn_pytorch(value, value_shapes, sampling_locations, attention_weights)
        return self.output_proj(output)


class DeformableTransformerDecoderLayer(nn.Module):
    """
    Deformable Transformer Decoder Layer inspired by PaddleDetection and Deformable-DETR implementations.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_transformer.py
    """

    def __init__(self, d_model=256, n_heads=8, d_ffn=1024, dropout=0.0, act=nn.ReLU(), n_levels=4, n_points=4):
        """Initialize the DeformableTransformerDecoderLayer with the given parameters."""
        super().__init__()

        # Self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.act = act
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        """Add positional embeddings to the input tensor, if provided."""
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        """Perform forward pass through the Feed-Forward Network part of the layer."""
        tgt2 = self.linear2(self.dropout3(self.act(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        return self.norm3(tgt)

    def forward(self, embed, refer_bbox, feats, shapes, padding_mask=None, attn_mask=None, query_pos=None):
        """Perform the forward pass through the entire decoder layer."""
        # Self attention
        q = k = self.with_pos_embed(embed, query_pos)
        tgt = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), embed.transpose(0, 1), attn_mask=attn_mask)[
            0
        ].transpose(0, 1)
        embed = embed + self.dropout1(tgt)
        embed = self.norm1(embed)

        # Cross attention
        tgt = self.cross_attn(
            self.with_pos_embed(embed, query_pos), refer_bbox.unsqueeze(2), feats, shapes, padding_mask
        )
        embed = embed + self.dropout2(tgt)
        embed = self.norm2(embed)

        # FFN
        return self.forward_ffn(embed)


class DeformableTransformerDecoder(nn.Module):
    """
    Implementation of Deformable Transformer Decoder based on PaddleDetection.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    """

    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        """Initialize the DeformableTransformerDecoder with the given parameters."""
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
        self,
        embed,  # decoder embeddings
        refer_bbox,  # anchor
        feats,  # image features
        shapes,  # feature shapes
        bbox_head,
        score_head,
        pos_mlp,
        attn_mask=None,
        padding_mask=None,
    ):
        """Perform the forward pass through the entire decoder."""
        output = embed
        dec_bboxes = []
        dec_cls = []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()
        for i, layer in enumerate(self.layers):
            output = layer(output, refer_bbox, feats, shapes, padding_mask, attn_mask, pos_mlp(refer_bbox))

            bbox = bbox_head[i](output)
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))

            if self.training:
                dec_cls.append(score_head[i](output))
                if i == 0:
                    dec_bboxes.append(refined_bbox)
                else:
                    dec_bboxes.append(torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox)))
            elif i == self.eval_idx:
                dec_cls.append(score_head[i](output))
                dec_bboxes.append(refined_bbox)
                break

            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack(dec_bboxes), torch.stack(dec_cls)


class SBSAtt(nn.Module):
    def __init__(self, dim, num_heads=4, bias=True):
        super(SBSAtt, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.factor = 2
        self.idx_dict = {}
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw // factor + 1) * factor - hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad

    def unpad(self, x, t_pad):
        hw = x.shape[-1]
        return x[..., t_pad[0]:hw - t_pad[1]]

    def comp2real(self, x):
        b, _, h, w = x.shape
        return torch.cat([x.real, x.imag], 1)

    #        return torch.stack([x.real, x.imag], 2).view(b,-1,h,w)
    def real2comp(self, x):
        xr, xi = x.chunk(2, dim=1)
        return torch.complex(xr, xi)

    def softmax_1(self, x, dim=-1):
        logit = x.exp()
        logit = logit / (logit.sum(dim, keepdim=True) + 1)
        return logit

    def get_idx_map(self, h, w):
        l1_u = torch.arange(h // 2).view(1, 1, -1, 1)
        l2_u = torch.arange(w).view(1, 1, 1, -1)
        half_map_u = l1_u @ l2_u
        l1_d = torch.arange(h - h // 2).flip(0).view(1, 1, -1, 1)
        l2_d = torch.arange(w).view(1, 1, 1, -1)
        half_map_d = l1_d @ l2_d
        return torch.cat([half_map_u, half_map_d], 2).view(1, 1, -1).argsort(-1)

    def get_idx(self, x):
        h, w = x.shape[-2:]
        if (h, w) in self.idx_dict:
            return self.idx_dict[(h, w)]
        idx_map = self.get_idx_map(h, w).to(x.device).detach()
        self.idx_dict[(h, w)] = idx_map
        return idx_map

    def attn(self, qkv):
        h = qkv.shape[2]
        q, k, v = qkv.chunk(3, dim=1)

        q, pad_w, idx = self.fft(q)
        q, pad = self.pad(q, self.factor)
        k, pad_w, _ = self.fft(k)
        k, pad = self.pad(k, self.factor)
        v, pad_w, _ = self.fft(v)
        v, pad = self.pad(v, self.factor)

        q = rearrange(q, 'b (head c) (factor hw) -> b head (c factor) hw', head=self.num_heads, factor=self.factor)
        k = rearrange(k, 'b (head c) (factor hw) -> b head (c factor) hw', head=self.num_heads, factor=self.factor)
        v = rearrange(v, 'b (head c) (factor hw) -> b head (c factor) hw', head=self.num_heads, factor=self.factor)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax_1(attn, dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head (c factor) hw -> b (head c) (factor hw)', head=self.num_heads, factor=self.factor)
        out = self.unpad(out, pad)
        out = self.ifft(out, pad_w, idx, h)
        return out

    def fft(self, x):
        x, pad = self.pad(x, 2)
        x = torch.fft.rfft2(x.float(), norm="ortho")
        x = self.comp2real(x)
        idx = self.get_idx(x)
        b, c = x.shape[:2]
        x = x.contiguous().view(b, c, -1)
        x = torch.gather(x, 2, index=idx.repeat(b, c, 1).to(x.device))  # b, 6c, h*(w//2+1)#x = torch.gather(x, 2, index=idx.repeat(b, c, 1))  # b, 6c, h*(w//2+1)
        return x, pad, idx

    def ifft(self, x, pad, idx, h):
        b, c = x.shape[:2]
        x = torch.scatter(x, 2, idx.repeat(b, c, 1), x)
        x = x.view(b, c, h, -1)
        x = self.real2comp(x)
        x = torch.fft.irfft2(x, norm='ortho')  # .abs()
        x = self.unpad(x, pad)
        return x

    def forward(self, x):
        qkv = self.qkv_dwconv(self.qkv(x))

        out = self.attn(qkv)
        out = self.project_out(out)
        return out


from einops import rearrange, repeat
from torch.nn import Softmax


def custom_complex_normalization(input_tensor, dim=-1):
    real_part = input_tensor.real
    imag_part = input_tensor.imag
    norm_real = F.softmax(real_part, dim=dim)
    norm_imag = F.softmax(imag_part, dim=dim)

    normalized_tensor = torch.complex(norm_real, norm_imag)

    return normalized_tensor


class FrequencyAttention(nn.Module):
    def __init__(self, in_dim):
        super(FrequencyAttention, self).__init__()

        down_dim = in_dim // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.ReLU(True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=3, padding=3), nn.BatchNorm2d(down_dim), nn.ReLU(True)
        )
        self.query_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.temperature = nn.Parameter(torch.ones(8, 1, 1))

        self.weight = nn.Sequential(
            nn.Conv2d(down_dim, down_dim // 16, 1, bias=True),
            nn.BatchNorm2d(down_dim // 16),
            nn.ReLU(True),
            nn.Conv2d(down_dim // 16, down_dim, 1, bias=True),
            nn.Sigmoid())

        self.softmax = Softmax(dim=-1)
        self.norm = nn.BatchNorm2d(down_dim)
        self.relu = nn.ReLU(True)
        self.num_heads = 8

    def forward(self, x):
        conv2 = self.conv2(x)
        b, c, h, w = conv2.shape

        q_f_2 = torch.fft.fft2(conv2.float())
        k_f_2 = torch.fft.fft2(conv2.float())
        v_f_2 = torch.fft.fft2(conv2.float())
        tepqkv = torch.fft.fft2(conv2.float())

        q_f_2 = rearrange(q_f_2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_f_2 = rearrange(k_f_2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_f_2 = rearrange(v_f_2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_f_2 = torch.nn.functional.normalize(q_f_2, dim=-1)
        k_f_2 = torch.nn.functional.normalize(k_f_2, dim=-1)
        attn_f_2 = (q_f_2 @ k_f_2.transpose(-2, -1)) * self.temperature
        attn_f_2 = custom_complex_normalization(attn_f_2, dim=-1)
        out_f_2 = torch.abs(torch.fft.ifft2(attn_f_2 @ v_f_2))
        out_f_2 = rearrange(out_f_2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_f_l_2 = torch.abs(torch.fft.ifft2(self.weight(tepqkv.real) * tepqkv))
        out_2 = torch.cat((out_f_2, out_f_l_2), 1)

        F_2 = torch.add(out_2, x)

        return F_2


def build_act_layer(act_type):
    # Build activation layer
    if act_type is None:
        return nn.Identity()
    assert act_type in ['GELU', 'ReLU', 'SiLU']
    if act_type == 'SiLU':
        return nn.SiLU()
    elif act_type == 'ReLU':
        return nn.ReLU()
    else:
        return nn.GELU()


class ElementScale(nn.Module):
    # A learnable element-wise scaler.

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale


class ChannelAggregationFFN(nn.Module):
    """An implementation of FFN with Channel Aggregation.
    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        kernel_size (int): The depth-wise conv kernel size as the
            depth-wise convolution. Defaults to 3.
        act_type (str): The type of activation. Defaults to 'GELU'.
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
    """

    def __init__(self,
                 embed_dims,
                 kernel_size=3,
                 act_type='GELU',
                 ffn_drop=0.):
        super(ChannelAggregationFFN, self).__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = int(embed_dims * 4)

        self.fc1 = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=self.feedforward_channels,
            kernel_size=1)
        self.dwconv = nn.Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=self.feedforward_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=self.feedforward_channels)
        self.act = build_act_layer(act_type)
        self.fc2 = nn.Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=embed_dims,
            kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)

        self.decompose = nn.Conv2d(
            in_channels=self.feedforward_channels,  # C -> 1
            out_channels=1, kernel_size=1,
        )
        self.sigma = ElementScale(
            self.feedforward_channels, init_value=1e-5, requires_grad=True)
        self.decompose_act = build_act_layer(act_type)

    def feat_decompose(self, x):
        # x_d: [B, C, H, W] -> [B, 1, H, W]
        x = x + self.sigma(x - self.decompose_act(self.decompose(x)))
        return x

    def forward(self, x):
        # proj 1
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        # proj 2
        x = self.feat_decompose(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CrossModelAtt(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        img_feat, text_feat = x
        B, C, H, W = img_feat.shape
        q = img_feat.view(B, C, -1)
        k = text_feat.view(B, C, -1).permute(0, 2, 1)
        attention_map = torch.bmm(q, k)  # [B, C, C]
        attention_map = self.softmax(attention_map)
        v = text_feat.view(B, C, -1)
        attention_info = torch.bmm(attention_map, v)
        attention_info = attention_info.view(B, C, H, W)
        output = self.gamma * attention_info + img_feat

        return output


try:
    from mmcv.cnn import build_norm_layer
except:
    pass


class Conv_Extra(nn.Module):
    def __init__(self, channel, norm_layer, act_layer):
        super(Conv_Extra, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(channel, 64, 1),
                                   build_norm_layer(norm_layer, 64)[1],
                                   act_layer(),
                                   nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1, bias=False),
                                   build_norm_layer(norm_layer, 64)[1],
                                   act_layer(),
                                   nn.Conv2d(64, channel, 1),
                                   build_norm_layer(norm_layer, channel)[1])

    def forward(self, x):
        out = self.block(x)
        return out


class EdgeGaussianAggregation(nn.Module):
    def __init__(self, dim, size=3, sigma=1, norm_layer=dict(type='BN', requires_grad=True), act_layer=nn.ReLU,
                 feature_extra=True):
        super().__init__()
        self.feature_extra = feature_extra
        gaussian = self.gaussian_kernel(size, sigma)
        gaussian = nn.Parameter(data=gaussian, requires_grad=False).clone()
        self.gaussian = nn.Conv2d(dim, dim, kernel_size=size, stride=1, padding=int(size // 2), groups=dim, bias=False)
        self.gaussian.weight.data = gaussian.repeat(dim, 1, 1, 1)
        self.norm = build_norm_layer(norm_layer, dim)[1]
        self.act = act_layer()
        if feature_extra == True:
            self.conv_extra = Conv_Extra(dim, norm_layer, act_layer)

    def forward(self, x):
        edges_o = self.gaussian(x)
        gaussian = self.act(self.norm(edges_o))
        if self.feature_extra == True:
            out = self.conv_extra(x + gaussian)
        else:
            out = gaussian
        return out

    def gaussian_kernel(self, size: int, sigma: float):
        kernel = torch.FloatTensor([
            [(1 / (2 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
             for x in range(-size // 2 + 1, size // 2 + 1)]
            for y in range(-size // 2 + 1, size // 2 + 1)
        ]).unsqueeze(0).unsqueeze(0)
        return kernel / kernel.sum()


class Pred_Layer(nn.Module):
    def __init__(self, in_c=256):
        super(Pred_Layer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_c, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.outlayer = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0), )

    def forward(self, x):
        x = self.enlayer(x)
        x1 = self.outlayer(x)
        return x, x1


class ASPP(nn.Module):
    def __init__(self, in_c):
        super(ASPP, self).__init__()

        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_c, 256, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_c, 256, 3, 1, padding=3, dilation=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_c, 256, 3, 1, padding=5, dilation=5),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(in_c, 256, 3, 1, padding=7, dilation=7),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x


# Dual-Stream Attention Module
class DSAM(nn.Module):
    def __init__(self, in_c):
        super(DSAM, self).__init__()
        self.ff_conv = ASPP(in_c)
        self.bf_conv = ASPP(in_c)
        self.rgbd_pred_layer = Pred_Layer(256 * 8)
        self.conv = nn.Conv2d(in_c, 1, 1, 1, 0, )

    def forward(self, x):
        pred = self.conv(x)
        feat = x
        [_, _, H, W] = feat.size()
        pred = torch.sigmoid(
            F.interpolate(pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))

        ff_feat = self.ff_conv(feat * pred)
        bf_feat = self.bf_conv(feat * (1 - pred))
        enhanced_feat, new_pred = self.rgbd_pred_layer(torch.cat((ff_feat, bf_feat), 1))
        # return enhanced_feat, new_pred
        return enhanced_feat


from timm.models.layers import trunc_normal_


class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 6, self.dim * 6 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 6 // reduction, self.dim * 2),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        std = torch.std(x, dim=(2, 3), keepdim=True).view(B, self.dim * 2)
        max = self.max_pool(x).view(B, self.dim * 2)
        y = torch.cat((avg, std, max), dim=1)  # B 6C
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)  # 2 B C 1 1
        return channel_weights


class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)  # B 2C H W
        spatial_weights = self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)  # 2 B 1 H W
        return spatial_weights


# 先空间校正再通道校正
class FCM(nn.Module):
    def __init__(self, dim, reduction=1, eps=1e-8):
        super(FCM, self).__init__()
        # 自定义可训练权重参数
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)

        self.apply(self._init_weights)

    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x1, x2 = x
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)

        spatial_weights = self.spatial_weights(x1, x2)
        x1_1 = x1 + fuse_weights[0] * spatial_weights[1] * x2
        x2_1 = x2 + fuse_weights[0] * spatial_weights[0] * x1

        channel_weights = self.channel_weights(x1_1, x2_1)

        main_out = x1_1 + fuse_weights[1] * channel_weights[1] * x2_1
        aux_out = x2_1 + fuse_weights[1] * channel_weights[0] * x1_1
        # return main_out, aux_out
        return torch.cat([main_out, aux_out], dim=1)


class ChannelPool(nn.Module):
    def forward(self, x):
        # 将maxpooling 与 global average pooling 结果拼接在一起
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class Basic(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, relu=True, bn=True, bias=False):
        super(Basic, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.LeakyReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        self.avgPoolW = nn.AdaptiveAvgPool2d((1, None))
        self.maxPoolW = nn.AdaptiveMaxPool2d((1, None))

        self.conv_1x1 = nn.Conv2d(in_channels=2 * channel, out_channels=2 * channel, kernel_size=1, padding=0, stride=1,
                                  bias=False)
        self.bn = nn.BatchNorm2d(2 * channel, eps=1e-5, momentum=0.01, affine=True)
        self.Relu = nn.LeakyReLU()

        self.F_h = nn.Sequential(  # 激发操作
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.BatchNorm2d(channel // reduction, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
        )
        self.F_w = nn.Sequential(  # 激发操作
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.BatchNorm2d(channel // reduction, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        N, C, H, W = x.size()
        res = x
        x_cat = torch.cat([self.avgPoolW(x), self.maxPoolW(x)], 1)
        x = self.Relu(self.bn(self.conv_1x1(x_cat)))
        x_1, x_2 = x.split(C, 1)

        x_1 = self.F_h(x_1)
        x_2 = self.F_w(x_2)
        s_h = self.sigmoid(x_1)
        s_w = self.sigmoid(x_2)

        out = res * s_h.expand_as(res) * s_w.expand_as(res)

        return out


class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=3):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = Basic(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bn=False, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class RCSSC(nn.Module):
    def __init__(self, n_feat, reduction=16):
        super(RCSSC, self).__init__()
        pooling_r = 4
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, stride=1, bias=True),
            nn.LeakyReLU(),
        )
        self.SC = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
            nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(n_feat)
        )
        self.SA = spatial_attn_layer()  ## Spatial Attention
        self.CA = CALayer(n_feat, reduction)  ## Channel Attention

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(n_feat * 2, n_feat, kernel_size=1),
            nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, stride=1, bias=True)
        )
        self.ReLU = nn.LeakyReLU()
        self.tail = nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1)

    def forward(self, x):
        res = x
        x = self.head(x)
        sa_branch = self.SA(x)
        ca_branch = self.CA(x)
        x1 = torch.cat([sa_branch, ca_branch], dim=1)  # 拼接
        x1 = self.conv1x1(x1)
        x2 = torch.sigmoid(
            torch.add(x, F.interpolate(self.SC(x), x.size()[2:])))
        out = torch.mul(x1, x2)
        out = self.tail(out)
        out = out + res
        out = self.ReLU(out)
        return out


class KernelSelectiveFusionAttention(nn.Module):
    def __init__(self, dim, r=16, L=32):
        super().__init__()
        d = max(dim // r, L)
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=4, groups=dim, dilation=2)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_maxpool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Conv2d(d, dim, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        dim = x.size(1)
        attn1 = self.conv0(x)  # conv_3*3
        attn2 = self.conv_spatial(attn1)  # conv_3*3 -> conv_5*5

        attn1 = self.conv1(attn1)  # b, dim/2, h, w
        attn2 = self.conv2(attn2)  # b, dim/2, h, w

        attn = torch.cat([attn1, attn2], dim=1)  # b,c,h,w
        avg_attn = torch.mean(attn, dim=1, keepdim=True)  # b,1,h,w
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)  # b,1,h,w
        agg = torch.cat([avg_attn, max_attn], dim=1)  # spa b,2,h,w

        ch_attn1 = self.global_pool(attn)  # b,dim,1, 1
        z = self.fc1(ch_attn1)
        a_b = self.fc2(z)
        a_b = a_b.reshape(batch_size, 2, dim // 2, -1)
        a_b = self.softmax(a_b)

        a1, a2 = a_b.chunk(2, dim=1)
        a1 = a1.reshape(batch_size, dim // 2, 1, 1)
        a2 = a2.reshape(batch_size, dim // 2, 1, 1)

        w1 = a1 * agg[:, 0, :, :].unsqueeze(1)
        w2 = a2 * agg[:, 0, :, :].unsqueeze(1)

        attn = attn1 * w1 + attn2 * w2
        attn = self.conv(attn).sigmoid()

        return x * attn


