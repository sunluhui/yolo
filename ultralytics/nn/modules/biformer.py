import torch
from torch import nn
from timm.models.layers import DropPath


class BiLevelRoutingAttention(nn.Module):
    def __init__(self, dim, num_heads=8, n_win=7, qk_scale=None, topk=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.n_win = n_win
        self.topk = topk
        self.qkv = nn.Linear(dim, dim * 3)
        self.scale = qk_scale or dim ** -0.5

    def forward(self, x):
        B, H, W, C = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 动态稀疏注意力计算逻辑
        # ...（此处实现BiFormer的核心逻辑，参考官方代码）

        return x


class BiFormerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = BiLevelRoutingAttention(dim, num_heads=num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        # 输入x形状: [B, C, H, W]
        B, C, H, W = x.shape

        # 第一次残差连接
        x_norm = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x_norm = self.norm1(x_norm)
        attn_out = self.attn(x_norm)  # [B, H, W, C]
        attn_out = attn_out.permute(0, 3, 1, 2)  # 恢复为 [B, C, H, W]
        x = x + self.drop_path(attn_out)

        # 第二次残差连接
        x_norm = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x_norm = self.norm2(x_norm)
        mlp_out = self.mlp(x_norm)  # [B, H, W, C]
        mlp_out = mlp_out.permute(0, 3, 1, 2)  # 恢复为 [B, C, H, W]
        x = x + self.drop_path(mlp_out)

        return x