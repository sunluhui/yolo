import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class TopkRouting(nn.Module):
    """BiFormer的核心路由机制，动态选择最相关的区域"""

    def __init__(self, dim, n_win=7, topk=4, qk_dim=None):
        super().__init__()
        self.dim = dim
        self.n_win = n_win
        self.topk = topk
        self.qk_dim = qk_dim or dim

        # 路由参数
        self.scale = nn.Parameter(torch.ones([]))
        self.router = nn.Conv2d(dim, n_win * n_win, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape

        # 生成路由分数
        routing_score = self.router(x)  # [B, n_win*n_win, H, W]
        routing_score = rearrange(routing_score, 'b (nw nh) h w -> b nw nh h w',
                                  nw=self.n_win, nh=self.n_win)

        # 聚合分数并选择topk区域
        routing_score = rearrange(routing_score, 'b nw nh h w -> b (h w) (nw nh)')
        routing_score = F.softmax(routing_score, dim=-1)

        # 选择topk区域
        _, topk_idx = torch.topk(routing_score, k=self.topk, dim=-1)

        # 生成路由掩码
        mask = torch.zeros_like(routing_score)
        mask.scatter_(-1, topk_idx, 1.0)

        return mask, routing_score


class BiLevelRoutingAttention(nn.Module):
    """BiFormer的双层路由注意力机制"""

    def __init__(self, dim, num_heads=8, n_win=7, topk=4, qk_scale=None,
                 attn_drop=0., proj_drop=0., qkv_bias=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.n_win = n_win
        self.topk = topk

        # 路由模块
        self.routing = TopkRouting(dim, n_win, topk)

        # QKV投影
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)

        # 相对位置偏置
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * n_win - 1) * (2 * n_win - 1), num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

        coords = torch.stack(torch.meshgrid(
            [torch.arange(n_win), torch.arange(n_win)]), dim=0).flatten(1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += n_win - 1
        relative_coords[:, :, 1] += n_win - 1
        relative_coords[:, :, 0] *= 2 * n_win - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x):
        B, C, H, W = x.shape
        num_windows = self.n_win * self.n_win

        # 生成路由掩码
        mask, routing_score = self.routing(x)
        mask = rearrange(mask, 'b (h w) (nw nh) -> b nw nh h w',
                         h=H // self.n_win, w=W // self.n_win)

        # 划分窗口
        x = rearrange(x, 'b c (h win_h) (w win_w) -> b (h w) (win_h win_w) c',
                      win_h=H // self.n_win, win_w=W // self.n_win)
        qkv = self.qkv(x).reshape(B, num_windows, -1, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(3, 0, 1, 4, 2, 5)  # [3, B, n_win, heads, win_size, dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # 每个 [B, n_win, heads, win_size, dim]

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 添加相对位置偏置
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            num_windows, num_windows, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0).unsqueeze(3)

        # 应用路由掩码
        mask = mask.view(B, num_windows, 1, 1, -1)
        attn = attn * mask + (1 - mask) * -1e9

        # 归一化注意力权重
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # 聚合值
        x = (attn @ v).transpose(2, 3).reshape(B, num_windows, -1, C)

        # 恢复空间维度
        x = rearrange(x, 'b (h w) (win_h win_w) c -> b c (h win_h) (w win_w)',
                      h=H // self.n_win, w=W // self.n_win,
                      win_h=H // self.n_win, win_w=W // self.n_win)

        # 最终投影
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BiFormerBlock(nn.Module):
    """完整的BiFormer模块，包含归一化和前馈网络"""

    def __init__(self, dim, num_heads=8, n_win=7, topk=4, mlp_ratio=4.,
                 qkv_bias=False, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = BiLevelRoutingAttention(
            dim, num_heads=num_heads, n_win=n_win, topk=topk,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, 1),
            act_layer(),
            nn.Dropout(drop),
            nn.Conv2d(mlp_hidden_dim, dim, 1),
            nn.Dropout(drop)
        )

        # 层缩放参数
        self.gamma1 = nn.Parameter(torch.ones(dim))
        self.gamma2 = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 保存原始输入用于残差连接
        shortcut = x

        # 注意力部分
        x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = shortcut + self.gamma1.unsqueeze(0).unsqueeze(2).unsqueeze(3) * self.attn(x)

        # MLP部分
        shortcut = x
        x = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = shortcut + self.gamma2.unsqueeze(0).unsqueeze(2).unsqueeze(3) * self.mlp(x)

        return x
