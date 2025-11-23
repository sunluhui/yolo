import torch.nn as nn
import torch
from einops import rearrange
import math


class AKConv(nn.Module):
    def __init__(self, inc, outc, num_param, stride=1, bias=None):
        super(AKConv, self).__init__()
        self.num_param = num_param
        self.stride = stride

        # 修复1: 使用正常的1x1卷积或调整卷积参数
        self.conv = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=1, stride=1, bias=bias),  # 改为1x1卷积
            nn.BatchNorm2d(outc),
            nn.SiLU()
        )

        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        # 修复2: 正确注册钩子
        self.p_conv.register_full_backward_hook(self._set_lr)

    # 修复3: 正确的梯度钩子实现
    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        new_grad_input = tuple(g * 0.1 if g is not None else None for g in grad_input)
        new_grad_output = tuple(g * 0.1 if g is not None else None for g in grad_output)
        return new_grad_input, new_grad_output

    def forward(self, x):
        # N is num_param.
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = self.num_param  # 直接使用num_param
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([
            torch.clamp(q_lt[..., :N], 0, x.size(2) - 1),
            torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)
        ], dim=-1).long()

        q_rb = torch.cat([
            torch.clamp(q_rb[..., :N], 0, x.size(2) - 1),
            torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)
        ], dim=-1).long()

        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([
            torch.clamp(p[..., :N], 0, x.size(2) - 1),
            torch.clamp(p[..., N:], 0, x.size(3) - 1)
        ], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # resampling the features based on the modified coordinates.
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # bilinear
        x_offset = (g_lt.unsqueeze(dim=1) * x_q_lt +
                    g_rb.unsqueeze(dim=1) * x_q_rb +
                    g_lb.unsqueeze(dim=1) * x_q_lb +
                    g_rt.unsqueeze(dim=1) * x_q_rt)

        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        out = self.conv(x_offset)

        return out

    # 修复4: 改进的采样点生成（中心对称）
    def _get_p_n(self, N, dtype):
        # 生成中心对称的采样点
        if N == 1:
            p_n = torch.tensor([[0.0], [0.0]])
        else:
            # 圆形采样点分布
            angles = torch.linspace(0, 2 * math.pi, N + 1)[:-1]
            radius = 1.0
            p_n_x = radius * torch.cos(angles)
            p_n_y = radius * torch.sin(angles)
            p_n = torch.stack([p_n_x, p_n_y], dim=0)

        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),
            torch.arange(0, w * self.stride, self.stride),
            indexing='ij'  # 修复5: 使用'ij'索引
        )
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    def _get_p(self, offset, dtype):
        N = self.num_param
        h, w = offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        index = index.clamp(min=0, max=x.shape[-1] - 1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        b, c, h, w, n = x_offset.size()
        x_offset = rearrange(x_offset, 'b c h w n -> b c h (w n)')
        return x_offset