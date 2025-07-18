import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicSnakeConv(nn.Module):
    """修复的动态蛇形卷积模块"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 偏移量生成卷积
        self.conv_offset = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,  # 每个采样点有(x,y)两个偏移量
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        # 主卷积核
        self.conv_main = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,  # 下采样由偏移卷积处理
            padding=0  # 不使用padding
        )

        # 初始化偏移量生成器
        self.init_weights()

    def init_weights(self):
        """初始化偏移量生成器"""
        nn.init.constant_(self.conv_offset.weight, 0)
        nn.init.constant_(self.conv_offset.bias, 0)

        # 初始化主卷积核
        nn.init.kaiming_normal_(self.conv_main.weight, mode='fan_out', nonlinearity='relu')
        if self.conv_main.bias is not None:
            nn.init.constant_(self.conv_main.bias, 0)

    def forward(self, x):
        # 1. 生成偏移量 [B, 2*K*K, H_out, W_out]
        offset = self.conv_offset(x)
        B, _, H_out, W_out = offset.size()
        K = self.kernel_size

        # 2. 创建基础网格 [-1,1] 范围
        y_range = torch.linspace(-1, 1, H_out, dtype=x.dtype, device=x.device)
        x_range = torch.linspace(-1, 1, W_out, dtype=x.dtype, device=x.device)
        y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='ij')
        grid = torch.stack((x_grid, y_grid), dim=-1)  # [H_out, W_out, 2]
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H_out, W_out, 2]

        # 3. 处理偏移量 [B, H_out, W_out, K*K, 2]
        offset = offset.permute(0, 2, 3, 1)  # [B, H_out, W_out, 2*K*K]
        offset = offset.reshape(B, H_out, W_out, K, K, 2)

        # 4. 构建蛇形偏移 (核心修复)
        # 水平方向: 每行的偏移累积
        for i in range(1, K):
            offset[:, :, :, i, :, 0] += offset[:, :, :, i - 1, :, 0]
            offset[:, :, :, i, :, 1] = offset[:, :, :, i - 1, :, 1]  # 保持y坐标一致

        # 垂直方向: 每列的偏移累积
        for j in range(1, K):
            offset[:, :, :, :, j, 1] += offset[:, :, :, :, j - 1, 1]
            offset[:, :, :, :, j, 0] = offset[:, :, :, :, j - 1, 0]  # 保持x坐标一致

        # 5. 展开偏移并叠加到基础网格
        offset = offset.reshape(B, H_out, W_out, K * K, 2)  # [B, H_out, W_out, K*K, 2]
        grid = grid.unsqueeze(3).repeat(1, 1, 1, K * K, 1)  # [B, H_out, W_out, K*K, 2]
        grid = grid + offset

        # 6. 重塑网格为grid_sample需要的格式 [B, H_out, W_out*K*K, 2]
        grid = grid.reshape(B, H_out, W_out * K * K, 2)

        # 7. 可变形卷积采样
        x_sampled = F.grid_sample(
            x,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )

        # 8. 重塑采样结果 [B, C, H_out, W_out, K*K]
        x_sampled = x_sampled.reshape(B, -1, H_out, W_out, K * K)
        x_sampled = x_sampled.permute(0, 1, 4, 2, 3)  # [B, C, K*K, H_out, W_out]

        # 9. 应用卷积权重
        weight = self.conv_main.weight.view(1,
                                            self.conv_main.out_channels,
                                            self.conv_main.in_channels,
                                            K * K,
                                            1, 1)

        output = (x_sampled.unsqueeze(1) * weight).sum(2).sum(2)

        # 10. 添加偏置
        if self.conv_main.bias is not None:
            output += self.conv_main.bias.view(1, -1, 1, 1)

        return output

