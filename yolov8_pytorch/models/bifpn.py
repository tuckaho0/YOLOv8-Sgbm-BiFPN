import torch
import torch.nn as nn
import torch.nn.functional as F


class BiFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels, num_outs, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.num_outs = num_outs

        # 横向连接卷积（调整通道数）
        self.lateral_convs = nn.ModuleList()
        for in_ch in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_ch, out_channels, 1))

        # 上采样与下采样卷积
        self.upsample_convs = nn.ModuleList()
        self.downsample_convs = nn.ModuleList()
        for _ in range(num_outs - 1):
            self.upsample_convs.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1))
            self.downsample_convs.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1))

        # 权重融合层
        self.weights = nn.Parameter(torch.ones(2, num_outs - 1))

    def forward(self, features):
        # 特征预处理（调整通道数）
        features = [lateral_conv(feature) for lateral_conv, feature in zip(self.lateral_convs, features)]

        # 双向特征融合
        for _ in range(self.num_layers):
            # 自顶向下路径
            for i in range(self.num_outs - 1, 0, -1):
                upsampled = F.interpolate(features[i], scale_factor=2, mode='nearest')
                features[i - 1] += upsampled * self.weights[0, i - 1]

            # 自底向上路径
            for i in range(self.num_outs - 1):
                downsampled = F.max_pool2d(features[i], 2)
                features[i + 1] += downsampled * self.weights[1, i]

        return features