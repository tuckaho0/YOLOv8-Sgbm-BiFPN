import torch
import torch.nn as nn
import torch.nn.functional as F


class BifpnNeck(nn.Module):
    def __init__(self, in_channels, out_channels, num_outs=3):
        super().__init__()
        self.num_outs = num_outs

        # 横向连接层（统一通道数）
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels
        ])

        # 自顶向下路径
        self.td_upsample = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels),
                nn.SiLU()
            ) for _ in range(num_outs - 1)
        ])

        # 自底向上路径
        self.bu_downsample = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, 2, 1),
                nn.BatchNorm2d(out_channels),
                nn.SiLU()
            ) for _ in range(num_outs - 1)
        ])

        # 动态权重
        self.td_weights = nn.Parameter(torch.ones(num_outs - 1))
        self.bu_weights = nn.Parameter(torch.ones(num_outs - 1))

    def forward(self, features):
        # 统一通道数
        features = [lateral_conv(f) for lateral_conv, f in zip(self.lateral_convs, features)]

        # 自顶向下融合
        for i in range(self.num_outs - 1, 0, -1):
            upsampled = self.td_upsample[i - 1](features[i])
            features[i - 1] = F.relu(self.td_weights[i - 1]) * features[i - 1] + upsampled

        # 自底向上融合
        for i in range(self.num_outs - 1):
            downsampled = self.bu_downsample[i](features[i])
            features[i + 1] = F.relu(self.bu_weights[i]) * features[i + 1] + downsampled

        return features