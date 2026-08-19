import torch
import torch.nn as nn

from damg_network.feature_transformer import NormDownsample


class ConvAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=padding, bias=False),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.block(x)


class SeparableConvAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size, stride=1, padding=padding, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.block(x)


def build_conv_block(in_ch, out_ch, kernel_size=3, block_type='standard'):
    if block_type == 'standard':
        return ConvAct(in_ch, out_ch, kernel_size=kernel_size)
    if block_type == 'separable':
        return SeparableConvAct(in_ch, out_ch, kernel_size=kernel_size)
    raise ValueError(f'Unsupported block_type: {block_type}')


class PriorPyramidEncoder(nn.Module):
    def __init__(self, in_channels, channels, norm=False):
        super().__init__()
        ch1, ch2, ch3, ch4 = channels
        self.stem = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels, ch1, 3, stride=1, padding=0, bias=False),
            nn.PReLU(),
        )
        self.down1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.down2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.down3 = NormDownsample(ch3, ch4, use_norm=norm)

    def forward(self, x):
        full = self.stem(x)
        level1 = self.down1(full)
        level2 = self.down2(level1)
        level3 = self.down3(level2)
        return {
            "full": full,
            "pyramid": [level1, level2, level3],
        }


class ConfidenceEstimator(nn.Module):
    def __init__(self, in_channels, hidden_channels, block_type='standard'):
        super().__init__()
        self.net = nn.Sequential(
            build_conv_block(in_channels, hidden_channels, block_type=block_type),
            build_conv_block(hidden_channels, hidden_channels, block_type=block_type),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class ResidualRefinementBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.act = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        residual = self.conv2(self.act(self.conv1(x)))
        return x + residual


class SeparableResidualRefinementBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.depthwise1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False)
        self.pointwise1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.act = nn.PReLU()
        self.depthwise2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False)
        self.pointwise2 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        residual = self.pointwise2(self.depthwise2(self.act(self.pointwise1(self.depthwise1(x)))))
        return x + residual


def build_residual_block(channels, block_type='standard'):
    if block_type == 'standard':
        return ResidualRefinementBlock(channels)
    if block_type == 'separable':
        return SeparableResidualRefinementBlock(channels)
    raise ValueError(f'Unsupported block_type: {block_type}')


class ReconstructionHead(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=3, block_type='standard'):
        super().__init__()
        self.net = nn.Sequential(
            build_conv_block(in_channels, hidden_channels, block_type=block_type),
            build_conv_block(hidden_channels, hidden_channels, block_type=block_type),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class RefinementHead(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=3, num_blocks=3, block_type='standard'):
        super().__init__()
        self.stem = build_conv_block(in_channels, hidden_channels, block_type=block_type)
        self.body = nn.Sequential(*[
            build_residual_block(hidden_channels, block_type=block_type)
            for _ in range(max(int(num_blocks), 1))
        ])
        self.out = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        feat = self.stem(x)
        feat = self.body(feat)
        return self.out(feat)


class PriorFusionBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=None, num_blocks=2, block_type='standard'):
        super().__init__()
        out_channels = hidden_channels if out_channels is None else out_channels
        self.stem = build_conv_block(in_channels, hidden_channels, block_type=block_type)
        self.body = nn.Sequential(*[
            build_residual_block(hidden_channels, block_type=block_type)
            for _ in range(max(int(num_blocks), 1))
        ])
        self.out = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        feat = self.stem(x)
        feat = self.body(feat)
        return self.out(feat)
