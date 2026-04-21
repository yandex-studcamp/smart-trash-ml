from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=True,
            ),
            nn.ReLU(inplace=True),
        )


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise_1 = ConvReLU(channels, channels, kernel_size=3, stride=1, groups=channels)
        self.pointwise_1 = ConvReLU(channels, channels, kernel_size=1, stride=1)
        self.depthwise_2 = ConvReLU(channels, channels, kernel_size=3, stride=1, groups=channels)
        self.pointwise_2 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.depthwise_1(x)
        out = self.pointwise_1(out)
        out = self.depthwise_2(out)
        out = self.pointwise_2(out)
        out = out + residual
        return self.activation(out)


class DownsampleStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.downsample = ConvReLU(in_channels, out_channels, kernel_size=3, stride=2)
        self.residual = DepthwiseSeparableBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        return self.residual(x)


class UpsampleStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.project = ConvReLU(in_channels, out_channels, kernel_size=3, stride=1)
        self.residual = DepthwiseSeparableBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.project(x)
        return self.residual(x)


class StudentAutoencoderV3(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        channels: tuple[int, int, int, int] = (24, 32, 48, 64),
        bottleneck_channels: int = 80,
    ) -> None:
        super().__init__()
        if in_channels != 1:
            raise ValueError("StudentAutoencoderV3 expects a single grayscale input channel.")

        c1, c2, c3, c4 = channels
        self.stem = nn.Sequential(
            ConvReLU(in_channels, c1, kernel_size=3, stride=1),
            DepthwiseSeparableBlock(c1),
        )
        self.encoder = nn.Sequential(
            DownsampleStage(c1, c2),
            DownsampleStage(c2, c3),
            DownsampleStage(c3, c4),
            DownsampleStage(c4, bottleneck_channels),
        )
        self.bottleneck = nn.Sequential(
            DepthwiseSeparableBlock(bottleneck_channels),
            DepthwiseSeparableBlock(bottleneck_channels),
        )
        self.decoder = nn.Sequential(
            UpsampleStage(bottleneck_channels, c4),
            UpsampleStage(c4, c3),
            UpsampleStage(c3, c2),
            UpsampleStage(c2, c1),
            DepthwiseSeparableBlock(c1),
            nn.Conv2d(c1, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.encoder(x)
        x = self.bottleneck(x)
        return self.decoder(x)
