from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_channels, out_channels),
            ConvBNAct(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.block(x)


class MobileNetV3Teacher(nn.Module):
    def __init__(self, encoder_name: str, use_pretrained_encoder: bool) -> None:
        super().__init__()
        if encoder_name != "mobilenet_v3_small":
            raise ValueError(f"Unsupported encoder: {encoder_name}")

        weights = models.MobileNet_V3_Small_Weights.DEFAULT if use_pretrained_encoder else None
        backbone = models.mobilenet_v3_small(weights=weights)
        self.features = backbone.features
        self.feature_indices = (2, 7, 10)
        self.bottleneck_index = 12
        self.out_channels = (24, 48, 96)
        self.in_channels = 576

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        collected_features: list[torch.Tensor] = []

        for index, layer in enumerate(self.features):
            x = layer(x)
            if index in self.feature_indices:
                collected_features.append(x)
            if index == self.bottleneck_index:
                break

        return x, tuple(collected_features)  # type: ignore[return-value]


class CompactBottleneck(nn.Module):
    def __init__(self, in_channels: int, bottleneck_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            ConvBNAct(in_channels, bottleneck_channels, kernel_size=1),
            ConvBNAct(bottleneck_channels, bottleneck_channels),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class FeatureDecoder(nn.Module):
    def __init__(self, bottleneck_channels: int) -> None:
        super().__init__()
        self.level3 = nn.Sequential(
            ConvBNAct(bottleneck_channels, 96),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
        )
        self.level2 = UpsampleBlock(96, 48)
        self.level1 = UpsampleBlock(48, 24)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_4x4 = self.level3(x)
        feature_8x8 = self.level2(feature_4x4)
        feature_16x16 = self.level1(feature_8x8)
        return feature_16x16, feature_8x8, feature_4x4


class ImageDecoder(nn.Module):
    def __init__(self, bottleneck_channels: int) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            UpsampleBlock(bottleneck_channels, 128),
            UpsampleBlock(128, 64),
            UpsampleBlock(64, 32),
            UpsampleBlock(32, 16),
            UpsampleBlock(16, 16),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


@dataclass(slots=True)
class ReverseDistillationOutput:
    teacher_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    reconstructed_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    reconstructed_image: torch.Tensor | None


class ReverseDistillationLiteModel(nn.Module):
    def __init__(
        self,
        encoder_name: str = "mobilenet_v3_small",
        bottleneck_channels: int = 128,
        use_pretrained_encoder: bool = True,
        use_image_decoder: bool = True,
    ) -> None:
        super().__init__()
        self.teacher = MobileNetV3Teacher(
            encoder_name=encoder_name,
            use_pretrained_encoder=use_pretrained_encoder,
        )
        self.bottleneck = CompactBottleneck(
            in_channels=self.teacher.in_channels,
            bottleneck_channels=bottleneck_channels,
        )
        self.feature_decoder = FeatureDecoder(bottleneck_channels=bottleneck_channels)
        self.image_decoder = ImageDecoder(bottleneck_channels) if use_image_decoder else None
        self._encoder_trainable = False
        self.set_encoder_trainable(False)

    def set_encoder_trainable(self, trainable: bool) -> None:
        self._encoder_trainable = trainable
        for parameter in self.teacher.parameters():
            parameter.requires_grad = trainable

        if trainable:
            self.teacher.train()
        else:
            self.teacher.eval()

    def forward(self, x: torch.Tensor) -> ReverseDistillationOutput:
        if self._encoder_trainable:
            latent, teacher_features = self.teacher(x)
            teacher_features = tuple(feature.detach() for feature in teacher_features)
        else:
            self.teacher.eval()
            with torch.no_grad():
                latent, teacher_features = self.teacher(x)

        bottleneck = self.bottleneck(latent)
        reconstructed_features = self.feature_decoder(bottleneck)
        reconstructed_image = None if self.image_decoder is None else self.image_decoder(bottleneck)

        return ReverseDistillationOutput(
            teacher_features=teacher_features,
            reconstructed_features=reconstructed_features,
            reconstructed_image=reconstructed_image,
        )


def compute_reverse_distillation_loss(
    output: ReverseDistillationOutput,
    reconstruction_target: torch.Tensor,
    feature_loss_weights: tuple[float, float, float],
    pixel_loss_weight: float,
) -> dict[str, torch.Tensor]:
    feature_losses: list[torch.Tensor] = []
    weighted_feature_loss = torch.zeros((), device=reconstruction_target.device)

    for weight, reconstructed_feature, teacher_feature in zip(
        feature_loss_weights,
        output.reconstructed_features,
        output.teacher_features,
    ):
        current_loss = F.mse_loss(reconstructed_feature, teacher_feature)
        feature_losses.append(current_loss)
        weighted_feature_loss = weighted_feature_loss + weight * current_loss

    pixel_loss = torch.zeros((), device=reconstruction_target.device)
    if output.reconstructed_image is not None and pixel_loss_weight > 0.0:
        pixel_loss = F.mse_loss(output.reconstructed_image, reconstruction_target)

    total_loss = weighted_feature_loss + pixel_loss_weight * pixel_loss

    return {
        "total_loss": total_loss,
        "feature_loss": weighted_feature_loss,
        "pixel_loss": pixel_loss,
        "scale_loss_1": feature_losses[0],
        "scale_loss_2": feature_losses[1],
        "scale_loss_3": feature_losses[2],
    }


def build_feature_anomaly_map(
    output: ReverseDistillationOutput,
    image_size: int,
    scale_weights: tuple[float, float, float],
) -> torch.Tensor:
    feature_maps: list[torch.Tensor] = []

    for weight, reconstructed_feature, teacher_feature in zip(
        scale_weights,
        output.reconstructed_features,
        output.teacher_features,
    ):
        residual_map = torch.mean((reconstructed_feature - teacher_feature) ** 2, dim=1, keepdim=True)
        residual_map = F.interpolate(
            residual_map,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
        feature_maps.append(weight * residual_map)

    total_weight = max(sum(scale_weights), 1e-6)
    return torch.sum(torch.stack(feature_maps, dim=0), dim=0) / total_weight


def build_pixel_anomaly_map(
    output: ReverseDistillationOutput,
    reconstruction_target: torch.Tensor,
) -> torch.Tensor | None:
    if output.reconstructed_image is None:
        return None

    return torch.mean(torch.abs(output.reconstructed_image - reconstruction_target), dim=1, keepdim=True)


def combine_anomaly_maps(
    feature_map: torch.Tensor,
    pixel_map: torch.Tensor | None,
    feature_map_weight: float,
    pixel_map_weight: float,
) -> torch.Tensor:
    combined = feature_map_weight * feature_map
    if pixel_map is not None and pixel_map_weight > 0.0:
        combined = combined + pixel_map_weight * pixel_map
    return combined


def topk_mean_score(anomaly_map: torch.Tensor, topk_ratio: float) -> torch.Tensor:
    batch_size = anomaly_map.shape[0]
    flattened = anomaly_map.view(batch_size, -1)
    k = max(1, int(flattened.shape[1] * topk_ratio))
    values, _ = torch.topk(flattened, k=k, dim=1)
    return values.mean(dim=1)
