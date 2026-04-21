from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.anomaly_reverse_distillation import MobileNetV3Teacher

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ConvReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
            ),
            nn.ReLU(inplace=True),
        )


class UpsampleConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvReLU(in_channels, out_channels, kernel_size=3, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class StudentOnlyAutoencoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        encoder_channels: tuple[int, int, int, int] = (16, 24, 32, 48),
        bottleneck_channels: int = 64,
    ) -> None:
        super().__init__()
        if in_channels != 1:
            raise ValueError("StudentOnlyAutoencoder expects a single grayscale input channel.")

        c1, c2, c3, c4 = encoder_channels
        self.encoder = nn.Sequential(
            ConvReLU(in_channels, c1, stride=1),
            ConvReLU(c1, c2, stride=2),
            ConvReLU(c2, c3, stride=2),
            ConvReLU(c3, c4, stride=2),
            ConvReLU(c4, bottleneck_channels, stride=2),
            ConvReLU(bottleneck_channels, bottleneck_channels, stride=1),
        )

        self.decoder = nn.Sequential(
            UpsampleConvBlock(bottleneck_channels, c4),
            UpsampleConvBlock(c4, c3),
            UpsampleConvBlock(c3, c2),
            UpsampleConvBlock(c2, c1),
            ConvReLU(c1, c1, stride=1),
            nn.Conv2d(c1, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        return self.decoder(latent)


def build_distillation_teacher(
    encoder_name: str = "mobilenet_v3_small",
    use_pretrained_encoder: bool = True,
) -> MobileNetV3Teacher:
    teacher = MobileNetV3Teacher(
        encoder_name=encoder_name,
        use_pretrained_encoder=use_pretrained_encoder,
    )
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher


def prepare_teacher_input(grayscale_tensor: torch.Tensor) -> torch.Tensor:
    if grayscale_tensor.ndim != 4:
        raise ValueError("Expected a 4D BCHW tensor.")
    if grayscale_tensor.shape[1] != 1:
        raise ValueError("Expected a single grayscale input channel.")

    teacher_input = grayscale_tensor.repeat(1, 3, 1, 1)
    mean = torch.as_tensor(IMAGENET_MEAN, device=teacher_input.device, dtype=teacher_input.dtype).view(1, 3, 1, 1)
    std = torch.as_tensor(IMAGENET_STD, device=teacher_input.device, dtype=teacher_input.dtype).view(1, 3, 1, 1)
    return (teacher_input - mean) / std


@dataclass(slots=True)
class StudentAutoencoderLosses:
    total_loss: torch.Tensor
    pixel_loss: torch.Tensor
    distillation_loss: torch.Tensor
    scale_loss_1: torch.Tensor
    scale_loss_2: torch.Tensor
    scale_loss_3: torch.Tensor


def compute_student_autoencoder_losses(
    reconstruction: torch.Tensor,
    reconstruction_target: torch.Tensor,
    teacher: MobileNetV3Teacher | None,
    feature_distillation_weights: tuple[float, float, float],
    pixel_loss_weight: float,
    distillation_weight: float,
) -> StudentAutoencoderLosses:
    pixel_loss = F.l1_loss(reconstruction, reconstruction_target)
    weighted_pixel_loss = pixel_loss_weight * pixel_loss

    distillation_loss = torch.zeros((), device=reconstruction.device, dtype=reconstruction.dtype)
    scale_losses = [
        torch.zeros((), device=reconstruction.device, dtype=reconstruction.dtype)
        for _ in range(3)
    ]

    if teacher is not None and distillation_weight > 0.0:
        with torch.no_grad():
            _, target_features = teacher(prepare_teacher_input(reconstruction_target))

        _, reconstructed_features = teacher(prepare_teacher_input(reconstruction))
        weighted_sum = torch.zeros((), device=reconstruction.device, dtype=reconstruction.dtype)
        total_weight = max(sum(feature_distillation_weights), 1e-6)
        for index, (weight, reconstructed_feature, target_feature) in enumerate(
            zip(feature_distillation_weights, reconstructed_features, target_features),
        ):
            current_loss = F.mse_loss(reconstructed_feature, target_feature)
            scale_losses[index] = current_loss
            weighted_sum = weighted_sum + weight * current_loss

        distillation_loss = weighted_sum / total_weight

    total_loss = weighted_pixel_loss + distillation_weight * distillation_loss
    return StudentAutoencoderLosses(
        total_loss=total_loss,
        pixel_loss=pixel_loss,
        distillation_loss=distillation_loss,
        scale_loss_1=scale_losses[0],
        scale_loss_2=scale_losses[1],
        scale_loss_3=scale_losses[2],
    )


def build_reconstruction_residual_map(
    reconstruction: torch.Tensor,
    reconstruction_target: torch.Tensor,
) -> torch.Tensor:
    return torch.abs(reconstruction - reconstruction_target)


def topk_mean_score(residual_map: torch.Tensor, topk_ratio: float) -> torch.Tensor:
    batch_size = residual_map.shape[0]
    flattened = residual_map.view(batch_size, -1)
    k = max(1, int(flattened.shape[1] * topk_ratio))
    values, _ = torch.topk(flattened, k=k, dim=1)
    return values.mean(dim=1)


def mean_score(residual_map: torch.Tensor) -> torch.Tensor:
    return residual_map.mean(dim=(1, 2, 3))
