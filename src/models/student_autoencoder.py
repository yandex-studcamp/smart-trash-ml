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
    pixel_topk_loss: torch.Tensor
    distillation_loss: torch.Tensor
    scale_loss_1: torch.Tensor
    scale_loss_2: torch.Tensor
    scale_loss_3: torch.Tensor


def build_stable_spatial_mask(
    height: int,
    width: int,
    *,
    top_fraction: float,
    bottom_fraction: float,
    left_fraction: float,
    right_fraction: float,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    mask = torch.ones((1, 1, height, width), device=device, dtype=dtype)

    top = int(round(height * top_fraction))
    bottom = int(round(height * bottom_fraction))
    left = int(round(width * left_fraction))
    right = int(round(width * right_fraction))

    if top > 0:
        mask[:, :, :top, :] = 0.0
    if bottom > 0:
        mask[:, :, height - bottom:, :] = 0.0
    if left > 0:
        mask[:, :, :, :left] = 0.0
    if right > 0:
        mask[:, :, :, width - right:] = 0.0

    return mask


def _expand_spatial_mask(
    spatial_mask: torch.Tensor | None,
    target_shape: torch.Size,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if spatial_mask is None:
        return None
    if spatial_mask.ndim != 4:
        raise ValueError("Expected spatial_mask to have BCHW dimensions.")
    if spatial_mask.shape[-2:] != target_shape[-2:]:
        raise ValueError(
            f"Spatial mask shape {tuple(spatial_mask.shape[-2:])} does not match target shape {tuple(target_shape[-2:])}.",
        )
    mask = spatial_mask.to(device=device, dtype=dtype)
    if mask.shape[0] == 1 and target_shape[0] > 1:
        mask = mask.expand(target_shape[0], -1, -1, -1)
    if mask.shape[1] == 1 and target_shape[1] > 1:
        mask = mask.expand(-1, target_shape[1], -1, -1)
    if mask.shape[0] != target_shape[0] or mask.shape[1] != target_shape[1]:
        raise ValueError(
            f"Expanded spatial mask BCHW {tuple(mask.shape)} does not match target BCHW {tuple(target_shape)}.",
        )
    return mask


def masked_mean_score(
    residual_map: torch.Tensor,
    spatial_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if spatial_mask is None:
        return residual_map.mean(dim=(1, 2, 3))

    mask = _expand_spatial_mask(
        spatial_mask,
        residual_map.shape,
        device=residual_map.device,
        dtype=residual_map.dtype,
    )
    weighted_sum = (residual_map * mask).sum(dim=(1, 2, 3))
    normalizer = mask.sum(dim=(1, 2, 3)).clamp_min(1e-6)
    return weighted_sum / normalizer


def compute_student_autoencoder_losses(
    reconstruction: torch.Tensor,
    reconstruction_target: torch.Tensor,
    teacher: MobileNetV3Teacher | None,
    feature_distillation_weights: tuple[float, float, float],
    pixel_loss_weight: float,
    distillation_weight: float,
    spatial_mask: torch.Tensor | None = None,
    pixel_topk_ratio: float = 0.0,
    pixel_topk_weight: float = 0.0,
) -> StudentAutoencoderLosses:
    residual_map = torch.abs(reconstruction - reconstruction_target)
    pixel_loss = masked_mean_score(residual_map, spatial_mask=spatial_mask)
    pixel_loss = pixel_loss.mean()

    pixel_topk_loss = torch.zeros((), device=reconstruction.device, dtype=reconstruction.dtype)
    if pixel_topk_ratio > 0.0 and pixel_topk_weight > 0.0:
        pixel_topk_loss = topk_mean_score(
            residual_map,
            topk_ratio=pixel_topk_ratio,
            spatial_mask=spatial_mask,
        ).mean()

    blended_pixel_loss = (1.0 - pixel_topk_weight) * pixel_loss + pixel_topk_weight * pixel_topk_loss
    weighted_pixel_loss = pixel_loss_weight * blended_pixel_loss

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
        pixel_topk_loss=pixel_topk_loss,
        distillation_loss=distillation_loss,
        scale_loss_1=scale_losses[0],
        scale_loss_2=scale_losses[1],
        scale_loss_3=scale_losses[2],
    )


def build_reconstruction_residual_map(
    reconstruction: torch.Tensor,
    reconstruction_target: torch.Tensor,
    spatial_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    residual_map = torch.abs(reconstruction - reconstruction_target)
    if spatial_mask is None:
        return residual_map

    mask = _expand_spatial_mask(
        spatial_mask,
        residual_map.shape,
        device=residual_map.device,
        dtype=residual_map.dtype,
    )
    return residual_map * mask


def topk_mean_score(
    residual_map: torch.Tensor,
    topk_ratio: float,
    spatial_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size = residual_map.shape[0]
    flattened = residual_map.view(batch_size, -1)

    if spatial_mask is None:
        k = max(1, int(flattened.shape[1] * topk_ratio))
        values, _ = torch.topk(flattened, k=k, dim=1)
        return values.mean(dim=1)

    mask = _expand_spatial_mask(
        spatial_mask,
        residual_map.shape,
        device=residual_map.device,
        dtype=residual_map.dtype,
    )
    flattened_mask = mask.view(batch_size, -1) > 0.5
    scores: list[torch.Tensor] = []
    for sample_values, sample_mask in zip(flattened, flattened_mask):
        valid_values = sample_values[sample_mask]
        if valid_values.numel() == 0:
            valid_values = sample_values
        k = max(1, int(valid_values.shape[0] * topk_ratio))
        top_values, _ = torch.topk(valid_values, k=k, dim=0)
        scores.append(top_values.mean())
    return torch.stack(scores)


def mean_score(
    residual_map: torch.Tensor,
    spatial_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return masked_mean_score(residual_map, spatial_mask=spatial_mask)
