from __future__ import annotations

import torch
import torch.nn as nn


def _build_fixed_spatial_mask(
    *,
    height: int,
    width: int,
    top_fraction: float,
    bottom_fraction: float,
    left_fraction: float,
    right_fraction: float,
) -> torch.Tensor:
    mask = torch.ones((1, 1, height, width), dtype=torch.float32)

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


class StudentAutoencoderScoreWrapper(nn.Module):
    """Wraps the reconstruction model and returns a single anomaly score per image."""

    def __init__(
        self,
        autoencoder: nn.Module,
        *,
        input_height: int,
        input_width: int,
        input_channels: int = 1,
        score_mode: str = "mse_mean",
        pixel_topk_ratio: float = 0.0,
        pixel_topk_weight: float = 0.0,
        use_stable_spatial_mask: bool = False,
        stable_mask_top_fraction: float = 0.0,
        stable_mask_bottom_fraction: float = 0.0,
        stable_mask_left_fraction: float = 0.0,
        stable_mask_right_fraction: float = 0.0,
    ) -> None:
        super().__init__()
        if input_height <= 0 or input_width <= 0:
            raise ValueError("input_height and input_width must be positive.")
        if input_channels not in {1, 3}:
            raise ValueError("input_channels must be 1 or 3.")
        if score_mode not in {"mae_mean", "mse_mean", "topk_mean"}:
            raise ValueError("score_mode must be one of: mae_mean, mse_mean, topk_mean.")
        if not 0.0 <= pixel_topk_ratio <= 1.0:
            raise ValueError("pixel_topk_ratio must be in [0, 1].")
        if not 0.0 <= pixel_topk_weight <= 1.0:
            raise ValueError("pixel_topk_weight must be in [0, 1].")

        self.autoencoder = autoencoder
        self.input_channels = int(input_channels)
        self.score_mode = score_mode
        self.use_topk = (
            score_mode == "topk_mean"
            and pixel_topk_ratio > 0.0
            and pixel_topk_weight > 0.0
        )
        self.pixel_topk_weight = float(pixel_topk_weight)

        if use_stable_spatial_mask:
            score_mask = _build_fixed_spatial_mask(
                height=input_height,
                width=input_width,
                top_fraction=stable_mask_top_fraction,
                bottom_fraction=stable_mask_bottom_fraction,
                left_fraction=stable_mask_left_fraction,
                right_fraction=stable_mask_right_fraction,
            )
        else:
            score_mask = torch.ones((1, 1, input_height, input_width), dtype=torch.float32)

        valid_pixel_count = int(score_mask.sum().item())
        if valid_pixel_count <= 0:
            raise ValueError("Spatial mask removed all pixels.")

        self.register_buffer("score_mask", score_mask, persistent=True)
        self.register_buffer(
            "score_normalizer",
            torch.tensor(float(valid_pixel_count), dtype=torch.float32),
            persistent=True,
        )

        topk_k = max(1, int(valid_pixel_count * pixel_topk_ratio))
        self.topk_k = topk_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_channels == 3:
            x = x.mean(dim=1, keepdim=True)

        reconstruction = self.autoencoder(x)
        difference = reconstruction - x

        if self.score_mode == "mse_mean":
            residual_map = difference * difference * self.score_mask
            flattened = residual_map.flatten(start_dim=1)
            return flattened.sum(dim=1, keepdim=True) / self.score_normalizer

        positive_residual = torch.relu(difference)
        negative_residual = torch.relu(-difference)
        residual_map = (positive_residual + negative_residual) * self.score_mask
        flattened = residual_map.flatten(start_dim=1)
        mean_score = flattened.sum(dim=1, keepdim=True) / self.score_normalizer

        if self.use_topk:
            top_values, _ = torch.topk(flattened, k=self.topk_k, dim=1)
            topk_score = top_values.mean(dim=1, keepdim=True)
            return (1.0 - self.pixel_topk_weight) * mean_score + self.pixel_topk_weight * topk_score

        return mean_score
