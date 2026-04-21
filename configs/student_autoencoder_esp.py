import json
from dataclasses import asdict, dataclass, field

import torch

from src.data.anomaly_dataset import ROIConfig
from src.models.student_autoencoder import (
    StudentOnlyAutoencoder,
    build_distillation_teacher,
    build_stable_spatial_mask,
)


@dataclass
class TrainConfig:
    train_csv: str = "data/anomaly_detection_dataset/labels/train.csv"
    valid_csv: str = "data/anomaly_detection_dataset/labels/validation.csv"
    test_csv: str = "data/anomaly_detection_dataset/labels/test.csv"
    img_dir: str = "data/anomaly_detection_dataset"

    input_size: int = 96
    input_channels: int = 1
    roi: ROIConfig | None = field(default_factory=lambda: ROIConfig(x=0, y=0, w=128, h=128))

    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 8e-4
    weight_decay: float = 1e-5
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    encoder_channels: tuple[int, int, int, int] = (16, 24, 32, 48)
    bottleneck_channels: int = 64

    pixel_loss_weight: float = 1.0
    distillation_weight: float = 0.15
    feature_distillation_weights: tuple[float, float, float] = (1.0, 0.75, 0.5)
    teacher_encoder_name: str = "mobilenet_v3_small"
    use_pretrained_teacher: bool = True
    use_teacher_distillation: bool = True
    pixel_topk_ratio: float = 0.0
    pixel_topk_weight: float = 0.0

    score_mode: str = "mean"
    score_topk_ratio: float = 0.05
    threshold_mode: str = "val_quantile"
    threshold_quantile: float = 0.995
    use_stable_spatial_mask: bool = False
    stable_mask_top_fraction: float = 0.16
    stable_mask_bottom_fraction: float = 0.04
    stable_mask_left_fraction: float = 0.05
    stable_mask_right_fraction: float = 0.05

    train_horizontal_flip: bool = True
    save_anomaly_maps: bool = True
    max_saved_anomaly_maps: int = 100

    def __post_init__(self) -> None:
        if self.input_size <= 0:
            raise ValueError("input_size must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if len(self.encoder_channels) != 4:
            raise ValueError("encoder_channels must contain exactly 4 values.")
        if self.bottleneck_channels <= 0:
            raise ValueError("bottleneck_channels must be positive.")
        if len(self.feature_distillation_weights) != 3:
            raise ValueError("feature_distillation_weights must contain exactly 3 values.")
        if self.pixel_loss_weight <= 0.0:
            raise ValueError("pixel_loss_weight must be positive.")
        if self.distillation_weight < 0.0:
            raise ValueError("distillation_weight must be non-negative.")
        if not 0.0 <= self.pixel_topk_ratio <= 1.0:
            raise ValueError("pixel_topk_ratio must be in [0, 1].")
        if not 0.0 <= self.pixel_topk_weight <= 1.0:
            raise ValueError("pixel_topk_weight must be in [0, 1].")
        if self.score_mode not in {"mean", "topk_mean"}:
            raise ValueError("score_mode must be either `mean` or `topk_mean`.")
        if not 0.0 < self.score_topk_ratio <= 1.0:
            raise ValueError("score_topk_ratio must be in (0, 1].")
        if not 0.0 < self.threshold_quantile < 1.0:
            raise ValueError("threshold_quantile must be in (0, 1).")
        for fraction_name in (
            "stable_mask_top_fraction",
            "stable_mask_bottom_fraction",
            "stable_mask_left_fraction",
            "stable_mask_right_fraction",
        ):
            fraction_value = getattr(self, fraction_name)
            if not 0.0 <= fraction_value < 1.0:
                raise ValueError(f"{fraction_name} must be in [0, 1).")
        if self.input_channels != 1:
            raise ValueError("Only single-channel grayscale input is supported for the student autoencoder.")
        if self.roi is not None:
            self.roi.validate()

    def save_to_json(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(asdict(self), file, indent=4)

    def get_model(self) -> StudentOnlyAutoencoder:
        return StudentOnlyAutoencoder(
            in_channels=self.input_channels,
            encoder_channels=self.encoder_channels,
            bottleneck_channels=self.bottleneck_channels,
        )

    def get_teacher(self):
        if not self.use_teacher_distillation or self.distillation_weight <= 0.0:
            return None
        return build_distillation_teacher(
            encoder_name=self.teacher_encoder_name,
            use_pretrained_encoder=self.use_pretrained_teacher,
        )

    def get_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def get_spatial_mask(
        self,
        *,
        height: int,
        width: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if not self.use_stable_spatial_mask:
            return None
        return build_stable_spatial_mask(
            height=height,
            width=width,
            top_fraction=self.stable_mask_top_fraction,
            bottom_fraction=self.stable_mask_bottom_fraction,
            left_fraction=self.stable_mask_left_fraction,
            right_fraction=self.stable_mask_right_fraction,
            device=device,
            dtype=dtype,
        )
