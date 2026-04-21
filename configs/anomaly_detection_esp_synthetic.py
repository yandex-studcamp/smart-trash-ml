import json
from dataclasses import asdict, dataclass, field

import torch

from src.data.anomaly_dataset import ROIConfig
from src.models.anomaly_reverse_distillation import ReverseDistillationLiteModel


@dataclass
class TrainConfig:
    train_csv: str = "data/anomaly_detection_dataset/labels/train.csv"
    valid_csv: str = "data/anomaly_detection_dataset/labels/validation.csv"
    test_csv: str = "data/anomaly_detection_dataset/labels/test.csv"
    img_dir: str = "data/anomaly_detection_dataset"

    encoder_name: str = "mobilenet_v3_small"
    use_pretrained_encoder: bool = True
    input_size: int = 128
    roi: ROIConfig | None = field(default_factory=lambda: ROIConfig(x=0, y=0, w=128, h=128))

    epochs: int = 25
    freeze_epochs: int = 25
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    bottleneck_channels: int = 128
    feature_loss_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    pixel_loss_weight: float = 0.0
    feature_map_weight: float = 1.0
    pixel_map_weight: float = 0.0
    score_topk_ratio: float = 0.05
    threshold_mode: str = "val_quantile"
    threshold_quantile: float = 0.995

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
        if self.freeze_epochs < 0:
            raise ValueError("freeze_epochs must be non-negative.")
        if len(self.feature_loss_weights) != 3:
            raise ValueError("feature_loss_weights must contain exactly 3 values.")
        if not 0.0 < self.score_topk_ratio <= 1.0:
            raise ValueError("score_topk_ratio must be in (0, 1].")
        if not 0.0 < self.threshold_quantile < 1.0:
            raise ValueError("threshold_quantile must be in (0, 1).")
        if self.roi is not None:
            self.roi.validate()

    def save_to_json(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(asdict(self), file, indent=4)

    def get_model(self) -> ReverseDistillationLiteModel:
        use_image_decoder = self.pixel_loss_weight > 0.0 or self.pixel_map_weight > 0.0
        return ReverseDistillationLiteModel(
            encoder_name=self.encoder_name,
            bottleneck_channels=self.bottleneck_channels,
            use_pretrained_encoder=self.use_pretrained_encoder,
            use_image_decoder=use_image_decoder,
        )

    def get_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
