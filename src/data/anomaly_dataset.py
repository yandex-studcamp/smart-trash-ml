from __future__ import annotations

import os
import random
from dataclasses import dataclass
from numbers import Number
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True, slots=True)
class ROIConfig:
    x: int
    y: int
    w: int
    h: int

    def as_xywh(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    def validate(self) -> None:
        if self.w <= 0 or self.h <= 0:
            raise ValueError("ROI width and height must be positive.")
        if self.x < 0 or self.y < 0:
            raise ValueError("ROI x and y must be non-negative.")


@dataclass(frozen=True, slots=True)
class AnomalySample:
    image: torch.Tensor
    reconstruction_target: torch.Tensor
    label: int
    path: str


class AnomalyDetectionDataset(Dataset):
    def __init__(
            self,
            csv_file: str,
            root_dir: str,
            image_size: int,
            roi: ROIConfig | None = None,
            normal_only: bool = False,
            augment_horizontal_flip: bool = False,
    ) -> None:
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.image_size = image_size
        self.roi = roi
        self.normal_only = normal_only
        self.augment_horizontal_flip = augment_horizontal_flip

        if "file_path" not in self.data_frame.columns:
            raise ValueError(f"`file_path` column is required in {csv_file}.")

        self.data_frame = self.data_frame.copy()
        self.data_frame["anomaly_label"] = self.data_frame.apply(
            self._resolve_label,
            axis=1,
        )
        if self.normal_only:
            self.data_frame = self.data_frame[self.data_frame["anomaly_label"] == 0].reset_index(drop=True)

        if self.data_frame.empty:
            raise ValueError(f"Dataset is empty after filtering: {csv_file}")

    def __len__(self) -> int:
        return len(self.data_frame)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.data_frame.iloc[idx]
        image_path = self._resolve_path(str(row["file_path"]))

        image = Image.open(image_path).convert("RGB")
        if self.roi is not None:
            image = self._crop_roi(image, self.roi)

        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=Image.Resampling.BILINEAR,
        )
        image = TF.rgb_to_grayscale(image, num_output_channels=3)

        if self.augment_horizontal_flip and random.random() < 0.5:
            image = TF.hflip(image)

        reconstruction_target = TF.to_tensor(image)
        model_input = TF.normalize(
            reconstruction_target.clone(),
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )

        sample = AnomalySample(
            image=model_input,
            reconstruction_target=reconstruction_target,
            label=int(row["anomaly_label"]),
            path=os.fspath(image_path),
        )
        return {
            "image": sample.image,
            "reconstruction_target": sample.reconstruction_target,
            "label": sample.label,
            "path": sample.path,
        }

    def _resolve_label(self, row: pd.Series) -> int:
        for column in ("label_id", "label", "target", "class_name", "class", "is_anomaly"):
            if column not in row.index:
                continue

            value = row[column]
            if pd.isna(value):
                continue
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"normal", "ok", "good", "0"}:
                    return 0
                if normalized in {"anomaly", "anomalous", "defect", "1"}:
                    return 1
            if isinstance(value, Number):
                return int(value != 0)

        path_parts = Path(str(row["file_path"]).replace("\\", "/")).parts
        return 1 if "anomaly" in {part.lower() for part in path_parts} else 0

    def _resolve_path(self, file_path: str) -> Path:
        path = Path(file_path)
        if path.is_absolute():
            return path
        return Path(self.root_dir) / path

    @staticmethod
    def _crop_roi(image: Image.Image, roi: ROIConfig) -> Image.Image:
        x, y, width, height = roi.as_xywh()
        return image.crop((x, y, x + width, y + height))
