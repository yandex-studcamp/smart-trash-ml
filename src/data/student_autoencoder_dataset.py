from __future__ import annotations

import os
import random
from numbers import Number
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from src.data.anomaly_dataset import ROIConfig


class StudentAutoencoderDataset(Dataset):
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
        self.data_frame["anomaly_label"] = self.data_frame.apply(self._resolve_label, axis=1)
        if self.normal_only:
            self.data_frame = self.data_frame[self.data_frame["anomaly_label"] == 0].reset_index(drop=True)

        if self.data_frame.empty:
            raise ValueError(f"Dataset is empty after filtering: {csv_file}")

    def __len__(self) -> int:
        return len(self.data_frame)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.data_frame.iloc[idx]
        image_path = self._resolve_path(str(row["file_path"]))

        image = Image.open(image_path).convert("L")
        if self.roi is not None:
            image = self._crop_roi(image, self.roi)

        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=Image.Resampling.BILINEAR,
        )

        if self.augment_horizontal_flip and random.random() < 0.5:
            image = TF.hflip(image)

        image_tensor = TF.to_tensor(image)

        return {
            "image": image_tensor.clone(),
            "reconstruction_target": image_tensor,
            "label": int(row["anomaly_label"]),
            "path": os.fspath(image_path),
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
