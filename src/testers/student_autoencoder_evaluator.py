from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader

from src.models.student_autoencoder import (
    build_reconstruction_residual_map,
    compute_student_autoencoder_losses,
    mean_score,
    topk_mean_score,
)


class StudentAutoencoderEvaluator:
    def __init__(self, config, device: str, save_dir: str | None = None) -> None:
        self.config = config
        self.device = device
        self.save_dir = save_dir

    def predict(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        teacher: torch.nn.Module | None = None,
        save_anomaly_maps: bool = False,
    ) -> dict[str, Any]:
        model = model.to(self.device)
        model.eval()

        if teacher is not None:
            teacher = teacher.to(self.device)
            teacher.eval()

        if save_anomaly_maps and self.save_dir is None:
            raise ValueError("save_dir must be provided to save anomaly maps.")

        if save_anomaly_maps and self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

        scores: list[float] = []
        labels: list[int] = []
        paths: list[str] = []
        losses: list[float] = []
        saved_maps = 0

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch["image"].to(self.device)
                reconstruction_target = batch["reconstruction_target"].to(self.device)
                batch_labels = batch["label"]
                batch_paths = batch["path"]

                reconstruction = model(inputs)
                losses_dict = compute_student_autoencoder_losses(
                    reconstruction=reconstruction,
                    reconstruction_target=reconstruction_target,
                    teacher=teacher,
                    feature_distillation_weights=self.config.feature_distillation_weights,
                    pixel_loss_weight=self.config.pixel_loss_weight,
                    distillation_weight=self.config.distillation_weight,
                )
                residual_map = build_reconstruction_residual_map(
                    reconstruction=reconstruction,
                    reconstruction_target=reconstruction_target,
                )
                batch_scores = self._score_residual_map(residual_map)

                scores.extend(batch_scores.cpu().tolist())
                labels.extend([int(label) for label in batch_labels])
                paths.extend(list(batch_paths))
                losses.extend([losses_dict.total_loss.item()] * len(batch_paths))

                if save_anomaly_maps and self.save_dir is not None:
                    saved_maps = self._save_anomaly_maps(
                        batch_paths=batch_paths,
                        inputs=inputs,
                        reconstruction=reconstruction,
                        residual_map=residual_map,
                        start_index=saved_maps,
                        limit=self.config.max_saved_anomaly_maps,
                    )

        return {
            "scores": np.asarray(scores, dtype=np.float32),
            "labels": np.asarray(labels, dtype=np.int64),
            "paths": paths,
            "avg_loss": float(np.mean(losses)) if losses else 0.0,
        }

    def compute_threshold(self, normal_scores: np.ndarray) -> float:
        if self.config.threshold_mode != "val_quantile":
            raise ValueError(f"Unsupported threshold mode: {self.config.threshold_mode}")
        return float(np.quantile(normal_scores, self.config.threshold_quantile))

    def compute_metrics(
        self,
        labels: np.ndarray,
        scores: np.ndarray,
        threshold: float,
    ) -> dict[str, float]:
        if labels.size == 0:
            raise ValueError("No labels provided for metric computation.")

        predictions = (scores >= threshold).astype(np.int64)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            zero_division=0,
        )

        metrics = {
            "threshold": threshold,
            "precision_at_threshold": float(precision),
            "recall_at_threshold": float(recall),
            "f1_at_threshold": float(f1),
        }

        if len(np.unique(labels)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(labels, scores))
            metrics["pr_auc"] = float(average_precision_score(labels, scores))
        else:
            metrics["roc_auc"] = float("nan")
            metrics["pr_auc"] = float("nan")

        return metrics

    def save_histogram(self, scores: np.ndarray, labels: np.ndarray, output_path: str) -> None:
        normal_scores = scores[labels == 0]
        anomaly_scores = scores[labels == 1]

        plt.figure(figsize=(10, 5))
        plt.hist(normal_scores, bins=30, alpha=0.7, label="normal")
        if anomaly_scores.size > 0:
            plt.hist(anomaly_scores, bins=30, alpha=0.7, label="anomaly")
        plt.xlabel("Image anomaly score")
        plt.ylabel("Count")
        plt.title("Normal vs anomaly score histogram")
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()

    def _score_residual_map(self, residual_map: torch.Tensor) -> torch.Tensor:
        if self.config.score_mode == "mean":
            return mean_score(residual_map)
        if self.config.score_mode == "topk_mean":
            return topk_mean_score(
                residual_map=residual_map,
                topk_ratio=self.config.score_topk_ratio,
            )
        raise ValueError(f"Unsupported score mode: {self.config.score_mode}")

    def _save_anomaly_maps(
        self,
        batch_paths: list[str],
        inputs: torch.Tensor,
        reconstruction: torch.Tensor,
        residual_map: torch.Tensor,
        start_index: int,
        limit: int,
    ) -> int:
        current_count = start_index

        for sample_path, input_tensor, reconstruction_tensor, residual_tensor in zip(
            batch_paths,
            inputs,
            reconstruction,
            residual_map,
        ):
            if current_count >= limit:
                break

            filename = f"{current_count:04d}_{Path(sample_path).stem}.png"
            output_path = Path(self.save_dir) / filename
            self._save_anomaly_map(input_tensor, reconstruction_tensor, residual_tensor, output_path)
            current_count += 1

        return current_count

    @staticmethod
    def _save_anomaly_map(
        input_tensor: torch.Tensor,
        reconstruction_tensor: torch.Tensor,
        residual_tensor: torch.Tensor,
        output_path: Path,
    ) -> None:
        input_image = input_tensor.detach().cpu().squeeze(0).numpy()
        reconstruction_image = reconstruction_tensor.detach().cpu().squeeze(0).numpy()
        residual_image = residual_tensor.detach().cpu().squeeze(0).numpy()

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(input_image, cmap="gray", vmin=0.0, vmax=1.0)
        plt.axis("off")
        plt.title("Input")

        plt.subplot(1, 3, 2)
        plt.imshow(reconstruction_image, cmap="gray", vmin=0.0, vmax=1.0)
        plt.axis("off")
        plt.title("Reconstruction")

        plt.subplot(1, 3, 3)
        plt.imshow(residual_image, cmap="inferno")
        plt.axis("off")
        plt.title("Residual map")

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
