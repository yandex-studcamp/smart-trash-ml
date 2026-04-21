from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.anomaly_reverse_distillation import MobileNetV3Teacher
from src.models.student_autoencoder import compute_student_autoencoder_losses


class StudentAutoencoderTrainer:
    def __init__(self, config, save_dir: str) -> None:
        self.config = config
        self.save_dir = save_dir
        self.device = config.device
        self.best_val_loss = float("inf")
        os.makedirs(self.save_dir, exist_ok=True)

    def fit(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        teacher: MobileNetV3Teacher | None = None,
    ) -> dict[str, list[float]]:
        model = model.to(self.device)
        optimizer = self.config.get_optimizer(model)

        if teacher is not None:
            teacher = teacher.to(self.device)
            teacher.eval()

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_pixel_loss": [],
            "val_pixel_loss": [],
            "train_pixel_topk_loss": [],
            "val_pixel_topk_loss": [],
            "train_distillation_loss": [],
            "val_distillation_loss": [],
        }

        for epoch in range(1, self.config.epochs + 1):
            print(f"\n--- Epoch {epoch}/{self.config.epochs} ---")
            train_metrics = self._run_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                teacher=teacher,
                training=True,
            )
            val_metrics = self._run_epoch(
                model=model,
                dataloader=val_loader,
                optimizer=None,
                teacher=teacher,
                training=False,
            )

            history["train_loss"].append(train_metrics["total_loss"])
            history["val_loss"].append(val_metrics["total_loss"])
            history["train_pixel_loss"].append(train_metrics["pixel_loss"])
            history["val_pixel_loss"].append(val_metrics["pixel_loss"])
            history["train_pixel_topk_loss"].append(train_metrics["pixel_topk_loss"])
            history["val_pixel_topk_loss"].append(val_metrics["pixel_topk_loss"])
            history["train_distillation_loss"].append(train_metrics["distillation_loss"])
            history["val_distillation_loss"].append(val_metrics["distillation_loss"])

            self._save_checkpoint(model, epoch)
            self._save_best(model, val_metrics["total_loss"])

            print(
                "Train Loss: "
                f"{train_metrics['total_loss']:.4f} | "
                f"Pixel: {train_metrics['pixel_loss']:.4f} | "
                f"TopK Pixel: {train_metrics['pixel_topk_loss']:.4f} | "
                f"Distill: {train_metrics['distillation_loss']:.4f}"
            )
            print(
                "Validation Loss: "
                f"{val_metrics['total_loss']:.4f} | "
                f"Pixel: {val_metrics['pixel_loss']:.4f} | "
                f"TopK Pixel: {val_metrics['pixel_topk_loss']:.4f} | "
                f"Distill: {val_metrics['distillation_loss']:.4f}"
            )

        return history

    def _run_epoch(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer | None,
        teacher: MobileNetV3Teacher | None,
        training: bool,
    ) -> dict[str, float]:
        model.train(training)
        if teacher is not None:
            teacher.eval()

        total_loss = 0.0
        pixel_loss = 0.0
        pixel_topk_loss = 0.0
        distillation_loss = 0.0

        context_manager = torch.enable_grad() if training else torch.no_grad()
        with context_manager:
            progress_bar = tqdm(
                dataloader,
                desc="Train" if training else "Validation",
                leave=False,
            )
            for batch in progress_bar:
                inputs = batch["image"].to(self.device)
                reconstruction_target = batch["reconstruction_target"].to(self.device)
                spatial_mask = self.config.get_spatial_mask(
                    height=inputs.shape[-2],
                    width=inputs.shape[-1],
                    device=self.device,
                    dtype=inputs.dtype,
                )

                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)

                reconstruction = model(inputs)
                losses = compute_student_autoencoder_losses(
                    reconstruction=reconstruction,
                    reconstruction_target=reconstruction_target,
                    teacher=teacher,
                    feature_distillation_weights=self.config.feature_distillation_weights,
                    pixel_loss_weight=self.config.pixel_loss_weight,
                    distillation_weight=self.config.distillation_weight,
                    spatial_mask=spatial_mask,
                    pixel_topk_ratio=self.config.pixel_topk_ratio,
                    pixel_topk_weight=self.config.pixel_topk_weight,
                )

                if optimizer is not None:
                    losses.total_loss.backward()
                    optimizer.step()

                total_loss += losses.total_loss.item()
                pixel_loss += losses.pixel_loss.item()
                pixel_topk_loss += losses.pixel_topk_loss.item()
                distillation_loss += losses.distillation_loss.item()
                progress_bar.set_postfix({"loss": f"{losses.total_loss.item():.4f}"})

        num_batches = max(len(dataloader), 1)
        return {
            "total_loss": total_loss / num_batches,
            "pixel_loss": pixel_loss / num_batches,
            "pixel_topk_loss": pixel_topk_loss / num_batches,
            "distillation_loss": distillation_loss / num_batches,
        }

    def _save_checkpoint(self, model: torch.nn.Module, epoch: int) -> None:
        torch.save(
            model.state_dict(),
            os.path.join(self.save_dir, f"latest_model_epoch_{epoch}.pth"),
        )

    def _save_best(self, model: torch.nn.Module, val_loss: float) -> None:
        if val_loss >= self.best_val_loss:
            return

        self.best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(self.save_dir, "best_model.pth"))
        print(f"*** Saved new best student autoencoder (val_loss={val_loss:.4f}) ***")
