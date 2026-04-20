from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.anomaly_reverse_distillation import compute_reverse_distillation_loss


class AnomalyTrainer:
    def __init__(self, config, save_dir: str) -> None:
        self.config = config
        self.save_dir = save_dir
        self.device = config.device
        self.best_val_loss = float("inf")
        os.makedirs(self.save_dir, exist_ok=True)

    def fit(self, model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> dict[str, list[float]]:
        model = model.to(self.device)
        optimizer = self.config.get_optimizer(model)
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_feature_loss": [],
            "val_feature_loss": [],
            "train_pixel_loss": [],
            "val_pixel_loss": [],
        }

        for epoch in range(1, self.config.epochs + 1):
            encoder_trainable = epoch > self.config.freeze_epochs
            model.set_encoder_trainable(encoder_trainable)
            print(f"\n--- Epoch {epoch}/{self.config.epochs} ---")
            print(f"Encoder trainable: {encoder_trainable}")

            train_metrics = self._run_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                training=True,
            )
            val_metrics = self._run_epoch(
                model=model,
                dataloader=val_loader,
                optimizer=None,
                training=False,
            )

            history["train_loss"].append(train_metrics["total_loss"])
            history["val_loss"].append(val_metrics["total_loss"])
            history["train_feature_loss"].append(train_metrics["feature_loss"])
            history["val_feature_loss"].append(val_metrics["feature_loss"])
            history["train_pixel_loss"].append(train_metrics["pixel_loss"])
            history["val_pixel_loss"].append(val_metrics["pixel_loss"])

            self._save_checkpoint(model, epoch)
            self._save_best(model, val_metrics["total_loss"])

            print(
                "Train Loss: "
                f"{train_metrics['total_loss']:.4f} | "
                f"Feature: {train_metrics['feature_loss']:.4f} | "
                f"Pixel: {train_metrics['pixel_loss']:.4f}"
            )
            print(
                "Validation Loss: "
                f"{val_metrics['total_loss']:.4f} | "
                f"Feature: {val_metrics['feature_loss']:.4f} | "
                f"Pixel: {val_metrics['pixel_loss']:.4f}"
            )

        return history

    def _run_epoch(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer | None,
        training: bool,
    ) -> dict[str, float]:
        if training:
            model.train()
            if not model._encoder_trainable:
                model.teacher.eval()
        else:
            model.eval()

        total_loss = 0.0
        feature_loss = 0.0
        pixel_loss = 0.0

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

                if optimizer is not None:
                    optimizer.zero_grad()

                output = model(inputs)
                losses = compute_reverse_distillation_loss(
                    output=output,
                    reconstruction_target=reconstruction_target,
                    feature_loss_weights=self.config.feature_loss_weights,
                    pixel_loss_weight=self.config.pixel_loss_weight,
                )

                if optimizer is not None:
                    losses["total_loss"].backward()
                    optimizer.step()

                total_loss += losses["total_loss"].item()
                feature_loss += losses["feature_loss"].item()
                pixel_loss += losses["pixel_loss"].item()
                progress_bar.set_postfix({"loss": f"{losses['total_loss'].item():.4f}"})

        num_batches = max(len(dataloader), 1)
        return {
            "total_loss": total_loss / num_batches,
            "feature_loss": feature_loss / num_batches,
            "pixel_loss": pixel_loss / num_batches,
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
        print(f"*** Saved new best anomaly model (val_loss={val_loss:.4f}) ***")
