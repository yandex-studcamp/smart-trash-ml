from __future__ import annotations

import argparse
import importlib
import json
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.data.student_autoencoder_dataset import StudentAutoencoderDataset
from src.testers.student_autoencoder_evaluator import StudentAutoencoderEvaluator
from src.trainers.student_autoencoder_trainer import StudentAutoencoderTrainer


def setup_experiment(exp_name: str) -> tuple[str, str]:
    base_dir = os.path.join("experiments", "anomaly_detection", exp_name)
    weights_dir = os.path.join(base_dir, "weights")
    artifacts_dir = os.path.join(base_dir, "artifacts")

    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)
    return weights_dir, artifacts_dir


def build_dataloader(
    dataset: StudentAutoencoderDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def plot_training_history(history: dict[str, list[float]], output_path: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="train_total_loss", marker="o")
    plt.plot(history["val_loss"], label="val_total_loss", marker="o")
    plt.plot(history["train_pixel_loss"], label="train_pixel_loss", linestyle="--")
    plt.plot(history["val_pixel_loss"], label="val_pixel_loss", linestyle="--")
    plt.plot(history["train_distillation_loss"], label="train_distillation_loss", linestyle=":")
    plt.plot(history["val_distillation_loss"], label="val_distillation_loss", linestyle=":")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Student-only autoencoder training history")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the student-only grayscale autoencoder for anomaly detection.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config module name from configs/ without the .py suffix.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="Experiment directory name under experiments/anomaly_detection/.",
    )
    args = parser.parse_args()

    config_module = importlib.import_module(f"configs.{args.config}")
    cfg = config_module.TrainConfig()

    weights_dir, artifacts_dir = setup_experiment(args.exp_name)
    cfg.save_to_json(os.path.join("experiments", "anomaly_detection", args.exp_name, "config.json"))

    print(f"=== Student autoencoder training: {args.exp_name} ===")
    print(f"Device: {cfg.device}")

    train_dataset = StudentAutoencoderDataset(
        csv_file=cfg.train_csv,
        root_dir=cfg.img_dir,
        image_size=cfg.input_size,
        roi=cfg.roi,
        normal_only=True,
        augment_horizontal_flip=cfg.train_horizontal_flip,
    )
    val_dataset = StudentAutoencoderDataset(
        csv_file=cfg.valid_csv,
        root_dir=cfg.img_dir,
        image_size=cfg.input_size,
        roi=cfg.roi,
        normal_only=True,
        augment_horizontal_flip=False,
    )

    train_loader = build_dataloader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = build_dataloader(
        dataset=val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    model = cfg.get_model()
    teacher = cfg.get_teacher()
    trainer = StudentAutoencoderTrainer(config=cfg, save_dir=weights_dir)
    history = trainer.fit(model=model, train_loader=train_loader, val_loader=val_loader, teacher=teacher)

    best_weights_path = os.path.join(weights_dir, "best_model.pth")
    model.load_state_dict(torch.load(best_weights_path, map_location=cfg.device))

    evaluator = StudentAutoencoderEvaluator(
        config=cfg,
        device=cfg.device,
        save_dir=os.path.join(artifacts_dir, "val_maps"),
    )
    val_predictions = evaluator.predict(model=model, dataloader=val_loader, teacher=teacher, save_anomaly_maps=False)
    threshold = evaluator.compute_threshold(val_predictions["scores"])
    threshold_payload = {
        "threshold_mode": cfg.threshold_mode,
        "threshold_quantile": cfg.threshold_quantile,
        "threshold": threshold,
        "val_avg_loss": val_predictions["avg_loss"],
        "val_num_samples": int(len(val_predictions["scores"])),
    }

    with open(os.path.join(artifacts_dir, "threshold.json"), "w", encoding="utf-8") as file:
        json.dump(threshold_payload, file, indent=4)

    plot_training_history(history, os.path.join(artifacts_dir, "loss_history.png"))
    evaluator.save_histogram(
        scores=val_predictions["scores"],
        labels=val_predictions["labels"],
        output_path=os.path.join(artifacts_dir, "val_score_histogram.png"),
    )

    print(f"Saved best student weights to: {best_weights_path}")
    print(f"Validation threshold ({cfg.threshold_mode}) = {threshold:.6f}")


if __name__ == "__main__":
    main()
