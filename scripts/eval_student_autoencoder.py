from __future__ import annotations

import argparse
import importlib
import json
import os

import torch
from torch.utils.data import DataLoader

from src.data.student_autoencoder_dataset import StudentAutoencoderDataset
from src.testers.student_autoencoder_evaluator import StudentAutoencoderEvaluator


def build_dataloader(dataset: StudentAutoencoderDataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def load_threshold(exp_dir: str) -> float:
    threshold_path = os.path.join(exp_dir, "artifacts", "threshold.json")
    if not os.path.exists(threshold_path):
        raise FileNotFoundError(f"Threshold file not found: {threshold_path}")

    with open(threshold_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    return float(payload["threshold"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the student-only grayscale autoencoder.",
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
    parser.add_argument(
        "--weights",
        type=str,
        default="best_model.pth",
        help="Weights filename inside experiments/anomaly_detection/<exp_name>/weights/.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["validation", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--save_maps",
        action="store_true",
        help="Save residual maps for samples from the selected split.",
    )
    args = parser.parse_args()

    config_module = importlib.import_module(f"configs.{args.config}")
    cfg = config_module.TrainConfig()

    exp_dir = os.path.join("experiments", "anomaly_detection", args.exp_name)
    weights_path = os.path.join(exp_dir, "weights", args.weights)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    split_csv = cfg.test_csv if args.split == "test" else cfg.valid_csv
    dataset = StudentAutoencoderDataset(
        csv_file=split_csv,
        root_dir=cfg.img_dir,
        image_size=cfg.input_size,
        roi=cfg.roi,
        normal_only=False,
        augment_horizontal_flip=False,
    )
    dataloader = build_dataloader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    model = cfg.get_model()
    model.load_state_dict(torch.load(weights_path, map_location=cfg.device))
    teacher = cfg.get_teacher()

    artifacts_dir = os.path.join(exp_dir, "artifacts")
    maps_dir = os.path.join(artifacts_dir, f"{args.split}_residual_maps")
    evaluator = StudentAutoencoderEvaluator(
        config=cfg,
        device=cfg.device,
        save_dir=maps_dir,
    )

    threshold = load_threshold(exp_dir)
    predictions = evaluator.predict(
        model=model,
        dataloader=dataloader,
        teacher=teacher,
        save_anomaly_maps=args.save_maps and cfg.save_anomaly_maps,
    )
    metrics = evaluator.compute_metrics(
        labels=predictions["labels"],
        scores=predictions["scores"],
        threshold=threshold,
    )
    metrics["avg_loss"] = predictions["avg_loss"]
    metrics["num_samples"] = int(len(predictions["scores"]))

    evaluator.save_histogram(
        scores=predictions["scores"],
        labels=predictions["labels"],
        output_path=os.path.join(artifacts_dir, f"{args.split}_score_histogram.png"),
    )

    metrics_path = os.path.join(artifacts_dir, f"{args.split}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"=== Student autoencoder evaluation: {args.exp_name} / {args.split} ===")
    print(f"Weights: {weights_path}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
