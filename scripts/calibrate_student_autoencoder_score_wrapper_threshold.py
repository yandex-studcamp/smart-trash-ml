from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.anomaly_dataset import ROIConfig
from src.data.student_autoencoder_dataset import StudentAutoencoderDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate threshold for a score-only student autoencoder wrapper on validation data.",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="Path to the exported TorchScript wrapper (.pt).",
    )
    parser.add_argument(
        "--config_path",
        type=Path,
        default=None,
        help="Optional path to experiment config.json. If omitted, it is inferred from model_path.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="Optional path to save threshold and validation metrics as JSON.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Execution device, usually `cpu`.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Optional batch size override. Defaults to config batch_size.",
    )
    parser.add_argument(
        "--threshold_quantile",
        type=float,
        default=None,
        help="Optional quantile override. Defaults to config threshold_quantile.",
    )
    return parser.parse_args()


def resolve_config_path(model_path: Path, explicit_config_path: Path | None) -> Path:
    if explicit_config_path is not None:
        config_path = explicit_config_path.expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file was not found: {config_path}")
        return config_path

    model_path = model_path.expanduser().resolve()
    if model_path.parent.name == "artifacts":
        candidate = model_path.parent.parent / "config.json"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not infer config.json from model_path. Pass --config_path explicitly.",
    )


def build_loader(
    *,
    csv_file: str,
    root_dir: str,
    input_size: int,
    roi: ROIConfig | None,
    normal_only: bool,
    batch_size: int,
) -> DataLoader:
    dataset = StudentAutoencoderDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        image_size=input_size,
        roi=roi,
        normal_only=normal_only,
        augment_horizontal_flip=False,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


def predict_scores(
    model: torch.jit.RecursiveScriptModule,
    dataloader: DataLoader,
    *,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores: list[float] = []
    labels: list[int] = []

    with torch.inference_mode():
        for batch in dataloader:
            inputs = batch["image"].to(device)
            outputs = model(inputs)
            scores.extend(outputs.squeeze(1).detach().cpu().tolist())
            labels.extend(int(label) for label in batch["label"])

    return np.asarray(scores, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def compute_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (scores >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )

    metrics = {
        "threshold": float(threshold),
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


def main() -> None:
    args = parse_args()
    device = args.device.strip().lower()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    model_path = args.model_path.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file was not found: {model_path}")

    config_path = resolve_config_path(model_path, args.config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    roi = ROIConfig(**config["roi"]) if config.get("roi") else None
    batch_size = int(args.batch_size or config["batch_size"])
    threshold_quantile = float(args.threshold_quantile or config["threshold_quantile"])

    model = torch.jit.load(model_path, map_location=device)
    model.eval()

    normal_loader = build_loader(
        csv_file=config["valid_csv"],
        root_dir=config["img_dir"],
        input_size=int(config["input_size"]),
        roi=roi,
        normal_only=True,
        batch_size=batch_size,
    )
    full_loader = build_loader(
        csv_file=config["valid_csv"],
        root_dir=config["img_dir"],
        input_size=int(config["input_size"]),
        roi=roi,
        normal_only=False,
        batch_size=batch_size,
    )

    normal_scores, _ = predict_scores(model, normal_loader, device=device)
    full_scores, full_labels = predict_scores(model, full_loader, device=device)
    threshold = float(np.quantile(normal_scores, threshold_quantile))
    metrics = compute_metrics(full_labels, full_scores, threshold)

    payload: dict[str, Any] = {
        "model_path": str(model_path),
        "config_path": str(config_path),
        "threshold_mode": "val_quantile",
        "threshold_quantile": threshold_quantile,
        "threshold": threshold,
        "num_validation_normals": int(len(normal_scores)),
        "num_validation_samples": int(len(full_scores)),
        "normal_score_mean": float(normal_scores.mean()) if len(normal_scores) > 0 else 0.0,
        "normal_score_std": float(normal_scores.std()) if len(normal_scores) > 0 else 0.0,
        "validation_metrics": metrics,
    }

    output_path = args.output_path
    if output_path is not None:
        output_path = output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=4), encoding="utf-8")

    print("=== Wrapper threshold calibration finished ===")
    print(f"Model: {model_path}")
    print(f"Config: {config_path}")
    print(f"Validation normal samples: {len(normal_scores)}")
    print(f"Validation total samples: {len(full_scores)}")
    print(f"Threshold quantile: {threshold_quantile}")
    print(f"Threshold: {threshold:.10f}")
    print(f"Precision@threshold: {metrics['precision_at_threshold']:.6f}")
    print(f"Recall@threshold: {metrics['recall_at_threshold']:.6f}")
    print(f"F1@threshold: {metrics['f1_at_threshold']:.6f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.6f}")
    print(f"PR-AUC: {metrics['pr_auc']:.6f}")
    if output_path is not None:
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
