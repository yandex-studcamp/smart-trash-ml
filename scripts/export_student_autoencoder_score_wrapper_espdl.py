from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import onnx
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.anomaly_dataset import ROIConfig
from src.data.student_autoencoder_dataset import StudentAutoencoderDataset
from src.models.student_autoencoder import StudentOnlyAutoencoder
from src.models.student_autoencoder_score_wrapper import StudentAutoencoderScoreWrapper
from src.models.student_autoencoder_v3 import StudentAutoencoderV3
from src.onnx_exporter import convert_to_external_data


FORBIDDEN_ESPDL_OPS = {
    "Abs",
    "ArgMax",
    "ArgMin",
    "NonZero",
    "Sort",
    "TopK",
}


class ScoreWrapperCalibrationDataset(Dataset):
    def __init__(
        self,
        *,
        csv_file: str,
        root_dir: str,
        image_size: int,
        roi: ROIConfig | None,
        input_channels: int,
        required_len: int,
    ) -> None:
        self.dataset = StudentAutoencoderDataset(
            csv_file=csv_file,
            root_dir=root_dir,
            image_size=image_size,
            roi=roi,
            normal_only=True,
            augment_horizontal_flip=False,
        )
        if len(self.dataset) == 0:
            raise ValueError(f"Calibration dataset is empty: {csv_file}")
        self.input_channels = input_channels
        self.required_len = max(required_len, len(self.dataset))

    def __len__(self) -> int:
        return self.required_len

    def __getitem__(self, index: int) -> torch.Tensor:
        sample = self.dataset[index % len(self.dataset)]
        image = sample["image"].float()
        if self.input_channels == 3 and image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export student autoencoder score wrapper to ONNX external data and ESPDL.",
    )
    parser.add_argument(
        "--architecture",
        choices=["v2", "v3"],
        required=True,
        help="Student autoencoder architecture used by the weights.",
    )
    parser.add_argument(
        "--weights_path",
        type=Path,
        required=True,
        help="Path to trained autoencoder weights (.pth).",
    )
    parser.add_argument(
        "--config_path",
        type=Path,
        default=None,
        help="Optional experiment config.json. If omitted, inferred from weights_path.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="Exact .espdl output path. If omitted, saved under experiment artifacts.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Optional model basename for generated artifacts.",
    )
    parser.add_argument(
        "--score_mode",
        choices=["mse_mean", "mae_mean"],
        default="mse_mean",
        help="ESPDL-safe scalar score. mse_mean is recommended.",
    )
    parser.add_argument(
        "--model_input_channels",
        type=int,
        choices=[1, 3],
        default=3,
        help="Wrapper input channels. Use 3 for firmware RGB input.",
    )
    parser.add_argument("--target", default="c", help="ESP-PPQ target. Use c for ESP32.")
    parser.add_argument("--bits", type=int, choices=[8, 16], default=8)
    parser.add_argument("--device", default="cuda", help="ESP-PPQ device: cuda or cpu.")
    parser.add_argument("--calib_steps", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--error_report", action="store_true")
    parser.add_argument("--export_test_values", action="store_true")
    return parser.parse_args()


def resolve_config_path(weights_path: Path, explicit_config_path: Path | None) -> Path:
    if explicit_config_path is not None:
        config_path = explicit_config_path.expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file was not found: {config_path}")
        return config_path

    weights_path = weights_path.expanduser().resolve()
    if weights_path.parent.name == "weights":
        candidate = weights_path.parent.parent / "config.json"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not infer config.json from weights_path. Pass --config_path explicitly.",
    )


def load_config(config_path: Path) -> dict[str, Any]:
    return json.loads(config_path.read_text(encoding="utf-8"))


def infer_v3_architecture_from_state_dict(state_dict: dict[str, torch.Tensor]) -> tuple[tuple[int, int, int, int], int]:
    channels = (
        int(state_dict["stem.0.0.weight"].shape[0]),
        int(state_dict["encoder.0.downsample.0.weight"].shape[0]),
        int(state_dict["encoder.1.downsample.0.weight"].shape[0]),
        int(state_dict["encoder.2.downsample.0.weight"].shape[0]),
    )
    bottleneck_channels = int(state_dict["encoder.3.downsample.0.weight"].shape[0])
    return channels, bottleneck_channels


def build_autoencoder(
    *,
    architecture: str,
    config: dict[str, Any],
    weights_path: Path,
    device: str,
) -> torch.nn.Module:
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    if architecture == "v2":
        autoencoder = StudentOnlyAutoencoder(
            in_channels=int(config["input_channels"]),
            encoder_channels=tuple(int(value) for value in config["encoder_channels"]),
            bottleneck_channels=int(config["bottleneck_channels"]),
        )
    elif architecture == "v3":
        channels, bottleneck_channels = infer_v3_architecture_from_state_dict(state_dict)
        autoencoder = StudentAutoencoderV3(
            in_channels=int(config["input_channels"]),
            channels=channels,
            bottleneck_channels=bottleneck_channels,
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    autoencoder.load_state_dict(state_dict, strict=True)
    return autoencoder.to(device).eval()


def build_wrapper(
    *,
    architecture: str,
    config: dict[str, Any],
    weights_path: Path,
    device: str,
    score_mode: str,
    model_input_channels: int,
) -> StudentAutoencoderScoreWrapper:
    autoencoder = build_autoencoder(
        architecture=architecture,
        config=config,
        weights_path=weights_path,
        device=device,
    )
    input_size = int(config["input_size"])
    wrapper = StudentAutoencoderScoreWrapper(
        autoencoder=autoencoder,
        input_height=input_size,
        input_width=input_size,
        input_channels=model_input_channels,
        score_mode=score_mode,
        pixel_topk_ratio=0.0,
        pixel_topk_weight=0.0,
        use_stable_spatial_mask=bool(config.get("use_stable_spatial_mask", False)),
        stable_mask_top_fraction=float(config.get("stable_mask_top_fraction", 0.0)),
        stable_mask_bottom_fraction=float(config.get("stable_mask_bottom_fraction", 0.0)),
        stable_mask_left_fraction=float(config.get("stable_mask_left_fraction", 0.0)),
        stable_mask_right_fraction=float(config.get("stable_mask_right_fraction", 0.0)),
    )
    return wrapper.to(device).eval()


def resolve_output_paths(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    weights_path: Path,
) -> tuple[Path, Path]:
    input_size = int(config["input_size"])
    model_name = args.model_name or (
        f"student_autoencoder_{args.architecture}_score_wrapper_{args.score_mode}_{input_size}x{input_size}"
    )

    if args.output_path is not None:
        espdl_path = args.output_path.expanduser().resolve()
    else:
        if weights_path.parent.name == "weights":
            output_root = weights_path.parent.parent / "artifacts" / "espdl"
        else:
            output_root = PROJECT_ROOT / "artifacts" / "student_autoencoder_espdl"
        espdl_path = output_root / model_name / "espdl" / f"{model_name}.espdl"

    if espdl_path.parent.name == "espdl":
        onnx_path = espdl_path.parent.parent / "onnx" / f"{espdl_path.stem}.onnx"
    else:
        onnx_path = espdl_path.parent / "onnx" / f"{espdl_path.stem}.onnx"

    return onnx_path, espdl_path


def export_onnx_external(
    *,
    wrapper: torch.nn.Module,
    onnx_path: Path,
    input_channels: int,
    input_size: int,
    device: str,
    opset: int,
) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper.eval()
    dummy_input = torch.rand((1, input_channels, input_size, input_size), device=device, dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        dummy_input,
        onnx_path.as_posix(),
        input_names=["input"],
        output_names=["mse"],
        export_params=True,
        do_constant_folding=True,
        opset_version=opset,
        dynamo=False,
    )
    convert_to_external_data(onnx_path)
    validate_onnx_ops(onnx_path)


def validate_onnx_ops(onnx_path: Path) -> None:
    model = onnx.load(onnx_path.as_posix(), load_external_data=True)
    onnx.checker.check_model(model)
    ops = sorted({node.op_type for node in model.graph.node})
    forbidden = sorted(set(ops) & FORBIDDEN_ESPDL_OPS)
    if forbidden:
        raise RuntimeError(
            f"ONNX contains ESP-DL incompatible ops: {', '.join(forbidden)}. "
            f"All ops: {', '.join(ops)}"
        )


def build_calibration_loader(
    *,
    config: dict[str, Any],
    input_channels: int,
    required_len: int,
    batch_size: int,
) -> DataLoader:
    roi = ROIConfig(**config["roi"]) if config.get("roi") else None
    dataset = ScoreWrapperCalibrationDataset(
        csv_file=config["valid_csv"],
        root_dir=config["img_dir"],
        image_size=int(config["input_size"]),
        roi=roi,
        input_channels=input_channels,
        required_len=required_len,
    )
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=False)


def resolve_device(device: str) -> str:
    normalized = device.strip().lower()
    if normalized == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU for ESP-PPQ.")
        return "cpu"
    return normalized


def export_espdl(
    *,
    onnx_path: Path,
    espdl_path: Path,
    config: dict[str, Any],
    args: argparse.Namespace,
    device: str,
) -> None:
    from esp_ppq.api import espdl_quantize_onnx

    batch_size = args.batch_size
    if batch_size != 1:
        print(f"ESPDL requires batch_size=1. Overriding batch_size={batch_size} to 1.")
        batch_size = 1

    espdl_path.parent.mkdir(parents=True, exist_ok=True)
    calib_loader = build_calibration_loader(
        config=config,
        input_channels=args.model_input_channels,
        required_len=max(args.calib_steps * batch_size, 1),
        batch_size=batch_size,
    )
    espdl_quantize_onnx(
        onnx_import_file=str(onnx_path),
        espdl_export_file=str(espdl_path),
        calib_dataloader=calib_loader,
        calib_steps=args.calib_steps,
        input_shape=[1, args.model_input_channels, int(config["input_size"]), int(config["input_size"])],
        target=args.target,
        num_of_bits=args.bits,
        device=device,
        error_report=args.error_report,
        skip_export=False,
        export_config=True,
        export_test_values=args.export_test_values,
        verbose=args.verbose,
    )
    convert_to_external_data(onnx_path)
    validate_onnx_ops(onnx_path)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    weights_path = args.weights_path.expanduser().resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file was not found: {weights_path}")

    config_path = resolve_config_path(weights_path, args.config_path)
    config = load_config(config_path)
    input_size = int(config["input_size"])

    wrapper = build_wrapper(
        architecture=args.architecture,
        config=config,
        weights_path=weights_path,
        device=device,
        score_mode=args.score_mode,
        model_input_channels=args.model_input_channels,
    )
    onnx_path, espdl_path = resolve_output_paths(args=args, config=config, weights_path=weights_path)

    export_onnx_external(
        wrapper=wrapper,
        onnx_path=onnx_path,
        input_channels=args.model_input_channels,
        input_size=input_size,
        device=device,
        opset=args.opset,
    )
    export_espdl(
        onnx_path=onnx_path,
        espdl_path=espdl_path,
        config=config,
        args=args,
        device=device,
    )

    print("=== Student autoencoder score wrapper ESPDL exported ===")
    print(f"Architecture: {args.architecture}")
    print(f"Score mode: {args.score_mode}")
    print(f"Config: {config_path}")
    print(f"Weights: {weights_path}")
    print(f"ONNX: {onnx_path}")
    print(f"ONNX data: {onnx_path.with_suffix(onnx_path.suffix + '.data')}")
    print(f"ESPDL: {espdl_path}")
    print(f"Input shape: [1, {args.model_input_channels}, {input_size}, {input_size}]")
    print(f"Target: {args.target}, bits: {args.bits}")


if __name__ == "__main__":
    main()
