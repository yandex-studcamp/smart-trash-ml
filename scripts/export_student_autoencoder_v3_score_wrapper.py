from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.student_autoencoder_score_wrapper import StudentAutoencoderScoreWrapper
from src.models.student_autoencoder_v3 import StudentAutoencoderV3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a score-only wrapper around the StudentAutoencoderV3 model and save it "
            "as a TorchScript .pt model."
        ),
    )
    parser.add_argument(
        "--weights_path",
        type=Path,
        required=True,
        help="Path to the trained autoencoder weights (.pth).",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path where the wrapped TorchScript model (.pt) will be saved.",
    )
    parser.add_argument(
        "--config_path",
        type=Path,
        default=None,
        help="Optional path to experiment config.json. If omitted, it is inferred from weights_path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for loading and tracing the model, usually `cpu`.",
    )
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
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def infer_v3_architecture_from_state_dict(state_dict: dict[str, torch.Tensor]) -> tuple[tuple[int, int, int, int], int]:
    channels = (
        int(state_dict["stem.0.0.weight"].shape[0]),
        int(state_dict["encoder.0.downsample.0.weight"].shape[0]),
        int(state_dict["encoder.1.downsample.0.weight"].shape[0]),
        int(state_dict["encoder.2.downsample.0.weight"].shape[0]),
    )
    bottleneck_channels = int(state_dict["encoder.3.downsample.0.weight"].shape[0])
    return channels, bottleneck_channels


def build_wrapper_model(config: dict[str, Any], weights_path: Path, device: str) -> StudentAutoencoderScoreWrapper:
    state_dict = torch.load(weights_path, map_location=device)
    channels, bottleneck_channels = infer_v3_architecture_from_state_dict(state_dict)

    autoencoder = StudentAutoencoderV3(
        in_channels=int(config["input_channels"]),
        channels=channels,
        bottleneck_channels=bottleneck_channels,
    )
    autoencoder.load_state_dict(state_dict, strict=True)
    autoencoder.eval()

    input_size = int(config["input_size"])
    wrapper = StudentAutoencoderScoreWrapper(
        autoencoder=autoencoder,
        input_height=input_size,
        input_width=input_size,
        pixel_topk_ratio=float(config.get("pixel_topk_ratio", 0.0)),
        pixel_topk_weight=float(config.get("pixel_topk_weight", 0.0)),
        use_stable_spatial_mask=bool(config.get("use_stable_spatial_mask", False)),
        stable_mask_top_fraction=float(config.get("stable_mask_top_fraction", 0.0)),
        stable_mask_bottom_fraction=float(config.get("stable_mask_bottom_fraction", 0.0)),
        stable_mask_left_fraction=float(config.get("stable_mask_left_fraction", 0.0)),
        stable_mask_right_fraction=float(config.get("stable_mask_right_fraction", 0.0)),
    )
    return wrapper.to(device).eval()


def save_torchscript_wrapper(
    wrapper: StudentAutoencoderScoreWrapper,
    *,
    output_path: Path,
    input_size: int,
    device: str,
) -> None:
    example_input = torch.rand((1, 1, input_size, input_size), device=device, dtype=torch.float32)

    with torch.inference_mode():
        eager_output = wrapper(example_input)
        traced = torch.jit.trace(wrapper, example_input, strict=True)
        frozen = torch.jit.freeze(traced.eval())
        scripted_output = frozen(example_input)

    if not torch.allclose(eager_output, scripted_output, atol=1e-6, rtol=1e-5):
        raise RuntimeError("TorchScript wrapper output does not match eager output.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(frozen, output_path)


def main() -> None:
    args = parse_args()
    device = args.device.strip().lower()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    weights_path = args.weights_path.expanduser().resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file was not found: {weights_path}")

    config_path = resolve_config_path(weights_path, args.config_path)
    config = load_config(config_path)
    wrapper = build_wrapper_model(config=config, weights_path=weights_path, device=device)

    output_path = args.output_path.expanduser().resolve()
    save_torchscript_wrapper(
        wrapper,
        output_path=output_path,
        input_size=int(config["input_size"]),
        device=device,
    )

    state_dict = torch.load(weights_path, map_location="cpu")
    channels, bottleneck_channels = infer_v3_architecture_from_state_dict(state_dict)

    print("=== Student autoencoder v3 score wrapper exported ===")
    print(f"Config: {config_path}")
    print(f"Weights: {weights_path}")
    print(f"Output: {output_path}")
    print(f"Channels: {channels}")
    print(f"Bottleneck channels: {bottleneck_channels}")
    print(f"Pixel top-k ratio: {config.get('pixel_topk_ratio', 0.0)}")
    print(f"Pixel top-k weight: {config.get('pixel_topk_weight', 0.0)}")
    print(f"Input size: {config['input_size']}x{config['input_size']}")


if __name__ == "__main__":
    main()
