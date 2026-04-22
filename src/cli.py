import argparse

from model_zoo import SUPPORTED_MOBILENETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export MobileNet models to ONNX/ESPDL with 3-class classifier heads."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Model names to export (space-separated). "
            f"Available: {', '.join(SUPPORTED_MOBILENETS)}"
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export all supported MobileNet models.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=3,
        help="Number of classes for classifier head.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=96,
        help="Square input size for dummy tensor during ONNX export.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Dummy batch size for ONNX export.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Export ONNX with dynamic batch axis.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/exports",
        help="Root directory where exported files and report are saved.",
    )
    parser.add_argument(
        "--export-formats",
        nargs="+",
        choices=["onnx", "espdl"],
        default=["onnx"],
        help="Export formats. Use 'onnx', 'espdl', or both.",
    )
    parser.add_argument(
        "--espdl-command",
        default=None,
        help=(
            "Optional command template for ONNX -> ESPDL conversion. "
            "Placeholders: {input_onnx}, {input_onnx_data}, {output_espdl}, "
            "{model_name}, {calibration_dir}, {quantization}, {image_size}, {num_classes}."
        ),
    )
    parser.add_argument(
        "--espdl-backend",
        choices=["esp-ppq", "command"],
        default="esp-ppq",
        help="Backend for ESPDL export.",
    )
    parser.add_argument(
        "--calibration-dir",
        default="",
        help="Path to calibration data directory for ESP-PPQ or command-based export.",
    )
    parser.add_argument(
        "--espdl-quantization",
        default="int8",
        help="Quantization mode for ESPDL export.",
    )
    parser.add_argument(
        "--espdl-target",
        default="c",
        help="ESP-PPQ target. Use 'c' for ESP32.",
    )
    parser.add_argument(
        "--espdl-calib-steps",
        type=int,
        default=32,
        help="Number of calibration steps for ESP-PPQ.",
    )
    parser.add_argument(
        "--espdl-batch-size",
        type=int,
        default=1,
        help="Calibration batch size for ESP-PPQ.",
    )
    parser.add_argument(
        "--espdl-device",
        default="cuda",
        help="Device used by ESP-PPQ, e.g. 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--espdl-export-test-values",
        action="store_true",
        help="Ask ESP-PPQ to save test input/output values into exported model artifacts.",
    )
    return parser.parse_args()
