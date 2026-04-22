import csv
from argparse import Namespace
from pathlib import Path

from espdl_exporter import export_to_espdl
from model_stats import (
    ModelStats,
    count_parameters,
    count_trainable_parameters,
    estimate_weight_size_mb,
    state_dict_size_mb,
)
from model_zoo import build_model, list_supported_models
from onnx_exporter import export_to_onnx


def run_export_pipeline(args: Namespace) -> None:
    model_names = resolve_model_names(args.models, args.all)
    export_formats = set(args.export_formats)
    if "espdl" in export_formats and args.espdl_backend == "command" and not args.espdl_command:
        raise ValueError("--espdl-command is required when --espdl-backend=command.")

    output_dir = Path(args.output_dir)
    report_path = output_dir / "model_report.csv"

    rows: list[ModelStats] = []
    for model_name in model_names:
        model_root = output_dir / model_name
        onnx_dir = model_root / "onnx"
        espdl_dir = model_root / "espdl"
        state_dict_dir = model_root / "state_dict"

        model = build_model(model_name=model_name, num_classes=args.num_classes)
        params = count_parameters(model)
        trainable = count_trainable_parameters(model)
        est_mb = estimate_weight_size_mb(model)
        state_mb = state_dict_size_mb(model, state_dict_dir / f"{model_name}.pt")

        onnx_path = onnx_dir / f"{model_name}.onnx"
        if "onnx" in export_formats or "espdl" in export_formats:
            onnx_path = export_to_onnx(
                model=model,
                output_path=onnx_path,
                image_size=args.image_size,
                batch_size=args.batch_size,
                opset=args.opset,
                dynamic_batch=args.dynamic_batch,
            )
        onnx_size_mb = onnx_path.stat().st_size / (1024 * 1024)
        onnx_data_path = onnx_path.with_suffix(onnx_path.suffix + ".data")
        onnx_data_size_mb = onnx_data_path.stat().st_size / (1024 * 1024)
        espdl_path = ""
        espdl_size_mb: float | None = None
        if "espdl" in export_formats:
            espdl_output_path = espdl_dir / f"{model_name}.espdl"
            espdl_file = export_to_espdl(
                backend=args.espdl_backend,
                model_name=model_name,
                input_onnx_path=onnx_path,
                input_onnx_data_path=onnx_data_path,
                output_espdl_path=espdl_output_path,
                command_template=args.espdl_command,
                calibration_dir=args.calibration_dir,
                quantization=args.espdl_quantization,
                image_size=args.image_size,
                num_classes=args.num_classes,
                calib_steps=args.espdl_calib_steps,
                batch_size=args.espdl_batch_size,
                target=args.espdl_target,
                device=args.espdl_device,
                export_test_values=args.espdl_export_test_values,
            )
            espdl_path = str(espdl_file.resolve())
            espdl_size_mb = espdl_file.stat().st_size / (1024 * 1024)

        rows.append(
            ModelStats(
                model_name=model_name,
                num_parameters=params,
                trainable_parameters=trainable,
                estimated_weight_size_mb=est_mb,
                state_dict_size_mb=state_mb,
                onnx_size_mb=onnx_size_mb,
                onnx_data_size_mb=onnx_data_size_mb,
                onnx_total_size_mb=onnx_size_mb + onnx_data_size_mb,
                onnx_path=str(onnx_path.resolve()),
                onnx_data_path=str(onnx_data_path.resolve()),
                espdl_size_mb=espdl_size_mb,
                espdl_path=espdl_path,
            )
        )
        print_model_summary(rows[-1])

    write_report(rows, report_path)
    print(f"\nReport saved: {report_path.resolve()}")


def resolve_model_names(selected_models: list[str] | None, export_all: bool) -> list[str]:
    available = list_supported_models()
    if export_all:
        return available

    if not selected_models:
        return ["mobilenet_v3_small"]

    unknown = [name for name in selected_models if name not in available]
    if unknown:
        available_str = ", ".join(available)
        unknown_str = ", ".join(unknown)
        raise ValueError(f"Unknown models: {unknown_str}. Available: {available_str}")
    return selected_models


def write_report(rows: list[ModelStats], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model_name",
                "num_parameters",
                "trainable_parameters",
                "estimated_weight_size_mb",
                "state_dict_size_mb",
                "onnx_size_mb",
                "onnx_data_size_mb",
                "onnx_total_size_mb",
                "onnx_path",
                "onnx_data_path",
                "espdl_size_mb",
                "espdl_path",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.model_name,
                    row.num_parameters,
                    row.trainable_parameters,
                    f"{row.estimated_weight_size_mb:.4f}",
                    f"{row.state_dict_size_mb:.4f}",
                    f"{row.onnx_size_mb:.4f}",
                    f"{row.onnx_data_size_mb:.4f}",
                    f"{row.onnx_total_size_mb:.4f}",
                    row.onnx_path,
                    row.onnx_data_path,
                    "" if row.espdl_size_mb is None else f"{row.espdl_size_mb:.4f}",
                    row.espdl_path,
                ]
            )


def print_model_summary(stats: ModelStats) -> None:
    print(f"\nModel: {stats.model_name}")
    print(f"  Params total: {stats.num_parameters:,}")
    print(f"  Params trainable: {stats.trainable_parameters:,}")
    print(f"  Estimated weights: {stats.estimated_weight_size_mb:.2f} MB")
    print(f"  state_dict size: {stats.state_dict_size_mb:.2f} MB")
    print(f"  ONNX proto size: {stats.onnx_size_mb:.2f} MB")
    print(f"  ONNX data size: {stats.onnx_data_size_mb:.2f} MB")
    print(f"  ONNX total size: {stats.onnx_total_size_mb:.2f} MB")
    print(f"  ONNX path: {stats.onnx_path}")
    print(f"  ONNX data path: {stats.onnx_data_path}")
    if stats.espdl_size_mb is not None:
        print(f"  ESPDL size: {stats.espdl_size_mb:.2f} MB")
        print(f"  ESPDL path: {stats.espdl_path}")
