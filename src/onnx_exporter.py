from pathlib import Path

import onnx
import torch
import torch.nn as nn


def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    image_size: int,
    batch_size: int,
    opset: int,
    dynamic_batch: bool,
) -> Path:
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(batch_size, 3, image_size, image_size, dtype=torch.float32)
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        }

    torch.onnx.export(
        model,
        dummy_input,
        output_path.as_posix(),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        export_params=True,
        do_constant_folding=True,
        opset_version=opset,
        dynamo=False,
    )
    convert_to_external_data(output_path)
    return output_path


def convert_to_external_data(output_path: Path) -> None:
    data_path = output_path.with_suffix(output_path.suffix + ".data")
    if data_path.exists():
        data_path.unlink()
    onnx_model = onnx.load(output_path.as_posix())
    onnx.save_model(
        onnx_model,
        output_path.as_posix(),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path.name,
        size_threshold=0,
    )
