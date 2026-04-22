import subprocess
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CalibrationImageDataset(Dataset):
    def __init__(self, image_paths: list[Path], image_size: int, required_len: int) -> None:
        self.image_paths = image_paths
        self.required_len = max(required_len, len(image_paths))
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return self.required_len

    def __getitem__(self, index: int) -> torch.Tensor:
        image_path = self.image_paths[index % len(self.image_paths)]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image)


class SyntheticCalibrationDataset(Dataset):
    def __init__(self, image_size: int, required_len: int) -> None:
        self.image_size = image_size
        self.required_len = max(required_len, 1)

    def __len__(self) -> int:
        return self.required_len

    def __getitem__(self, _: int) -> torch.Tensor:
        return torch.randn(3, self.image_size, self.image_size, dtype=torch.float32)


def export_to_espdl(
    *,
    backend: str,
    model_name: str,
    input_onnx_path: Path,
    input_onnx_data_path: Path,
    output_espdl_path: Path,
    command_template: str | None,
    calibration_dir: str,
    quantization: str,
    image_size: int,
    num_classes: int,
    calib_steps: int,
    batch_size: int,
    target: str,
    device: str,
    export_test_values: bool,
) -> Path:
    if backend == "command":
        if not command_template:
            raise ValueError("command_template is required for command backend.")
        return export_to_espdl_via_command(
            model_name=model_name,
            input_onnx_path=input_onnx_path,
            input_onnx_data_path=input_onnx_data_path,
            output_espdl_path=output_espdl_path,
            command_template=command_template,
            calibration_dir=calibration_dir,
            quantization=quantization,
            image_size=image_size,
            num_classes=num_classes,
        )

    if backend == "esp-ppq":
        return export_to_espdl_via_esp_ppq(
            input_onnx_path=input_onnx_path,
            output_espdl_path=output_espdl_path,
            calibration_dir=calibration_dir,
            quantization=quantization,
            image_size=image_size,
            calib_steps=calib_steps,
            batch_size=batch_size,
            target=target,
            device=device,
            export_test_values=export_test_values,
        )

    raise ValueError(f"Unknown ESPDL backend: {backend}")


def export_to_espdl_via_command(
    *,
    model_name: str,
    input_onnx_path: Path,
    input_onnx_data_path: Path,
    output_espdl_path: Path,
    command_template: str,
    calibration_dir: str,
    quantization: str,
    image_size: int,
    num_classes: int,
) -> Path:
    output_espdl_path.parent.mkdir(parents=True, exist_ok=True)
    command = command_template.format(
        input_onnx=input_onnx_path.resolve(),
        input_onnx_data=input_onnx_data_path.resolve(),
        output_espdl=output_espdl_path.resolve(),
        model_name=model_name,
        calibration_dir=calibration_dir,
        quantization=quantization,
        image_size=image_size,
        num_classes=num_classes,
    )
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "ESPDL conversion failed.\n"
            f"Command: {command}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    if not output_espdl_path.exists():
        raise RuntimeError(
            "ESPDL conversion command succeeded, but output file was not created: "
            f"{output_espdl_path.resolve()}"
        )
    return output_espdl_path


def export_to_espdl_via_esp_ppq(
    *,
    input_onnx_path: Path,
    output_espdl_path: Path,
    calibration_dir: str,
    quantization: str,
    image_size: int,
    calib_steps: int,
    batch_size: int,
    target: str,
    device: str,
    export_test_values: bool,
) -> Path:
    from esp_ppq.api import espdl_quantize_onnx

    output_espdl_path.parent.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device)

    if batch_size != 1:
        print(
            f"ESPDL quantization expects batch_size=1. Overriding requested batch_size={batch_size} to 1."
        )
        batch_size = 1

    required_len = max(calib_steps * batch_size, 1)
    calib_dataset = build_calibration_dataset(
        calibration_dir=calibration_dir,
        image_size=image_size,
        required_len=required_len,
    )
    calib_loader = DataLoader(
        dataset=calib_dataset,
        batch_size=max(batch_size, 1),
        shuffle=False,
        drop_last=False,
    )

    num_of_bits = parse_num_bits(quantization)
    espdl_quantize_onnx(
        onnx_import_file=str(input_onnx_path.resolve()),
        espdl_export_file=str(output_espdl_path.resolve()),
        calib_dataloader=calib_loader,
        calib_steps=calib_steps,
        input_shape=[1, 3, image_size, image_size],
        target=target,
        num_of_bits=num_of_bits,
        device=device,
        error_report=False,
        skip_export=False,
        export_config=True,
        export_test_values=export_test_values,
        verbose=0,
    )

    if not output_espdl_path.exists():
        raise RuntimeError(
            "esp-ppq finished without creating ESPDL file: "
            f"{output_espdl_path.resolve()}"
        )
    return output_espdl_path


def parse_num_bits(quantization: str) -> int:
    q = quantization.lower().strip()
    if q in {"int8", "8"}:
        return 8
    if q in {"int16", "16"}:
        return 16
    raise ValueError(
        f"Unsupported quantization mode '{quantization}'. Use int8 or int16."
    )


def build_calibration_dataset(
    *,
    calibration_dir: str,
    image_size: int,
    required_len: int,
) -> Dataset:
    image_paths: list[Path] = []
    calib_path = Path(calibration_dir)
    if calibration_dir and calib_path.exists():
        image_paths = collect_image_paths(calib_path)

    if image_paths:
        return CalibrationImageDataset(
            image_paths=image_paths,
            image_size=image_size,
            required_len=required_len,
        )

    print(
        "Calibration images not found. Falling back to synthetic random tensors. "
        "This is for smoke tests only; real calibration images are recommended."
    )
    return SyntheticCalibrationDataset(image_size=image_size, required_len=required_len)


def collect_image_paths(root: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def resolve_device(device: str) -> str:
    if device.lower() == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU for esp-ppq quantization.")
        return "cpu"
    return device
