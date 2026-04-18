#!/usr/bin/env python3
"""
Скрипт для подсчёта параметров модели PyTorch и оценки пикового RAM на ESP32.
Запуск: uv run -m scripts.count_params --path /путь/до/модели.pt [--model_type baseline|mobilenetv2]
"""
import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn

# Добавляем корень проекта в sys.path для импорта моделей
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.models import get_model


def count_parameters_from_state_dict(state_dict: dict) -> int:
    total = 0
    for tensor in state_dict.values():
        if isinstance(tensor, torch.Tensor):
            total += tensor.numel()
    return total


def count_parameters_from_model(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_file_size_mb(file_path: Path) -> float:
    return os.path.getsize(file_path) / (1024 * 1024)


def estimate_peak_ram(
    model: nn.Module,
    input_shape: tuple = (3, 96, 96),  # (C, H, W) в PyTorch (NCHW)
    quantized: bool = True,
) -> dict:
    bytes_per_element = 1 if quantized else 4
    dtype_name = "INT8" if quantized else "FP32"

    c, h, w = input_shape
    input_size_bytes = h * w * c * bytes_per_element

    device = next(model.parameters()).device
    dummy_input = torch.randn(1, c, h, w).to(device)

    shapes = []

    def hook(module, inp, out):
        # Обрабатываем случай, когда out — тензор или кортеж
        if isinstance(out, torch.Tensor):
            t = out
        elif isinstance(out, (tuple, list)):
            t = out[0]
        else:
            return

        # Проверяем размерность
        if t.dim() == 4:
            # NCHW -> (H, W, C)
            n, c_out, h_out, w_out = t.shape
            shapes.append((h_out, w_out, c_out))
        elif t.dim() == 2:
            # Линейный слой: (N, features)
            # Для оценки RAM считаем как 1x1xC
            n, features = t.shape
            shapes.append((1, 1, features))
        else:
            # Игнорируем нестандартные размерности
            pass

    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.AvgPool2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook))

    model.eval()
    with torch.no_grad():
        _ = model(dummy_input)

    for hk in hooks:
        hk.remove()

    peak_bytes = input_size_bytes
    prev_bytes = input_size_bytes
    for shape in shapes:
        h_out, w_out, c_out = shape
        out_bytes = h_out * w_out * c_out * bytes_per_element
        simultaneous = prev_bytes + out_bytes
        if simultaneous > peak_bytes:
            peak_bytes = simultaneous
        prev_bytes = out_bytes

    peak_bytes = int(peak_bytes * 1.1)

    return {
        "dtype": dtype_name,
        "bytes": peak_bytes,
        "kb": peak_bytes / 1024,
        "input_bytes": input_size_bytes,
        "input_kb": input_size_bytes / 1024,
        "shapes": shapes,
    }


def main():
    parser = argparse.ArgumentParser(description="Подсчёт параметров PyTorch модели и оценка RAM")
    parser.add_argument(
        "--path", "-p",
        type=str,
        required=True,
        help="Путь к файлу модели (.pt, .pth)"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="baseline",
        choices=["baseline", "mobilenetv2"],
        help="Тип архитектуры модели (по умолчанию baseline)"
    )
    args = parser.parse_args()

    model_path = Path(args.path).resolve()
    if not model_path.exists():
        print(f"❌ Файл не найден: {model_path}")
        return

    # Фиксированный размер входа (как при обучении)
    INPUT_SHAPE = (3, 96, 96)  # C, H, W

    print(f"📁 Анализ модели: {model_path}")
    print(f"📦 Размер файла: {get_file_size_mb(model_path):.2f} МБ")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    total_params = 0
    model = None

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
        total_params = count_parameters_from_model(model)
        model_type = "полная модель (nn.Module)"
    elif isinstance(checkpoint, dict):
        # Если загружен state_dict, создаём модель нужной архитектуры и загружаем веса
        model = get_model(args.model_type, num_classes=3)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        total_params = count_parameters_from_state_dict(state_dict)
        model_type = f"state_dict -> {args.model_type}"
    else:
        print("❌ Неизвестный формат файла. Ожидался nn.Module или словарь.")
        return

    print(f"🧠 Тип данных: {model_type}")
    print(f"🔢 Количество параметров: {total_params:,}")

    int8_size_kb = (total_params * 1) / 1024

    print("\n📊 Размер весов:")
    print(f"   - INT8 (квантованная модель .espdl): ~{int8_size_kb:.1f} КБ")

    # Оценка RAM (теперь model всегда определён)
    print("\n🧮 Оценка пикового потребления RAM на ESP32 (при инференсе):")
    est_int8 = estimate_peak_ram(model, INPUT_SHAPE, quantized=True)
    print(f"\n   🔹 Квантованная модель (INT8):")
    print(f"      - Входной буфер: {est_int8['input_kb']:.1f} КБ")
    print(f"      - Пиковое RAM (активации + запас 10%): ~{est_int8['kb']:.1f} КБ")
    print(f"      - Общий объём (веса + пиковое RAM): ~{int8_size_kb + est_int8['kb']:.1f} КБ")



if __name__ == "__main__":
    main()