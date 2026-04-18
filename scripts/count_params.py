#!/usr/bin/env python3
"""
Скрипт для подсчёта параметров модели PyTorch и оценки пикового RAM на ESP32.
Запуск: uv run -m scripts.count_params --path /путь/до/модели.pt
"""
import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn


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
    """
    Оценка пикового использования RAM при инференсе.
    Возвращает словарь с размерами в байтах и килобайтах для FP32 и INT8.
    """
    bytes_per_element = 1 if quantized else 4
    dtype_name = "INT8" if quantized else "FP32"

    # Переводим в NHWC для ESP-DL (H, W, C)
    c, h, w = input_shape
    input_size_bytes = h * w * c * bytes_per_element

    # Фиктивный вход для трассировки формы
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, c, h, w).to(device)

    # Регистрируем хук для сбора форм выходов
    shapes = []

    def hook(module, inp, out):
        # out может быть тензором или кортежем (берём первый)
        if isinstance(out, torch.Tensor):
            t = out
        else:
            t = out[0]
        # Формат NCHW -> преобразуем в NHWC
        n, c_out, h_out, w_out = t.shape
        shapes.append((h_out, w_out, c_out))

    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.AvgPool2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook))

    # Прогон модели
    model.eval()
    with torch.no_grad():
        _ = model(dummy_input)

    # Убираем хуки
    for hk in hooks:
        hk.remove()

    # Оценка пикового RAM: ищем максимальную сумму двух последовательных тензоров
    # (входной + выходной) или просто максимальный одиночный тензор.
    peak_bytes = input_size_bytes
    prev_bytes = input_size_bytes
    for shape in shapes:
        h_out, w_out, c_out = shape
        out_bytes = h_out * w_out * c_out * bytes_per_element
        # В худшем случае одновременно держим вход и выход слоя
        simultaneous = prev_bytes + out_bytes
        if simultaneous > peak_bytes:
            peak_bytes = simultaneous
        prev_bytes = out_bytes

    # Добавляем запас 10% на временные буферы и выравнивание
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
        "--input-shape",
        type=str,
        default="3,96,96",
        help="Размер входа в формате C,H,W (по умолчанию 3,96,96)"
    )
    args = parser.parse_args()

    model_path = Path(args.path).resolve()
    if not model_path.exists():
        print(f"❌ Файл не найден: {model_path}")
        return

    # Парсим входную форму
    try:
        c, h, w = map(int, args.input_shape.split(","))
        input_shape = (c, h, w)
    except Exception:
        print("❌ Неверный формат --input-shape. Пример: 3,96,96")
        return

    print(f"📁 Анализ модели: {model_path}")
    print(f"📦 Размер файла: {get_file_size_mb(model_path):.2f} МБ")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    total_params = 0
    model_type = "неизвестно"
    model = None

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
        total_params = count_parameters_from_model(model)
        model_type = "полная модель (nn.Module)"
    elif isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            total_params = count_parameters_from_state_dict(checkpoint["state_dict"])
            model_type = "чекпоинт обучения (state_dict)"
        else:
            total_params = count_parameters_from_state_dict(checkpoint)
            model_type = "state_dict"
    else:
        print("❌ Неизвестный формат файла. Ожидался nn.Module или словарь.")
        return

    print(f"🧠 Тип данных: {model_type}")
    print(f"🔢 Количество параметров: {total_params:,}")

    int8_size_kb = (total_params * 1) / 1024

    print("\n📊 Размер весов:")
    print(f"   - INT8 (квантованная модель .espdl): ~{int8_size_kb:.1f} КБ")
    print(f"   - FP32 (оригинальные веса): ~{float32_size_kb:.1f} КБ")

    # Оценка RAM, если есть полная модель
    if model is not None:
        print("\n🧮 Оценка пикового потребления RAM на ESP32 (при инференсе):")

        # Для INT8
        est_int8 = estimate_peak_ram(model, input_shape, quantized=True)
        print(f"\n   🔹 Квантованная модель (INT8):")
        print(f"      - Входной буфер: {est_int8['input_kb']:.1f} КБ")
        print(f"      - Пиковое RAM (активации + запас 10%): ~{est_int8['kb']:.1f} КБ")
        print(f"      - Общий объём (веса + пиковое RAM): ~{int8_size_kb + est_int8['kb']:.1f} КБ")

    else:
        print("\n⚠️  Для оценки пикового RAM необходима полная модель (nn.Module),")
        print("    а не только state_dict. Сохраните модель целиком через torch.save(model, ...).")


if __name__ == "__main__":
    main()