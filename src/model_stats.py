from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


@dataclass
class ModelStats:
    model_name: str
    num_parameters: int
    trainable_parameters: int
    estimated_weight_size_mb: float
    state_dict_size_mb: float
    onnx_size_mb: float
    onnx_data_size_mb: float
    onnx_total_size_mb: float
    onnx_path: str
    onnx_data_path: str
    espdl_size_mb: float | None
    espdl_path: str


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_weight_size_mb(model: nn.Module) -> float:
    bytes_total = sum(p.numel() * p.element_size() for p in model.parameters())
    return bytes_to_mb(bytes_total)


def state_dict_size_mb(model: nn.Module, save_path: Path) -> float:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    return bytes_to_mb(save_path.stat().st_size)


def bytes_to_mb(bytes_count: int) -> float:
    return bytes_count / (1024 * 1024)
