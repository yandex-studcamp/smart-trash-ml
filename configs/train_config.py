import json
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torchvision import transforms


# Заглушка, чтобы код не падал. Замени на свой импорт


@dataclass
class TrainConfig:
    # Базовые пути
    train_csv: str = "data/union_dataset/labels/train.csv"
    valid_csv: str = "data/union_dataset/labels/valid.csv"
    test_csv: str = "data/union_dataset/labels/test.csv"
    img_dir: str = "data/union_dataset/"

    # Гиперпараметры
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def save_to_json(self, filepath: str) -> None:
        """Сохраняет только поля датакласса (без методов) в JSON"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=4)

    # --- Фабричные методы (не попадут в JSON) ---
    @staticmethod
    def get_train_transforms() -> transforms.Compose:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def get_valid_transforms() -> transforms.Compose:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def get_model() -> nn.Module:
        return SimpleCNN(num_classes=3)

    @staticmethod
    def get_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.Adam(model.parameters(), lr=self.learning_rate)

    @staticmethod
    def get_criterion() -> nn.Module:
        return nn.CrossEntropyLoss()
