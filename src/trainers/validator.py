import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.metrics import ClassificationMetricsCalculator


class Validator:
    def __init__(self, dataloader: DataLoader, criterion: nn.Module, metric_calc: ClassificationMetricsCalculator,
                 device: str) -> None:
        self.dataloader = dataloader
        self.criterion = criterion
        self.metric_calc = metric_calc
        self.device = device

    def validate(self, model: nn.Module) -> tuple:
        model.eval()
        self.metric_calc.reset()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in self.dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
                self.metric_calc.update(outputs, targets)

        metrics = self.metric_calc.compute()
        avg_loss = val_loss / len(self.dataloader)

        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"Validation Loss: {avg_loss:.4f} | {metrics_str}")

        return avg_loss, metrics
