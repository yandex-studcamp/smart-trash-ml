import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.metrics import ClassificationMetricsCalculator


class Tester:
    def __init__(self, dataloader: DataLoader, metric_calc: ClassificationMetricsCalculator, device: str) -> None:
        self.dataloader = dataloader
        self.metric_calc = metric_calc
        self.device = device

    def test(self, model: nn.Module) -> dict:
        model.eval()
        self.metric_calc.reset()

        with torch.no_grad():
            for inputs, targets in self.dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                self.metric_calc.update(outputs, targets)

        return self.metric_calc.compute()
