import os

import torch
import torch.nn as nn


class ModelSaver:
    def __init__(self, save_dir: str, monitor_metric: str = "f1") -> None:
        self.save_dir = save_dir
        self.monitor_metric = monitor_metric
        self.best_metric_value = 0.0

    def save(self, model: nn.Module, metrics: dict) -> None:
        latest_path = os.path.join(self.save_dir, "latest_model.pth")
        torch.save(model.state_dict(), latest_path)

        current_metric = metrics.get(self.monitor_metric, 0.0)

        if current_metric > self.best_metric_value:
            self.best_metric_value = current_metric
            best_path = os.path.join(self.save_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"*** Сохранена новая лучшая модель! ({self.monitor_metric}: {current_metric:.4f}) ***")
