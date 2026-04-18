import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class ClassificationMetricsCalculator:
    def __init__(self) -> None:
        self.reset()
        self.all_preds = []
        self.all_targets = []

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        _, predicted = torch.max(preds.data, 1)
        # Сохраняем предсказания и таргеты для подсчета метрик в конце эпохи
        self.all_preds.extend(predicted.cpu().numpy())
        self.all_targets.extend(targets.cpu().numpy())

    def compute(self) -> dict:
        acc = accuracy_score(self.all_targets, self.all_preds)
        # average='macro' считает метрику для каждого класса и берет среднее (хорошо для дисбаланса)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.all_targets, self.all_preds, average='macro', zero_division=0
        )
        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def reset(self) -> None:
        self.all_preds = []
        self.all_targets = []
