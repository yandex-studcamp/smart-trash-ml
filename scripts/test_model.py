import argparse
import importlib
import os

import torch
from torch.utils.data import DataLoader

from src.data.torch_dataset import TrashDataset
from src.testers.tester import Tester
from src.utils.metrics import ClassificationMetricsCalculator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Имя конфига (без .py)")
    parser.add_argument("--exp_name", type=str, required=True, help="Имя эксперимента")
    parser.add_argument("--weights", type=str, default="best_model.pth", help="Имя файла весов")
    args = parser.parse_args()

    # Пути
    exp_dir = os.path.join("experiments", args.exp_name)
    weights_path = os.path.join(exp_dir, "weights", args.weights)

    # Загрузка конфига
    config_module = importlib.import_module(f"configs.{args.config}")
    cfg = config_module.TrainConfig()

    print(f"=== Тестирование эксперимента: {args.exp_name} ===")
    print(f"Загрузка весов из: {weights_path}")

    # Датасет и Лоадер
    test_dataset = TrashDataset(cfg.test_csv, cfg.img_dir, transform=cfg.get_valid_transforms())
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    # Модель
    model = cfg.get_model()
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Файл весов не найден: {weights_path}")

    model.load_state_dict(torch.load(weights_path, map_location=cfg.device))
    model.to(cfg.device)

    # Тестирование
    metric_calc = ClassificationMetricsCalculator()
    tester = Tester(
        dataloader=test_loader,
        metric_calc=metric_calc,
        device=cfg.device
    )

    results = tester.test(model)

    # Прямой вывод результатов
    print("\n=== Итоговые результаты тестирования ===")
    for metric, value in results.items():
        print(f"{metric.capitalize()}: {value:.4f}")


if __name__ == "__main__":
    main()
