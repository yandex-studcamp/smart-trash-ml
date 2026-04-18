import argparse
import importlib
import os

from torch.utils.data import DataLoader

from src.data.torch_dataset import TrashDataset
from src.trainers.trainer import Trainer
from src.trainers.validator import Validator
from src.utils.metrics import ClassificationMetricsCalculator
from src.utils.plotter import MetricsPlotter
from src.utils.saver import ModelSaver


def setup_experiment(exp_name: str) -> tuple[str, str]:
    base_dir = os.path.join("experiments", exp_name)
    weights_dir = os.path.join(base_dir, "weights")
    artifacts_dir = os.path.join(base_dir, "artifacts")

    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    return weights_dir, artifacts_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Имя файла конфига (без .py)")
    parser.add_argument("--exp_name", type=str, required=True, help="Уникальное имя эксперимента")
    args = parser.parse_args()

    # Загрузка конфига
    config_module = importlib.import_module(f"configs.{args.config}")
    cfg = config_module.TrainConfig()

    # Подготовка папок
    weights_dir, artifacts_dir = setup_experiment(args.exp_name)
    cfg.save_to_json(os.path.join("experiments", args.exp_name, "config.json"))

    print(f"=== Запуск эксперимента: {args.exp_name} ===")
    print(f"Устройство: {cfg.device}")

    # Инициализация Датасетов
    train_dataset = TrashDataset(cfg.train_csv, cfg.img_dir, transform=cfg.get_train_transforms())
    valid_dataset = TrashDataset(cfg.valid_csv, cfg.img_dir, transform=cfg.get_valid_transforms())

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False)

    # Инициализация компонентов
    model = cfg.get_model()
    metric_calc = ClassificationMetricsCalculator()
    saver = ModelSaver(save_dir=weights_dir, monitor_metric="f1")  # Следим за F1-score
    plotter = MetricsPlotter(save_dir=artifacts_dir)

    validator = Validator(
        dataloader=valid_loader,
        criterion=cfg.get_criterion(),
        metric_calc=metric_calc,
        device=cfg.device
    )

    trainer = Trainer(
        config=cfg,
        validator=validator,
        saver=saver,
        plotter=plotter
    )

    trainer.train(model, train_loader)


if __name__ == "__main__":
    main()
