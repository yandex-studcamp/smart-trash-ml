import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.trainers.validator import Validator
from src.utils.plotter import MetricsPlotter
from src.utils.saver import ModelSaver


class Trainer:
    def __init__(self, config, validator: Validator, saver: ModelSaver, plotter: MetricsPlotter) -> None:
        self.config = config
        self.validator = validator
        self.saver = saver
        self.plotter = plotter
        self.device = config.device

    def train(self, model: nn.Module, train_loader: DataLoader) -> None:
        model = model.to(self.device)
        optimizer = self.config.get_optimizer(model)
        criterion = self.config.get_criterion()

        train_losses = []
        val_losses = []
        val_metrics_history = []

        for epoch in range(1, self.config.epochs + 1):
            print(f"\n--- Эпоха {epoch}/{self.config.epochs} ---")
            model.train()
            train_loss = 0.0

            # Прогресс-бар для батчей
            pbar = tqdm(train_loader, desc="Обучение", leave=False)
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                # Обновляем лосс прямо в прогресс-баре
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            avg_train_loss = train_loss / len(train_loader)
            print(f"Train Loss: {avg_train_loss:.4f}")

            # Валидация
            val_loss, val_metrics = self.validator.validate(model)

            # Сохраняем историю для отрисовки
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            val_metrics_history.append(val_metrics)

            # Сохранение модели (передаем метрики, saver сам достанет f1)
            self.saver.save(model, val_metrics, epoch)

            # Обновляем графики (перезаписываем их после каждой эпохи)
            self.plotter.plot_and_save(train_losses, val_losses, val_metrics_history)

        print("\nОбучение успешно завершено!")
