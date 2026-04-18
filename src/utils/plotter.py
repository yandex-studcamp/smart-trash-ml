import os

import matplotlib.pyplot as plt


class MetricsPlotter:
    def __init__(self, save_dir: str) -> None:
        self.save_dir = save_dir

    def plot_and_save(self, train_losses: list, val_losses: list, val_metrics: list) -> None:
        # График лоссов
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss', marker='o')
        plt.plot(val_losses, label='Validation Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'loss_history.png'))
        plt.close()

        # График метрик
        plt.figure(figsize=(10, 5))
        epochs = range(1, len(val_metrics) + 1)

        # Динамически достаем все метрики из словаря
        for metric_name in val_metrics[0].keys():
            values = [m[metric_name] for m in val_metrics]
            plt.plot(epochs, values, label=metric_name.capitalize(), marker='o')

        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Validation Metrics')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'metrics_history.png'))
        plt.close()

        print(f"Графики сохранены в {self.save_dir}")
