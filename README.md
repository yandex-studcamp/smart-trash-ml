# Smart Trash ML

Модуль машинного обучения для проекта умной мусорки.

## Быстрый старт

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/yandex-studcamp/smart-trash-ml.git
   cd smart-trash-ml
   ```

2. Синхронизируйте зависимости:
   Убедитесь, что у вас установлен [uv](https://github.com/astral-sh/uv).
   ```bash
   uv sync
   ```

Модуль машинного обучения для системы классификации мусора.

## Структура проекта
* `configs/`: Конфигурационные файлы (гиперпараметры, пути).
* `data/`: Исходные данные (игнорируется Git).
* `docs/`: Документация по runtime-пайплайну и рабочим сценариям.
* `experiments/`: Артефакты обучения (логи, графики, веса моделей).
* `scripts/`: Скрипты для подготовки данных и запуска обучения/тестирования.
* `src/`: Основная логика (модели, датасеты, тренеры, утилиты).

## Runtime Presence Pipeline

В репозитории есть отдельный rule-based runtime-пайплайн для детекции появления объекта в фиксированном ROI перед вызовом 3-class classifier.

Кратко:
- detector работает по схеме `ROI -> grayscale -> resize -> GaussianBlur -> absdiff -> threshold -> morphology -> foreground_ratio -> brightness check -> hysteresis gate`;
- classifier вызывается не на каждом кадре, а один раз на объект;
- калибровка и replay выполняются локально на ноутбуке через отдельные desktop-скрипты;
- при переносе на ESP desktop debug-поля и утилиты не являются частью embedded API.

Подробное описание, калибровка, replay и заметки по переносу на ESP:
[docs/runtime_presence_pipeline.md](docs/runtime_presence_pipeline.md)

## Использование

Для запуска скриптов используйте команду `uv run`:
```bash
uv run -m папка.имя_скрипта
```

## Быстрый старт

### 1. Подготовка данных
Сначала необходимо обработать изображения и сгенерировать файлы разметки:
```bash
# Ресайз и перемещение картинок
uv run -m scripts.process_images

# Генерация train/val/test сплитов
uv run -m scripts.generate_splits
```

### 2. Обучение модели
Запуск обучения с использованием выбранного конфига:
```bash
uv run -m scripts.train_model --config train_config --exp_name my_first_experiment
```
*   `--config`: имя файла из папки `configs/` (без расширения).
*   `--exp_name`: уникальное имя для папки эксперимента.
*   **Результат:** В `experiments/my_first_experiment/` появятся веса (`weights/`), графики метрик и копия конфига (`artifacts/`).

### 3. Тестирование
Проверка обученной модели на тестовой выборке:
```bash
uv run -m scripts.test_model --config train_config --exp_name my_first_experiment --weights best_model.pth
```

## Работа с экспериментами
* Все результаты обучения автоматически сохраняются в `experiments/`.
* После каждой эпохи скрипт обновляет графики `loss_history.png` и `metrics_history.png` в папке `artifacts/`.
* Используйте `config.json` внутри папки эксперимента, чтобы вспомнить настройки обучения.

## Разработка
* **Новая модель:** добавьте класс в `src/models/` и метод инициализации в `configs/train_config.py`.
* **Новые метрики:** добавьте метод в `ClassificationMetricsCalculator` (`src/utils/metrics.py`).

## Данные

Папка `data/` исключена из отслеживания Git. 
Поместите необходимые датасеты в эту директорию перед началом обучения.