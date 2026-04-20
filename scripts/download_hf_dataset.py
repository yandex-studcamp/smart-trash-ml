import os
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import login
from tqdm import tqdm

# Залогиниться программно
login(token="ТВОЙ ТОКЕН")

# Настройки
DATASET_NAME = "kdkd1/waste-garbage-management-dataset"
SAVE_DIR = Path("../data/raw/hf_dataset")


def main():
    print(f"Загрузка датасета {DATASET_NAME} из Hugging Face...")
    # Загружаем датасет
    full_dataset = load_dataset(DATASET_NAME)

    print("Доступные части датасета:", full_dataset.keys())

    # Берем 'train' (или любой другой, который есть в списке)
    if 'train' in full_dataset:
        dataset = full_dataset['train']
    else:
        # Если нет 'train', берем первую доступную часть
        first_key = list(full_dataset.keys())[0]
        dataset = full_dataset[first_key]
        print(f"Сплит 'train' не найден, используем '{first_key}'")

    # Теперь у нас есть нужный объект Dataset, у которого ЕСТЬ .features
    features = dataset.features
    if 'label' in features:
        class_names = features['label'].names
    else:
        print("ВНИМАНИЕ: Не найдена колонка 'label'. Проверьте структуру датасета на HF.")
        return

    print(f"Найдено классов: {len(class_names)} ({class_names})")

    # Создаем папки для каждого класса
    for class_name in class_names:
        os.makedirs(SAVE_DIR / class_name, exist_ok=True)

    print(f"Сохранение изображений в {SAVE_DIR}...")

    # Итерируемся по датасету и сохраняем картинки
    for i, item in enumerate(tqdm(dataset)):
        image = item['image']  # PIL Image
        label_id = item['label']
        class_name = class_names[label_id]

        # Конвертируем в RGB, если попалась ч/б или RGBA картинка
        if image.mode != 'RGB':
            image = image.convert('RGB')

        file_path = SAVE_DIR / class_name / f"img_{i:05d}.jpg"
        image.save(file_path, "JPEG", quality=100)

    print("\nГотово! Датасет успешно скачан и структурирован.")


if __name__ == "__main__":
    main()
