from pathlib import Path

# Директории
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
DIR_RESIZED = DATA_DIR / 'dataset-resized'
DIR_REALWASTE = DATA_DIR / 'realwaste' / 'RealWaste'
DIR_OUT_IMAGES = DATA_DIR / 'union_dataset' / 'images'

# Локальная структура датасета
DATASET_DIR = BASE_DIR / 'data' / 'union_dataset'
DIR_IMAGES = DATA_DIR / 'images'
DIR_LABELS = DATA_DIR / 'labels'

# Конфиг на гугл диске
GDRIVE_FILE_ID = "17zCevNzcD_vU19_d4EIyoMTy8Qj7CyJY"
ARCHIVE_NAME = "dataset.zip"

# В какое разрешение приводить данные
TARGET_SIZE = (224, 224)

# Как кодировать классы
CLASS_MAPPING = {
    'plastic': 'plastic', 'Plastic': 'plastic',
    'paper': 'paper', 'Paper': 'paper',
    'cardboard': 'paper', 'Cardboard': 'paper',
}
CLASS_IDS = {'paper': 0, 'plastic': 1, 'other': 2}

# Настройки разбиения
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15
