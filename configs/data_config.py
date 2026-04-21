from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DIR = DATA_DIR / "raw"

DATASET_NAME = 'union_dataset'

DIR_RESIZED = RAW_DIR / "dataset-resized"
DIR_GARBAGE_12 = RAW_DIR / "garbage_classification_12_classes"
DIR_HF = RAW_DIR / "hf_dataset"
DIR_KERENBERKE = RAW_DIR / "kerenberke"
DIR_REALWASTE = RAW_DIR / "realwaste" / "RealWaste"

RAW_DATASET_DIRS = {
    "dataset-resized": DIR_RESIZED,
    "garbage_classification_12_classes": DIR_GARBAGE_12,
    "hf_dataset": DIR_HF,
    "kerenberke": DIR_KERENBERKE,
    "realwaste": DIR_REALWASTE,
}

DEFAULT_SOURCES = tuple(RAW_DATASET_DIRS.keys())

TARGET_SIZE = (224, 224)

CLASS_MAPPING = {
    "paper": "paper",
    "cardboard": "paper",
    "plastic": "plastic",
    "battery": "other",
    "biological": "other",
    "biodegradable": "other",
    "brown-glass": "other",
    "clothes": "other",
    "food organics": "other",
    "glass": "other",
    "green-glass": "other",
    "metal": "other",
    "miscellaneous trash": "other",
    "shoes": "other",
    "textile trash": "other",
    "trash": "other",
    "vegetation": "other",
    "white-glass": "other",
}

CLASS_IDS = {"paper": 0, "plastic": 1, "other": 2}

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

GDRIVE_FILE_ID = "17zCevNzcD_vU19_d4EIyoMTy8Qj7CyJY"
ARCHIVE_NAME = "dataset.zip"


def get_dataset_dir(dataset_name: str = DATASET_NAME) -> Path:
    return DATA_DIR / dataset_name


def get_images_dir(dataset_name: str = DATASET_NAME) -> Path:
    return get_dataset_dir(dataset_name) / "images"


def get_labels_dir(dataset_name: str = DATASET_NAME) -> Path:
    return get_dataset_dir(dataset_name) / "labels"


DATASET_DIR = get_dataset_dir()
DIR_IMAGES = get_images_dir()
DIR_LABELS = get_labels_dir()
DIR_OUT_IMAGES = DIR_IMAGES
