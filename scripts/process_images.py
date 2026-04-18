import os
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from configs.data_config import DIR_OUT_IMAGES, DIR_RESIZED, DIR_REALWASTE, TARGET_SIZE, CLASS_MAPPING


def crop_center_and_resize(img, size):
    width, height = img.size
    new_dim = min(width, height)
    left, top = (width - new_dim) / 2, (height - new_dim) / 2
    right, bottom = (width + new_dim) / 2, (height + new_dim) / 2
    return img.crop((left, top, right, bottom)).resize(size, Image.Resampling.LANCZOS)


def main():
    if DIR_OUT_IMAGES.exists():
        shutil.rmtree(DIR_OUT_IMAGES)

    for cls in set(CLASS_MAPPING.values()) | {'other'}:
        os.makedirs(DIR_OUT_IMAGES / cls, exist_ok=True)

    all_files = []
    for ds_name, ds_path in [("resized", DIR_RESIZED), ("realwaste", DIR_REALWASTE)]:
        if not ds_path.exists(): continue
        for root, _, files in os.walk(ds_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_files.append((ds_name, Path(root), file))

    for ds_name, folder_path, file_name in tqdm(all_files, desc="Обработка фото"):
        target_class = CLASS_MAPPING.get(folder_path.name, 'other')
        src_file = folder_path / file_name
        new_name = f"{ds_name}_{folder_path.name}_{file_name}"
        target_file = DIR_OUT_IMAGES / target_class / new_name

        try:
            with Image.open(src_file) as img:
                if img.mode != 'RGB': img = img.convert('RGB')
                crop_center_and_resize(img, TARGET_SIZE).save(target_file, 'JPEG', quality=90)
        except Exception as e:
            print(f"\nОшибка {src_file}: {e}")


if __name__ == "__main__":
    main()
