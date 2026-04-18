import zipfile

import gdown

from configs.data_config import DATA_DIR, GDRIVE_FILE_ID, ARCHIVE_NAME


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = DATA_DIR / ARCHIVE_NAME

    print("Скачивание архива...")
    gdown.download(id=GDRIVE_FILE_ID, output=str(archive_path), quiet=False)

    if archive_path.exists():
        print("Распаковка...")
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)

        archive_path.unlink()
        print(f"Датасет успешно загружен и распакован в {DATA_DIR}")
    else:
        print("Ошибка скачивания файла.")


if __name__ == "__main__":
    main()
