import os

import pandas as pd
from sklearn.model_selection import train_test_split

from configs.data_config import DIR_LABELS, DIR_IMAGES, CLASS_IDS, TEST_RATIO, VALID_RATIO


def main():
    os.makedirs(DIR_LABELS, exist_ok=True)
    data_records = []

    for class_name in os.listdir(DIR_IMAGES):
        class_dir = DIR_IMAGES / class_name
        if not class_dir.is_dir(): continue

        for file_name in os.listdir(class_dir):
            rel_path = f"images/{class_name}/{file_name}"
            data_records.append({
                'file_path': rel_path,
                'label_name': class_name,
                'label_id': CLASS_IDS.get(class_name, 2)
            })

    df = pd.DataFrame(data_records)

    # Стратифицированное разбиение
    test_val_ratio = VALID_RATIO + TEST_RATIO
    train_df, temp_df = train_test_split(
        df, test_size=test_val_ratio, random_state=42, stratify=df['label_id']
    )

    valid_df, test_df = train_test_split(
        temp_df, test_size=(TEST_RATIO / test_val_ratio), random_state=42, stratify=temp_df['label_id']
    )

    train_df.to_csv(DIR_LABELS / 'train.csv', index=False)
    valid_df.to_csv(DIR_LABELS / 'valid.csv', index=False)
    test_df.to_csv(DIR_LABELS / 'test.csv', index=False)

    print(f"Сплиты созданы в {DIR_LABELS}")
    print(f"Train: {len(train_df)} | Valid: {len(valid_df)} | Test: {len(test_df)}")


if __name__ == "__main__":
    main()
