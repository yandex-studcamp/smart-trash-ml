import os
import pandas as pd
from PIL import Image
from typing import Callable, Optional, Tuple
from torch.utils.data import Dataset
import torch


class TrashDataset(Dataset):
    def __init__(self, csv_file: str, root_dir: str, transform: Optional[Callable] = None) -> None:
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data_frame)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = os.path.join(self.root_dir, self.data_frame.iloc[idx]['file_path'])
        image = Image.open(img_path).convert('RGB')
        label = int(self.data_frame.iloc[idx]['label_id'])

        if self.transform:
            image = self.transform(image)

        return image, label
