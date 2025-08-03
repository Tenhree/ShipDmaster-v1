import os
import re
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def get_cbvalue(filename):
    pattern = r"Cb=(\d+\.\d+)"
    match = re.search(pattern, filename)

    if match:
        cb_value = float(match.group(1))  
    else:
        cb_value = 0
    return torch.tensor(cb_value, dtype=torch.float32)

class CsvHullDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.file_paths = []
        self.labels = []
        self.transform = transform
        self.label_encoder = LabelEncoder()

        # 遍历目录
        for class_dir in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path):
                for file in os.listdir(class_path):
                    if file.endswith(".csv"):
                        self.file_paths.append(os.path.join(class_path, file))
                        self.labels.append(class_dir)


        self.labels = self.label_encoder.fit_transform(self.labels)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 加载csv为numpy数组
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        try:
            data = pd.read_csv(file_path, header=None).values.astype(np.float32).round(5)
            data = data[:,:2]
            # print(file_path)
            # data=data.reshape(40,10,2)
            data = data.reshape(40, 40, 2)
            data = data.transpose(2, 0, 1)
            ship_Cb = get_cbvalue(file_path)
        except Exception as e:
            raise RuntimeError(f"Error reading {file_path}: {e}")


        if self.transform:
            data = self.transform(data)


        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)  # [1, H, W]

        return torch.tensor(data), label, ship_Cb