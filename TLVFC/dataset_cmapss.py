import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class CMAPSSDataset(Dataset):
    def __init__(self, data_path, rul_path=None, sequence_length=30, mode="train"):
        self.sequence_length = sequence_length
        self.mode = mode

        column_names = ["unit", "cycle"] + [f"op_setting_{i}" for i in range(1, 4)] + \
                       [f"sensor_{i}" for i in range(1, 22)]
        df = pd.read_csv(data_path, sep="\s+", header=None, names=column_names)

        selected_sensors = [2, 4, 7, 8, 11, 12, 13, 15, 17, 20, 21]
        sensor_cols = [f"sensor_{i}" for i in selected_sensors]
        df = df[["unit", "cycle"] + sensor_cols]

        df[sensor_cols] = (df[sensor_cols] - df[sensor_cols].mean()) / df[sensor_cols].std()

        self.samples = []
        grouped = df.groupby("unit")

        if mode == "train":
            for unit_id, group in grouped:
                data = group[sensor_cols].values
                for i in range(len(data) - sequence_length + 1):
                    seq = data[i:i + sequence_length]
                    rul = len(data) - i - sequence_length
                    self.samples.append((seq, rul))
        elif mode == "test":
            rul_labels = pd.read_csv(rul_path, header=None).values.flatten()
            for unit_id, group in grouped:
                data = group[sensor_cols].values
                seq = data[-sequence_length:]
                rul = rul_labels[unit_id - 1]
                self.samples.append((seq, rul))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, rul = self.samples[idx]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(rul, dtype=torch.float32)
