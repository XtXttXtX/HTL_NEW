import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
USEFUL_COLUMNS = [
        "功出电压", "功出电流", "主轨入电压", "主轨出电压",
        "小轨入电压", "小轨出电压", "送端分线盘电压", "受端分线盘电压"
    ]

class RailwayDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.label_map = {}
        root_dir = os.path.expanduser(root_dir)
        folders = sorted(os.listdir(root_dir))

        for label, folder in enumerate(folders):
            if folder.startswith('~') or folder.startswith('.'):
                continue
            folder_path = os.path.join(root_dir, folder)
            self.label_map[label] = folder
            for file in os.listdir(folder_path):
                if file.endswith(".xlsx") and not file.startswith('~$'):
                    self.samples.append((os.path.join(folder_path, file), label))

    def __len__(self):
        return len(self.samples)


    #
    # def __getitem__(self, idx):
    #     path, label = self.samples[idx]
    #     df = pd.read_excel(path)
    #
    #     # ✅ 找出所有有用的列名（中文名匹配）
    #     selected_cols = [col for col in df.columns if
    #                      isinstance(col, str) and any(key in col for key in USEFUL_COLUMNS)]
    #
    #     # ✅ 选中的列组成一个 [时间步, 特征数] 的 numpy 数组
    #     signal = df[selected_cols].dropna().astype(np.float32).values
    #
    #     # ✅ 转为 PyTorch tensor
    #     signal = torch.from_numpy(signal)  # shape: [seq_len, 8]
    #     return signal, label


    def __getitem__(self, idx):
        path, label = self.samples[idx]
        df = pd.read_excel(path, sheet_name="data_interpt", engine="openpyxl")


        print(f"\n📄 当前读取文件: {path}")
        print("🧾 所有原始列名:", list(df.columns))

        selected_cols = [col for col in df.columns if
                         isinstance(col, str) and any(key in col for key in USEFUL_COLUMNS)]

        print("✅ 匹配到的有效列名:", selected_cols)

        signal = df[selected_cols].dropna().astype(np.float32).values
        signal = torch.from_numpy(signal)
        return signal, label

if __name__ == "__main__":
    dataset = RailwayDataset("./分类数据")
    print("✅ 成功加载数据集！样本总数：", len(dataset))

    signal, label = dataset[0]
    print("第一个样本的 shape:", signal.shape)
    print("第一个样本的标签:", label)


