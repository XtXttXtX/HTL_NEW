import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

USEFUL_COLUMNS = [
    "功出电压", "功出电流", "主轨入电压", "主轨出电压",
    "小轨入电压", "小轨出电压", "送端分线盘电压", "受端分线盘电压"
]

class WindowedRailwayDataset(Dataset):
    def __init__(self, data_dir, fault_range_file, window_size=60, stride=20):
        self.samples = []
        self.labels = []
        self.label_map = {}
        self.window_size = window_size
        self.stride = stride

        fault_ranges = pd.read_excel(fault_range_file)
        fault_range_dict = {
            str(row['filename']).strip().lower(): (int(row['start']), int(row['end']))
            for _, row in fault_ranges.iterrows()
        }

        folders = sorted(os.listdir(data_dir))
        for label, folder in enumerate(folders):
            folder_path = os.path.join(data_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            self.label_map[label] = folder

            for file in os.listdir(folder_path):
                if not file.endswith(".xlsx") or file.startswith("~$"):
                    continue
                standard_name = str(file).strip().lower()

                if standard_name not in fault_range_dict:
                    print(f"⚠️ 警告：{standard_name} 不在 fault_ranges 中，跳过")
                    continue

                file_path = os.path.join(folder_path, file)
                start, end = fault_range_dict[standard_name]

                df = pd.read_excel(file_path, sheet_name="data_interpt", engine="openpyxl")
                df = df.iloc[start - 1:end]

                selected_cols = [col for col in df.columns if col in USEFUL_COLUMNS]
                data = df[selected_cols].dropna().astype(np.float32).values  # shape: [T, 8]

                # 滑窗
                for i in range(0, len(data) - window_size + 1, stride):
                    window = data[i:i + window_size]
                    self.samples.append((torch.tensor(window, dtype=torch.float32), label))
                    self.labels.append(label)

            # ✅ 文件夹读取完之后，统计该类的样本数量
            count_for_label = sum(1 for l in self.labels if l == label)
            print(f"标签 {label}（{folder}）: {count_for_label} 条样本")

        print(f"✅ 样本加载完成，总数: {len(self.samples)}，每个样本维度: [{window_size}, 8]")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


