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
    def __init__(self, data_dir, fault_range_file,
                 window_size=60, stride=20,
                 split: str = "train",          # 标记当前数据集：'train'/'val'/'test'
                 split_by_time: bool = False,
                 ratios=(0.6, 0.2, 0.2) ):
        self.samples = []
        self.labels = []
        self.label_map = {}
        self.window_size = window_size
        self.stride = stride
        # === 新增：把这三个入参保存到对象上，后面会用到 ===
        self.split = split  # 'train' / 'val' / 'test'，标记当前数据集是哪一段
        self.split_by_time = split_by_time  # 是否启用“按时间切分 6:2:2”的开关（默认 False）
        self.ratios = ratios  # (0.6, 0.2, 0.2) 三段的比例

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
                T = len(data) #得到当前这段时序的总长度（行数）
                if T < self.window_size:
                    # 行数少于一个窗口，跳过本文件
                    continue

                # === 新增：如果开启 split_by_time，就在单个 Excel 内按 6:2:2 切出本 split 的“时间段” ===
                if self.split_by_time:
                    r_train, r_val, r_test = self.ratios  #读取比例（默认 0.6/0.2/0.2）
                    idx1 = int(T * r_train)  # 前 60% 的分界点
                    idx2 = int(T * (r_train + r_val))  # 前 60% + 中 20% 的分界点

                    #根据构造数据集时传入的 split 选择单个 Excel 的“前 60% / 中 20% / 后 20%”
                    if self.split == "train":
                        data_seg = data[:idx1]  # 取前 60%
                    elif self.split == "val":
                        data_seg = data[idx1:idx2]  # 取中间 20%
                    elif self.split == "test":
                        data_seg = data[idx2:]  # 取最后 20%
                    else:
                        raise ValueError("split must be 'train'/'val'/'test' when split_by_time=True")
                else:
                    # 默认（不开开关）：保持你的旧行为——整段 data 用来滑窗
                    data_seg = data

                # === 下面改成对 data_seg 做滑窗（把你原先 for 循环里的 data 替换为 data_seg）===
                for i in range(0, len(data_seg) - self.window_size + 1, self.stride):
                    window = data_seg[i:i + self.window_size]  # [window_size, 8]
                    self.samples.append((torch.tensor(window, dtype=torch.float32), label))
                    self.labels.append(label)

                # 滑窗
                #for i in range(0, len(data) - window_size + 1, stride):
                    #window = data[i:i + window_size]
                    #self.samples.append((torch.tensor(window, dtype=torch.float32), label))
                    #self.labels.append(label)

            # ✅ 文件夹读取完之后，统计该类的样本数量
            count_for_label = sum(1 for l in self.labels if l == label)
            print(f"标签 {label}（{folder}）: {count_for_label} 条样本")

        print(f"✅ 样本加载完成，总数: {len(self.samples)}，每个样本维度: [{window_size}, 8]")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


