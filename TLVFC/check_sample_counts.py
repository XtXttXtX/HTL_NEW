import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from collections import Counter
from tqdm import tqdm

class WindowedRailwayDataset(Dataset):
    def __init__(self, data_dir, fault_range_file, window_size=60, stride=20):
        self.samples = []
        self.labels = []
        self.window_size = window_size
        self.stride = stride

        # 加载起止范围文件
        self.range_dict = {}
        df_range = pd.read_excel(fault_range_file)
        for _, row in df_range.iterrows():
            fname = str(row["filename"]).strip().lower()
            self.range_dict[fname] = (int(row["start"]), int(row["end"]))

        # 遍历每类目录
        for class_idx, folder in enumerate(sorted(os.listdir(data_dir))):
            folder_path = os.path.join(data_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            for file in os.listdir(folder_path):
                if not file.endswith(".xlsx") or file.startswith("~$"):
                    continue
                file_path = os.path.join(folder_path, file)
                filename = os.path.basename(file_path).strip().lower()  # ✅ 现在 file_path 已定义

                try:
                    df = pd.read_excel(file_path, sheet_name="data_interpt", engine="openpyxl")
                    if filename in self.range_dict:
                        start, end = self.range_dict[filename]
                        df = df.iloc[start:end]
                        print(f"📌 使用裁剪 → {filename} | {start}-{end} | 剩余: {len(df)} 行")
                    else:
                        print(f"⚠️ 未匹配到裁剪范围: {filename}")
                        continue

                    values = df.values[:, :8]  # 前 8 个特征

                    for i in range(0, len(values) - window_size + 1, stride):
                        window = values[i:i + window_size]
                        if window.shape == (window_size, 8):
                            self.samples.append(window.astype(np.float32))
                            self.labels.append(class_idx)

                except Exception as e:
                    print(f"⚠️ 无法读取文件 {file_path}：{e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


# ✅ 启动统计流程
if __name__ == "__main__":
    dataset = WindowedRailwayDataset(
        data_dir="分类数据",
        fault_range_file="fault_ranges.xlsx",
        window_size=60,
        stride=20
    )

    label_counter = Counter()
    for _, label in tqdm(dataset, desc="🔍 正在遍历样本"):
        label_counter[int(label)] += 1

    print("\n📊 每类标签样本数统计：")
    for label, count in sorted(label_counter.items()):
        print(f"标签 {label}: {count} 条样本")

    print(f"\n✅ 样本总数: {len(dataset)}")

