from dataset_windowed import WindowedRailwayDataset

dataset = WindowedRailwayDataset(
    data_dir="分类数据",
    fault_range_file="fault_ranges.xlsx",
    window_size=60,
    stride=20
)

x, y = dataset[0]
print("✅ 样本 shape:", x.shape)  # 应该是 torch.Size([60, 8])
print("✅ 标签:", y)

