import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt








from torch.utils.data import DataLoader
from dataset_windowed import WindowedRailwayDataset
from models.bilstm_encoder import BiLSTMEncoder
from models.ddfn import DDFN
from models.decoder_full import DecoderFull

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from datetime import datetime

from sklearn.metrics import classification_report


from datetime import datetime  # 导入时间模块
# 动态生成文件名（使用当前日期和时间）
logdir = f"./confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

# ========== 配置 ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 8
WINDOW_SIZE = 60
STRIDE = 20



# ========== 加载模型 ==========
encoder = BiLSTMEncoder(input_size=8).to(DEVICE)
ddfn = DDFN().to(DEVICE)
decoder = DecoderFull(input_size=128, num_classes=NUM_CLASSES).to(DEVICE)

# ✅ 从 best_ckpt.pt 一次性加载三段权重
ckpt = torch.load("best_ckpt.pt", map_location=DEVICE)
encoder.load_state_dict(ckpt["encoder"])
ddfn.load_state_dict(ckpt["ddfn"])
decoder.load_state_dict(ckpt["decoder"])
print("✅ 已从 best_ckpt.pt 加载 encoder / ddfn / decoder 权重")



# ❗ 如果 encoder 和 ddfn 是冻结状态，也可以加载训练中保存的（可选）
encoder.eval()
ddfn.eval()
decoder.eval()

# ========== 加载数据 ==========
from torch.utils.data import Subset

# 加载完整数据集（保持一致的路径和参数）
full_dataset = WindowedRailwayDataset(
    data_dir="分类数据",  # 替换为你的实际路径
    fault_range_file="fault_ranges.xlsx",
    window_size=WINDOW_SIZE,
    stride=STRIDE
)

# ✅ 从保存的 split_idx.npz 中加载测试集索引
# === 如果训练时用了 --split-by-time，这里也要用同样的时间划分 ===
test_dataset = WindowedRailwayDataset(
    data_dir="分类数据",                 # ⚠️ 改成你自己的数据路径
    fault_range_file="fault_ranges.xlsx",
    window_size=60,
    stride=20,
    split="test",
    split_by_time=True,                # ⚠️ 一定要加这个
    ratios=(0.6, 0.2, 0.2)             # ⚠️ 跟训练时的比例保持一致
)


# ✅ 创建 DataLoader
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"✅ 已加载测试集，仅包含 {len(test_dataset)} 条样本（来自 split_idx.npz）")


# ========== 推理 ==========
all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)  # [B, T, 8]
        y = y.to(DEVICE)

        features = encoder(x)        # [B, 256]
        aligned = ddfn(features)     # [B, 128]
        logits = decoder(aligned)   # [B, 8]

        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())

def log_confusion_matrix_tensorboard(y_true, y_pred, class_names, writer, global_step=0):
    # ✅ 更稳妥：用 from_predictions（不会把 y_true/y_pred 搞反）
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=range(8), cmap="Blues", normalize=None
    )
    fig = disp.figure_
    fig.set_size_inches(6, 6)
    disp.ax_.set_xlabel("Predicted label")
    disp.ax_.set_ylabel("True label")

    if opt.tensorboard_log:
        writer.add_figure("ConfusionMatrix/test", fig)
    plt.close(fig)


# ========== 评估 ==========
y_true = np.concatenate(all_labels)
y_pred = np.concatenate(all_preds)

report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
conf_mat = confusion_matrix(y_true, y_pred)

# 打印简要指标
acc = report["accuracy"]
f1 = report["macro avg"]["f1-score"]

print(f"\n✅ Accuracy: {acc:.4f}")
print(f"✅ Macro F1-score: {f1:.4f}")


# ✅ 创建 writer（每次一个新目录）
logdir = f"./tb_logs/infer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir=logdir)
writer.add_scalar("Accuracy/test", acc, 0)
writer.add_scalar("F1/test", f1, 0)

# ✅ 写入图像
class_names = ['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']
log_confusion_matrix_tensorboard(y_true, y_pred, class_names, writer)



# 计算并输出分类报告（字典格式）
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)





writer.close()



# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=[f"F{i}" for i in range(NUM_CLASSES)],
            yticklabels=[f"F{i}" for i in range(NUM_CLASSES)])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")

# 本地保存混淆矩阵图像
plt.savefig(logdir)
plt.close()




