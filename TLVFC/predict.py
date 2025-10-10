import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# ========== 强制WandB纯离线模式 ==========
# 彻底禁用所有网络连接，但保留本地记录功能
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_API_KEY"] = "fake_key"
os.environ["WANDB_SILENT"] = "true"  # 减少输出
os.environ["WANDB_DISABLE_CODE"] = "true"  # 禁用代码保存
os.environ["WANDB_DISABLE_GIT"] = "true"   # 禁用Git集成

import wandb

# 创建完全离线的配置
offline_settings = wandb.Settings(
    disable_networking=True,  # 完全禁用网络
    _disable_meta=True,       # 禁止收集系统信息
    _disable_stats=True,      # 禁用统计收集
    start_method="thread",     # 避免多进程问题
    save_code=False,           # 不保存代码
    console="off"             # 关闭控制台输出
)

# ✅ 初始化WandB（纯离线模式，但保留本地记录）
run = wandb.init(
    project="RUL迁移学习",
    name="测试推理",
    mode="offline",           # 明确指定离线模式
    settings=offline_settings
)
print("✅ WandB已初始化为离线模式，将只进行本地记录")




from torch.utils.data import DataLoader
from dataset_windowed import WindowedRailwayDataset
from models.bilstm_encoder import BiLSTMEncoder
from models.ddfn import DDFN
from models.decoder_full import DecoderFull

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from datetime import datetime

from sklearn.metrics import classification_report
import wandb


from datetime import datetime  # 导入时间模块
# 动态生成文件名（使用当前日期和时间）
logdir = f"./confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

# ========== 配置 ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 8
WINDOW_SIZE = 60
STRIDE = 20

# ✅ 设置 WandB 离线模式
os.environ["WANDB_MODE"] = "offline"
wandb.init(project="RUL迁移学习", name="测试推理")

# ========== 加载模型 ==========
encoder = BiLSTMEncoder(input_size=8).to(DEVICE)
ddfn = DDFN().to(DEVICE)
decoder = DecoderFull(input_size=128, num_classes=NUM_CLASSES).to(DEVICE)

# ✅ 加载你保存的 decoder 权重
decoder.load_state_dict(torch.load("decoder_model.pt", map_location=DEVICE))

# ❗ 如果 encoder 和 ddfn 是冻结状态，也可以加载训练中保存的（可选）
encoder.eval()
ddfn.eval()
decoder.eval()

# ========== 加载数据 ==========
test_dataset = WindowedRailwayDataset(
    data_dir="分类数据",  # 替换为你的实际路径
    fault_range_file="fault_ranges.xlsx",
    window_size=WINDOW_SIZE,
    stride=STRIDE
)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

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
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    writer.add_figure("ConfusionMatrix", fig, global_step=global_step)
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

# ✅ 转为 WandB 表格记录
table = wandb.Table(columns=["Class", "Precision", "Recall", "F1-score", "Support"])
for cls_name, metrics in report.items():
    if cls_name in class_names:  # 排除 'accuracy', 'macro avg' 等
        table.add_data(
            cls_name,
            round(metrics["precision"], 4),
            round(metrics["recall"], 4),
            round(metrics["f1-score"], 4),
            int(metrics["support"])
        )

# ✅ 记录到 WandB（将出现在 Panels > Table 中）
wandb.log({"Per-class metrics": table})

writer.close()

# ========== WandB 记录 ==========
wandb.log({
    "test/accuracy": acc,
    "test/f1": f1
})

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

# # 保存图像并上传到 WandB
# plt.savefig("confusion_matrix.png")
# wandb.log({"confusion_matrix": wandb.Image("confusion_matrix.png")})

print("\n✅ 推理完成，指标已记录 WandB（本地模式）。")

