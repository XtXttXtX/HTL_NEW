import torch
import argparse
from models.bilstm_encoder import BiLSTMEncoder
from torch.utils.data import DataLoader
from dataset_railway import RailwayDataset
from models.ddfn import DDFN  # 我们后面要补充 DDFN
from models.decoder import Decoder  # 我们后面要补充 Decoder
from torch.optim import Adam
from models.decoder_full import DecoderFull
from dataset_windowed import WindowedRailwayDataset

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import os
os.environ["WANDB_MODE"] = "offline"
import wandb

# ==================== 引入 TensorBoard ====================
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

# ==================== 初始化 SummaryWriter ====================
now = datetime.now().strftime("%Y%m%d_%H%M%S")
tb_log_dir = os.path.join("tb_logs", f"run_{now}")
writer = SummaryWriter(log_dir=tb_log_dir)


# 加载预训练模型的函数
def load_pretrained_model(model_path, device):
    # ✅ 第 1 步：创建一个新模型 —— 输入维度改为 8，适配你的 TC-A 数据
    model = BiLSTMEncoder(input_size=8)


    # ✅ 第 2 步：加载原模型的参数（rul_model.pt 来自 FD001 训练）
    state_dict = torch.load(model_path)

    # ✅ 第 3 步：过滤掉不兼容的参数（比如输入层、输出层维度不一样就不能加载）
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if k in model.state_dict() and v.size() == model.state_dict()[k].size()
    }

    # ✅ 第 4 步：加载匹配成功的参数；strict=False 允许部分加载，不报错
    model.load_state_dict(filtered_state_dict, strict=False)

    # ✅ 第 5 步：把模型放到显卡（或 CPU）
    model.to(device)

    # ✅ 第 6 步（可选）：打印哪些参数被加载了，哪些跳过了（用于调试确认）
    print(f"💡 加载了 {len(filtered_state_dict)} 个匹配参数（总共 {len(model.state_dict())} 层）")

    # ✅ 第 7 步：冻结参数（不更新），只保留 output 层可训练
    for name, param in model.named_parameters():
        if name in filtered_state_dict:
            param.requires_grad = False  # ✅ 冻结迁移来的层
        else:
            param.requires_grad = True   # ✅ 保留未迁移的层可训练（输入层 / 输出层）

    return model



def main(opt):
    # 初始化 wandb（如果需要）
    if opt.wandb_log:
        wandb.init(project="RUL迁移学习", name="源模型迁移")

    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载预训练的 BiLSTM 模型
    model = load_pretrained_model(opt.pretrained_model_path, device)

    # ⚠️ window_size 和 stride 可以根据你实际设定传入
    railway_dataset = WindowedRailwayDataset(
        data_dir=opt.data_path,
        fault_range_file="fault_ranges.xlsx",  # 请确保路径正确
        window_size=60,
        stride=20
    )
    #初始化 EarlyStopping 状态变量
    best_val_f1 = 0.0  # 当前观察到的最佳验证集 F1
    no_improve_epochs = 0  # 连续未提升的 epoch 计数器
    early_stop_patience = opt.early_stop_patience

    # 按标签分层划分（stratify），保证各类别比例一致
    indices = list(range(len(railway_dataset)))
    labels = railway_dataset.labels
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=42
    )
    from collections import Counter

    train_labels = [labels[i] for i in train_idx]
    val_labels = [labels[i] for i in val_idx]

    print("\n📊 训练集标签分布:")
    print(Counter(train_labels))

    print("\n📊 验证集标签分布:")
    print(Counter(val_labels))

    train_subset = Subset(railway_dataset, train_idx)
    val_subset = Subset(railway_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=opt.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=opt.batch_size, shuffle=False)

    # 创建 DDFN 模块（特征对齐）
    ddfn = DDFN().to(device)
    decoder = DecoderFull().to(device)


    # 定义优化器和损失函数
    optimizer = Adam(decoder.parameters(), lr=opt.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # 训练过程
    for epoch in range(opt.epochs):
        model.eval()
        decoder.train()

        total_loss = 0
        all_preds, all_labels = [], []

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                features = model(inputs, return_features=True)

            aligned_features = ddfn(features)
            outputs = decoder(aligned_features)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # === 🟩 Train metrics ===
        from sklearn.metrics import accuracy_score, f1_score

        acc_train = accuracy_score(all_labels, all_preds)
        f1_train = f1_score(all_labels, all_preds, average="macro")
        avg_loss = total_loss / len(train_loader)

        # === 🟩 验证阶段 ===
        decoder.eval()
        val_loss = 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                features = model(inputs, return_features=True)
                outputs = decoder(ddfn(features))
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        acc_val = accuracy_score(val_labels, val_preds)
        f1_val = f1_score(val_labels, val_preds, average="macro")

        # === EarlyStopping 判断 ===
        if f1_val > best_val_f1:
            best_val_f1 = f1_val
            no_improve_epochs = 0
            torch.save(decoder.state_dict(), "best_decoder.pt")
            print(f"✅ 验证集 F1 提升为 {f1_val:.4f}，保存当前模型为 best_decoder.pt")
        else:
            no_improve_epochs += 1
            print(f"⚠️ 验证集 F1 未提升，已连续 {no_improve_epochs} 次")

            if no_improve_epochs >= early_stop_patience:
                print(f"🛑 早停触发：验证集 F1 连续 {early_stop_patience} 次无提升，提前终止训练。")
                break

        # 绘制混淆矩阵
        cm = confusion_matrix(val_labels, val_preds)
        fig, ax = plt.subplots(figsize=(6, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax)
        plt.close(fig)  # 避免多余图像弹出

        # TensorBoard 写入图像
        if opt.tensorboard_log:
            writer.add_figure("ConfusionMatrix/val", fig, epoch)

        avg_val_loss = val_loss / len(val_loader)

        # === 📊 记录 ===
        if opt.wandb_log:
            wandb.log({
                "epoch": epoch + 1,
                "Loss/train": avg_loss,
                "Accuracy/train": acc_train,
                "F1/train": f1_train,
                "Loss/val": avg_val_loss,
                "Accuracy/val": acc_val,
                "F1/val": f1_val
            }, step=epoch)

        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Accuracy/train", acc_train, epoch)
        writer.add_scalar("F1/train", f1_train, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/val", acc_val, epoch)
        writer.add_scalar("F1/val", f1_val, epoch)

        print(
            f"Epoch {epoch + 1}/{opt.epochs} - Train Loss: {avg_loss:.4f} - Val Acc: {acc_val:.4f} - Val F1: {f1_val:.4f}")

    # 保存训练后的 Decoder 权重
    torch.save(decoder.state_dict(), "decoder_model.pt")
    print("✅ 训练完成，Decoder 权重已保存为 decoder_model.pt")
    # ✅ 训练完成后
    print("✅ 训练完成，Decoder 权重已保存为 decoder_model.pt")

    # # ✅ 自动调用推理模块
    # import subprocess
    # print("🔍 正在执行模型推理以生成混淆矩阵与评估指标...")
    # subprocess.run(["python", "predict.py"])
    #
    # writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensorboard-log', action='store_true', help='Log results to TensorBoard')

    parser.add_argument('--pretrained-model-path', type=str, default='rul_model.pt',
                        help='Path to the pretrained BiLSTM model')
    parser.add_argument('--data-path', type=str, default='./data/TC-A', help='Path to the Railway dataset')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs for training')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--wandb-log', action='store_true', help='Log rsults to WandB')
    parser.add_argument('--early-stop-patience', type=int, default=1000,
                        help='验证集 F1 连续多少个 epoch 无提升则提前终止训练（EarlyStopping）')

    opt = parser.parse_args()
    main(opt)
