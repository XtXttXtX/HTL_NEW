import torch
import seaborn as sns

import argparse
from models.bilstm_encoder import BiLSTMEncoder
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


from dataset_railway import RailwayDataset
from models.ddfn import DDFN  # 我们后面要补充 DDFN
from models.decoder import Decoder  # 我们后面要补充 Decoder
from torch.optim import Adam
from models.decoder_full import DecoderFull
from dataset_windowed import WindowedRailwayDataset
from collections import Counter



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
    # ✅ 初始化占位，防止 test_dataset 未定义
    test_dataset = None
    # 初始化 wandb（如果需要）
    if opt.wandb_log:
        wandb.init(project="RUL迁移学习", name="源模型迁移")

    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载预训练的 BiLSTM 模型
    model = load_pretrained_model(opt.pretrained_model_path, device)

    # ⚠️ 根据是否启用 split-by-time 选择数据集划分方式
    if opt.split_by_time:
        print("🕒 启用按时间比例 (6:2:2) 划分 train/val/test ...")

        # 三个数据集分别使用 split='train'/'val'/'test'
        train_dataset = WindowedRailwayDataset(
            data_dir=opt.data_path,
            fault_range_file="fault_ranges.xlsx",
            window_size=60,
            stride=20,
            split="train",
            split_by_time=True,
            ratios=tuple(opt.ratios)
        )
        val_dataset = WindowedRailwayDataset(
            data_dir=opt.data_path,
            fault_range_file="fault_ranges.xlsx",
            window_size=60,
            stride=20,
            split="val",
            split_by_time=True,
            ratios=tuple(opt.ratios)
        )
        test_dataset = WindowedRailwayDataset(
            data_dir=opt.data_path,
            fault_range_file="fault_ranges.xlsx",
            window_size=60,
            stride=20,
            split="test",
            split_by_time=True,
            ratios=tuple(opt.ratios)
        )

        # === 为时间划分模式创建 DataLoader ===
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)

    else:
        print("🎲 使用原始随机划分方式 (6:2:2) ...")
        import random, numpy as np

        # 1️⃣ 固定随机种子
        def set_seed(seed=42):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        set_seed(42)

        # 2️⃣ 构造完整数据集
        railway_dataset = WindowedRailwayDataset(
            data_dir=opt.data_path,
            fault_range_file="fault_ranges.xlsx",
            window_size=60,
            stride=20
        )

        # 3️⃣ 按 6:2:2 划分并保存索引
        N = len(railway_dataset)
        n_train = int(0.6 * N)
        n_val = int(0.2 * N)
        n_test = N - n_train - n_val
        g = torch.Generator().manual_seed(42)
        train_ds, val_ds, test_ds = random_split(railway_dataset, [n_train, n_val, n_test], generator=g)

        np.savez("split_idx.npz",
                 train_idx=np.array(train_ds.indices),
                 val_idx=np.array(val_ds.indices),
                 test_idx=np.array(test_ds.indices))
        print(f"✅ 已保存划分索引 split_idx.npz (train/val/test = {len(train_ds)}/{len(val_ds)}/{len(test_ds)})")

        # 4️⃣ 创建 DataLoader
        train_loader = DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=opt.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=opt.batch_size, shuffle=False)

        # 5️⃣ 记录数据分布
        train_dataset, val_dataset, test_dataset = train_ds, val_ds, test_ds

    #初始化 EarlyStopping 状态变量
    best_val_f1 = 0.0  # 当前观察到的最佳验证集 F1
    no_improve_epochs = 0  # 连续未提升的 epoch 计数器
    early_stop_patience = opt.early_stop_patience

    # —— 通用：从任意数据集对象提取标签列表（兼容 Subset 和自定义 Dataset）——
    def get_labels_from_dataset(ds):
        # 1) Subset 的情况：用 indices 在底层 dataset.labels 里索引
        try:
            from torch.utils.data import Subset
            if isinstance(ds, Subset) and hasattr(ds.dataset, "labels"):
                return [ds.dataset.labels[i] for i in ds.indices]
        except Exception:
            pass
        # 2) 自定义 WindowedRailwayDataset：直接有 labels
        if hasattr(ds, "labels"):
            return list(ds.labels)
        # 3) 兜底：从 dataset 迭代拿（慢，但通用）
        return [y for _, y in ds]

    train_labels = get_labels_from_dataset(train_dataset)
    val_labels = get_labels_from_dataset(val_dataset)

    print("\n📊 训练集标签分布:")
    print(Counter(train_labels))

    print("\n📊 验证集标签分布:")
    print(Counter(val_labels))
    if test_dataset is not None:
        test_labels = get_labels_from_dataset(test_dataset)
        print("\n📊 测试集标签分布:")
        print(Counter(test_labels))







    # 创建 DDFN 模块（特征对齐）
    ddfn = DDFN().to(device)
    decoder = DecoderFull().to(device)


    # 定义优化器和损失函数
    for p in model.parameters():
        p.requires_grad = False  # 冻结 BiLSTM encoder

    ddfn.train()
    decoder.train()

    optimizer = Adam(list(ddfn.parameters()) + list(decoder.parameters()), lr=opt.lr)
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
            torch.save({
                "encoder": model.state_dict(),
                "ddfn": ddfn.state_dict(),
                "decoder": decoder.state_dict(),
                "label_map": getattr(train_dataset, "label_map", None)
            }, "best_ckpt.pt")
            print(f"✅ 验证集 F1 提升为 {f1_val:.4f}，保存当前模型为 best_ckpt.pt")


        else:
            no_improve_epochs += 1
            print(f"⚠️ 验证集 F1 未提升，已连续 {no_improve_epochs} 次")

            if no_improve_epochs >= early_stop_patience:
                print(f"🛑 早停触发：验证集 F1 连续 {early_stop_patience} 次无提升，提前终止训练。")
                break

        # 绘制混淆矩阵
        disp = ConfusionMatrixDisplay.from_predictions(
            val_labels, val_preds,
            labels=list(range(8)),
            cmap="Blues",
            normalize=None
        )
        fig = disp.figure_
        fig.set_size_inches(6, 6)
        disp.ax_.set_xlabel("Predicted label")
        disp.ax_.set_ylabel("True label")
        disp.ax_.set_title(f"Confusion Matrix (VAL) epoch={epoch + 1}")

        if opt.tensorboard_log:
            writer.add_figure("ConfusionMatrix/val", fig, epoch + 1)
        plt.close(fig)



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
    # 保存三段权重，供 predict.py 使用
    torch.save(model.state_dict(), "encoder.pt")
    torch.save(ddfn.state_dict(), "ddfn.pt")
    torch.save(decoder.state_dict(), "decoder_model.pt")

    print("✅ 训练完成，已保存 encoder.pt / ddfn.pt / decoder_model.pt")

    print("✅ 训练完成，Decoder 权重已保存为 decoder_model.pt")
    if test_dataset is not None:
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        ckpt = torch.load("best_ckpt.pt", map_location=device)
        model.load_state_dict(ckpt["encoder"], strict=False)
        ddfn.load_state_dict(ckpt["ddfn"])
        decoder.load_state_dict(ckpt["decoder"])
        model.eval();
        ddfn.eval();
        decoder.eval()

        ys, ps = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                feats = model(x, return_features=True)
                out = decoder(ddfn(feats))
                ps.append(out.argmax(1).cpu().numpy())
                ys.append(y.cpu().numpy())

        import numpy as np
        y_true = np.concatenate(ys)
        y_pred = np.concatenate(ps)
        acc = accuracy_score(y_true, y_pred)
        f1m = f1_score(y_true, y_pred, average="macro")
        print(f"[TEST] acc={acc:.4f}  f1_macro={f1m:.4f}")
        print(classification_report(y_true, y_pred, digits=4))
        if opt.tensorboard_log:
            writer.add_scalar("Accuracy/test", acc, 0)
            writer.add_scalar("F1/test", f1m, 0)
            writer.flush()

        # ===== 绘制并写入测试集混淆矩阵（最稳妥：from_predictions）=====


        disp = ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred,  # ← 用测试集的 y_true / y_pred
            labels=list(range(8)),
            cmap="Blues",
            normalize=None  # 想看百分比就改成 "true"
        )
        fig = disp.figure_
        fig.set_size_inches(6, 6)
        disp.ax_.set_xlabel("Predicted label")
        disp.ax_.set_ylabel("True label")
        disp.ax_.set_title("Confusion Matrix (TEST)")

        if opt.tensorboard_log:
            writer.add_figure("ConfusionMatrix/test", fig)  # ← 写到 test
        plt.close(fig)

    # ✅ 训练完成后
    print("✅ 训练完成，Decoder 权重已保存为 decoder_model.pt")

    # =======================
    # 🎯 训练结束提示信息
    # =======================
    if os.path.exists("best_ckpt.pt"):
        print("\n✅ 训练结束，最优模型已保存为：best_ckpt.pt")
        print("👉 你可以在测试或预测脚本中通过以下方式加载：")
        print("    ckpt = torch.load('best_ckpt.pt', map_location=device)")
        print("    encoder.load_state_dict(ckpt['encoder'])")
        print("    ddfn.load_state_dict(ckpt['ddfn'])")
        print("    decoder.load_state_dict(ckpt['decoder'])")
    else:
        print("\n⚠️ 警告：未找到 best_ckpt.pt，请检查 EarlyStopping 或保存逻辑是否执行。")

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
    parser.add_argument('--split-by-time', action='store_true',
                        help='在每个 Excel 内按时间比例 (6:2:2) 划分 train/val/test')
    parser.add_argument('--ratios', type=float, nargs=3, default=[0.6, 0.2, 0.2],
                        help='train/val/test 比例，仅在 --split-by-time 启用时有效')

    opt = parser.parse_args()
    main(opt)
