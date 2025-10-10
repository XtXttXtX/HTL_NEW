import sys
import os

# 忽略项目中的 wandb 文件夹，优先使用官方库
sys.path = [p for p in sys.path if not p.endswith(os.path.sep + "wandb")]

import argparse
import os
import torch
import torch.nn as nn
import importlib
from torch.utils.data import DataLoader
from dataset_cmapss import CMAPSSDataset
from models.bilstm import BiLSTMNet  # 正确类名
from datetime import datetime


def train(opt):
    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')

    wandb = None
    if opt.wandb_log:
        import importlib
        wandb = importlib.import_module("wandb")
        wandb.login()
        wandb.init(project=opt.wandb_project, name=opt.wandb_run, config=vars(opt))

    # 初始化 wandb

    # 加载数据
    train_dataset = CMAPSSDataset(opt.data_path, sequence_length=opt.seq_len, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)

    # 模型
    model = BiLSTMNet(input_size=11, hidden_size=64, num_layers=2, num_classes=1)  # RUL -> 回归
    model.to(device)

    # 损失函数 & 优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    model.train()
    for epoch in range(opt.epochs):
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)  # [B] -> [B, 1]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{opt.epochs}] - Loss: {avg_loss:.4f}")
        if opt.wandb_log:
            wandb.log({"epoch": epoch + 1, "train_loss": avg_loss})

    if opt.save_path:
        os.makedirs(os.path.dirname(opt.save_path), exist_ok=True)
        torch.save(model.state_dict(), opt.save_path)
        print(f"✅ 模型已保存到: {opt.save_path}")

    if opt.wandb_log:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/CMAPSSData/train_FD001.txt')
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save_path', type=str, default='saved_models/rul_model.pt')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--wandb_log', action='store_true', help='是否记录到 WandB')
    parser.add_argument('--wandb_project', type=str, default='FD001-RUL')
    parser.add_argument('--wandb_run', type=str, default=f"FD001_Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    opt = parser.parse_args()
    train(opt)
