import os
import copy
import time
import logging
import argparse
import yaml
from models.bilstm import BiLSTMNet
from yaml.loader import SafeLoader
from tqdm import tqdm
import os
os.environ["WANDB_MODE"] = "offline"

import wandb
import torch
from torchvision import models as torchmodel
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from utils.general import AppPath, colorstr, save_ckpt_, plot_and_log_result, seed_everything
from dataset_railway import RailwayDataset
from torch.utils.data import DataLoader, random_split
from dataset_cmapss import CMAPSSDataset


from models import CustomResnet

from toolkit import TLVFC
from toolkit.standardization import FlattenStandardization
from toolkit.matching import IndexMatching
from toolkit.transfer import VarTransfer

seed_everything(2)

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format="%(message)s", level=logging.INFO)
LOGGER = logging.getLogger("Torch-Cls")

p_crossover  = 0.1



def train_model(model, support_model, dataloaders, optimizer, opt, wandb, lr_scheduler=None):
    since = time.perf_counter()
    num_epochs, device = opt.epochs, opt.device

    DIR_SAVE = AppPath.RUN_TRAIN_DIR / f"{opt.name}/run_seed_{opt.seed}"
    os.makedirs(DIR_SAVE / "weights", exist_ok=True)

    LOGGER.info(f"\n{colorstr('Hyperparameter:')} {opt}")
    LOGGER.info(f"\n{colorstr('Device:')} {device}")
    LOGGER.info(f"\n{colorstr('Optimizer:')} {optimizer}")
    if opt.log_result:
        with open(DIR_SAVE / "opt.yaml", "w") as f:
            yaml.dump(vars(opt), f)
    if opt.lr_scheduler:
        LOGGER.info(
            f"\n{colorstr('LR Scheduler:')} {type(lr_scheduler).__name__}")
    else:
        lr_scheduler = None
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=list(
            range(torch.cuda.device_count())))

    criterion = nn.CrossEntropyLoss()
    LOGGER.info(f"\n{colorstr('Loss:')} {type(criterion).__name__}")

    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": [], "lr": []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_model_optim = copy.deepcopy(optimizer.state_dict())
    best_val_acc = 0.0

    model.to(device)
    if support_model is not None:
        support_model.to(device)
    for epoch in range(num_epochs):
        LOGGER.info(colorstr(f'\nEpoch {epoch}/{num_epochs-1}:'))
        for phase in ["train", "val"]:
            if phase == "train":
                LOGGER.info(colorstr('bright_yellow', 'bold', '\n%20s' + '%15s' * 3) %
                            ('Training:', 'gpu_mem', 'loss', 'acc'))
                model.train()
            else:
                LOGGER.info(colorstr('bright_yellow', 'bold', '\n%20s' + '%15s' * 3) %
                            ('Validation:', 'gpu_mem', 'loss', 'acc'))
                model.eval()
            running_items = 0
            running_loss = 0.0
            running_corrects = 0
            _phase = tqdm(dataloaders[phase],
                          total=len(dataloaders[phase]),
                          bar_format='{desc} {percentage:>7.0f}%|{bar:10}{r_bar}{bar:-10b}',
                          unit='batch')

            for inputs, labels in _phase:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # x_pretrain = None
                # if support_model is not None:
                   # x_pretrain = support_model(inputs)
                x_pretrain = None
                if support_model is not None:
                    x_pretrain = support_model(inputs)
                    outputs = model(inputs, phase, x_pretrain, p_crossover)
                else:
                    outputs = model(inputs)  # BiLSTM 的标准调用方式

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    #outputs = model(inputs, phase, x_pretrain, p_crossover)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        history['lr'].append(lr_scheduler.optimizer.param_groups[0]
                                             ["lr"]) if lr_scheduler else history['lr'].append(opt.lr)
                        if lr_scheduler is not None:
                            lr_scheduler.step()
                running_items += inputs.size(0)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}GB'
                desc = ('%35s' + '%15.6g' * 2) % (mem, running_loss /
                                                  running_items, running_corrects / running_items)
                _phase.set_description_str(desc)

            epoch_loss = running_loss / running_items
            epoch_acc = running_corrects / running_items

            print(f"🐸 Logging val_loss = {epoch_loss:.4f}, val_accuracy = {epoch_acc:.4f} at epoch {epoch}")

            if opt.wandb_log:
                wandb.log({
                    "val_loss": float(epoch_loss),
                    "val_accuracy": float(epoch_acc)
                }, step=epoch)

            if phase == 'train':
                if opt.wandb_log:
                    wandb.log({"train_acc": epoch_acc,"train_accuracy": epoch_acc, "train_loss": epoch_loss}, step = epoch)
                    if lr_scheduler:
                        wandb.log(
                            {"lr": lr_scheduler.optimizer.param_groups[0]["lr"]}, step = epoch)
                    else:
                        wandb.log({"lr": opt.lr})
                history["train_loss"].append(epoch_loss)
                history["train_acc"].append(epoch_acc)
            else:
                history["val_loss"].append(epoch_loss)
                history["val_acc"].append(epoch_acc)
                #增加的代码
                if opt.wandb_log:
                    print(f"📤 Logging val_loss = {epoch_loss}, val_accuracy = {epoch_acc.item()} (epoch {epoch})")
                    wandb.log({
                        "val_accuracy": float(epoch_acc.item()),
                        "val_loss": float(epoch_loss)
                    }, step=epoch)

                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_model_optim = copy.deepcopy(optimizer.state_dict())
                    if opt.save_ckpt:
                        save_ckpt_(model, DIR_SAVE, "best.pt")

    time_elapsed = time.perf_counter() - since
    LOGGER.info(f"Training complete in \
                {time_elapsed // 3600}h \
                {time_elapsed % 3600 // 60}m \
                { time_elapsed % 60}s with \
                {num_epochs} epochs")
    LOGGER.info(f"Best val Acc: {round(best_val_acc, 6)}")
    if opt.save_ckpt:
        save_ckpt_(model, DIR_SAVE, "last.pt")
    model.load_state_dict(best_model_wts)
    optimizer.load_state_dict(best_model_optim)

    if opt.log_result:
        plot_and_log_result(DIR_SAVE, history)
    if opt.save_ckpt:
        LOGGER.info(f"Best model weight saved at {DIR_SAVE}/weights/best.pt")
        LOGGER.info(f"Last model weight saved at {DIR_SAVE}/weights/last.pt")

    #if opt.wandb_log:
    #    wandb.log({"val_accuracy": best_val_acc.item()}, step=num_epochs)
    #if opt.wandb_log:
    #    wandb.log({"val_loss": epoch_loss}, step=num_epochs)

    return model, best_val_acc

    #测试模型
def test_model(model, test_loader, device):
    model.to(device)
    model.eval()
    totals = 0
    corrects = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            totals += inputs.size(0)
            corrects += torch.sum(preds == labels.data)

    acc = corrects / totals
    return acc.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                        help='cuda device or cpu (default: %(default)s)')
    parser.add_argument('--pretrain-epochs', type=int, default=20,
                        help='Number of epochs to train RUL model (default: 20)')

    parser.add_argument('--pretrain-cmapss', action='store_true',
                        help='Pretrain BiLSTM+Attention model on C-MAPSS FD001 dataset')

    parser.add_argument('--seed', type=int, default=2,
                        help='random seed will start at seed = 2 (default: %(default)s)')
    parser.add_argument('--model-base', type=str, default="resnet18",
                        help='The model name of target model (default: %(default)s)')
    parser.add_argument('--base-init', type=str, default="He",
                        help='The method to initialize for parameters ["He", "Glorot"] (default: %(default)s)')
    # choices= ['CIFAR10', 'Intel', 'PetImages']
    parser.add_argument('--data-name', type=str, default="CIFAR10",
                        help='The name of dataset (default: %(default)s)')
    parser.add_argument('--data-root', type=str, default="./data",
                        help='Folder where the dataset is saved (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Mini-batch size for each iteration when training model (default: %(default)s)')
    parser.add_argument('--workers', type=int, default=2,
                        help='The number of worker for dataloader (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='The number of epochs in training (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for optimizer (default: %(default)s)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay for optimizer (default: %(default)s)')
    parser.add_argument('--lr-scheduler', action='store_true',
                        help='Learning rate scheduler during training. Default: MultiStepLR')
    parser.add_argument('--lr-step', nargs='+', type=float,
                        help='Lmilestones for learning rate scheduler (defaut: [0.7, 0.9])')
    parser.add_argument('--show-summary', action='store_true',
                        help='Show model summary with default input size (3, 224, 224)')
    parser.add_argument('--name', default='exp',
                        help='Project name will saved at runs/train/__name__')
    parser.add_argument('--save_ckpt', action='store_true',
                        help='Save model into checkpoint folder')
    parser.add_argument('--log-result', action='store_true',
                        help='Save result of training progress into checkpoint folder')
    parser.add_argument('--wandb-log', action='store_true',
                        help='Log result into WanDB')
    parser.add_argument('--wandb-name', type=str, default="wandb_log",
                        help='Log name in WanDB')
    opt = parser.parse_args()
    if opt.pretrain_cmapss:
        print("🚀 开始在 C-MAPSS FD001 数据上预训练 RUL 模型")
        import wandb

        wandb.init(project="RUL预训练", name="FD001-BiLSTM", config={"model": "BiLSTM+Attn", "epochs": 20})


        class BiLSTMAttention(nn.Module):
            def __init__(self, input_size=11, hidden_size=128, num_layers=2, num_heads=4, dropout=0.3):
                super(BiLSTMAttention, self).__init__()
                self.lstm = nn.LSTM(input_size=input_size,
                                    hidden_size=hidden_size,
                                    num_layers=num_layers,
                                    batch_first=True,
                                    bidirectional=True,
                                    dropout=dropout)
                self.attn = nn.MultiheadAttention(embed_dim=hidden_size * 2,
                                                  num_heads=num_heads,
                                                  batch_first=True)
                self.flatten = nn.Flatten()
                self.dropout = nn.Dropout(p=dropout)
                self.out = nn.Linear(hidden_size * 2 * 30, 1)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
                flat = self.flatten(attn_out)
                drop = self.dropout(flat)
                return self.out(drop).squeeze(1)


        rul_dataset = CMAPSSDataset("data/CMAPSSData/train_FD001.txt", mode="train")
        rul_loader = DataLoader(rul_dataset, batch_size=64, shuffle=True)

        model = BiLSTMAttention().to(opt.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        for epoch in range(opt.pretrain_epochs):
            model.train()
            total_loss = 0
            for x, y in rul_loader:
                x, y = x.to(opt.device), y.to(opt.device)
                optimizer.zero_grad()
                preds = model(x)
                loss = criterion(preds, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x.size(0)
            print(f"[RUL Epoch {epoch + 1}] Loss: {total_loss / len(rul_loader.dataset):.4f}")
            wandb.log({"epoch": epoch + 1, "loss": total_loss / len(rul_loader.dataset)})

        torch.save(model.state_dict(), "rul_model.pt")
        print("✅ 预训练完成，模型已保存为 rul_model.pt")
        wandb.finish()  # ✅ 正确关闭 wandb 会话

        exit(0)

    try:
        device_name = os.getlogin()
    except:
        device_name = "Colab/Cloud"
    if opt.wandb_log:
        wandb.init(
            project=opt.wandb_name,
            name=opt.name,
            tags=[device_name],
            config=vars(opt))
        print(f"✅ WandB 已初始化！项目名: {opt.wandb_name}, 实验名: {opt.name}")
    else:
        wandb = None

    if opt.log_result:
        AppPath.RUN_DIR.mkdir(parents=True, exist_ok=True)
        AppPath.RUN_TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    with open('./config/dataset.yaml') as f:
        dataset_config = yaml.load(f, Loader=SafeLoader)

    LOGGER.info(f"\n*** RUN ON SEED {opt.seed} ***")
    num_classes = 8  # ✅ 你自己的数据是8类故障


    # Define the model before training
    #Source model  源模型：预训练的vgg19(从ImageNet迁移)
    source_model = torchmodel.vgg19(weights=torchmodel.VGG19_Weights.IMAGENET1K_V1)

    #Target mdoel  目标模型：自定义 ResNet（输出类别数根据数据集调整）
    target_model = BiLSTMNet(input_size=8, hidden_size=64, num_layers=2, num_classes=num_classes)

    # Transfer process
    #这段代码是整个迁移学习的核心思想实现，用三步实现源模型（VGG19）到目标模型（自定义 ResNet）的知识转移：
    #FlattenStandardization：将高维特征扁平化，方便特征匹配
    #IndexMatching：匹配目标模型和源模型中最相似的层
    #VarTransfer：基于方差分布进行迁移，对齐特征变化性（TLVFC 的关键）
    transfer_tool = TLVFC(
        standardization=FlattenStandardization(),
        matching=IndexMatching(),
        transfer=VarTransfer()
    )

    transfer_tool(
        from_module=source_model,
        to_module=target_model
    )
    # Finish define model
    #
    # train_loader, val_loader = get_train_valid_loader(
    #     dataset_name=opt.data_name,
    #     data_dir=opt.data_root,
    #     batch_size=opt.batch_size,
    #     augment=True,
    #     random_seed=opt.seed,
    #     num_workers=opt.workers
    # )
    # test_loader = get_test_loader(
    #     dataset_name=opt.data_name,
    #     data_dir=opt.data_root,
    #     batch_size=opt.batch_size,
    #     num_workers=opt.workers
    # )

    # ✅ 使用自定义的 RailwayDataset 代替图像数据
    dataset = RailwayDataset("./分类数据")  # 确保路径正确

    total_len = len(dataset)
    train_len = int(0.6 * total_len)
    val_len = int(0.2 * total_len)
    test_len = total_len - train_len - val_len

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)

    # ✅ 封装成原来结构一样的 dataloaders 字典
    dataloaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }

    dataloaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
    steps_per_epoch = len(dataloaders['train'])

    optimizer = torch.optim.AdamW(target_model.parameters(),
                                  lr=opt.lr,
                                  weight_decay=opt.weight_decay)
    total_step = steps_per_epoch * opt.epochs
    if opt.lr_step is not None:
        milestones = [int(opt.lr_step[i] * total_step)
                      for i in range(len(opt.lr_step))]
    else:  # use default
        milestones = [int(0.6 * total_step), int(0.8 * total_step)]
    lr_scheduler = MultiStepLR(optimizer, milestones=milestones)

    #用于支持迁移学习的辅助模型：
    #这个 support_model 只提取特征，不输出分类结果，辅助目标模型进行 feature crossover（特征融合）
    # support_model = torchmodel.vgg19(weights = torchmodel.VGG19_Weights.IMAGENET1K_V1)
    # # scale feature of vgg16 model from 25088 to 512
    # support_model.avgpool = nn.AdaptiveAvgPool2d((1,1))
    # support_model.classifier = torch.nn.Identity()
    # support_model.eval()
    support_model = None  # ✅ 我们现在不使用 VGG 支持模型

    if torch.cuda.device_count() > 1:
        target_model = nn.DataParallel(target_model,
                                       device_ids=list(range(torch.cuda.device_count())))


    best_model, val_acc = train_model(model=target_model,
                                      support_model=support_model,
                                      dataloaders=dataloaders,
                                      optimizer=optimizer,
                                      opt=opt,
                                      wandb=wandb,
                                      lr_scheduler=lr_scheduler)
    test_acc = test_model(best_model, dataloaders["test"], opt.device)
    LOGGER.info(f"Validation accuracy: {round(val_acc, 6)}")
    LOGGER.info(f"Test accuracy: {round(test_acc, 6)}")
    if opt.wandb_log:
        wandb.summary["val_accuracy"] = val_acc
        wandb.summary["test_accuracy"] = test_acc
        wandb.finish()



