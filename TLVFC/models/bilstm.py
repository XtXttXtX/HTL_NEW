import torch
import torch.nn as nn

class BiLSTMNet(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, num_classes=8, dropout=0.5):
        super(BiLSTMNet, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 双向LSTM -> hidden*2

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, seq_len, hidden*2)
        out = lstm_out[:, -1, :]    # 取最后一个时间步的输出
        out = self.fc(out)          # 映射到分类数
        return out
