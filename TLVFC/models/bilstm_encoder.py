import torch
import torch.nn as nn

class BiLSTMEncoder(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, num_heads=4, dropout=0.3):
        super(BiLSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=num_heads,
            batch_first=True
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, return_features=True):
        """
        输入：
            x: Tensor [B, T, 8]
        返回：
            如果 return_features=True：返回 [B, 256] 特征向量
            否则抛出异常（本模块不做最终输出，只提特征）
        """
        lstm_out, _ = self.lstm(x)  # → [B, T, 256]
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)  # → [B, T, 256]

        pooled = attn_out.mean(dim=1)  # Temporal Mean Pooling → [B, 256]
        dropped = self.dropout(pooled)

        if return_features:
            return dropped
        else:
            raise ValueError("BiLSTMEncoder 只负责特征提取，请设置 return_features=True。")
