import torch
import torch.nn as nn

# BiLSTM + Attention 网络结构
class BiLSTMAttention(nn.Module):
    def __init__(self, input_size=11, hidden_size=128, num_layers=2, num_heads=4, dropout=0.3):
        super(BiLSTMAttention, self).__init__()
        # LSTM 层，双向
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout)
        # MultiheadAttention
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size * 2,
                                          num_heads=num_heads,
                                          batch_first=True)
        # 全连接层输出
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(hidden_size * 2 * 30, 1)  # 输出单一数值

    def forward(self, x, return_features=False):
        lstm_out, _ = self.lstm(x)  # shape: [B, T, H*2]处理时间序列，提取时序上下文
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)  # [B, T, H*2]加入注意力机制，增强重要位置信息

        # ✅ 使用 temporal mean pooling：取每个序列的平均 → [B, H*2]
        pooled = attn_out.mean(dim=1)  #压缩时间维度，变成固定向量

        drop = self.dropout(pooled)  # shape: [B, H*2] 防止过拟合，保持 shape 不变 压缩时间维度，变成固定维度的向量

        if return_features:
            return drop  # 迁移任务中输出特征（送 DDFN）
        else:
            return self.out(drop).squeeze(1)  # 源任务中输出 RUL

# 这个类会被用来加载你的预训练模型。
