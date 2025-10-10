import torch
import torch.nn as nn


class DDFN(nn.Module):
    def __init__(self, input_size=256, output_size=128, hidden_size=256):
        super(DDFN, self).__init__()

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.ReLU()
        self.layer5 = nn.Linear(hidden_size, output_size)
       # self.layer6 = nn.Softmax(dim=1)  # Softmax 用于特征分布归一化

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        #x = self.layer6(x)  # 输出归一化特征
        return x
