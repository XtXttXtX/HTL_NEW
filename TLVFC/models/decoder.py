import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, input_size=128, num_classes=8):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)
