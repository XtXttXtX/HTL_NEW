import torch
import torch.nn as nn

class DecoderFull(nn.Module):
    def __init__(self, input_size=128, num_classes=8, dropout=0.3):
        super(DecoderFull, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)
