"""Stage 1: 1D-CNN baseline. Convolves over time, with the 14 sensors as input channels."""

import torch.nn as nn

import config
from models.heads import MultiHead


class CNNBaseline(nn.Module):
    def __init__(self, channels=(32, 64), kernel_size=3, hidden=64, dropout=0.1):
        super().__init__()
        layers, in_channels = [], config.N_SENSORS
        for out_channels in channels:
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
            ]
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * config.WINDOW, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.heads = MultiHead(hidden)

    def forward(self, x):
        # (N, window, sensors) -> (N, sensors, window)
        features = self.conv(x.transpose(1, 2))
        return self.heads(self.fc(features))
