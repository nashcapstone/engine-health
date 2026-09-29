"""Stage 2: CNN + LSTM. The CNN extracts local patterns, the LSTM reads them in order.

The LSTM sees trends inside the 30-cycle window only, not the whole engine life.
"""

import torch.nn as nn

import config
from models.heads import MultiHead


class CNNLSTM(nn.Module):
    def __init__(self, channels=(32, 64), kernel_size=3, lstm_hidden=64, lstm_layers=1, dropout=0.1):
        super().__init__()
        layers, in_channels = [], config.N_SENSORS
        for out_channels in channels:
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
            ]
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)
        self.lstm = nn.LSTM(in_channels, lstm_hidden, num_layers=lstm_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.heads = MultiHead(lstm_hidden)

    def forward(self, x):
        # (N, window, sensors) -> (N, sensors, window) for the convs, then back for the LSTM
        features = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.lstm(features)
        return self.heads(self.dropout(out[:, -1]))
