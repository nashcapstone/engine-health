"""Stage 3: sensor attention + CNN + LSTM + the three heads.

At each timestep a small network scores the 14 sensors, a softmax turns the scores
into weights that sum to 1 across sensors, and the input is multiplied by them. The
weights are returned as `attn`, shape (N, window, sensors), for the explainability work.

Softmax weights average 1/14, which would shrink the input to about 1/14 of its size.
With scale=True the weights are multiplied by 14 before they are applied, so uniform
attention leaves the input unchanged. `attn` is always the weights that sum to 1.
"""

import torch
import torch.nn as nn

import config
from models.cnn_lstm import CNNLSTM


class SensorAttention(nn.Module):
    """(N, window, sensors) -> weights (N, window, sensors), softmax over sensors.

    The scorer is a conv over time, so a sensor's weight can depend on its recent trend
    and on the other sensors, not only on its current value.
    """

    def __init__(self, hidden=32, kernel_size=3):
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv1d(config.N_SENSORS, hidden, kernel_size, padding=kernel_size // 2),
            nn.Tanh(),
            nn.Conv1d(hidden, config.N_SENSORS, 1),
        )

    def forward(self, x):
        logits = self.score(x.transpose(1, 2)).transpose(1, 2)
        return torch.softmax(logits, dim=-1)


class AttentionCNNLSTM(nn.Module):
    def __init__(
        self,
        attn_hidden=32,
        attn_kernel=3,
        scale=True,
        channels=(32, 64),
        kernel_size=3,
        lstm_hidden=64,
        lstm_layers=1,
        dropout=0.1,
    ):
        super().__init__()
        self.attention = SensorAttention(attn_hidden, attn_kernel)
        self.scale = float(config.N_SENSORS) if scale else 1.0
        self.backbone = CNNLSTM(channels, kernel_size, lstm_hidden, lstm_layers, dropout)

    def forward(self, x):
        attn = self.attention(x)
        out = self.backbone(x * attn * self.scale)
        out["attn"] = attn
        return out
