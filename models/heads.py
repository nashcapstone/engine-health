"""The three output heads, shared by every model so the contract 2 dict is built in one place."""

import torch.nn as nn

import config

HEALTH_MAX = 100.0


class MultiHead(nn.Module):
    """Features (N, in_features) -> contract 2 dict.

    The regression heads work on a 0-1 scale and are multiplied back to cycles and
    health points on the way out, so the three losses start at similar sizes.
    """

    def __init__(self, in_features):
        super().__init__()
        self.rul = nn.Linear(in_features, 1)
        self.health = nn.Linear(in_features, 1)
        self.stage = nn.Linear(in_features, config.N_STAGES)

    def forward(self, features, attn=None):
        return {
            "rul": self.rul(features).squeeze(-1) * config.RUL_CAP,
            "health": self.health(features).squeeze(-1) * HEALTH_MAX,
            "stage_logits": self.stage(features),
            "attn": attn,
        }
