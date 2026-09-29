"""Multi-task loss over the three heads."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from models.heads import HEALTH_MAX

DEFAULT_WEIGHTS = {"rul": 1.0, "health": 1.0, "stage": 0.1}


def stage_class_weights(y_stage):
    """Inverse-frequency weights, mean 1 across classes. CRITICAL is the rare stage."""
    counts = torch.bincount(torch.as_tensor(y_stage), minlength=config.N_STAGES).float()
    weights = counts.sum() / counts.clamp(min=1)
    return weights / weights.mean()


class MultiTaskLoss(nn.Module):
    """weights['rul'] * MSE(rul) + weights['health'] * MSE(health) + weights['stage'] * CE(stage).

    Both MSE terms are taken on targets scaled to 0-1, so they are comparable to each
    other. Cross-entropy is still larger, which is why its default weight is lower.
    """

    def __init__(self, weights=None, class_weights=None):
        super().__init__()
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.register_buffer("class_weights", class_weights)

    def forward(self, out, batch):
        """Returns (total, parts). parts holds the three unweighted losses as floats."""
        rul = F.mse_loss(out["rul"] / config.RUL_CAP, batch["y_rul"].float() / config.RUL_CAP)
        health = F.mse_loss(out["health"] / HEALTH_MAX, batch["y_health"].float() / HEALTH_MAX)
        stage = F.cross_entropy(out["stage_logits"], batch["y_stage"].long(), weight=self.class_weights)
        total = self.weights["rul"] * rul + self.weights["health"] * health + self.weights["stage"] * stage
        return total, {"rul": rul.item(), "health": health.item(), "stage": stage.item()}
