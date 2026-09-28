"""The three targets, all derived from RUL. Shared so data, models and evaluation agree."""

import numpy as np

import config


def cap_rul(rul):
    return np.minimum(np.asarray(rul, dtype=np.float32), config.RUL_CAP)


def health_score(rul):
    """100 * (RUL / cap) ^ exponent, on capped RUL."""
    rul = cap_rul(rul)
    return (100.0 * (rul / config.RUL_CAP) ** config.HEALTH_EXPONENT).astype(np.float32)


def health_stage(rul):
    """0 = HEALTHY, 1 = WARNING, 2 = CRITICAL."""
    rul = np.asarray(rul)
    stage = np.ones(rul.shape, dtype=np.int64)
    stage[rul > config.HEALTHY_ABOVE] = 0
    stage[rul < config.CRITICAL_BELOW] = 2
    return stage
