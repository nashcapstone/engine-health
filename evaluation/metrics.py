"""Metrics on the predictions file (contract 3).

Every function takes plain arrays, so it works on a predictions DataFrame column,
a numpy array or a list. RUL errors are in cycles.
"""

import numpy as np

import config
from contracts import check_predictions
from labels import health_score, health_stage

# NASA PHM08 scoring: late predictions (pred > true) cost more than early ones
NASA_EARLY = 13.0
NASA_LATE = 10.0


def _pair(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    assert y_true.shape == y_pred.shape, f"shapes differ: {y_true.shape} and {y_pred.shape}"
    assert y_true.size > 0, "no rows to score"
    return y_true, y_pred


def rmse(y_true, y_pred):
    y_true, y_pred = _pair(y_true, y_pred)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true, y_pred):
    y_true, y_pred = _pair(y_true, y_pred)
    return float(np.mean(np.abs(y_pred - y_true)))


def nasa_score(y_true, y_pred):
    """Sum over rows of exp(-d/13) - 1 if d < 0, else exp(d/10) - 1, with d = pred - true.

    Lower is better, 0 is perfect. It is a sum, so it grows with the number of rows:
    compare scores only on the same split.
    """
    y_true, y_pred = _pair(y_true, y_pred)
    d = y_pred - y_true
    per_row = np.where(d < 0, np.exp(-d / NASA_EARLY), np.exp(d / NASA_LATE)) - 1.0
    return float(per_row.sum())


def stage_accuracy(stage_true, stage_pred):
    stage_true, stage_pred = _pair(stage_true, stage_pred)
    return float(np.mean(stage_true == stage_pred))


def confusion_matrix(stage_true, stage_pred):
    """(N_STAGES, N_STAGES) counts. Rows are the true stage, columns the predicted stage."""
    stage_true = np.asarray(stage_true, dtype=np.int64)
    stage_pred = np.asarray(stage_pred, dtype=np.int64)
    cm = np.zeros((config.N_STAGES, config.N_STAGES), dtype=np.int64)
    np.add.at(cm, (stage_true, stage_pred), 1)
    return cm


def stage_recall(stage_true, stage_pred):
    """Fraction of each true stage predicted correctly. NaN for a stage with no rows."""
    cm = confusion_matrix(stage_true, stage_pred)
    totals = cm.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.diag(cm) / totals


def head_agreement(df):
    """How often the health and stage heads agree with the RUL head.

    The three heads are separate, so health_pred and stage_pred are not computed from
    rul_pred. Returns the stage agreement rate and the mean health gap in points.
    """
    implied_stage = health_stage(df["rul_pred"].to_numpy())
    implied_health = health_score(df["rul_pred"].to_numpy())
    return {
        "stage_agreement": float(np.mean(df["stage_pred"].to_numpy() == implied_stage)),
        "health_gap": float(np.mean(np.abs(df["health_pred"].to_numpy() - implied_health))),
    }


def summarize(df):
    """All headline metrics for one predictions DataFrame, as a flat dict."""
    check_predictions(df)
    recall = stage_recall(df["stage_true"], df["stage_pred"])
    return {
        "n": len(df),
        "rul_rmse": rmse(df["rul_true"], df["rul_pred"]),
        "rul_mae": mae(df["rul_true"], df["rul_pred"]),
        "nasa_score": nasa_score(df["rul_true"], df["rul_pred"]),
        "health_rmse": rmse(df["health_true"], df["health_pred"]),
        "health_mae": mae(df["health_true"], df["health_pred"]),
        "stage_acc": stage_accuracy(df["stage_true"], df["stage_pred"]),
        **{f"recall_{name.lower()}": float(r) for name, r in zip(config.STAGE_NAMES, recall)},
        **head_agreement(df),
    }
