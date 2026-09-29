"""Plots on the predictions file (contract 3).

Each function draws on `ax` if given, else on a new figure, and returns the figure.
Save with fig.savefig(path).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from evaluation.metrics import confusion_matrix
from labels import health_score

STAGE_COLORS = ("#2e7d32", "#f9a825", "#c62828")


def _axes(ax, figsize=(6, 5)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    return ax.figure, ax


def _stage_bands(ax, boundaries, limits):
    """Shade the HEALTHY / WARNING / CRITICAL regions and set the y limits.

    boundaries is (critical, healthy) on the y scale, limits is (bottom, top).
    """
    critical, healthy = boundaries
    bottom, top = limits
    for (lo, hi), color in zip(((healthy, top), (critical, healthy), (bottom, critical)), STAGE_COLORS):
        ax.axhspan(lo, hi, color=color, alpha=0.08, zorder=0)
    ax.set_ylim(bottom, top)


def plot_rul_scatter(df, ax=None, title="Predicted vs actual RUL"):
    """One point per row, coloured by true stage, with the perfect-prediction diagonal."""
    fig, ax = _axes(ax)
    for stage, (name, color) in enumerate(zip(config.STAGE_NAMES, STAGE_COLORS)):
        rows = df[df["stage_true"] == stage]
        ax.scatter(rows["rul_true"], rows["rul_pred"], s=14, alpha=0.7, color=color, label=name)
    ax.plot([0, config.RUL_CAP], [0, config.RUL_CAP], color="black", lw=1, ls="--")
    ax.set_xlim(-3, config.RUL_CAP + 3)
    ax.set_ylim(-3, config.RUL_CAP + 3)
    ax.set_aspect("equal")
    ax.set_xlabel("Actual RUL (cycles)")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    return fig


def plot_rul_error_hist(df, ax=None, bins=30, title="RUL error (predicted - actual)"):
    """Positive error is a late prediction, which the NASA score penalises more."""
    fig, ax = _axes(ax)
    err = df["rul_pred"].to_numpy() - df["rul_true"].to_numpy()
    ax.hist(err, bins=bins, color="#546e7a", alpha=0.85)
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_xlabel("Error (cycles)   < early | late >")
    ax.set_ylabel("Rows")
    ax.set_title(title)
    return fig


def plot_error_by_rul(df, bin_width=10, ax=None, title="RUL error by actual RUL"):
    """Mean error (bias) and RMSE in bins of actual RUL, with the stage regions shaded."""
    fig, ax = _axes(ax, figsize=(8, 4))
    rul_true = df["rul_true"].to_numpy()
    err = df["rul_pred"].to_numpy() - rul_true
    edges = np.arange(0, config.RUL_CAP + bin_width, bin_width)
    idx = np.clip(np.digitize(rul_true, edges) - 1, 0, len(edges) - 2)
    centers, bias, rmse = [], [], []
    for b in np.unique(idx):
        e = err[idx == b]
        centers.append(edges[b] + bin_width / 2)
        bias.append(e.mean())
        rmse.append(np.sqrt((e**2).mean()))
    for (lo, hi), color in zip(
        ((config.HEALTHY_ABOVE, config.RUL_CAP), (config.CRITICAL_BELOW, config.HEALTHY_ABOVE), (0, config.CRITICAL_BELOW)),
        STAGE_COLORS,
    ):
        ax.axvspan(lo, hi, color=color, alpha=0.08, zorder=0)
    ax.bar(centers, bias, width=bin_width * 0.8, color="#546e7a", alpha=0.8, label="Mean error (bias)")
    ax.plot(centers, rmse, color="#c62828", marker="o", ms=4, label="RMSE")
    ax.axhline(0, color="black", lw=1)
    ax.set_xlim(0, config.RUL_CAP)
    ax.set_xlabel("Actual RUL (cycles)")
    ax.set_ylabel("Cycles   (bias > 0 is late)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    return fig


def plot_health_over_life(df, engine_id, ax=None, title=None):
    """True and predicted health score across one engine's windows.

    Needs several rows per engine, so use predictions on the validation split,
    not the test split (one row per engine).
    """
    rows = df[df["engine_id"] == engine_id].sort_values("cycle")
    assert len(rows) > 1, f"engine {engine_id} has {len(rows)} rows, need several: use the val split"
    fig, ax = _axes(ax, figsize=(8, 4))
    _stage_bands(ax, (health_score(config.CRITICAL_BELOW), health_score(config.HEALTHY_ABOVE)), (-3, 103))
    ax.plot(rows["cycle"], rows["health_true"], color="black", lw=1.5, label="Actual")
    ax.plot(rows["cycle"], rows["health_pred"], color="#1565c0", lw=1.2, label="Predicted")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Health score")
    ax.set_title(title or f"Engine {engine_id}: health score over life")
    ax.legend(loc="lower left", fontsize=8)
    return fig


def plot_rul_over_life(df, engine_id, ax=None, title=None):
    """True and predicted RUL across one engine's windows. Same data needs as health over life."""
    rows = df[df["engine_id"] == engine_id].sort_values("cycle")
    assert len(rows) > 1, f"engine {engine_id} has {len(rows)} rows, need several: use the val split"
    fig, ax = _axes(ax, figsize=(8, 4))
    _stage_bands(ax, (config.CRITICAL_BELOW, config.HEALTHY_ABOVE), (-3, config.RUL_CAP + 3))
    ax.plot(rows["cycle"], rows["rul_true"], color="black", lw=1.5, label="Actual")
    ax.plot(rows["cycle"], rows["rul_pred"], color="#1565c0", lw=1.2, label="Predicted")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("RUL (cycles)")
    ax.set_title(title or f"Engine {engine_id}: RUL over life")
    ax.legend(loc="lower left", fontsize=8)
    return fig


def plot_health_grid(df, engine_ids, ncols=3):
    """plot_health_over_life for several engines, one panel each."""
    nrows = int(np.ceil(len(engine_ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows), squeeze=False)
    for ax, eid in zip(axes.flat, engine_ids):
        plot_health_over_life(df, eid, ax=ax, title=f"Engine {eid}")
    for ax in axes.flat[len(engine_ids) :]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def plot_confusion(df, ax=None, title="Health stage"):
    """Confusion matrix, rows true stage, columns predicted stage, with counts."""
    fig, ax = _axes(ax, figsize=(4.5, 4))
    cm = confusion_matrix(df["stage_true"], df["stage_pred"])
    ax.imshow(cm, cmap="Blues")
    for i in range(config.N_STAGES):
        for j in range(config.N_STAGES):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)
    ticks = range(config.N_STAGES)
    ax.set_xticks(ticks, config.STAGE_NAMES, fontsize=8)
    ax.set_yticks(ticks, config.STAGE_NAMES, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    return fig
