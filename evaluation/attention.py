"""Attention analysis on attn.npy (contract 3): shape (N, 30, 14), rows in predictions.csv order.

Attention sums to 1 across the 14 sensors at each timestep, so 1/14 is the "no
preference" level. Built on fakes.fake_predictions() until the attention model lands.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

import config
from contracts import check_attention

UNIFORM = 1.0 / config.N_SENSORS


def sensor_attention(attn):
    """(N, WINDOW, N_SENSORS) -> (N, N_SENSORS): each row's attention averaged over its timesteps."""
    return np.asarray(attn).mean(axis=1)


def attention_by_stage(df, attn, stage_col="stage_true"):
    """Mean attention per sensor in each health stage.

    Returns a DataFrame, one row per stage name and one column per sensor. A stage
    with no rows is NaN.
    """
    check_attention(attn, len(df))
    per_row = sensor_attention(attn)
    stages = df[stage_col].to_numpy()
    rows = [
        per_row[stages == s].mean(axis=0) if (stages == s).any() else np.full(config.N_SENSORS, np.nan)
        for s in range(config.N_STAGES)
    ]
    return pd.DataFrame(rows, index=list(config.STAGE_NAMES), columns=list(config.SENSORS))


def attention_over_time(attn):
    """(WINDOW, N_SENSORS): mean attention at each timestep of the window, across all rows."""
    return np.asarray(attn).mean(axis=0)


def top_sensors(by_stage, k=3):
    """{stage name: [k sensor names with the highest mean attention]} from attention_by_stage."""
    return {stage: row.sort_values(ascending=False).index[:k].tolist() for stage, row in by_stage.iterrows()}


def stage_shift(by_stage):
    """Per sensor, CRITICAL minus HEALTHY attention. Positive: the sensor gains attention near failure."""
    return (by_stage.loc["CRITICAL"] - by_stage.loc["HEALTHY"]).sort_values(ascending=False)


def _heatmap(ax, values, xlabels, ylabels, center=UNIFORM, cmap="RdBu_r", annotate=False):
    """Colours from 0 (blue) through `center` (white) to the max (red): red is above-uniform attention."""
    values = np.asarray(values, dtype=float)
    top = max(np.nanmax(values), center * 1.01)
    norm = TwoSlopeNorm(vmin=0.0, vcenter=center, vmax=top)
    im = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(xlabels)), xlabels, fontsize=8)
    ax.set_yticks(range(len(ylabels)), ylabels, fontsize=8)
    if annotate:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                dark = norm(values[i, j]) > 0.8 or norm(values[i, j]) < 0.2
                ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=6,
                        color="white" if dark else "black")
    return im


def plot_window_attention(attn_row, ax=None, title="Attention in one window"):
    """Heatmap of one window: timesteps down, sensors across."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    ticks = [str(t) if t % 5 == 0 else "" for t in range(1, config.WINDOW + 1)]
    im = _heatmap(ax, attn_row, list(config.SENSORS), ticks)
    ax.set_xlabel("Sensor")
    ax.set_ylabel("Timestep in window (30 = latest)")
    ax.set_title(title)
    ax.figure.colorbar(im, ax=ax, label="Attention (1/14 = uniform)")
    return ax.figure


def plot_attention_by_stage(df, attn, ax=None, title="Mean attention per sensor by health stage"):
    """3 x 14 heatmap: which sensors the model looks at in each stage."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))
    by_stage = attention_by_stage(df, attn)
    im = _heatmap(ax, by_stage.to_numpy(), list(config.SENSORS), list(config.STAGE_NAMES), annotate=True)
    ax.set_xlabel("Sensor")
    ax.set_title(title)
    ax.figure.colorbar(im, ax=ax, label="Attention (1/14 = uniform)")
    return ax.figure


def plot_attention_over_time(attn, ax=None, title="Mean attention across the window"):
    """30 x 14 heatmap, averaged over all rows: does attention move toward recent cycles?"""
    fig = plot_window_attention(attention_over_time(attn), ax=ax, title=title)
    return fig


def plot_attention_over_life(df, attn, engine_id, ax=None, title=None):
    """One engine, cycle down the y axis, sensors across: how attention shifts as it wears.

    Needs several rows per engine, so use the validation split.
    """
    check_attention(attn, len(df))
    mask = (df["engine_id"] == engine_id).to_numpy()
    assert mask.sum() > 1, f"engine {engine_id} has {mask.sum()} rows, need several: use the val split"
    order = np.argsort(df["cycle"].to_numpy()[mask])
    cycles = df["cycle"].to_numpy()[mask][order]
    per_row = sensor_attention(attn[mask][order])
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    im = _heatmap(ax, per_row, list(config.SENSORS), [str(c) for c in cycles])
    # One tick per row is unreadable for a long engine: keep about 10
    step = max(1, len(cycles) // 10)
    ax.set_yticks(range(0, len(cycles), step), [str(c) for c in cycles[::step]], fontsize=8)
    ax.set_xlabel("Sensor")
    ax.set_ylabel("Cycle")
    ax.set_title(title or f"Engine {engine_id}: attention over life")
    ax.figure.colorbar(im, ax=ax, label="Attention (1/14 = uniform)")
    return ax.figure
