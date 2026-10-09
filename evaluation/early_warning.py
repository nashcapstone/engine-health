"""Early warning: which sensors gain attention while RUL is still high.

    python -m evaluation.early_warning attention_real

Two views, both on every window of the validation engines:
  attention  mean attention per sensor in bins of true RUL. A sensor's onset is the
             highest RUL from which its attention stays above its HEALTHY baseline
             (RUL > 100) by ONSET_RISE, all the way down to failure.
  signal     the same for the raw reading: drift from each engine's own first window,
             in units of that engine's spread. Onset is where |drift| stays above
             ONSET_DRIFT. This is what the sensors actually do, independent of any model.
Comparing the two shows whether the model watches the sensors that degrade first.

Writes into artifacts/runs/<run>/eval/: early_warning.csv and early_warning.png.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from contracts import check_attention
from evaluation.attention import UNIFORM, sensor_attention

BIN_WIDTH = 10
ONSET_RISE = 1.10
ONSET_DRIFT = 1.0


def _bins(rul, bin_width=BIN_WIDTH):
    """Lower edge of each row's RUL bin. RUL_CAP gets its own bin with the flat, capped rows."""
    return (np.minimum(np.asarray(rul), config.RUL_CAP) // bin_width * bin_width).astype(int)


def attention_by_rul(df, attn, bin_width=BIN_WIDTH):
    """DataFrame: one row per RUL bin (lower edge, descending), one column per sensor."""
    check_attention(attn, len(df))
    per_row = pd.DataFrame(sensor_attention(attn), columns=list(config.SENSORS))
    per_row["rul_bin"] = _bins(df["rul_true"], bin_width)
    return per_row.groupby("rul_bin").mean().sort_index(ascending=False)


def signal_drift_by_rul(batch, bin_width=BIN_WIDTH):
    """Mean drift per sensor in bins of true RUL, from the raw (scaled) readings.

    Drift is each window's last reading minus the engine's first window mean, divided by
    the engine's first window std. Positive means the reading rose with wear.
    """
    X, engine = batch["X"], batch["engine_id"]
    last = X[:, -1, :]
    drift = np.empty_like(last)
    for eid in np.unique(engine):
        rows = np.where(engine == eid)[0]
        first = X[rows[np.argmin(batch["cycle"][rows])]]
        drift[rows] = (last[rows] - first.mean(axis=0)) / np.maximum(first.std(axis=0), 1e-3)
    out = pd.DataFrame(drift, columns=list(config.SENSORS))
    out["rul_bin"] = _bins(batch["y_rul"], bin_width)
    return out.groupby("rul_bin").mean().sort_index(ascending=False)


def onset(curve, above):
    """Highest RUL bin from which `above(curve)` holds in every lower bin. NaN if never."""
    ok = above(curve).sort_index(ascending=True)
    result = {}
    for sensor in curve.columns:
        flags = ok[sensor].to_numpy()
        # walk up from failure while the condition holds
        n = 0
        while n < len(flags) and flags[n]:
            n += 1
        result[sensor] = float(ok.index[n - 1]) if n else np.nan
    return pd.Series(result)


def early_warning_table(att_curve, drift_curve):
    """One row per sensor, sorted by attention onset (earliest warning first)."""
    baseline = att_curve.loc[att_curve.index > config.HEALTHY_ABOVE].mean()
    early = att_curve.loc[(att_curve.index >= 60) & (att_curve.index < config.HEALTHY_ABOVE)].mean()
    late = att_curve.loc[att_curve.index < config.CRITICAL_BELOW].mean()
    table = pd.DataFrame(
        {
            "attn_healthy": baseline,
            "attn_rul_60_100": early,
            "attn_critical": late,
            "early_rise": early / baseline,
            "attn_onset_rul": onset(att_curve, lambda c: c > baseline * ONSET_RISE),
            "signal_onset_rul": onset(drift_curve, lambda c: c.abs() > ONSET_DRIFT),
            "signal_direction": np.sign(drift_curve.loc[drift_curve.index < config.CRITICAL_BELOW].mean()),
        }
    )
    return table.sort_values(["attn_onset_rul", "early_rise"], ascending=False)


def plot_early_warning(att_curve, drift_curve, table, k=4, title="Early warning"):
    """Left: attention vs RUL. Right: signal drift vs RUL. The k earliest sensors are coloured."""
    highlight = table.index[:k].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    for ax, curve, ylabel, ref in (
        (axes[0], att_curve, "Mean attention", UNIFORM),
        (axes[1], drift_curve.abs(), "|Drift| from the engine's first window (std units)", ONSET_DRIFT),
    ):
        x = curve.index + BIN_WIDTH / 2
        for sensor in curve.columns:
            if sensor in highlight:
                ax.plot(x, curve[sensor], lw=2, marker="o", ms=3, label=sensor)
            else:
                ax.plot(x, curve[sensor], lw=0.8, color="grey", alpha=0.5)
        ax.axhline(ref, color="black", lw=1, ls="--")
        for edge in (config.CRITICAL_BELOW, config.HEALTHY_ABOVE):
            ax.axvline(edge, color="grey", lw=1, ls=":")
        ax.set_xlim(config.RUL_CAP + BIN_WIDTH, 0)
        ax.set_xlabel("True RUL (cycles), failure on the right")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, title=f"earliest {k} by attention", title_fontsize=8)
    axes[0].set_title(f"{title}: attention as the engine wears (dashed = uniform)")
    axes[1].set_title(f"{title}: what the sensors actually do (dashed = onset level)")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    from data.preprocess import load_split
    from models.train import load_checkpoint, make_predictions

    from evaluation.report import resolve_run

    parser = argparse.ArgumentParser()
    parser.add_argument("run", help="run folder name under artifacts/runs/, or a path")
    args = parser.parse_args()

    run_dir = resolve_run(args.run)
    out_dir = os.path.join(run_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)
    batch = load_split("val")
    model, _ = load_checkpoint(os.path.join(run_dir, "best.pt"))
    df, attn = make_predictions(model, batch)
    assert attn is not None, f"{run_dir} has no attention: use an attention model"

    att_curve = attention_by_rul(df, attn)
    drift_curve = signal_drift_by_rul(batch)
    table = early_warning_table(att_curve, drift_curve)
    table.to_csv(os.path.join(out_dir, "early_warning.csv"))
    fig = plot_early_warning(att_curve, drift_curve, table, title=os.path.basename(run_dir))
    fig.savefig(os.path.join(out_dir, "early_warning.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(table.round(3).to_string())
    print(f"wrote early_warning.csv and early_warning.png to {out_dir}")
