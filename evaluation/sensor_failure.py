"""Sensor failure demo: corrupt one sensor, see how much the predictions move.

    python -m evaluation.sensor_failure attention_real

If attention is a faithful explanation, the sensors with high attention should be the
ones whose failure moves the prediction most. For each sensor and each failure mode it
measures the change in predicted RUL, then compares that against the sensor's attention.

Writes into artifacts/runs/<run>/eval/: sensor_failure.csv and sensor_failure.png.
Inputs are min-max scaled, so normal readings lie in about [0, 1].
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from evaluation.attention import UNIFORM, sensor_attention
from evaluation.metrics import rmse

NOISE_STD = 0.3


def corrupt(batch, sensor, mode, seed=config.SEED):
    """Copy of a contract 1 dict with one sensor broken in every window.

    stuck  the sensor holds its first reading of the window for all 30 timesteps
    noise  Gaussian noise with std NOISE_STD is added
    dead   the sensor reads 0, the training minimum
    """
    j = config.SENSORS.index(sensor)
    X = batch["X"].copy()
    if mode == "stuck":
        X[:, :, j] = X[:, :1, j]
    elif mode == "noise":
        X[:, :, j] += np.random.default_rng(seed).normal(0, NOISE_STD, X[:, :, j].shape).astype(np.float32)
    elif mode == "dead":
        X[:, :, j] = 0.0
    else:
        raise ValueError(f"unknown mode {mode!r}, expected stuck, noise or dead")
    return {**batch, "X": X}


def failure_table(model, batch, modes=("stuck", "noise", "dead")):
    """One row per (mode, sensor).

    Columns: attention (clean attention on that sensor, NaN without attention),
    attention_after (its attention once broken), mean_abs_change and mean_change in
    predicted RUL, rmse_clean and rmse_broken against the true RUL.
    """
    from models.train import make_predictions

    clean_df, clean_attn = make_predictions(model, batch)
    clean_rul = clean_df["rul_pred"].to_numpy()
    clean_rmse = rmse(clean_df["rul_true"], clean_rul)
    clean_by_sensor = sensor_attention(clean_attn).mean(axis=0) if clean_attn is not None else None

    rows = []
    for mode in modes:
        for j, sensor in enumerate(config.SENSORS):
            df, attn = make_predictions(model, corrupt(batch, sensor, mode))
            change = df["rul_pred"].to_numpy() - clean_rul
            rows.append(
                {
                    "mode": mode,
                    "sensor": sensor,
                    "attention": clean_by_sensor[j] if clean_by_sensor is not None else np.nan,
                    "attention_after": sensor_attention(attn).mean(axis=0)[j] if attn is not None else np.nan,
                    "mean_abs_change": float(np.abs(change).mean()),
                    "mean_change": float(change.mean()),
                    "rmse_clean": clean_rmse,
                    "rmse_broken": rmse(df["rul_true"], df["rul_pred"]),
                }
            )
    return pd.DataFrame(rows)


def rank_agreement(table):
    """Spearman correlation between attention and mean_abs_change, per mode. NaN without attention."""
    return {
        mode: float(part["attention"].rank().corr(part["mean_abs_change"].rank()))
        for mode, part in table.groupby("mode", sort=False)
    }


def plot_failure(table, title="Sensor failure: change in predicted RUL"):
    """Left: change per sensor and mode. Right: attention vs change, one point per sensor."""
    modes = list(dict.fromkeys(table["mode"]))
    has_attn = table["attention"].notna().all()
    fig, axes = plt.subplots(1, 2 if has_attn else 1, figsize=(15 if has_attn else 9, 4.5), squeeze=False)
    ax = axes[0, 0]
    width = 0.8 / len(modes)
    x = np.arange(config.N_SENSORS)
    for i, mode in enumerate(modes):
        part = table[table["mode"] == mode].set_index("sensor").loc[list(config.SENSORS)]
        ax.bar(x + (i - (len(modes) - 1) / 2) * width, part["mean_abs_change"], width, label=mode)
    ax.set_xticks(x, config.SENSORS, fontsize=8)
    ax.set_xlabel("Broken sensor")
    ax.set_ylabel("Mean |change| in predicted RUL (cycles)")
    ax.set_title(title)
    ax.legend(fontsize=8)

    if has_attn:
        ax = axes[0, 1]
        agreement = rank_agreement(table)
        for mode in modes:
            part = table[table["mode"] == mode]
            ax.scatter(part["attention"], part["mean_abs_change"], s=25, label=f"{mode} (rank corr {agreement[mode]:.2f})")
        part = table[table["mode"] == modes[0]]
        for _, r in part.iterrows():
            ax.annotate(r["sensor"], (r["attention"], r["mean_abs_change"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        ax.axvline(UNIFORM, color="grey", lw=1, ls="--")
        ax.set_xlabel("Attention on the sensor (clean data, dashed = uniform)")
        ax.set_ylabel("Mean |change| in predicted RUL (cycles)")
        ax.set_title("Does attention point at the sensors that matter?")
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    from data.preprocess import load_split
    from models.train import load_checkpoint

    from evaluation.report import resolve_run

    parser = argparse.ArgumentParser()
    parser.add_argument("run", help="run folder name under artifacts/runs/, or a path")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    args = parser.parse_args()

    run_dir = resolve_run(args.run)
    out_dir = os.path.join(run_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)
    model, _ = load_checkpoint(os.path.join(run_dir, "best.pt"))
    table = failure_table(model, load_split(args.split))
    table.to_csv(os.path.join(out_dir, "sensor_failure.csv"), index=False)
    fig = plot_failure(table, title=f"{os.path.basename(run_dir)}: sensor failure ({args.split})")
    fig.savefig(os.path.join(out_dir, "sensor_failure.png"), dpi=120, bbox_inches="tight")

    print(f"clean RUL RMSE {table['rmse_clean'].iloc[0]:.2f} on {args.split}")
    for mode, part in table.groupby("mode", sort=False):
        top = part.sort_values("mean_abs_change", ascending=False).head(4)
        moves = ", ".join(f"{r.sensor} {r.mean_abs_change:.1f}" for r in top.itertuples())
        print(f"{mode:5s}  biggest moves: {moves}  (RMSE up to {part['rmse_broken'].max():.2f})")
    print(f"rank agreement attention vs change: {rank_agreement(table)}")
    print(f"wrote sensor_failure.csv and sensor_failure.png to {out_dir}")
