"""Metrics and plots for finished training runs.

    python -m evaluation.report cnn_real cnn_lstm_real

For each run in artifacts/runs/<run>/ it reads predictions.csv (test split), loads
best.pt to predict on the validation split, and writes into artifacts/runs/<run>/eval/:
    metrics.json            test and val metrics
    val_predictions.csv     contract 3, every window of the validation engines
    rul_scatter.png, rul_error.png, confusion.png        test split
    health_over_life.png, rul_over_life.png              validation engines
Then prints one table comparing the runs.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from contracts import check_predictions
from evaluation import plots
from evaluation.metrics import summarize

RUNS_DIR = os.path.join("artifacts", "runs")
TABLE_COLUMNS = ("rul_rmse", "rul_mae", "nasa_score", "health_rmse", "stage_acc")


def load_predictions(run_dir, name="predictions.csv"):
    """Returns (DataFrame, attention array or None) for one run."""
    df = pd.read_csv(os.path.join(run_dir, name))
    attn_path = os.path.join(run_dir, "attn.npy")
    attn = np.load(attn_path) if name == "predictions.csv" and os.path.exists(attn_path) else None
    check_predictions(df, attn)
    return df, attn


def val_predictions(run_dir):
    """Predict on every validation window with the run's best checkpoint."""
    from data.preprocess import load_split
    from models.train import load_checkpoint, make_predictions

    model, _ = load_checkpoint(os.path.join(run_dir, "best.pt"))
    df, _ = make_predictions(model, load_split("val"))
    return df


def pick_engines(df, k=6):
    """k engines spread from the shortest to the longest life, for the over-life plots."""
    lives = df.groupby("engine_id")["cycle"].max().sort_values()
    idx = np.linspace(0, len(lives) - 1, min(k, len(lives))).round().astype(int)
    return lives.index[idx].tolist()


def _save(fig, path):
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def evaluate_run(run_dir, with_val=True):
    """Writes the eval/ folder for one run. Returns {'test': metrics, 'val': metrics or None}."""
    out_dir = os.path.join(run_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.basename(os.path.normpath(run_dir))

    test_df, _ = load_predictions(run_dir)
    metrics = {"test": summarize(test_df), "val": None}
    _save(plots.plot_rul_scatter(test_df, title=f"{name}: predicted vs actual RUL (test)"),
          os.path.join(out_dir, "rul_scatter.png"))
    _save(plots.plot_rul_error_hist(test_df, title=f"{name}: RUL error (test)"),
          os.path.join(out_dir, "rul_error.png"))
    _save(plots.plot_confusion(test_df, title=f"{name}: health stage (test)"),
          os.path.join(out_dir, "confusion.png"))

    if with_val:
        val_df = val_predictions(run_dir)
        val_df.to_csv(os.path.join(out_dir, "val_predictions.csv"), index=False)
        metrics["val"] = summarize(val_df)
        engines = pick_engines(val_df)
        _save(plots.plot_health_grid(val_df, engines), os.path.join(out_dir, "health_over_life.png"))
        fig, axes = plt.subplots(2, 3, figsize=(15, 6.4))
        for ax, eid in zip(axes.flat, engines):
            plots.plot_rul_over_life(val_df, eid, ax=ax, title=f"Engine {eid}")
        fig.tight_layout()
        _save(fig, os.path.join(out_dir, "rul_over_life.png"))

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def comparison_table(results, split="test"):
    """results maps run name -> evaluate_run output. Returns a DataFrame, one row per run."""
    rows = {name: m[split] for name, m in results.items() if m[split] is not None}
    return pd.DataFrame(rows).T[list(TABLE_COLUMNS)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="run folder names under artifacts/runs/, e.g. cnn_real")
    parser.add_argument("--no-val", action="store_true", help="skip validation predictions and life plots")
    args = parser.parse_args()

    results = {run: evaluate_run(os.path.join(RUNS_DIR, run), with_val=not args.no_val) for run in args.runs}
    for split in ("test", "val"):
        table = comparison_table(results, split)
        if len(table):
            print(f"\n{split} split ({int(next(iter(results.values()))[split]['n'])} rows)")
            print(table.round(3).to_string())
    print(f"\nwrote metrics.json and plots to artifacts/runs/<run>/eval/ for {', '.join(args.runs)}")
