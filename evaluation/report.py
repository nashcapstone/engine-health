"""Metrics and plots for finished training runs.

    python -m evaluation.report cnn_real cnn_lstm_real
    python fakes.py && python -m evaluation.report artifacts/fake --no-val

A run is a folder name under artifacts/runs/, or any folder holding predictions.csv.
For each run it reads predictions.csv (test split), loads best.pt to predict on the
validation split, and writes into <run>/eval/:
    metrics.json            test and val metrics, plus RUL errors by stage
    val_predictions.csv     contract 3, every window of the validation engines
    val_attn.npy            only for models with attention
    rul_scatter.png, rul_error.png, confusion.png        test split
    error_by_rul.png                                     validation split, else test
    health_over_life.png, rul_over_life.png              validation engines
    attn_by_stage.png, attn_over_time.png                only for models with attention
    attn_over_life.png                                   same, validation engines
Then prints one table comparing the runs.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from contracts import check_predictions
from evaluation import attention, plots
from evaluation.metrics import errors_by_stage, summarize

RUNS_DIR = os.path.join("artifacts", "runs")
TABLE_COLUMNS = ("rul_rmse", "rul_mae", "nasa_score", "health_rmse", "stage_acc")


def resolve_run(run):
    """A folder name under artifacts/runs/, or a path to any run folder."""
    return run if os.path.isdir(run) else os.path.join(RUNS_DIR, run)


def load_predictions(run_dir, name="predictions.csv"):
    """Returns (DataFrame, attention array or None) for one run."""
    df = pd.read_csv(os.path.join(run_dir, name))
    attn_path = os.path.join(run_dir, "attn.npy")
    attn = np.load(attn_path) if name == "predictions.csv" and os.path.exists(attn_path) else None
    check_predictions(df, attn)
    return df, attn


def val_predictions(run_dir):
    """Predict on every validation window with the run's best checkpoint. Returns (df, attn or None)."""
    from data.preprocess import load_split
    from models.train import load_checkpoint, make_predictions

    model, _ = load_checkpoint(os.path.join(run_dir, "best.pt"))
    return make_predictions(model, load_split("val"))


def pick_engines(df, k=6):
    """k engines spread from the shortest to the longest life, for the over-life plots."""
    lives = df.groupby("engine_id")["cycle"].max().sort_values()
    idx = np.linspace(0, len(lives) - 1, min(k, len(lives))).round().astype(int)
    return lives.index[idx].tolist()


def _save(fig, path):
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _metrics(df, attn):
    out = summarize(df)
    out["errors_by_stage"] = errors_by_stage(df).to_dict(orient="index")
    if attn is not None:
        by_stage = attention.attention_by_stage(df, attn)
        out["attention_by_stage"] = by_stage.to_dict(orient="index")
        out["top_sensors"] = attention.top_sensors(by_stage)
    return out


def _attention_plots(df, attn, out_dir, name, split):
    _save(attention.plot_attention_by_stage(df, attn, title=f"{name}: attention by stage ({split})"),
          os.path.join(out_dir, "attn_by_stage.png"))
    _save(attention.plot_attention_over_time(attn, title=f"{name}: attention across the window ({split})"),
          os.path.join(out_dir, "attn_over_time.png"))


def evaluate_run(run_dir, with_val=True):
    """Writes the eval/ folder for one run. Returns {'test': metrics, 'val': metrics or None}."""
    out_dir = os.path.join(run_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.basename(os.path.normpath(run_dir))

    test_df, test_attn = load_predictions(run_dir)
    metrics = {"test": _metrics(test_df, test_attn), "val": None}
    _save(plots.plot_rul_scatter(test_df, title=f"{name}: predicted vs actual RUL (test)"),
          os.path.join(out_dir, "rul_scatter.png"))
    _save(plots.plot_rul_error_hist(test_df, title=f"{name}: RUL error (test)"),
          os.path.join(out_dir, "rul_error.png"))
    _save(plots.plot_confusion(test_df, title=f"{name}: health stage (test)"),
          os.path.join(out_dir, "confusion.png"))

    if not with_val:
        _save(plots.plot_error_by_rul(test_df, title=f"{name}: RUL error by actual RUL (test)"),
              os.path.join(out_dir, "error_by_rul.png"))
        if test_attn is not None:
            _attention_plots(test_df, test_attn, out_dir, name, "test")
    else:
        val_df, val_attn = val_predictions(run_dir)
        val_df.to_csv(os.path.join(out_dir, "val_predictions.csv"), index=False)
        metrics["val"] = _metrics(val_df, val_attn)
        _save(plots.plot_error_by_rul(val_df, title=f"{name}: RUL error by actual RUL (val)"),
              os.path.join(out_dir, "error_by_rul.png"))
        engines = pick_engines(val_df)
        _save(plots.plot_health_grid(val_df, engines), os.path.join(out_dir, "health_over_life.png"))
        fig, axes = plt.subplots(2, 3, figsize=(15, 6.4))
        for ax, eid in zip(axes.flat, engines):
            plots.plot_rul_over_life(val_df, eid, ax=ax, title=f"Engine {eid}")
        fig.tight_layout()
        _save(fig, os.path.join(out_dir, "rul_over_life.png"))
        if val_attn is not None:
            np.save(os.path.join(out_dir, "val_attn.npy"), val_attn)
            _attention_plots(val_df, val_attn, out_dir, name, "val")
            _save(attention.plot_attention_over_life(val_df, val_attn, engines[-1]),
                  os.path.join(out_dir, "attn_over_life.png"))

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def comparison_table(results, split="test"):
    """results maps run name -> evaluate_run output. Returns a DataFrame, one row per run."""
    rows = {name: m[split] for name, m in results.items() if m[split] is not None}
    if not rows:
        return pd.DataFrame(columns=list(TABLE_COLUMNS), dtype=float)
    return pd.DataFrame(rows).T[list(TABLE_COLUMNS)].astype(float)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="run folder names under artifacts/runs/, or paths")
    parser.add_argument("--no-val", action="store_true", help="skip validation predictions and life plots")
    args = parser.parse_args()

    results = {run: evaluate_run(resolve_run(run), with_val=not args.no_val) for run in args.runs}
    for split in ("test", "val"):
        table = comparison_table(results, split)
        if len(table):
            print(f"\n{split} split ({int(next(iter(results.values()))[split]['n'])} rows)")
            print(table.round(3).to_string())
            for run, m in results.items():
                print(f"\n{run}, RUL error by true stage ({split}):")
                print(pd.DataFrame(m[split]["errors_by_stage"]).T.round(2).to_string())
                if "top_sensors" in m[split]:
                    print(f"top attention sensors: {m[split]['top_sensors']}")
    print(f"\nwrote metrics.json and plots to <run>/eval/ for {', '.join(args.runs)}")
