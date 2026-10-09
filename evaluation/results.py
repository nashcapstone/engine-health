"""Results table across the ablation runs: mean and std over seeds, per variant.

    python -m evaluation.results              after `python -m models.ablation`

Reads artifacts/runs/ablation/<variant>_s<seed>/ and writes into artifacts/runs/ablation/:
    results_runs.csv      one row per run and split
    results.csv           mean and std per variant and split
    results.md            the validation and test tables as markdown
    attention_seeds.csv   attention per sensor for every attention run (validation)
    results.png           validation RUL RMSE per variant, one dot per seed
"""

import argparse
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from evaluation.attention import sensor_attention
from evaluation.metrics import summarize
from labels import health_stage

ABLATION_DIR = os.path.join(config.ARTIFACTS_DIR, "runs", "ablation")
RUN_NAME = re.compile(r"^(?P<variant>.+)_s(?P<seed>\d+)$")
SPLIT_FILES = {"val": ("val_predictions.csv", "val_attn.npy"), "test": ("predictions.csv", "attn.npy")}
METRICS = ("rul_rmse", "rul_mae", "nasa_score", "health_rmse", "stage_acc", "stage_acc_from_rul", "recall_warning")
# Lower is better for these, higher for the rest
LOWER_IS_BETTER = {"rul_rmse", "rul_mae", "nasa_score", "health_rmse"}


def find_runs(ablation_dir=ABLATION_DIR):
    """{(variant, seed): run folder} for every finished run."""
    runs = {}
    for name in sorted(os.listdir(ablation_dir)):
        m = RUN_NAME.match(name)
        path = os.path.join(ablation_dir, name)
        if m and os.path.exists(os.path.join(path, "predictions.csv")):
            runs[(m["variant"], int(m["seed"]))] = path
    return runs


def run_metrics(df):
    out = summarize(df)
    # Stage read off the RUL head: the fair comparison for variants trained without the stage loss
    out["stage_acc_from_rul"] = float(np.mean(health_stage(df["rul_pred"].to_numpy()) == df["stage_true"].to_numpy()))
    return {k: out[k] for k in METRICS}


def collect(runs):
    """(per-run DataFrame, attention per sensor DataFrame for attention runs on validation)."""
    rows, attn_rows = [], []
    for (variant, seed), path in runs.items():
        for split, (pred_file, attn_file) in SPLIT_FILES.items():
            pred_path = os.path.join(path, pred_file)
            if not os.path.exists(pred_path):
                continue
            rows.append({"variant": variant, "seed": seed, "split": split, **run_metrics(pd.read_csv(pred_path))})
            attn_path = os.path.join(path, attn_file)
            if split == "val" and os.path.exists(attn_path):
                per_sensor = sensor_attention(np.load(attn_path)).mean(axis=0)
                attn_rows.append({"variant": variant, "seed": seed, **dict(zip(config.SENSORS, per_sensor))})
    return pd.DataFrame(rows), pd.DataFrame(attn_rows)


def aggregate(per_run, order=None):
    """Mean and std over seeds. Index (split, variant), columns (metric, mean|std) plus n_seeds."""
    grouped = per_run.groupby(["split", "variant"])[list(METRICS)]
    table = grouped.agg(["mean", "std"])
    table["n_seeds"] = per_run.groupby(["split", "variant"]).size()
    if order:
        table = table.reindex([(s, v) for s in ("val", "test") for v in order if (s, v) in table.index])
    return table


def attention_stability(attn_seeds):
    """Per attention variant: mean Spearman correlation of per-sensor attention between seed pairs."""
    out = {}
    if attn_seeds.empty:
        return out
    for variant, part in attn_seeds.groupby("variant"):
        vectors = part[list(config.SENSORS)].to_numpy()
        corrs = [
            pd.Series(vectors[i]).rank().corr(pd.Series(vectors[j]).rank())
            for i in range(len(vectors))
            for j in range(i + 1, len(vectors))
        ]
        out[variant] = float(np.mean(corrs)) if corrs else np.nan
    return out


def to_markdown(table, split):
    """One split as a markdown table, 'mean ± std', best value per column in bold."""
    part = table.loc[split]
    lines = ["| Variant | " + " | ".join(METRICS) + " | seeds |", "|---" * (len(METRICS) + 2) + "|"]
    best = {
        m: (part[(m, "mean")].idxmin() if m in LOWER_IS_BETTER else part[(m, "mean")].idxmax()) for m in METRICS
    }
    for variant, row in part.iterrows():
        cells = []
        for m in METRICS:
            digits = 0 if m == "nasa_score" else 3 if "acc" in m or "recall" in m else 2
            cell = f"{row[(m, 'mean')]:.{digits}f} ± {row[(m, 'std')]:.{digits}f}"
            cells.append(f"**{cell}**" if best[m] == variant else cell)
        lines.append(f"| {variant} | " + " | ".join(cells) + f" | {int(row['n_seeds'].iloc[0])} |")
    return "\n".join(lines)


def plot_results(per_run, order, metric="rul_rmse", split="val"):
    part = per_run[per_run["split"] == split]
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, variant in enumerate(order):
        values = part.loc[part["variant"] == variant, metric]
        ax.scatter(np.full(len(values), i), values, color="#546e7a", s=25, zorder=3)
        ax.errorbar(i, values.mean(), yerr=values.std(), fmt="o", color="#c62828", ms=8, capsize=5, zorder=4)
    ax.set_xticks(range(len(order)), order, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(f"{metric} ({split})")
    ax.set_title(f"{metric} on {split} per variant: grey = one seed, red = mean ± std")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    from models.ablation import VARIANTS

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=ABLATION_DIR)
    args = parser.parse_args()

    runs = find_runs(args.dir)
    assert runs, f"no finished runs in {args.dir}: run `python -m models.ablation` first"
    per_run, attn_seeds = collect(runs)
    order = [v for v in VARIANTS if v in set(per_run["variant"])]
    table = aggregate(per_run, order)
    stability = attention_stability(attn_seeds)

    per_run.to_csv(os.path.join(args.dir, "results_runs.csv"), index=False)
    table.to_csv(os.path.join(args.dir, "results.csv"))
    attn_seeds.to_csv(os.path.join(args.dir, "attention_seeds.csv"), index=False)
    with open(os.path.join(args.dir, "results.md"), "w") as f:
        for split in ("val", "test"):
            f.write(f"### {split} split\n\n{to_markdown(table, split)}\n\n")
        f.write("### Attention stability across seeds (validation, Spearman of per-sensor attention)\n\n")
        f.write("\n".join(f"- {v}: {c:.2f}" for v, c in stability.items()) + "\n")
    fig = plot_results(per_run, order)
    fig.savefig(os.path.join(args.dir, "results.png"), dpi=120, bbox_inches="tight")

    for split in ("val", "test"):
        print(f"\n{split} split, mean ± std over seeds")
        print(to_markdown(table, split))
    print(f"\nattention stability across seeds: { {k: round(v, 2) for k, v in stability.items()} }")
    top = attn_seeds.groupby("variant")[list(config.SENSORS)].mean() if len(attn_seeds) else pd.DataFrame()
    for variant, row in top.iterrows():
        print(f"{variant}: top sensors {row.sort_values(ascending=False).index[:4].tolist()}")
    print(f"\nwrote results.csv, results.md, results.png to {args.dir}")
