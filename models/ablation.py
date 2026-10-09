"""Ablation runs: every model variant, several seeds, same data.

    python -m models.ablation                 all variants, seeds 0 1 2
    python -m models.ablation --seeds 0 --variants attention cnn_lstm

Each run writes into artifacts/runs/ablation/<variant>_s<seed>/:
    best.pt, history.csv          as in models.train
    predictions.csv, attn.npy     test split
    val_predictions.csv, val_attn.npy     validation split
Runs that already have predictions.csv are skipped, so an interrupted sweep resumes.
"""

import argparse
import os
import time

import numpy as np

from models.train import RUNS_DIR, load_data, make_predictions, train, write_predictions

ABLATION_DIR = os.path.join(RUNS_DIR, "ablation")

# name -> (model, model kwargs, loss weights). Loss weights not given keep the defaults.
VARIANTS = {
    "cnn": ("cnn", {}, {}),
    "cnn_lstm": ("cnn_lstm", {}, {}),
    "attention": ("attention", {}, {}),
    "attention_no_scale": ("attention", {"scale": False}, {}),
    "attention_no_stage": ("attention", {}, {"stage": 0.0}),
    "attention_no_health": ("attention", {}, {"health": 0.0}),
    "attention_stage_0.5": ("attention", {}, {"stage": 0.5}),
}
SEEDS = (0, 1, 2)


def run_variant(name, seed, data, epochs=30, device="cpu"):
    """Train one variant with one seed. Returns the run folder."""
    model_name, model_kwargs, loss_weights = VARIANTS[name]
    out_dir = os.path.join(ABLATION_DIR, f"{name}_s{seed}")
    if os.path.exists(os.path.join(out_dir, "predictions.csv")):
        return out_dir
    model, _ = train(
        model_name,
        data,
        out_dir,
        model_kwargs=model_kwargs,
        loss_weights=loss_weights,
        epochs=epochs,
        device=device,
        seed=seed,
        verbose=False,
    )
    write_predictions(model, data["test"], out_dir, device)
    val_df, val_attn = make_predictions(model, data["val"], device)
    val_df.to_csv(os.path.join(out_dir, "val_predictions.csv"), index=False)
    if val_attn is not None:
        np.save(os.path.join(out_dir, "val_attn.npy"), val_attn)
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--data", choices=("fake", "real"), default="real")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    data = load_data(args.data)
    total = len(args.variants) * len(args.seeds)
    for i, (name, seed) in enumerate(((n, s) for n in args.variants for s in args.seeds), 1):
        start = time.time()
        out_dir = run_variant(name, seed, data, args.epochs, args.device)
        print(f"[{i}/{total}] {name} seed {seed}: {time.time() - start:.0f}s -> {out_dir}", flush=True)
