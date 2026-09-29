"""Training loop, checkpointing and the predictions file.

    python -m models.train --model cnn --data fake       smoke run on fakes.fake_dataset()
    python -m models.train --model cnn --data real       needs `python -m data.preprocess` first

Each run writes into artifacts/runs/<model>_<data>/:
    best.pt            checkpoint with the lowest validation RUL RMSE
    history.csv        one row per epoch
    predictions.csv    contract 3, on the test split (validation split for fake data)
    attn.npy           only for models with attention
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config
from contracts import check_batch, check_predictions
from data.loaders import EngineWindows
from models import MODELS, build_model
from models.heads import HEALTH_MAX
from models.losses import MultiTaskLoss, stage_class_weights

RUNS_DIR = os.path.join(config.ARTIFACTS_DIR, "runs")


def split_by_engine(batch, val_fraction=0.2, seed=config.SEED):
    """Split a contract 1 dict into (train, val) by engine, so overlapping windows do not leak."""
    engines = np.unique(batch["engine_id"])
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(len(engines) * val_fraction)))
    is_val = np.isin(batch["engine_id"], rng.choice(engines, n_val, replace=False))
    return {k: v[~is_val] for k, v in batch.items()}, {k: v[is_val] for k, v in batch.items()}


def load_data(kind):
    """kind is 'fake' or 'real'. Returns {'train', 'val', 'test'} contract 1 dicts."""
    if kind == "fake":
        from fakes import fake_dataset

        train, val = split_by_engine(fake_dataset())
        return {"train": train, "val": val, "test": val}
    from data.preprocess import SPLITS, load_split

    return {name: load_split(name) for name in SPLITS}


def _to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def predict(model, batch, batch_size=1024, device="cpu"):
    """Run a contract 1 dict through the model. Returns the contract 2 dict as numpy arrays."""
    model.to(device).eval()
    loader = DataLoader(EngineWindows(batch), batch_size=batch_size)
    outs = [model(b["X"].to(device)) for b in loader]
    merged = {
        k: torch.cat([o[k] for o in outs]).cpu().numpy() for k in ("rul", "health", "stage_logits")
    }
    has_attn = outs[0]["attn"] is not None
    merged["attn"] = torch.cat([o["attn"] for o in outs]).cpu().numpy() if has_attn else None
    return merged


def make_predictions(model, batch, device="cpu"):
    """Returns (contract 3 DataFrame, attention array or None). Rows follow the order of batch."""
    out = predict(model, batch, device=device)
    df = pd.DataFrame(
        {
            "engine_id": batch["engine_id"],
            "cycle": batch["cycle"],
            "rul_true": batch["y_rul"],
            "rul_pred": np.clip(out["rul"], 0, config.RUL_CAP),
            "health_true": batch["y_health"],
            "health_pred": np.clip(out["health"], 0, HEALTH_MAX),
            "stage_true": batch["y_stage"],
            "stage_pred": out["stage_logits"].argmax(axis=1),
        }
    )[list(config.PREDICTION_COLUMNS)]
    check_predictions(df, out["attn"])
    return df, out["attn"]


def write_predictions(model, batch, out_dir, device="cpu"):
    df, attn = make_predictions(model, batch, device)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    if attn is not None:
        np.save(os.path.join(out_dir, "attn.npy"), attn)
    return df, attn


@torch.no_grad()
def evaluate(model, loader, loss_fn, device="cpu"):
    """Mean losses over the loader, plus RUL RMSE in cycles and stage accuracy."""
    model.eval()
    sums = {"loss": 0.0, "rul": 0.0, "health": 0.0, "stage": 0.0, "sq_err": 0.0, "correct": 0.0}
    n = 0
    for batch in loader:
        batch = _to_device(batch, device)
        out = model(batch["X"])
        total, parts = loss_fn(out, batch)
        size = len(batch["X"])
        n += size
        sums["loss"] += total.item() * size
        for k, v in parts.items():
            sums[k] += v * size
        rul = out["rul"].clamp(0, config.RUL_CAP)
        sums["sq_err"] += ((rul - batch["y_rul"]) ** 2).sum().item()
        sums["correct"] += (out["stage_logits"].argmax(dim=1) == batch["y_stage"]).sum().item()
    return {
        "loss": sums["loss"] / n,
        "rul_loss": sums["rul"] / n,
        "health_loss": sums["health"] / n,
        "stage_loss": sums["stage"] / n,
        "rul_rmse": (sums["sq_err"] / n) ** 0.5,
        "stage_acc": sums["correct"] / n,
    }


def save_checkpoint(path, model, model_name, model_kwargs, **extra):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "model_kwargs": model_kwargs,
            "state_dict": model.state_dict(),
            **extra,
        },
        path,
    )


def load_checkpoint(path, device="cpu"):
    """Rebuild the model from a checkpoint. Returns (model in eval mode, checkpoint dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build_model(ckpt["model_name"], **ckpt["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


def train(
    model_name,
    data,
    out_dir,
    model_kwargs=None,
    epochs=30,
    batch_size=256,
    lr=1e-3,
    loss_weights=None,
    use_class_weights=True,
    device="cpu",
    seed=config.SEED,
    verbose=True,
):
    """Train on data['train'], keep the checkpoint with the best RUL RMSE on data['val'].

    Returns (best model, history DataFrame).
    """
    torch.manual_seed(seed)
    for batch in data.values():
        check_batch(batch)

    model_kwargs = model_kwargs or {}
    model = build_model(model_name, **model_kwargs).to(device)
    class_weights = stage_class_weights(data["train"]["y_stage"]) if use_class_weights else None
    loss_fn = MultiTaskLoss(loss_weights, class_weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        EngineWindows(data["train"]), batch_size=batch_size, shuffle=True, generator=generator
    )
    val_loader = DataLoader(EngineWindows(data["val"]), batch_size=1024)

    ckpt_path = os.path.join(out_dir, "best.pt")
    history, best_rmse = [], float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, n = 0.0, 0
        for batch in train_loader:
            batch = _to_device(batch, device)
            total, _ = loss_fn(model(batch["X"]), batch)
            assert torch.isfinite(total), f"loss is {total.item()} at epoch {epoch}"
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            train_loss += total.item() * len(batch["X"])
            n += len(batch["X"])

        val = evaluate(model, val_loader, loss_fn, device)
        row = {"epoch": epoch, "train_loss": train_loss / n, **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        is_best = val["rul_rmse"] < best_rmse
        if is_best:
            best_rmse = val["rul_rmse"]
            save_checkpoint(
                ckpt_path,
                model,
                model_name,
                model_kwargs,
                epoch=epoch,
                val_rul_rmse=best_rmse,
                loss_weights=loss_fn.weights,
            )
        if verbose:
            print(
                f"epoch {epoch:3d}  train {row['train_loss']:.4f}  val {val['loss']:.4f}  "
                f"rmse {val['rul_rmse']:6.2f}  stage acc {val['stage_acc']:.3f}"
                + ("  *" if is_best else "")
            )

    history = pd.DataFrame(history)
    history.to_csv(os.path.join(out_dir, "history.csv"), index=False)
    model, _ = load_checkpoint(ckpt_path, device)
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODELS), default="cnn")
    parser.add_argument("--data", choices=("fake", "real"), default="fake")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out_dir = os.path.join(RUNS_DIR, f"{args.model}_{args.data}")
    data = load_data(args.data)
    model, history = train(
        args.model,
        data,
        out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
    df, attn = write_predictions(model, data["test"], out_dir, args.device)
    rmse = float(np.sqrt(np.mean((df["rul_pred"] - df["rul_true"]) ** 2)))
    acc = float((df["stage_pred"] == df["stage_true"]).mean())
    print(f"best val rmse {history['val_rul_rmse'].min():.2f}")
    print(f"test rmse {rmse:.2f}  stage acc {acc:.3f}  ({len(df)} rows)")
    print(f"wrote best.pt, history.csv and predictions.csv to {out_dir}")
