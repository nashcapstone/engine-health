"""Stand-ins for real inputs, in the contract formats.

Models builds against fake_dataset() until the real dataloaders land.
Evaluation builds against fake_predictions() until a real model lands.

    python fakes.py      writes both into artifacts/fake/
"""

import os

import numpy as np
import pandas as pd

import config
from labels import cap_rul, health_score, health_stage


def _softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _engine_cycles(n_engines, rng):
    """Every windowed cycle of n_engines run-to-failure engines."""
    engine_id, cycle, rul = [], [], []
    for eid in range(1, n_engines + 1):
        life = int(rng.integers(130, 360))
        cycles = np.arange(config.WINDOW, life + 1)
        engine_id.append(np.full(len(cycles), eid))
        cycle.append(cycles)
        rul.append(life - cycles)
    return np.concatenate(engine_id), np.concatenate(cycle), np.concatenate(rul)


def fake_dataset(n_engines=20, noise=0.05, seed=config.SEED):
    """Contract 1. Sensors drift with wear, each at its own strength, so the loss can fall."""
    rng = np.random.default_rng(seed)
    engine_id, cycle, raw_rul = _engine_cycles(n_engines, rng)
    rul = cap_rul(raw_rul)

    n = len(rul)
    wear = 1.0 - rul / config.RUL_CAP
    ramp = np.linspace(0.8, 1.0, config.WINDOW)
    strength = rng.uniform(0.0, 1.0, config.N_SENSORS)
    # A few sensors carry no signal, so attention has something to ignore
    strength[rng.choice(config.N_SENSORS, 3, replace=False)] = 0.0

    X = wear[:, None, None] * ramp[None, :, None] * strength[None, None, :]
    X = X + noise * rng.standard_normal((n, config.WINDOW, config.N_SENSORS))

    return {
        "X": X.astype(np.float32),
        "y_rul": rul,
        "y_health": health_score(rul),
        "y_stage": health_stage(rul),
        "engine_id": engine_id.astype(np.int64),
        "cycle": cycle.astype(np.int64),
    }


def fake_predictions(n_engines=10, rul_noise=12.0, seed=config.SEED):
    """Contract 3. Returns (predictions DataFrame, attention array)."""
    rng = np.random.default_rng(seed)
    engine_id, cycle, raw_rul = _engine_cycles(n_engines, rng)
    rul_true = cap_rul(raw_rul)

    n = len(rul_true)
    rul_pred = np.clip(rul_true + rul_noise * rng.standard_normal(n), 0, config.RUL_CAP)
    health_pred = np.clip(health_score(rul_true) + 6.0 * rng.standard_normal(n), 0, 100)

    df = pd.DataFrame(
        {
            "engine_id": engine_id,
            "cycle": cycle,
            "rul_true": rul_true,
            "rul_pred": rul_pred.astype(np.float32),
            "health_true": health_score(rul_true),
            "health_pred": health_pred.astype(np.float32),
            "stage_true": health_stage(rul_true),
            "stage_pred": health_stage(rul_pred),
        }
    )[list(config.PREDICTION_COLUMNS)]

    # Attention favours a different group of sensors in each stage
    focus = np.zeros((config.N_STAGES, config.N_SENSORS))
    for stage in range(config.N_STAGES):
        focus[stage, rng.choice(config.N_SENSORS, 3, replace=False)] = 2.0
    logits = focus[df["stage_true"].to_numpy()][:, None, :]
    logits = logits + 0.3 * rng.standard_normal((n, config.WINDOW, config.N_SENSORS))
    attn = _softmax(logits).astype(np.float32)

    return df, attn


if __name__ == "__main__":
    out_dir = os.path.join(config.ARTIFACTS_DIR, "fake")
    os.makedirs(out_dir, exist_ok=True)

    np.savez(os.path.join(out_dir, "dataset.npz"), **fake_dataset())
    df, attn = fake_predictions()
    df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    np.save(os.path.join(out_dir, "attn.npy"), attn)
    print(f"wrote dataset.npz, predictions.csv ({len(df)} rows) and attn.npy to {out_dir}")
