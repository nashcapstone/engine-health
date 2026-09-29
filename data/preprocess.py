"""FD001 preprocessing: load, split by engine, normalize, window, label.

    python -m data.preprocess      writes train/val/test .npz and the scaler to artifacts/data/

Every split is a dict in the contract 1 format (see contracts.check_batch).
"""

import json
import os

import numpy as np
import pandas as pd

import config
from contracts import check_batch
from data.download import download, raw_path
from labels import cap_rul, health_score, health_stage

VAL_FRACTION = 0.2
OUT_DIR = os.path.join(config.ARTIFACTS_DIR, "data")
SPLITS = ("train", "val", "test")


def load_raw(kind):
    """kind is 'train' or 'test'. Returns a DataFrame with config.RAW_COLUMNS."""
    path = raw_path(f"{kind}_{config.DATASET}.txt")
    return pd.read_csv(path, sep=r"\s+", header=None, names=list(config.RAW_COLUMNS))


def load_test_rul():
    """True RUL at the last test cycle, one value per test engine, in engine order."""
    path = raw_path(f"RUL_{config.DATASET}.txt")
    return pd.read_csv(path, sep=r"\s+", header=None)[0].to_numpy()


def add_train_rul(df):
    """Run-to-failure: RUL is cycles left until the engine's last recorded cycle. Uncapped."""
    last = df.groupby("engine_id")["cycle"].transform("max")
    return df.assign(rul=last - df["cycle"])


def add_test_rul(df, final_rul):
    """Test engines stop before failure: RUL = final RUL + cycles left in the recorded run."""
    engines = np.sort(df["engine_id"].unique())
    assert len(engines) == len(final_rul), f"{len(engines)} test engines but {len(final_rul)} RUL values"
    offset = pd.Series(final_rul, index=engines)
    last = df.groupby("engine_id")["cycle"].transform("max")
    return df.assign(rul=last - df["cycle"] + df["engine_id"].map(offset))


def split_engines(engine_ids, val_fraction=VAL_FRACTION, seed=config.SEED):
    """Split by engine, not by window: overlapping windows from one engine would leak."""
    engines = np.sort(np.unique(engine_ids))
    rng = np.random.default_rng(seed)
    n_val = int(round(len(engines) * val_fraction))
    val = np.sort(rng.choice(engines, n_val, replace=False))
    train = np.setdiff1d(engines, val)
    return train, val


class MinMaxScaler:
    """Per-sensor min-max to [0, 1]. Fit on training engines only."""

    def fit(self, df):
        values = df[list(config.SENSORS)].to_numpy(dtype=np.float64)
        self.min_ = values.min(axis=0)
        self.max_ = values.max(axis=0)
        return self

    def transform(self, df):
        span = np.where(self.max_ > self.min_, self.max_ - self.min_, 1.0)
        out = df.copy()
        out[list(config.SENSORS)] = (df[list(config.SENSORS)].to_numpy() - self.min_) / span
        return out

    def to_dict(self):
        return {"sensors": list(config.SENSORS), "min": self.min_.tolist(), "max": self.max_.tolist()}

    @classmethod
    def from_dict(cls, d):
        assert d["sensors"] == list(config.SENSORS), "scaler was fitted on a different sensor list"
        scaler = cls()
        scaler.min_ = np.asarray(d["min"])
        scaler.max_ = np.asarray(d["max"])
        return scaler


def _batch(X, rul, engine_id, cycle):
    rul = cap_rul(rul)
    return {
        "X": np.asarray(X, dtype=np.float32),
        "y_rul": rul,
        "y_health": health_score(rul),
        "y_stage": health_stage(rul),
        "engine_id": np.asarray(engine_id, dtype=np.int64),
        "cycle": np.asarray(cycle, dtype=np.int64),
    }


def make_windows(df, window=config.WINDOW):
    """Every full sliding window of each engine. Labels come from the window's last row."""
    X, rul, engine_id, cycle = [], [], [], []
    for eid, g in df.groupby("engine_id", sort=True):
        g = g.sort_values("cycle")
        values = g[list(config.SENSORS)].to_numpy(dtype=np.float32)
        if len(values) < window:
            continue
        # (n_windows, n_sensors, window) -> (n_windows, window, n_sensors)
        X.append(np.lib.stride_tricks.sliding_window_view(values, window, axis=0).transpose(0, 2, 1))
        rul.append(g["rul"].to_numpy()[window - 1 :])
        cycle.append(g["cycle"].to_numpy()[window - 1 :])
        engine_id.append(np.full(len(values) - window + 1, eid))
    return _batch(np.concatenate(X), np.concatenate(rul), np.concatenate(engine_id), np.concatenate(cycle))


def make_last_windows(df, window=config.WINDOW):
    """One window per engine: its last `window` cycles. Short engines are padded with their first row."""
    X, rul, engine_id, cycle = [], [], [], []
    for eid, g in df.groupby("engine_id", sort=True):
        g = g.sort_values("cycle")
        values = g[list(config.SENSORS)].to_numpy(dtype=np.float32)[-window:]
        if len(values) < window:
            values = np.concatenate([np.repeat(values[:1], window - len(values), axis=0), values])
        X.append(values)
        rul.append(g["rul"].iloc[-1])
        cycle.append(g["cycle"].iloc[-1])
        engine_id.append(eid)
    return _batch(np.stack(X), np.array(rul), np.array(engine_id), np.array(cycle))


def build_datasets(val_fraction=VAL_FRACTION, seed=config.SEED):
    """Returns ({'train', 'val', 'test'} contract 1 dicts, fitted scaler)."""
    download()
    train_df = add_train_rul(load_raw("train"))
    test_df = add_test_rul(load_raw("test"), load_test_rul())

    train_ids, val_ids = split_engines(train_df["engine_id"], val_fraction, seed)
    fit_df = train_df[train_df["engine_id"].isin(train_ids)]
    val_df = train_df[train_df["engine_id"].isin(val_ids)]

    scaler = MinMaxScaler().fit(fit_df)
    splits = {
        "train": make_windows(scaler.transform(fit_df)),
        "val": make_windows(scaler.transform(val_df)),
        "test": make_last_windows(scaler.transform(test_df)),
    }
    for batch in splits.values():
        check_batch(batch)
    check_no_leakage(splits)
    return splits, scaler


def check_no_leakage(splits):
    train = set(np.unique(splits["train"]["engine_id"]).tolist())
    val = set(np.unique(splits["val"]["engine_id"]).tolist())
    overlap = train & val
    assert not overlap, f"engines in both train and val: {sorted(overlap)}"


def save_datasets(splits, scaler, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    for name, batch in splits.items():
        np.savez(os.path.join(out_dir, f"{name}.npz"), **batch)
    with open(os.path.join(out_dir, "scaler.json"), "w") as f:
        json.dump(scaler.to_dict(), f, indent=2)


def load_split(name, out_dir=OUT_DIR):
    """Load a saved split as a contract 1 dict. Run `python -m data.preprocess` first."""
    with np.load(os.path.join(out_dir, f"{name}.npz")) as f:
        return {k: f[k] for k in f.files}


if __name__ == "__main__":
    splits, scaler = build_datasets()
    save_datasets(splits, scaler)
    for name, batch in splits.items():
        n_engines = len(np.unique(batch["engine_id"]))
        stages = np.bincount(batch["y_stage"], minlength=config.N_STAGES).tolist()
        print(f"{name:5s} X {batch['X'].shape}  engines {n_engines}  stages {stages}")
    print(f"wrote {', '.join(SPLITS)} .npz and scaler.json to {OUT_DIR}")
