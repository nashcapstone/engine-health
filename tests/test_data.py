import os

import numpy as np
import pandas as pd
import pytest

import config
from contracts import check_batch
from data.download import FILES, raw_path
from data.preprocess import (
    MinMaxScaler,
    add_test_rul,
    add_train_rul,
    check_no_leakage,
    make_last_windows,
    make_windows,
    split_engines,
)


def _raw(lives, seed=0):
    """Synthetic raw frame in the FD001 layout: one run of `life` cycles per engine."""
    rng = np.random.default_rng(seed)
    rows = []
    for eid, life in enumerate(lives, start=1):
        for c in range(1, life + 1):
            rows.append([eid, c, 0.0, 0.0, 100.0] + rng.normal(size=21).tolist())
    return pd.DataFrame(rows, columns=list(config.RAW_COLUMNS))


def test_train_rul_counts_down_to_zero():
    df = add_train_rul(_raw([40, 50]))
    for _, g in df.groupby("engine_id"):
        assert g["rul"].tolist() == list(range(len(g) - 1, -1, -1))


def test_test_rul_adds_final_offset():
    df = add_test_rul(_raw([35, 40]), np.array([10, 200]))
    last = df.groupby("engine_id").tail(1)["rul"].tolist()
    assert last == [10, 200]
    assert df[df["engine_id"] == 1]["rul"].iloc[0] == 10 + 34


def test_windows_meet_contract_and_count():
    df = add_train_rul(_raw([40, 50, 29]))
    batch = make_windows(df)
    check_batch(batch)
    # An engine with fewer cycles than the window yields no windows
    assert len(batch["X"]) == (40 - 29) + (50 - 29)
    assert 3 not in batch["engine_id"]


def test_window_label_comes_from_last_row():
    df = add_train_rul(_raw([45]))
    batch = make_windows(df)
    assert batch["cycle"][0] == config.WINDOW
    assert batch["y_rul"][0] == 45 - config.WINDOW
    assert batch["y_rul"][-1] == 0
    first = df[config.SENSORS[0]].to_numpy()[: config.WINDOW]
    np.testing.assert_allclose(batch["X"][0, :, 0], first, rtol=1e-6)


def test_rul_is_capped():
    df = add_train_rul(_raw([300]))
    batch = make_windows(df)
    assert batch["y_rul"].max() == config.RUL_CAP
    assert batch["y_health"].max() == 100


def test_last_windows_one_per_engine_with_padding():
    df = add_test_rul(_raw([60, 20]), np.array([5, 50]))
    batch = make_last_windows(df)
    check_batch(batch)
    assert batch["engine_id"].tolist() == [1, 2]
    assert batch["y_rul"].tolist() == [5, 50]
    assert batch["cycle"].tolist() == [60, 20]
    # Short engine is front-padded with its first row
    np.testing.assert_array_equal(batch["X"][1, 0], batch["X"][1, config.WINDOW - 20])


def test_split_is_by_engine_and_deterministic():
    engines = np.arange(1, 101)
    train, val = split_engines(engines)
    assert len(val) == 20 and len(train) == 80
    assert not set(train) & set(val)
    assert set(train) | set(val) == set(engines)
    train2, val2 = split_engines(engines)
    assert val.tolist() == val2.tolist()


def test_scaler_fits_train_only():
    train = _raw([40], seed=1)
    other = _raw([40], seed=2)
    other[list(config.SENSORS)] += 5.0
    scaler = MinMaxScaler().fit(train)
    scaled = scaler.transform(train)[list(config.SENSORS)].to_numpy()
    assert scaled.min() == pytest.approx(0) and scaled.max() == pytest.approx(1)
    # Unseen data is not refitted, so it can fall outside [0, 1]
    assert scaler.transform(other)[list(config.SENSORS)].to_numpy().max() > 1


def test_scaler_round_trips():
    scaler = MinMaxScaler().fit(_raw([40]))
    restored = MinMaxScaler.from_dict(scaler.to_dict())
    np.testing.assert_array_equal(restored.min_, scaler.min_)
    np.testing.assert_array_equal(restored.max_, scaler.max_)


def test_leakage_check_catches_shared_engine():
    splits = {"train": {"engine_id": np.array([1, 2])}, "val": {"engine_id": np.array([2, 3])}}
    with pytest.raises(AssertionError, match="both train and val"):
        check_no_leakage(splits)


have_raw = all(os.path.exists(raw_path(f)) for f in FILES)


@pytest.mark.skipif(not have_raw, reason="FD001 not in data/raw; run python -m data.download")
def test_real_fd001():
    from data.preprocess import build_datasets

    splits, scaler = build_datasets()
    n_windows = len(splits["train"]["X"]) + len(splits["val"]["X"])
    # 20,631 rows over 100 engines, minus 29 per engine for the window
    assert n_windows == 20631 - 100 * (config.WINDOW - 1)
    assert len(splits["test"]["X"]) == 100
    assert len(np.unique(splits["val"]["engine_id"])) == 20
    assert splits["train"]["X"].min() == 0 and splits["train"]["X"].max() == 1
