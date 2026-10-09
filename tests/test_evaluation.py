import json
import math
import os

import numpy as np
import pytest

import config
from evaluation import attention, plots
from evaluation.metrics import (
    confusion_matrix,
    errors_by_stage,
    head_agreement,
    mae,
    nasa_score,
    rmse,
    stage_accuracy,
    stage_recall,
    summarize,
)
from evaluation.report import comparison_table, evaluate_run, pick_engines
from fakes import fake_predictions


def test_rmse_and_mae_by_hand():
    y_true = [10, 20, 30, 40]
    y_pred = [12, 18, 30, 44]  # errors 2, -2, 0, 4
    assert rmse(y_true, y_pred) == pytest.approx(math.sqrt((4 + 4 + 0 + 16) / 4))
    assert mae(y_true, y_pred) == pytest.approx((2 + 2 + 0 + 4) / 4)


def test_perfect_predictions_score_zero():
    y = np.array([0.0, 50.0, 125.0])
    assert rmse(y, y) == 0
    assert mae(y, y) == 0
    assert nasa_score(y, y) == 0


def test_nasa_score_by_hand():
    # d = -13 -> e^1 - 1, d = +10 -> e^1 - 1, d = 0 -> 0
    score = nasa_score([50, 50, 50], [37, 60, 50])
    assert score == pytest.approx(2 * (math.e - 1))


def test_nasa_score_penalises_late_more_than_early():
    early = nasa_score([50], [40])
    late = nasa_score([50], [60])
    assert late > early > 0


def test_metrics_reject_mismatched_shapes():
    with pytest.raises(AssertionError, match="shapes differ"):
        rmse([1, 2, 3], [1, 2])


def test_confusion_matrix_and_recall():
    stage_true = [0, 0, 1, 1, 2, 2]
    stage_pred = [0, 1, 1, 1, 2, 0]
    cm = confusion_matrix(stage_true, stage_pred)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert cm.sum() == 6
    assert stage_recall(stage_true, stage_pred).tolist() == [0.5, 1.0, 0.5]
    assert stage_accuracy(stage_true, stage_pred) == pytest.approx(4 / 6)


def test_stage_recall_is_nan_for_missing_stage():
    recall = stage_recall([0, 0], [0, 1])
    assert recall[0] == 0.5
    assert np.isnan(recall[1]) and np.isnan(recall[2])


def test_head_agreement_on_consistent_heads():
    df, _ = fake_predictions()
    # fake stage_pred is computed from rul_pred, so the heads agree fully
    assert head_agreement(df)["stage_agreement"] == 1.0


def test_summarize_on_fake_predictions():
    df, _ = fake_predictions()
    m = summarize(df)
    assert m["n"] == len(df)
    assert m["rul_rmse"] >= m["rul_mae"] > 0
    assert 0 <= m["stage_acc"] <= 1
    assert all(f"recall_{name.lower()}" in m for name in config.STAGE_NAMES)


def test_plots_draw_on_fake_predictions(tmp_path):
    df, _ = fake_predictions()
    eid = int(df["engine_id"].iloc[0])
    figs = [
        plots.plot_rul_scatter(df),
        plots.plot_rul_error_hist(df),
        plots.plot_error_by_rul(df),
        plots.plot_confusion(df),
        plots.plot_health_over_life(df, eid),
        plots.plot_rul_over_life(df, eid),
        plots.plot_health_grid(df, pick_engines(df, k=4), ncols=2),
    ]
    for i, fig in enumerate(figs):
        path = tmp_path / f"{i}.png"
        fig.savefig(path)
        assert path.stat().st_size > 0


def test_life_plot_refuses_one_row_per_engine():
    df, _ = fake_predictions()
    last = df.groupby("engine_id").tail(1)
    with pytest.raises(AssertionError, match="use the val split"):
        plots.plot_health_over_life(last, int(last["engine_id"].iloc[0]))


def test_evaluate_run_on_a_fake_run_folder(tmp_path):
    df, attn = fake_predictions()
    df.to_csv(tmp_path / "predictions.csv", index=False)
    np.save(tmp_path / "attn.npy", attn)

    metrics = evaluate_run(str(tmp_path), with_val=False)
    out = tmp_path / "eval"
    expected = ("metrics.json", "rul_scatter.png", "rul_error.png", "confusion.png", "error_by_rul.png",
                "attn_by_stage.png", "attn_over_time.png")
    for name in expected:
        assert (out / name).exists(), f"{name} was not written"
    assert set(metrics["test"]["top_sensors"]) == set(config.STAGE_NAMES)
    with open(out / "metrics.json") as f:
        assert json.load(f)["test"]["rul_rmse"] == pytest.approx(metrics["test"]["rul_rmse"])

    table = comparison_table({"fake": metrics})
    assert list(table.index) == ["fake"]
    assert table.loc["fake", "rul_rmse"] == pytest.approx(metrics["test"]["rul_rmse"])
    assert len(comparison_table({"fake": metrics}, split="val")) == 0


def test_errors_by_stage_by_hand():
    df, _ = fake_predictions()
    df = df.iloc[:4].copy()
    df["stage_true"] = [0, 0, 2, 2]
    df["rul_true"] = [110.0, 120.0, 10.0, 20.0]
    df["rul_pred"] = [100.0, 120.0, 16.0, 22.0]  # errors -10, 0, +6, +2
    table = errors_by_stage(df)
    assert table.loc["HEALTHY", "n"] == 2
    assert table.loc["HEALTHY", "bias"] == pytest.approx(-5.0)
    assert table.loc["CRITICAL", "mae"] == pytest.approx(4.0)
    assert table.loc["CRITICAL", "rmse"] == pytest.approx(math.sqrt((36 + 4) / 2))
    assert table.loc["WARNING", "n"] == 0 and np.isnan(table.loc["WARNING", "rmse"])


def test_attention_by_stage_by_hand():
    df, _ = fake_predictions()
    df = df.iloc[:2].copy()
    df["stage_true"] = [0, 2]
    attn = np.full((2, config.WINDOW, config.N_SENSORS), 0.0, dtype=np.float32)
    attn[0, :, 0] = 1.0  # HEALTHY row looks only at the first sensor
    attn[1, :, -1] = 1.0  # CRITICAL row looks only at the last sensor
    by_stage = attention.attention_by_stage(df, attn)
    first, last = config.SENSORS[0], config.SENSORS[-1]
    assert by_stage.loc["HEALTHY", first] == 1.0
    assert by_stage.loc["CRITICAL", last] == 1.0
    assert by_stage.loc["WARNING"].isna().all()
    assert attention.top_sensors(by_stage, k=1)["CRITICAL"] == [last]
    shift = attention.stage_shift(by_stage)
    assert shift.index[0] == last and shift.iloc[0] == 1.0


def test_attention_by_stage_rows_sum_to_one():
    df, attn = fake_predictions()
    by_stage = attention.attention_by_stage(df, attn)
    assert np.allclose(by_stage.sum(axis=1), 1.0, atol=1e-4)
    assert attention.attention_over_time(attn).shape == (config.WINDOW, config.N_SENSORS)


def test_attention_finds_the_fake_focus_sensors():
    # fake attention boosts 3 sensors per stage; they should be each stage's top 3
    df, attn = fake_predictions()
    by_stage = attention.attention_by_stage(df, attn)
    for stage, sensors in attention.top_sensors(by_stage, k=3).items():
        assert all(by_stage.loc[stage, s] > 2 * attention.UNIFORM for s in sensors)


def test_attention_plots_draw(tmp_path):
    df, attn = fake_predictions()
    eid = int(df["engine_id"].iloc[0])
    figs = [
        attention.plot_window_attention(attn[0]),
        attention.plot_attention_by_stage(df, attn),
        attention.plot_attention_over_time(attn),
        attention.plot_attention_over_life(df, attn, eid),
    ]
    for i, fig in enumerate(figs):
        path = tmp_path / f"{i}.png"
        fig.savefig(path)
        assert path.stat().st_size > 0


def test_attention_rejects_mismatched_rows():
    df, attn = fake_predictions()
    with pytest.raises(AssertionError, match="attn has shape"):
        attention.attention_by_stage(df, attn[:-1])


def _tiny_attention_model():
    import torch

    from models import build_model

    torch.manual_seed(0)
    return build_model("attention").eval()


def test_corrupt_modes_break_only_one_sensor():
    from evaluation.sensor_failure import corrupt
    from fakes import fake_dataset

    batch = fake_dataset(n_engines=2)
    j = 3
    sensor = config.SENSORS[j]
    others = [k for k in range(config.N_SENSORS) if k != j]
    for mode in ("stuck", "noise", "dead"):
        broken = corrupt(batch, sensor, mode)
        assert np.array_equal(broken["X"][:, :, others], batch["X"][:, :, others]), f"{mode} touched other sensors"
        assert not np.array_equal(broken["X"][:, :, j], batch["X"][:, :, j])
        assert broken["X"].dtype == np.float32
    stuck = corrupt(batch, sensor, "stuck")["X"][:, :, j]
    assert np.allclose(stuck, stuck[:, :1])
    assert (corrupt(batch, sensor, "dead")["X"][:, :, j] == 0).all()
    assert batch["X"][:, :, j].std() > 0, "corrupt must not change the input batch"
    with pytest.raises(ValueError, match="unknown mode"):
        corrupt(batch, sensor, "melted")


def test_failure_table_and_plot():
    from evaluation.sensor_failure import failure_table, plot_failure, rank_agreement
    from fakes import fake_dataset

    batch = fake_dataset(n_engines=2)
    table = failure_table(_tiny_attention_model(), batch, modes=("noise", "dead"))
    assert len(table) == 2 * config.N_SENSORS
    assert (table["mean_abs_change"] >= 0).all()
    assert table["attention"].between(0, 1).all()
    assert set(rank_agreement(table)) == {"noise", "dead"}
    plot_failure(table)


def test_onset_walks_up_from_failure():
    import pandas as pd

    from evaluation.early_warning import onset

    curve = pd.DataFrame({"a": [1, 1, 5, 5, 5], "b": [5, 1, 5, 5, 5], "c": [1, 1, 1, 1, 1]}, index=[40, 30, 20, 10, 0])
    result = onset(curve, lambda c: c > 2)
    assert result["a"] == 20  # above from RUL 20 down to 0
    assert result["b"] == 20  # the blip at 40 does not count, 30 breaks the run
    assert np.isnan(result["c"])


def test_early_warning_on_fake_data():
    from evaluation.early_warning import (
        attention_by_rul,
        early_warning_table,
        plot_early_warning,
        signal_drift_by_rul,
    )
    from fakes import fake_dataset
    from models.train import make_predictions

    batch = fake_dataset(n_engines=4)
    df, attn = make_predictions(_tiny_attention_model(), batch)
    att_curve = attention_by_rul(df, attn)
    assert np.allclose(att_curve.sum(axis=1), 1.0, atol=1e-4)
    drift = signal_drift_by_rul(batch)
    assert list(drift.columns) == list(config.SENSORS)
    # fake sensors drift with wear, so most should show a signal onset
    table = early_warning_table(att_curve, drift)
    assert table["signal_onset_rul"].notna().sum() >= config.N_SENSORS // 2
    plot_early_warning(att_curve, drift, table)


def test_format_card_matches_the_spec():
    from evaluation.card import format_card

    result = {
        "engine_id": 42, "cycle": 150, "health": 68.2, "stage_name": "WARNING", "rul": 38.4,
        "top_sensors": [("s14", 0.18), ("s11", 0.11)],
    }
    card = format_card(result, true_rul=40)
    for text in ("Engine 42 - Cycle 150", "Health Score   68 / 100", "Stage          WARNING",
                 "Estimated RUL  38 cycles", "Actual RUL     40 cycles", "s14 (2.5x)"):
        assert text in card, f"{text!r} missing from the card"
    widths = {len(line) for line in card.splitlines()}
    assert len(widths) == 1, "card lines should all be the same width"
    assert "Watching" not in format_card({**result, "top_sensors": []})


def test_fleet_table_sorts_most_urgent_first():
    import pandas as pd

    from data.preprocess import MinMaxScaler
    from evaluation.card import fleet_table
    from models.inference import Predictor

    rng = np.random.default_rng(0)
    parts = []
    for eid, n in ((1, 40), (2, 35), (3, 50)):
        part = pd.DataFrame(rng.uniform(0, 1, (n, len(config.RAW_COLUMNS))), columns=list(config.RAW_COLUMNS))
        part["engine_id"], part["cycle"] = eid, np.arange(1, n + 1)
        parts.append(part)
    scaler = MinMaxScaler()
    scaler.min_, scaler.max_ = np.zeros(config.N_SENSORS), np.ones(config.N_SENSORS)
    table = fleet_table(Predictor(_tiny_attention_model(), scaler), pd.concat(parts), pd.Series({1: 5, 2: 200, 3: 50}))
    assert sorted(table["engine_id"]) == [1, 2, 3]
    assert table["rul_pred"].is_monotonic_increasing
    assert table.set_index("engine_id").loc[2, "rul_true"] == config.RUL_CAP
    assert table.set_index("engine_id").loc[3, "cycle"] == 50
