import json
import math
import os

import numpy as np
import pytest

import config
from evaluation import plots
from evaluation.metrics import (
    confusion_matrix,
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
    for name in ("metrics.json", "rul_scatter.png", "rul_error.png", "confusion.png"):
        assert (out / name).exists(), f"{name} was not written"
    with open(out / "metrics.json") as f:
        assert json.load(f)["test"]["rul_rmse"] == pytest.approx(metrics["test"]["rul_rmse"])

    table = comparison_table({"fake": metrics})
    assert list(table.index) == ["fake"]
    assert table.loc["fake", "rul_rmse"] == pytest.approx(metrics["test"]["rul_rmse"])
