import numpy as np
import pytest

import config
from contracts import check_attention, check_batch, check_model_output, check_predictions
from fakes import fake_dataset, fake_predictions
from labels import health_score, health_stage


def test_config_has_14_sensors():
    assert config.N_SENSORS == 14
    assert not set(config.SENSORS) & set(config.DROPPED_SENSORS)


def test_health_score_endpoints():
    assert health_score(0) == 0
    assert health_score(config.RUL_CAP) == 100
    assert health_score(300) == 100


def test_health_stage_boundaries():
    stages = health_stage(np.array([125, 101, 100, 30, 29, 0]))
    assert stages.tolist() == [0, 0, 1, 1, 2, 2]


def test_fake_dataset_meets_contract():
    batch = fake_dataset()
    check_batch(batch)
    assert set(np.unique(batch["y_stage"])) == {0, 1, 2}


def test_fake_predictions_meet_contract():
    df, attn = fake_predictions()
    check_predictions(df, attn)


def test_check_batch_rejects_wrong_window():
    batch = fake_dataset()
    batch["X"] = batch["X"][:, :-1, :]
    with pytest.raises(AssertionError, match="X has shape"):
        check_batch(batch)


def test_check_attention_rejects_unnormalized():
    _, attn = fake_predictions()
    with pytest.raises(AssertionError, match="sum to 1"):
        check_attention(attn * 2, len(attn))


def test_model_output_allows_missing_attention():
    n = 8
    out = {
        "rul": np.zeros(n),
        "health": np.zeros(n),
        "stage_logits": np.zeros((n, config.N_STAGES)),
        "attn": None,
    }
    check_model_output(out, n)
