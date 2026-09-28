"""Validators for the three interfaces between data, models and evaluation.

Call these at every handoff. They raise AssertionError with a message naming the field.
"""

import numpy as np

import config

BATCH_KEYS = ("X", "y_rul", "y_health", "y_stage", "engine_id", "cycle")
MODEL_OUTPUT_KEYS = ("rul", "health", "stage_logits", "attn")


def _shape(a):
    return tuple(a.shape)


def check_batch(batch):
    """Contract 1, data -> model. Accepts numpy arrays or torch tensors."""
    missing = [k for k in BATCH_KEYS if k not in batch]
    assert not missing, f"batch is missing keys: {missing}"

    n = _shape(batch["X"])[0]
    expected = (n, config.WINDOW, config.N_SENSORS)
    assert _shape(batch["X"]) == expected, f"X has shape {_shape(batch['X'])}, expected {expected}"
    assert "float32" in str(batch["X"].dtype), f"X has dtype {batch['X'].dtype}, expected float32"

    for key in BATCH_KEYS[1:]:
        assert _shape(batch[key]) == (n,), f"{key} has shape {_shape(batch[key])}, expected {(n,)}"

    rul = np.asarray(batch["y_rul"])
    assert rul.min() >= 0 and rul.max() <= config.RUL_CAP, (
        f"y_rul must lie in [0, {config.RUL_CAP}], got [{rul.min()}, {rul.max()}]"
    )
    health = np.asarray(batch["y_health"])
    assert health.min() >= 0 and health.max() <= 100, (
        f"y_health must lie in [0, 100], got [{health.min()}, {health.max()}]"
    )
    stages = set(np.unique(np.asarray(batch["y_stage"])).tolist())
    assert stages <= set(range(config.N_STAGES)), f"y_stage has unexpected values: {stages}"


def check_model_output(out, n):
    """Contract 2, model -> evaluation. Accepts numpy arrays or torch tensors."""
    missing = [k for k in MODEL_OUTPUT_KEYS if k not in out]
    assert not missing, f"model output is missing keys: {missing}"

    assert _shape(out["rul"]) == (n,), f"rul has shape {_shape(out['rul'])}, expected {(n,)}"
    assert _shape(out["health"]) == (n,), f"health has shape {_shape(out['health'])}, expected {(n,)}"
    expected = (n, config.N_STAGES)
    assert _shape(out["stage_logits"]) == expected, (
        f"stage_logits has shape {_shape(out['stage_logits'])}, expected {expected}"
    )
    if out["attn"] is not None:
        check_attention(out["attn"], n)


def check_attention(attn, n):
    expected = (n, config.WINDOW, config.N_SENSORS)
    assert _shape(attn) == expected, f"attn has shape {_shape(attn)}, expected {expected}"
    if hasattr(attn, "detach"):
        attn = attn.detach().cpu().numpy()
    sums = np.asarray(attn).sum(axis=-1)
    assert np.allclose(sums, 1.0, atol=1e-4), "attn must sum to 1 across sensors at each timestep"


def check_predictions(df, attn=None):
    """Contract 3, the predictions file. attn rows follow the same order as df."""
    assert tuple(df.columns) == config.PREDICTION_COLUMNS, (
        f"columns are {tuple(df.columns)}, expected {config.PREDICTION_COLUMNS}"
    )
    assert not df.isna().any().any(), "predictions contain missing values"
    for col in ("stage_true", "stage_pred"):
        stages = set(df[col].unique().tolist())
        assert stages <= set(range(config.N_STAGES)), f"{col} has unexpected values: {stages}"
    if attn is not None:
        check_attention(attn, len(df))
