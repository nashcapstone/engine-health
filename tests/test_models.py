import numpy as np
import pytest
import torch

import config
from contracts import check_model_output, check_predictions
from fakes import fake_dataset
from models import MODELS, build_model
from models.losses import MultiTaskLoss, stage_class_weights
from models.train import load_checkpoint, make_predictions, split_by_engine, train

MODEL_NAMES = sorted(MODELS)


def small_batch(n=32, seed=0):
    """n random windows of the fake dataset, as tensors."""
    data = fake_dataset()
    idx = np.random.default_rng(seed).choice(len(data["X"]), n, replace=False)
    return {k: torch.from_numpy(v[idx]) for k, v in data.items()}


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_forward_meets_contract(name):
    batch = small_batch()
    out = build_model(name)(batch["X"])
    check_model_output(out, len(batch["X"]))


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_loss_and_backward_are_finite(name):
    torch.manual_seed(0)
    batch = small_batch()
    model = build_model(name)
    total, parts = MultiTaskLoss()(model(batch["X"]), batch)
    total.backward()
    assert torch.isfinite(total)
    assert all(np.isfinite(v) for v in parts.values())
    for p_name, p in model.named_parameters():
        assert p.grad is not None, f"{p_name} got no gradient"
        assert torch.isfinite(p.grad).all(), f"{p_name} has a non-finite gradient"


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_overfits_one_small_batch(name):
    torch.manual_seed(0)
    batch = small_batch()
    model = build_model(name, dropout=0.0)
    loss_fn = MultiTaskLoss({"stage": 1.0})
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(600):
        total, _ = loss_fn(model(batch["X"]), batch)
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
    assert total.item() < 1e-2, f"loss stayed at {total.item():.4f}"


def test_stage_class_weights_favour_rare_class():
    weights = stage_class_weights(np.array([0] * 6 + [1] * 3 + [2] * 1))
    assert weights[2] > weights[1] > weights[0]
    assert weights.mean().item() == pytest.approx(1.0)


def test_split_by_engine_has_no_overlap():
    train_batch, val_batch = split_by_engine(fake_dataset())
    assert not set(train_batch["engine_id"]) & set(val_batch["engine_id"])
    assert len(val_batch["X"]) > 0


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_train_checkpoint_and_predictions(name, tmp_path):
    train_batch, val_batch = split_by_engine(fake_dataset(n_engines=6))
    data = {"train": train_batch, "val": val_batch}
    model, history = train(name, data, str(tmp_path), epochs=3, verbose=False)
    assert len(history) == 3
    assert history["train_loss"].iloc[-1] < history["train_loss"].iloc[0]

    df, attn = make_predictions(model, val_batch)
    check_predictions(df, attn)
    assert df["engine_id"].tolist() == val_batch["engine_id"].tolist()
    assert df["rul_pred"].between(0, config.RUL_CAP).all()

    reloaded, ckpt = load_checkpoint(str(tmp_path / "best.pt"))
    assert ckpt["model_name"] == name
    df_reloaded, _ = make_predictions(reloaded, val_batch)
    assert np.allclose(df["rul_pred"], df_reloaded["rul_pred"])
