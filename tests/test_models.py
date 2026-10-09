import os

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
    best = float("inf")
    for _ in range(600):
        total, _ = loss_fn(model(batch["X"]), batch)
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        # Best, not last: Adam can spike briefly late in a run without the model losing the fit
        best = min(best, total.item())
    assert best < 1e-2, f"loss never went below {best:.4f}"


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


def test_attention_sums_to_one_over_sensors():
    out = build_model("attention")(small_batch()["X"])
    attn = out["attn"].detach()
    assert attn.shape == (32, config.WINDOW, config.N_SENSORS)
    assert (attn >= 0).all()
    assert torch.allclose(attn.sum(dim=-1), torch.ones(32, config.WINDOW), atol=1e-5)


def test_uniform_attention_with_scale_leaves_input_unchanged():
    torch.manual_seed(0)
    model = build_model("attention", dropout=0.0).eval()
    last = model.attention.score[-1]
    torch.nn.init.zeros_(last.weight)
    torch.nn.init.zeros_(last.bias)
    x = small_batch()["X"]
    with torch.no_grad():
        out = model(x)
        plain = model.backbone(x)
    assert torch.allclose(out["attn"], torch.full_like(out["attn"], 1 / config.N_SENSORS))
    assert torch.allclose(out["rul"], plain["rul"], atol=1e-5)


def test_attention_run_writes_attn_file(tmp_path):
    from models.train import write_predictions

    train_batch, val_batch = split_by_engine(fake_dataset(n_engines=6))
    model, _ = train("attention", {"train": train_batch, "val": val_batch}, str(tmp_path), epochs=2, verbose=False)
    df, attn = write_predictions(model, val_batch, str(tmp_path))
    saved = np.load(tmp_path / "attn.npy")
    check_predictions(df, saved)
    assert np.allclose(saved, attn)


def test_ablation_variants_build_and_run(tmp_path, monkeypatch):
    import pandas as pd

    from models import ablation

    monkeypatch.setattr(ablation, "ABLATION_DIR", str(tmp_path))
    for name, (model_name, model_kwargs, _) in ablation.VARIANTS.items():
        assert model_name in MODELS, f"{name} uses unknown model {model_name}"
        build_model(model_name, **model_kwargs)

    train_batch, val_batch = split_by_engine(fake_dataset(n_engines=6))
    data = {"train": train_batch, "val": val_batch, "test": val_batch}
    out_dir = ablation.run_variant("attention_no_stage", 0, data, epochs=1)
    for name in ("best.pt", "predictions.csv", "attn.npy", "val_predictions.csv", "val_attn.npy"):
        assert os.path.exists(os.path.join(out_dir, name)), f"{name} was not written"
    check_predictions(pd.read_csv(os.path.join(out_dir, "val_predictions.csv")))
    # a finished run is skipped, not retrained
    mtime = os.path.getmtime(os.path.join(out_dir, "best.pt"))
    ablation.run_variant("attention_no_stage", 0, data, epochs=1)
    assert os.path.getmtime(os.path.join(out_dir, "best.pt")) == mtime
