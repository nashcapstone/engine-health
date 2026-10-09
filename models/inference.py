"""Inference on raw sensor readings, for the output card.

    from models.inference import Predictor
    predictor = Predictor.load("artifacts/runs/attention_real")
    result = predictor.predict(raw_rows)      # one engine's raw rows, config.RAW_COLUMNS

Raw rows are scaled with the scaler fitted on the training engines
(artifacts/data/scaler.json, written by `python -m data.preprocess`), then the last
config.WINDOW cycles up to the requested cycle go through the model.
"""

import json
import os

import numpy as np
import torch

import config
from data.preprocess import OUT_DIR as DATA_DIR
from data.preprocess import MinMaxScaler
from models.heads import HEALTH_MAX
from models.train import load_checkpoint


def window_from_rows(rows, scaler, cycle=None):
    """(WINDOW, N_SENSORS) float32 input from one engine's raw rows, ending at `cycle`.

    cycle defaults to the engine's last recorded cycle. An engine with fewer than
    WINDOW cycles is padded with its first reading, as in data.preprocess.
    """
    engines = rows["engine_id"].unique()
    assert len(engines) == 1, f"rows hold {len(engines)} engines, expected one"
    rows = rows.sort_values("cycle")
    if cycle is not None:
        assert cycle in set(rows["cycle"]), f"engine {engines[0]} has no cycle {cycle}"
        rows = rows[rows["cycle"] <= cycle]
    values = scaler.transform(rows)[list(config.SENSORS)].to_numpy(dtype=np.float32)[-config.WINDOW :]
    if len(values) < config.WINDOW:
        values = np.concatenate([np.repeat(values[:1], config.WINDOW - len(values), axis=0), values])
    return values


class Predictor:
    def __init__(self, model, scaler, model_name=None):
        self.model = model.eval()
        self.scaler = scaler
        self.model_name = model_name

    @classmethod
    def load(cls, run_dir, data_dir=DATA_DIR, device="cpu"):
        """run_dir holds best.pt. data_dir holds scaler.json."""
        model, ckpt = load_checkpoint(os.path.join(run_dir, "best.pt"), device)
        with open(os.path.join(data_dir, "scaler.json")) as f:
            scaler = MinMaxScaler.from_dict(json.load(f))
        return cls(model, scaler, ckpt["model_name"])

    @torch.no_grad()
    def predict(self, rows, cycle=None, top_k=3):
        """Prediction for one engine at one cycle. Returns a plain dict.

        Keys: engine_id, cycle, rul, health, stage, stage_name, stage_probs, and, for
        models with attention, attn (WINDOW, N_SENSORS) and top_sensors, the top_k
        (sensor, mean attention over the window) pairs.
        """
        x = window_from_rows(rows, self.scaler, cycle)
        device = next(self.model.parameters()).device
        out = self.model(torch.from_numpy(x)[None].to(device))
        probs = torch.softmax(out["stage_logits"][0], dim=-1).cpu().numpy()
        stage = int(probs.argmax())
        result = {
            "engine_id": int(rows["engine_id"].iloc[0]),
            "cycle": int(cycle if cycle is not None else rows["cycle"].max()),
            "rul": float(np.clip(out["rul"][0].item(), 0, config.RUL_CAP)),
            "health": float(np.clip(out["health"][0].item(), 0, HEALTH_MAX)),
            "stage": stage,
            "stage_name": config.STAGE_NAMES[stage],
            "stage_probs": probs.tolist(),
            "attn": None,
            "top_sensors": [],
        }
        if out["attn"] is not None:
            attn = out["attn"][0].cpu().numpy()
            per_sensor = attn.mean(axis=0)
            order = np.argsort(per_sensor)[::-1][:top_k]
            result["attn"] = attn
            result["top_sensors"] = [(config.SENSORS[j], float(per_sensor[j])) for j in order]
        return result
