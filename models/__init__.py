"""Member B: CNN baseline, CNN+LSTM, attention multi-head model, training loop.

Every model's forward returns a dict that passes contracts.check_model_output.
Every run writes predictions.csv and attn.npy that pass contracts.check_predictions.
"""

from models.cnn import CNNBaseline
from models.cnn_lstm import CNNLSTM

MODELS = {"cnn": CNNBaseline, "cnn_lstm": CNNLSTM}


def build_model(name, **kwargs):
    assert name in MODELS, f"unknown model {name!r}, expected one of {sorted(MODELS)}"
    return MODELS[name](**kwargs)
