"""Member B: CNN baseline, CNN+LSTM, attention multi-head model, training loop.

Every model's forward returns a dict that passes contracts.check_model_output.
Every run writes predictions.csv and attn.npy that pass contracts.check_predictions.
"""
