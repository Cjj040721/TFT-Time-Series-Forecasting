"""
Author     : Cao Junjia

Model factory — returns a model instance given a name string and config dict.

Usage
-----
from src.model.factory import build_model

model = build_model(
    model_name = "tft",          # "tft" | "lstm" | "informer"
    n_features      = 7,
    n_time_features = 4,
    seq_len    = 96,
    label_len  = 48,
    pred_len   = 24,
    freq       = "h",
)
"""
import torch.nn as nn


def build_model(
    model_name:      str,
    n_features:      int,
    n_time_features: int,
    seq_len:         int,
    label_len:       int,
    pred_len:        int,
    freq:            str   = "h",
    d_model:         int   = 128,
    n_heads:         int   = 4,
    lstm_layers:     int   = 2,
    hidden_size:     int   = 256,
    dropout:         float = 0.1,
    third_party_root: str  = "third_party/informer2020",
    **kwargs,
) -> nn.Module:
    """
    Parameters
    ----------
    model_name       : one of "tft", "lstm", "informer"
    n_features       : total observed feature count (D)
    n_time_features  : number of time-encoding features (F)
    Everything else  : forwarded to the relevant model constructor.

    Returns
    -------
    nn.Module — not yet moved to device; call .to(device) afterwards.
    """
    name = model_name.lower().strip()

    if name == "tft":
        from src.model.tft import TemporalFusionTransformer
        return TemporalFusionTransformer(
            n_hist_vars   = n_features,
            n_future_vars = n_time_features,
            d_model       = d_model,
            n_heads       = n_heads,
            lstm_layers   = lstm_layers,
            pred_len      = pred_len,
            dropout       = dropout,
        )

    elif name == "lstm":
        from src.model.baselines.lstm import LSTMBaseline
        return LSTMBaseline(
            n_features  = n_features,
            hidden_size = hidden_size,
            num_layers  = lstm_layers,
            pred_len    = pred_len,
            dropout     = dropout,
        )

    elif name in ("informer", "informer2020"):
        from src.model.baselines.informer_wrapper import build_informer
        return build_informer(
            n_features       = n_features,
            seq_len          = seq_len,
            label_len        = label_len,
            pred_len         = pred_len,
            freq             = freq,
            third_party_root = third_party_root,
            **kwargs,
        )

    else:
        raise ValueError(
            f"Unknown model_name='{model_name}'. "
            "Choose from: 'tft', 'lstm', 'informer'."
        )
