"""
Author     : Cao Junjia

Predict module — loads a trained model and runs inference on a given
numpy input array. Used by the FastAPI backend.

Usage (programmatic)
--------------------
from src.experiment.predict import Predictor

p = Predictor("configs/tft_etth1_96_24.yaml")
pred = p.predict(enc_x, enc_t, dec_x, dec_t)   # → np.ndarray (pred_len,)
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.model.factory import build_model
from src.utils.io import load_config


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Predictor:
    """
    Thin inference wrapper.
    Handles model loading, device placement, and inverse scaling.
    """

    def __init__(self, config_path: str, scaler=None):
        self.cfg    = load_config(config_path)
        self.device = _device()
        self.scaler = scaler   # optional StandardScaler for inverse transform

        self.model = build_model(
            model_name      = self.cfg["model"],
            n_features      = self.cfg["n_features"],
            n_time_features = self.cfg["n_time_features"],
            seq_len         = self.cfg["seq_len"],
            label_len       = self.cfg["label_len"],
            pred_len        = self.cfg["pred_len"],
            freq            = self.cfg.get("freq", "h"),
            d_model         = self.cfg.get("d_model", 128),
            n_heads         = self.cfg.get("n_heads", 4),
            lstm_layers     = self.cfg.get("lstm_layers", 2),
            hidden_size     = self.cfg.get("hidden_size", 256),
            dropout         = self.cfg.get("dropout", 0.1),
            c_out           = self.cfg.get("c_out", 1),
        ).to(self.device)

        ckpt_dir = os.path.join(
            "results/checkpoints",
            f"{self.cfg['model']}_{self.cfg['dataset']}_"
            f"{self.cfg['seq_len']}_{self.cfg['pred_len']}",
        )
        ckpt = os.path.join(ckpt_dir, "best_model.pt")
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device, weights_only=True))
        self.model.eval()

    def predict(
        self,
        enc_x: np.ndarray,     # (seq_len, D)
        enc_t: np.ndarray,     # (seq_len, F)
        dec_x: np.ndarray,     # (label_len+pred_len, D)
        dec_t: np.ndarray,     # (label_len+pred_len, F)
        inverse: bool = True,
    ) -> np.ndarray:
        """
        Returns
        -------
        pred : (pred_len,)  float32
        """
        def _t(x):
            return torch.tensor(x[None], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            if self.cfg["model"] == "lstm":
                pred = self.model(_t(enc_x)).squeeze(0)
            else:
                # Zero out the future pred_len steps to prevent data leakage
                label_len = self.cfg.get("label_len", 48)
                dec_x = dec_x.copy()
                dec_x[label_len:] = 0.0
                pred = self.model(_t(enc_x), _t(enc_t),
                                  _t(dec_x), _t(dec_t)).squeeze(0)

        pred = pred.cpu().numpy()   # (pred_len,)

        if inverse and self.scaler is not None:
            ti   = self.cfg.get("target_idx", -1)
            pred = pred * self.scaler.std_[0, ti] + self.scaler.mean_[0, ti]

        return pred

    @property
    def attention(self) -> np.ndarray | None:
        """Return last temporal attention (TFT only)."""
        if hasattr(self.model, "last_attention") and self.model.last_attention is not None:
            return self.model.last_attention[0].cpu().numpy()
        return None

    @property
    def var_weights(self) -> np.ndarray | None:
        """Return last variable selection weights (TFT only)."""
        if hasattr(self.model, "last_var_weights") and self.model.last_var_weights is not None:
            return self.model.last_var_weights[0].mean(0).cpu().numpy()
        return None
