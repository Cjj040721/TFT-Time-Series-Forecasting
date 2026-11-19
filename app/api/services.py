"""
Author     : Cao Junjia

ForecastService — business logic layer between the API routes and the ML code.

Responsibilities:
1. Load model via Predictor
2. Run inference
3. Compute residuals & MAD anomaly flags
4. Return a unified result dict
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.experiment.predict import Predictor
from src.evaluation.anomaly import ResidualMADDetector
from src.utils.io import load_config


class ForecastService:
    def __init__(self, config_path: str):
        self.cfg = load_config(config_path)
        self.predictor = Predictor(config_path)
        self.detector = ResidualMADDetector(
            window=self.cfg.get("anomaly_window", 48),
            lam=self.cfg.get("anomaly_lam", 3.0),
        )

    def forecast(
        self,
        enc_x: np.ndarray,
        enc_t: np.ndarray,
        dec_x: np.ndarray,
        dec_t: np.ndarray,
    ) -> dict:
        """
        Returns
        -------
        dict with keys:
            prediction    (pred_len,)
            anomaly_flag  (pred_len,)
            threshold     (pred_len,)
            residual      (pred_len,)   — zeros on first call (no ground truth)
            attention     (pred_len, seq_len) | None
            var_weights   (n_vars,) | None
        """
        pred = self.predictor.predict(enc_x, enc_t, dec_x, dec_t, inverse=True)

        # Residual is zero when ground truth is unavailable at serving time.
        # The frontend can pass actual values later for monitoring.
        residual = np.zeros_like(pred)

        anom = self.detector.detect(residual)

        return {
            "prediction":   pred,
            "residual":     residual,
            "anomaly_flag": anom["anomaly_flag"],
            "threshold":    anom["threshold"],
            "attention":    _to_list_or_none(self.predictor.attention),
            "var_weights":  _to_list_or_none(self.predictor.var_weights),
        }


def _to_list_or_none(arr):
    if arr is None:
        return None
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    return arr
