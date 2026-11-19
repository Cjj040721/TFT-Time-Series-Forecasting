"""
Author     : Cao Junjia

Residual-based anomaly detection with MAD dynamic threshold.

Algorithm (matches Section 3.5 of the thesis):
    1.  e_t  = |y_t − ŷ_t|                 (absolute residual)
    2.  m_t  = median(e_{t-w+1} … e_t)      (rolling median)
    3.  MAD_t = median(|e_i − m_t|)          (rolling MAD)
    4.  τ_t  = m_t + λ · MAD_t              (dynamic threshold)
    5.  flag  = 1  if  e_t > τ_t  else 0

inject_synthetic_anomalies() creates labelled test data for evaluation
(since ETT has no native anomaly labels).
"""
from typing import Literal

import numpy as np


# ─────────────────────────── core detector ────────────────────────────────────

class ResidualMADDetector:
    """
    Parameters
    ----------
    window : rolling window size for MAD computation
    lam    : threshold multiplier  (τ = median + lam * MAD)
    """

    def __init__(self, window: int = 48, lam: float = 3.0):
        self.window = window
        self.lam    = lam

    def fit(self, residuals: np.ndarray) -> "ResidualMADDetector":
        """
        Optionally fit on a calibration set (validation).
        For the current design lam is set at construction time, so
        fit() is a no-op — kept for API consistency.
        """
        return self

    def detect(self, residuals: np.ndarray) -> dict:
        """
        Parameters
        ----------
        residuals : 1-D array |y_t - ŷ_t|, shape (T,)

        Returns
        -------
        dict with keys:
            residuals   : same input
            threshold   : τ_t  shape (T,)
            anomaly_flag: binary 0/1, shape (T,)
            rolling_med : rolling median m_t
            rolling_mad : rolling MAD
        """
        residuals = np.asarray(residuals, dtype=np.float64).flatten()
        T = len(residuals)
        w = self.window

        rolling_med = np.full(T, np.nan)
        rolling_mad = np.full(T, np.nan)
        threshold   = np.full(T, np.nan)

        for t in range(T):
            start = max(0, t - w + 1)
            window_data = residuals[start: t + 1]

            m_t = np.median(window_data)
            mad_t = np.median(np.abs(window_data - m_t))

            rolling_med[t] = m_t
            rolling_mad[t] = mad_t
            threshold[t]   = m_t + self.lam * mad_t

        anomaly_flag = (residuals > threshold).astype(int)

        return {
            "residuals":    residuals,
            "threshold":    threshold,
            "anomaly_flag": anomaly_flag,
            "rolling_med":  rolling_med,
            "rolling_mad":  rolling_mad,
        }

    def __call__(self, residuals: np.ndarray) -> dict:
        return self.detect(residuals)


# ─────────────────────────── anomaly injection ───────────────────────────────

AnomalyType = Literal["spike", "level_shift", "drift", "dropout"]


def inject_synthetic_anomalies(
    series:     np.ndarray,
    anomaly_types: list = None,
    seed:       int  = 42,
    n_anomalies: int = 5,
    duration_range: tuple = (4, 24),
    magnitude:  float = 3.0,
) -> tuple:
    """
    Inject synthetic anomalies into a 1-D time series and return
    the contaminated series plus a binary label array.

    Parameters
    ----------
    series          : original 1-D float array (T,)
    anomaly_types   : list of AnomalyType to sample from
                      default: ["spike", "level_shift", "drift", "dropout"]
    seed            : reproducibility
    n_anomalies     : number of anomaly segments to inject
    duration_range  : (min_len, max_len) steps per segment
    magnitude       : anomaly amplitude as multiple of series std

    Returns
    -------
    contaminated : (T,)  float — series with anomalies injected
    labels       : (T,)  int   — 1 at anomaly positions, 0 otherwise
    """
    rng = np.random.default_rng(seed)
    if anomaly_types is None:
        anomaly_types = ["spike", "level_shift", "drift", "dropout"]

    series = series.copy().astype(float)
    T      = len(series)
    std    = float(np.std(series))
    labels = np.zeros(T, dtype=int)

    for _ in range(n_anomalies):
        atype    = rng.choice(anomaly_types)
        dur      = int(rng.integers(*duration_range))
        start    = int(rng.integers(0, max(1, T - dur)))
        end      = min(start + dur, T)
        amp      = magnitude * std

        if atype == "spike":
            # Single-point impulse
            idx = rng.integers(start, end)
            series[idx] += float(rng.choice([-1, 1])) * amp
            labels[idx] = 1

        elif atype == "level_shift":
            shift = float(rng.choice([-1, 1])) * amp
            series[start:end] += shift
            labels[start:end] = 1

        elif atype == "drift":
            drift = np.linspace(0, float(rng.choice([-1, 1])) * amp, end - start)
            series[start:end] += drift
            labels[start:end] = 1

        elif atype == "dropout":
            # Flat-line (sensor loss)
            fill = float(series[start]) if start > 0 else 0.0
            series[start:end] = fill
            labels[start:end] = 1

    return series, labels
