"""
Author     : Cao Junjia

Time-feature engineering.
Converts a pandas DatetimeIndex into a float32 array of cyclic/ordinal features.
Supports hourly (ETTh) and minutely (ETTm) granularity.
"""
import numpy as np
import pandas as pd


# ─────────────────────── individual feature extractors ────────────────────────

def _month_of_year(idx: pd.DatetimeIndex) -> np.ndarray:
    return (idx.month - 1) / 11.0                     # [0, 1]


def _day_of_month(idx: pd.DatetimeIndex) -> np.ndarray:
    return (idx.day - 1) / 30.0


def _day_of_week(idx: pd.DatetimeIndex) -> np.ndarray:
    return idx.dayofweek / 6.0                         # Mon=0, Sun=6 → [0,1]


def _hour_of_day(idx: pd.DatetimeIndex) -> np.ndarray:
    return idx.hour / 23.0


def _minute_of_hour(idx: pd.DatetimeIndex) -> np.ndarray:
    return idx.minute / 59.0


# ─────────────────────────── public API ───────────────────────────────────────

GRANULARITY_FEATURES = {
    "h":  [_month_of_year, _day_of_month, _day_of_week, _hour_of_day],
    "t":  [_month_of_year, _day_of_month, _day_of_week, _hour_of_day, _minute_of_hour],
}


def time_features(
    timestamps: pd.Series,
    freq: str = "h",
) -> np.ndarray:
    """
    Parameters
    ----------
    timestamps : pd.Series of datetime64 (or string dates)
    freq       : 'h' for hourly ETTh*, 't' for 15-min ETTm*

    Returns
    -------
    np.ndarray  shape (T, n_features)  float32, values in [0, 1]
    """
    idx = pd.DatetimeIndex(timestamps)
    funcs = GRANULARITY_FEATURES.get(freq, GRANULARITY_FEATURES["h"])
    feats = np.stack([f(idx) for f in funcs], axis=-1).astype(np.float32)
    return feats


def n_time_features(freq: str = "h") -> int:
    return len(GRANULARITY_FEATURES.get(freq, GRANULARITY_FEATURES["h"]))
