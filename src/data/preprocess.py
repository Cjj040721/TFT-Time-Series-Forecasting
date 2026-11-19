"""
Author     : Cao Junjia

Data preprocessing utilities.

StandardScaler
    - fit on training split only (no data leakage)
    - transform / inverse_transform on any split

train_val_test_split
    - chronological 12:4:4 ratio, returns (train_df, val_df, test_df)
"""
import numpy as np
import pandas as pd


# ─────────────────────────── scaler ──────────────────────────────────────────

class StandardScaler:
    """
    Z-score normalisation fitted on training data.
    Statistics are per-feature (column-wise).
    """

    def __init__(self):
        self.mean_: np.ndarray = None
        self.std_: np.ndarray = None

    def fit(self, data: np.ndarray) -> "StandardScaler":
        """
        data : shape (N, D)  — training split only
        """
        self.mean_ = data.mean(axis=0, keepdims=True)     # (1, D)
        self.std_  = data.std(axis=0, keepdims=True)      # (1, D)
        self.std_  = np.where(self.std_ < 1e-8, 1.0, self.std_)  # avoid div-by-zero
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean_) / self.std_

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.std_ + self.mean_

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.fit(data).transform(data)

    # Allow pickling for API serving
    def __repr__(self):
        return f"StandardScaler(mean={self.mean_}, std={self.std_})"


# ─────────────────────────── split ───────────────────────────────────────────

def train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 12 / 20,
    val_ratio: float   =  4 / 20,
) -> tuple:
    """
    Chronological split (no shuffling).
    Default: 12:4:4  (matches ETT benchmark convention).

    Returns
    -------
    (train_df, val_df, test_df)  — pandas DataFrames with identical columns
    """
    n = len(df)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df   = df.iloc[n_train: n_train + n_val].reset_index(drop=True)
    test_df  = df.iloc[n_train + n_val:].reset_index(drop=True)

    return train_df, val_df, test_df
