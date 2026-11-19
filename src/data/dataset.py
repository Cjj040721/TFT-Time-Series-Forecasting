"""
Author     : Cao Junjia

PyTorch Dataset for ETT (and generic CSV) time-series.

Sliding-window construction:
    Input  : x[i : i+seq_len]          shape (seq_len,  n_features)
    Label  : x[i+seq_len-label_len :   shape (label_len+pred_len, n_features)
              i+seq_len+pred_len]
    Time   : same windows from time_feature array

The label window overlaps the input by label_len steps — this mirrors the
Informer / TFT convention where the decoder sees the last label_len steps of
historical data before predicting pred_len future steps.

Usage
-----
from src.data.dataset import ETTDatasetManager

mgr = ETTDatasetManager("data/raw/ETT/ETTh1.csv", seq_len=96, label_len=48,
                         pred_len=24, freq="h", target="OT")
train_loader = mgr.get_loader("train", batch_size=32)
val_loader   = mgr.get_loader("val",   batch_size=32, shuffle=False)
test_loader  = mgr.get_loader("test",  batch_size=32, shuffle=False)
"""
import os
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.features import time_features
from src.data.preprocess import StandardScaler, train_val_test_split


# ──────────────────────────────────────────────────────────────────────────────
#  Core sliding-window dataset
# ──────────────────────────────────────────────────────────────────────────────

class _SlidingWindowDataset(Dataset):
    """
    Internal class — built by ETTDatasetManager.
    """

    def __init__(
        self,
        data:     np.ndarray,   # (T, D)   normalised features (includes target)
        time_enc: np.ndarray,   # (T, F)   time features
        seq_len:   int,
        label_len: int,
        pred_len:  int,
        target_idx: int,        # column index of the target variable
    ):
        self.data       = data.astype(np.float32)
        self.time_enc   = time_enc.astype(np.float32)
        self.seq_len    = seq_len
        self.label_len  = label_len
        self.pred_len   = pred_len
        self.target_idx = target_idx

        # Total valid start positions
        self.n_samples = len(data) - seq_len - pred_len + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        s  = idx
        e  = s + self.seq_len
        # Decoder input: last label_len of encoder + pred_len zeros (generative)
        ds = e - self.label_len
        de = ds + self.label_len + self.pred_len

        # ── encoder ──
        enc_x    = self.data[s:e]                 # (seq_len, D)
        enc_t    = self.time_enc[s:e]             # (seq_len, F)

        # ── decoder ──
        dec_x    = self.data[ds:de].copy()        # (label_len+pred_len, D)
        dec_x[self.label_len:] = 0.0              # zero out future pred_len steps to prevent leakage
        dec_t    = self.time_enc[ds:de]

        # ── target (prediction ground truth) ──
        target   = self.data[e: e + self.pred_len, self.target_idx]  # (pred_len,)

        return (
            torch.tensor(enc_x),
            torch.tensor(enc_t),
            torch.tensor(dec_x),
            torch.tensor(dec_t),
            torch.tensor(target),
        )


# ──────────────────────────────────────────────────────────────────────────────
#  Public manager
# ──────────────────────────────────────────────────────────────────────────────

class ETTDatasetManager:
    """
    Loads an ETT CSV (or any CSV with a 'date' column), fits the scaler on
    train, and exposes get_loader() for each split.

    Parameters
    ----------
    csv_path   : path to ETTh1.csv / ETTh2.csv / ETTm1.csv etc.
    seq_len    : encoder input length  (e.g. 96, 168, 336)
    label_len  : decoder known history (e.g. 48)
    pred_len   : forecast horizon      (e.g. 24, 48, 168)
    freq       : 'h' hourly | 't' 15-min
    target     : name of target column (default 'OT')
    """

    SPLITS: dict = {
        "train": 0,
        "val":   1,
        "test":  2,
    }

    def __init__(
        self,
        csv_path:  str,
        seq_len:   int = 96,
        label_len: int = 48,
        pred_len:  int = 24,
        freq:      str = "h",
        target:    str = "OT",
    ):
        self.seq_len   = seq_len
        self.label_len = label_len
        self.pred_len  = pred_len
        self.freq      = freq
        self.target    = target

        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["date"])

        # ── build time features across the whole series ──
        self.time_enc = time_features(df["date"], freq=freq)   # (T, F)

        # ── feature columns (everything except 'date') ──
        feat_cols = [c for c in df.columns if c != "date"]
        self.target_idx = feat_cols.index(target)
        raw_data = df[feat_cols].values.astype(np.float32)     # (T, D)
        self.n_features = raw_data.shape[1]

        # ── chronological split ──
        splits = self._split_indices(len(raw_data))

        # ── fit scaler on training portion only ──
        self.scaler = StandardScaler()
        train_slice = slice(*splits["train"])
        self.scaler.fit(raw_data[train_slice])

        # ── normalise full series ──
        norm_data = self.scaler.transform(raw_data)

        # ── build per-split datasets ──
        self._datasets: dict = {}
        for name, (start, end) in splits.items():
            self._datasets[name] = _SlidingWindowDataset(
                data       = norm_data[start:end],
                time_enc   = self.time_enc[start:end],
                seq_len    = seq_len,
                label_len  = label_len,
                pred_len   = pred_len,
                target_idx = self.target_idx,
            )

    # ── index calculation ─────────────────────────────────────────────────────

    def _split_indices(self, n: int) -> dict:
        n_train = int(n * 12 / 20)
        n_val   = int(n *  4 / 20)
        return {
            "train": (0,              n_train),
            "val":   (n_train,        n_train + n_val),
            "test":  (n_train + n_val, n),
        }

    # ── public API ────────────────────────────────────────────────────────────

    def get_dataset(self, split: Literal["train", "val", "test"]) -> Dataset:
        return self._datasets[split]

    def get_loader(
        self,
        split:      Literal["train", "val", "test"],
        batch_size: int  = 32,
        shuffle:    bool = True,
        num_workers: int = 0,
    ) -> DataLoader:
        ds = self._datasets[split]
        return DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = shuffle if split == "train" else False,
            num_workers = num_workers,
            pin_memory  = False,
            drop_last   = split == "train",
        )

    @property
    def n_time_features(self) -> int:
        return self.time_enc.shape[1]
