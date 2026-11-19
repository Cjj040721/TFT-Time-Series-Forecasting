"""
Author     : Cao Junjia

Unified I/O helpers for experiment results.
Saves metrics, predictions, residuals, anomaly flags to disk.
"""
import json
import os

import numpy as np
import pandas as pd
import yaml


# ─────────────────────────── config ──────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_config(cfg: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, allow_unicode=True)


# ─────────────────────────── metrics ─────────────────────────────────────────

def save_metrics(metrics: dict, path: str) -> None:
    """Save a dict of scalar metrics as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    print(f"[io] metrics saved → {path}")


def load_metrics(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


# ─────────────────────────── arrays ──────────────────────────────────────────

def save_array(arr: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.endswith(".npy"):
        np.save(path, arr)
    else:
        np.savetxt(path, arr, delimiter=",")
    print(f"[io] array saved → {path}")


def load_array(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path)
    return np.loadtxt(path, delimiter=",")


# ─────────────────────────── dataframes ──────────────────────────────────────

def save_df(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[io] dataframe saved → {path}")


def load_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# ─────────────────────────── standard result bundle ─────────────────────────

def save_result_bundle(
    out_dir: str,
    pred: np.ndarray,
    true: np.ndarray,
    metrics: dict,
    residual: np.ndarray = None,
    anomaly: np.ndarray = None,
    attention: np.ndarray = None,
    config: dict = None,
) -> None:
    """
    Save a complete experiment result to out_dir/.
    Directory layout matches the thesis results/ specification:
        metrics.json, pred.csv, true.csv,
        residual.csv, anomaly.csv, attention.npy, config.yaml
    """
    os.makedirs(out_dir, exist_ok=True)

    save_metrics(metrics, os.path.join(out_dir, "metrics.json"))

    pd.DataFrame(pred).to_csv(os.path.join(out_dir, "pred.csv"), index=False)
    pd.DataFrame(true).to_csv(os.path.join(out_dir, "true.csv"), index=False)

    if residual is not None:
        pd.DataFrame(residual).to_csv(os.path.join(out_dir, "residual.csv"), index=False)

    if anomaly is not None:
        pd.DataFrame(anomaly).to_csv(os.path.join(out_dir, "anomaly.csv"), index=False)

    if attention is not None:
        np.save(os.path.join(out_dir, "attention.npy"), attention)

    if config is not None:
        save_config(config, os.path.join(out_dir, "config.yaml"))

    print(f"[io] full result bundle saved → {out_dir}")
