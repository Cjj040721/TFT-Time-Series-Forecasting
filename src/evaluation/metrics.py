"""
Author     : Cao Junjia

Evaluation metrics for both forecasting and anomaly detection tasks.

Forecasting
-----------
    mae, mse, rmse, mape

Anomaly Detection
-----------------
    precision_recall_f1   — requires binary labels

Usage
-----
from src.evaluation.metrics import compute_forecast_metrics, compute_anomaly_metrics
"""
import numpy as np


# ─────────────────────────── forecasting ─────────────────────────────────────

def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def mse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean((pred - true) ** 2))


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(mse(pred, true)))


def mape(pred: np.ndarray, true: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean(np.abs((pred - true) / (np.abs(true) + eps))) * 100)


def compute_forecast_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """
    Returns
    -------
    dict with keys: mae, mse, rmse, mape
    All values are Python floats (JSON-serialisable).
    """
    return {
        "mae":  mae(pred, true),
        "mse":  mse(pred, true),
        "rmse": rmse(pred, true),
        "mape": mape(pred, true),
    }


# ─────────────────────────── anomaly detection ────────────────────────────────

def precision_recall_f1(
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
) -> dict:
    """
    Parameters
    ----------
    pred_labels : binary (0/1) array — model's anomaly flags
    true_labels : binary (0/1) array — ground-truth labels

    Returns
    -------
    dict with keys: precision, recall, f1, tp, fp, fn
    """
    pred_labels = np.asarray(pred_labels, dtype=int).flatten()
    true_labels = np.asarray(true_labels, dtype=int).flatten()

    tp = int(np.sum((pred_labels == 1) & (true_labels == 1)))
    fp = int(np.sum((pred_labels == 1) & (true_labels == 0)))
    fn = int(np.sum((pred_labels == 0) & (true_labels == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
        "tp":        tp,
        "fp":        fp,
        "fn":        fn,
    }


def compute_anomaly_metrics(
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
) -> dict:
    return precision_recall_f1(pred_labels, true_labels)
