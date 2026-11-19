"""
Author     : Cao Junjia

Centralised plotting module — generates every figure required by the thesis.

All figures:
  - Use English labels (required for thesis)
  - Are saved as PDF (vector) or PNG (screenshots / bitmaps)
  - Follow naming convention: fig{ch}_{idx}_{description}.pdf

Entry points
------------
plot_data_slice          → Chapter 6: dataset overview
plot_loss_curve          → Chapter 6: training convergence
plot_prediction_vs_truth → Chapter 6: forecasting quality
plot_residual_anomaly    → Chapter 6: anomaly detection
plot_metric_comparison   → Chapter 6: model / horizon comparison
plot_attention_heatmap   → Chapter 6: interpretability
plot_variable_importance → Chapter 6: variable selection weights
"""
import os

import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import numpy as np

# ── global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "legend.fontsize":   10,
    "figure.dpi":        150,
    "axes.grid":         True,
    "grid.alpha":        0.3,
})

COLORS = {
    "truth":     "#2C3E50",
    "pred":      "#E74C3C",
    "residual":  "#3498DB",
    "threshold": "#E67E22",
    "anomaly":   "#E74C3C",
    "train":     "#2980B9",
    "val":       "#E67E22",
}


# ─────────────────────────── helpers ─────────────────────────────────────────

def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] saved → {path}")


# ─────────────────────────── 1. data slice ────────────────────────────────────

def plot_data_slice(
    df,
    columns: list,
    n_points: int = 500,
    title: str = "ETTh1 — Sample Time Series",
    save_path: str = None,
):
    """
    Plot a subset of raw ETT columns.

    Parameters
    ----------
    df       : pandas DataFrame with a 'date' column
    columns  : list of column names to plot
    n_points : number of time steps to show
    """
    fig, axes = plt.subplots(len(columns), 1,
                              figsize=(12, 2.5 * len(columns)),
                              sharex=True)
    if len(columns) == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        vals = df[col].values[:n_points]
        ax.plot(vals, linewidth=0.8, color=COLORS["truth"])
        ax.set_ylabel(col)

    axes[-1].set_xlabel("Time step")
    axes[0].set_title(title)
    fig.tight_layout()

    if save_path:
        _save(fig, save_path)
    return fig


# ─────────────────────────── 2. loss curve ───────────────────────────────────

def plot_loss_curve(
    train_losses: list,
    val_losses:   list,
    title: str    = "Training & Validation Loss",
    save_path: str = None,
):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="Train MSE", color=COLORS["train"])
    ax.plot(val_losses,   label="Val MSE",   color=COLORS["val"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if save_path:
        _save(fig, save_path)
    return fig


# ─────────────────────────── 3. prediction vs truth ─────────────────────────

def plot_prediction_vs_truth(
    pred:       np.ndarray,
    true:       np.ndarray,
    n_show:     int   = 200,
    title:      str   = "Ground Truth vs Prediction",
    save_path:  str   = None,
):
    """
    pred, true : 1-D arrays (flattened over samples × pred_len)
    """
    pred = pred.flatten()[:n_show]
    true = true.flatten()[:n_show]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(true, label="Ground Truth", color=COLORS["truth"], linewidth=1.2)
    ax.plot(pred, label="Prediction",   color=COLORS["pred"],  linewidth=1.0,
            linestyle="--")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Normalised OT")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if save_path:
        _save(fig, save_path)
    return fig


# ─────────────────────────── 4. residual + anomaly ───────────────────────────

def plot_residual_anomaly(
    residuals:    np.ndarray,
    threshold:    np.ndarray,
    anomaly_flag: np.ndarray,
    true_labels:  np.ndarray = None,
    n_show:       int  = 300,
    title:        str  = "Residual & MAD Anomaly Detection",
    save_path:    str  = None,
):
    """
    residuals, threshold, anomaly_flag : 1-D arrays (T,)
    true_labels : optional ground-truth binary labels (for visualising TP/FP)
    """
    t = np.arange(min(n_show, len(residuals)))
    res = residuals[:n_show]
    thr = threshold[:n_show]
    flg = anomaly_flag[:n_show]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, res, label="Residual |e_t|", color=COLORS["residual"],
            linewidth=0.9)
    ax.plot(t, thr, label="MAD Threshold τ_t", color=COLORS["threshold"],
            linewidth=1.2, linestyle="--")

    # Mark detected anomalies
    anom_idx = np.where(flg > 0)[0]
    ax.scatter(t[anom_idx], res[anom_idx], color=COLORS["anomaly"],
               zorder=5, s=20, label="Detected anomaly")

    if true_labels is not None:
        tl = true_labels[:n_show]
        true_idx = np.where(tl > 0)[0]
        ax.axvspan(0, 0, alpha=0, label="Injected anomaly (shaded)")   # legend entry
        for idx in true_idx:
            ax.axvspan(max(0, idx - 0.5), idx + 0.5, alpha=0.15,
                       color="#8E44AD", linewidth=0)

    ax.set_xlabel("Time step")
    ax.set_ylabel("Absolute Residual")
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()

    if save_path:
        _save(fig, save_path)
    return fig


# ─────────────────────────── 5. metric comparison ────────────────────────────

def plot_metric_comparison(
    results:   dict,
    metric:    str   = "mae",
    xlabel:    str   = "Model",
    title:     str   = None,
    save_path: str   = None,
):
    """
    Bar chart comparing a metric across models or horizons.

    Parameters
    ----------
    results : dict  {label: value}
              e.g. {"TFT": 0.42, "LSTM": 0.57, "Informer": 0.51}
              or   {"96→24": 0.42, "168→48": 0.50, "336→168": 0.61}
    """
    labels = list(results.keys())
    values = [results[k] for k in labels]
    title  = title or f"{metric.upper()} Comparison"

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4), 4))
    bars = ax.bar(labels, values, color="#3498DB", edgecolor="white", width=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel(metric.upper())
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()

    if save_path:
        _save(fig, save_path)
    return fig


# ─────────────────────────── 6. attention heatmap ────────────────────────────

def plot_attention_heatmap(
    attn:      np.ndarray,
    save_path: str = None,
    title:     str = "Temporal Self-Attention Weights",
):
    """
    attn : (pred_len, seq_len)  — from extract_attention_weights()
    """
    fig, ax = plt.subplots(figsize=(12, max(3, attn.shape[0] * 0.4)))
    im = ax.imshow(attn, aspect="auto", cmap="YlOrRd", origin="upper")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    ax.set_xlabel("Encoder time step (seq_len)")
    ax.set_ylabel("Decoder prediction step (pred_len)")
    ax.set_title(title)
    fig.tight_layout()

    if save_path:
        _save(fig, save_path)
    return fig


# ─────────────────────────── 7. variable importance ─────────────────────────

def plot_variable_importance(
    weights:    np.ndarray,
    var_names:  list,
    save_path:  str = None,
    title:      str = "Variable Selection Weights",
):
    """
    weights   : (n_vars,)  — from extract_variable_importance()
    var_names : list of str  matching the variable order
    """
    idx    = np.argsort(weights)[::-1]
    sorted_w = weights[idx]
    sorted_n = [var_names[i] for i in idx]

    fig, ax = plt.subplots(figsize=(max(6, len(var_names)), 4))
    ax.barh(sorted_n[::-1], sorted_w[::-1], color="#2ECC71", edgecolor="white")
    ax.set_xlabel("Mean Selection Weight")
    ax.set_title(title)
    fig.tight_layout()

    if save_path:
        _save(fig, save_path)
    return fig
