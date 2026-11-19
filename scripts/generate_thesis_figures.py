"""
Author     : Cao Junjia

generate_thesis_figures.py — Generate all 9 thesis figures (Fig 18 – Fig 26).

Figures:
  Fig 18 — ETTh1 dataset sample (raw time series)
  Fig 19 — TFT prediction vs ground truth
  Fig 20 — Three-model prediction comparison on same test segment
  Fig 21 — TFT under different input window lengths (mae bar chart)
  Fig 22 — 24-step prediction example (single forecast window)
  Fig 23 — Prediction error vs forecast horizon (line chart)
  Fig 24 — Anomaly detection case (residuals + threshold + anomaly marks)
  Fig 25 — Variable importance (VSN weights bar chart)
  Fig 26 — Temporal attention heatmap

All figures saved to: results/figures/

Usage:
  python scripts/generate_thesis_figures.py
  python scripts/generate_thesis_figures.py --fig 19 --fig 24  # specific figures only
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.anomaly import ResidualMADDetector, inject_synthetic_anomalies
from src.evaluation.plots import (
    plot_data_slice, plot_prediction_vs_truth, plot_residual_anomaly,
    plot_metric_comparison, plot_attention_heatmap, plot_variable_importance,
    plot_loss_curve,
)

RESULTS_ROOT = Path("results")
FIGURES_DIR  = RESULTS_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi":     150,
    "axes.grid":      True,
    "grid.alpha":     0.3,
})

COLORS = {
    "truth":     "#2C3E50",
    "tft":       "#E74C3C",
    "lstm":      "#3498DB",
    "informer":  "#2ECC71",
    "threshold": "#E67E22",
    "anomaly":   "#E74C3C",
    "residual":  "#3498DB",
}

MODEL_LABELS = {"tft": "TFT (ours)", "lstm": "LSTM", "informer": "Informer"}


# ── helpers ───────────────────────────────────────────────────────────────────

def _save(fig, name: str) -> Path:
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] saved → {path}")
    return path


def _load_pred_true(dataset: str, model: str, seq_len: int, pred_len: int) -> tuple:
    """Load pred.csv and true.csv from result directory."""
    base = RESULTS_ROOT / dataset / model / f"seq{seq_len}_pred{pred_len}_run1"
    pred_path = base / "pred.csv"
    true_path = base / "true.csv"
    if not pred_path.exists() or not true_path.exists():
        return None, None
    pred = pd.read_csv(pred_path, header=None).values
    true = pd.read_csv(true_path, header=None).values
    return pred, true


def _load_metrics(dataset: str, model: str, seq_len: int, pred_len: int) -> dict:
    path = RESULTS_ROOT / dataset / model / f"seq{seq_len}_pred{pred_len}_run1" / "metrics.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _load_loss_history(model: str, dataset: str, seq_len: int, pred_len: int) -> tuple:
    path = (RESULTS_ROOT / "checkpoints" /
            f"{model}_{dataset}_{seq_len}_{pred_len}" / "loss_history.json")
    if not path.exists():
        return None, None
    with open(path) as f:
        h = json.load(f)
    return h.get("train", h.get("train_loss", [])), h.get("val", h.get("val_loss", []))


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 18 — ETTh1 dataset sample
# ═══════════════════════════════════════════════════════════════════════════════

def fig18_dataset_sample():
    csv_path = Path("data/raw/ETT/ETTh1.csv")
    if not csv_path.exists():
        print("[figures] SKIP fig18: ETTh1.csv not found")
        return

    df = pd.read_csv(csv_path, parse_dates=["date"])
    columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    n_points = 720  # 30 days of hourly data

    fig, axes = plt.subplots(len(columns), 1,
                              figsize=(14, 2.2 * len(columns)), sharex=True)
    col_labels = {
        "HUFL": "High Useful Fan Load",
        "HULL": "High Useless Fan Load",
        "MUFL": "Mid Useful Fan Load",
        "MULL": "Mid Useless Fan Load",
        "LUFL": "Low Useful Fan Load",
        "LULL": "Low Useless Fan Load",
        "OT":   "Oil Temperature (target)",
    }
    color_map = ["#2980B9", "#3498DB", "#27AE60", "#2ECC71",
                 "#8E44AD", "#9B59B6", "#E74C3C"]

    for ax, col, color in zip(axes, columns, color_map):
        vals = df[col].values[:n_points]
        ax.plot(vals, linewidth=0.7, color=color)
        ax.set_ylabel(col_labels[col], fontsize=9)
        ax.tick_params(axis="y", labelsize=8)

    # X-axis: show dates every 7 days
    tick_positions = list(range(0, n_points, 168))
    tick_labels    = [df["date"].iloc[i].strftime("%Y-%m-%d") for i in tick_positions]
    axes[-1].set_xticks(tick_positions)
    axes[-1].set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=8)
    axes[-1].set_xlabel("Date")
    axes[0].set_title("Figure 18 — ETTh1 Dataset Sample (First 30 Days, All Variables)",
                       fontsize=12)
    fig.tight_layout()
    _save(fig, "fig18_dataset_sample.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 19 — TFT prediction vs ground truth
# ═══════════════════════════════════════════════════════════════════════════════

def fig19_tft_prediction():
    pred, true = _load_pred_true("ETTh1", "tft", 96, 24)
    if pred is None:
        print("[figures] SKIP fig19: TFT results not found")
        return

    metrics = _load_metrics("ETTh1", "tft", 96, 24)
    n_show  = min(300, len(pred.flatten()))

    fig, ax = plt.subplots(figsize=(13, 4))
    true_flat = true.flatten()[:n_show]
    pred_flat = pred.flatten()[:n_show]

    ax.plot(true_flat, label="Ground Truth", color=COLORS["truth"], linewidth=1.2)
    ax.plot(pred_flat, label="TFT Prediction", color=COLORS["tft"],
            linewidth=1.0, linestyle="--")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Normalised Oil Temperature")
    mae  = metrics.get("mae",  "N/A")
    rmse = metrics.get("rmse", "N/A")
    title_str = "Figure 19 — TFT Forecast vs Ground Truth (ETTh1, seq=96, pred=24)"
    if mae != "N/A":
        title_str += f"  [MAE={mae:.4f}, RMSE={rmse:.4f}]"
    ax.set_title(title_str, fontsize=11)
    ax.legend()
    fig.tight_layout()
    _save(fig, "fig19_tft_prediction.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 20 — Three-model comparison on same test segment
# ═══════════════════════════════════════════════════════════════════════════════

def fig20_model_comparison():
    models = ["tft", "informer", "lstm"]
    data   = {}
    true_ref = None

    for m in models:
        pred, true = _load_pred_true("ETTh1", m, 96, 24)
        if pred is not None:
            data[m] = pred
            if true_ref is None:
                true_ref = true

    if true_ref is None:
        print("[figures] SKIP fig20: no model results found")
        return

    n_show = 200
    true_flat = true_ref.flatten()[:n_show]

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(true_flat, label="Ground Truth", color=COLORS["truth"],
            linewidth=1.4, zorder=10)

    for m, pred in data.items():
        ax.plot(pred.flatten()[:n_show],
                label=MODEL_LABELS[m],
                color=COLORS[m],
                linewidth=1.0, linestyle="--", alpha=0.85)

    ax.set_xlabel("Time step")
    ax.set_ylabel("Normalised Oil Temperature")
    ax.set_title("Figure 20 — Prediction Comparison: TFT vs Informer vs LSTM\n"
                 "(ETTh1, seq=96, pred=24, test set segment)", fontsize=11)
    ax.legend()
    fig.tight_layout()
    _save(fig, "fig20_model_comparison.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 21 — TFT under different input window lengths (MAE bar chart)
# ═══════════════════════════════════════════════════════════════════════════════

def fig21_window_ablation():
    configs = [
        ("seq=96",  96,  24),
        ("seq=168", 168, 24),
        ("seq=336", 336, 24),
    ]
    mae_vals = {}
    mse_vals = {}

    for label, seq, pred in configs:
        m = _load_metrics("ETTh1", "tft", seq, pred)
        mae_vals[label] = m.get("mae")
        mse_vals[label] = m.get("mse")

    # Skip if all missing
    if all(v is None for v in mae_vals.values()):
        print("[figures] SKIP fig21: no window ablation results")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, metric, vals, color in zip(
        axes,
        ["MAE", "MSE"],
        [mae_vals, mse_vals],
        ["#3498DB", "#E74C3C"],
    ):
        labels = list(vals.keys())
        values = [v if v is not None else 0 for v in vals.values()]
        bars = ax.bar(labels, values, color=color, edgecolor="white", width=0.45)
        for bar, v in zip(bars, values):
            if v:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(v * 0.02, 0.001),
                        f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel(metric)
        ax.set_xlabel("Input Window Length")
        ax.set_title(f"{metric} by Input Window Length")

    fig.suptitle("Figure 21 — TFT Prediction Performance vs Input Window Length\n"
                 "(ETTh1, pred=24)", fontsize=11)
    fig.tight_layout()
    _save(fig, "fig21_window_ablation.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 22 — 24-step prediction example (single forecast window)
# ═══════════════════════════════════════════════════════════════════════════════

def fig22_single_window_prediction():
    pred, true = _load_pred_true("ETTh1", "tft", 96, 24)
    if pred is None:
        print("[figures] SKIP fig22: TFT results not found")
        return

    # Pick a representative window (sample index 50, looks "interesting")
    sample_idx = 50
    if len(pred) <= sample_idx:
        sample_idx = len(pred) // 2

    pred_window = pred[sample_idx]   # (pred_len,)
    true_window = true[sample_idx]   # (pred_len,)
    pred_len    = len(pred_window)

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(pred_len)
    ax.plot(x, true_window, "o-", label="Ground Truth", color=COLORS["truth"],
            linewidth=1.5, markersize=5)
    ax.plot(x, pred_window, "s--", label="TFT Prediction", color=COLORS["tft"],
            linewidth=1.3, markersize=4)

    # Fill error band
    ax.fill_between(x, true_window, pred_window,
                    alpha=0.12, color=COLORS["tft"], label="Prediction error")

    ax.set_xlabel("Forecast step (hours ahead)")
    ax.set_ylabel("Normalised Oil Temperature")
    ax.set_xticks(x)
    ax.set_xticklabels([f"+{i+1}h" for i in x], fontsize=8)
    ax.set_title(f"Figure 22 — Single 24-Step Forecast Window\n"
                 f"(TFT on ETTh1, sample #{sample_idx})", fontsize=11)
    ax.legend()
    fig.tight_layout()
    _save(fig, "fig22_single_window_prediction.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 23 — Error vs forecast horizon (MAE by pred_len)
# ═══════════════════════════════════════════════════════════════════════════════

def fig23_error_vs_horizon():
    horizons = [24, 48, 96, 168]
    mae_tft = []
    mse_tft = []

    for pred_len in horizons:
        m = _load_metrics("ETTh1", "tft", 96, pred_len)
        mae_tft.append(m.get("mae") if m else None)
        mse_tft.append(m.get("mse") if m else None)

    if all(v is None for v in mae_tft):
        print("[figures] SKIP fig23: no horizon ablation results")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, vals, ylabel, title in zip(
        axes,
        [mae_tft, mse_tft],
        ["MAE", "MSE"],
        ["MAE vs Prediction Horizon", "MSE vs Prediction Horizon"],
    ):
        valid_x = [h for h, v in zip(horizons, vals) if v is not None]
        valid_y = [v for v in vals if v is not None]
        ax.plot(valid_x, valid_y, "o-", color=COLORS["tft"],
                linewidth=2, markersize=7, label="TFT")
        ax.set_xlabel("Prediction Horizon (hours)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(valid_x)
        ax.set_title(title)
        ax.legend()

    fig.suptitle("Figure 23 — Forecasting Error vs Prediction Horizon\n"
                 "(TFT on ETTh1, seq=96)", fontsize=11)
    fig.tight_layout()
    _save(fig, "fig23_error_vs_horizon.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 24 — Anomaly detection case (residuals + threshold + marks)
# ═══════════════════════════════════════════════════════════════════════════════

def fig24_anomaly_detection():
    pred, true = _load_pred_true("ETTh1", "tft", 96, 24)
    if pred is None:
        print("[figures] SKIP fig24: TFT results not found")
        return

    true_flat = true.flatten()
    pred_flat = pred.flatten()

    # Inject mixed synthetic anomalies for illustration
    contaminated, labels = inject_synthetic_anomalies(
        series        = true_flat,
        anomaly_types = ["spike", "level_shift", "drift", "dropout"],
        seed          = 42,
        n_anomalies   = 6,
        duration_range= (6, 20),
        magnitude     = 3.0,
    )
    residuals = np.abs(contaminated - pred_flat)

    detector = ResidualMADDetector(window=48, lam=3.0)
    det_result = detector.detect(residuals)

    n_show = 400
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    # Top: original signal with anomaly regions shaded
    ax = axes[0]
    ax.plot(true_flat[:n_show], label="Ground Truth", color=COLORS["truth"],
            linewidth=0.9)
    ax.plot(contaminated[:n_show], label="Contaminated Signal",
            color="#8E44AD", linewidth=0.8, linestyle="--", alpha=0.8)
    # Shade anomaly segments
    in_anomaly = False
    start_idx  = None
    for i, l in enumerate(labels[:n_show]):
        if l == 1 and not in_anomaly:
            in_anomaly = True; start_idx = i
        elif l == 0 and in_anomaly:
            ax.axvspan(start_idx, i, alpha=0.15, color="#8E44AD", linewidth=0)
            in_anomaly = False
    if in_anomaly:
        ax.axvspan(start_idx, n_show, alpha=0.15, color="#8E44AD", linewidth=0)
    ax.set_ylabel("Normalised OT")
    ax.set_title("Injected Anomalies (purple shading)")
    ax.legend(fontsize=9)

    # Bottom: residuals + threshold + detected points
    ax2 = axes[1]
    t   = np.arange(n_show)
    res = det_result["residuals"][:n_show]
    thr = det_result["threshold"][:n_show]
    flg = det_result["anomaly_flag"][:n_show]

    ax2.plot(t, res, label="Residual |e_t|",
             color=COLORS["residual"], linewidth=0.9)
    ax2.plot(t, thr, label="MAD Threshold τ_t",
             color=COLORS["threshold"], linewidth=1.2, linestyle="--")

    anom_idx = np.where(flg > 0)[0]
    ax2.scatter(t[anom_idx], res[anom_idx],
                color=COLORS["anomaly"], zorder=5, s=18,
                label="Detected anomaly")

    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Absolute Residual")
    ax2.set_title("Residual + MAD Threshold + Detected Anomalies")
    ax2.legend(fontsize=9)

    fig.suptitle("Figure 24 — Anomaly Detection Case Study\n"
                 "(TFT on ETTh1, MAD detector λ=3.0)", fontsize=12)
    fig.tight_layout()
    _save(fig, "fig24_anomaly_detection.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 25 — Variable importance (VSN weights)
# ═══════════════════════════════════════════════════════════════════════════════

def fig25_variable_importance():
    import pandas as pd

    weights   = None
    var_names = None

    # Priority 1: variable_importance.csv (saved by test.py)
    csv_path = RESULTS_ROOT / "variable_importance" / "variable_importance.csv"
    if csv_path.exists():
        vi = pd.read_csv(csv_path)
        hist = vi[vi["group"] == "historical"].copy()
        name_map = {f"hist_var_{i}": n
                    for i, n in enumerate(["HUFL","HULL","MUFL","MULL","LUFL","LULL","OT"])}
        hist["label"] = hist["variable"].map(name_map).fillna(hist["variable"])
        weights   = hist["importance"].values
        var_names = hist["label"].tolist()
        print(f"[figures] loaded var importance from {csv_path}")

    # Priority 2: legacy .npy files
    if weights is None:
        for p in [
            RESULTS_ROOT / "variable_importance" / "ETTh1_tft_96_24_var_weights.npy",
            RESULTS_ROOT / "ETTh1" / "tft" / "seq96_pred24_run1" / "var_weights.npy",
        ]:
            if p.exists():
                weights = np.load(p)
                print(f"[figures] loaded var weights from {p}")
                break

    if weights is None:
        print("[figures] SKIP fig25: variable importance not found — run test first")
        _placeholder_figure("fig25_variable_importance.pdf",
                            "Figure 25 — Variable Selection Weights\n"
                            "(Run evaluation to generate this figure)")
        return

    weights = np.array(weights).flatten()
    if var_names is None:
        feat = ["HUFL","HULL","MUFL","MULL","LUFL","LULL","OT"]
        var_names = feat[:len(weights)] if len(weights) <= 7 else [f"Var {i}" for i in range(len(weights))]

    idx      = np.argsort(weights)[::-1]
    sorted_w = weights[idx]
    sorted_n = [var_names[i] for i in idx]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = plt.cm.RdYlGn(np.linspace(0.35, 0.85, len(sorted_n)))[::-1]
    bars = ax.barh(sorted_n[::-1], sorted_w[::-1], color=colors[::-1], edgecolor="white")

    for bar, val in zip(bars, sorted_w[::-1]):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    ax.set_xlabel("Mean Variable Selection Weight")
    ax.set_title("Figure 25 — Variable Importance (TFT VSN Weights)\n"
                 "(ETTh1, seq=96, pred=24)", fontsize=11)
    ax.set_xlim(0, sorted_w.max() * 1.15)
    fig.tight_layout()
    _save(fig, "fig25_variable_importance.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 26 — Temporal attention heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def fig26_attention_heatmap():
    att_path = RESULTS_ROOT / "ETTh1" / "tft" / "seq96_pred24_run1" / "attention.npy"

    if not att_path.exists():
        print("[figures] SKIP fig26: attention.npy not found — run test first")
        _placeholder_figure("fig26_attention_heatmap.pdf",
                            "Figure 26 — Temporal Attention Heatmap\n"
                            "(Run evaluation to generate this figure)")
        return

    attn = np.load(att_path)  # shape: (pred_len, seq_len) or (n_heads, pred_len, seq_len)
    if attn.ndim == 3:
        attn = attn.mean(axis=0)  # average over heads

    pred_len, seq_len = attn.shape

    fig, ax = plt.subplots(figsize=(13, max(3, pred_len * 0.35)))
    im = ax.imshow(attn, aspect="auto", cmap="YlOrRd", origin="upper",
                   vmin=0, vmax=attn.max())
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Attention Weight")

    # Y-axis: forecast steps
    if pred_len <= 24:
        yticks = np.arange(pred_len)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"+{i+1}h" for i in yticks], fontsize=8)

    # X-axis: encoder steps (last seq_len points before forecast)
    step = max(1, seq_len // 12)
    xticks = np.arange(0, seq_len, step)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"-{seq_len - i}h" for i in xticks], fontsize=8, rotation=45)

    ax.set_xlabel("Encoder Time Step (hours before forecast)")
    ax.set_ylabel("Forecast Step")
    ax.set_title("Figure 26 — Temporal Self-Attention Weights (TFT)\n"
                 "(ETTh1, seq=96, pred=24)", fontsize=11)
    fig.tight_layout()
    _save(fig, "fig26_attention_heatmap.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Utility: placeholder figure
# ═══════════════════════════════════════════════════════════════════════════════

def _placeholder_figure(filename: str, text: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=12,
            bbox=dict(boxstyle="round", facecolor="#FDEBD0", alpha=0.7))
    ax.set_axis_off()
    _save(fig, filename)


# ═══════════════════════════════════════════════════════════════════════════════
# Bonus: Training loss curves (not numbered but often useful in thesis)
# ═══════════════════════════════════════════════════════════════════════════════

def bonus_loss_curves():
    models = [
        ("tft",      "ETTh1", 96,  24, "TFT"),
        ("lstm",     "ETTh1", 96,  24, "LSTM"),
        ("informer", "ETTh1", 96,  24, "Informer"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    found_any = False
    for ax, (model, dataset, seq, pred, label) in zip(axes, models):
        train_loss, val_loss = _load_loss_history(model, dataset, seq, pred)
        if train_loss:
            found_any = True
            ax.plot(train_loss, label="Train MSE", color="#2980B9")
            ax.plot(val_loss,   label="Val MSE",   color="#E67E22")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MSE")
            ax.set_title(f"{label} Training Curve")
            ax.legend()
        else:
            ax.text(0.5, 0.5, f"No loss history\n({label})",
                    transform=ax.transAxes, ha="center", va="center", fontsize=10)
            ax.set_axis_off()

    if found_any:
        fig.suptitle("Training & Validation Loss Curves (ETTh1, seq=96, pred=24)", fontsize=12)
        fig.tight_layout()
        _save(fig, "fig_bonus_loss_curves.pdf")
    else:
        plt.close(fig)
        print("[figures] SKIP bonus loss curves: no loss history found")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

FIGURE_REGISTRY = {
    18: ("fig18_dataset_sample",      fig18_dataset_sample),
    19: ("fig19_tft_prediction",      fig19_tft_prediction),
    20: ("fig20_model_comparison",    fig20_model_comparison),
    21: ("fig21_window_ablation",     fig21_window_ablation),
    22: ("fig22_single_window_prediction", fig22_single_window_prediction),
    23: ("fig23_error_vs_horizon",    fig23_error_vs_horizon),
    24: ("fig24_anomaly_detection",   fig24_anomaly_detection),
    25: ("fig25_variable_importance", fig25_variable_importance),
    26: ("fig26_attention_heatmap",   fig26_attention_heatmap),
    0:  ("bonus_loss_curves",         bonus_loss_curves),
}


def main():
    parser = argparse.ArgumentParser(description="Generate thesis figures 18-26")
    parser.add_argument("--fig", type=int, nargs="*", default=None,
                        help="Figure numbers to generate (default: all)")
    args = parser.parse_args()

    to_run = sorted(FIGURE_REGISTRY.keys()) if args.fig is None else args.fig

    print(f"\n[figures] Output directory: {FIGURES_DIR.resolve()}")
    print(f"[figures] Generating figures: {to_run}\n")

    for num in to_run:
        if num not in FIGURE_REGISTRY:
            print(f"[figures] Unknown figure number: {num}")
            continue
        name, func = FIGURE_REGISTRY[num]
        print(f"[figures] --- Figure {num}: {name} ---")
        try:
            func()
        except Exception as e:
            print(f"[figures] ERROR in figure {num}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n[figures] Done. Check: {FIGURES_DIR.resolve()}")


if __name__ == "__main__":
    main()
