"""
Author     : Cao Junjia

Test / evaluation script.
Loads the best checkpoint, runs on the test split, saves a full result bundle.

Usage
-----
python -m src.experiment.test --config configs/tft_etth1_96_24.yaml
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.dataset import ETTDatasetManager
from src.evaluation.anomaly import ResidualMADDetector
from src.evaluation.metrics import compute_forecast_metrics, compute_anomaly_metrics
from src.evaluation.explain import extract_attention_weights, extract_variable_importance
from src.model.factory import build_model
from src.utils.io import load_config, save_result_bundle
from src.utils.logger import get_logger
from src.utils.seed import fix_seed


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _forward(model, batch, device, model_name):
    enc_x, enc_t, dec_x, dec_t, target = [b.to(device) for b in batch]
    if model_name == "lstm":
        pred = model(enc_x)
    else:
        pred = model(enc_x, enc_t, dec_x, dec_t)
    return pred.cpu().numpy(), target.cpu().numpy()


def test(cfg: dict) -> dict:
    fix_seed(cfg.get("seed", 42))
    device = _device()

    logger = get_logger(
        f"test_{cfg['model']}_{cfg['dataset']}_{cfg['seq_len']}_{cfg['pred_len']}",
        log_dir="results/logs",
    )

    # ── data ──────────────────────────────────────────────────────────────────
    mgr = ETTDatasetManager(
        csv_path  = cfg["data_path"],
        seq_len   = cfg["seq_len"],
        label_len = cfg["label_len"],
        pred_len  = cfg["pred_len"],
        freq      = cfg.get("freq", "h"),
        target    = cfg.get("target", "OT"),
    )
    test_loader = mgr.get_loader("test", batch_size=cfg.get("batch_size", 32), shuffle=False)

    # ── model ─────────────────────────────────────────────────────────────────
    model = build_model(
        model_name      = cfg["model"],
        n_features      = mgr.n_features,
        n_time_features = mgr.n_time_features,
        seq_len         = cfg["seq_len"],
        label_len       = cfg["label_len"],
        pred_len        = cfg["pred_len"],
        freq            = cfg.get("freq", "h"),
        d_model         = cfg.get("d_model", 128),
        n_heads         = cfg.get("n_heads", 4),
        lstm_layers     = cfg.get("lstm_layers", 2),
        hidden_size     = cfg.get("hidden_size", 256),
        dropout         = cfg.get("dropout", 0.1),
    ).to(device)

    # ── load checkpoint ───────────────────────────────────────────────────────
    ckpt_dir = os.path.join(
        "results/checkpoints",
        f"{cfg['model']}_{cfg['dataset']}_{cfg['seq_len']}_{cfg['pred_len']}",
    )
    ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    logger.info(f"Loaded checkpoint: {ckpt_path}")

    # ── inference ─────────────────────────────────────────────────────────────
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            p, t = _forward(model, batch, device, cfg["model"])
            all_pred.append(p)
            all_true.append(t)

    pred = np.concatenate(all_pred, axis=0)   # (N, pred_len)
    true = np.concatenate(all_true, axis=0)

    # ── inverse-scale (optional) ──────────────────────────────────────────────
    if cfg.get("inverse_scale", True):
        scaler = mgr.scaler
        ti     = mgr.target_idx
        # pred / true are already in normalised space; inverse:
        pred_inv = pred * scaler.std_[0, ti] + scaler.mean_[0, ti]
        true_inv = true * scaler.std_[0, ti] + scaler.mean_[0, ti]
    else:
        pred_inv, true_inv = pred, true

    # ── metrics ───────────────────────────────────────────────────────────────
    metrics = compute_forecast_metrics(pred_inv.flatten(), true_inv.flatten())
    logger.info(f"Test results: {metrics}")

    # ── anomaly detection (residual + MAD) ────────────────────────────────────
    residuals = np.abs(pred_inv.flatten() - true_inv.flatten())
    detector  = ResidualMADDetector(
        window = cfg.get("anomaly_window", 48),
        lam    = cfg.get("anomaly_lam", 3.0),
    )
    anom_result = detector.detect(residuals)

    # ── interpretability (TFT only) ───────────────────────────────────────────
    attention = None
    if cfg["model"] == "tft":
        try:
            sample_batch = next(iter(test_loader))
            attention = extract_attention_weights(model, *sample_batch[:4], device=device)

            vi = extract_variable_importance(model, *sample_batch[:4], device=device)
            if vi is not None:
                import pandas as pd
                vi_dir = "results/variable_importance"
                os.makedirs(vi_dir, exist_ok=True)
                vi_df = pd.DataFrame({
                    "group": (
                        ["historical"] * mgr.n_features
                        + ["future"] * mgr.n_time_features
                    ),
                    "variable": (
                        [f"hist_var_{i}" for i in range(mgr.n_features)]
                        + [f"future_var_{i}" for i in range(mgr.n_time_features)]
                    ),
                    "importance": vi,
                })
                vi_df.to_csv(os.path.join(vi_dir, "variable_importance.csv"), index=False)
                logger.info("Variable importance saved.")
        except Exception as e:
            logger.warning(f"Could not extract interpretability: {e}")

    # ── save results ──────────────────────────────────────────────────────────
    out_dir = os.path.join(
        "results",
        cfg.get("dataset", "ETTh1"),
        cfg["model"],
        f"seq{cfg['seq_len']}_pred{cfg['pred_len']}_run1",
    )
    save_result_bundle(
        out_dir   = out_dir,
        pred      = pred_inv,
        true      = true_inv,
        metrics   = metrics,
        residual  = anom_result["residuals"].reshape(-1, 1),
        anomaly   = anom_result["anomaly_flag"].reshape(-1, 1),
        attention = attention,
        config    = cfg,
    )

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    test(cfg)


if __name__ == "__main__":
    main()
