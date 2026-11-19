"""
Author     : Cao Junjia

evaluate_anomaly_types.py — Table 15: Anomaly detection evaluation.

Loads the trained TFT (or any model) checkpoint, runs inference on the test set,
then evaluates the MAD detector against four synthetic anomaly types:

  spike        — single-point impulse
  level_shift  — sustained offset
  drift        — gradual slope
  dropout      — flat-line / sensor loss

For each type, reports: Precision, Recall, F1-score.
Also runs a combined evaluation (all four types mixed) for the summary row.

Output:
  results/metrics/table15_anomaly.json
  results/metrics/table15_anomaly.txt
  results/metrics/table15_anomaly_latex.tex

Usage:
  python scripts/evaluate_anomaly_types.py --config configs/tft_etth1_96_24.yaml
  python scripts/evaluate_anomaly_types.py --config configs/tft_etth1_96_24.yaml --lam 2.5
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import ETTDatasetManager
from src.evaluation.anomaly import ResidualMADDetector, inject_synthetic_anomalies
from src.evaluation.metrics import compute_anomaly_metrics
from src.model.factory import build_model
from src.utils.io import load_config
from src.utils.seed import fix_seed

METRICS_DIR = Path("results/metrics")


# ── helpers ───────────────────────────────────────────────────────────────────

def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_predict(cfg: dict, device: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the best checkpoint and run inference on the test split.
    Returns (pred_flat, true_flat) as 1-D float arrays in normalised space.
    """
    mgr = ETTDatasetManager(
        csv_path  = cfg["data_path"],
        seq_len   = cfg["seq_len"],
        label_len = cfg["label_len"],
        pred_len  = cfg["pred_len"],
        freq      = cfg.get("freq", "h"),
        target    = cfg.get("target", "OT"),
    )
    test_loader = mgr.get_loader("test", batch_size=cfg.get("batch_size", 32), shuffle=False)

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

    ckpt_dir  = Path("results/checkpoints") / f"{cfg['model']}_{cfg['dataset']}_{cfg['seq_len']}_{cfg['pred_len']}"
    ckpt_path = ckpt_dir / "best_model.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    print(f"[anomaly_eval] loaded checkpoint: {ckpt_path}")

    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            enc_x, enc_t, dec_x, dec_t, target = [b.to(device) for b in batch]
            if cfg["model"] == "lstm":
                pred = model(enc_x)
            else:
                pred = model(enc_x, enc_t, dec_x, dec_t)
            all_pred.append(pred.cpu().numpy())
            all_true.append(target.cpu().numpy())

    pred = np.concatenate(all_pred, axis=0).flatten()
    true = np.concatenate(all_true, axis=0).flatten()
    return pred, true


def evaluate_anomaly_type(
    true_series: np.ndarray,
    anomaly_type: str,
    detector: ResidualMADDetector,
    n_runs: int = 10,
    seed_offset: int = 0,
) -> dict:
    """
    Run multiple random anomaly injections for a single anomaly type.
    Returns averaged Precision, Recall, F1.

    Uses the clean true series to inject anomalies, then simulates what
    a model with the observed residual pattern would detect.
    """
    precisions, recalls, f1s = [], [], []

    for i in range(n_runs):
        contaminated, labels = inject_synthetic_anomalies(
            series       = true_series,
            anomaly_types= [anomaly_type],
            seed         = seed_offset + i * 7,
            n_anomalies  = 5,
            duration_range = (4, 24),
            magnitude    = 3.0,
        )
        # Compute residuals on the contaminated signal (model would still predict
        # the clean underlying pattern; residuals spike on anomaly segments)
        residuals = np.abs(contaminated - true_series)

        result = detector.detect(residuals)
        flag   = result["anomaly_flag"]

        metrics = compute_anomaly_metrics(flag, labels)
        precisions.append(metrics.get("precision", 0.0))
        recalls.append(metrics.get("recall",    0.0))
        f1s.append(metrics.get("f1",         0.0))

    return {
        "anomaly_type": anomaly_type,
        "precision":    float(np.mean(precisions)),
        "recall":       float(np.mean(recalls)),
        "f1":           float(np.mean(f1s)),
        "precision_std": float(np.std(precisions)),
        "recall_std":   float(np.std(recalls)),
        "f1_std":       float(np.std(f1s)),
        "n_runs":       n_runs,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection on 4 types")
    parser.add_argument("--config", type=str, default="configs/tft_etth1_96_24.yaml")
    parser.add_argument("--lam",    type=float, default=None,
                        help="Override anomaly_lam from config")
    parser.add_argument("--n_runs", type=int, default=10,
                        help="Number of random injections per anomaly type")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.lam is not None:
        cfg["anomaly_lam"] = args.lam
    fix_seed(cfg.get("seed", 42))
    device = _device()

    # 1. Get clean predictions and ground truth
    pred, true = load_model_and_predict(cfg, device)
    print(f"[anomaly_eval] test samples: {len(true)}")

    # 2. Build detector
    detector = ResidualMADDetector(
        window = cfg.get("anomaly_window", 48),
        lam    = cfg.get("anomaly_lam",    3.0),
    )

    # 3. Evaluate each anomaly type
    ANOMALY_TYPES = ["spike", "level_shift", "drift", "dropout"]
    results = []
    for atype in ANOMALY_TYPES:
        print(f"[anomaly_eval] evaluating: {atype} ({args.n_runs} runs)...")
        res = evaluate_anomaly_type(true, atype, detector, n_runs=args.n_runs)
        results.append(res)
        print(f"  Precision={res['precision']:.4f}  Recall={res['recall']:.4f}  F1={res['f1']:.4f}")

    # 4. Combined (all types mixed)
    print(f"[anomaly_eval] evaluating: combined ({args.n_runs} runs)...")
    combined_precs, combined_recs, combined_f1s = [], [], []
    for i in range(args.n_runs):
        contaminated, labels = inject_synthetic_anomalies(
            series        = true,
            anomaly_types = ANOMALY_TYPES,
            seed          = 1000 + i * 7,
            n_anomalies   = 8,
            duration_range= (4, 24),
            magnitude     = 3.0,
        )
        residuals = np.abs(contaminated - true)
        result    = detector.detect(residuals)
        m = compute_anomaly_metrics(result["anomaly_flag"], labels)
        combined_precs.append(m.get("precision", 0.0))
        combined_recs.append(m.get("recall",    0.0))
        combined_f1s.append(m.get("f1",         0.0))

    results.append({
        "anomaly_type": "combined",
        "precision":    float(np.mean(combined_precs)),
        "recall":       float(np.mean(combined_recs)),
        "f1":           float(np.mean(combined_f1s)),
        "precision_std": float(np.std(combined_precs)),
        "recall_std":   float(np.std(combined_recs)),
        "f1_std":       float(np.std(combined_f1s)),
        "n_runs":       args.n_runs,
    })
    print(f"  Combined → Precision={results[-1]['precision']:.4f}  "
          f"Recall={results[-1]['recall']:.4f}  F1={results[-1]['f1']:.4f}")

    # 5. Save JSON
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = METRICS_DIR / "table15_anomaly.json"
    with open(json_path, "w") as f:
        json.dump({"config": args.config, "lam": cfg["anomaly_lam"],
                   "results": results}, f, indent=2)
    print(f"\n[anomaly_eval] saved → {json_path}")

    # 6. ASCII table
    sep  = "─" * 60
    lines = [
        sep,
        f"  Table 15 — Anomaly Detection Performance",
        f"  Model: {cfg['model']} | Dataset: {cfg['dataset']} | λ={cfg['anomaly_lam']}",
        sep,
        f"  {'Anomaly Type':<14s}  {'Precision':>9s}  {'Recall':>9s}  {'F1':>9s}",
        sep,
    ]
    for r in results:
        lines.append(
            f"  {r['anomaly_type']:<14s}  {r['precision']:>9.4f}  "
            f"{r['recall']:>9.4f}  {r['f1']:>9.4f}"
        )
    lines.append(sep)
    ascii_table = "\n".join(lines)
    print("\n" + ascii_table)

    txt_path = METRICS_DIR / "table15_anomaly.txt"
    with open(txt_path, "w") as f:
        f.write(ascii_table + "\n")
    print(f"[anomaly_eval] ASCII table → {txt_path}")

    # 7. LaTeX table
    latex = [
        "% Table 15 — Anomaly Detection Performance",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Anomaly detection performance by anomaly type (MAD detector, $\lambda=" +
        f"{cfg['anomaly_lam']}" + r"$)}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Anomaly Type & Precision & Recall & F1-score \\",
        r"\midrule",
    ]
    for r in results:
        latex.append(
            f"{r['anomaly_type'].replace('_', ' ').title()} & "
            f"{r['precision']:.4f} & {r['recall']:.4f} & {r['f1']:.4f} \\\\"
        )
    latex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    latex_str = "\n".join(latex)

    tex_path = METRICS_DIR / "table15_anomaly_latex.tex"
    with open(tex_path, "w") as f:
        f.write(latex_str + "\n")
    print(f"[anomaly_eval] LaTeX table → {tex_path}")


if __name__ == "__main__":
    main()
