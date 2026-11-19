"""
Author     : Cao Junjia

collect_results_tables.py — Aggregate all experiment metrics into thesis tables.

Reads metrics.json from every result directory and assembles:
  Table 10 — ETTh1 three-model comparison (seq=96, pred=24)
  Table 11 — ETTh2, ETTm1 three-model comparison (seq=96, pred=24)
  Table 13 — TFT window-length ablation   (pred=24)
  Table 14 — TFT prediction-horizon ablation (seq=96)

Output:
  results/metrics/thesis_tables.csv     — machine-readable summary
  results/metrics/thesis_tables.txt     — formatted ASCII tables for copy-paste
  results/metrics/thesis_tables_latex.tex — LaTeX tabular code

Usage:
  python scripts/collect_results_tables.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_ROOT = Path("results")
METRICS_DIR  = RESULTS_ROOT / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_metrics(dataset: str, model: str, seq_len: int, pred_len: int) -> dict | None:
    """
    Load metrics.json from the standard result directory layout:
        results/{dataset}/{model}/seq{seq}_pred{pred}_run1/metrics.json
    """
    run_dir = (RESULTS_ROOT / dataset / model /
               f"seq{seq_len}_pred{pred_len}_run1" / "metrics.json")
    if not run_dir.exists():
        return None
    with open(run_dir) as f:
        return json.load(f)


def fmt(val, decimals=4):
    """Format a float for display; return '—' if None."""
    if val is None:
        return "—"
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def row_str(label: str, metrics: dict | None, cols=("mae", "mse", "rmse", "mape")):
    """Format a single table row."""
    if metrics is None:
        values = ["—"] * len(cols)
    else:
        values = [fmt(metrics.get(c)) for c in cols]
    return f"  {label:<30s}  " + "  ".join(f"{v:>8s}" for v in values)


# ── Table builders ────────────────────────────────────────────────────────────

def build_table(title: str, rows: list[tuple[str, str, str, int, int]]) -> tuple[str, list[dict]]:
    """
    rows: list of (row_label, dataset, model, seq_len, pred_len)
    Returns (ascii_table, records_list)
    """
    HEADER = ("MAE", "MSE", "RMSE", "MAPE%")
    sep  = "─" * 70
    lines = [sep, f"  {title}", sep]
    lines.append(f"  {'Configuration':<30s}  " +
                 "  ".join(f"{h:>8s}" for h in HEADER))
    lines.append(sep)

    records = []
    for label, dataset, model, seq_len, pred_len in rows:
        m = load_metrics(dataset, model, seq_len, pred_len)
        lines.append(row_str(label, m))
        rec = {
            "table": title,
            "label": label,
            "dataset": dataset,
            "model": model,
            "seq_len": seq_len,
            "pred_len": pred_len,
        }
        if m:
            rec.update({k: m.get(k) for k in ("mae", "mse", "rmse", "mape")})
        else:
            rec.update({"mae": None, "mse": None, "rmse": None, "mape": None})
        records.append(rec)

    lines.append(sep)
    return "\n".join(lines), records


def build_latex_table(title: str, rows: list[tuple], caption: str = "") -> str:
    """Generate a LaTeX tabular block."""
    lines = [
        f"% {title}",
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption or title}}}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Configuration & MAE & MSE & RMSE & MAPE (\%) \\",
        r"\midrule",
    ]
    for label, dataset, model, seq_len, pred_len in rows:
        m = load_metrics(dataset, model, seq_len, pred_len)
        values = [fmt(m.get(k) if m else None) for k in ("mae", "mse", "rmse", "mape")]
        lines.append(f"{label} & {' & '.join(values)} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


# ── Table definitions ─────────────────────────────────────────────────────────

TABLE10_ROWS = [
    ("TFT (ours)",          "ETTh1", "tft",      96,  24),
    ("Informer",            "ETTh1", "informer",  96,  24),
    ("LSTM",                "ETTh1", "lstm",      96,  24),
]

TABLE11_ROWS = [
    # ETTh2
    ("TFT — ETTh2",         "ETTh2", "tft",      96,  24),
    ("Informer — ETTh2",    "ETTh2", "informer",  96,  24),
    ("LSTM — ETTh2",        "ETTh2", "lstm",      96,  24),
    # ETTm1
    ("TFT — ETTm1",         "ETTm1", "tft",      96,  24),
    ("Informer — ETTm1",    "ETTm1", "informer",  96,  24),
    ("LSTM — ETTm1",        "ETTm1", "lstm",      96,  24),
]

TABLE12_ROWS = [
    # ECL
    ("TFT — ECL",           "ECL",     "tft",      96,  24),
    ("Informer — ECL",      "ECL",     "informer",  96,  24),
    ("LSTM — ECL",          "ECL",     "lstm",      96,  24),
    # Weather
    ("TFT — Weather",       "Weather", "tft",      96,  24),
    ("Informer — Weather",  "Weather", "informer",  96,  24),
    ("LSTM — Weather",      "Weather", "lstm",      96,  24),
]

TABLE13_ROWS = [
    ("seq=96,  pred=24",    "ETTh1", "tft",       96,  24),
    ("seq=168, pred=24",    "ETTh1", "tft",      168,  24),
    ("seq=336, pred=24",    "ETTh1", "tft",      336,  24),
]

TABLE14_ROWS = [
    ("seq=96,  pred=24",    "ETTh1", "tft",       96,  24),
    ("seq=96,  pred=48",    "ETTh1", "tft",       96,  48),
    ("seq=96,  pred=96",    "ETTh1", "tft",       96,  96),
    ("seq=96,  pred=168",   "ETTh1", "tft",       96, 168),
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    all_records = []
    all_ascii   = []
    all_latex   = []

    # Table 10
    t, recs = build_table(
        "Table 10 — ETTh1 Model Comparison (seq=96, pred=24)", TABLE10_ROWS)
    all_ascii.append(t); all_records.extend(recs)
    all_latex.append(build_latex_table(
        "Table 10", TABLE10_ROWS,
        caption="Forecasting results on ETTh1 (seq=96, pred=24)"))

    # Table 11
    t, recs = build_table(
        "Table 11 — ETTh2 & ETTm1 Model Comparison (seq=96, pred=24)", TABLE11_ROWS)
    all_ascii.append(t); all_records.extend(recs)
    all_latex.append(build_latex_table(
        "Table 11", TABLE11_ROWS,
        caption="Forecasting results on ETTh2 and ETTm1 (seq=96, pred=24)"))

    # Table 12
    t, recs = build_table(
        "Table 12 — ECL & Weather Model Comparison (seq=96, pred=24)", TABLE12_ROWS)
    all_ascii.append(t); all_records.extend(recs)
    all_latex.append(build_latex_table(
        "Table 12", TABLE12_ROWS,
        caption="Forecasting results on ECL and Weather datasets (seq=96, pred=24)"))

    # Table 13
    t, recs = build_table(
        "Table 13 — TFT Input Window Ablation (ETTh1, pred=24)", TABLE13_ROWS)
    all_ascii.append(t); all_records.extend(recs)
    all_latex.append(build_latex_table(
        "Table 13", TABLE13_ROWS,
        caption="TFT forecasting results under different input window lengths (ETTh1, pred=24)"))

    # Table 14
    t, recs = build_table(
        "Table 14 — TFT Prediction Horizon Ablation (ETTh1, seq=96)", TABLE14_ROWS)
    all_ascii.append(t); all_records.extend(recs)
    all_latex.append(build_latex_table(
        "Table 14", TABLE14_ROWS,
        caption="TFT forecasting results under different prediction horizons (ETTh1, seq=96)"))

    # ── Write ASCII ──────────────────────────────────────────────────────────
    txt_path = METRICS_DIR / "thesis_tables.txt"
    with open(txt_path, "w") as f:
        f.write("\n\n".join(all_ascii) + "\n")
    print(f"[collect] ASCII tables → {txt_path}")

    # ── Write CSV ────────────────────────────────────────────────────────────
    import csv
    csv_path = METRICS_DIR / "thesis_tables.csv"
    fieldnames = ["table", "label", "dataset", "model", "seq_len", "pred_len",
                  "mae", "mse", "rmse", "mape"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)
    print(f"[collect] CSV summary    → {csv_path}")

    # ── Write LaTeX ──────────────────────────────────────────────────────────
    tex_path = METRICS_DIR / "thesis_tables_latex.tex"
    with open(tex_path, "w") as f:
        f.write("% Auto-generated LaTeX tables\n")
        f.write("% \\usepackage{booktabs} required in preamble\n\n")
        f.write("\n\n".join(all_latex))
    print(f"[collect] LaTeX tables   → {tex_path}")

    # ── Print to stdout ──────────────────────────────────────────────────────
    print()
    for t in all_ascii:
        print(t)
        print()


if __name__ == "__main__":
    main()
