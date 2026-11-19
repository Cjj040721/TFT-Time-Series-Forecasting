"""
Author     : Cao Junjia

Ablation / grid-search script.
Sweeps over (model, seq_len, pred_len) combinations, calls train+test,
collects results into a summary CSV.

Usage
-----
python -m src.experiment.ablation --base_config configs/tft_etth1_base.yaml
"""
import argparse
import copy
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.experiment.train import train
from src.experiment.test  import test
from src.utils.io import load_config
from src.utils.logger import get_logger


# ── default ablation grid ─────────────────────────────────────────────────────
WINDOW_COMBINATIONS = [
    (96,  48, 24),
    (168, 48, 48),
    (336, 48, 168),
]

MODELS = ["tft", "lstm", "informer"]


def run_ablation(base_cfg: dict, models: list = None, combos: list = None) -> pd.DataFrame:
    logger = get_logger("ablation")
    if models is None:
        models = [base_cfg["model"]]
    if combos is None:
        combos = WINDOW_COMBINATIONS

    records = []

    for model_name in models:
        for seq_len, label_len, pred_len in combos:
            cfg = copy.deepcopy(base_cfg)
            cfg["model"]     = model_name
            cfg["seq_len"]   = seq_len
            cfg["label_len"] = label_len
            cfg["pred_len"]  = pred_len

            run_id = f"{model_name}_{cfg.get('dataset','ETTh1')}_{seq_len}_{pred_len}"
            logger.info(f"▶ Running {run_id}")

            try:
                train(cfg)
                metrics = test(cfg)
                records.append({
                    "model":    model_name,
                    "dataset":  cfg.get("dataset", "ETTh1"),
                    "seq_len":  seq_len,
                    "pred_len": pred_len,
                    **metrics,
                })
                logger.info(f"  {run_id}: {metrics}")
            except Exception as e:
                logger.error(f"  FAILED {run_id}: {e}")
                records.append({
                    "model":    model_name,
                    "dataset":  cfg.get("dataset", "ETTh1"),
                    "seq_len":  seq_len,
                    "pred_len": pred_len,
                    "error":    str(e),
                })

    df = pd.DataFrame(records)
    out_path = "results/metrics/ablation_summary.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Ablation results saved → {out_path}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", type=str, required=True)
    parser.add_argument("--models", nargs="+", default=None,
                        help="e.g. --models tft lstm")
    args = parser.parse_args()
    cfg = load_config(args.base_config)
    run_ablation(cfg, models=args.models)


if __name__ == "__main__":
    main()
