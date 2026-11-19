#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# run_all_experiments.sh — Master script to run every training + evaluation job
# required for the thesis.
#
# Usage:
#   bash scripts/run_all_experiments.sh [--skip-train] [--skip-test]
#
# Options:
#   --skip-train   Skip training (run evaluation only on existing checkpoints)
#   --skip-test    Skip evaluation (train only)
#
# Tables covered:
#   Table 10 — ETTh1  three-model comparison (96→24)
#   Table 11 — ETTh2, ETTm1 cross-dataset comparison (96→24)
#   Table 12 — ECL, Weather cross-dataset comparison (96→24)
#   Table 13 — TFT window-length ablation (96/168/336 → 24)
#   Table 14 — TFT prediction-horizon ablation (96 → 24/48/96/168)
#   Table 15 — Anomaly detection (run separately via evaluate_anomaly_types.py)
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

SKIP_TRAIN=false
SKIP_TEST=false

for arg in "$@"; do
  case $arg in
    --skip-train) SKIP_TRAIN=true ;;
    --skip-test)  SKIP_TEST=true  ;;
  esac
done

LOG_DIR="results/logs"
mkdir -p "$LOG_DIR"

run_experiment() {
  local config="$1"
  local label="$2"
  echo ""
  echo "══════════════════════════════════════════════════════════"
  echo "  $label"
  echo "  config: $config"
  echo "══════════════════════════════════════════════════════════"

  if [ "$SKIP_TRAIN" = false ]; then
    echo "[train] starting..."
    python -m src.experiment.train --config "$config" \
      2>&1 | tee "$LOG_DIR/train_$(basename "$config" .yaml).log"
    echo "[train] done."
  else
    echo "[train] skipped."
  fi

  if [ "$SKIP_TEST" = false ]; then
    echo "[test]  starting..."
    python -m src.experiment.test --config "$config" \
      2>&1 | tee "$LOG_DIR/test_$(basename "$config" .yaml).log"
    echo "[test]  done."
  else
    echo "[test] skipped."
  fi
}

# ── Table 10: ETTh1, all three models (96→24) ─────────────────────────────────
echo "=== TABLE 10: ETTh1 three-model comparison ==="
run_experiment "configs/tft_etth1_96_24.yaml"      "TFT  | ETTh1 | seq=96  pred=24"
run_experiment "configs/lstm_etth1_96_24.yaml"     "LSTM | ETTh1 | seq=96  pred=24"
run_experiment "configs/informer_etth1_96_24.yaml" "Informer | ETTh1 | seq=96  pred=24"

# ── Table 11: ETTh2, all three models (96→24) ─────────────────────────────────
echo ""
echo "=== TABLE 11a: ETTh2 three-model comparison ==="
run_experiment "configs/tft_etth2_96_24.yaml"      "TFT  | ETTh2 | seq=96  pred=24"
run_experiment "configs/lstm_etth2_96_24.yaml"     "LSTM | ETTh2 | seq=96  pred=24"
run_experiment "configs/informer_etth2_96_24.yaml" "Informer | ETTh2 | seq=96  pred=24"

# ── Table 11: ETTm1, all three models (96→24) ─────────────────────────────────
echo ""
echo "=== TABLE 11b: ETTm1 three-model comparison ==="
run_experiment "configs/tft_ettm1_96_24.yaml"      "TFT  | ETTm1 | seq=96  pred=24"
run_experiment "configs/lstm_ettm1_96_24.yaml"     "LSTM | ETTm1 | seq=96  pred=24"
run_experiment "configs/informer_ettm1_96_24.yaml" "Informer | ETTm1 | seq=96  pred=24"

# ── Table 13: TFT window-length ablation (pred=24 fixed) ──────────────────────
echo ""
echo "=== TABLE 13: TFT input window ablation ==="
# seq=96 already done in Table 10
run_experiment "configs/tft_etth1_168_24.yaml"     "TFT  | ETTh1 | seq=168 pred=24"
run_experiment "configs/tft_etth1_336_24.yaml"     "TFT  | ETTh1 | seq=336 pred=24"

# ── Table 14: TFT prediction-horizon ablation (seq=96 fixed) ──────────────────
echo ""
echo "=== TABLE 14: TFT prediction horizon ablation ==="
# pred=24 already done in Table 10
run_experiment "configs/tft_etth1_96_48.yaml"      "TFT  | ETTh1 | seq=96  pred=48"
run_experiment "configs/tft_etth1_96_96.yaml"      "TFT  | ETTh1 | seq=96  pred=96"
run_experiment "configs/tft_etth1_96_168.yaml"     "TFT  | ETTh1 | seq=96  pred=168"

# ── Bonus: full multi-horizon baselines (for comprehensive tables) ─────────────
echo ""
echo "=== BONUS: full horizon baselines ==="
run_experiment "configs/tft_etth1_168_48.yaml"     "TFT  | ETTh1 | seq=168 pred=48"
run_experiment "configs/tft_etth1_336_168.yaml"    "TFT  | ETTh1 | seq=336 pred=168"
run_experiment "configs/lstm_etth1_168_48.yaml"    "LSTM | ETTh1 | seq=168 pred=48"
run_experiment "configs/lstm_etth1_336_168.yaml"   "LSTM | ETTh1 | seq=336 pred=168"
run_experiment "configs/informer_etth1_168_48.yaml"  "Informer | ETTh1 | seq=168 pred=48"
run_experiment "configs/informer_etth1_336_168.yaml" "Informer | ETTh1 | seq=336 pred=168"

# ── Table 12: ECL, all three models (96→24) ───────────────────────────────────
echo ""
echo "=== TABLE 12a: ECL three-model comparison ==="
run_experiment "configs/tft_ecl_96_24.yaml"      "TFT  | ECL     | seq=96  pred=24"
run_experiment "configs/lstm_ecl_96_24.yaml"     "LSTM | ECL     | seq=96  pred=24"
run_experiment "configs/informer_ecl_96_24.yaml" "Informer | ECL | seq=96  pred=24"

# ── Table 12: Weather, all three models (96→24) ───────────────────────────────
echo ""
echo "=== TABLE 12b: Weather three-model comparison ==="
run_experiment "configs/tft_weather_96_24.yaml"      "TFT  | Weather  | seq=96  pred=24"
run_experiment "configs/lstm_weather_96_24.yaml"     "LSTM | Weather  | seq=96  pred=24"
run_experiment "configs/informer_weather_96_24.yaml" "Informer | Weather | seq=96 pred=24"

# ── Table 15: anomaly detection evaluation ────────────────────────────────────
echo ""
echo "=== TABLE 15: Anomaly detection evaluation ==="
python scripts/evaluate_anomaly_types.py \
  --config configs/tft_etth1_96_24.yaml \
  2>&1 | tee "$LOG_DIR/anomaly_eval.log"
echo "[anomaly] done."

# ── Collect all results into tables ───────────────────────────────────────────
echo ""
echo "=== Collecting results into summary tables ==="
python scripts/collect_results_tables.py \
  2>&1 | tee "$LOG_DIR/collect_results.log"
echo "[collect] done."

# ── Generate all thesis figures ───────────────────────────────────────────────
echo ""
echo "=== Generating thesis figures ==="
python scripts/generate_thesis_figures.py \
  2>&1 | tee "$LOG_DIR/figures.log"
echo "[figures] done."

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  ALL EXPERIMENTS COMPLETE"
echo "  Results:  results/"
echo "  Tables:   results/metrics/thesis_tables.csv"
echo "  Figures:  results/figures/"
echo "══════════════════════════════════════════════════════════"
