<div align="center">

# TFT Time-Series Forecasting

**Interpretable multivariate forecasting with a from-scratch Temporal Fusion Transformer.**

A PyTorch research project combining a custom TFT implementation, LSTM and Informer baselines, multi-dataset experiments, attention and variable-selection analysis, and a FastAPI + Streamlit application.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-TFT_from_Scratch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Inference_API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Verification](https://img.shields.io/badge/Evaluation_%26_API-Verified-2AA887)](#verification)

[Overview](#overview) · [Results](#results) · [Quick Start](#quick-start) · [Application](#application) · [Verification](#verification) · [Limitations](#limitations)

**[Final thesis (Chinese, PDF)](docs/thesis.pdf) · [Experiment tables](results/metrics/thesis_tables.csv) · [Third-party notices](THIRD_PARTY_NOTICES.md)**

</div>

---

<img src="docs/streamlit-dashboard.jpg" alt="TFT Streamlit dashboard with the FastAPI backend online" width="100%">

<a id="overview"></a>
## 1. Overview

This project explores interpretable multi-step forecasting on multivariate time-series data. It was developed as Cao Junjia's undergraduate graduation project and preserves both the research code and the application used to demonstrate it.

The TFT implementation includes GLU, GRN, Variable Selection Networks, an LSTM encoder–decoder, and interpretable multi-head attention. The decoder consumes both historical observations and known time features, while future observed values are zeroed to prevent leakage.

Beyond the model, the repository contains chronological data splitting, training and evaluation scripts, preserved experiment outputs, an inference API, and a dashboard for data exploration, forecasting, results, anomaly monitoring, and interpretability.

## 2. Model and Experiment Pipeline

```text
Public time-series CSV
  → Chronological split and train-only scaling
  → Encoder / decoder windows with masked future observations
  → TFT, LSTM, or Informer
  → MAE / MSE / RMSE / MAPE and ablation tables
  → Attention weights and variable-selection analysis
  → FastAPI inference + Streamlit visualisation
```

Experiments cover ETTh1, ETTh2, ETTm1, Weather, and ECL configurations. Input-window and prediction-horizon ablations are collected in [`results/metrics/thesis_tables.csv`](results/metrics/thesis_tables.csv).

<a id="results"></a>
## 3. Selected Results

ETTh1, `seq_len=96`, `pred_len=24`. These are the preserved thesis experiment results, not metrics from a new training run during repository cleanup.

| Model | MAE | MSE | RMSE | MAPE |
| --- | ---: | ---: | ---: | ---: |
| TFT | **0.1887** | **0.0589** | **0.2426** | 23.65% |
| LSTM | 0.2170 | 0.0840 | 0.2898 | **22.29%** |
| Informer | 0.2849 | 0.1302 | 0.3608 | 28.82% |

TFT achieves the lowest MAE, MSE, and RMSE on this run; LSTM has the lowest MAPE.

<img src="results/figures/fig21_window_ablation.png" alt="TFT input-window ablation" width="100%">

### Interpretability Example

<img src="results/figures/fig26_attention_heatmap.png" alt="TFT temporal attention heatmap" width="100%">

Additional prediction, error-analysis, anomaly-monitoring, and variable-importance figures are in [`results/figures/`](results/figures/).

<a id="quick-start"></a>
## 4. Quick Start

### Requirements

- Python 3.10+
- The packages in `requirements.txt`
- Public datasets downloaded separately as described in [`data/README.md`](data/README.md)

```bash
git clone https://github.com/Cjj040721/TFT-Time-Series-Forecasting.git
cd TFT-Time-Series-Forecasting
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the main experiment, place ETTh1 at:

```text
data/raw/ETT/ETTh1.csv
```

Train and evaluate:

```bash
python -m src.experiment.train --config configs/tft_etth1_96_24.yaml
python -m src.experiment.test --config configs/tft_etth1_96_24.yaml
```

The checkpoint is written to `results/checkpoints/tft_ETTh1_96_24/best_model.pt`. Model weights are not tracked, so a fresh clone must train one or supply an equivalent checkpoint.

To reproduce the full experiment matrix:

```bash
bash scripts/run_all_experiments.sh
```

The full matrix trains many models and can take substantial time and storage.

<a id="application"></a>
## 5. Run the Application

Start the API from the repository root:

```bash
uvicorn app.api.main:app --reload --port 8000
```

In another terminal:

```bash
streamlit run app/streamlit/app.py
```

Interactive API documentation is available at `http://localhost:8000/docs`.

| Route | Purpose |
| --- | --- |
| `GET /health` | Liveness check |
| `GET /models` | List configurations and checkpoint availability |
| `POST /predict` | Run inference on supplied arrays |
| `GET /results/{run_id}/metrics` | Read preserved metrics |
| `GET /results/{run_id}/attention` | Read preserved TFT attention weights |

<a id="verification"></a>
## 6. Verification

| Check | Result |
| --- | --- |
| Python syntax | `src`, `app`, and `scripts` compiled successfully |
| Dataset windows and leakage mask | Correct shapes; future `dec_x` values are zero |
| TFT forward pass | Prediction `(1, 24)`, attention `(1, 24, 96)`, 11 variable weights summing to 1 |
| Informer adapter | Forward output shape verified |
| Main experiment evaluation | MAE `0.18865186`, consistent with the preserved result within floating-point tolerance |
| Variable importance | Regression check passed after correcting the saved DataFrame schema |
| Real HTTP workflow | Health, model list, inference, nested metrics, and attention routes passed |
| Streamlit | Dashboard opened with **API Online**; README screenshot captured locally |

<a id="limitations"></a>
## 7. Limitations

- This is a thesis research implementation, not a production forecasting service.
- MAE, MSE, and RMSE follow the preserved configurations, including normalized targets where `inverse_scale: False`.
- MAPE can be unstable near zero, especially on ETTh2, Weather, and ECL.
- The preserved summary has no TFT result for ECL; it is left blank rather than backfilled.
- MAD residual monitoring is evaluated with synthetically injected anomalies, not ground-truth ETT anomaly labels.
- Attention and variable-selection weights are model-oriented signals, not causal explanations.

## 8. Repository Structure

```text
TFT-Time-Series-Forecasting/
├── app/api/                   # FastAPI routes and inference service
├── app/streamlit/             # Interactive dashboard
├── configs/                   # YAML experiment configurations
├── data/                      # Dataset download instructions
├── docs/                      # Final thesis, licence, and real dashboard screenshot
├── results/
│   ├── ETTh1/tft/             # Preserved main result bundle
│   ├── figures/               # Selected thesis figures
│   ├── metrics/               # Comparison and ablation tables
│   └── variable_importance/
├── scripts/                   # Experiment and figure pipelines
├── src/                       # Data, models, evaluation, and training code
├── third_party/informer2020/  # Upstream baseline and its licence
└── requirements.txt
```

## 9. Author and Licence

**Cao Junjia** — undergraduate graduation project, 2025–26.

Original source code and project documentation are available under the [MIT License](LICENSE). The final thesis is available separately under [CC BY 4.0](docs/LICENSE.md). Third-party code and datasets remain subject to their original terms; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

---

<div align="center">

**TFT Time-Series Forecasting — Implement the model. Measure the result. Inspect the attention.**

PyTorch + FastAPI + Streamlit

</div>
