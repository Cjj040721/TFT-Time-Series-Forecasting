"""
Author     : Cao Junjia

FastAPI backend for the thesis forecasting system.

## Endpoints

GET  /health                    — liveness probe
POST /predict                   — run inference, return forecast + anomaly flags
GET  /models                    — list available trained models
GET  /results/{run_id}/metrics  — load saved metrics JSON
GET  /results/{run_id}/attention — load attention weights (TFT only)

## Run

cd THESIS/
uvicorn app.api.main:app --reload --port 8000
"""
import os
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.api.schemas import PredictRequest, PredictResponse, MetricsResponse
from app.api.services import ForecastService

# ── app ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="TFT Time-Series Forecasting API",
    description="TFT / LSTM / Informer forecasting + MAD anomaly detection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── lazy service cache ────────────────────────────────────────────────────────

_services: dict = {}


def _get_service(config_path: str) -> ForecastService:
    if config_path not in _services:
        _services[config_path] = ForecastService(config_path)
    return _services[config_path]


# ── routes ────────────────────────────────────────────────────────────────────

_RESULTS_ROOT = Path("results").resolve()


def _result_path(run_id: str, filename: str) -> Path:
    path = (_RESULTS_ROOT / run_id / filename).resolve()
    if not path.is_relative_to(_RESULTS_ROOT):
        raise HTTPException(status_code=400, detail="Invalid result run ID.")
    return path


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def list_models():
    """List all YAML configs that have a matching trained checkpoint."""
    configs_dir = "configs"
    available = []
    if not os.path.isdir(configs_dir):
        return {"models": []}

    for fn in sorted(os.listdir(configs_dir)):
        if not fn.endswith(".yaml"):
            continue
        cfg_path = os.path.join(configs_dir, fn)
        try:
            from src.utils.io import load_config
            cfg = load_config(cfg_path)
            ckpt = os.path.join(
                "results/checkpoints",
                f"{cfg['model']}_{cfg['dataset']}_{cfg['seq_len']}_{cfg['pred_len']}",
                "best_model.pt",
            )
            available.append({
                "config":    fn,
                "model":     cfg.get("model"),
                "dataset":   cfg.get("dataset"),
                "seq_len":   cfg.get("seq_len"),
                "label_len": cfg.get("label_len", 48),
                "pred_len":  cfg.get("pred_len"),
                "trained":   os.path.exists(ckpt),
                "data_path": cfg.get("data_path", ""),
                "target":    cfg.get("target", "OT"),
                "freq":      cfg.get("freq", "h"),
            })
        except Exception:
            pass
    return {"models": available}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Run inference on the supplied input arrays.

    All arrays are passed as nested Python lists (JSON-serialisable).
    Shape requirements:
        enc_x  : (seq_len, n_features)
        enc_t  : (seq_len, n_time_features)
        dec_x  : (label_len + pred_len, n_features)
        dec_t  : (label_len + pred_len, n_time_features)
    """
    try:
        svc = _get_service(req.config_path)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Could not load model: {e}")

    try:
        result = svc.forecast(
            enc_x=np.array(req.enc_x, dtype=np.float32),
            enc_t=np.array(req.enc_t, dtype=np.float32),
            dec_x=np.array(req.dec_x, dtype=np.float32),
            dec_t=np.array(req.dec_t, dtype=np.float32),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    return PredictResponse(
        prediction=result["prediction"].tolist(),
        anomaly_flag=result["anomaly_flag"].tolist(),
        threshold=result["threshold"].tolist(),
        residual=result["residual"].tolist(),
        attention=result.get("attention"),        # None for non-TFT
        var_weights=result.get("var_weights"),
    )


@app.get("/results/{run_id:path}/metrics", response_model=MetricsResponse)
def get_metrics(run_id: str):
    """
    run_id example: ETTh1/tft/seq96_pred24_run1
    """
    path = _result_path(run_id, "metrics.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Metrics not found: {path}")
    from src.utils.io import load_metrics
    m = load_metrics(path)
    return MetricsResponse(run_id=run_id, metrics=m)


@app.get("/results/{run_id:path}/attention")
def get_attention(run_id: str):
    path = _result_path(run_id, "attention.npy")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Attention not found (TFT only).")
    attn = np.load(path)
    return {"run_id": run_id, "attention": attn.tolist()}
