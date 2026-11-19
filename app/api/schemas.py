"""
Author     : Cao Junjia

Pydantic request / response schemas for the FastAPI backend.
"""
from typing import List, Optional
from pydantic import BaseModel


class PredictRequest(BaseModel):
    config_path: str = "configs/tft_etth1_96_24.yaml"
    # All arrays as nested lists (JSON-serialisable)
    enc_x: List[List[float]]    # (seq_len, n_features)
    enc_t: List[List[float]]    # (seq_len, n_time_features)
    dec_x: List[List[float]]    # (label_len+pred_len, n_features)
    dec_t: List[List[float]]    # (label_len+pred_len, n_time_features)


class PredictResponse(BaseModel):
    prediction:   List[float]
    anomaly_flag: List[int]
    threshold:    List[float]
    residual:     List[float]
    # TFT-only optional fields
    attention:    Optional[List[List[float]]] = None    # (pred_len, seq_len)
    var_weights:  Optional[List[float]] = None          # (n_vars,)


class MetricsResponse(BaseModel):
    run_id:  str
    metrics: dict
