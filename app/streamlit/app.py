"""
Author     : Cao Junjia

Streamlit dashboard for the thesis forecasting system.

Pages:
  1. 📊 Data Explorer
  2. 🔮 Live Forecast
  3. 🚨 Anomaly Monitor
  4. 📈 Results Viewer
  5. 🔍 Interpretability

Run:
  streamlit run app/streamlit/app.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Power Transformer Forecasting System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
[data-testid="stAppViewContainer"] {
    background: #0f1117;
}
[data-testid="stSidebar"] {
    background: #161b27;
    border-right: 1px solid #2d3447;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #1e2535;
    border: 1px solid #2d3447;
    border-radius: 10px;
    padding: 16px 20px;
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 700;
    color: #e8eaf6;
}
[data-testid="stMetricLabel"] {
    color: #90a4ae;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stMetricDelta"] svg { display: none; }

/* Section headers */
h1 { color: #e8eaf6 !important; font-weight: 700; letter-spacing: -0.5px; }
h2 { color: #cfd8dc !important; font-weight: 600; }
h3 { color: #b0bec5 !important; }

/* Info / warning / error boxes */
.stAlert { border-radius: 8px; }

/* Sidebar nav radio */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] label span,
[data-testid="stSidebar"] p {
    color: #b0bec5 !important;
    font-size: 0.9rem;
}

/* Divider */
hr { border-color: #2d3447 !important; }

/* Status badge */
.status-ok   { color: #66bb6a; font-weight: 600; }
.status-fail { color: #ef5350; font-weight: 600; }

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════════════════════════

FEAT_NAMES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
TIME_NAMES = ["month", "day_of_week", "day_of_month", "hour"]

PALETTE = {
    "truth":    "#4fc3f7",
    "pred":     "#ff7043",
    "residual": "#64b5f6",
    "thresh":   "#ffa726",
    "anomaly":  "#ef5350",
    "tft":      "#ab47bc",
    "informer": "#42a5f5",
    "lstm":     "#26a69a",
    "grid":     "rgba(255,255,255,0.07)",
}


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def _check_api() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _api_models():
    try:
        r = requests.get(f"{API_URL}/models", timeout=5)
        r.raise_for_status()
        return r.json()["models"]
    except Exception as e:
        st.warning(f"Cannot reach API: {e}")
        return []


def _api_predict(config_path, enc_x, enc_t, dec_x, dec_t):
    payload = {
        "config_path": config_path,
        "enc_x": enc_x.tolist(),
        "enc_t": enc_t.tolist(),
        "dec_x": dec_x.tolist(),
        "dec_t": dec_t.tolist(),
    }
    r = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def _plotly_layout(title="", xlabel="", ylabel=""):
    """Shared dark-theme layout for all Plotly figures."""
    import plotly.graph_objects as go
    return dict(
        title=dict(text=title, font=dict(color="#cfd8dc", size=15)),
        paper_bgcolor="#1e2535",
        plot_bgcolor="#1e2535",
        font=dict(color="#90a4ae", size=12),
        xaxis=dict(
            title=xlabel,
            gridcolor=PALETTE["grid"],
            zerolinecolor=PALETTE["grid"],
            showline=True, linecolor="#2d3447",
        ),
        yaxis=dict(
            title=ylabel,
            gridcolor=PALETTE["grid"],
            zerolinecolor=PALETTE["grid"],
            showline=True, linecolor="#2d3447",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="#2d3447",
            borderwidth=1,
        ),
        margin=dict(l=60, r=20, t=50, b=50),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Splash / Landing page
# ══════════════════════════════════════════════════════════════════════════════

if "entered" not in st.session_state:
    st.session_state.entered = False

if not st.session_state.entered:
    st.markdown("""
<style>
[data-testid="stSidebar"]          { display: none !important; }
header[data-testid="stHeader"]     { display: none !important; }
[data-testid="stToolbar"]          { display: none !important; }
.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}
/* ── CTA button ── */
div[data-testid="stButton"] > button {
    background: transparent !important;
    color: #4fc3f7 !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    padding: 13px 52px !important;
    border-radius: 50px !important;
    border: 1.5px solid rgba(79,195,247,0.6) !important;
    box-shadow: 0 0 30px rgba(79,195,247,0.15), inset 0 0 30px rgba(79,195,247,0.03) !important;
    letter-spacing: 0.22em !important;
    text-transform: uppercase !important;
    transition: all 0.3s ease !important;
}
div[data-testid="stButton"] > button:hover {
    background: rgba(79,195,247,0.08) !important;
    box-shadow: 0 0 60px rgba(79,195,247,0.35), inset 0 0 30px rgba(79,195,247,0.06) !important;
    border-color: #4fc3f7 !important;
    transform: translateY(-2px) !important;
}
</style>
""", unsafe_allow_html=True)

    # ── Hero HTML ──────────────────────────────────────────────────────────────
    st.markdown("""
<div style="
    background: radial-gradient(ellipse at 20% 20%, #0a1628 0%, #040c1f 40%, #020810 70%, #010610 100%);
    min-height: calc(100vh - 80px);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    position: relative;
    overflow: hidden;
    padding: 60px 24px 40px;
">

<style>
@keyframes gridDrift {
    0%   { background-position: 0 0; }
    100% { background-position: 80px 80px; }
}
@keyframes orb1Pulse {
    0%,100% { opacity:.18; transform:scale(1); }
    50%     { opacity:.35; transform:scale(1.08); }
}
@keyframes orb2Pulse {
    0%,100% { opacity:.12; transform:scale(1); }
    50%     { opacity:.25; transform:scale(1.06); }
}
@keyframes up1 { from{opacity:0;transform:translateY(50px)} to{opacity:1;transform:translateY(0)} }
@keyframes up2 { from{opacity:0;transform:translateY(40px)} to{opacity:1;transform:translateY(0)} }
@keyframes up3 { from{opacity:0;transform:translateY(30px)} to{opacity:1;transform:translateY(0)} }
@keyframes towerPulse {
    0%,100% { filter: drop-shadow(0 0 8px rgba(79,195,247,.45)); }
    50%     { filter: drop-shadow(0 0 22px rgba(79,195,247,.9)) drop-shadow(0 0 40px rgba(79,195,247,.3)); }
}
@keyframes arcFlicker {
    0%,100%{opacity:.55} 30%{opacity:.9} 60%{opacity:.35} 80%{opacity:.75}
}
@keyframes scanLine {
    0%   { top: -2px; opacity: .6; }
    100% { top: 100%; opacity: 0; }
}
.sp-grid {
    position:absolute; inset:0; pointer-events:none;
    background-image:
        linear-gradient(rgba(79,195,247,.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(79,195,247,.04) 1px, transparent 1px);
    background-size: 80px 80px;
    animation: gridDrift 20s linear infinite;
}
.sp-orb1 {
    position:absolute; width:800px; height:800px; border-radius:50%;
    background: radial-gradient(circle, rgba(13,71,161,.22) 0%, transparent 65%);
    top:-300px; left:-250px; pointer-events:none;
    animation: orb1Pulse 6s ease-in-out infinite;
}
.sp-orb2 {
    position:absolute; width:600px; height:600px; border-radius:50%;
    background: radial-gradient(circle, rgba(1,87,155,.18) 0%, transparent 65%);
    bottom:-200px; right:-150px; pointer-events:none;
    animation: orb2Pulse 8s ease-in-out infinite;
}
.sp-orb3 {
    position:absolute; width:350px; height:350px; border-radius:50%;
    background: radial-gradient(circle, rgba(74,20,140,.12) 0%, transparent 65%);
    top:35%; right:8%; pointer-events:none;
    animation: orb2Pulse 10s ease-in-out infinite reverse;
}
.sp-scan {
    position:absolute; left:0; right:0; height:2px;
    background: linear-gradient(90deg, transparent, rgba(79,195,247,.15), transparent);
    animation: scanLine 8s linear infinite; pointer-events:none;
}
.sp-tower {
    animation: up1 .9s ease .1s both, towerPulse 3.5s ease-in-out 1.2s infinite;
    margin-bottom: 28px;
    position: relative; z-index:2;
}
.sp-eyebrow {
    color:#4fc3f7; font-size:.7rem; letter-spacing:.4em;
    text-transform:uppercase; margin-bottom:14px; opacity:.75;
    animation: up2 .8s ease .35s both; position:relative; z-index:2;
}
.sp-title {
    font-size: clamp(1.9rem, 4.5vw, 3.4rem);
    font-weight: 800; line-height: 1.12;
    background: linear-gradient(130deg, #e8f4fd 0%, #90caf9 35%, #4fc3f7 65%, #29b6f6 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; text-align:center; letter-spacing:-.5px;
    margin-bottom:14px; max-width:720px;
    animation: up2 .8s ease .5s both; position:relative; z-index:2;
}
.sp-subtitle {
    color:#455a64; font-size:.93rem; text-align:center;
    max-width:540px; line-height:1.75; margin-bottom:40px;
    animation: up3 .8s ease .65s both; position:relative; z-index:2;
}
.sp-stats {
    display:flex; gap:14px; flex-wrap:wrap; justify-content:center;
    margin-bottom:30px;
    animation: up3 .8s ease .75s both; position:relative; z-index:2;
}
.sp-stat {
    background: rgba(255,255,255,.025);
    border: 1px solid rgba(79,195,247,.15);
    border-radius: 14px; padding:18px 28px; text-align:center;
    backdrop-filter: blur(12px); min-width:100px;
}
.sp-stat-val { font-size:1.65rem; font-weight:800; font-family:monospace; }
.sp-stat-lbl { font-size:.68rem; color:#37474f; text-transform:uppercase; letter-spacing:.1em; margin-top:5px; }
.sp-badges {
    display:flex; gap:8px; flex-wrap:wrap; justify-content:center;
    margin-bottom:50px;
    animation: up3 .8s ease .85s both; position:relative; z-index:2;
}
.sp-badge {
    background: rgba(79,195,247,.06);
    border: 1px solid rgba(79,195,247,.2);
    color: #78909c; padding:5px 15px; border-radius:20px;
    font-size:.76rem; font-weight:500;
}
.sp-btn-wrap {
    animation: up3 .8s ease .95s both; position:relative; z-index:2;
}
/* Decorative power lines (left & right edges) */
.sp-powerline-l, .sp-powerline-r {
    position:absolute; top:0; bottom:0; width:1px;
    background: linear-gradient(to bottom, transparent, rgba(79,195,247,.25) 30%, rgba(79,195,247,.25) 70%, transparent);
    pointer-events:none;
}
.sp-powerline-l { left: 60px; }
.sp-powerline-r { right: 60px; }
</style>

<div class="sp-grid"></div>
<div class="sp-orb1"></div>
<div class="sp-orb2"></div>
<div class="sp-orb3"></div>
<div class="sp-scan"></div>
<div class="sp-powerline-l"></div>
<div class="sp-powerline-r"></div>

<!-- High-Voltage Transmission Tower SVG -->
<div class="sp-tower">
<svg width="160" height="240" viewBox="0 0 160 240" fill="none" xmlns="http://www.w3.org/2000/svg">
  <!-- Base glow shadow -->
  <ellipse cx="80" cy="235" rx="44" ry="7" fill="#4fc3f7" opacity="0.12"/>
  <!-- Main legs -->
  <line x1="80" y1="18" x2="38" y2="220" stroke="#4fc3f7" stroke-width="2.5" stroke-linecap="round"/>
  <line x1="80" y1="18" x2="122" y2="220" stroke="#4fc3f7" stroke-width="2.5" stroke-linecap="round"/>
  <!-- Base feet -->
  <line x1="38"  y1="220" x2="14"  y2="238" stroke="#4fc3f7" stroke-width="2.5" stroke-linecap="round"/>
  <line x1="38"  y1="220" x2="50"  y2="238" stroke="#4fc3f7" stroke-width="2.5" stroke-linecap="round"/>
  <line x1="122" y1="220" x2="146" y2="238" stroke="#4fc3f7" stroke-width="2.5" stroke-linecap="round"/>
  <line x1="122" y1="220" x2="110" y2="238" stroke="#4fc3f7" stroke-width="2.5" stroke-linecap="round"/>
  <!-- Horizontal cross-bars -->
  <line x1="55"  y1="80"  x2="105" y2="80"  stroke="#4fc3f7" stroke-width="2"   stroke-linecap="round"/>
  <line x1="51"  y1="118" x2="109" y2="118" stroke="#4fc3f7" stroke-width="2"   stroke-linecap="round"/>
  <line x1="46"  y1="162" x2="114" y2="162" stroke="#4fc3f7" stroke-width="2"   stroke-linecap="round"/>
  <line x1="42"  y1="195" x2="118" y2="195" stroke="#4fc3f7" stroke-width="1.5" stroke-linecap="round"/>
  <!-- X-bracing -->
  <line x1="55"  y1="80"  x2="105" y2="118" stroke="#4fc3f7" stroke-width="1" opacity="0.35"/>
  <line x1="105" y1="80"  x2="55"  y2="118" stroke="#4fc3f7" stroke-width="1" opacity="0.35"/>
  <line x1="51"  y1="118" x2="109" y2="162" stroke="#4fc3f7" stroke-width="1" opacity="0.35"/>
  <line x1="109" y1="118" x2="51"  y2="162" stroke="#4fc3f7" stroke-width="1" opacity="0.35"/>
  <line x1="46"  y1="162" x2="114" y2="195" stroke="#4fc3f7" stroke-width="1" opacity="0.35"/>
  <line x1="114" y1="162" x2="46"  y2="195" stroke="#4fc3f7" stroke-width="1" opacity="0.35"/>
  <!-- Conductor arms — top -->
  <line x1="55"  y1="80"  x2="4"   y2="66"  stroke="#4fc3f7" stroke-width="2" stroke-linecap="round"/>
  <line x1="105" y1="80"  x2="156" y2="66"  stroke="#4fc3f7" stroke-width="2" stroke-linecap="round"/>
  <!-- Conductor arms — middle -->
  <line x1="51"  y1="118" x2="4"   y2="108" stroke="#4fc3f7" stroke-width="2" stroke-linecap="round"/>
  <line x1="109" y1="118" x2="156" y2="108" stroke="#4fc3f7" stroke-width="2" stroke-linecap="round"/>
  <!-- Insulators — top -->
  <circle cx="4"   cy="66"  r="4.5" fill="#4fc3f7" opacity=".9"/>
  <circle cx="156" cy="66"  r="4.5" fill="#4fc3f7" opacity=".9"/>
  <circle cx="4"   cy="66"  r="10"  fill="#4fc3f7" opacity=".1"/>
  <circle cx="156" cy="66"  r="10"  fill="#4fc3f7" opacity=".1"/>
  <!-- Insulators — middle -->
  <circle cx="4"   cy="108" r="4.5" fill="#4fc3f7" opacity=".9"/>
  <circle cx="156" cy="108" r="4.5" fill="#4fc3f7" opacity=".9"/>
  <circle cx="4"   cy="108" r="10"  fill="#4fc3f7" opacity=".1"/>
  <circle cx="156" cy="108" r="10"  fill="#4fc3f7" opacity=".1"/>
  <!-- Crown -->
  <circle cx="80" cy="18"  r="6"   fill="#4fc3f7"/>
  <circle cx="80" cy="18"  r="14"  fill="#4fc3f7" opacity=".18"/>
  <circle cx="80" cy="18"  r="22"  fill="#4fc3f7" opacity=".07"/>
  <!-- Animated conductor catenary curves -->
  <path d="M 4 66 Q 80 80 156 66"  stroke="#4fc3f7" stroke-width="1" fill="none"
        stroke-dasharray="6 5" opacity=".5"
        style="animation:arcFlicker 2.2s ease-in-out infinite"/>
  <path d="M 4 108 Q 80 122 156 108" stroke="#4fc3f7" stroke-width="1" fill="none"
        stroke-dasharray="6 5" opacity=".4"
        style="animation:arcFlicker 2.8s ease-in-out infinite .6s"/>
  <!-- Voltage sparks at crown -->
  <line x1="80" y1="4"  x2="74" y2="12" stroke="#e3f2fd" stroke-width="1.5" opacity=".8"
        style="animation:arcFlicker 1.5s ease-in-out infinite"/>
  <line x1="80" y1="4"  x2="86" y2="11" stroke="#e3f2fd" stroke-width="1.5" opacity=".8"
        style="animation:arcFlicker 1.5s ease-in-out infinite .4s"/>
</svg>
</div>

<div class="sp-eyebrow">⚡ &nbsp; Final Year Project &nbsp;·&nbsp; BJTU-LU &nbsp;·&nbsp; 2025–26</div>

<div class="sp-title">Power Transformer<br>Forecasting System</div>

<div class="sp-subtitle">
    An end-to-end platform for multivariate time-series forecasting of transformer
    oil temperature, with anomaly detection and interpretability analysis.
</div>

<div class="sp-stats">
    <div class="sp-stat">
        <div class="sp-stat-val" style="color:#4fc3f7">17,420</div>
        <div class="sp-stat-lbl">Data Records</div>
    </div>
    <div class="sp-stat">
        <div class="sp-stat-val" style="color:#ab47bc">3</div>
        <div class="sp-stat-lbl">Models</div>
    </div>
    <div class="sp-stat">
        <div class="sp-stat-val" style="color:#26a69a">7</div>
        <div class="sp-stat-lbl">Input Features</div>
    </div>
    <div class="sp-stat">
        <div class="sp-stat-val" style="color:#ff7043">MAD</div>
        <div class="sp-stat-lbl">Anomaly Det.</div>
    </div>
    <div class="sp-stat">
        <div class="sp-stat-val" style="color:#ffd54f">TFT</div>
        <div class="sp-stat-lbl">Primary Model</div>
    </div>
</div>

<div class="sp-badges">
    <span class="sp-badge">PyTorch</span>
    <span class="sp-badge">Temporal Fusion Transformer</span>
    <span class="sp-badge">Informer</span>
    <span class="sp-badge">LSTM</span>
    <span class="sp-badge">FastAPI</span>
    <span class="sp-badge">ETT Dataset</span>
</div>

<div class="sp-btn-wrap" id="sp-btn-anchor"></div>

</div>
""", unsafe_allow_html=True)

    # ── CTA Button ─────────────────────────────────────────────────────────────
    _, btn_col, _ = st.columns([2.2, 1, 2.2])
    with btn_col:
        if st.button("ENTER  SYSTEM  →", key="enter_btn"):
            st.session_state.entered = True
            st.rerun()

    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚡ FYP Dashboard")
    st.markdown("**Power Transformer Forecasting**")
    st.markdown("---")

    PAGES = [
        "🏠 Home",
        "📊 Data Explorer",
        "🔮 Live Forecast",
        "🚨 Anomaly Monitor",
        "📈 Results Viewer",
        "🔍 Interpretability",
    ]
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")

    st.markdown("---")

    # API health check
    api_ok = _check_api()
    status_cls  = "status-ok"   if api_ok else "status-fail"
    status_text = "● API Online" if api_ok else "● API Offline"
    st.markdown(f'<span class="{status_cls}">{status_text}</span>', unsafe_allow_html=True)
    if not api_ok:
        st.caption("Start with: `uvicorn app.api.main:app --reload --port 8000`")

    st.markdown("---")
    st.caption("Cao Junjia · BJTU-LU FYP 2025–26")


# ══════════════════════════════════════════════════════════════════════════════
#  Page 0 — Home
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Home":
    # ── hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
<div style="
    background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #01579b 100%);
    border-radius: 16px;
    padding: 48px 40px 40px 40px;
    margin-bottom: 32px;
    border: 1px solid #1565c0;
">
    <h1 style="color:#e3f2fd; font-size:2.2rem; margin:0 0 8px 0; font-weight:800;">
        ⚡ Power Transformer Forecasting System
    </h1>
    <p style="color:#90caf9; font-size:1.05rem; margin:0 0 20px 0;">
        Multivariate Time-Series Analysis &amp; Prediction based on ETT Dataset
    </p>
    <div style="display:flex; gap:12px; flex-wrap:wrap;">
        <span style="background:#1565c0; color:#e3f2fd; padding:4px 14px; border-radius:20px; font-size:0.82rem;">TFT · Temporal Fusion Transformer</span>
        <span style="background:#1565c0; color:#e3f2fd; padding:4px 14px; border-radius:20px; font-size:0.82rem;">LSTM Baseline</span>
        <span style="background:#1565c0; color:#e3f2fd; padding:4px 14px; border-radius:20px; font-size:0.82rem;">Informer Baseline</span>
        <span style="background:#1565c0; color:#e3f2fd; padding:4px 14px; border-radius:20px; font-size:0.82rem;">MAD Anomaly Detection</span>
        <span style="background:#1565c0; color:#e3f2fd; padding:4px 14px; border-radius:20px; font-size:0.82rem;">Interpretability Analysis</span>
    </div>
</div>
""", unsafe_allow_html=True)

    # ── overview cards ─────────────────────────────────────────────────────────
    st.markdown("### System Overview")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
<div style="background:#1e2535;border:1px solid #2d3447;border-radius:12px;padding:24px;min-height:260px;">
    <div style="font-size:2rem;margin-bottom:10px;">🎯</div>
    <div style="color:#4fc3f7;font-weight:700;font-size:1rem;margin-bottom:8px;">Research Objective</div>
    <div style="color:#b0bec5;font-size:0.88rem;line-height:1.6;">
        Predict oil temperature (OT) of power transformers using multivariate
        time-series data, enabling proactive maintenance and fault detection.
    </div>
</div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
<div style="background:#1e2535;border:1px solid #2d3447;border-radius:12px;padding:24px;min-height:260px;">
    <div style="font-size:2rem;margin-bottom:10px;">🧠</div>
    <div style="color:#ab47bc;font-weight:700;font-size:1rem;margin-bottom:8px;">Primary Model — TFT</div>
    <div style="color:#b0bec5;font-size:0.88rem;line-height:1.6;">
        Temporal Fusion Transformer integrates multi-scale attention,
        variable selection networks, and gated residual connections for
        interpretable multi-step forecasting.
    </div>
</div>""", unsafe_allow_html=True)

    with c3:
        st.markdown("""
<div style="background:#1e2535;border:1px solid #2d3447;border-radius:12px;padding:24px;min-height:260px;">
    <div style="font-size:2rem;margin-bottom:10px;">🚨</div>
    <div style="color:#ff7043;font-weight:700;font-size:1rem;margin-bottom:8px;">Anomaly Detection</div>
    <div style="color:#b0bec5;font-size:0.88rem;line-height:1.6;">
        Residual-based detection using MAD (Median Absolute Deviation)
        dynamic thresholding converts forecast errors into actionable
        operational alerts.
    </div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── pipeline diagram ───────────────────────────────────────────────────────
    st.markdown("### System Pipeline")
    st.markdown("""
<div style="background:#1e2535;border:1px solid #2d3447;border-radius:12px;padding:28px 32px;margin-bottom:24px;">
    <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
        <div style="text-align:center;flex:1;min-width:100px;">
            <div style="background:#0d47a1;border-radius:8px;padding:12px 8px;color:#e3f2fd;font-size:0.82rem;font-weight:600;">
                📂 ETT Dataset<br><span style="font-weight:400;color:#90caf9;">ETTh1/h2/m1</span>
            </div>
        </div>
        <div style="color:#4fc3f7;font-size:1.4rem;">→</div>
        <div style="text-align:center;flex:1;min-width:100px;">
            <div style="background:#1a237e;border-radius:8px;padding:12px 8px;color:#e3f2fd;font-size:0.82rem;font-weight:600;">
                ⚙️ Preprocessing<br><span style="font-weight:400;color:#90caf9;">Normalise · Split</span>
            </div>
        </div>
        <div style="color:#4fc3f7;font-size:1.4rem;">→</div>
        <div style="text-align:center;flex:1;min-width:100px;">
            <div style="background:#4a148c;border-radius:8px;padding:12px 8px;color:#e3f2fd;font-size:0.82rem;font-weight:600;">
                🧠 TFT Model<br><span style="font-weight:400;color:#ce93d8;">VSN · Attention</span>
            </div>
        </div>
        <div style="color:#4fc3f7;font-size:1.4rem;">→</div>
        <div style="text-align:center;flex:1;min-width:100px;">
            <div style="background:#bf360c;border-radius:8px;padding:12px 8px;color:#e3f2fd;font-size:0.82rem;font-weight:600;">
                🚨 Anomaly Det.<br><span style="font-weight:400;color:#ffab91;">MAD Threshold</span>
            </div>
        </div>
        <div style="color:#4fc3f7;font-size:1.4rem;">→</div>
        <div style="text-align:center;flex:1;min-width:100px;">
            <div style="background:#004d40;border-radius:8px;padding:12px 8px;color:#e3f2fd;font-size:0.82rem;font-weight:600;">
                📊 Dashboard<br><span style="font-weight:400;color:#80cbc4;">FastAPI · Streamlit</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    # ── dataset stats ──────────────────────────────────────────────────────────
    st.markdown("### Dataset at a Glance")
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Primary Dataset", "ETTh1")
    d2.metric("Sampling Rate",   "1 hour")
    d3.metric("Total Records",   "17,420")
    d4.metric("Input Features",  "7")
    d5.metric("Target Variable", "OT")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── navigation guide ───────────────────────────────────────────────────────
    st.markdown("### Navigate the System")
    n1, n2, n3, n4, n5 = st.columns(5)
    for col, icon, label, desc in [
        (n1, "📊", "Data Explorer",  "Browse ETT datasets and visualise raw feature time series."),
        (n2, "🔮", "Live Forecast",  "Run real-time inference via the FastAPI backend."),
        (n3, "🚨", "Anomaly Monitor","Inspect residuals and MAD-based anomaly flags."),
        (n4, "📈", "Results Viewer", "Compare MAE/MSE across models and training loss curves."),
        (n5, "🔍", "Interpretability","View variable importance and temporal attention heatmaps."),
    ]:
        col.markdown(f"""
<div style="background:#1e2535;border:1px solid #2d3447;border-radius:10px;padding:16px 12px;text-align:center;">
    <div style="font-size:1.6rem;">{icon}</div>
    <div style="color:#e8eaf6;font-weight:600;font-size:0.85rem;margin:6px 0 4px;">{label}</div>
    <div style="color:#78909c;font-size:0.78rem;line-height:1.4;">{desc}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Cao Junjia · BJTU-LU FYP 2025–26 · TFT-based Power Transformer Forecasting System")


# ══════════════════════════════════════════════════════════════════════════════
#  Page 1 — Data Explorer
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Data Explorer":
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    st.title("📊 Data Explorer")
    st.markdown("Browse and inspect the ETT dataset used for training and evaluation.")

    data_dir = "data/raw/ETT"
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")]) \
        if os.path.isdir(data_dir) else []

    if not csv_files:
        st.error("No CSV files found in `data/raw/ETT/`.")
        st.stop()

    chosen = st.selectbox("Select dataset", csv_files)
    df = load_csv(os.path.join(data_dir, chosen))
    feat_cols = [c for c in df.columns if c != "date"]

    # ── summary metrics ───────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total rows",   f"{len(df):,}")
    c2.metric("Features",     len(feat_cols))
    c3.metric("Start date",   str(df["date"].min().date()))
    c4.metric("End date",     str(df["date"].max().date()))

    st.markdown("---")

    # ── interactive time-series plot ──────────────────────────────────────────
    st.subheader("Feature Time Series")
    col_left, col_right = st.columns([3, 1])
    with col_right:
        selected_cols = st.multiselect("Variables", feat_cols,
                                        default=["OT", "HUFL", "MUFL"])
        n_points = st.slider("Time steps", 200, min(8000, len(df)), 1000, step=200)

    if selected_cols:
        with col_left:
            fig = make_subplots(rows=len(selected_cols), cols=1,
                                shared_xaxes=True, vertical_spacing=0.04)
            colors = ["#4fc3f7", "#ff7043", "#66bb6a", "#ffa726", "#ab47bc",
                      "#ef5350", "#42a5f5"]
            for i, col in enumerate(selected_cols):
                fig.add_trace(
                    go.Scatter(
                        x=df["date"].values[:n_points],
                        y=df[col].values[:n_points],
                        name=col,
                        line=dict(color=colors[i % len(colors)], width=1.2),
                        mode="lines",
                    ),
                    row=i + 1, col=1,
                )
                fig.update_yaxes(title_text=col, row=i + 1, col=1,
                                 gridcolor=PALETTE["grid"],
                                 title_font=dict(color="#90a4ae"),
                                 tickfont=dict(color="#90a4ae"))
                fig.update_xaxes(gridcolor=PALETTE["grid"],
                                 tickfont=dict(color="#90a4ae"))

            fig.update_layout(
                height=200 * len(selected_cols),
                paper_bgcolor="#1e2535",
                plot_bgcolor="#1e2535",
                font=dict(color="#90a4ae"),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=60, r=20, t=20, b=40),
                showlegend=True,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── statistics ────────────────────────────────────────────────────────────
    with st.expander("Descriptive Statistics"):
        st.dataframe(df[feat_cols].describe().round(4), use_container_width=True)

    with st.expander("Raw Data Preview"):
        st.dataframe(df.head(200), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Page 2 — Live Forecast
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔮 Live Forecast":
    import plotly.graph_objects as go

    st.title("🔮 Live Forecast")

    if not api_ok:
        st.error("API is offline. Please start the FastAPI backend first.")
        st.stop()

    models = _api_models()
    trained = [
        m for m in models
        if m.get("trained") and os.path.exists(m.get("data_path", ""))
    ]

    if not trained:
        st.warning("No trained models with available local data found.")
        st.stop()

    # ── mode selector ─────────────────────────────────────────────────────────
    mode = st.radio(
        "Forecast Mode",
        ["Single Model", "Multi-Model Comparison"],
        horizontal=True,
    )
    st.markdown("---")

    # ── helpers ───────────────────────────────────────────────────────────────
    def _build_inputs(df, feat_cols, seq_len, pred_len, label_len, start_idx, freq="h"):
        from src.data.features import time_features
        from src.data.preprocess import StandardScaler
        raw = df[feat_cols].values.astype(np.float32)
        scaler = StandardScaler()
        scaler.fit(raw[: int(len(df) * 0.6)])
        norm = scaler.transform(raw)
        enc_x = norm[start_idx: start_idx + seq_len]
        enc_t = time_features(df["date"].iloc[start_idx: start_idx + seq_len], freq=freq)
        ds    = start_idx + seq_len - label_len
        dec_x = norm[ds: ds + label_len + pred_len].copy()
        dec_x[label_len:] = 0.0
        dec_t = time_features(df["date"].iloc[ds: ds + label_len + pred_len], freq=freq)
        return raw, scaler, enc_x, enc_t, dec_x, dec_t

    def _inv_scale(pred_norm, scaler, feat_cols, target):
        target_idx = feat_cols.index(target)
        return pred_norm * scaler.std_[0, target_idx] + scaler.mean_[0, target_idx]

    # ══════════════════════════════════════════════════════════════════════════
    #  Single-model mode
    # ══════════════════════════════════════════════════════════════════════════
    if mode == "Single Model":
        st.markdown("Select a trained model and run real-time inference on the test set.")

        model_labels = [
            f"{m['model'].upper()} · {m['dataset']} · seq={m['seq_len']} pred={m['pred_len']}"
            for m in trained
        ]
        col_sel, col_info = st.columns([2, 1])
        with col_sel:
            chosen_idx = st.selectbox("Model", range(len(model_labels)),
                                       format_func=lambda i: model_labels[i])
        chosen_model = trained[chosen_idx]
        config_path = (
            f"configs/{chosen_model['model']}_{chosen_model['dataset'].lower()}"
            f"_{chosen_model['seq_len']}_{chosen_model['pred_len']}.yaml"
        )
        with col_info:
            st.info(f"Config: `{config_path}`")

        df        = load_csv(chosen_model["data_path"])
        feat_cols = [c for c in df.columns if c != "date"]
        seq_len   = chosen_model["seq_len"]
        pred_len  = chosen_model["pred_len"]
        label_len = chosen_model.get("label_len", 48)
        target    = chosen_model.get("target", "OT")
        freq      = chosen_model.get("freq", "h")
        max_start = len(df) - seq_len - pred_len - 1
        start_idx = st.slider("Window start (test region)",
                              int(len(df) * 0.8), max_start, int(len(df) * 0.8))

        if st.button("▶ Run Forecast", type="primary"):
            raw, scaler, enc_x, enc_t, dec_x, dec_t = _build_inputs(
                df, feat_cols, seq_len, pred_len, label_len, start_idx, freq=freq)
            with st.spinner("Running inference..."):
                try:
                    result = _api_predict(config_path, enc_x, enc_t, dec_x, dec_t)
                except Exception as e:
                    st.error(f"API error: {e}")
                    st.stop()

            pred = _inv_scale(np.array(result["prediction"]), scaler, feat_cols, target)
            target_idx = feat_cols.index(target)
            gt_start   = start_idx + seq_len
            true       = raw[gt_start: gt_start + pred_len, target_idx]
            hist       = raw[start_idx: start_idx + seq_len, target_idx]
            dates_hist = df["date"].iloc[start_idx: start_idx + seq_len].values
            dates_pred = df["date"].iloc[gt_start: gt_start + pred_len].values

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates_hist[-48:], y=hist[-48:],
                name="History", line=dict(color="#90a4ae", width=1.2, dash="dot"),
                opacity=0.7,
            ))
            fig.add_trace(go.Scatter(
                x=dates_pred, y=true, name="Ground Truth",
                line=dict(color=PALETTE["truth"], width=2),
            ))
            fig.add_trace(go.Scatter(
                x=dates_pred, y=pred, name="Prediction",
                line=dict(color=PALETTE["pred"], width=2, dash="dash"),
            ))
            fig.update_layout(
                **_plotly_layout(
                    title=f"Forecast — {chosen_model['model'].upper()} | pred_len={pred_len}",
                    xlabel="Date", ylabel=f"{target} (original scale)",
                ),
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True)

            from src.evaluation.metrics import compute_forecast_metrics
            m = compute_forecast_metrics(pred, true)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("MAE",  f"{m['mae']:.4f}")
            c2.metric("MSE",  f"{m['mse']:.4f}")
            c3.metric("RMSE", f"{m['rmse']:.4f}")
            c4.metric("MAPE", f"{m['mape']:.2f}%")

    # ══════════════════════════════════════════════════════════════════════════
    #  Multi-model comparison mode
    # ══════════════════════════════════════════════════════════════════════════
    else:
        st.markdown(
            "Run **TFT, LSTM, and Informer** on the same data window and compare "
            "their predictions side-by-side."
        )

        # Group models by (dataset, seq_len, pred_len)
        from collections import defaultdict
        groups = defaultdict(dict)
        for m in trained:
            key = (m["dataset"], m["seq_len"], m["pred_len"])
            groups[key][m["model"]] = m

        # Keep only groups that have all three model types
        full_groups = {
            k: v for k, v in groups.items()
            if all(t in v for t in ("tft", "lstm", "informer"))
        }

        if not full_groups:
            st.warning(
                "No configuration found where TFT, LSTM, and Informer are all trained "
                "on the same dataset/seq_len/pred_len. Train all three models first."
            )
            st.stop()

        group_labels = [
            f"{ds} · seq={sq} · pred={pr}"
            for (ds, sq, pr) in full_groups
        ]
        chosen_group_key = list(full_groups.keys())[
            st.selectbox("Configuration", range(len(group_labels)),
                         format_func=lambda i: group_labels[i])
        ]
        group = full_groups[chosen_group_key]
        anchor = group["tft"]

        df        = load_csv(anchor["data_path"])
        feat_cols = [c for c in df.columns if c != "date"]
        seq_len   = anchor["seq_len"]
        pred_len  = anchor["pred_len"]
        label_len = anchor.get("label_len", 48)
        target    = anchor.get("target", "OT")
        freq      = anchor.get("freq", "h")
        max_start = len(df) - seq_len - pred_len - 1
        start_idx = st.slider("Window start (test region)",
                              int(len(df) * 0.8), max_start, int(len(df) * 0.8))

        if st.button("▶ Compare All Models", type="primary"):
            raw, scaler, enc_x, enc_t, dec_x, dec_t = _build_inputs(
                df, feat_cols, seq_len, pred_len, label_len, start_idx, freq=freq)
            target_idx = feat_cols.index(target)
            gt_start   = start_idx + seq_len
            true       = raw[gt_start: gt_start + pred_len, target_idx]
            hist       = raw[start_idx: start_idx + seq_len, target_idx]
            dates_hist = df["date"].iloc[start_idx: start_idx + seq_len].values
            dates_pred = df["date"].iloc[gt_start: gt_start + pred_len].values

            model_style = {
                "tft":      (PALETTE["tft"],      "solid",  "TFT (Ours)"),
                "lstm":     (PALETTE["lstm"],     "dash",   "LSTM"),
                "informer": (PALETTE["informer"], "dot",    "Informer"),
            }

            from src.evaluation.metrics import compute_forecast_metrics
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates_hist[-48:], y=hist[-48:],
                name="History", line=dict(color="#90a4ae", width=1.2, dash="dot"),
                opacity=0.6,
            ))
            fig.add_trace(go.Scatter(
                x=dates_pred, y=true, name="Ground Truth",
                line=dict(color=PALETTE["truth"], width=2.5),
            ))

            metric_rows = []
            for model_type in ("tft", "lstm", "informer"):
                m_cfg = group[model_type]
                cfg_path = (
                    f"configs/{model_type}_{m_cfg['dataset'].lower()}"
                    f"_{seq_len}_{pred_len}.yaml"
                )
                color, dash, label = model_style[model_type]
                with st.spinner(f"Running {label}..."):
                    try:
                        res = _api_predict(cfg_path, enc_x, enc_t, dec_x, dec_t)
                    except Exception as e:
                        st.warning(f"{label} failed: {e}")
                        continue
                pred = _inv_scale(np.array(res["prediction"]), scaler, feat_cols, target)
                fig.add_trace(go.Scatter(
                    x=dates_pred, y=pred, name=label,
                    line=dict(color=color, width=2, dash=dash),
                ))
                met = compute_forecast_metrics(pred, true)
                metric_rows.append({
                    "Model": label,
                    "MAE":   round(met["mae"],  4),
                    "MSE":   round(met["mse"],  4),
                    "RMSE":  round(met["rmse"], 4),
                    "MAPE":  f"{met['mape']:.2f}%",
                })

            fig.update_layout(
                **_plotly_layout(
                    title=f"Multi-Model Comparison | {anchor['dataset']} | pred={pred_len}",
                    xlabel="Date", ylabel=f"{target} (original scale)",
                ),
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── metrics table ─────────────────────────────────────────────────
            if metric_rows:
                st.markdown("#### Model Performance on This Window")
                df_met = pd.DataFrame(metric_rows).set_index("Model")
                st.dataframe(
                    df_met.style.highlight_min(
                        subset=["MAE", "MSE", "RMSE"], color="#1b3a2b"
                    ),
                    use_container_width=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
#  Page 3 — Anomaly Monitor
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🚨 Anomaly Monitor":
    import plotly.graph_objects as go

    st.title("🚨 Anomaly Monitor")
    st.markdown("Visualise residuals and MAD-based anomaly flags from a saved test run.")

    results_root = "results"
    run_dirs = []
    for root, dirs, files in os.walk(results_root):
        if "anomaly.csv" in files and "residual.csv" in files:
            run_dirs.append(root)

    if not run_dirs:
        st.warning("No anomaly results found. Run `python -m src.experiment.test --config ...`.")
        st.stop()

    chosen_run = st.selectbox("Select experiment run", run_dirs)
    residual = pd.read_csv(os.path.join(chosen_run, "residual.csv")).values.flatten()
    n_show   = st.slider("Time steps to display", 100, min(2000, len(residual)), 600, step=100)

    # ── MAD controls ──────────────────────────────────────────────────────────
    col_w, col_l = st.columns(2)
    with col_w:
        window = st.slider("MAD window", 12, 168, 48, step=12)
    with col_l:
        lam = st.slider("λ (threshold multiplier)", 1.0, 6.0, 3.0, step=0.5)

    from src.evaluation.anomaly import ResidualMADDetector
    detector    = ResidualMADDetector(window=window, lam=lam)
    anom_result = detector.detect(residual)

    n_flagged = int(anom_result["anomaly_flag"].sum())
    flag_rate = n_flagged / len(residual) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Total time steps", f"{len(residual):,}")
    c2.metric("Anomalies detected", n_flagged)
    c3.metric("Flag rate", f"{flag_rate:.2f}%")

    st.markdown("---")

    # ── interactive plot ──────────────────────────────────────────────────────
    t   = np.arange(n_show)
    res = residual[:n_show]
    thr = anom_result["threshold"][:n_show]
    flg = anom_result["anomaly_flag"][:n_show]
    anom_idx = np.where(flg > 0)[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=res,
        name="|Residual|",
        line=dict(color=PALETTE["residual"], width=1.0),
        fill="tozeroy",
        fillcolor="rgba(100,181,246,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=t, y=thr,
        name=f"MAD Threshold (λ={lam})",
        line=dict(color=PALETTE["thresh"], width=1.8, dash="dash"),
    ))
    if len(anom_idx) > 0:
        fig.add_trace(go.Scatter(
            x=t[anom_idx], y=res[anom_idx],
            name="Anomaly",
            mode="markers",
            marker=dict(color=PALETTE["anomaly"], size=7, symbol="x",
                        line=dict(width=1.5, color=PALETTE["anomaly"])),
        ))
    fig.update_layout(
        **_plotly_layout(
            title="Residual & MAD Anomaly Detection",
            xlabel="Time step", ylabel="Absolute Residual",
        ),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── anomaly table ──────────────────────────────────────────────────────────
    if len(anom_idx) > 0:
        with st.expander(f"Anomaly positions ({len(anom_idx)} points)"):
            anom_df = pd.DataFrame({
                "Step": t[anom_idx],
                "Residual": res[anom_idx].round(4),
                "Threshold": thr[anom_idx].round(4),
                "Excess": (res[anom_idx] - thr[anom_idx]).round(4),
            })
            st.dataframe(anom_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Page 4 — Results Viewer
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Results Viewer":
    import plotly.graph_objects as go

    st.title("📈 Results Viewer")
    st.markdown("Compare saved experiment metrics across models and configurations.")

    results_root = "results"
    if not os.path.isdir(results_root):
        st.warning("No results directory found.")
        st.stop()

    # ── collect metrics ────────────────────────────────────────────────────────
    metric_files = []
    for root, dirs, files in os.walk(results_root):
        if "metrics.json" in files:
            metric_files.append(os.path.join(root, "metrics.json"))

    if not metric_files:
        st.warning("No metrics found. Train and test models first.")
        st.stop()

    records = []
    for mf in metric_files:
        with open(mf) as f:
            m = json.load(f)
        rel = os.path.relpath(mf, results_root)
        parts = rel.split(os.sep)
        # Try to parse model/dataset from path
        model_tag = parts[1] if len(parts) > 1 else "unknown"
        records.append({
            "run":   rel.replace(f"{os.sep}metrics.json", ""),
            "model": model_tag,
            **m,
        })

    df_metrics = pd.DataFrame(records)

    # ── summary table ──────────────────────────────────────────────────────────
    st.subheader("All Experiments")
    styled = df_metrics.style.format({
        "mae":  "{:.4f}",
        "mse":  "{:.4f}",
        "rmse": "{:.4f}",
        "mape": "{:.2f}",
    }).background_gradient(subset=["mae", "mse"], cmap="RdYlGn_r")
    st.dataframe(styled, use_container_width=True)

    # ── bar chart ──────────────────────────────────────────────────────────────
    if len(df_metrics) > 1:
        st.markdown("---")
        metric_choice = st.selectbox("Metric to visualise", ["mae", "mse", "rmse", "mape"])

        if metric_choice in df_metrics.columns:
            model_colors = {
                "tft": PALETTE["tft"], "informer": PALETTE["informer"],
                "lstm": PALETTE["lstm"],
            }
            bar_colors = [
                model_colors.get(r.split("/")[0].split("_")[0], "#78909c")
                for r in df_metrics["run"]
            ]

            fig = go.Figure(go.Bar(
                x=df_metrics["run"],
                y=df_metrics[metric_choice],
                marker_color=bar_colors,
                text=df_metrics[metric_choice].round(4),
                textposition="outside",
                textfont=dict(color="#cfd8dc", size=11),
            ))
            fig.update_layout(
                **_plotly_layout(
                    title=f"{metric_choice.upper()} Comparison",
                    xlabel="Experiment", ylabel=metric_choice.upper(),
                ),
                height=420,
                xaxis_tickangle=-35,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── loss curves ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Training Loss Curves")
    ckpt_root = "results/checkpoints"
    loss_files = []
    if os.path.isdir(ckpt_root):
        for root, dirs, files in os.walk(ckpt_root):
            if "loss_history.json" in files:
                loss_files.append(os.path.join(root, "loss_history.json"))

    if loss_files:
        chosen_loss = st.selectbox("Select run", loss_files,
                                    format_func=lambda p: os.path.relpath(p, ckpt_root))
        with open(chosen_loss) as f:
            lh = json.load(f)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=lh["train"], name="Train loss",
            line=dict(color="#42a5f5", width=1.8),
        ))
        fig.add_trace(go.Scatter(
            y=lh["val"], name="Val loss",
            line=dict(color="#ff7043", width=1.8, dash="dash"),
        ))
        fig.update_layout(
            **_plotly_layout(title="Loss Curve", xlabel="Epoch", ylabel="MSE Loss"),
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No loss history found yet.")

    # ── ablation summary ───────────────────────────────────────────────────────
    ablation_path = "results/metrics/ablation_summary.csv"
    if os.path.exists(ablation_path):
        st.markdown("---")
        st.subheader("Ablation Summary")
        st.dataframe(pd.read_csv(ablation_path), use_container_width=True)

    # ── thesis model comparison table ─────────────────────────────────────────
    baseline_path = "results/metrics/baselines_comparison.csv"
    if os.path.exists(baseline_path):
        st.markdown("---")
        st.subheader("Thesis: Model Comparison (ETTh1 · seq=96 · pred=24)")
        df_base = pd.read_csv(baseline_path, index_col=0)

        # Highlight the TFT row (best model) in green
        def _highlight_tft(row):
            return [
                "background-color: #1b3a2b; color: #66bb6a; font-weight: 700"
                if row.name == "tft" else ""
            ] * len(row)

        styled_base = (
            df_base.style
            .apply(_highlight_tft, axis=1)
            .format("{:.4f}")
            .set_table_styles([{
                "selector": "th",
                "props": [("color", "#90a4ae"), ("font-size", "0.85rem")],
            }])
        )
        st.dataframe(styled_base, use_container_width=True)
        st.caption(
            "TFT achieves the lowest MAE and MSE, outperforming both LSTM and Informer baselines."
        )

    # ── prediction chart replay ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Prediction Chart Replay")
    st.markdown("Visualise saved forecast results from any completed experiment run.")

    pred_runs = []
    for root, dirs, files in os.walk("results"):
        if "pred.csv" in files and "true.csv" in files:
            pred_runs.append(root)

    if not pred_runs:
        st.info("No saved prediction files found. Run `python -m src.experiment.test` first.")
    else:
        chosen_pred_run = st.selectbox(
            "Select experiment run",
            pred_runs,
            format_func=lambda p: os.path.relpath(p, "results"),
            key="pred_run_sel",
        )
        pred_vals = pd.read_csv(os.path.join(chosen_pred_run, "pred.csv")).values.flatten()
        true_vals = pd.read_csv(os.path.join(chosen_pred_run, "true.csv")).values.flatten()

        n_show = st.slider(
            "Time steps to display", 24,
            min(500, len(pred_vals)), min(200, len(pred_vals)),
            step=24, key="pred_nshow",
        )
        pred_show = pred_vals[:n_show]
        true_show = true_vals[:n_show]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=true_show, name="Ground Truth",
            line=dict(color=PALETTE["truth"], width=2),
        ))
        fig.add_trace(go.Scatter(
            y=pred_show, name="Prediction",
            line=dict(color=PALETTE["pred"], width=2, dash="dash"),
        ))
        fig.update_layout(
            **_plotly_layout(
                title=f"Saved Predictions — {os.path.relpath(chosen_pred_run, 'results')}",
                xlabel="Time step", ylabel="OT (normalised)",
            ),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Quick metrics on the shown window
        from src.evaluation.metrics import compute_forecast_metrics
        qm = compute_forecast_metrics(pred_show, true_show)
        qc1, qc2, qc3, qc4 = st.columns(4)
        qc1.metric("MAE",  f"{qm['mae']:.4f}")
        qc2.metric("MSE",  f"{qm['mse']:.4f}")
        qc3.metric("RMSE", f"{qm['rmse']:.4f}")
        qc4.metric("MAPE", f"{qm['mape']:.2f}%")


# ══════════════════════════════════════════════════════════════════════════════
#  Page 5 — Interpretability
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Interpretability":
    import plotly.graph_objects as go

    st.title("🔍 Interpretability (TFT)")
    st.markdown("Visualise variable selection weights and temporal attention patterns.")

    results_root = "results"

    # ── variable importance ────────────────────────────────────────────────────
    vi_path = "results/variable_importance/variable_importance.csv"
    if os.path.exists(vi_path):
        st.subheader("Variable Selection Weights")
        vi = pd.read_csv(vi_path)

        # Map generic names to real names
        hist_map   = {f"hist_var_{i}": n for i, n in enumerate(FEAT_NAMES)}
        future_map = {f"future_var_{i}": n for i, n in enumerate(TIME_NAMES)}
        name_map   = {**hist_map, **future_map}
        vi["label"] = vi["variable"].map(name_map).fillna(vi["variable"])

        col_h, col_f = st.columns(2)

        for (grp, title, col) in [
            ("historical", "Historical Variables", col_h),
            ("future",     "Future (Time) Variables", col_f),
        ]:
            subset = vi[vi["group"] == grp].sort_values("importance", ascending=True)
            if subset.empty:
                continue
            fig = go.Figure(go.Bar(
                x=subset["importance"],
                y=subset["label"],
                orientation="h",
                marker=dict(
                    color=subset["importance"],
                    colorscale="Viridis",
                    showscale=False,
                ),
                text=subset["importance"].map("{:.4f}".format),
                textposition="outside",
                textfont=dict(color="#cfd8dc"),
            ))
            fig.update_layout(
                **_plotly_layout(title=title, xlabel="Importance", ylabel=""),
                height=max(250, len(subset) * 50),
            )
            col.plotly_chart(fig, use_container_width=True)

        with st.expander("Raw importance data"):
            st.dataframe(vi[["group", "label", "importance"]], use_container_width=True)
    else:
        st.info("Variable importance data not found. Run TFT test first.")

    st.markdown("---")

    # ── attention heatmap ──────────────────────────────────────────────────────
    st.subheader("Temporal Self-Attention Heatmap")

    attn_runs = []
    for root, dirs, files in os.walk(results_root):
        if "attention.npy" in files:
            attn_runs.append(root)

    if not attn_runs:
        st.info("No attention data found. Run TFT test first.")
    else:
        chosen_run = st.selectbox("Select TFT run", attn_runs)
        attn = np.load(os.path.join(chosen_run, "attention.npy"))

        fig = go.Figure(go.Heatmap(
            z=attn,
            colorscale="YlOrRd",
            colorbar=dict(
                title=dict(text="Weight", font=dict(color="#90a4ae")),
                tickfont=dict(color="#90a4ae"),
            ),
        ))
        fig.update_layout(
            **_plotly_layout(
                title="Temporal Attention Weights (pred_len × seq_len)",
                xlabel="Encoder time step (seq_len)",
                ylabel="Decoder prediction step (pred_len)",
            ),
            height=max(300, attn.shape[0] * 18),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Brighter cells indicate higher attention. Each row is a forecast step; "
            "each column is a historical encoder step."
        )
