"""
NLP Pipeline Dashboard (Minimal, "auto-20" from CSV)

What changed (per request)
--------------------------
- Reads CSV path from env var DATA_CSV (set it at CLI when running uvicorn)
- On the homepage, automatically samples 20 rows from DATA_CSV and shows predictions
- A single "Predict 20 random rows" action re-samples 20 from the same CSV
- Artifacts (metrics + images) are displayed directly inline (no "open" buttons)
- Removed: upload flow, config link/buttons, pr_curve/confusion_matrix buttons, /sample route

Run
---
DATA_CSV=/absolute/path/to/data.csv ARTIFACTS_DIR=artifacts uvicorn dashboard:app --reload --port 7860

Optional env:
- TARGET_COL   : if your CSV includes the target column and you want it dropped before inference
- DASH_FAST_HOME=1 : skip expected-columns inspection on home to load faster
"""

from __future__ import annotations

import html
import json
import os
import time
from typing import Any, List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

# -----------------------------------------------------------------------------
# Configuration (CLI via environment variables)
# -----------------------------------------------------------------------------

ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")
RUNCFG_PATH = os.path.join(ARTIFACTS_DIR, "run_config.json")  # optional, not linked in UI
DATA_CSV = "./data.csv"
TARGET_COL = os.environ.get("TARGET_COL", "").strip()         # optional
DASH_FAST_HOME = os.environ.get("DASH_FAST_HOME", "0") == "1"

# How many rows to preview on page
AUTO_N = 20
MAX_PREVIEW_ROWS = 500

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

app = FastAPI(
    title="NLP Pipeline Dashboard (Auto-20)",
    description="Loads a trained model, samples 20 rows from DATA_CSV, and displays predictions.",
    version="3.0",
)

# Serve artifacts directory (metrics.json, model.pkl, plots, etc.)
app.mount("/artifacts", StaticFiles(directory=ARTIFACTS_DIR), name="artifacts")

# In-process cache for model
_MODEL_CACHE: dict[str, Any] = {"mtime": None, "obj": None}


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _exists(path: str) -> bool:
    return os.path.isfile(path)


def _safe_html(x: Any) -> str:
    return html.escape(str(x), quote=True)


def _load_json(path: str) -> dict:
    try:
        if _exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _load_model():
    if not _exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Train first (artifacts/model.pkl).")
    mtime = os.path.getmtime(MODEL_PATH)
    if _MODEL_CACHE["mtime"] != mtime or _MODEL_CACHE["obj"] is None:
        _MODEL_CACHE["obj"] = joblib.load(MODEL_PATH)
        _MODEL_CACHE["mtime"] = mtime
    return _MODEL_CACHE["obj"]


def _expected_feature_columns(pipe) -> List[str]:
    """
    Best-effort: if the pipeline has a 'preprocessor' ColumnTransformer with
    explicit list/tuple selectors, we collect those as expected columns.
    """
    pre = getattr(pipe, "named_steps", {}).get("preprocessor")
    if pre is None:
        return []
    cols: List[str] = []
    transformers = getattr(pre, "transformers_", None) or getattr(pre, "transformers", [])
    for _name, _trans, sel in transformers:
        if isinstance(sel, (list, tuple)):
            cols.extend([str(c) for c in sel])
    # dedupe
    seen, uniq = set(), []
    for c in cols:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def _predict_df(pipe, df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict on df, returning a new df with 'prediction' and optional 'proba_1'.
    """
    out = df.copy()
    preds = pipe.predict(df)
    out["prediction"] = preds

    # Probability if available
    try:
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(df)
            if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] >= 2:
                out["proba_1"] = proba[:, 1]
                return out
    except Exception:
        pass

    # decision_function mapped to [0,1]
    try:
        if hasattr(pipe, "decision_function"):
            dec = pipe.decision_function(df)
            dec = np.asarray(dec)
            if dec.ndim == 1:
                out["proba_1"] = 1.0 / (1.0 + np.exp(-dec))
    except Exception:
        pass
    return out


def _html_table(df: pd.DataFrame) -> str:
    head = df.head(min(MAX_PREVIEW_ROWS, len(df)))
    th = "".join(f"<th>{_safe_html(c)}</th>" for c in head.columns)
    trs = []
    for _, r in head.iterrows():
        tds = "".join(f"<td>{_safe_html(v)}</td>" for v in r.values)
        trs.append(f"<tr>{tds}</tr>")
    return f"""
    <div class="table-wrap">
      <table class="table">
        <thead><tr>{th}</tr></thead>
        <tbody>{''.join(trs)}</tbody>
      </table>
    </div>
    """


def _layout(title: str, body_html: str) -> HTMLResponse:
    """Dark-themed minimal layout; artifacts shown inline, no buttons."""
    doc = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1"/>
      <title>{_safe_html(title)}</title>
      <style>
        :root {{
          --bg: #0f1117; --card: #161a22; --muted: #9aa3b2; --text: #e6eaf0;
          --accent: #5ee6a8; --accent-2: #7aa2f7; --stroke: #2a2f3a;
        }}
        * {{ box-sizing: border-box; }}
        body {{ margin: 24px; font-family: ui-sans-serif, system-ui; background: var(--bg); color: var(--text); }}
        h1 {{ margin: 0 0 12px; font-size: 28px; }}
        h2 {{ margin: 0 0 10px; font-size: 20px; }}
        .muted {{ color: var(--muted); font-size: 0.95rem; }}
        .row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; margin: 16px 0; }}
        .card {{ background: var(--card); border: 1px solid var(--stroke); border-radius: 12px; padding: 16px; }}
        .btn {{ display:inline-block; background: var(--accent); color: #0b1220; padding: 8px 12px; border-radius: 8px; font-weight: 600; text-decoration:none; }}
        .table {{ width: 100%; border-collapse: collapse; font-size: 0.95rem; }}
        .table th, .table td {{ border-bottom: 1px solid var(--stroke); padding: 8px 10px; text-align: left; }}
        .img-wrap img {{ max-width: 100%; height: auto; border: 1px solid var(--stroke); border-radius: 8px; }}
        .sep {{ height: 1px; background: var(--stroke); margin: 12px 0; }}
        code {{ background: #0b1220; color: #d7e0ef; padding: 2px 6px; border-radius: 6px; }}
      </style>
    </head>
    <body>
      <h1>NLP Pipeline Dashboard</h1>
      <div class="muted">Artifacts dir: <code>{_safe_html(ARTIFACTS_DIR)}</code> • CSV: <code>{_safe_html(DATA_CSV or 'NOT SET')}</code></div>
      {body_html}
    </body>
    </html>
    """
    return HTMLResponse(doc)


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/healthz", response_class=PlainTextResponse)
def healthz() -> str:
    return "ok"


@app.get("/metrics.json", response_class=JSONResponse)
def metrics_json() -> JSONResponse:
    data = _load_json(METRICS_PATH)
    if not data:
        return JSONResponse({"error": "No metrics found. Train first."}, status_code=404)
    return JSONResponse(data)


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    """
    Home: shows metrics + inline artifacts + auto 20-row prediction preview from DATA_CSV.
    Removes config links and "open" buttons; keeps a single "Predict 20 random rows" action.
    """
    if not DATA_CSV or not os.path.exists(DATA_CSV):
        raise HTTPException(status_code=400, detail="DATA_CSV not set or file not found. Set DATA_CSV env.")

    # Metrics card (inline values)
    metrics = _load_json(METRICS_PATH)
    if metrics:
        rows = "".join(
            f"<tr><td>{_safe_html(k)}</td><td>{_safe_html(v)}</td></tr>"
            for k, v in metrics.items() if not isinstance(v, dict)
        )
        metrics_card = f"""
        <div class="card">
          <h2>Metrics</h2>
          <table class="table">
            <thead><tr><th>Metric</th><th>Value</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>
        """
    else:
        metrics_card = """
        <div class="card">
          <h2>Metrics</h2>
          <p class="muted">No metrics found in artifacts/metrics.json.</p>
        </div>
        """

    # Artifacts images displayed directly (if present)
    imgs_html = []
    for name in ("confusion_matrix.png", "pr_curve.png"):
        path = os.path.join(ARTIFACTS_DIR, name)
        if _exists(path):
            imgs_html.append(f'<div class="img-wrap"><img src="/artifacts/{_safe_html(name)}" alt="{_safe_html(name)}"/></div>')
    artifacts_card = f"""
    <div class="card">
      <h2>Artifacts</h2>
      {"".join(imgs_html) if imgs_html else '<p class="muted">No images found (confusion_matrix.png, pr_curve.png).</p>'}
    </div>
    """

    # Auto-20 predictions preview
    pipe = _load_model()
    # best-effort: drop target if provided
    df = pd.read_csv(DATA_CSV)
    if TARGET_COL and TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])
    if len(df) == 0:
        raise HTTPException(status_code=400, detail="DATA_CSV has no rows to sample.")

    sample_df = df.sample(n=min(AUTO_N, len(df)), random_state=int(time.time()) % (2**32 - 1))
    t0 = time.perf_counter()
    preds_df = _predict_df(pipe, sample_df)
    dt = time.perf_counter() - t0

    pred_table = _html_table(preds_df)
    predict_card = f"""
    <div class="card">
      <h2>Predictions (auto 20)</h2>
      <div class="muted">Predicted {len(preds_df)} sampled rows in {dt:.2f}s from <code>{_safe_html(DATA_CSV)}</code>.</div>
      {pred_table}
      <div class="sep"></div>
      <a class="btn" href="/predict">Predict 20 random rows</a>
    </div>
    """

    body = f"""
    <div class="row">{metrics_card}{artifacts_card}</div>
    <div class="row">{predict_card}</div>
    """
    return _layout("NLP Pipeline Dashboard", body)


@app.get("/predict", response_class=HTMLResponse)
def predict() -> HTMLResponse:
    """
    Re-sample 20 rows from DATA_CSV and display predictions.
    This replaces upload/sample flows; uses the same CSV given via DATA_CSV env.
    """
    if not DATA_CSV or not os.path.exists(DATA_CSV):
        raise HTTPException(status_code=400, detail="DATA_CSV not set or file not found. Set DATA_CSV env.")

    pipe = _load_model()
    df = pd.read_csv(DATA_CSV)
    if TARGET_COL and TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])
    if len(df) == 0:
        raise HTTPException(status_code=400, detail="DATA_CSV has no rows to sample.")

    sample_df = df.sample(n=min(AUTO_N, len(df)), random_state=int(time.time()) % (2**32 - 1))
    t0 = time.perf_counter()
    preds_df = _predict_df(pipe, sample_df)
    dt = time.perf_counter() - t0

    pred_table = _html_table(preds_df)
    body = f"""
    <div class="card">
      <h2>Predictions (20 random rows)</h2>
      <div class="muted">Predicted {len(preds_df)} sampled rows in {dt:.2f}s from <code>{_safe_html(DATA_CSV)}</code>.</div>
      {pred_table}
      <div class="sep"></div>
      <a class="btn" href="/">Back</a>
    </div>
    """
    return _layout("Predictions", body)


@app.get("/healthz", response_class=PlainTextResponse)
def _healthz_duplicate() -> str:
    # Keep a second health endpoint name stable in case external probes target it
    return "ok"
