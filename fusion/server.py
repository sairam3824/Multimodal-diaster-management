"""
Unified Disaster Response Server
==================================
FastAPI server that merges the existing CrisisMMD classifier with
all 6 IoT sensor models through the FusionLayer.

Endpoints:
  GET  /health
  POST /analyze        → full pipeline (image + tweet + IoT fields)
  POST /crisis/predict → crisis model only (mirrors crisis/server.py)
  POST /iot/predict    → IoT models only
  GET  /               → serves static/index.html dashboard

Run:
    uvicorn fusion.server:app --host 0.0.0.0 --port 8001 --reload
"""

from __future__ import annotations

import io
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env manually — no external dependency needed
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from fusion.pipeline import DisasterPipeline, CRISIS_CLASSES
from fusion.xai import GradCAMViT, generate_openai_summary

# ─────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────
state: dict = {}
STATIC_DIR  = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Server] Initializing Disaster Response Pipeline…")
    state["pipeline"] = DisasterPipeline(load_crisis_model=True)
    # Set up Grad-CAM if crisis model loaded
    pipe = state["pipeline"]
    if pipe._crisis_model is not None:
        try:
            state["gradcam"] = GradCAMViT(pipe._crisis_model)
            print("[Server] Grad-CAM ready.")
        except Exception as e:
            import traceback
            print(f"[Server] Grad-CAM init FAILED: {e}")
            traceback.print_exc()
    state["openai_key"] = os.getenv("OPENAI_API_KEY", "")
    print(f"[Server] OpenAI key loaded: {'yes' if state['openai_key'] else 'NO KEY FOUND'}")
    state["openai_key"] = os.getenv("OPENAI_API_KEY", "")
    print("[Server] Ready.")
    yield
    state.clear()


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(
    title="Unified Disaster Response API",
    description="IoT × CrisisMMD fusion for real-time disaster assessment",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if (STATIC_DIR / "index.html").exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/")
async def root():
    html = STATIC_DIR / "index.html"
    if html.exists():
        return FileResponse(str(html))
    return JSONResponse({"message": "Unified Disaster Response API v1.0"})


@app.get("/health")
async def health():
    return {
        "status":       "ok",
        "pipeline":     "pipeline" in state,
        "crisis_model": state.get("pipeline") is not None
                        and state["pipeline"]._crisis_model is not None,
        "iot_model":    "iot_model.pth loaded"
                        if "pipeline" in state else "not loaded",
    }


@app.post("/analyze")
async def analyze(
    image:         UploadFile = File(...),
    tweet:         str        = Form(...),
    lat:           Optional[float] = Form(None),
    lon:           Optional[float] = Form(None),
    # fire
    max_temp:          Optional[float] = Form(None),
    min_temp:          Optional[float] = Form(None),
    avg_wind_speed:    Optional[float] = Form(None),
    precipitation:     Optional[float] = Form(None),
    month:             Optional[int]   = Form(None),
    # storm
    wind_kts:      Optional[float] = Form(None),
    pressure:      Optional[float] = Form(None),
    # earthquake
    depth:         Optional[float] = Form(None),
    # flood
    elevation_m:         Optional[float] = Form(None),
    distance_to_river_m: Optional[float] = Form(None),
    rainfall_7d:         Optional[float] = Form(None),
    monthly_rainfall:    Optional[float] = Form(None),
    drainage_index:      Optional[float] = Form(None),
    ndvi:                Optional[float] = Form(None),
    ndwi:                Optional[float] = Form(None),
):
    if "pipeline" not in state:
        raise HTTPException(503, "Pipeline not ready")

    # Save uploaded image to temp file
    img_bytes = await image.read()
    tmp_path  = f"/tmp/crisis_upload_{int(time.time())}.jpg"
    with open(tmp_path, "wb") as f:
        f.write(img_bytes)

    # Build sensor kwargs — only include fields the user actually submitted
    _all_sensor: Dict[str, Any] = {
        "lat": lat, "lon": lon,
        "max_temp": max_temp, "min_temp": min_temp,
        "avg_wind_speed": avg_wind_speed, "precipitation": precipitation,
        "month": month,
        "wind_kts": wind_kts, "pressure": pressure,
        "depth": depth,
        "elevation_m": elevation_m, "distance_to_river_m": distance_to_river_m,
        "rainfall_7d": rainfall_7d, "monthly_rainfall": monthly_rainfall,
        "drainage_index": drainage_index, "ndvi": ndvi, "ndwi": ndwi,
    }
    sensor_kwargs = {k: v for k, v in _all_sensor.items() if v is not None}

    t0 = time.perf_counter()
    try:
        result = state["pipeline"].analyze(
            image_path = tmp_path,
            tweet      = tweet,
            **sensor_kwargs,
        )

        # ── Grad-CAM ──────────────────────────────────────────────────────
        gradcam_b64 = ""
        if "gradcam" in state:
            try:
                pipe    = state["pipeline"]
                img_pil = Image.open(tmp_path).convert("RGB")
                img_t   = pipe.IMG_TRANSFORM(img_pil).unsqueeze(0).to(pipe.device)
                enc     = pipe._crisis_tokenizer(
                    tweet, max_length=128, padding="max_length",
                    truncation=True, return_tensors="pt"
                )
                gradcam_b64 = state["gradcam"].compute(
                    img_t,
                    enc["input_ids"].to(pipe.device),
                    enc["attention_mask"].to(pipe.device),
                    img_pil,
                )
            except Exception as e:
                print(f"[Server] Grad-CAM skipped: {e}")

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    # Build response dict for OpenAI summary
    response_dict = {
        "alert_level":    result.alert_level,
        "disaster_type":  result.confirmed_type,
        "priority":       result.priority,
        "fused_severity": round(result.fused_severity, 4),
        "iot": {
            "type":          result.iot_disaster_type,
            "fire_prob":     round(result.iot_fire_prob, 4),
            "storm_cat":     round(result.iot_storm_cat, 4),
            "eq_magnitude":  round(result.iot_eq_magnitude, 4),
            "flood_risk":    round(result.iot_flood_risk, 4),
            "casualty_risk": round(result.iot_casualty_risk, 4),
            "sensor_weights": {k: round(v, 4) for k, v in result.iot_sensor_weights.items()},
        },
        "crisis": {
            "category":      result.crisis_category,
            "confidence":    round(result.crisis_confidence, 4),
            "vision_weight": round(result.vision_weight, 4),
            "text_weight":   round(result.text_weight, 4),
        },
        "fusion": {
            "population_impact": round(result.population_impact, 4),
            "resource_needs":    {k: round(v, 4) for k, v in result.resource_needs.items()},
        },
    }
    xai_summary = generate_openai_summary(response_dict, state.get("openai_key", ""))

    return JSONResponse({
        "alert_level":     result.alert_level,
        "summary":         result.summary,
        "disaster_type":   result.confirmed_type,
        "priority":        result.priority,
        "fused_severity":  round(result.fused_severity, 4),
        "iot": {
            "type":           result.iot_disaster_type,
            "fire_prob":      round(result.iot_fire_prob, 4),
            "storm_cat":      round(result.iot_storm_cat, 4),
            "eq_magnitude":   round(result.iot_eq_magnitude, 4),
            "flood_risk":     round(result.iot_flood_risk, 4),
            "casualty_risk":  round(result.iot_casualty_risk, 4),
            "severity":       round(result.iot_severity, 4),
            "sensor_weights": {k: round(v, 4) for k, v in result.iot_sensor_weights.items()},
        },
        "crisis": {
            "category":     result.crisis_category,
            "confidence":   round(result.crisis_confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in result.crisis_probs.items()},
            "vision_weight": round(result.vision_weight, 4),
            "text_weight":   round(result.text_weight, 4),
        },
        "fusion": {
            "priority_probs":    {k: round(v, 4) for k, v in result.priority_probs.items()},
            "type_probs":        {k: round(v, 4) for k, v in result.type_probs.items()},
            "population_impact": round(result.population_impact, 4),
            "resource_needs":    {k: round(v, 4) for k, v in result.resource_needs.items()},
        },
        "xai": {
            "gradcam_b64": gradcam_b64,
            "summary":     xai_summary,
        },
        "inference_ms": elapsed_ms,
    })


@app.post("/iot/predict")
async def iot_predict(
    disaster_type: str   = Form(...),
    lat:           float = Form(0.0),
    lon:           float = Form(0.0),
    month:         int   = Form(6),
    # fire
    max_temp:       float = Form(30.0),
    min_temp:       float = Form(15.0),
    avg_wind_speed: float = Form(5.0),
    precipitation:  float = Form(0.0),
    year:           int   = Form(2024),
    day_of_year:    int   = Form(180),
    season:         str   = Form("Summer"),
    # storm
    wind_kts:  float = Form(0.0),
    pressure:  float = Form(1013.0),
    basin:     str   = Form("North Atlantic"),
    # earthquake
    depth:     float = Form(10.0),
    # flood
    elevation_m:         float = Form(50.0),
    distance_to_river_m: float = Form(500.0),
    rainfall_7d:         float = Form(0.0),
    monthly_rainfall:    float = Form(0.0),
    drainage_index:      float = Form(0.5),
    ndvi:                float = Form(0.3),
    ndwi:                float = Form(0.1),
):
    if "pipeline" not in state:
        raise HTTPException(503, "Pipeline not ready")

    pred = state["pipeline"].iot.predict(
        disaster_type=disaster_type, lat=lat, lon=lon,
        month=month, max_temp=max_temp, min_temp=min_temp,
        avg_wind_speed=avg_wind_speed, precipitation=precipitation,
        year=year, day_of_year=day_of_year, season=season,
        wind_kts=wind_kts, pressure=pressure, basin=basin,
        depth=depth,
        elevation_m=elevation_m, distance_to_river_m=distance_to_river_m,
        rainfall_7d=rainfall_7d, monthly_rainfall=monthly_rainfall,
        drainage_index=drainage_index, ndvi=ndvi, ndwi=ndwi,
    )
    return JSONResponse({
        "disaster_type":  pred.disaster_type,
        "fire_prob":      round(pred.fire_prob, 4),
        "storm_cat":      round(pred.storm_cat, 1),
        "eq_magnitude":   round(pred.eq_magnitude, 2),
        "flood_risk":     round(pred.flood_risk, 1),
        "severity_score": round(pred.severity_score, 4),
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fusion.server:app", host="0.0.0.0", port=8001, reload=False)
