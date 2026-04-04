"""
Unified Disaster Response Server (Tri-Fusion)
===============================================
FastAPI server that merges the CrisisMMD classifier, IoT sensor models,
and xBD satellite damage model through the TriFusionLayer.

Endpoints:
  GET  /health
  POST /analyze              → full tri-fusion pipeline (image + tweet + IoT + satellite)
  POST /crisis/predict       → crisis model only
  POST /iot/predict          → IoT models only
  POST /satellite/predict    → satellite damage only (NEW)
  GET  /                     → serves static/index.html dashboard

Run:
    uvicorn fusion.server:app --host 0.0.0.0 --port 8001 --reload
"""

from __future__ import annotations

import asyncio
import io
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Dict, Any
from uuid import uuid4

import torch
import torch.nn.functional as F
from collections import defaultdict
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
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
from fusion.satellite_xai import SatelliteGradCAM, overlay_heatmap, overlay_to_base64

# ─────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────
state: dict = {}
STATIC_DIR  = Path(__file__).parent / "static"
PAGES_DIR   = STATIC_DIR / "pages"
JOB_UPLOAD_DIR = Path("/tmp/fusion_analysis_jobs")
JOBS_STORE_FILE = Path(__file__).parent / ".jobs_store.json"
STATIC_DIR.mkdir(exist_ok=True)
JOB_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Disk-backed job persistence
# ─────────────────────────────────────────────
import json

def _load_jobs_from_disk() -> dict:
    """Load persisted jobs from disk on startup."""
    if not JOBS_STORE_FILE.exists():
        return {}
    try:
        data = json.loads(JOBS_STORE_FILE.read_text())
        # Only keep completed/failed jobs (running jobs can't resume)
        kept = {}
        for job_id, job in data.items():
            if job.get("status") in ("completed", "failed"):
                kept[job_id] = job
            else:
                # Mark orphaned running/queued jobs as failed
                job["status"] = "failed"
                job["error"] = "Server restarted before job completed"
                job["updated_at"] = time.time()
                kept[job_id] = job
        return kept
    except Exception as e:
        print(f"[Server] Warning: could not load jobs from disk: {e}")
        return {}


def _persist_jobs_to_disk():
    """Save current jobs dict to disk."""
    try:
        jobs = state.get("jobs", {})
        JOBS_STORE_FILE.write_text(json.dumps(jobs, default=str))
    except Exception as e:
        print(f"[Server] Warning: could not persist jobs to disk: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Server] Initializing Disaster Response Pipeline…")
    state["pipeline"] = DisasterPipeline(load_crisis_model=True)
    state["analysis_lock"] = asyncio.Lock()
    state["jobs"] = _load_jobs_from_disk()
    if state["jobs"]:
        print(f"[Server] Restored {len(state['jobs'])} jobs from disk.")
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
    # Set up Satellite Grad-CAM if satellite model loaded
    if pipe._satellite_model is not None:
        try:
            state["sat_gradcam"] = SatelliteGradCAM(pipe._satellite_model)
            print("[Server] Satellite Grad-CAM ready.")
        except Exception as e:
            print(f"[Server] Satellite Grad-CAM init FAILED: {e}")
    state["openai_key"] = os.getenv("OPENAI_API_KEY", "")
    print(f"[Server] OpenAI key loaded: {'yes' if state['openai_key'] else 'NO KEY FOUND'}")
    print("[Server] Ready.")
    yield
    state.clear()


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(
    title="Unified Disaster Response API (Tri-Fusion)",
    description="IoT × CrisisMMD × Satellite(xBD) tri-fusion for real-time disaster assessment",
    version="2.0.0",
    lifespan=lifespan,
)

ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8001,http://127.0.0.1:8001").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─────────────────────────────────────────────
# Rate limiting (in-memory, per-client IP)
# ─────────────────────────────────────────────
_rate_limit_store: Dict[str, list] = defaultdict(list)
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "10"))       # requests
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds


def _check_rate_limit(request: Request):
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    timestamps = _rate_limit_store[client_ip]
    # Purge old entries
    _rate_limit_store[client_ip] = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW]
    if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_MAX:
        raise HTTPException(429, "Too many requests. Please try again later.")
    _rate_limit_store[client_ip].append(now)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
def _page_response(filename: str) -> FileResponse:
    html = PAGES_DIR / filename
    if not html.resolve().is_relative_to(PAGES_DIR.resolve()):
        raise HTTPException(403, "Access denied")
    if not html.exists():
        raise HTTPException(404, f"Page not found: {filename}")
    return FileResponse(str(html))


@app.get("/")
async def root():
    return RedirectResponse(url="/overview", status_code=302)


@app.get("/overview")
async def overview_page():
    return _page_response("overview.html")


@app.get("/analysis")
async def analysis_page():
    return _page_response("analysis.html")


@app.get("/incident")
async def incident_page():
    return _page_response("incident.html")


@app.get("/iot-monitor")
async def iot_monitor_page():
    return _page_response("iot-monitor.html")


@app.get("/reports")
async def reports_page():
    return _page_response("reports.html")


@app.get("/health")
async def health():
    pipe = state.get("pipeline")
    return {
        "status":           "ok",
        "pipeline":         pipe is not None,
        "crisis_model":     pipe is not None and pipe._crisis_model is not None,
        "iot_model":        "loaded" if pipe is not None else "not loaded",
        "satellite_model":  pipe is not None and pipe._satellite_model is not None,
        "tri_fusion":       pipe is not None,
        "background_jobs":  len(state.get("jobs", {})),
    }


def _build_sensor_kwargs(**values: Any) -> Dict[str, Any]:
    return {k: v for k, v in values.items() if v is not None}


def _run_analysis_sync(
    *,
    image_path: str,
    tweet: str,
    sensor_kwargs: Dict[str, Any],
    satellite_image_path: Optional[str] = None,
) -> Dict[str, Any]:
    if "pipeline" not in state:
        raise HTTPException(503, "Pipeline not ready")

    t0 = time.perf_counter()
    result = state["pipeline"].analyze(
        image_path=image_path,
        tweet=tweet,
        satellite_image_path=satellite_image_path,
        **sensor_kwargs,
    )

    # Crisis Grad-CAM
    gradcam_b64 = ""
    if "gradcam" in state:
        try:
            pipe = state["pipeline"]
            img_pil = Image.open(image_path).convert("RGB")
            img_t = pipe.IMG_TRANSFORM(img_pil).unsqueeze(0).to(pipe.device)
            enc = pipe._crisis_tokenizer(
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
            print(f"[Server] Crisis Grad-CAM skipped: {e}")

    # Satellite Grad-CAM
    sat_gradcam_b64 = ""
    if satellite_image_path and "sat_gradcam" in state:
        try:
            import cv2
            import numpy as np
            from XBD.xbd_model import preprocess_satellite_image, IMAGE_SIZE
            pipe = state["pipeline"]
            sat_tensor = preprocess_satellite_image(satellite_image_path, device=pipe.device)
            cam = state["sat_gradcam"].compute(sat_tensor, target="F_sat")
            orig_img = cv2.imread(satellite_image_path)
            if orig_img is not None:
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                orig_img = cv2.resize(orig_img, (IMAGE_SIZE, IMAGE_SIZE))
                overlay_img = overlay_heatmap(cam, orig_img)
                sat_gradcam_b64 = overlay_to_base64(overlay_img)
        except Exception as e:
            print(f"[Server] Satellite Grad-CAM skipped: {e}")

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

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

    return {
        "alert_level":      result.alert_level,
        "summary":          result.summary,
        "disaster_type":    result.confirmed_type,
        "priority":         result.priority,
        "fused_severity":   round(result.fused_severity, 4),
        "active_modalities": result.active_modalities,
        "modality_weights":  {k: round(v, 4) for k, v in result.modality_weights.items()},
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
            "category":      result.crisis_category,
            "confidence":    round(result.crisis_confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in result.crisis_probs.items()},
            "vision_weight": round(result.vision_weight, 4),
            "text_weight":   round(result.text_weight, 4),
        },
        "satellite": {
            "damage_class":  result.satellite_damage,
            "damage_probs":  {k: round(v, 4) for k, v in result.satellite_damage_probs.items()},
        } if result.satellite_damage else None,
        "fusion": {
            "priority_probs":    {k: round(v, 4) for k, v in result.priority_probs.items()},
            "type_probs":        {k: round(v, 4) for k, v in result.type_probs.items()},
            "population_impact": round(result.population_impact, 4),
            "resource_needs":    {k: round(v, 4) for k, v in result.resource_needs.items()},
        },
        "xai": {
            "crisis_gradcam_b64":    gradcam_b64,
            "satellite_gradcam_b64": sat_gradcam_b64,
            "summary":               xai_summary,
        },
        "inference_ms": elapsed_ms,
    }


async def _execute_analysis(
    *,
    image_path: str,
    tweet: str,
    sensor_kwargs: Dict[str, Any],
    satellite_image_path: Optional[str] = None,
) -> Dict[str, Any]:
    async with state["analysis_lock"]:
        return await asyncio.to_thread(
            _run_analysis_sync,
            image_path=image_path,
            tweet=tweet,
            sensor_kwargs=sensor_kwargs,
            satellite_image_path=satellite_image_path,
        )


def _update_job(job_id: str, **updates):
    """Update job dict fields and persist to disk."""
    jobs = state["jobs"]
    if job_id in jobs:
        jobs[job_id].update(updates)
        jobs[job_id]["updated_at"] = time.time()
        _persist_jobs_to_disk()


async def _run_background_job(
    *,
    job_id: str,
    image_path: str,
    tweet: str,
    sensor_kwargs: Dict[str, Any],
    satellite_image_path: Optional[str] = None,
) -> None:
    _update_job(job_id, status="running")

    try:
        result = await _execute_analysis(
            image_path=image_path,
            tweet=tweet,
            sensor_kwargs=sensor_kwargs,
            satellite_image_path=satellite_image_path,
        )
        _update_job(job_id, status="completed", completed_at=time.time(), result=result)
    except Exception as e:
        _update_job(job_id, status="failed", error=str(e))
    finally:
        for p in [image_path, satellite_image_path]:
            if p:
                try:
                    os.remove(p)
                except Exception as e:
                    print(f"[Server] Warning: failed to clean up {p}: {e}")


async def _save_upload_to_job_file(image: UploadFile) -> str:
    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    tmp_path = JOB_UPLOAD_DIR / f"{uuid4().hex}{suffix}"
    img_bytes = await image.read()
    with open(tmp_path, "wb") as f:
        f.write(img_bytes)
    return str(tmp_path)


@app.post("/analyze")
async def analyze(
    request:       Request,
    image:         UploadFile = File(...),
    tweet:         str        = Form(...),
    satellite_image: Optional[UploadFile] = File(None),
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
    _check_rate_limit(request)
    sensor_kwargs = _build_sensor_kwargs(
        lat=lat, lon=lon,
        max_temp=max_temp, min_temp=min_temp,
        avg_wind_speed=avg_wind_speed, precipitation=precipitation,
        month=month,
        wind_kts=wind_kts, pressure=pressure,
        depth=depth,
        elevation_m=elevation_m, distance_to_river_m=distance_to_river_m,
        rainfall_7d=rainfall_7d, monthly_rainfall=monthly_rainfall,
        drainage_index=drainage_index, ndvi=ndvi, ndwi=ndwi,
    )
    tmp_path = await _save_upload_to_job_file(image)
    sat_tmp_path = None
    if satellite_image is not None and satellite_image.filename:
        sat_tmp_path = await _save_upload_to_job_file(satellite_image)
    try:
        result = await _execute_analysis(
            image_path=tmp_path,
            tweet=tweet,
            sensor_kwargs=sensor_kwargs,
            satellite_image_path=sat_tmp_path,
        )
    finally:
        for p in [tmp_path, sat_tmp_path]:
            if p:
                try:
                    os.remove(p)
                except Exception as e:
                    print(f"[Server] Warning: failed to clean up {p}: {e}")

    return JSONResponse(result)


@app.post("/satellite/predict")
async def satellite_predict(
    request: Request,
    satellite_image: UploadFile = File(...),
):
    """Standalone satellite damage prediction (no fusion)."""
    _check_rate_limit(request)
    if "pipeline" not in state:
        raise HTTPException(503, "Pipeline not ready")
    pipe = state["pipeline"]
    if pipe._satellite_predictor is None:
        raise HTTPException(503, "Satellite model not loaded")

    sat_tmp = await _save_upload_to_job_file(satellite_image)
    try:
        result = pipe._satellite_predictor.predict(sat_tmp)
        # Generate Grad-CAM
        sat_gradcam_b64 = ""
        if "sat_gradcam" in state:
            try:
                import cv2
                import numpy as np
                from XBD.xbd_model import preprocess_satellite_image, IMAGE_SIZE
                sat_tensor = preprocess_satellite_image(sat_tmp, device=pipe.device)
                cam = state["sat_gradcam"].compute(sat_tensor, target="F_sat")
                orig_img = cv2.imread(sat_tmp)
                if orig_img is not None:
                    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                    orig_img = cv2.resize(orig_img, (IMAGE_SIZE, IMAGE_SIZE))
                    overlay_img = overlay_heatmap(cam, orig_img)
                    sat_gradcam_b64 = overlay_to_base64(overlay_img)
            except Exception as e:
                print(f"[Server] Satellite Grad-CAM skipped: {e}")
        return JSONResponse({
            "damage_class": result["damage_class"],
            "damage_probs": {k: round(v, 4) for k, v in result["damage_probs"].items()},
            "xai": {"satellite_gradcam_b64": sat_gradcam_b64},
        })
    finally:
        try:
            os.remove(sat_tmp)
        except Exception:
            pass


@app.post("/analysis/jobs")
async def create_analysis_job(
    request:       Request,
    image:         UploadFile = File(...),
    tweet:         str        = Form(...),
    satellite_image: Optional[UploadFile] = File(None),
    lat:           Optional[float] = Form(None),
    lon:           Optional[float] = Form(None),
    max_temp:          Optional[float] = Form(None),
    min_temp:          Optional[float] = Form(None),
    avg_wind_speed:    Optional[float] = Form(None),
    precipitation:     Optional[float] = Form(None),
    month:             Optional[int]   = Form(None),
    wind_kts:      Optional[float] = Form(None),
    pressure:      Optional[float] = Form(None),
    depth:         Optional[float] = Form(None),
    elevation_m:         Optional[float] = Form(None),
    distance_to_river_m: Optional[float] = Form(None),
    rainfall_7d:         Optional[float] = Form(None),
    monthly_rainfall:    Optional[float] = Form(None),
    drainage_index:      Optional[float] = Form(None),
    ndvi:                Optional[float] = Form(None),
    ndwi:                Optional[float] = Form(None),
):
    _check_rate_limit(request)
    if "pipeline" not in state:
        raise HTTPException(503, "Pipeline not ready")

    sensor_kwargs = _build_sensor_kwargs(
        lat=lat, lon=lon,
        max_temp=max_temp, min_temp=min_temp,
        avg_wind_speed=avg_wind_speed, precipitation=precipitation,
        month=month,
        wind_kts=wind_kts, pressure=pressure,
        depth=depth,
        elevation_m=elevation_m, distance_to_river_m=distance_to_river_m,
        rainfall_7d=rainfall_7d, monthly_rainfall=monthly_rainfall,
        drainage_index=drainage_index, ndvi=ndvi, ndwi=ndwi,
    )
    image_path = await _save_upload_to_job_file(image)
    sat_image_path = None
    if satellite_image is not None and satellite_image.filename:
        sat_image_path = await _save_upload_to_job_file(satellite_image)
    created_at = time.time()
    job_id = uuid4().hex
    state["jobs"][job_id] = {
        "job_id": job_id,
        "status": "queued",
        "created_at": created_at,
        "updated_at": created_at,
        "result": None,
        "error": None,
    }
    _persist_jobs_to_disk()

    asyncio.create_task(
        _run_background_job(
            job_id=job_id,
            image_path=image_path,
            tweet=tweet,
            sensor_kwargs=sensor_kwargs,
            satellite_image_path=sat_image_path,
        )
    )

    return JSONResponse({
        "job_id": job_id,
        "status": "queued",
        "created_at": created_at,
        "updated_at": created_at,
    })


@app.get("/analysis/jobs/{job_id}")
async def get_analysis_job(job_id: str):
    job = state.get("jobs", {}).get(job_id)
    if not job:
        raise HTTPException(404, "Analysis job not found")
    return JSONResponse(job)


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
