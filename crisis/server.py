"""
CrisisMMD Adaptive Multimodal Classifier — FastAPI server
"""
from __future__ import annotations

import io
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    XLMRobertaTokenizer,
    XLMRobertaModel,
)
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "best_adaptive_model.pth"
STATIC_DIR = BASE_DIR / "static"

# ---------------------------------------------------------------------------
# Label classes (sklearn LabelEncoder sorts alphabetically)
# ---------------------------------------------------------------------------
CLASSES = [
    "affected_individuals",
    "infrastructure_and_utility_damage",
    "not_humanitarian",
    "other_relevant_information",
    "rescue_volunteering_or_donation_effort",
]

# ---------------------------------------------------------------------------
# Model definition (must match training code exactly)
# ---------------------------------------------------------------------------

class ConfidenceEstimator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


class AdaptiveFusionClassifier(nn.Module):
    def __init__(self, blip_model, xlm_model, hidden_dim: int = 512, n_classes: int = 8):
        super().__init__()
        self.vision_encoder = blip_model.vision_model
        self.text_encoder   = xlm_model

        self.vision_dim = 768
        self.text_dim   = 768

        self.vision_confidence = ConfidenceEstimator(self.vision_dim)
        self.text_confidence   = ConfidenceEstimator(self.text_dim)

        self.vision_proj = nn.Sequential(
            nn.Linear(self.vision_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, dropout=0.1, batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, images, input_ids, attention_mask, return_attention: bool = False):
        vision_outputs  = self.vision_encoder(images)
        vision_features = vision_outputs.last_hidden_state[:, 0, :]

        text_outputs  = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]

        vision_conf = self.vision_confidence(vision_features)
        text_conf   = self.text_confidence(text_features)

        total_conf    = vision_conf + text_conf + 1e-8
        vision_weight = vision_conf / total_conf
        text_weight   = text_conf   / total_conf

        vision_proj     = self.vision_proj(vision_features)
        text_proj       = self.text_proj(text_features)
        vision_weighted = vision_weight * vision_proj
        text_weighted   = text_weight   * text_proj

        vision_attended, vision_attn = self.cross_attention(
            vision_weighted.unsqueeze(1), text_weighted.unsqueeze(1), text_weighted.unsqueeze(1)
        )
        text_attended, text_attn = self.cross_attention(
            text_weighted.unsqueeze(1), vision_weighted.unsqueeze(1), vision_weighted.unsqueeze(1)
        )

        vision_attended = vision_attended.squeeze(1)
        text_attended   = text_attended.squeeze(1)

        logits = self.classifier(torch.cat([vision_attended, text_attended], dim=1))

        if return_attention:
            return logits, {
                "vision_conf":   vision_conf.squeeze(-1),
                "text_conf":     text_conf.squeeze(-1),
                "vision_weight": vision_weight.squeeze(-1),
                "text_weight":   text_weight.squeeze(-1),
            }
        return logits


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
state: dict = {}

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup."""
    print("Loading backbone models…")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state["device"] = device

    BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")  # warms up cache
    blip_model     = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    xlm_tokenizer  = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    xlm_model      = XLMRobertaModel.from_pretrained("xlm-roberta-base")

    state["tokenizer"] = xlm_tokenizer

    print(f"Building AdaptiveFusionClassifier (n_classes={len(CLASSES)})…")
    model = AdaptiveFusionClassifier(
        blip_model=blip_model,
        xlm_model=xlm_model,
        hidden_dim=512,
        n_classes=len(CLASSES),
    )

    print(f"Loading checkpoint from {MODEL_PATH} …")
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    # checkpoint may be full state_dict or wrapped
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    state["model"] = model

    print(f"Model ready on {device}.")
    yield
    state.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="CrisisMMD Dashboard", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    from fastapi.responses import FileResponse
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    tweet: str = Form(...),
):
    if "model" not in state:
        raise HTTPException(503, "Model not loaded yet, please wait.")

    # --- load image ---
    try:
        img_bytes = await image.read()
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_tensor = IMG_TRANSFORM(pil_img).unsqueeze(0).to(state["device"])
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    # --- tokenise text ---
    tokenizer = state["tokenizer"]
    enc = tokenizer(
        tweet,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = enc["input_ids"].to(state["device"])
    attention_mask = enc["attention_mask"].to(state["device"])

    # --- inference ---
    t0 = time.perf_counter()
    with torch.no_grad():
        logits, attn_info = state["model"](
            img_tensor, input_ids, attention_mask, return_attention=True
        )
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    probs      = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    pred_idx   = int(np.argmax(probs))
    pred_class = CLASSES[pred_idx]
    confidence = float(probs[pred_idx])

    vision_weight = float(attn_info["vision_weight"].cpu().item())
    text_weight   = float(attn_info["text_weight"].cpu().item())
    vision_conf   = float(attn_info["vision_conf"].cpu().item())
    text_conf     = float(attn_info["text_conf"].cpu().item())

    return JSONResponse({
        "prediction":     pred_class,
        "confidence":     confidence,
        "probabilities":  {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))},
        "modality": {
            "vision_weight": vision_weight,
            "text_weight":   text_weight,
            "vision_conf":   vision_conf,
            "text_conf":     text_conf,
        },
        "inference_ms": elapsed_ms,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
