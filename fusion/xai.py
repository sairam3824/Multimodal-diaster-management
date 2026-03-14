"""
XAI Module — Grad-CAM + OpenAI Natural Language Explanation
=============================================================
Two independent components:

1. GradCAMViT
   Computes Grad-CAM on BLIP's ViT vision encoder inside AdaptiveFusionClassifier.
   crisis_model.vision_encoder = blip_model.vision_model (BlipVisionModel)
   Target layer: vision_encoder.encoder.layers[-1]  → [B, 197, 768] (196 patch + 1 CLS)
   Returns a base64-encoded PNG of the heatmap overlaid on the original image.

2. generate_openai_summary(result_dict, api_key)
   Sends the full assessment to GPT-4o.
   Returns a structured, human-readable explanation for field responders.
"""

from __future__ import annotations
import base64, io, os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image


# ─────────────────────────────────────────────
# 1. Grad-CAM for BLIP ViT
# ─────────────────────────────────────────────

class GradCAMViT:
    """
    Hook-based Grad-CAM for BLIP's Vision Transformer encoder
    inside AdaptiveFusionClassifier (crisis_model.vision_encoder).

    BLIP ViT patch layout:
      Input  : 224 × 224
      Patches: 16 × 16 → 14 × 14 spatial grid = 196 patch tokens + 1 CLS
      Last encoder layer output: [B, 197, 768]

    We skip CLS, reshape to [14, 14], weight by gradients,
    then upsample back to 224 × 224 and overlay on the original image.
    """

    def __init__(self, crisis_model):
        self.model = crisis_model

    def compute(
        self,
        img_tensor: torch.Tensor,      # [1, 3, 224, 224]
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        original_pil: Image.Image,
    ) -> str:
        """
        Input Gradient Saliency Map overlaid on the original image.

        Why not patch-token Grad-CAM:
          AdaptiveFusionClassifier only uses the CLS token (index 0) from BLIP ViT.
          Gradients through the 196 patch tokens are zero → flat CAM.

        Instead we compute d(score)/d(pixel) directly.
        The gradient flows: score → CLS token → self-attention → ALL patch embeddings
        → patch projection → input pixels.  This gives a 224×224 saliency map.
        """
        try:
            return self._compute_inner(img_tensor, input_ids, attention_mask, original_pil)
        except Exception as e:
            import traceback
            print(f"[GradCAM] WARNING: {e}")
            traceback.print_exc()
            return ""

    def _compute_inner(self, img_tensor, input_ids, attention_mask, original_pil):
        self.model.zero_grad()

        with torch.enable_grad():
            img_g = img_tensor.detach().clone().requires_grad_(True)
            logits, _ = self.model(img_g, input_ids, attention_mask, return_attention=True)
            pred_class = int(logits.argmax(dim=-1).item())
            score = logits[0, pred_class]
            score.backward()

        if img_g.grad is None:
            print("[GradCAM] No gradient on input image.")
            return ""

        # img_g.grad: [1, 3, 224, 224]
        # Take max absolute gradient across RGB channels → [224, 224]
        saliency = img_g.grad[0].abs().max(dim=0)[0].cpu().numpy()

        # Smooth with small Gaussian to reduce pixel noise
        saliency = cv2.GaussianBlur(saliency, (11, 11), 0)

        # Normalise
        s_min, s_max = saliency.min(), saliency.max()
        if s_max - s_min < 1e-8:
            print("[GradCAM] Saliency map is flat. Skipping.")
            return ""
        saliency = (saliency - s_min) / (s_max - s_min)

        # Convert original image to numpy 224×224
        orig    = original_pil.convert("RGB").resize((224, 224))
        orig_np = np.array(orig, dtype=np.float32) / 255.0

        # Apply JET colormap and blend
        heatmap = cv2.applyColorMap(np.uint8(255 * saliency), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        overlay = 0.45 * heatmap + 0.55 * orig_np
        overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

        # Encode to base64 PNG
        buf = io.BytesIO()
        Image.fromarray(overlay).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


# ─────────────────────────────────────────────
# 2. OpenAI Natural Language Explanation
# ─────────────────────────────────────────────

def generate_openai_summary(result: dict, api_key: Optional[str] = None) -> str:
    """
    Sends the full assessment dict to GPT-4o and returns a structured,
    human-readable explanation for field responders.

    result: the JSON dict returned by /analyze
    """
    key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not key or key == "your_key_here":
        return ""

    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)

        iot    = result.get("iot", {})
        crisis = result.get("crisis", {})
        fusion = result.get("fusion", {})
        res    = fusion.get("resource_needs", {})

        prompt = f"""You are a disaster response AI assistant.
Analyze the following automated sensor and social media assessment and produce a concise,
actionable briefing for field responders. Be direct, factual, and specific.

=== ASSESSMENT DATA ===
Alert Level      : {result.get('alert_level')}
Disaster Type    : {result.get('disaster_type')}
Priority         : {result.get('priority')}
Fused Severity   : {result.get('fused_severity', 0):.2f} / 1.0

IoT Sensor Analysis:
  Detected type  : {iot.get('type')}
  Fire prob      : {iot.get('fire_prob', 0):.1%}
  Storm category : {iot.get('storm_cat', 0) * 5:.1f} / 5
  Earthquake     : M{iot.get('eq_magnitude', 0) * 9:.1f}
  Flood risk     : {iot.get('flood_risk', 0) * 100:.0f} / 100
  Casualty risk  : {iot.get('casualty_risk', 0):.1%}
  Sensor weights : {iot.get('sensor_weights', {})}

Social Media (Crisis Model):
  Category       : {crisis.get('category')}
  Confidence     : {crisis.get('confidence', 0):.1%}
  Vision weight  : {crisis.get('vision_weight', 0):.2f}
  Text weight    : {crisis.get('text_weight', 0):.2f}

Fusion Output:
  Population impact : {fusion.get('population_impact', 0):.1%}
  Resource needs    : water={res.get('water', 0):.0%}  medical={res.get('medical', 0):.0%}  rescue={res.get('rescue', 0):.0%}  shelter={res.get('shelter', 0):.0%}

=== YOUR TASK ===
Write a structured field briefing with exactly these 4 sections:

**SITUATION**: 2-3 sentences explaining what is happening, what the sensors detected, and what social media confirms.

**KEY RISKS**: Bullet list of the top 3 specific risks (use actual numbers from the data — e.g. casualty risk %, flood score, etc.)

**RECOMMENDED ACTIONS**: Numbered list of 3-4 specific, actionable steps for first responders (deploy X, evacuate Y, prioritize Z).

**WHY THIS ALERT**: 1-2 sentences explaining which data signals drove this assessment and why the alert level was set to {result.get('alert_level')}.

Keep total response under 220 words. Use plain language — no jargon."""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[XAI] OpenAI error: {e}")
        return ""
