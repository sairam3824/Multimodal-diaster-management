"""
XAI Module — Gradient-weighted Attention Rollout + OpenAI Explanation
======================================================================
Two independent components:

1. GradCAMViT
   Combines ViT attention maps with gradient signals to produce accurate
   heatmaps showing what drives the model's classification decision.

   Method: Gradient-weighted Attention Rollout
   - Forward pass with output_attentions=True to get attention matrices
   - Backward pass to get gradients on the attention weights
   - Weight attention by gradient importance (positive contributions only)
   - Roll up through last N layers to produce a CLS-to-patch attention map
   - Overlay as heatmap on the original image

2. generate_openai_summary(result_dict, api_key)
   Sends the full assessment to GPT-4o for a structured field briefing.
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
# 1. Gradient-weighted Attention for BLIP ViT
# ─────────────────────────────────────────────

class GradCAMViT:
    """
    Gradient-weighted Attention Rollout for BLIP ViT.

    Pure attention shows "where the model looks" but not "what matters for
    the prediction". By weighting attention with gradients, we highlight
    regions that both receive attention AND influence the final classification.

    BLIP ViT: 224×224 → 14×14 patches (196 + 1 CLS), 12 encoder layers.
    """

    GRID_SIZE = 14

    def __init__(self, crisis_model):
        self.model = crisis_model

    def compute(
        self,
        img_tensor: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        original_pil: Image.Image,
    ) -> str:
        try:
            result = self._compute_gradient_attention(
                img_tensor, input_ids, attention_mask, original_pil
            )
            if result:
                return result
            # Fallback: pure attention rollout (no gradients)
            result = self._compute_attention_only(img_tensor, original_pil)
            if result:
                return result
            # Final fallback: input gradient saliency
            return self._compute_input_gradient(
                img_tensor, input_ids, attention_mask, original_pil
            )
        except Exception as e:
            import traceback
            print(f"[GradCAM] WARNING: {e}")
            traceback.print_exc()
            return ""

    def _compute_gradient_attention(self, img_tensor, input_ids, attention_mask, original_pil):
        """
        Gradient-weighted Attention Rollout.

        Uses backward hooks to capture gradients on attention weight tensors,
        then weights attention by clamped-positive gradients to highlight
        regions that both receive attention AND drive the classification.
        """
        vision_enc = getattr(self.model, 'vision_encoder', None)
        if vision_enc is None:
            return ""

        self.model.zero_grad()

        # Storage for attention gradients captured via hooks
        attn_grads = {}
        hook_handles = []

        def make_grad_hook(layer_idx):
            def hook(grad):
                attn_grads[layer_idx] = grad.detach()
            return hook

        with torch.enable_grad():
            # Forward: get attention maps from ViT
            vision_outputs = vision_enc(img_tensor, output_attentions=True)
            attentions = getattr(vision_outputs, 'attentions', None)
            if attentions is None or len(attentions) == 0:
                return ""

            # Register gradient hooks on attention tensors
            for i, attn_tensor in enumerate(attentions):
                if attn_tensor.requires_grad:
                    h = attn_tensor.register_hook(make_grad_hook(i))
                    hook_handles.append(h)

            # Complete the forward pass through the full model
            cls_feat = vision_outputs.last_hidden_state[:, 0, :]
            vis_proj = self.model.vision_proj(cls_feat)

            txt_feat = self.model.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state[:, 0, :]
            txt_proj = self.model.text_proj(txt_feat)

            vis_conf = self.model.vision_confidence(cls_feat)
            txt_conf = self.model.text_confidence(txt_feat)
            total = vis_conf + txt_conf + 1e-8
            vis_w = vis_conf / total
            txt_w = txt_conf / total

            weighted_vis = vis_proj * vis_w
            weighted_txt = txt_proj * txt_w

            q = weighted_vis.unsqueeze(1)
            k = weighted_txt.unsqueeze(1)
            attended_vis, _ = self.model.cross_attention(q, k, k)
            attended_vis = attended_vis.squeeze(1)

            fused = torch.cat([attended_vis, weighted_txt], dim=1)
            logits = self.model.classifier(fused)

            pred_class = int(logits.argmax(dim=-1).item())
            score = logits[0, pred_class]
            score.backward()

        # Clean up hooks
        for h in hook_handles:
            h.remove()

        # Build gradient-weighted attention rollout from last 4 layers
        n_layers = len(attentions)
        start_layer = max(0, n_layers - 4)

        rollout = None
        for i in range(start_layer, n_layers):
            attn = attentions[i][0].detach()  # [num_heads, 197, 197]

            # Weight by positive gradients if available
            if i in attn_grads:
                grad = torch.clamp(attn_grads[i][0], min=0)
                attn = attn * grad

            head_avg = attn.mean(dim=0)  # [197, 197]

            # Add residual connection
            head_avg = 0.5 * head_avg + 0.5 * torch.eye(
                head_avg.size(0), device=head_avg.device
            )
            row_sum = torch.clamp(head_avg.sum(dim=-1, keepdim=True), min=1e-8)
            head_avg = head_avg / row_sum

            if rollout is None:
                rollout = head_avg
            else:
                rollout = torch.matmul(rollout, head_avg)

        cls_attn = rollout[0, 1:].cpu().numpy()  # [196]
        return self._postprocess_and_overlay(cls_attn, original_pil)

    def _compute_attention_only(self, img_tensor, original_pil):
        """Pure attention rollout fallback (no gradients)."""
        vision_enc = getattr(self.model, 'vision_encoder', None)
        if vision_enc is None:
            return ""

        with torch.no_grad():
            vision_outputs = vision_enc(img_tensor, output_attentions=True)
            attentions = getattr(vision_outputs, 'attentions', None)
            if attentions is None or len(attentions) == 0:
                return ""

            n_layers = len(attentions)
            deep_layers = attentions[max(0, n_layers - 4):]

            rollout = None
            for layer_attn in deep_layers:
                head_avg = layer_attn[0].mean(dim=0)
                head_avg = 0.5 * head_avg + 0.5 * torch.eye(
                    head_avg.size(0), device=head_avg.device
                )
                head_avg = head_avg / head_avg.sum(dim=-1, keepdim=True)
                if rollout is None:
                    rollout = head_avg
                else:
                    rollout = torch.matmul(rollout, head_avg)

            cls_attn = rollout[0, 1:].cpu().numpy()

        return self._postprocess_and_overlay(cls_attn, original_pil)

    def _compute_input_gradient(self, img_tensor, input_ids, attention_mask, original_pil):
        """Final fallback: pixel-level gradient saliency."""
        self.model.zero_grad()

        with torch.enable_grad():
            img_g = img_tensor.detach().clone().requires_grad_(True)
            logits, _ = self.model(img_g, input_ids, attention_mask, return_attention=True)
            pred_class = int(logits.argmax(dim=-1).item())
            logits[0, pred_class].backward()

        if img_g.grad is None:
            return ""

        saliency = img_g.grad[0].abs().max(dim=0)[0].cpu().numpy()
        saliency = cv2.GaussianBlur(saliency, (21, 21), 0)
        s_min, s_max = saliency.min(), saliency.max()
        if s_max - s_min < 1e-8:
            return ""
        saliency = (saliency - s_min) / (s_max - s_min)
        return self._overlay_heatmap(saliency, original_pil)

    def _postprocess_and_overlay(self, cls_attn: np.ndarray, original_pil: Image.Image) -> str:
        """Normalize, threshold, reshape, smooth, and overlay."""
        a_min, a_max = cls_attn.min(), cls_attn.max()
        if a_max - a_min < 1e-8:
            return ""
        cls_attn = (cls_attn - a_min) / (a_max - a_min)

        # Suppress bottom 40% to focus on strong activations
        threshold = np.percentile(cls_attn, 40)
        cls_attn = np.clip(cls_attn - threshold, 0, 1)
        c_max = cls_attn.max()
        if c_max > 1e-8:
            cls_attn = cls_attn / c_max

        # Reshape [196] → [14, 14]
        attn_map = cls_attn.reshape(self.GRID_SIZE, self.GRID_SIZE)

        # Upsample to 224×224
        attn_map = cv2.resize(attn_map, (224, 224), interpolation=cv2.INTER_CUBIC)

        # Smooth to reduce grid artifacts
        attn_map = cv2.GaussianBlur(attn_map, (15, 15), 0)

        # Final normalize
        a_min, a_max = attn_map.min(), attn_map.max()
        if a_max - a_min > 1e-8:
            attn_map = (attn_map - a_min) / (a_max - a_min)

        return self._overlay_heatmap(attn_map, original_pil)

    def _overlay_heatmap(self, cam: np.ndarray, original_pil: Image.Image) -> str:
        """Overlay a [224, 224] heatmap on the original image → base64 PNG."""
        orig = original_pil.convert("RGB").resize((224, 224))
        orig_np = np.array(orig, dtype=np.float32) / 255.0

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        overlay = 0.45 * heatmap + 0.55 * orig_np
        overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

        buf = io.BytesIO()
        Image.fromarray(overlay).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


# ─────────────────────────────────────────────
# 2. OpenAI Natural Language Explanation
# ─────────────────────────────────────────────

def generate_openai_summary(result: dict, api_key: Optional[str] = None) -> str:
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
  Sensor weights : {iot.get('sensor_weights', {{}})}

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

**KEY RISKS**: Bullet list of the top 3 specific risks (use actual numbers from the data)

**RECOMMENDED ACTIONS**: Numbered list of 3-4 specific, actionable steps for first responders.

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
