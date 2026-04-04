"""
Satellite Explainability — Grad-CAM for DeepLabV3Plus_xBD
==========================================================
Generates Grad-CAM heatmaps for the xBD satellite damage model.

The xBD model uses a ResNet101 encoder. We hook into the final
encoder block (layer4) to capture activations and gradients, then
compute Grad-CAM w.r.t. either:
  - The F_sat embedding (which damage features does the model see?)
  - A specific damage class prediction

Outputs:
  - Raw heatmap (numpy float32, 0-1)
  - Overlay image (RGB numpy uint8)
  - Base64-encoded overlay PNG
  - Metadata JSON

If the .pkl model is not directly Grad-CAM compatible (e.g., pickle
load fails), a fallback using input-gradient saliency is provided.

Limitation: If the model cannot be loaded at all, a surrogate
lightweight CNN can be trained for approximate explainability.
This is documented but not auto-built.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from XBD.xbd_model import (
    DeepLabV3Plus_xBD,
    XBD_DAMAGE_CLASSES,
    IMAGE_SIZE,
    preprocess_satellite_image,
    preprocess_satellite_pil,
    extract_satellite_features,
)


OUTPUT_DIR = Path(os.path.dirname(__file__)) / ".." / "outputs" / "satellite_xai"


class SatelliteGradCAM:
    """
    Grad-CAM for the xBD DeepLabV3Plus model.

    Hooks into the ResNet101 encoder's final block to compute
    gradient-weighted class activation maps.
    """

    def __init__(self, model: DeepLabV3Plus_xBD):
        self.model = model
        self.device = next(model.parameters()).device
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._hooks: list = []
        self._register_hooks()

    def _register_hooks(self):
        """Hook into the last encoder block (features[-1])."""
        # In segmentation-models-pytorch, the encoder is model.base_model.encoder
        # ResNet101 layers: layer1, layer2, layer3, layer4
        encoder = self.model.base_model.encoder
        target_layer = encoder.layer4  # Final residual block

        def fwd_hook(module, input, output):
            self._activations = output.detach()

        def bwd_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self._hooks.append(target_layer.register_forward_hook(fwd_hook))
        self._hooks.append(target_layer.register_full_backward_hook(bwd_hook))

    def cleanup(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def compute(
        self,
        image_tensor: torch.Tensor,
        target: str = "F_sat",
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap.

        Args:
            image_tensor: [1, 3, 512, 512] preprocessed satellite image
            target: "F_sat" to explain the satellite embedding,
                    "damage_class" to explain a specific damage prediction
            target_class: damage class index (0-3) if target="damage_class".
                          If None, uses the predicted class.

        Returns:
            cam: [512, 512] float32 heatmap, values 0-1
        """
        self.model.zero_grad()
        image_tensor = image_tensor.to(self.device).requires_grad_(True)

        # Forward pass with features
        outputs = self.model(image_tensor, return_features=True)

        if target == "F_sat":
            # Explain: what drives the F_sat embedding magnitude?
            score = outputs["F_sat"].norm(dim=1).sum()
        elif target == "damage_class":
            # Explain: what drives prediction of a specific damage class?
            p_x = outputs["P_x"]  # [B, 4, H, W]
            if target_class is None:
                target_class = p_x.mean(dim=(2, 3)).argmax(dim=1).item()
            # Sum of probabilities for target class across spatial dims
            score = p_x[:, target_class, :, :].sum()
        else:
            raise ValueError(f"Unknown target: {target}")

        score.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")

        # Grad-CAM: global-average-pool gradients -> channel weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # [B, 1, H', W']
        cam = F.relu(cam)

        # Upsample to input resolution
        cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to 0-1
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def compute_input_saliency(
        self, image_tensor: torch.Tensor,
    ) -> np.ndarray:
        """
        Fallback: input-gradient saliency map.
        Works even if encoder hooks fail.
        """
        self.model.zero_grad()
        img = image_tensor.to(self.device).clone().requires_grad_(True)

        outputs = self.model(img, return_features=True)
        score = outputs["F_sat"].norm(dim=1).sum()
        score.backward()

        if img.grad is None:
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

        saliency = img.grad[0].abs().max(dim=0)[0].cpu().numpy()
        saliency = cv2.GaussianBlur(saliency, (21, 21), 0)
        s_min, s_max = saliency.min(), saliency.max()
        if s_max - s_min > 1e-8:
            saliency = (saliency - s_min) / (s_max - s_min)
        return saliency


def overlay_heatmap(
    cam: np.ndarray,
    original_image: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Overlay a heatmap on the original image.

    Args:
        cam: [H, W] float32 heatmap (0-1)
        original_image: [H, W, 3] RGB uint8 image
        alpha: heatmap opacity

    Returns:
        overlay: [H, W, 3] RGB uint8 image
    """
    h, w = original_image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    orig_f = original_image.astype(np.float32) / 255.0
    result = alpha * heatmap + (1 - alpha) * orig_f
    return np.clip(result * 255, 0, 255).astype(np.uint8)


def overlay_to_base64(overlay: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def save_satellite_xai(
    image_path: str,
    model: DeepLabV3Plus_xBD,
    output_dir: Optional[str] = None,
    target: str = "F_sat",
    target_class: Optional[int] = None,
) -> Dict:
    """
    Full satellite explainability pipeline:
    generate Grad-CAM, overlay, and save all artifacts.

    Returns metadata dict.
    """
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and preprocess
    device = next(model.parameters()).device
    tensor = preprocess_satellite_image(image_path, device=device)

    # Load original image for overlay
    orig_img = cv2.imread(str(image_path))
    if orig_img is not None:
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        orig_img = cv2.resize(orig_img, (IMAGE_SIZE, IMAGE_SIZE))
    else:
        orig_img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

    # Compute Grad-CAM
    gradcam = SatelliteGradCAM(model)
    try:
        cam = gradcam.compute(tensor, target=target, target_class=target_class)
        method = "grad_cam"
    except Exception as e:
        print(f"[SatXAI] Grad-CAM failed ({e}), falling back to input saliency")
        cam = gradcam.compute_input_saliency(tensor)
        method = "input_saliency"
    finally:
        gradcam.cleanup()

    # Get damage prediction
    feats = extract_satellite_features(model, tensor.to(device))
    damage_idx = int(feats["damage_pred"].item())
    damage_probs = feats["damage_probs_summary"].squeeze(0).cpu().numpy()

    # Generate overlay
    overlay_img = overlay_heatmap(cam, orig_img)

    # Generate filenames
    stem = Path(image_path).stem
    timestamp = int(time.time())
    prefix = f"{stem}_{timestamp}"

    # Save artifacts
    heatmap_path = out_dir / f"{prefix}_heatmap.npy"
    overlay_path = out_dir / f"{prefix}_overlay.png"
    metadata_path = out_dir / f"{prefix}_metadata.json"

    np.save(str(heatmap_path), cam)
    Image.fromarray(overlay_img).save(str(overlay_path))

    metadata = {
        "source_image": str(image_path),
        "method": method,
        "target": target,
        "target_class": target_class if target == "damage_class" else None,
        "predicted_damage": XBD_DAMAGE_CLASSES[damage_idx],
        "damage_probabilities": {
            XBD_DAMAGE_CLASSES[i]: float(damage_probs[i]) for i in range(4)
        },
        "heatmap_file": str(heatmap_path),
        "overlay_file": str(overlay_path),
        "timestamp": timestamp,
        "image_size": IMAGE_SIZE,
        "model_encoder": "ResNet101",
        "limitations": (
            "Grad-CAM computed on encoder layer4 features. "
            "If .pkl model was loaded via redirect unpickler, architecture "
            "is identical to training. If model structure differs, input "
            "saliency fallback is used (less spatially precise)."
        ),
    }
    with open(str(metadata_path), "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def visualize_topk_predictions(
    model: DeepLabV3Plus_xBD,
    image_paths: List[str],
    output_dir: Optional[str] = None,
    k: int = 5,
) -> List[Dict]:
    """
    Generate Grad-CAM overlays for the top-k satellite predictions
    (ranked by maximum damage probability).
    """
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    device = next(model.parameters()).device

    scored = []
    for path in image_paths:
        try:
            tensor = preprocess_satellite_image(path, device=device)
            feats = extract_satellite_features(model, tensor)
            max_damage_prob = float(feats["damage_probs_summary"][0, 1:].max().item())
            scored.append((path, max_damage_prob))
        except Exception as e:
            print(f"[SatXAI] Skipping {path}: {e}")

    scored.sort(key=lambda x: x[1], reverse=True)
    top_k = scored[:k]

    results = []
    for path, score in top_k:
        meta = save_satellite_xai(path, model, output_dir=str(out_dir))
        meta["damage_score"] = score
        results.append(meta)
        print(f"  {Path(path).name}: {meta['predicted_damage']} (max_damage_prob={score:.3f})")

    return results


# ──────────────────────────────────────────────
# Modality Contribution Report
# ──────────────────────────────────────────────

def modality_contribution_report(
    tri_fusion_model,
    crisis_embedding: torch.Tensor,
    iot_embedding: Optional[torch.Tensor] = None,
    satellite_embedding: Optional[torch.Tensor] = None,
) -> Dict[str, Dict]:
    """
    Estimate modality contributions via ablation-at-inference:
    1. Predict with all available modalities
    2. Drop one modality at a time
    3. Report confidence shift

    Returns dict with per-modality contribution estimates.
    """
    tri_fusion_model.eval()
    device = crisis_embedding.device

    def _predict(crisis, iot, sat):
        with torch.no_grad():
            out = tri_fusion_model(crisis, iot, sat)
        severity = float(out.severity_score.item())
        priority_conf = float(torchF.softmax(out.priority_logits, dim=-1).max().item())
        disaster_conf = float(torchF.softmax(out.disaster_logits, dim=-1).max().item())
        return {
            "severity": severity,
            "priority_confidence": priority_conf,
            "disaster_confidence": disaster_conf,
        }

    import torch.nn.functional as torchF

    # Full prediction
    full = _predict(crisis_embedding, iot_embedding, satellite_embedding)

    report = {"full_prediction": full, "modality_contributions": {}}

    # Crisis is always present, so we measure its contribution
    # by comparing full vs a degraded crisis (zeros)
    # But crisis is mandatory — we can still measure how much it matters
    # by comparing with and without optional modalities

    # Drop IoT
    if iot_embedding is not None:
        no_iot = _predict(crisis_embedding, None, satellite_embedding)
        report["modality_contributions"]["iot"] = {
            "severity_shift": full["severity"] - no_iot["severity"],
            "priority_confidence_shift": full["priority_confidence"] - no_iot["priority_confidence"],
            "disaster_confidence_shift": full["disaster_confidence"] - no_iot["disaster_confidence"],
        }
    else:
        report["modality_contributions"]["iot"] = {"status": "not_provided"}

    # Drop Satellite
    if satellite_embedding is not None:
        no_sat = _predict(crisis_embedding, iot_embedding, None)
        report["modality_contributions"]["satellite"] = {
            "severity_shift": full["severity"] - no_sat["severity"],
            "priority_confidence_shift": full["priority_confidence"] - no_sat["priority_confidence"],
            "disaster_confidence_shift": full["disaster_confidence"] - no_sat["disaster_confidence"],
        }
    else:
        report["modality_contributions"]["satellite"] = {"status": "not_provided"}

    # Crisis contribution: drop both optional modalities
    crisis_only = _predict(crisis_embedding, None, None)
    report["modality_contributions"]["crisis"] = {
        "severity_baseline": crisis_only["severity"],
        "priority_confidence_baseline": crisis_only["priority_confidence"],
        "disaster_confidence_baseline": crisis_only["disaster_confidence"],
        "severity_gain_from_optional": full["severity"] - crisis_only["severity"],
    }

    # Gate weights from the model (if available from last forward)
    with torch.no_grad():
        out = tri_fusion_model(crisis_embedding, iot_embedding, satellite_embedding)
    report["gate_weights"] = {
        k: float(v.mean().item()) for k, v in out.modality_weights.items()
    }

    return report
