"""
xBD Satellite Damage Assessment Model
=======================================
Extracts the DeepLabV3Plus_xBD architecture from the training notebook
and provides a clean inference + feature-extraction API.

Model outputs (per the notebook):
  F_sat    : [B, 512]       — satellite embedding for physical damage
  P_x      : [B, 4, H, W]  — damage probability maps (no-damage / minor / major / destroyed)
  F_region : [B, 128]       — regional spatial statistics

Loading:
  The .pkl was saved via `pickle.dump(model)` from a Jupyter __main__,
  so we use a redirect unpickler that maps __main__.* -> this module.
"""

import os
import io
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None
    print("[xBD] WARNING: segmentation-models-pytorch not installed. "
          "Install with: pip install segmentation-models-pytorch")


# ──────────────────────────────────────────────
# Architecture (extracted from notebook Cell 16)
# ──────────────────────────────────────────────

XBD_DAMAGE_CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
NUM_CLASSES = 4
F_SAT_DIM = 512
F_REGION_DIM = 128
IMAGE_SIZE = 512

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


class RegionalStatsModule(nn.Module):
    """
    Extract F_region: Regional spatial context statistics.
    Combines decoder features with damage probability maps.
    """

    def __init__(self, in_channels: int = 256, out_dim: int = 128):
        super().__init__()
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(in_channels + 4, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, out_dim),
        )

    def forward(self, decoder_features: torch.Tensor, damage_probs: torch.Tensor) -> torch.Tensor:
        if decoder_features.shape[2:] != damage_probs.shape[2:]:
            decoder_features = F.interpolate(
                decoder_features, size=damage_probs.shape[2:],
                mode="bilinear", align_corners=False,
            )
        x = torch.cat([decoder_features, damage_probs], dim=1)
        x = self.conv_fusion(x)
        x = self.spatial_pool(x)
        x = x.flatten(1)
        return self.fc(x)


class DeepLabV3Plus_xBD(nn.Module):
    """
    Modified DeepLabV3+ for xBD damage assessment.

    Outputs:
      F_sat    : [B, F_SAT_DIM]       — satellite embedding
      P_x      : [B, 4, H, W]         — damage probabilities
      F_region : [B, F_REGION_DIM]     — regional statistics
    """

    def __init__(
        self,
        num_classes: int = 4,
        encoder_name: str = "resnet101",
        encoder_weights: str = "imagenet",
        F_sat_dim: int = 512,
        F_region_dim: int = 128,
    ):
        super().__init__()
        if smp is None:
            raise ImportError("segmentation-models-pytorch is required")

        self.num_classes = num_classes
        self.F_sat_dim = F_sat_dim

        self.base_model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=None,
        )

        encoder_channels = 2048  # ResNet101/50 final block
        self.embedding_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(encoder_channels, F_sat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.regional_stats = RegionalStatsModule(
            in_channels=256, out_dim=F_region_dim,
        )

    def forward(self, x: torch.Tensor, return_features: bool = True) -> dict:
        features = self.base_model.encoder(x)
        if return_features:
            F_sat = self.embedding_extractor(features[-1])
        # smp >=0.4 decoder takes a list; smp 0.3.x takes unpacked args
        try:
            decoder_output = self.base_model.decoder(features)
        except TypeError:
            decoder_output = self.base_model.decoder(*features)
        logits = self.base_model.segmentation_head(decoder_output)
        P_x = torch.softmax(logits, dim=1)
        if return_features:
            F_region = self.regional_stats(decoder_output, P_x)

        output = {"logits": logits, "P_x": P_x}
        if return_features:
            output["F_sat"] = F_sat
            output["F_region"] = F_region
        return output


# ──────────────────────────────────────────────
# Custom unpickler to remap __main__ classes
# ──────────────────────────────────────────────

_LOCAL_CLASSES = {
    "DeepLabV3Plus_xBD": DeepLabV3Plus_xBD,
    "RegionalStatsModule": RegionalStatsModule,
}


class _RedirectUnpickler(pickle.Unpickler):
    """Redirect __main__.ClassName to this module's classes."""

    def find_class(self, module: str, name: str):
        if module == "__main__" and name in _LOCAL_CLASSES:
            return _LOCAL_CLASSES[name]
        return super().find_class(module, name)


def _pkl_load_with_map_location(pkl_path: str, device: torch.device):
    """
    Load a pickled model with proper CUDA->CPU mapping.

    The .pkl was saved on a CUDA machine via pickle.dump(model).
    Nested tensor storages also contain CUDA references, so we
    patch torch.storage._load_from_bytes to force CPU mapping.
    """
    import __main__
    import torch.storage as _ts

    # 1) Inject local classes into __main__ so pickle can find them
    originals = {}
    for name, cls in _LOCAL_CLASSES.items():
        originals[name] = getattr(__main__, name, None)
        setattr(__main__, name, cls)

    # 2) Patch _load_from_bytes to force map_location=cpu
    #    (the pkl was saved with pickle.dump, so nested tensor
    #    storages call torch.load internally without map_location)
    _orig_load_from_bytes = _ts._load_from_bytes

    def _patched_load_from_bytes(b):
        return torch.load(io.BytesIO(b), map_location=device, weights_only=False)

    _ts._load_from_bytes = _patched_load_from_bytes

    try:
        with open(pkl_path, "rb") as f:
            model = pickle.load(f)
    finally:
        # Restore everything
        _ts._load_from_bytes = _orig_load_from_bytes
        for name, orig in originals.items():
            if orig is None and hasattr(__main__, name):
                delattr(__main__, name)
            elif orig is not None:
                setattr(__main__, name, orig)

    return model


# ──────────────────────────────────────────────
# Preprocessing (from notebook Cell 5)
# ──────────────────────────────────────────────

def preprocess_satellite_image(
    image_path: str,
    target_size: int = IMAGE_SIZE,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Load and preprocess a satellite image for DeepLabV3Plus_xBD.

    Returns [1, 3, target_size, target_size] tensor on the specified device.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load satellite image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    # Normalize (ImageNet)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)

    if device is not None:
        tensor = tensor.to(device)
    return tensor


def preprocess_satellite_pil(
    pil_image,
    target_size: int = IMAGE_SIZE,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Preprocess a PIL Image for the xBD model."""
    img = np.array(pil_image.convert("RGB"))
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────

_DEFAULT_PKL = os.path.join(os.path.dirname(__file__), "deeplabv3plus_xbd_trained.pkl")


def load_xbd_model(
    pkl_path: str = _DEFAULT_PKL,
    device: Optional[torch.device] = None,
) -> DeepLabV3Plus_xBD:
    """
    Load the trained xBD model from the .pkl checkpoint.

    The .pkl was saved via pickle.dump(model) from a notebook (__main__),
    so we use a redirect unpickler to remap classes.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"xBD checkpoint not found: {pkl_path}")

    print(f"[xBD] Loading model from {pkl_path}...")
    loaded = _pkl_load_with_map_location(pkl_path, device)

    # The pkl may be a checkpoint dict with 'model' and/or 'model_state_dict' keys
    if isinstance(loaded, dict) and "model" in loaded:
        model = loaded["model"]
        if isinstance(model, DeepLabV3Plus_xBD):
            print(f"[xBD] Extracted model from checkpoint dict (epoch {loaded.get('epoch', '?')})")
        else:
            print(f"[xBD] 'model' key is {type(model)}, trying model_state_dict...")
            model = None

        if model is None and "model_state_dict" in loaded:
            arch = loaded.get("architecture", {})
            print("[xBD] Reconstructing model from state_dict...")
            model = DeepLabV3Plus_xBD(
                num_classes=arch.get("num_classes", NUM_CLASSES),
                encoder_name=arch.get("encoder", "resnet101"),
                encoder_weights=None,
                F_sat_dim=arch.get("F_sat_dim", F_SAT_DIM),
                F_region_dim=arch.get("F_region_dim", F_REGION_DIM),
            )
            model.load_state_dict(loaded["model_state_dict"], strict=False)
    elif isinstance(loaded, DeepLabV3Plus_xBD):
        model = loaded
    elif isinstance(loaded, dict) and any("base_model" in k for k in loaded.keys()):
        print("[xBD] pkl contains raw state_dict, reconstructing model...")
        model = DeepLabV3Plus_xBD(
            num_classes=NUM_CLASSES,
            encoder_name="resnet101",
            encoder_weights=None,
            F_sat_dim=F_SAT_DIM,
            F_region_dim=F_REGION_DIM,
        )
        model.load_state_dict(loaded, strict=False)
    else:
        raise TypeError(f"Expected DeepLabV3Plus_xBD or checkpoint dict, got {type(loaded)}")

    model.to(device).eval()
    print(f"[xBD] Model loaded on {device}.")
    return model


# ──────────────────────────────────────────────
# Feature extraction for fusion
# ──────────────────────────────────────────────

SATELLITE_EMB_DIM = F_SAT_DIM + F_REGION_DIM  # 512 + 128 = 640


def extract_satellite_features(
    model: DeepLabV3Plus_xBD,
    image_tensor: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Extract features from a satellite image for tri-modal fusion.

    Args:
        model: Trained DeepLabV3Plus_xBD
        image_tensor: [B, 3, 512, 512] preprocessed tensor

    Returns:
        dict with:
          'F_sat'      : [B, 512]  satellite embedding
          'P_x'        : [B, 4, H, W] damage probabilities
          'F_region'   : [B, 128]  regional statistics
          'embedding'  : [B, 640]  concatenated F_sat + F_region (for fusion)
          'damage_pred': [B]       predicted damage class per image
          'damage_probs_summary': [B, 4]  mean damage probability per class
    """
    model.eval()
    with torch.no_grad():
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        outputs = model(image_tensor, return_features=True)

    f_sat = outputs["F_sat"]
    f_region = outputs["F_region"]
    p_x = outputs["P_x"]

    # Concatenated embedding for fusion layer
    embedding = torch.cat([f_sat, f_region], dim=1)

    # Per-image damage summary: average probability across all pixels
    # P_x is [B, 4, H, W], mean over spatial dims -> [B, 4]
    damage_probs_summary = p_x.mean(dim=(2, 3))
    damage_pred = damage_probs_summary.argmax(dim=1)

    return {
        "F_sat": f_sat,
        "P_x": p_x,
        "F_region": f_region,
        "embedding": embedding,
        "damage_pred": damage_pred,
        "damage_probs_summary": damage_probs_summary,
    }


# ──────────────────────────────────────────────
# Convenience: full satellite inference
# ──────────────────────────────────────────────

class SatellitePredictor:
    """High-level wrapper for satellite image damage assessment and feature extraction."""

    def __init__(self, pkl_path: str = _DEFAULT_PKL, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_xbd_model(pkl_path, self.device)

    def predict(self, image_path: str) -> Dict:
        """Run full satellite inference on an image file."""
        tensor = preprocess_satellite_image(image_path, device=self.device)
        feats = extract_satellite_features(self.model, tensor)
        damage_idx = int(feats["damage_pred"].item())
        probs = feats["damage_probs_summary"].squeeze(0).cpu().numpy()

        return {
            "damage_class": XBD_DAMAGE_CLASSES[damage_idx],
            "damage_class_idx": damage_idx,
            "damage_probs": {
                XBD_DAMAGE_CLASSES[i]: float(probs[i]) for i in range(NUM_CLASSES)
            },
            "embedding": feats["embedding"].cpu(),
            "F_sat": feats["F_sat"].cpu(),
            "F_region": feats["F_region"].cpu(),
        }

    def get_embedding(self, image_path: str) -> torch.Tensor:
        """Extract the 640-dim satellite embedding for fusion."""
        tensor = preprocess_satellite_image(image_path, device=self.device)
        feats = extract_satellite_features(self.model, tensor)
        return feats["embedding"].cpu()

    def get_embedding_from_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Extract embedding from a preprocessed tensor."""
        image_tensor = image_tensor.to(self.device)
        feats = extract_satellite_features(self.model, image_tensor)
        return feats["embedding"].cpu()
