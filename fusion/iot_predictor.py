"""
IoT Predictor
=============
Loads the single trained AdaptiveIoTClassifier and provides inference.
Disaster type is auto-detected — no user input required.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "IOT", "models", "iot_model.pth")

DISASTER_TYPES = ["fire", "storm", "earthquake", "flood", "unknown"]
RISK_NAMES     = ["fire_prob", "storm_cat_norm", "eq_magnitude_norm", "flood_risk_norm"]


@dataclass
class IoTPrediction:
    disaster_type:        str           # auto-detected
    type_probabilities:   Dict[str, float]
    severity_score:       float         # 0-1
    fire_prob:            float
    storm_cat_norm:       float         # 0-1 (divide by 5 to get Saffir-Simpson cat)
    eq_magnitude_norm:    float         # 0-1 (multiply by 9 to get Richter)
    flood_risk_norm:      float         # 0-1 (multiply by 100 for score)
    casualty_risk:        float         # 0-1 estimated human casualty likelihood
    sensor_weights:       Dict[str, float]  # per-group confidence weights
    embedding:            torch.Tensor  # [128] for FusionLayer


class IoTPredictor:
    """
    Loads the single AdaptiveIoTClassifier and runs inference.
    Call predict_from_features() with a raw 32-dim sensor vector,
    or use the helper build_features_*() constructors.
    """

    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"IoT model not found: {MODEL_PATH}\n"
                "Run: python IOT/train_iot.py"
            )
        # Import model class from training module
        from IOT.train_iot import AdaptiveIoTClassifier
        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        cfg  = ckpt["config"]

        self.model = AdaptiveIoTClassifier(
            group_size       = cfg["group_size"],
            hidden_dim       = cfg["hidden_dim"],
            n_disaster_types = cfg["n_disaster_types"],
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.hidden_dim = cfg["hidden_dim"]
        print(f"  [IoT] Model loaded — val F1={ckpt.get('val_f1', 'N/A'):.4f}  "
              f"val acc={ckpt.get('val_acc', 'N/A'):.4f}")

    @torch.no_grad()
    def predict_from_features(self, features: list) -> IoTPrediction:
        """Run inference on a 32-dim sensor feature vector."""
        if len(features) != 32:
            raise ValueError(
                f"Expected 32-dim feature vector, got {len(features)}-dim. "
                "Check feature extraction function output."
            )
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # [1, 32]
        dis_logits, severity, risk, casualty, emb, attn = self.model(x, return_attention=True)

        probs    = F.softmax(dis_logits, dim=-1).squeeze(0)
        pred_idx = int(probs.argmax())
        type_str = DISASTER_TYPES[pred_idx]

        return IoTPrediction(
            disaster_type      = type_str,
            type_probabilities = {DISASTER_TYPES[i]: float(probs[i]) for i in range(5)},
            severity_score     = float(severity.item()),
            fire_prob          = float(risk[0, 0]),
            storm_cat_norm     = float(risk[0, 1]),
            eq_magnitude_norm  = float(risk[0, 2]),
            flood_risk_norm    = float(risk[0, 3]),
            casualty_risk      = float(casualty.item()),
            sensor_weights     = {
                "weather":  float(attn["weather_weight"].item()),
                "storm":    float(attn["storm_weight"].item()),
                "seismic":  float(attn["seismic_weight"].item()),
                "hydro":    float(attn["hydro_weight"].item()),
            },
            embedding = emb.squeeze(0).cpu(),  # [128]
        )

    # ── Convenience constructors ──────────────────────────────────────────
    def predict_fire_conditions(self, precipitation, max_temp, min_temp,
                                avg_wind_speed, month, temp_range=None,
                                lagged_precipitation=0.0) -> IoTPrediction:
        from IOT.train_iot import features_from_fire
        import pandas as pd
        tr = temp_range if temp_range is not None else max_temp - min_temp
        row = pd.Series({
            "PRECIPITATION": precipitation, "MAX_TEMP": max_temp,
            "MIN_TEMP": min_temp, "AVG_WIND_SPEED": avg_wind_speed,
            "TEMP_RANGE": tr, "WIND_TEMP_RATIO": avg_wind_speed / (max_temp + 1e-6),
            "MONTH": month, "LAGGED_PRECIPITATION": lagged_precipitation,
        })
        return self.predict_from_features(features_from_fire(row))

    def predict_storm(self, lat, lon, wind_kts, pressure, month,
                      category=None, shape_leng=1.0) -> IoTPrediction:
        from IOT.train_iot import features_from_storm_hist, CAT_SEV
        import pandas as pd
        cat_str = str(category) if category else "TS"
        row = pd.Series({
            "WIND_KTS": wind_kts, "PRESSURE": pressure,
            "LAT": lat, "LONG": lon, "CAT": cat_str,
            "MONTH": month, "Shape_Leng": shape_leng,
        })
        return self.predict_from_features(features_from_storm_hist(row))

    def predict_earthquake(self, lat, lon, depth, magnitude=None,
                           rms=0.2, n_stations=10, n_phases=20,
                           azimuth_gap=180, month=6) -> IoTPrediction:
        from IOT.train_iot import features_from_eq_iran
        import pandas as pd
        row = pd.Series({
            "Lat": lat, "Long": lon, "Depth": depth,
            "Magnitude": magnitude or 0.0,
            "RMS": rms, "Number of stations": n_stations,
            "Number of phases": n_phases, "Azimuth GAP": azimuth_gap,
            "Date": f"2024/{month:02d}/01",
        })
        return self.predict_from_features(features_from_eq_iran(row))

    def predict_flood(self, lat, lon, elevation_m, distance_to_river_m,
                      rainfall_7d, monthly_rainfall, drainage_index=0.5,
                      ndvi=0.3, ndwi=0.1, historical_flood_count=0) -> IoTPrediction:
        from IOT.train_iot import features_from_flood
        import pandas as pd
        row = pd.Series({
            "latitude": lat, "longitude": lon,
            "elevation_m": elevation_m,
            "distance_to_river_m": distance_to_river_m,
            "rainfall_7d_mm": rainfall_7d,
            "monthly_rainfall_mm": monthly_rainfall,
            "drainage_index": drainage_index,
            "ndvi": ndvi, "ndwi": ndwi,
            "historical_flood_count": historical_flood_count,
            "flood_risk_score": 0.0,   # unknown at predict time
        })
        return self.predict_from_features(features_from_flood(row))
