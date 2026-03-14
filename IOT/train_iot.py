"""
Adaptive IoT Sensor Classifier  —  train_iot.py
=================================================
Novel Architecture (mirrors the crisis AdaptiveFusionClassifier):

  4 Sensor Groups  →  Group Encoders
                   →  Per-group Confidence Estimators  (NOVEL: adaptive weighting)
                   →  Cross-Group Multi-Head Attention  (NOVEL: sensor interaction)
                   →  Shared Representation  [B, 128]
                   →  3 Multi-Task Heads:
                         • disaster_type  (fire / storm / earthquake / flood / unknown)
                         • severity_score (0-1 regression)
                         • risk_details   (4 per-hazard risk scores)

Input: 32-dim unified sensor vector — 4 groups × 8 features
  Group 0  Weather  : precipitation, max_temp, min_temp, wind_speed, temp_range,
                      drought_index, month_sin, month_cos
  Group 1  Storm    : wind_intensity, pressure_anomaly, lat, lon,
                      storm_cat_norm, hour_sin, hour_cos, track_speed
  Group 2  Seismic  : depth, rms, stations, phases, azimuth_gap,
                      eq_lat, eq_lon, magnitude_proxy
  Group 3  Hydro    : elevation_inv, river_proximity, rainfall_7d,
                      monthly_rain, drainage, ndvi, ndwi, flood_history

No user input required — disaster type is auto-detected from sensor signals.

Saved to: IOT/models/iot_model.pth
"""

import os, sys, math, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
DATASETS_DIR   = os.path.join(os.path.dirname(__file__), "datasets")
MODELS_DIR     = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

DISASTER_TYPES = ["fire", "storm", "earthquake", "flood", "unknown"]
DISASTER_IDX   = {d: i for i, d in enumerate(DISASTER_TYPES)}

GROUP_SIZE     = 8
N_GROUPS       = 4
HIDDEN_DIM     = 128     # per-group encoding size; final embedding = 128
TOTAL_FEATURES = GROUP_SIZE * N_GROUPS   # 32

BATCH_SIZE     = 256
EPOCHS         = 40
LR             = 1e-3
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# Feature Extraction
# ─────────────────────────────────────────────
def _sin_cos(val, period):
    return math.sin(2 * math.pi * val / period), math.cos(2 * math.pi * val / period)

def _clamp(v, lo=0.0, hi=1.0):
    return float(max(lo, min(hi, v)))

CAT_SEV = {
    "TD": 0.10, "TS": 0.30, "E": 0.15, "L": 0.20,
    "SD": 0.15, "SS": 0.25, "W": 0.10,
    "H1": 0.40, "H2": 0.55, "H3": 0.70, "H4": 0.85, "H5": 1.00,
}

def features_from_fire(row) -> list:
    precip   = _clamp(row["PRECIPITATION"] / 50.0)
    max_t    = _clamp(row["MAX_TEMP"] / 55.0)
    min_t    = _clamp(row["MIN_TEMP"] / 55.0)
    wind     = _clamp(row["AVG_WIND_SPEED"] / 50.0)
    t_range  = _clamp(row["TEMP_RANGE"] / 35.0)
    drought  = _clamp(1.0 - precip)                          # low rain = high drought
    ms, mc   = _sin_cos(int(row["MONTH"]), 12)

    weather  = [precip, max_t, min_t, wind, t_range, drought, ms, mc]
    # California approximate centroid; populate weather-wind in storm slot too
    lat_n, lon_n = 36.5 / 90.0, -119.5 / 180.0
    storm    = [wind, 0.0, lat_n, lon_n, 0.0, 0.0, 0.0, 0.0]
    seismic  = [0.0] * 8
    hydro    = [0.0, 0.0, precip, _clamp(row.get("LAGGED_PRECIPITATION", 0) / 50),
                0.5, 0.3, 0.1, 0.0]
    return weather + storm + seismic + hydro


def features_from_storm_hist(row) -> list:
    wind_n    = _clamp(float(row["WIND_KTS"]) / 200.0)
    press     = float(row["PRESSURE"])
    press_n   = _clamp((1013 - press) / 500.0) if press > 0 else 0.0
    lat_n     = float(row["LAT"]) / 90.0
    lon_n     = float(row["LONG"]) / 180.0
    cat_sev   = CAT_SEV.get(str(row.get("CAT", "TS")), 0.3)
    month     = int(row["MONTH"])
    ms, mc    = _sin_cos(month, 12)
    track     = _clamp(float(row.get("Shape_Leng", 1.0)) / 10.0)

    weather   = [0.0, 0.0, 0.0, wind_n, 0.0, 0.0, ms, mc]
    storm     = [wind_n, press_n, lat_n, lon_n, cat_sev, 0.0, 0.0, track]
    seismic   = [0.0] * 8
    hydro     = [0.0, 0.0, 0.0, 0.0, 0.5, 0.3, 0.1, 0.0]
    return weather + storm + seismic + hydro


def features_from_storm_atl(row) -> list:
    wind_n  = _clamp(float(row["wind"]) / 180.0)
    press   = float(row["pressure"])
    press_n = _clamp((1013 - press) / 500.0) if press > 0 else 0.0
    lat_n   = float(row["lat"]) / 90.0
    lon_n   = float(row["long"]) / 180.0
    cat     = float(row["category"]) if not pd.isna(row["category"]) else 0
    cat_n   = _clamp(cat / 5.0)
    month   = int(row["month"])
    hour    = int(row["hour"])
    ms, mc  = _sin_cos(month, 12)
    hs, hc  = _sin_cos(hour, 24)
    ts_d    = _clamp(float(row.get("tropicalstorm_force_diameter", 0)) / 1000.0)

    weather  = [0.0, 0.0, 0.0, wind_n, 0.0, 0.0, ms, mc]
    storm    = [wind_n, press_n, lat_n, lon_n, cat_n, hs, hc, ts_d]
    seismic  = [0.0] * 8
    hydro    = [0.0, 0.0, 0.0, 0.0, 0.5, 0.3, 0.1, 0.0]
    return weather + storm + seismic + hydro


def features_from_eq_global(row) -> list:
    depth   = _clamp(float(row["location.depth"]) / 700.0)
    mag     = _clamp(float(row["impact.magnitude"]) / 9.0)
    lat_n   = float(row["location.latitude"]) / 90.0
    lon_n   = float(row["location.longitude"]) / 180.0
    month   = int(row["time.month"])
    hour    = int(row.get("time.hour", 12))
    ms, mc  = _sin_cos(month, 12)
    gap_n   = _clamp(float(row.get("impact.gap", 180)) / 360.0)

    weather  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ms, mc]
    storm    = [0.0, 0.0, lat_n, lon_n, 0.0, 0.0, 0.0, 0.0]
    seismic  = [depth, 0.2, 0.5, 0.5, gap_n, lat_n, lon_n, mag]
    hydro    = [0.0] * 8
    return weather + storm + seismic + hydro


def features_from_eq_iran(row) -> list:
    depth   = _clamp(float(row["Depth"]) / 700.0)
    mag     = _clamp(float(row["Magnitude"]) / 9.0)
    lat_n   = float(row["Lat"]) / 90.0
    lon_n   = float(row["Long"]) / 180.0
    rms     = _clamp(float(row["RMS"]) / 2.0)
    sta     = _clamp(float(row["Number of stations"]) / 200.0)
    phs     = _clamp(float(row["Number of phases"]) / 500.0)
    gap     = _clamp(float(row["Azimuth GAP"]) / 360.0)
    try:
        month = int(str(row["Date"]).split("/")[1])
    except Exception:
        month = 6
    ms, mc  = _sin_cos(month, 12)

    weather  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ms, mc]
    storm    = [0.0, 0.0, lat_n, lon_n, 0.0, 0.0, 0.0, 0.0]
    seismic  = [depth, rms, sta, phs, gap, lat_n, lon_n, mag]
    hydro    = [0.0] * 8
    return weather + storm + seismic + hydro


def features_from_flood(row) -> list:
    elev    = _clamp(1.0 - float(row["elevation_m"]) / 500.0)  # low elev = higher risk
    d_river = _clamp(1.0 - float(row["distance_to_river_m"]) / 5000.0)
    r7d     = _clamp(float(row["rainfall_7d_mm"]) / 500.0)
    rmon    = _clamp(float(row["monthly_rainfall_mm"]) / 2000.0)
    drain   = _clamp(float(row["drainage_index"]))
    ndvi    = _clamp((float(row["ndvi"]) + 1) / 2)
    ndwi    = _clamp((float(row["ndwi"]) + 1) / 2)
    fhist   = _clamp(float(row["historical_flood_count"]) / 20.0)
    lat_n   = float(row["latitude"]) / 90.0
    lon_n   = float(row["longitude"]) / 180.0

    weather  = [r7d, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    storm    = [0.0, 0.0, lat_n, lon_n, 0.0, 0.0, 0.0, 0.0]
    seismic  = [0.0] * 8
    hydro    = [elev, d_river, r7d, rmon, drain, ndvi, ndwi, fhist]
    return weather + storm + seismic + hydro


# ─────────────────────────────────────────────
# Build unified dataset
# ─────────────────────────────────────────────
def build_dataset():
    all_feats, all_types, all_severity, all_risk, all_casualty = [], [], [], [], []

    # ── 1. CA Wildfire (fire=True rows only → label fire) ─────────────────
    print("  Loading CA Wildfire…")
    df = pd.read_csv(os.path.join(DATASETS_DIR, "CA_Weather_Fire_Dataset_1984-2025.csv"))
    fire_df = df[df["FIRE_START_DAY"] == True]
    no_fire = df[df["FIRE_START_DAY"] == False].sample(
        n=min(len(fire_df), 2000), random_state=42
    )
    for _, row in fire_df.iterrows():
        sev = _clamp((row["WIND_TEMP_RATIO"] * 5 + (1 - row["PRECIPITATION"] / 50) +
                      row["MAX_TEMP"] / 55) / 3)
        all_feats.append(features_from_fire(row))
        all_types.append(DISASTER_IDX["fire"])
        all_severity.append(sev)
        all_risk.append([sev, 0.0, 0.0, 0.0])
        all_casualty.append(_clamp(sev * 0.70))
    for _, row in no_fire.iterrows():
        all_feats.append(features_from_fire(row))
        all_types.append(DISASTER_IDX["unknown"])
        all_severity.append(0.05)
        all_risk.append([0.05, 0.0, 0.0, 0.0])
        all_casualty.append(0.04)
    print(f"    Fire: {len(fire_df)} pos, {len(no_fire)} neg")

    # ── 2. Historical Tropical Storms ──────────────────────────────────────
    print("  Loading Historical Storms…")
    df = pd.read_csv(os.path.join(DATASETS_DIR, "Historical_Tropical_Storm_Tracks.csv"))
    df["CAT"] = df["CAT"].fillna("TS")
    df_s = df.sample(n=min(len(df), 8000), random_state=42)
    for _, row in df_s.iterrows():
        sev = CAT_SEV.get(str(row.get("CAT", "TS")), 0.3)
        cat_str = str(row.get("CAT", "TS"))
        mult = 0.90 if cat_str in ["H4", "H5"] else 0.75
        all_feats.append(features_from_storm_hist(row))
        all_types.append(DISASTER_IDX["storm"])
        all_severity.append(sev)
        all_risk.append([0.0, sev, 0.0, 0.0])
        all_casualty.append(_clamp(sev * mult))
    print(f"    Storm hist: {len(df_s)} samples")

    # ── 3. Atlantic Storms ─────────────────────────────────────────────────
    print("  Loading Atlantic Storms…")
    df = pd.read_csv(os.path.join(DATASETS_DIR, "storms.csv"))
    df = df.dropna(subset=["category"])
    df_s = df.sample(n=min(len(df), 6000), random_state=42)
    for _, row in df_s.iterrows():
        sev  = _clamp(float(row["category"]) / 5.0)
        mult = 0.90 if float(row["category"]) >= 4 else 0.75
        all_feats.append(features_from_storm_atl(row))
        all_types.append(DISASTER_IDX["storm"])
        all_severity.append(sev)
        all_risk.append([0.0, sev, 0.0, 0.0])
        all_casualty.append(_clamp(sev * mult))
    print(f"    Storm ATL: {len(df_s)} samples")

    # ── 4. Global Earthquakes ─────────────────────────────────────────────
    print("  Loading Global Earthquakes…")
    df = pd.read_csv(os.path.join(DATASETS_DIR, "earthquakes.csv"))
    df.columns = df.columns.str.strip('"')
    df = df.dropna(subset=["impact.magnitude"])
    for _, row in df.iterrows():
        sev = _clamp(float(row["impact.magnitude"]) / 9.0)
        all_feats.append(features_from_eq_global(row))
        all_types.append(DISASTER_IDX["earthquake"])
        all_severity.append(sev)
        all_risk.append([0.0, 0.0, sev, 0.0])
        all_casualty.append(_clamp(sev * 0.92))
    print(f"    EQ global: {len(df)} samples")

    # ── 5. Iran Earthquakes ───────────────────────────────────────────────
    print("  Loading Iran Earthquakes…")
    df = pd.read_csv(os.path.join(DATASETS_DIR, "iran earthquakes.csv"))
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["Magnitude"])
    df_s = df.sample(n=min(len(df), 10000), random_state=42)
    for _, row in df_s.iterrows():
        sev = _clamp(float(row["Magnitude"]) / 9.0)
        all_feats.append(features_from_eq_iran(row))
        all_types.append(DISASTER_IDX["earthquake"])
        all_severity.append(sev)
        all_risk.append([0.0, 0.0, sev, 0.0])
        all_casualty.append(_clamp(sev * 0.92))
    print(f"    EQ Iran:   {len(df_s)} samples")

    # ── 6. Sri Lanka Floods ───────────────────────────────────────────────
    print("  Loading Floods…")
    df = pd.read_csv(os.path.join(DATASETS_DIR, "sri_lanka_flood_risk_dataset_25000.csv"))
    df = df.dropna(subset=["flood_risk_score"])
    for _, row in df.iterrows():
        sev = _clamp(float(row["flood_risk_score"]) / 100.0)
        all_feats.append(features_from_flood(row))
        all_types.append(DISASTER_IDX["flood"])
        all_severity.append(sev)
        all_risk.append([0.0, 0.0, 0.0, sev])
        all_casualty.append(_clamp(sev * 0.60))
    print(f"    Floods:    {len(df)} samples")

    X  = torch.tensor(all_feats,     dtype=torch.float32)
    Yt = torch.tensor(all_types,     dtype=torch.long)
    Ys = torch.tensor(all_severity,  dtype=torch.float32)
    Yr = torch.tensor(all_risk,      dtype=torch.float32)
    Yc = torch.tensor(all_casualty,  dtype=torch.float32)
    print(f"\n  Total samples: {len(X)}")
    return X, Yt, Ys, Yr, Yc


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class SensorDataset(Dataset):
    def __init__(self, X, Yt, Ys, Yr, Yc):
        self.X, self.Yt, self.Ys, self.Yr, self.Yc = X, Yt, Ys, Yr, Yc
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.Yt[i], self.Ys[i], self.Yr[i], self.Yc[i]


# ─────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────
class SensorGroupEncoder(nn.Module):
    """Encodes one group of 8 sensor features → hidden_dim embedding."""
    def __init__(self, in_dim: int = GROUP_SIZE, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )
    def forward(self, x):
        return self.net(x)


class SensorConfidenceEstimator(nn.Module):
    """
    Estimates how much information / signal is present in a sensor group.
    Mirrors ConfidenceEstimator from the crisis AdaptiveFusionClassifier.
    Output: scalar in (0, 1).
    """
    def __init__(self, in_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x)


class AdaptiveIoTClassifier(nn.Module):
    """
    Adaptive IoT Multi-Hazard Classifier

    Novel Contributions:
      1. Per-sensor-group confidence estimation — groups with no active
         sensors are automatically down-weighted.
      2. Cross-group multi-head attention — seismic signals can reinforce
         hydro signals (earthquake-triggered floods) etc.
      3. Shared multi-task head — jointly learns type, severity & risk.
      4. No disaster_type input needed — fully self-supervised type detection.

    Architecture mirrors the crisis AdaptiveFusionClassifier for
    architectural consistency across the fusion pipeline.
    """
    def __init__(
        self,
        group_size:       int = GROUP_SIZE,
        hidden_dim:       int = HIDDEN_DIM,
        n_disaster_types: int = len(DISASTER_TYPES),
        n_risk_outputs:   int = 4,
        n_attn_heads:     int = 4,
        dropout:          float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # ── Group Encoders ─────────────────────────────────────────────────
        self.encoders = nn.ModuleList([
            SensorGroupEncoder(group_size, hidden_dim) for _ in range(N_GROUPS)
        ])

        # ── Confidence Estimators (NOVEL #1) ──────────────────────────────
        self.confidence = nn.ModuleList([
            SensorConfidenceEstimator(hidden_dim) for _ in range(N_GROUPS)
        ])

        # ── Cross-Group Attention (NOVEL #2) ──────────────────────────────
        self.cross_group_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_attn_heads,
            dropout=0.1,
            batch_first=True,
        )

        # ── Multi-Task Heads (NOVEL #3) ───────────────────────────────────
        self.disaster_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_disaster_types),
        )
        self.severity_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_risk_outputs),
            nn.Sigmoid(),
        )
        # ── Casualty Risk Head (NOVEL #4) ──────────────────────────────────
        # Estimates human casualty likelihood (0-1) from sensor signals.
        # Earthquake → highest (no warning); Flood → lower (slower onset).
        self.casualty_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        x : [B, 32]  (4 groups × 8 features, already normalised)
        Returns
        -------
        disaster_logits : [B, 5]
        severity        : [B]
        risk_details    : [B, 4]
        embedding       : [B, hidden_dim]   ← used by FusionLayer
        (optional) attention_info : dict
        """
        # ── Encode each sensor group ───────────────────────────────────────
        group_feats = [
            enc(x[:, i * GROUP_SIZE: (i + 1) * GROUP_SIZE])
            for i, enc in enumerate(self.encoders)
        ]   # list of 4 × [B, hidden_dim]

        # ── Confidence scores ──────────────────────────────────────────────
        confs = [conf(g) for conf, g in zip(self.confidence, group_feats)]
        # confs: list of 4 × [B, 1]

        total_conf = sum(confs) + 1e-8
        weights    = [c / total_conf for c in confs]   # normalised to sum=1

        # ── Confidence-weighted group features ────────────────────────────
        weighted = [w * g for w, g in zip(weights, group_feats)]   # [B, hidden]

        # ── Cross-group attention ─────────────────────────────────────────
        stacked  = torch.stack(weighted, dim=1)               # [B, 4, hidden]
        attended, attn_weights = self.cross_group_attn(stacked, stacked, stacked)

        # ── Global representation via mean-pooling ────────────────────────
        embedding = attended.mean(dim=1)                       # [B, hidden_dim]

        # ── Output heads ──────────────────────────────────────────────────
        disaster_logits = self.disaster_head(embedding)              # [B, 5]
        severity        = self.severity_head(embedding).squeeze(-1)  # [B]
        risk_details    = self.risk_head(embedding)                  # [B, 4]
        casualty_risk   = self.casualty_head(embedding).squeeze(-1)  # [B]

        if return_attention:
            return disaster_logits, severity, risk_details, casualty_risk, embedding, {
                "weather_conf":   confs[0].squeeze(-1),
                "storm_conf":     confs[1].squeeze(-1),
                "seismic_conf":   confs[2].squeeze(-1),
                "hydro_conf":     confs[3].squeeze(-1),
                "weather_weight": weights[0].squeeze(-1),
                "storm_weight":   weights[1].squeeze(-1),
                "seismic_weight": weights[2].squeeze(-1),
                "hydro_weight":   weights[3].squeeze(-1),
                "cross_group_attn": attn_weights,
            }
        return disaster_logits, severity, risk_details, casualty_risk, embedding


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
def train():
    print("=" * 60)
    print("   AdaptiveIoTClassifier  —  Training")
    print("=" * 60)
    print(f"Device: {DEVICE}\n")

    print("Building dataset…")
    X, Yt, Ys, Yr, Yc = build_dataset()

    # ── Weighted sampler for class balance ────────────────────────────────
    counts   = torch.bincount(Yt)
    weights  = 1.0 / counts.float()
    s_weights = weights[Yt]
    sampler  = WeightedRandomSampler(s_weights, len(s_weights), replacement=True)

    n       = len(X)
    split   = int(n * 0.85)
    idx     = torch.randperm(n)
    tr_idx, va_idx = idx[:split], idx[split:]

    train_ds  = SensorDataset(X[tr_idx], Yt[tr_idx], Ys[tr_idx], Yr[tr_idx], Yc[tr_idx])
    val_ds    = SensorDataset(X[va_idx], Yt[va_idx], Ys[va_idx], Yr[va_idx], Yc[va_idx])

    # Weighted sampling only for train
    tr_sw     = s_weights[tr_idx]
    tr_sampler = WeightedRandomSampler(tr_sw, len(tr_sw), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=tr_sampler)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    model     = AdaptiveIoTClassifier().to(DEVICE)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_p:,}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    ce_loss   = nn.CrossEntropyLoss()
    mse_loss  = nn.MSELoss()

    best_f1   = 0.0
    save_path = os.path.join(MODELS_DIR, "iot_model.pth")

    print(f"\nTraining for {EPOCHS} epochs…\n")
    print(f"{'Epoch':>5} | {'T-loss':>8} | {'V-loss':>8} | "
          f"{'V-acc':>6} | {'V-F1':>6} | {'Best':>5}")
    print("-" * 55)

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        tr_loss = 0.0
        for xb, ytb, ysb, yrb, ycb in train_loader:
            xb  = xb.to(DEVICE);  ytb = ytb.to(DEVICE)
            ysb = ysb.to(DEVICE); yrb = yrb.to(DEVICE); ycb = ycb.to(DEVICE)

            dis_logits, sev, risk, cas, _ = model(xb)
            loss  = ce_loss(dis_logits, ytb)
            loss += mse_loss(sev, ysb)
            loss += mse_loss(risk, yrb)
            loss += mse_loss(cas, ycb)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()

        scheduler.step()

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        va_loss   = 0.0
        all_pred, all_true = [], []
        with torch.no_grad():
            for xb, ytb, ysb, yrb, ycb in val_loader:
                xb  = xb.to(DEVICE);  ytb = ytb.to(DEVICE)
                ysb = ysb.to(DEVICE); yrb = yrb.to(DEVICE); ycb = ycb.to(DEVICE)
                dis_logits, sev, risk, cas, _ = model(xb)
                loss  = ce_loss(dis_logits, ytb)
                loss += mse_loss(sev, ysb)
                loss += mse_loss(risk, yrb)
                loss += mse_loss(cas, ycb)
                va_loss += loss.item()
                all_pred.extend(dis_logits.argmax(1).cpu().tolist())
                all_true.extend(ytb.cpu().tolist())

        avg_tr  = tr_loss / len(train_loader)
        avg_va  = va_loss / len(val_loader)
        acc     = accuracy_score(all_true, all_pred)
        f1      = f1_score(all_true, all_pred, average="macro", zero_division=0)
        is_best = f1 > best_f1

        if is_best:
            best_f1 = f1
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "group_size":       GROUP_SIZE,
                    "hidden_dim":       HIDDEN_DIM,
                    "n_disaster_types": len(DISASTER_TYPES),
                    "disaster_types":   DISASTER_TYPES,
                    "total_features":   TOTAL_FEATURES,
                },
                "val_f1":  best_f1,
                "val_acc": acc,
            }, save_path)

        marker = " ✓" if is_best else ""
        print(f"{epoch:5d} | {avg_tr:8.4f} | {avg_va:8.4f} | "
              f"{acc:6.4f} | {f1:6.4f} |{marker}")

    # ── Final report ───────────────────────────────────────────────────────
    print(f"\nBest val macro-F1: {best_f1:.4f}")
    print("\nPer-class report (validation):")
    print(classification_report(all_true, all_pred,
                                target_names=DISASTER_TYPES, zero_division=0))

    size_mb = os.path.getsize(save_path) / 1024 / 1024
    print(f"Saved → {save_path}  ({size_mb:.2f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    train()
