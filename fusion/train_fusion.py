"""
Fusion Layer Training
======================
Generates IoT embeddings from the trained AdaptiveIoTClassifier by
creating per-event sensor vectors and running them through the model.
Uses the CrisisMMD humanitarian split (13K+ samples) as training signal.

Run AFTER training the IoT model:
    python IOT/train_iot.py
    python fusion/train_fusion.py
"""

import os, sys, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def _clamp(v, lo=0.0, hi=1.0):
    return float(max(lo, min(hi, v)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

from fusion.fusion_layer import FusionLayer, PRIORITY_LABELS, DISASTER_LABELS, RESOURCE_NAMES
from IOT.train_iot import (
    AdaptiveIoTClassifier,
    features_from_fire, features_from_storm_hist,
    features_from_eq_iran, features_from_flood,
    HIDDEN_DIM, GROUP_SIZE, DISASTER_TYPES,
)

CRISIS_ROOT  = os.path.join(os.path.dirname(__file__), "..", "crisis", "Dataset", "CrisisMMD_v2.0")
SPLITS_DIR   = os.path.join(CRISIS_ROOT, "crisismmd_datasplit_all", "crisismmd_datasplit_all")
IOT_CKPT     = os.path.join(os.path.dirname(__file__), "..", "IOT", "models", "iot_model.pth")
SAVE_PATH    = os.path.join(os.path.dirname(__file__), "fusion_model.pth")

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CRISIS_DIM   = 1024     # AdaptiveFusionClassifier: 512 (vision) + 512 (text)
IOT_EMB_DIM  = HIDDEN_DIM   # 128
BATCH_SIZE   = 128
EPOCHS       = 40
LR           = 1e-3

EVENT_TO_TYPE = {
    "california_wildfires":  "fire",
    "hurricane_harvey":      "storm",
    "hurricane_irma":        "storm",
    "hurricane_maria":       "storm",
    "iraq_iran_earthquake":  "earthquake",
    "mexico_earthquake":     "earthquake",
    "srilanka_floods":       "flood",
}
LABEL_TO_PRIORITY = {
    "affected_individuals":                  2,
    "infrastructure_and_utility_damage":     2,
    "rescue_volunteering_or_donation_effort":3,
    "not_humanitarian":                      0,
    "other_relevant_information":            1,
    "vehicle_damage":                        1,
    "missing_or_found_people":               3,
}
LABEL_TO_POPULATION = {
    "not_humanitarian":                       0.02,
    "other_relevant_information":             0.20,
    "vehicle_damage":                         0.25,
    "affected_individuals":                   0.65,
    "infrastructure_and_utility_damage":      0.55,
    "rescue_volunteering_or_donation_effort": 0.75,
    "missing_or_found_people":                0.85,
}
# Base resource needs [water, medical, rescue, shelter] per disaster type
DISASTER_RESOURCES = {
    "fire":       [0.35, 0.45, 0.65, 0.40],
    "storm":      [0.65, 0.55, 0.75, 0.80],
    "earthquake": [0.75, 0.85, 0.90, 0.70],
    "flood":      [0.90, 0.50, 0.75, 0.60],
    "unknown":    [0.10, 0.10, 0.10, 0.10],
}
DISASTER_TO_IDX = {d: i for i, d in enumerate(DISASTER_LABELS)}


def _r():
    return random.random()


def generate_sensor_features(event: str, label: str) -> list:
    """Generate a realistic 32-dim sensor vector for the given CrisisMMD event."""
    dtype = EVENT_TO_TYPE.get(event, "unknown")
    import pandas as pd

    if dtype == "fire":
        row = pd.Series({
            "PRECIPITATION":      max(0, _r() * 3),
            "MAX_TEMP":           38 + _r() * 15,
            "MIN_TEMP":           18 + _r() * 10,
            "AVG_WIND_SPEED":     10 + _r() * 25,
            "TEMP_RANGE":         20 + _r() * 10,
            "WIND_TEMP_RATIO":    0.3 + _r() * 0.4,
            "MONTH":              random.randint(7, 11),
            "LAGGED_PRECIPITATION": max(0, _r() * 2),
        })
        return features_from_fire(row)

    elif dtype == "storm":
        cat_map = {
            "affected_individuals": "H3",
            "infrastructure_and_utility_damage": "H4",
            "rescue_volunteering_or_donation_effort": "H5",
        }
        cat = cat_map.get(label, "H2")
        row = pd.Series({
            "WIND_KTS":   100 + _r() * 100,
            "PRESSURE":   920 + _r() * 60,
            "LAT":        15 + _r() * 20,
            "LONG":       -80 + _r() * 20,
            "CAT":        cat,
            "MONTH":      random.randint(8, 10),
            "Shape_Leng": 1.0 + _r() * 4,
        })
        return features_from_storm_hist(row)

    elif dtype == "earthquake":
        mag_base = 5.5 if "damage" in label or "affected" in label else 4.5
        row = pd.Series({
            "Lat":                  30 + _r() * 10,
            "Long":                 44 + _r() * 20,
            "Depth":                8 + _r() * 30,
            "Magnitude":            mag_base + _r() * 2,
            "RMS":                  0.1 + _r() * 0.4,
            "Number of stations":   int(5 + _r() * 50),
            "Number of phases":     int(10 + _r() * 80),
            "Azimuth GAP":          int(100 + _r() * 180),
            "Date":                 "2017/11/12",
        })
        return features_from_eq_iran(row)

    elif dtype == "flood":
        risk_base = 60.0 if "damage" in label or "affected" in label else 40.0
        row = pd.Series({
            "latitude":             6 + _r() * 4,
            "longitude":            80 + _r() * 4,
            "elevation_m":          int(5 + _r() * 40),
            "distance_to_river_m":  50 + _r() * 500,
            "rainfall_7d_mm":       100 + _r() * 200,
            "monthly_rainfall_mm":  200 + _r() * 400,
            "drainage_index":       0.2 + _r() * 0.4,
            "ndvi":                 -0.3 + _r() * 0.4,
            "ndwi":                 0.1 + _r() * 0.5,
            "historical_flood_count": random.randint(0, 5),
            "flood_risk_score":     risk_base + _r() * 30,
        })
        return features_from_flood(row)

    else:
        return [0.0] * 32


def generate_crisis_embedding(event: str, label: str) -> torch.Tensor:
    """Synthetic 1024-dim crisis embedding (vision_proj + text_proj block pattern)."""
    dtype    = EVENT_TO_TYPE.get(event, "unknown")
    bias     = torch.zeros(CRISIS_DIM)
    t_idx    = DISASTER_LABELS.index(dtype) if dtype in DISASTER_LABELS else 4
    block    = CRISIS_DIM // 5
    bias[t_idx * block: (t_idx + 1) * block] = 1.5
    if "damage" in label or "infrastructure" in label:
        bias[CRISIS_DIM // 2:] += 0.8
    if "rescue" in label or "volunteering" in label:
        bias[:CRISIS_DIM // 4] += 1.0
    return bias + torch.randn(CRISIS_DIM) * 0.3


def build_training_data(iot_model: AdaptiveIoTClassifier):
    """Extract real 128-dim IoT embeddings + synthetic crisis embeddings from CrisisMMD."""
    path = os.path.join(SPLITS_DIR, "task_humanitarian_text_img_train.tsv")
    df   = pd.read_csv(path, sep="\t").dropna(subset=["label", "event_name"])
    print(f"  Loaded {len(df)} CrisisMMD samples")

    iot_embs, crisis_embs = [], []
    sev_targets, pri_targets, dis_targets = [], [], []
    pop_targets, res_targets = [], []

    iot_model.eval()
    with torch.no_grad():
        for _, row in df.iterrows():
            event = str(row["event_name"])
            label = str(row["label"])

            # Real IoT embedding from trained model
            feats    = generate_sensor_features(event, label)
            x_tensor = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
            _, sev, risk, cas, emb = iot_model(x_tensor)
            iot_embs.append(emb.squeeze(0))

            # Synthetic crisis embedding
            crisis_embs.append(generate_crisis_embedding(event, label))

            # Severity target
            sev_val = 0.3
            if "damage" in label or "affected" in label:
                sev_val = 0.65 + _r() * 0.35
            elif "rescue" in label:
                sev_val = 0.55 + _r() * 0.2
            elif "not_humanitarian" in label:
                sev_val = _r() * 0.15
            sev_targets.append(sev_val)

            pri_targets.append(LABEL_TO_PRIORITY.get(label, 1))
            dtype = EVENT_TO_TYPE.get(event, "unknown")
            dis_targets.append(DISASTER_TO_IDX.get(dtype, 4))

            # Population impact target
            pop_base = LABEL_TO_POPULATION.get(label, 0.2)
            pop_targets.append(_clamp(pop_base + (_r() - 0.5) * 0.1))

            # Resource needs target — base per disaster type, scaled by severity
            base_res = DISASTER_RESOURCES.get(dtype, [0.1, 0.1, 0.1, 0.1])
            res_scaled = [_clamp(v * (0.7 + sev_val * 0.6)) for v in base_res]
            res_targets.append(res_scaled)

    return (
        torch.stack(iot_embs),
        torch.stack(crisis_embs),
        torch.tensor(sev_targets, dtype=torch.float32),
        torch.tensor(pri_targets, dtype=torch.long),
        torch.tensor(dis_targets, dtype=torch.long),
        torch.tensor(pop_targets, dtype=torch.float32),
        torch.tensor(res_targets, dtype=torch.float32),
    )


def train():
    print(f"Device: {DEVICE}")

    # Load trained IoT model
    print("Loading AdaptiveIoTClassifier…")
    ckpt = torch.load(IOT_CKPT, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]
    iot_model = AdaptiveIoTClassifier(
        group_size       = cfg["group_size"],
        hidden_dim       = cfg["hidden_dim"],
        n_disaster_types = cfg["n_disaster_types"],
    )
    iot_model.load_state_dict(ckpt["model_state_dict"])
    print(f"  IoT model loaded  (val F1={ckpt.get('val_f1', 'N/A'):.4f})")

    print("Building fusion training data…")
    iot_X, crisis_X, sev_Y, pri_Y, dis_Y, pop_Y, res_Y = build_training_data(iot_model)
    print(f"  {len(iot_X)} samples  |  IoT emb: {iot_X.shape}  |  Crisis emb: {crisis_X.shape}")

    n     = len(iot_X)
    idx   = torch.randperm(n)
    split = int(n * 0.8)
    tr, va = idx[:split], idx[split:]

    def loader(idx, shuffle=False):
        ds = TensorDataset(
            iot_X[idx], crisis_X[idx],
            sev_Y[idx], pri_Y[idx], dis_Y[idx],
            pop_Y[idx], res_Y[idx],
        )
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = loader(tr, shuffle=True)
    val_loader   = loader(va)

    model = FusionLayer(
        iot_dim    = IOT_EMB_DIM,
        crisis_dim = CRISIS_DIM,
        proj_dim   = 256,
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    mse_loss  = nn.MSELoss()
    ce_loss   = nn.CrossEntropyLoss()

    best_val  = float("inf")
    print(f"\nTraining FusionLayer for {EPOCHS} epochs…")
    print(f"{'Epoch':>5} | {'T-loss':>8} | {'V-loss':>8} | {'Pri-acc':>7} | {'Dis-acc':>7}")
    print("-" * 50)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for ib, cb, sb, pb, db, popb, resb in train_loader:
            ib, cb   = ib.to(DEVICE), cb.to(DEVICE)
            sb, pb   = sb.to(DEVICE), pb.to(DEVICE)
            db       = db.to(DEVICE)
            popb     = popb.to(DEVICE)
            resb     = resb.to(DEVICE)
            out  = model(ib, cb)
            loss = mse_loss(out.severity_score, sb) + \
                   ce_loss(out.priority_logits, pb) + \
                   ce_loss(out.disaster_logits, db) + \
                   mse_loss(out.population_impact, popb) + \
                   mse_loss(out.resource_needs, resb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
        scheduler.step()

        model.eval()
        va_loss = 0.0
        c_pri = c_dis = total = 0
        with torch.no_grad():
            for ib, cb, sb, pb, db, popb, resb in val_loader:
                ib, cb   = ib.to(DEVICE), cb.to(DEVICE)
                sb, pb   = sb.to(DEVICE), pb.to(DEVICE)
                db       = db.to(DEVICE)
                popb     = popb.to(DEVICE)
                resb     = resb.to(DEVICE)
                out  = model(ib, cb)
                loss = mse_loss(out.severity_score, sb) + \
                       ce_loss(out.priority_logits, pb) + \
                       ce_loss(out.disaster_logits, db) + \
                       mse_loss(out.population_impact, popb) + \
                       mse_loss(out.resource_needs, resb)
                va_loss += loss.item()
                c_pri  += (out.priority_logits.argmax(1) == pb).sum().item()
                c_dis  += (out.disaster_logits.argmax(1) == db).sum().item()
                total  += len(sb)

        avg_tr  = tr_loss / len(train_loader)
        avg_va  = va_loss / len(val_loader)
        p_acc   = c_pri / total
        d_acc   = c_dis / total
        marker  = ""
        if avg_va < best_val:
            best_val = avg_va
            torch.save(model.state_dict(), SAVE_PATH)
            marker = " ✓"
        print(f"{epoch:5d} | {avg_tr:8.4f} | {avg_va:8.4f} | "
              f"{p_acc:7.3f} | {d_acc:7.3f}{marker}")

    print(f"\nBest val loss: {best_val:.4f}")
    print(f"Fusion model saved → {SAVE_PATH}")


if __name__ == "__main__":
    train()
