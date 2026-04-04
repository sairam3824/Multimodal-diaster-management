"""
Unified Disaster Response Pipeline (Tri-Fusion)
=================================================
Combines:
  * AdaptiveIoTClassifier  — auto-detects disaster type from sensor data
  * AdaptiveFusionClassifier (BLIP + XLM-RoBERTa) — social media analysis
  * DeepLabV3Plus_xBD — satellite damage assessment
  * TriFusionLayer (cross-modal attention) — merges all modalities

Modality requirements:
  - Crisis (image + tweet) : MANDATORY
  - IoT (sensor readings)  : OPTIONAL
  - Satellite (xBD image)  : OPTIONAL

Supported configurations:
  1. crisis only
  2. crisis + iot
  3. crisis + satellite
  4. crisis + iot + satellite

Usage:
    from fusion.pipeline import DisasterPipeline
    pipe = DisasterPipeline()
    result = pipe.analyze(
        image_path="path/to/tweet_image.jpg",
        tweet="Flooding on main street!",
        satellite_image_path="path/to/satellite.png",  # optional
        lat=6.5, lon=80.1,
        rainfall_7d=200,
    )
    print(result)
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from PIL import Image
from torchvision import transforms

from fusion.tri_fusion_layer import TriFusionLayer, TriFusionOutput
from fusion.fusion_layer import FusionLayer, PRIORITY_LABELS, DISASTER_LABELS, RESOURCE_NAMES
from fusion.iot_predictor import IoTPredictor, IoTPrediction

CRISIS_DIR       = os.path.join(os.path.dirname(__file__), "..", "crisis")
FUSION_MODEL     = os.path.join(os.path.dirname(__file__), "fusion_model.pth")
TRI_FUSION_MODEL = os.path.join(os.path.dirname(__file__), "tri_fusion_model.pth")
IOT_EMB_DIM      = 128
CRISIS_DIM       = 1024    # 512 (vision) + 512 (text) from AdaptiveFusionClassifier
SATELLITE_DIM    = 640     # 512 (F_sat) + 128 (F_region)
INPUT_SIZE       = (224, 224)

CRISIS_CLASSES = [
    "affected_individuals",
    "infrastructure_and_utility_damage",
    "not_humanitarian",
    "other_relevant_information",
    "rescue_volunteering_or_donation_effort",
]


@dataclass
class DisasterAssessment:
    # -- IoT --
    iot_disaster_type:   str
    iot_type_probs:      Dict[str, float]
    iot_severity:        float
    iot_fire_prob:       float
    iot_storm_cat:       float
    iot_eq_magnitude:    float
    iot_flood_risk:      float
    iot_casualty_risk:   float
    iot_sensor_weights:  Dict[str, float]
    # -- Crisis --
    crisis_category:     str
    crisis_confidence:   float
    crisis_probs:        Dict[str, float]
    vision_weight:       float
    text_weight:         float
    # -- Satellite --
    satellite_damage:    Optional[str] = None
    satellite_damage_probs: Optional[Dict[str, float]] = None
    # -- Fusion --
    fused_severity:      float = 0.0
    priority:            str = "Low"
    priority_probs:      Dict[str, float] = field(default_factory=dict)
    confirmed_type:      str = "unknown"
    type_probs:          Dict[str, float] = field(default_factory=dict)
    population_impact:   float = 0.0
    resource_needs:      Dict[str, float] = field(default_factory=dict)
    modality_weights:    Dict[str, float] = field(default_factory=dict)
    # -- Meta --
    active_modalities:   List[str] = field(default_factory=list)
    alert_level:         str = ""
    summary:             str = ""

    def __str__(self):
        no_sensor = self.iot_disaster_type == "unknown" and self.iot_severity == 0.0
        iot_lines = (
            ["  IoT Sensor Analysis : no sensor data provided"]
            if no_sensor else [
                "  IoT Sensor Analysis (auto-detected):",
                f"    Detected type : {self.iot_disaster_type}",
                f"    Dominant group: {max(self.iot_sensor_weights, key=self.iot_sensor_weights.get)} "
                f"({max(self.iot_sensor_weights.values()):.0%} weight)",
                f"    Fire prob     : {self.iot_fire_prob:.3f}",
                f"    Storm cat     : {self.iot_storm_cat * 5:.1f} / 5",
                f"    EQ magnitude  : {self.iot_eq_magnitude * 9:.1f} M",
                f"    Flood risk    : {self.iot_flood_risk * 100:.0f} / 100",
                f"    Casualty risk : {self.iot_casualty_risk:.3f}",
            ]
        )
        sat_lines = (
            ["  Satellite Analysis  : no satellite data provided"]
            if self.satellite_damage is None else [
                "  Satellite Damage Assessment:",
                f"    Predicted     : {self.satellite_damage}",
                f"    Probabilities : {self.satellite_damage_probs}",
            ]
        )
        res = self.resource_needs
        modw = self.modality_weights
        lines = [
            "=" * 58,
            "  UNIFIED DISASTER RESPONSE ASSESSMENT (TRI-FUSION)",
            "=" * 58,
            f"  Alert Level      : {self.alert_level}",
            f"  Disaster Type    : {self.confirmed_type}",
            f"  Priority         : {self.priority}",
            f"  Fused Severity   : {self.fused_severity:.2f}",
            f"  Population Impact: {self.population_impact:.3f}",
            f"  Resources Needed : water={res.get('water',0):.2f}  "
            f"medical={res.get('medical',0):.2f}  "
            f"rescue={res.get('rescue',0):.2f}  "
            f"shelter={res.get('shelter',0):.2f}",
            f"  Active Modalities: {', '.join(self.active_modalities)}",
            f"  Modality Weights : " + "  ".join(f"{k}={v:.2f}" for k, v in modw.items()) if modw else "",
            "",
            *iot_lines,
            "",
            *sat_lines,
            "",
            "  Crisis Social Media Analysis:",
            f"    Category      : {self.crisis_category} ({self.crisis_confidence:.1%})",
            f"    Vision weight : {self.vision_weight:.2f}",
            f"    Text weight   : {self.text_weight:.2f}",
            "",
            f"  Summary: {self.summary}",
            "=" * 58,
        ]
        return "\n".join(lines)


_PRIORITY_SCORE = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}


def _alert_level(severity: float, priority: str) -> str:
    pri_idx = _PRIORITY_SCORE.get(priority, 0)
    score   = (severity + pri_idx / 3.0) / 2
    if score >= 0.75: return "RED"
    if score >= 0.5:  return "ORANGE"
    if score >= 0.25: return "YELLOW"
    return "GREEN"


_DISASTER_KEYWORDS = {
    "fire": ["fire", "wildfire", "blaze", "burning", "arson", "smoke", "flames",
             "bushfire", "forest fire", "inferno", "charred", "ember", "burnt",
             "scorched", "ash"],
    "storm": ["storm", "hurricane", "typhoon", "cyclone", "tornado",
              "irma", "harvey", "maria", "dorian", "katrina", "ian",
              "sandy", "helene", "haiyan", "matthew",
              "tropical storm", "tropical depression",
              "blown", "ripped", "uprooted",
              "power outage", "powerline"],
    "earthquake": ["earthquake", "quake", "seismic", "tremor", "aftershock",
                   "magnitude", "richter", "epicenter", "collapsed", "rubble",
                   "crumbled", "shaking", "fault line"],
    "flood": ["flood", "flooding", "submerged", "inundation", "deluge",
              "water level", "overflow", "flash flood", "waterlogged",
              "underwater", "washed away", "rising water", "levee"],
}

_STORM_CONTEXTUAL = ["damage", "damaged", "destruction", "destroyed", "debris",
                     "wreckage", "demolished", "devastat", "wind", "winds",
                     "gust", "rooftop", "roof", "trees down", "category"]


def _infer_disaster_type_from_text(tweet: str, caption: str = "") -> str:
    combined = f"{tweet} {caption}".lower()
    scores = {"fire": 0, "storm": 0, "earthquake": 0, "flood": 0}
    for disaster, keywords in _DISASTER_KEYWORDS.items():
        for kw in keywords:
            if kw in combined:
                scores[disaster] += 2
    storm_contextual_hits = sum(1 for kw in _STORM_CONTEXTUAL if kw in combined)
    scores["storm"] += storm_contextual_hits
    max_score = max(scores.values())
    if max_score == 0:
        return "unknown"
    top_types = [t for t, s in scores.items() if s == max_score]
    if len(top_types) == 1:
        return top_types[0]
    strong = {}
    for t in top_types:
        strong[t] = sum(2 for kw in _DISASTER_KEYWORDS[t] if kw in combined)
    return max(strong, key=strong.get)


def _summary(a: DisasterAssessment, iot_available: bool, sat_available: bool) -> str:
    if not iot_available and not sat_available:
        type_note = ""
        if a.confirmed_type != "unknown":
            type_note = f" Inferred hazard: {a.confirmed_type}."
        return (
            f"No sensor/satellite data provided. Social media analysis: '{a.crisis_category}' "
            f"({a.crisis_confidence:.1%} confidence).{type_note} Priority: {a.priority}."
        )
    parts = []
    if iot_available:
        if a.iot_fire_prob > 0.5:
            parts.append(f"wildfire conditions ({a.iot_fire_prob:.0%})")
        if a.iot_storm_cat * 5 >= 1:
            parts.append(f"Category {a.iot_storm_cat * 5:.0f} storm")
        if a.iot_eq_magnitude * 9 >= 4.0:
            parts.append(f"M{a.iot_eq_magnitude * 9:.1f} earthquake")
        if a.iot_flood_risk * 100 > 40:
            parts.append(f"flood risk {a.iot_flood_risk * 100:.0f}/100")
    if sat_available and a.satellite_damage:
        parts.append(f"satellite: {a.satellite_damage}")
    parts.append(f"social media: '{a.crisis_category}'")
    return f"Sensors detect {'; '.join(parts)}. Priority: {a.priority}."


class DisasterPipeline:
    IMG_TRANSFORM = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def __init__(
        self,
        load_crisis_model: bool = True,
        load_satellite_model: bool = True,
        device: Optional[str] = None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"[Pipeline] Device: {self.device}")

        # -- IoT --
        print("[Pipeline] Loading IoT model...")
        self.iot = IoTPredictor()

        # -- Crisis --
        self._crisis_model = None
        self._crisis_tokenizer = None
        self._blip_captioner = None
        self._blip_processor = None
        if load_crisis_model:
            self._load_crisis_model()

        # -- Satellite --
        self._satellite_model = None
        self._satellite_predictor = None
        if load_satellite_model:
            self._load_satellite_model()

        # -- Tri-Fusion --
        print("[Pipeline] Loading TriFusionLayer...")
        self.tri_fusion = TriFusionLayer(
            crisis_dim=CRISIS_DIM,
            iot_dim=IOT_EMB_DIM,
            satellite_dim=SATELLITE_DIM,
            proj_dim=256,
        ).to(self.device)

        if os.path.exists(TRI_FUSION_MODEL):
            state = torch.load(TRI_FUSION_MODEL, map_location=self.device, weights_only=True)
            self.tri_fusion.load_state_dict(state)
            print("[Pipeline] Tri-fusion model loaded.")
        elif os.path.exists(FUSION_MODEL):
            # Attempt partial load from old dual-fusion model
            from fusion.tri_fusion_layer import upgrade_fusion_state_dict
            old_state = torch.load(FUSION_MODEL, map_location=self.device, weights_only=True)
            partial = upgrade_fusion_state_dict(old_state)
            if partial:
                self.tri_fusion.load_state_dict(partial, strict=False)
                print("[Pipeline] Partial dual-fusion weights loaded into tri-fusion.")
            else:
                print("[Pipeline] WARNING: no compatible weights found in old fusion model.")
        else:
            print("[Pipeline] WARNING: no fusion model found — run train_tri_fusion.py")
        self.tri_fusion.eval()

        # -- Legacy dual-fusion (backward compat) --
        self.fusion = FusionLayer(
            iot_dim=IOT_EMB_DIM, crisis_dim=CRISIS_DIM, proj_dim=256
        ).to(self.device)
        if os.path.exists(FUSION_MODEL):
            state = torch.load(FUSION_MODEL, map_location=self.device, weights_only=True)
            self.fusion.load_state_dict(state)
        self.fusion.eval()

    def _load_crisis_model(self):
        try:
            sys.path.insert(0, CRISIS_DIR)
            from server import AdaptiveFusionClassifier
            from transformers import (
                BlipForConditionalGeneration, BlipProcessor,
                XLMRobertaTokenizer, XLMRobertaModel,
            )
            print("[Pipeline] Loading BLIP + XLM-RoBERTa...")
            blip = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            self._blip_captioner = blip
            self._blip_processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            xlm_tok = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
            xlm_model = XLMRobertaModel.from_pretrained("xlm-roberta-base")
            model = AdaptiveFusionClassifier(
                blip_model=blip, xlm_model=xlm_model,
                hidden_dim=512, n_classes=len(CRISIS_CLASSES)
            )
            ckpt_path = os.path.join(CRISIS_DIR, "best_adaptive_model.pth")
            try:
                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            except Exception:
                print("[Pipeline] Warning: weights_only=True failed, using unsafe load")
                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict(sd, strict=False)
            model.to(self.device).eval()
            self._crisis_model = model
            self._crisis_tokenizer = xlm_tok
            print("[Pipeline] Crisis model ready (with BLIP captioning).")
        except Exception as e:
            print(f"[Pipeline] WARNING: Could not load crisis model: {e}")
            self._blip_captioner = None
            self._blip_processor = None

    def _load_satellite_model(self):
        try:
            from XBD.xbd_model import SatellitePredictor
            xbd_pkl = os.path.join(os.path.dirname(__file__), "..", "XBD", "deeplabv3plus_xbd_trained.pkl")
            if os.path.exists(xbd_pkl):
                self._satellite_predictor = SatellitePredictor(xbd_pkl, self.device)
                self._satellite_model = self._satellite_predictor.model
                print("[Pipeline] Satellite (xBD) model ready.")
            else:
                print("[Pipeline] WARNING: xBD .pkl not found — satellite modality disabled.")
        except Exception as e:
            print(f"[Pipeline] WARNING: Could not load satellite model: {e}")

    def _generate_caption(self, image_path: str) -> str:
        if self._blip_captioner is None or self._blip_processor is None:
            return ""
        try:
            img = Image.open(image_path).convert("RGB")
            inputs = self._blip_processor(
                images=img, text="a photograph of", return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                out = self._blip_captioner.generate(**inputs, max_new_tokens=50, num_beams=3)
            return self._blip_processor.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            print(f"[Pipeline] Caption generation failed: {e}")
            return ""

    def _run_crisis(self, image_path: str, tweet: str):
        if self._crisis_model is None:
            n = len(CRISIS_CLASSES)
            probs = {c: 1.0 / n for c in CRISIS_CLASSES}
            return probs, torch.zeros(CRISIS_DIM), 0.5, 0.5

        try:
            img = Image.open(image_path).convert("RGB")
            img_t = self.IMG_TRANSFORM(img).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"[Pipeline] Warning: failed to load image {image_path}: {e}")
            img_t = torch.zeros(1, 3, *INPUT_SIZE).to(self.device)

        enc = self._crisis_tokenizer(
            tweet, max_length=128, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            logits, attn = self._crisis_model(img_t, ids, mask, return_attention=True)
            vis_feat = self._crisis_model.vision_encoder(img_t).last_hidden_state[:, 0, :]
            txt_feat = self._crisis_model.text_encoder(
                input_ids=ids, attention_mask=mask
            ).last_hidden_state[:, 0, :]
            vis_proj = self._crisis_model.vision_proj(vis_feat)
            txt_proj = self._crisis_model.text_proj(txt_feat)
            emb = torch.cat([vis_proj, txt_proj], dim=1).squeeze(0)
            temperature = 2.5
            probs_t = F.softmax(logits / temperature, dim=-1).squeeze(0).cpu()
            probs = {CRISIS_CLASSES[i]: float(probs_t[i]) for i in range(len(CRISIS_CLASSES))}
            vis_w = float(attn["vision_weight"].cpu().item())
            txt_w = float(attn["text_weight"].cpu().item())

        return probs, emb.cpu(), vis_w, txt_w

    def _run_satellite(self, satellite_image_path: str):
        """Run satellite model and return embedding + prediction dict."""
        if self._satellite_predictor is None:
            return None
        try:
            result = self._satellite_predictor.predict(satellite_image_path)
            print(f"[Pipeline] Satellite damage: {result['damage_class']}")
            return result
        except Exception as e:
            print(f"[Pipeline] WARNING: Satellite inference failed: {e}")
            return None

    def analyze(
        self,
        image_path: str,
        tweet: str,
        crisis_embedding: Optional[torch.Tensor] = None,
        satellite_image_path: Optional[str] = None,
        satellite_embedding: Optional[torch.Tensor] = None,
        **sensor_kwargs,
    ) -> DisasterAssessment:
        """
        Full tri-fusion pipeline inference.

        Parameters
        ----------
        image_path            : path to the tweet/social media image (REQUIRED)
        tweet                 : tweet text (REQUIRED)
        crisis_embedding      : optional pre-computed [1024] tensor
        satellite_image_path  : path to satellite image (OPTIONAL)
        satellite_embedding   : optional pre-computed [640] tensor
        **sensor_kwargs       : raw sensor readings forwarded to IoTPredictor
        """
        active_modalities = ["crisis"]

        # -- Step 1: IoT --
        from IOT.train_iot import (
            features_from_fire, features_from_storm_hist,
            features_from_eq_iran, features_from_flood,
        )
        import pandas as pd

        def _pd(d): return pd.Series(d)

        candidates = []

        if any(k in sensor_kwargs for k in ["precipitation", "max_temp", "avg_wind_speed"]):
            row = _pd({
                "PRECIPITATION":       sensor_kwargs.get("precipitation", 0),
                "MAX_TEMP":            sensor_kwargs.get("max_temp", 30),
                "MIN_TEMP":            sensor_kwargs.get("min_temp", 15),
                "AVG_WIND_SPEED":      sensor_kwargs.get("avg_wind_speed", 5),
                "TEMP_RANGE":          sensor_kwargs.get("max_temp", 30) - sensor_kwargs.get("min_temp", 15),
                "WIND_TEMP_RATIO":     sensor_kwargs.get("avg_wind_speed", 5) / (sensor_kwargs.get("max_temp", 30) + 1e-6),
                "MONTH":               sensor_kwargs.get("month", 6),
                "LAGGED_PRECIPITATION":sensor_kwargs.get("lagged_precipitation", 0),
            })
            candidates.append(self.iot.predict_from_features(features_from_fire(row)))

        if any(k in sensor_kwargs for k in ["wind_kts", "pressure"]):
            row = _pd({
                "WIND_KTS":    sensor_kwargs.get("wind_kts", 0),
                "PRESSURE":    sensor_kwargs.get("pressure", 1013),
                "LAT":         sensor_kwargs.get("lat", 0),
                "LONG":        sensor_kwargs.get("lon", 0),
                "CAT":         sensor_kwargs.get("category", "TS"),
                "MONTH":       sensor_kwargs.get("month", 6),
                "Shape_Leng":  sensor_kwargs.get("shape_leng", 1.0),
            })
            candidates.append(self.iot.predict_from_features(features_from_storm_hist(row)))

        if any(k in sensor_kwargs for k in ["depth", "rms", "n_stations"]):
            row = _pd({
                "Lat":                  sensor_kwargs.get("lat", 0),
                "Long":                 sensor_kwargs.get("lon", 0),
                "Depth":                sensor_kwargs.get("depth", 10),
                "Magnitude":            sensor_kwargs.get("magnitude", 0),
                "RMS":                  sensor_kwargs.get("rms", 0.2),
                "Number of stations":   sensor_kwargs.get("n_stations", 10),
                "Number of phases":     sensor_kwargs.get("n_phases", 20),
                "Azimuth GAP":          sensor_kwargs.get("azimuth_gap", 180),
                "Date":                 f"2024/{sensor_kwargs.get('month', 6):02d}/01",
            })
            candidates.append(self.iot.predict_from_features(features_from_eq_iran(row)))

        if any(k in sensor_kwargs for k in ["rainfall_7d", "elevation_m", "ndvi"]):
            row = _pd({
                "latitude":              sensor_kwargs.get("lat", 0),
                "longitude":             sensor_kwargs.get("lon", 0),
                "elevation_m":           sensor_kwargs.get("elevation_m", 50),
                "distance_to_river_m":   sensor_kwargs.get("distance_to_river_m", 500),
                "rainfall_7d_mm":        sensor_kwargs.get("rainfall_7d", 0),
                "monthly_rainfall_mm":   sensor_kwargs.get("monthly_rainfall", 0),
                "drainage_index":        sensor_kwargs.get("drainage_index", 0.5),
                "ndvi":                  sensor_kwargs.get("ndvi", 0.3),
                "ndwi":                  sensor_kwargs.get("ndwi", 0.1),
                "historical_flood_count":sensor_kwargs.get("historical_flood_count", 0),
                "flood_risk_score":      0.0,
            })
            candidates.append(self.iot.predict_from_features(features_from_flood(row)))

        iot_available = len(candidates) > 0
        iot = max(candidates, key=lambda p: p.severity_score) if iot_available else None
        if iot_available:
            active_modalities.append("iot")
            print(f"[Pipeline] IoT active: {iot.disaster_type}")

        # -- Step 2: Crisis --
        if crisis_embedding is not None:
            crisis_emb = crisis_embedding
            crisis_probs = {c: 1.0 / len(CRISIS_CLASSES) for c in CRISIS_CLASSES}
            vision_w = text_w = 0.5
        else:
            crisis_probs, crisis_emb, vision_w, text_w = self._run_crisis(image_path, tweet)

        top_crisis = max(crisis_probs, key=crisis_probs.get)
        top_conf = crisis_probs[top_crisis]

        # -- Step 3: Satellite --
        sat_result = None
        sat_emb = None
        if satellite_embedding is not None:
            sat_emb = satellite_embedding
            active_modalities.append("satellite")
            print("[Pipeline] Satellite active (pre-computed embedding)")
        elif satellite_image_path is not None:
            sat_result = self._run_satellite(satellite_image_path)
            if sat_result is not None:
                sat_emb = sat_result["embedding"].squeeze(0)
                active_modalities.append("satellite")

        sat_available = sat_emb is not None

        # -- Step 4: Tri-Fusion --
        use_fusion = iot_available or sat_available
        if use_fusion:
            crisis_vec = crisis_emb.unsqueeze(0).to(self.device)
            iot_vec = iot.embedding.unsqueeze(0).to(self.device) if iot_available else None
            sat_vec = sat_emb.unsqueeze(0).to(self.device) if sat_available else None

            with torch.no_grad():
                out = self.tri_fusion(crisis_vec, iot_vec, sat_vec)

            fused_sev = float(out.severity_score.item())
            pri_probs = F.softmax(out.priority_logits, dim=-1).squeeze(0).cpu()
            dis_probs = F.softmax(out.disaster_logits, dim=-1).squeeze(0).cpu()
            priority = PRIORITY_LABELS[int(pri_probs.argmax())]
            conf_type = DISASTER_LABELS[int(dis_probs.argmax())]
            pop_impact = float(out.population_impact.item())
            res_needs = {RESOURCE_NAMES[i]: float(out.resource_needs[0, i])
                         for i in range(len(RESOURCE_NAMES))}
            modality_weights = {k: float(v.mean().item()) for k, v in out.modality_weights.items()}

            iot_disaster_type = iot.disaster_type if iot else "unknown"
            iot_type_probs = iot.type_probabilities if iot else {d: 0.2 for d in DISASTER_LABELS}
            iot_severity = iot.severity_score if iot else 0.0
            iot_fire_prob = iot.fire_prob if iot else 0.0
            iot_storm_cat = iot.storm_cat_norm if iot else 0.0
            iot_eq_magnitude = iot.eq_magnitude_norm if iot else 0.0
            iot_flood_risk = iot.flood_risk_norm if iot else 0.0
            iot_casualty_risk = iot.casualty_risk if iot else 0.0
            iot_sensor_weights = iot.sensor_weights if iot else {"weather": 0.25, "storm": 0.25, "seismic": 0.25, "hydro": 0.25}
            priority_probs = {PRIORITY_LABELS[i]: float(pri_probs[i]) for i in range(4)}
            type_probs = {DISASTER_LABELS[i]: float(dis_probs[i]) for i in range(5)}
        else:
            # Crisis-only fallback (no IoT, no satellite)
            fused_sev = 0.0
            crisis_to_priority = {
                "not_humanitarian":                       "Low",
                "other_relevant_information":             "Low",
                "vehicle_damage":                         "Medium",
                "affected_individuals":                   "High",
                "infrastructure_and_utility_damage":      "High",
                "rescue_volunteering_or_donation_effort": "Critical",
                "missing_or_found_people":                "Critical",
            }
            _CRISIS_SEVERITY_SCALE = {
                "not_humanitarian":                       0.0,
                "other_relevant_information":             0.25,
                "vehicle_damage":                         0.50,
                "affected_individuals":                   0.80,
                "infrastructure_and_utility_damage":      0.90,
                "rescue_volunteering_or_donation_effort": 0.95,
                "missing_or_found_people":                0.95,
            }
            priority = crisis_to_priority.get(top_crisis, "Low")
            caption = self._generate_caption(image_path)
            conf_type = _infer_disaster_type_from_text(tweet, caption)
            sev_scale = _CRISIS_SEVERITY_SCALE.get(top_crisis, 0.4)
            fused_sev = top_conf * sev_scale

            iot_disaster_type = "unknown"
            iot_type_probs = {d: 0.2 for d in DISASTER_LABELS}
            iot_severity = 0.0
            iot_fire_prob = 0.0
            iot_storm_cat = 0.0
            iot_eq_magnitude = 0.0
            iot_flood_risk = 0.0
            iot_casualty_risk = 0.0
            iot_sensor_weights = {"weather": 0.25, "storm": 0.25, "seismic": 0.25, "hydro": 0.25}
            priority_probs = {p: 0.25 for p in PRIORITY_LABELS}
            type_probs = {d: 0.2 for d in DISASTER_LABELS}
            modality_weights = {"crisis": 1.0, "iot": 0.0, "satellite": 0.0}

            _c = top_crisis
            _s = top_conf
            _CRISIS_POP = {
                "not_humanitarian":                       0.0,
                "other_relevant_information":             0.10,
                "affected_individuals":                   0.70,
                "infrastructure_and_utility_damage":      0.55,
                "rescue_volunteering_or_donation_effort": 0.65,
            }
            _CRISIS_RES = {
                "not_humanitarian":                       [0.0,  0.0,  0.0,  0.0],
                "other_relevant_information":             [0.10, 0.05, 0.05, 0.05],
                "affected_individuals":                   [0.50, 0.65, 0.60, 0.50],
                "infrastructure_and_utility_damage":      [0.40, 0.30, 0.55, 0.45],
                "rescue_volunteering_or_donation_effort": [0.45, 0.50, 0.75, 0.60],
            }
            _base_pop = _CRISIS_POP.get(_c, 0.10)
            _base_res = _CRISIS_RES.get(_c, [0.1, 0.1, 0.1, 0.1])
            pop_impact = round(_base_pop * _s, 4)
            res_needs = {RESOURCE_NAMES[i]: round(_base_res[i] * _s, 4)
                         for i in range(len(RESOURCE_NAMES))}

        # Satellite fields
        satellite_damage = None
        satellite_damage_probs = None
        if sat_result is not None:
            satellite_damage = sat_result["damage_class"]
            satellite_damage_probs = sat_result["damage_probs"]

        assessment = DisasterAssessment(
            iot_disaster_type=iot_disaster_type,
            iot_type_probs=iot_type_probs,
            iot_severity=iot_severity,
            iot_fire_prob=iot_fire_prob,
            iot_storm_cat=iot_storm_cat,
            iot_eq_magnitude=iot_eq_magnitude,
            iot_flood_risk=iot_flood_risk,
            iot_casualty_risk=iot_casualty_risk,
            iot_sensor_weights=iot_sensor_weights,
            crisis_category=top_crisis,
            crisis_confidence=top_conf,
            crisis_probs=crisis_probs,
            vision_weight=vision_w,
            text_weight=text_w,
            satellite_damage=satellite_damage,
            satellite_damage_probs=satellite_damage_probs,
            fused_severity=fused_sev,
            priority=priority,
            priority_probs=priority_probs,
            confirmed_type=conf_type,
            type_probs=type_probs,
            population_impact=pop_impact,
            resource_needs=res_needs,
            modality_weights=modality_weights,
            active_modalities=active_modalities,
            alert_level="",
            summary="",
        )

        if top_crisis == "not_humanitarian":
            assessment.alert_level = "GREEN"
        else:
            assessment.alert_level = _alert_level(fused_sev, priority)
        assessment.summary = _summary(assessment, iot_available, sat_available)
        return assessment


if __name__ == "__main__":
    import glob

    pipe = DisasterPipeline(load_crisis_model=False, load_satellite_model=False)

    pattern = os.path.join(CRISIS_DIR, "CrisisMMD_v2.0", "data_image",
                           "srilanka_floods", "**", "*.jpg")
    samples = glob.glob(pattern, recursive=True)
    img_path = samples[0] if samples else "/dev/null"

    result = pipe.analyze(
        image_path=img_path,
        tweet="Flooding in Colombo, roads submerged, families need rescue!",
        lat=6.93, lon=79.84,
        rainfall_7d=180.0, monthly_rainfall=320.0,
        elevation_m=8, distance_to_river_m=120.0,
        drainage_index=0.25, ndvi=-0.1, ndwi=0.6,
    )
    print(result)
