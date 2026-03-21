"""
Unified Disaster Response Pipeline
====================================
Combines:
  • AdaptiveIoTClassifier  — auto-detects disaster type from sensor data
  • AdaptiveFusionClassifier (BLIP + XLM-RoBERTa) — social media analysis
  • FusionLayer (cross-modal attention) — merges both for richer assessment

No disaster_type input required — the IoT model determines it automatically.

Usage:
    from fusion.pipeline import DisasterPipeline
    pipe = DisasterPipeline()
    result = pipe.analyze(
        image_path="path/to/tweet_image.jpg",
        tweet="Flooding on main street, people trapped!",
        lat=6.5, lon=80.1,
        rainfall_7d=200, monthly_rainfall=350,
        elevation_m=8, distance_to_river_m=120,
        drainage_index=0.25, ndvi=-0.1, ndwi=0.6,
    )
    print(result)
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict
from PIL import Image
from torchvision import transforms

from fusion.fusion_layer import FusionLayer, PRIORITY_LABELS, DISASTER_LABELS, RESOURCE_NAMES
from fusion.iot_predictor import IoTPredictor, IoTPrediction

CRISIS_DIR   = os.path.join(os.path.dirname(__file__), "..", "crisis")
FUSION_MODEL = os.path.join(os.path.dirname(__file__), "fusion_model.pth")
IOT_EMB_DIM  = 128
CRISIS_DIM   = 1024    # 512 (vision) + 512 (text) from AdaptiveFusionClassifier
INPUT_SIZE   = (224, 224)

CRISIS_CLASSES = [
    "affected_individuals",
    "infrastructure_and_utility_damage",
    "not_humanitarian",
    "other_relevant_information",
    "rescue_volunteering_or_donation_effort",
]


@dataclass
class DisasterAssessment:
    # ── IoT ──────────────────────────────────────────────────────────────
    iot_disaster_type:   str
    iot_type_probs:      Dict[str, float]
    iot_severity:        float
    iot_fire_prob:       float
    iot_storm_cat:       float        # raw 0-1; multiply × 5 for Saffir-Simpson
    iot_eq_magnitude:    float        # raw 0-1; multiply × 9 for Richter
    iot_flood_risk:      float        # raw 0-1; multiply × 100 for score
    iot_casualty_risk:   float        # 0-1 estimated human casualty likelihood
    iot_sensor_weights:  Dict[str, float]
    # ── Crisis ───────────────────────────────────────────────────────────
    crisis_category:     str
    crisis_confidence:   float
    crisis_probs:        Dict[str, float]
    vision_weight:       float
    text_weight:         float
    # ── Fusion ───────────────────────────────────────────────────────────
    fused_severity:      float
    priority:            str
    priority_probs:      Dict[str, float]
    confirmed_type:      str
    type_probs:          Dict[str, float]
    population_impact:   float        # 0-1 estimated fraction of population affected
    resource_needs:      Dict[str, float]  # water/medical/rescue/shelter each 0-1
    # ── Summary ──────────────────────────────────────────────────────────
    alert_level:         str
    summary:             str

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
        res = self.resource_needs
        lines = [
            "=" * 58,
            "  UNIFIED DISASTER RESPONSE ASSESSMENT",
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
            "",
            *iot_lines,
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

# Generic damage words that apply to storm/hurricane contexts
# These boost storm score only when combined with other signals
_STORM_CONTEXTUAL = ["damage", "damaged", "destruction", "destroyed", "debris",
                     "wreckage", "demolished", "devastat", "wind", "winds",
                     "gust", "rooftop", "roof", "trees down", "category"]


def _infer_disaster_type_from_text(tweet: str, caption: str = "") -> str:
    """Infer disaster type from tweet text + optional BLIP image caption.
    Uses keyword matching with contextual scoring.
    The caption provides visual context the text alone may lack."""
    # Combine tweet and caption for richer signal
    combined = f"{tweet} {caption}".lower()
    scores = {"fire": 0, "storm": 0, "earthquake": 0, "flood": 0}

    for disaster, keywords in _DISASTER_KEYWORDS.items():
        for kw in keywords:
            if kw in combined:
                scores[disaster] += 2  # Strong keyword = 2 points

    # Contextual storm words: only count as storm if no other type dominates
    storm_contextual_hits = sum(1 for kw in _STORM_CONTEXTUAL if kw in combined)
    scores["storm"] += storm_contextual_hits  # 1 point each (weaker signal)

    max_score = max(scores.values())
    if max_score == 0:
        return "unknown"

    # Resolve ties: prefer the type with more strong keyword hits
    top_types = [t for t, s in scores.items() if s == max_score]
    if len(top_types) == 1:
        return top_types[0]

    # Tie-breaking: count only strong keywords (from _DISASTER_KEYWORDS)
    strong = {}
    for t in top_types:
        strong[t] = sum(2 for kw in _DISASTER_KEYWORDS[t] if kw in combined)
    return max(strong, key=strong.get)


def _summary(a: DisasterAssessment, iot_available: bool = True) -> str:
    if not iot_available:
        type_note = ""
        if a.confirmed_type != "unknown":
            type_note = f" Inferred hazard: {a.confirmed_type}."
        return (
            f"No sensor data provided. Social media analysis: '{a.crisis_category}' "
            f"({a.crisis_confidence:.1%} confidence).{type_note} Priority: {a.priority}."
        )
    parts = []
    if a.iot_fire_prob > 0.5:
        parts.append(f"wildfire conditions ({a.iot_fire_prob:.0%})")
    if a.iot_storm_cat * 5 >= 1:
        parts.append(f"Category {a.iot_storm_cat * 5:.0f} storm")
    if a.iot_eq_magnitude * 9 >= 4.0:
        parts.append(f"M{a.iot_eq_magnitude * 9:.1f} earthquake")
    if a.iot_flood_risk * 100 > 40:
        parts.append(f"flood risk {a.iot_flood_risk * 100:.0f}/100")
    parts.append(f"social media: '{a.crisis_category}'")
    return f"Sensors detect {'; '.join(parts)}. Priority: {a.priority}."


class DisasterPipeline:
    IMG_TRANSFORM = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def __init__(self, load_crisis_model: bool = True, device: Optional[str] = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"[Pipeline] Device: {self.device}")

        print("[Pipeline] Loading IoT model…")
        self.iot = IoTPredictor()

        self._crisis_model    = None
        self._crisis_tokenizer = None
        if load_crisis_model:
            self._load_crisis_model()

        print("[Pipeline] Loading FusionLayer…")
        self.fusion = FusionLayer(
            iot_dim=IOT_EMB_DIM, crisis_dim=CRISIS_DIM, proj_dim=256
        ).to(self.device)
        if os.path.exists(FUSION_MODEL):
            state = torch.load(FUSION_MODEL, map_location=self.device, weights_only=True)
            self.fusion.load_state_dict(state)
            print(f"[Pipeline] Fusion model loaded.")
        else:
            print("[Pipeline] WARNING: fusion_model.pth not found — run train_fusion.py")
        self.fusion.eval()

    def _load_crisis_model(self):
        try:
            sys.path.insert(0, CRISIS_DIR)
            from server import AdaptiveFusionClassifier
            from transformers import (
                BlipForConditionalGeneration, BlipProcessor,
                XLMRobertaTokenizer, XLMRobertaModel,
            )
            print("[Pipeline] Loading BLIP + XLM-RoBERTa…")
            blip  = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            # Keep full BLIP model for captioning (used in hazard inference)
            self._blip_captioner = blip
            self._blip_processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )

            xlm_tok   = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
            xlm_model = XLMRobertaModel.from_pretrained("xlm-roberta-base")
            model = AdaptiveFusionClassifier(
                blip_model=blip, xlm_model=xlm_model,
                hidden_dim=512, n_classes=len(CRISIS_CLASSES)
            )
            ckpt_path = os.path.join(CRISIS_DIR, "best_adaptive_model.pth")
            try:
                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            except Exception:
                print("[Pipeline] Warning: weights_only=True failed for crisis checkpoint, using unsafe load")
                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            sd   = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict(sd, strict=False)
            model.to(self.device).eval()
            self._crisis_model     = model
            self._crisis_tokenizer = xlm_tok
            print("[Pipeline] Crisis model ready (with BLIP captioning).")
        except Exception as e:
            print(f"[Pipeline] WARNING: Could not load crisis model: {e}")
            self._blip_captioner = None
            self._blip_processor = None

    def _generate_caption(self, image_path: str) -> str:
        """Generate an image caption using BLIP for visual disaster type inference."""
        if self._blip_captioner is None or self._blip_processor is None:
            return ""
        try:
            img = Image.open(image_path).convert("RGB")
            inputs = self._blip_processor(
                images=img,
                text="a photograph of",
                return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                out = self._blip_captioner.generate(
                    **inputs, max_new_tokens=50, num_beams=3
                )
            caption = self._blip_processor.decode(out[0], skip_special_tokens=True)
            print(f"[Pipeline] BLIP caption: {caption}")
            return caption
        except Exception as e:
            print(f"[Pipeline] Caption generation failed: {e}")
            return ""

    def _run_crisis(self, image_path: str, tweet: str):
        if self._crisis_model is None:
            n     = len(CRISIS_CLASSES)
            probs = {c: 1.0 / n for c in CRISIS_CLASSES}
            return probs, torch.zeros(CRISIS_DIM), 0.5, 0.5

        try:
            img   = Image.open(image_path).convert("RGB")
            img_t = self.IMG_TRANSFORM(img).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"[Pipeline] Warning: failed to load image {image_path}: {e}")
            img_t = torch.zeros(1, 3, *INPUT_SIZE).to(self.device)

        enc   = self._crisis_tokenizer(
            tweet, max_length=128, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        ids   = enc["input_ids"].to(self.device)
        mask  = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            logits, attn = self._crisis_model(img_t, ids, mask, return_attention=True)
            vis_feat = self._crisis_model.vision_encoder(img_t).last_hidden_state[:, 0, :]
            txt_feat = self._crisis_model.text_encoder(
                input_ids=ids, attention_mask=mask
            ).last_hidden_state[:, 0, :]
            vis_proj = self._crisis_model.vision_proj(vis_feat)   # [1, 512]
            txt_proj = self._crisis_model.text_proj(txt_feat)     # [1, 512]
            emb      = torch.cat([vis_proj, txt_proj], dim=1).squeeze(0)  # [1024]
            # Temperature scaling to prevent overconfident predictions
            # Higher temperature → more distributed probabilities (avoids 100% / 0.0%)
            temperature = 2.5
            probs_t  = F.softmax(logits / temperature, dim=-1).squeeze(0).cpu()
            probs    = {CRISIS_CLASSES[i]: float(probs_t[i]) for i in range(len(CRISIS_CLASSES))}
            vis_w    = float(attn["vision_weight"].cpu().item())
            txt_w    = float(attn["text_weight"].cpu().item())

        return probs, emb.cpu(), vis_w, txt_w

    def analyze(
        self,
        image_path: str,
        tweet:      str,
        crisis_embedding: Optional[torch.Tensor] = None,
        **sensor_kwargs,
    ) -> DisasterAssessment:
        """
        Full pipeline inference.

        Parameters
        ----------
        image_path       : path to the tweet image
        tweet            : tweet text
        crisis_embedding : optional pre-computed [1024] tensor (skips model run)
        **sensor_kwargs  : raw sensor readings forwarded to IoTPredictor

        Available sensor kwargs (all optional, default 0):
          Fire     : precipitation, max_temp, min_temp, avg_wind_speed, month,
                     temp_range, lagged_precipitation
          Storm    : lat, lon, wind_kts, pressure, month, category, shape_leng
          Quake    : lat, lon, depth, magnitude, rms, n_stations, n_phases,
                     azimuth_gap, month
          Flood    : lat, lon, elevation_m, distance_to_river_m, rainfall_7d,
                     monthly_rainfall, drainage_index, ndvi, ndwi,
                     historical_flood_count
        """
        # ── Step 1: Build sensor feature vector + IoT prediction ──────────
        # Try all helper constructors; pick whichever has the highest severity
        # (in practice the caller passes only one set of kwargs)
        from IOT.train_iot import (
            features_from_fire, features_from_storm_hist,
            features_from_eq_iran, features_from_flood,
        )
        import pandas as pd

        def _pd(d): return pd.Series(d)

        candidates = []

        # Fire
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

        # Storm
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

        # Earthquake
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

        # Flood
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

        if not iot_available:
            # No sensor data provided — skip IoT entirely
            iot = None
        else:
            # Pick IoT prediction with highest severity
            iot: IoTPrediction = max(candidates, key=lambda p: p.severity_score)

        # ── Step 2: Crisis inference ───────────────────────────────────────
        if crisis_embedding is not None:
            crisis_emb   = crisis_embedding
            crisis_probs = {c: 1.0 / len(CRISIS_CLASSES) for c in CRISIS_CLASSES}
            vision_w = text_w = 0.5
        else:
            crisis_probs, crisis_emb, vision_w, text_w = self._run_crisis(image_path, tweet)

        top_crisis = max(crisis_probs, key=crisis_probs.get)
        top_conf   = crisis_probs[top_crisis]

        # ── Step 3: Fusion / crisis-only fallback ────────────────────────
        if iot_available:
            iot_vec    = iot.embedding.unsqueeze(0).to(self.device)
            crisis_vec = crisis_emb.unsqueeze(0).to(self.device)

            with torch.no_grad():
                out = self.fusion(iot_vec, crisis_vec)

            fused_sev  = float(out.severity_score.item())
            pri_probs  = F.softmax(out.priority_logits, dim=-1).squeeze(0).cpu()
            dis_probs  = F.softmax(out.disaster_logits, dim=-1).squeeze(0).cpu()
            priority   = PRIORITY_LABELS[int(pri_probs.argmax())]
            conf_type  = DISASTER_LABELS[int(dis_probs.argmax())]
            pop_impact = float(out.population_impact.item())
            res_needs  = {RESOURCE_NAMES[i]: float(out.resource_needs[0, i])
                          for i in range(len(RESOURCE_NAMES))}

            iot_disaster_type  = iot.disaster_type
            iot_type_probs     = iot.type_probabilities
            iot_severity       = iot.severity_score
            iot_fire_prob      = iot.fire_prob
            iot_storm_cat      = iot.storm_cat_norm
            iot_eq_magnitude   = iot.eq_magnitude_norm
            iot_flood_risk     = iot.flood_risk_norm
            iot_casualty_risk  = iot.casualty_risk
            iot_sensor_weights = iot.sensor_weights
            priority_probs     = {PRIORITY_LABELS[i]: float(pri_probs[i]) for i in range(4)}
            type_probs         = {DISASTER_LABELS[i]: float(dis_probs[i]) for i in range(5)}
        else:
            # No sensor data — drive entirely from crisis model
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
            # Category-specific severity multipliers — severe categories
            # should produce higher severity even without sensor data
            _CRISIS_SEVERITY_SCALE = {
                "not_humanitarian":                       0.0,
                "other_relevant_information":             0.25,
                "vehicle_damage":                         0.50,
                "affected_individuals":                   0.80,
                "infrastructure_and_utility_damage":      0.90,
                "rescue_volunteering_or_donation_effort": 0.95,
                "missing_or_found_people":                0.95,
            }
            priority  = crisis_to_priority.get(top_crisis, "Low")
            # Generate image caption for visual context in hazard inference
            caption = self._generate_caption(image_path)
            conf_type = _infer_disaster_type_from_text(tweet, caption)
            sev_scale = _CRISIS_SEVERITY_SCALE.get(top_crisis, 0.4)
            fused_sev = top_conf * sev_scale

            iot_disaster_type  = "unknown"
            iot_type_probs     = {d: 0.2 for d in DISASTER_LABELS}
            iot_severity       = 0.0
            iot_fire_prob      = 0.0
            iot_storm_cat      = 0.0
            iot_eq_magnitude   = 0.0
            iot_flood_risk     = 0.0
            iot_casualty_risk  = 0.0
            iot_sensor_weights = {"weather": 0.25, "storm": 0.25, "seismic": 0.25, "hydro": 0.25}
            priority_probs     = {p: 0.25 for p in PRIORITY_LABELS}
            type_probs         = {d: 0.2 for d in DISASTER_LABELS}

            # Estimate Population Impact & Resource Needs from crisis category alone
            # These are domain-informed mappings (no sensor data available)
            _c = top_crisis
            _s = top_conf  # confidence as scaling factor
            _CRISIS_POP = {
                "not_humanitarian":                       0.0,
                "other_relevant_information":             0.10,
                "affected_individuals":                   0.70,
                "infrastructure_and_utility_damage":      0.55,
                "rescue_volunteering_or_donation_effort": 0.65,
            }
            # resource_needs per crisis category [water, medical, rescue, shelter]
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
            res_needs  = {RESOURCE_NAMES[i]: round(_base_res[i] * _s, 4)
                          for i in range(len(RESOURCE_NAMES))}

        assessment = DisasterAssessment(
            iot_disaster_type  = iot_disaster_type,
            iot_type_probs     = iot_type_probs,
            iot_severity       = iot_severity,
            iot_fire_prob      = iot_fire_prob,
            iot_storm_cat      = iot_storm_cat,
            iot_eq_magnitude   = iot_eq_magnitude,
            iot_flood_risk     = iot_flood_risk,
            iot_casualty_risk  = iot_casualty_risk,
            iot_sensor_weights = iot_sensor_weights,
            crisis_category    = top_crisis,
            crisis_confidence  = top_conf,
            crisis_probs       = crisis_probs,
            vision_weight      = vision_w,
            text_weight        = text_w,
            fused_severity     = fused_sev,
            priority           = priority,
            priority_probs     = priority_probs,
            confirmed_type     = conf_type,
            type_probs         = type_probs,
            population_impact  = pop_impact,
            resource_needs     = res_needs,
            alert_level        = "",
            summary            = "",
        )
        # Cap alert level: not_humanitarian → max GREEN regardless of severity
        if top_crisis == "not_humanitarian":
            assessment.alert_level = "GREEN"
        else:
            assessment.alert_level = _alert_level(fused_sev, priority)
        assessment.summary = _summary(assessment, iot_available)
        return assessment


if __name__ == "__main__":
    import glob

    pipe = DisasterPipeline(load_crisis_model=False)

    pattern  = os.path.join(CRISIS_DIR, "CrisisMMD_v2.0", "data_image",
                            "srilanka_floods", "**", "*.jpg")
    samples  = glob.glob(pattern, recursive=True)
    img_path = samples[0] if samples else "/dev/null"

    result = pipe.analyze(
        image_path          = img_path,
        tweet               = "Flooding in Colombo, roads submerged, families need rescue!",
        lat                 = 6.93,
        lon                 = 79.84,
        rainfall_7d         = 180.0,
        monthly_rainfall    = 320.0,
        elevation_m         = 8,
        distance_to_river_m = 120.0,
        drainage_index      = 0.25,
        ndvi                = -0.1,
        ndwi                = 0.6,
    )
    print(result)
