# Multimodal Disaster Intelligence Platform

## Complete Technical Documentation

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Layer 1: IoT Sensor Classifier](#3-layer-1-iot-sensor-classifier)
4. [Layer 2: Crisis Social Media Classifier](#4-layer-2-crisis-social-media-classifier)
5. [Layer 3: Cross-Modal Fusion Layer](#5-layer-3-cross-modal-fusion-layer)
6. [Layer 4: Explainability (XAI)](#6-layer-4-explainability-xai)
7. [Layer 5: Alert and Resource Recommendation](#7-layer-5-alert-and-resource-recommendation)
8. [Server and API](#8-server-and-api)
9. [Web Dashboard](#9-web-dashboard)
10. [Datasets](#10-datasets)
11. [Training Procedures](#11-training-procedures)
12. [Quick Start](#12-quick-start)
13. [Directory Structure](#13-directory-structure)

---

## 1. Project Overview

### What This System Does

This platform provides **tri-modal disaster assessment** by fusing three independent data streams:

- **Environmental sensor metadata** (weather, storm-track, seismic, and hydrological features)
- **Social media** (images and text from disaster-affected areas)
- **Satellite imagery** (damage assessment features extracted from xBD-style post-disaster imagery)

It automatically detects the disaster type, estimates severity, predicts resource needs, and generates actionable briefings for first responders. No manual disaster-type input is needed; the system infers everything from raw data.

The repository retains the historical `IoT` naming used during development (`IOT/`, `AdaptiveIoTClassifier`), but the current model is trained on archival environmental and seismological datasets plus a synthetic flood-risk dataset rather than on a live sensor network. The architecture is sensor-agnostic and can be adapted to operational feeds, but the published results should be read in that scope.

### Why This Approach

Traditional disaster response systems often rely on a single data source. Environmental metadata provides physical hazard evidence but limited human context. Social media offers real-time on-the-ground reports but is noisy and unstructured. Satellite imagery gives broad spatial evidence of structural damage but lacks immediate field-level semantics. This platform fuses all three streams through learned pairwise cross-attention and adaptive gating, producing assessments that no single modality can provide alone.

### The Five-Layer Pipeline

```
Layer 1: Environmental Sensor Analysis ("IoT" path in code)
    32-dim sensor vector -> AdaptiveIoTClassifier -> disaster type + severity + risk scores
                                                      |
                                                      | 128-dim embedding
                                                      v
Layer 3: Tri-Modal Fusion ----------> TriFusionLayer (pairwise cross-attention + gating)
                                                      ^                      ^
                                                      |                      |
                                                      | 1024-dim embedding   | 640-dim embedding
                                                      |                      |
Layer 2: Crisis Social Media         Layer 2b: Satellite Damage Assessment
    Image + Tweet ->                     Post-disaster image ->
    BLIP ViT + XLM-RoBERTa ->           DeepLabV3+ ->
    AdaptiveFusionClassifier            F_sat (512) + F_region (128)
                                                      |
                                                      v
Layer 4: Explainability
    Gradient-weighted Attention Rollout (Grad-CAM) + GPT-4o Natural Language Briefing
                                                      |
                                                      v
Layer 5: Alert and Resource Recommendation
    Alert level (RED/ORANGE/YELLOW/GREEN) + Resource needs (water/medical/rescue/shelter)
```

---

## 2. System Architecture

### High-Level Data Flow

```
                    +------------------+
                    |   User Input     |
                    | Image + Tweet +  |
                    | Sensor Readings  |
                    | + Satellite Img  |
                    +--------+---------+
              +--------------+------------------+--------------+
              |                                 |              |
              v                                 v              v
    +---------+----------+          +-----------+----------+  +-----------+----------+
    | Env. Metadata      |          | Crisis Media Module  |  | Satellite Damage     |
    | Module             |          | AdaptiveFusion       |  | Module               |
    | AdaptiveIoT        |          | Classifier           |  | DeepLabV3+           |
    | Classifier         |          |                      |  |                      |
    | Input: 32 floats   |          | Input: 224x224 img   |  | Input: 512x512 post  |
    | Output: 128-dim    |          | + tokenized text     |  | image                |
    | embedding          |          | Output: 1024-dim     |  | Output: 640-dim      |
    +---------+----------+          | embedding            |  | embedding            |
              |                     +-----------+----------+  +-----------+----------+
              |                                 |                         |
              +-------------------+-------------+-------------------------+
                                  |
                             v
                  +----------+---------+
                  |   TriFusionLayer   |
                  | Pairwise Cross-    |
                  | Attention + Gate   |
                  |                    |
                  | Crisis/Env/Sat     |
                  | pair interactions  |
                  +----------+---------+
                             |
               +-------------+-------------+
               |             |             |             |
               v             v             v             v
         +---------+   +---------+   +---------+   +---------+
         |Severity |   |Priority |   |Disaster |   |Resource |
         | Head    |   | Head    |   | Head    |   | Heads   |
         +---------+   +---------+   +---------+   +---------+
```

### Model Checkpoints

| Model | File | Size | What It Contains |
|-------|------|------|-----------------|
| Environmental Metadata Classifier | `IOT/models/iot_model.pth` | ~180 KB | AdaptiveIoTClassifier state_dict + config + validation metrics |
| Crisis Classifier | `crisis/best_adaptive_model.pth` | ~450 MB | AdaptiveFusionClassifier state_dict (includes frozen BLIP + XLM-RoBERTa weights) |
| Dual Fusion Layer | `fusion/fusion_model.pth` | ~2 MB | Legacy IoT + Crisis fusion checkpoint |
| Tri-Fusion Layer | `fusion/tri_fusion_model.pth` | ~3 MB | TriFusionLayer state_dict for crisis + environmental metadata + satellite fusion |

### Technology Stack

| Component | Technology |
|-----------|-----------|
| ML Framework | PyTorch >= 2.0 |
| Vision Backbone | BLIP ViT (Salesforce/blip-image-captioning-base) |
| Text Backbone | XLM-RoBERTa (xlm-roberta-base) |
| Image Captioning | BLIP (BlipForConditionalGeneration) |
| API Server | FastAPI + Uvicorn |
| XAI Explanations | GPT-4o (OpenAI API) |
| Frontend | Vanilla JavaScript, CSS (no framework) |
| State Management | LocalStorage (browser) + JSON file (server jobs) |

---

## 3. Layer 1: IoT Sensor Classifier

**Source**: `IOT/train_iot.py`
**Inference wrapper**: `fusion/iot_predictor.py`

### Purpose

Automatically detects disaster type from raw sensor readings across 4 sensor groups. No user input about disaster type is required; the model infers it from the data.

### Input: 32-Dimensional Unified Sensor Vector

The input is divided into 4 groups of 8 features each. All values are normalized to [0, 1].

#### Group 0: Weather (indices 0-7)

| Index | Feature | Normalization | Source |
|-------|---------|---------------|--------|
| 0 | Precipitation | / 50.0 mm | Weather stations |
| 1 | Max Temperature | / 55.0 C | Weather stations |
| 2 | Min Temperature | / 55.0 C | Weather stations |
| 3 | Wind Speed | / 50.0 km/h | Anemometers |
| 4 | Temperature Range | / 35.0 C | Derived (max - min) |
| 5 | Drought Index | 1 - precip (inverted) | Derived |
| 6 | Month (sine) | sin(2pi * month / 12) | Calendar |
| 7 | Month (cosine) | cos(2pi * month / 12) | Calendar |

#### Group 1: Storm (indices 8-15)

| Index | Feature | Normalization | Source |
|-------|---------|---------------|--------|
| 8 | Wind Intensity | / 200.0 knots | Storm trackers |
| 9 | Pressure Anomaly | (1013 - pressure) / 500.0 | Barometers |
| 10 | Latitude | / 90.0 | GPS |
| 11 | Longitude | / 180.0 | GPS |
| 12 | Storm Category (normalized) | CAT_SEV lookup | Saffir-Simpson |
| 13 | Hour (sine) | sin(2pi * hour / 24) | Timestamp |
| 14 | Hour (cosine) | cos(2pi * hour / 24) | Timestamp |
| 15 | Track Speed | / 10.0 | Storm trackers |

**Storm Category Severity Mapping (CAT_SEV)**:
```
TD: 0.10    TS: 0.30    E: 0.15     L: 0.20
SD: 0.15    SS: 0.25    W: 0.10
H1: 0.40    H2: 0.55    H3: 0.70    H4: 0.85    H5: 1.00
```

#### Group 2: Seismic (indices 16-23)

| Index | Feature | Normalization | Source |
|-------|---------|---------------|--------|
| 16 | Depth | / 700.0 km | Seismographs |
| 17 | RMS Error | / 2.0 | Seismic networks |
| 18 | Number of Stations | / 200.0 | Seismic networks |
| 19 | Number of Phases | / 500.0 | Phase pickers |
| 20 | Azimuth Gap | / 360.0 degrees | Station geometry |
| 21 | Latitude | / 90.0 | Epicenter location |
| 22 | Longitude | / 180.0 | Epicenter location |
| 23 | Magnitude Proxy | / 9.0 Richter | Seismographs |

#### Group 3: Hydro (indices 24-31)

| Index | Feature | Normalization | Source |
|-------|---------|---------------|--------|
| 24 | Elevation (inverted) | 1 - (elev / 500.0) | DEM |
| 25 | River Proximity (inverted) | 1 - (dist / 5000.0) | GIS |
| 26 | 7-day Rainfall | / 500.0 mm | Rain gauges |
| 27 | Monthly Rainfall | / 2000.0 mm | Rain gauges |
| 28 | Drainage Index | direct [0, 1] | Terrain analysis |
| 29 | NDVI | (ndvi + 1) / 2 | Satellite |
| 30 | NDWI | (ndwi + 1) / 2 | Satellite |
| 31 | Historical Flood Count | / 20.0 | Historical records |

### Architecture: AdaptiveIoTClassifier

```
Input [B, 32]
  |
  |-- Split into 4 groups of 8 features each
  |
  v
4x SensorGroupEncoder
  |  Each: Linear(8, 128) + LayerNorm + ReLU + Dropout(0.2)
  |         Linear(128, 128) + LayerNorm
  |  Output: [B, 128] per group
  |
  v
4x SensorConfidenceEstimator (NOVEL)
  |  Each: Linear(128, 64) + ReLU + Dropout(0.2)
  |         Linear(64, 32) + ReLU
  |         Linear(32, 1) + Sigmoid
  |  Output: scalar confidence in (0, 1) per group
  |
  |  Purpose: Groups with no active sensors (all zeros) get
  |  low confidence, automatically down-weighting them.
  |  This prevents noise from irrelevant sensor groups.
  |
  v
Confidence-Weighted Aggregation
  |  weights[i] = confidence[i] / sum(all_confidences)
  |  weighted[i] = weights[i] * group_features[i]
  |
  v
Cross-Group Multi-Head Attention (NOVEL)
  |  nn.MultiheadAttention(embed_dim=128, num_heads=4)
  |  Input: stacked weighted features [B, 4, 128]
  |  Self-attention: each sensor group attends to all others
  |
  |  Purpose: Allows cross-sensor interaction. For example,
  |  seismic activity can reinforce hydro signals to detect
  |  earthquake-triggered floods (cascading disasters).
  |
  v
Mean Pooling: [B, 4, 128] -> [B, 128]
  |
  v  (This 128-dim embedding is passed to FusionLayer)
  |
  +---> Disaster Type Head
  |     Linear(128, 128) + LayerNorm + ReLU + Dropout(0.2)
  |     Linear(128, 5) -> softmax -> [fire, storm, earthquake, flood, unknown]
  |
  +---> Severity Head
  |     Linear(128, 64) + ReLU + Linear(64, 1) + Sigmoid -> [0, 1]
  |
  +---> Risk Detail Head
  |     Linear(128, 64) + ReLU + Linear(64, 4) + Sigmoid
  |     -> [fire_prob, storm_cat, eq_magnitude, flood_risk] each [0, 1]
  |
  +---> Casualty Risk Head
        Linear(128, 64) + ReLU + Dropout(0.2) + Linear(64, 1) + Sigmoid -> [0, 1]
```

### Weight Initialization

All linear layers use **Xavier Uniform** initialization. Biases are initialized to zero.

### Output Interpretation

| Field | Range | How to Read |
|-------|-------|-------------|
| `disaster_type` | string | Auto-detected: fire, storm, earthquake, flood, or unknown |
| `severity_score` | 0-1 | Overall severity (0 = none, 1 = catastrophic) |
| `fire_prob` | 0-1 | Fire probability score |
| `storm_cat_norm` | 0-1 | Multiply by 5 for Saffir-Simpson category |
| `eq_magnitude_norm` | 0-1 | Multiply by 9 for approximate Richter magnitude |
| `flood_risk_norm` | 0-1 | Multiply by 100 for risk score |
| `casualty_risk` | 0-1 | Estimated human casualty likelihood |
| `sensor_weights` | dict, sum=1 | Shows which sensor group dominated the decision |
| `embedding` | [128] tensor | Passed downstream to FusionLayer |

### Training Details

| Parameter | Value |
|-----------|-------|
| Batch Size | 256 |
| Epochs | 40 |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR (T_max=40) |
| Loss | CrossEntropy(type) + MSE(severity) + MSE(risk) + MSE(casualty) |
| Gradient Clipping | max_norm=1.0 |
| Class Balancing | WeightedRandomSampler |
| Train/Val Split | 85% / 15% |
| Total Parameters | ~180K |
| Training Data | ~50K samples from 6 datasets |

### Severity Label Generation

Training labels for severity are derived from the raw sensor data:

```
Fire:       severity = (wind_temp_ratio * 5 + (1 - precip/50) + max_temp/55) / 3
Storm:      severity = CAT_SEV[category]  (0.10 for TD ... 1.00 for H5)
Earthquake: severity = magnitude / 9.0
Flood:      severity = flood_risk_score / 100.0
```

Casualty risk labels are derived from severity with hazard-specific multipliers:
- Fire: severity * 0.70
- Storm: severity * 0.75 (0.90 for Cat 4-5)
- Earthquake: severity * 0.92 (highest: no warning time)
- Flood: severity * 0.60 (lowest: slower onset)

---

## 4. Layer 2: Crisis Social Media Classifier

**Model Definition**: `crisis/server.py` (AdaptiveFusionClassifier class)
**Inference via pipeline**: `fusion/pipeline.py` (_run_crisis method)

### Purpose

Classifies image-text pairs from social media into humanitarian categories relevant to disaster response.

### Backbone Models

#### Vision: BLIP ViT

- **Model**: `Salesforce/blip-image-captioning-base`
- **Architecture**: Vision Transformer (ViT-B/16)
- **Input**: 224 x 224 RGB image
- **Patch Size**: 16 x 16 pixels -> 14 x 14 = 196 patches
- **Output**: [B, 197, 768] (196 patch tokens + 1 CLS token)
- **Used**: CLS token [B, 768] as the global image representation
- **Pre-trained**: On COCO captions (129K images)
- **Fine-tuned**: End-to-end with the classifier

BLIP is also used separately for **image captioning** (hazard inference when IoT is unavailable).

#### Text: XLM-RoBERTa

- **Model**: `xlm-roberta-base`
- **Architecture**: Transformer encoder, 12 layers
- **Input**: Tokenized text, max 128 tokens, padded to max_length
- **Output**: [B, 128, 768]
- **Used**: CLS token [B, 768] as the global text representation
- **Why XLM-RoBERTa**: Supports 100 languages; disaster tweets are multilingual

### Image Preprocessing

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),          # [0, 255] -> [0.0, 1.0]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet means
        std=[0.229, 0.224, 0.225]    # ImageNet stds
    ),
])
```

### Architecture: AdaptiveFusionClassifier

```
Image [224, 224, 3]                         Tweet (max 128 tokens)
       |                                            |
       v                                            v
BLIP ViT Encoder                           XLM-RoBERTa Encoder
[B, 197, 768]                               [B, 128, 768]
       |                                            |
       | CLS token                                  | CLS token
       v                                            v
vision_features [B, 768]                   text_features [B, 768]
       |                                            |
       +-----> ConfidenceEstimator ----+    +---- ConfidenceEstimator <-----+
       |       [768] -> [1] sigmoid    |    |     [768] -> [1] sigmoid     |
       |                               |    |                              |
       |            vision_conf -------+----+------- text_conf             |
       |            Normalize: weight = conf / (sum + eps)                 |
       |                                                                   |
       v                                            v
Vision Projection                          Text Projection
  Linear(768, 512)                           Linear(768, 512)
  + LayerNorm + ReLU                         + LayerNorm + ReLU
  + Dropout(0.2)                             + Dropout(0.2)
       |                                            |
       v                                            v
vision_proj [B, 512]                       text_proj [B, 512]
       |                                            |
       | * vision_weight                            | * text_weight
       v                                            v
vision_weighted [B, 512]                   text_weighted [B, 512]
       |                                            |
       +--------+----------------------------------+--------+
                |                                           |
                v                                           v
Cross-Attention (vision queries text)      Cross-Attention (text queries vision)
  Q = vision_weighted                        Q = text_weighted
  K/V = text_weighted                        K/V = vision_weighted
  8 heads, dim=512                           8 heads, dim=512
                |                                           |
                v                                           v
  vision_attended [B, 512]                  text_attended [B, 512]
                |                                           |
                +------- Concatenate -----------------------+
                                   |
                                   v
                        fused [B, 1024]
                                   |
                                   v
                     Classification MLP
                       Linear(1024, 512)
                       + LayerNorm + ReLU + Dropout(0.3)
                       Linear(512, 256)
                       + ReLU + Dropout(0.2)
                       Linear(256, 5)
                                   |
                                   v
                        logits [B, 5]
                                   |
                                   v
              softmax(logits / temperature=2.5)
                                   |
                                   v
                     probabilities [B, 5]
```

### ConfidenceEstimator (Adaptive Modality Weighting)

This is the mechanism that makes the model "adaptive." Rather than treating image and text equally, the model learns to estimate how much useful information each modality carries for the current input.

```
Input [B, dim]
  -> Linear(dim, 256) + ReLU + Dropout(0.2)
  -> Linear(256, 128) + ReLU
  -> Linear(128, 1) + Sigmoid
  -> confidence [B, 1] in (0, 1)
```

The two confidence scores are normalized to weights that sum to 1:
```
vision_weight = vision_conf / (vision_conf + text_conf + 1e-8)
text_weight   = text_conf   / (vision_conf + text_conf + 1e-8)
```

**Why this matters**: When an image is blurry or uninformative, the model learns to lower vision_conf and rely more on text. When a tweet is generic ("pray for victims"), the model shifts weight to the image.

### Temperature Scaling

Raw softmax on the crisis model logits produces overconfident predictions (99.8% / 0.0% / 0.0% / 0.0% / 0.0%). This is a known problem with deep neural networks.

**Solution**: Temperature scaling divides logits by T=2.5 before softmax:

```python
probabilities = softmax(logits / 2.5, dim=-1)
```

- T=1.0: Standard softmax (overconfident)
- T=2.5: Calibrated probabilities (e.g., 45% / 25% / 15% / 10% / 5%)
- Higher T: More uniform (less decisive)

This does not change the predicted class (argmax is unaffected), only the confidence values.

### Output Classes (5 Humanitarian Categories)

| Class | Description | Example |
|-------|-------------|---------|
| `affected_individuals` | People displaced, injured, or needing aid | "Families stranded on rooftops" |
| `infrastructure_and_utility_damage` | Roads, buildings, power, water systems damaged | "Bridge collapsed, power lines down" |
| `not_humanitarian` | Off-topic, noise, not crisis-related | "Beautiful sunset over the city" |
| `other_relevant_information` | General logistics, news updates | "Red Cross setting up at convention center" |
| `rescue_volunteering_or_donation_effort` | Emergency response, volunteering, donations | "Volunteers needed for sandbagging" |

### The 1024-dim Crisis Embedding

The embedding passed to the FusionLayer is the concatenation of the vision and text projections:

```
crisis_embedding = cat([vision_proj, text_proj], dim=1)  # [B, 1024]
                         [B, 512]     [B, 512]
```

This preserves both visual and textual information for the fusion layer to work with.

---

## 5. Layer 3: Cross-Modal Fusion Layer

**Source**: `fusion/fusion_layer.py`
**Training**: `fusion/train_fusion.py`

### Purpose

Combines the IoT sensor embedding (128-dim) with the crisis social media embedding (1024-dim) using **cross-modal attention**, producing enriched disaster assessments that leverage both data streams.

### Why Not Just Concatenate?

Simple concatenation treats all features equally. Cross-modal attention lets the sensor data **query** the social media context:
- "I see seismic readings. Does the social media confirm building damage?"
- "Wind sensors show high speeds. Are people reporting a hurricane or just a storm?"

This selective information retrieval produces more accurate assessments than concatenation.

### Architecture: FusionLayer

```
IoT Embedding [B, 128]                    Crisis Embedding [B, 1024]
       |                                            |
       v                                            v
IoT Projection                             Crisis Projection
  Linear(128, 512)                           Linear(1024, 512)
  + LayerNorm + ReLU                         + LayerNorm + ReLU
  + Dropout(0.2)                             + Dropout(0.2)
  Linear(512, 256)                           Linear(512, 256)
  + LayerNorm                                + LayerNorm
       |                                            |
       v                                            v
iot_proj [B, 256]                          crisis_proj [B, 256]
       |                                            |
       +---------> CrossModalAttention <------------+
       |           Q = iot_proj (queries)            |
       |           K/V = crisis_proj (context)       |
       |                                             |
       |           Q_proj = Linear(256, 256)         |
       |           K_proj = Linear(256, 256)         |
       |           V_proj = Linear(256, 256)         |
       |           scale  = 256^(-0.5)               |
       |                                             |
       |           attn = softmax(Q * K^T * scale)   |
       |           out  = attn * V                   |
       |           out_proj = Linear(256, 256)        |
       |                    |                        |
       |                    v                        |
       |              attended [B, 256]              |
       |                    |                        |
       +-------- Concatenate -----------------------+
       |              |                              |
       v              v                              v
    iot_proj      attended                    crisis_proj
    [B, 256]      [B, 256]                   [B, 256]
                       |
                       v
               fused [B, 768]
                       |
                       v
              Shared MLP
                Linear(768, 512) + LayerNorm + ReLU + Dropout(0.2)
                Linear(512, 256) + LayerNorm + ReLU + Dropout(0.2)
                       |
                       v
                rep [B, 256]    (shared representation)
                       |
         +------+------+------+------+
         |      |      |      |      |
         v      v      v      v      v
     Severity Priority Disaster Pop.  Resource
      Head     Head    Head   Head    Head
```

### Output Heads

| Head | Architecture | Output | Range |
|------|-------------|--------|-------|
| **Severity** | Linear(256, 1) + Sigmoid | [B] | 0.0 - 1.0 |
| **Priority** | Linear(256, 4) | [B, 4] logits | Low / Medium / High / Critical |
| **Disaster Type** | Linear(256, 5) | [B, 5] logits | fire / storm / earthquake / flood / unknown |
| **Population Impact** | Linear(256, 64) + ReLU + Dropout + Linear(64, 1) + Sigmoid | [B] | 0.0 - 1.0 |
| **Resource Needs** | Linear(256, 64) + ReLU + Dropout + Linear(64, 4) + Sigmoid | [B, 4] | water / medical / rescue / shelter, each 0-1 |

### How Training Data Is Generated

The fusion layer requires paired (IoT embedding, crisis embedding) samples. Since no dataset has both sensor readings and social media posts for the same events, the training uses a **hybrid approach**:

1. **Real IoT embeddings**: The CrisisMMD dataset contains events (hurricane_harvey, srilanka_floods, etc.). For each event, realistic sensor readings are **synthetically generated** based on the event type. These are passed through the pre-trained IoT model to get real 128-dim embeddings.

2. **Synthetic crisis embeddings**: 1024-dim vectors with type-biased block patterns + Gaussian noise (std=0.3). Different crisis categories activate different embedding regions.

3. **Labels are derived from domain knowledge**:
   - Severity from crisis category + random variation
   - Priority from `LABEL_TO_PRIORITY` mapping
   - Disaster type from `EVENT_TO_TYPE` mapping
   - Population impact from `LABEL_TO_POPULATION` mapping
   - Resource needs from `DISASTER_RESOURCES` base values scaled by severity

### Crisis-Only Fallback Path

When no IoT sensor data is provided, the system bypasses the FusionLayer entirely and uses heuristic mappings:

#### Severity Scaling by Crisis Category

```
not_humanitarian:                        0.0 (no disaster)
other_relevant_information:              0.25
vehicle_damage:                          0.50
affected_individuals:                    0.80
infrastructure_and_utility_damage:       0.90
rescue_volunteering_or_donation_effort:  0.95
missing_or_found_people:                 0.95
```

Formula: `fused_severity = crisis_confidence * severity_scale`

#### Disaster Type Inference Without Sensors

When IoT data is unavailable, the system infers disaster type from text using a two-stage approach:

1. **BLIP Image Captioning**: Generates a description of the image (e.g., "a photograph of hurricane damage to buildings")
2. **Keyword Matching**: Scores each disaster type by keyword hits in the combined tweet + caption text

**Scoring System**:
- Strong keywords (from `_DISASTER_KEYWORDS` dict) = 2 points each
- Contextual storm words (damage, debris, wind, etc.) = 1 point each (storm only)
- Ties broken by counting only strong keyword hits

**Example keyword sets**:
```
fire:       fire, wildfire, blaze, burning, smoke, flames, charred, ember...
storm:      storm, hurricane, typhoon, cyclone, tornado, irma, harvey, maria...
earthquake: earthquake, quake, seismic, tremor, aftershock, collapsed, rubble...
flood:      flood, flooding, submerged, inundation, deluge, waterlogged...
```

---

## 6. Layer 4: Explainability (XAI)

**Source**: `fusion/xai.py`

### Purpose

Provides two forms of explainability:
1. **Visual**: Heatmap showing which image regions drove the classification
2. **Textual**: GPT-4o-generated structured briefing for field responders

### Gradient-Weighted Attention Rollout (Visual XAI)

#### The Problem

Standard Grad-CAM does not work well with Vision Transformers (ViTs). In BLIP's ViT, only the CLS token is used downstream for classification. This means gradients on individual patch tokens are near-zero, making standard Grad-CAM highlight nothing meaningful.

#### The Solution

**Gradient-weighted Attention Rollout** combines two signals:
- **Attention maps**: Where the model "looks" (from ViT self-attention layers)
- **Gradients on attention weights**: Which of those attention patterns actually matter for the final prediction

#### Algorithm

```
1. FORWARD PASS (with gradient tracking)
   - Run image through BLIP ViT with output_attentions=True
   - This gives attention matrices [num_heads, 197, 197] for each of 12 layers
   - Register backward hooks on each attention tensor to capture gradients
   - Complete forward pass through full classification pipeline:
     vision_proj -> text_proj -> cross_attention -> classifier -> logits

2. BACKWARD PASS
   - Compute loss = logits[predicted_class]
   - loss.backward()
   - Hooks capture gradient tensors for each attention layer

3. GRADIENT WEIGHTING
   For each of the last 4 layers (layers 8-11):
     - attention [num_heads, 197, 197]
     - gradient  [num_heads, 197, 197] (captured by hooks)
     - Clamp gradients to positive values only (keep features that
       INCREASE the predicted class score)
     - weighted_attention = attention * clamp(gradient, min=0)

4. ATTENTION ROLLOUT
   Accumulate through layers using matrix multiplication:
     - Average across attention heads: [197, 197]
     - Add residual connection: 0.5 * attention + 0.5 * identity
     - Normalize rows to sum to 1
     - Multiply with previous layer's rollout matrix

   Result: rollout matrix showing how information flows from
   each patch to the CLS token through all 4 layers

5. EXTRACT CLS ATTENTION
   - Take row 0 (CLS token), columns 1-196 (patch tokens)
   - Result: [196] attention values

6. POST-PROCESSING
   - Normalize to [0, 1]
   - Threshold at 40th percentile (suppress low-activation areas)
   - Reshape [196] -> [14, 14] grid
   - Upsample to [224, 224] using bicubic interpolation
   - Gaussian blur with 15x15 kernel (smooth grid artifacts)
   - Normalize again to [0, 1]

7. OVERLAY
   - Apply JET colormap to heatmap
   - Blend: 45% heatmap + 55% original image
   - Encode as base64 PNG string
```

#### Why Last 4 Layers Only

Using all 12 layers dilutes the signal. Early layers learn generic features (edges, textures). Later layers specialize in task-relevant patterns (damage, flooding). Using only layers 8-11 produces focused heatmaps.

#### Why 40th Percentile Threshold

Without thresholding, low-activation areas create a diffuse background glow that obscures the important regions. The 40th percentile cutoff keeps only the top 60% of activations.

#### Fallback Chain

If gradient-weighted attention fails (e.g., no gradients captured):
1. **Pure Attention Rollout**: Same as above but without gradient weighting
2. **Input Gradient Saliency**: Pixel-level gradients showing which pixels affect the output
3. **Empty string**: If all methods fail, no heatmap is shown

### GPT-4o Natural Language Briefing (Textual XAI)

The full assessment (sensor data, crisis classification, fusion output) is sent to GPT-4o with a structured prompt requesting:

1. **SITUATION**: 2-3 sentences on what is happening
2. **KEY RISKS**: Top 3 specific risks with numbers from the data
3. **RECOMMENDED ACTIONS**: 3-4 actionable steps for first responders
4. **WHY THIS ALERT**: 1-2 sentences explaining which signals drove the alert level

```
Model:       gpt-4o
Max tokens:  400
Temperature: 0.3 (low = factual, consistent)
Max words:   ~220
```

Requires `OPENAI_API_KEY` environment variable. If not set, the briefing is silently skipped.

---

## 7. Layer 5: Alert and Resource Recommendation

### Alert Level Calculation

```python
combined_score = (severity + priority_index / 3.0) / 2
```

Where `priority_index` maps: Low=0, Medium=1, High=2, Critical=3.

| Alert Level | Score Range | Meaning |
|------------|-------------|---------|
| **RED** | >= 0.75 | Immediate life threat, deploy all resources |
| **ORANGE** | >= 0.50 | Significant damage, coordinate response |
| **YELLOW** | >= 0.25 | Moderate situation, monitor closely |
| **GREEN** | < 0.25 | Low risk, routine monitoring |

**Special rule**: If the crisis category is `not_humanitarian`, the alert is always GREEN regardless of other scores.

### Resource Needs Estimation

When the FusionLayer is active, resource needs come from the resource_head output (4 values in [0, 1]).

When in crisis-only fallback mode, resource needs are estimated from lookup tables:

| Resource | affected_individuals | infrastructure_damage | rescue_effort | not_humanitarian |
|----------|---------------------|----------------------|---------------|-----------------|
| Water | 0.50 | 0.40 | 0.45 | 0.0 |
| Medical | 0.65 | 0.30 | 0.50 | 0.0 |
| Rescue | 0.60 | 0.55 | 0.75 | 0.0 |
| Shelter | 0.50 | 0.45 | 0.60 | 0.0 |

These base values are scaled by the crisis model's confidence score.

### Base Resource Needs by Disaster Type (FusionLayer Training)

| Disaster | Water | Medical | Rescue | Shelter |
|----------|-------|---------|--------|---------|
| Fire | 0.35 | 0.45 | 0.65 | 0.40 |
| Storm | 0.65 | 0.55 | 0.75 | 0.80 |
| Earthquake | 0.75 | 0.85 | 0.90 | 0.70 |
| Flood | 0.90 | 0.50 | 0.75 | 0.60 |
| Unknown | 0.10 | 0.10 | 0.10 | 0.10 |

---

## 8. Server and API

**Source**: `fusion/server.py`

### FastAPI Server

- **Port**: 8001
- **Startup**: Loads all 3 models (IoT, Crisis, Fusion) into memory
- **Rate Limiting**: 10 requests per 60 seconds per IP (configurable via `RATE_LIMIT_MAX` and `RATE_LIMIT_WINDOW` env vars)
- **CORS**: Configurable via `ALLOWED_ORIGINS` env var (defaults to `*`)
- **Job Persistence**: Background jobs saved to `.jobs_store.json` on disk (survives server restart)

### API Endpoints

#### GET /health
Returns model load status.
```json
{
  "status": "ok",
  "pipeline": true,
  "crisis_model": true,
  "iot_model": true,
  "background_jobs": 2
}
```

#### POST /analyze (Synchronous)
Full pipeline analysis. Accepts multipart form data.

**Parameters**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | File | Yes | Image file (JPG/PNG) |
| `tweet` | string | Yes | Tweet or report text |
| `precipitation` | float | No | Weather: mm rainfall |
| `max_temp` | float | No | Weather: degrees C |
| `min_temp` | float | No | Weather: degrees C |
| `avg_wind_speed` | float | No | Weather: km/h |
| `wind_kts` | float | No | Storm: wind in knots |
| `pressure` | float | No | Storm: pressure in hPa |
| `lat` | float | No | Geographic latitude |
| `lon` | float | No | Geographic longitude |
| `depth` | float | No | Earthquake: depth in km |
| `magnitude` | float | No | Earthquake: Richter scale |
| `rms` | float | No | Seismic: RMS error |
| `n_stations` | int | No | Seismic: station count |
| `n_phases` | int | No | Seismic: phase count |
| `azimuth_gap` | int | No | Seismic: azimuth gap degrees |
| `elevation_m` | float | No | Flood: elevation meters |
| `distance_to_river_m` | float | No | Flood: distance to river |
| `rainfall_7d` | float | No | Flood: 7-day rainfall mm |
| `monthly_rainfall` | float | No | Flood: monthly rainfall mm |
| `drainage_index` | float | No | Flood: 0-1 drainage index |
| `ndvi` | float | No | Flood: vegetation index |
| `ndwi` | float | No | Flood: water index |
| `month` | int | No | Calendar month (1-12) |

**Response**:
```json
{
  "alert_level": "RED",
  "summary": "Sensors detect flood risk 72/100; social media: 'affected_individuals'. Priority: High.",
  "disaster_type": "flood",
  "priority": "High",
  "fused_severity": 0.7234,
  "iot": {
    "type": "flood",
    "fire_prob": 0.001,
    "storm_cat": 0.002,
    "eq_magnitude": 0.001,
    "flood_risk": 0.723,
    "casualty_risk": 0.434,
    "severity": 0.723,
    "sensor_weights": {
      "weather": 0.02,
      "storm": 0.01,
      "seismic": 0.02,
      "hydro": 0.95
    }
  },
  "crisis": {
    "category": "affected_individuals",
    "confidence": 0.423,
    "probabilities": {
      "affected_individuals": 0.423,
      "infrastructure_and_utility_damage": 0.312,
      "not_humanitarian": 0.089,
      "other_relevant_information": 0.121,
      "rescue_volunteering_or_donation_effort": 0.055
    },
    "vision_weight": 0.45,
    "text_weight": 0.55
  },
  "fusion": {
    "priority_probs": { "Low": 0.05, "Medium": 0.15, "High": 0.60, "Critical": 0.20 },
    "type_probs": { "fire": 0.01, "storm": 0.05, "earthquake": 0.02, "flood": 0.88, "unknown": 0.04 },
    "population_impact": 0.623,
    "resource_needs": { "water": 0.80, "medical": 0.50, "rescue": 0.70, "shelter": 0.60 }
  },
  "xai": {
    "gradcam_b64": "iVBORw0KGgo...",
    "summary": "**SITUATION**: Flooding detected in the region..."
  },
  "inference_ms": 2341
}
```

#### POST /analysis/jobs (Asynchronous)
Creates a background analysis job. Same parameters as `/analyze`.

```json
// Response
{ "job_id": "abc123", "status": "queued" }
```

#### GET /analysis/jobs/{job_id}
Poll job status.

```json
// While running
{ "job_id": "abc123", "status": "running" }

// When complete
{ "job_id": "abc123", "status": "completed", "result": { ... } }

// On failure
{ "job_id": "abc123", "status": "failed", "error": "..." }
```

#### POST /iot/predict
IoT-only prediction (bypasses crisis model).

**Parameters**: `disaster_type` (string) + sensor readings as form fields.

### Background Job Persistence

Jobs are stored in `.jobs_store.json` on disk. On server restart:
- Completed and failed jobs are preserved
- Jobs that were running or queued are marked as failed (since the process died)

The frontend polls for job status every 3 seconds via `app-shell.js`.

### Security Features

- **Path traversal prevention**: Page file serving validates that resolved paths stay within the pages directory
- **Rate limiting**: Per-IP request throttling on analysis endpoints
- **CORS restrictions**: Configurable allowed origins (vs. wide-open `*`)
- **Input validation**: Sensor readings validated for numeric type and range on the frontend

---

## 9. Web Dashboard

**Source**: `fusion/static/`

### Architecture

Vanilla JavaScript with an app-shell pattern. No build tools, no framework. Each page is a separate HTML file loaded by the server, with shared header/footer rendered by `app-shell.js`.

### Pages

| Page | URL | Purpose |
|------|-----|---------|
| Overview | `/overview` | Dashboard summary: current alert, recent incidents, workflow guide |
| New Analysis | `/analysis` | Upload image + tweet + sensor data, run analysis, see results |
| Incident Details | `/incident` | Deep dive into latest result: Grad-CAM heatmap, briefing, all metrics |
| IoT Monitor | `/iot-monitor` | IoT-only predictions, sensor weight visualization |
| Reports | `/reports` | History of all analyses (40 max), search, compare, export JSON |

### State Management (store.js)

Uses browser LocalStorage with three keys:

| Key | Purpose | Max Items |
|-----|---------|-----------|
| `fusion.analysis.history.v1` | Saved analysis records | 40 |
| `fusion.analysis.active.v1` | Currently viewed record ID | 1 |
| `fusion.analysis.pending.v1` | Background jobs in-flight | unlimited |

### Background Job Sync (app-shell.js)

Every 3 seconds, the shell:
1. Checks all pending jobs via `GET /analysis/jobs/{id}`
2. If completed: saves result to history, removes from pending, dispatches `fusion:job-saved` event
3. If failed: removes from pending, dispatches `fusion:job-failed` event
4. Polls `/health` for pipeline status indicator

### CSS Design System (base.css)

Dark theme with warm accent colors:

```
Background:  #071018  (deep navy)
Accent:      #d6b07a  (warm gold)
Accent Soft: #8bbabf  (cool cyan)
Success:     #79c7a4  (green)
Warning:     #d6b07a  (gold)
Danger:      #d07a74  (red)
```

Max container width: 1480px. Responsive grid layouts (2/3/4 column).

---

## 10. Datasets

### IoT Training Data (~50K+ samples)

| Dataset | Source File | Samples | Features | Disaster Type |
|---------|-----------|---------|----------|---------------|
| CA Wildfires (1984-2025) | `CA_Weather_Fire_Dataset_1984-2025.csv` | ~5K fire + 2K non-fire | Precipitation, temp, wind, drought | Fire / Unknown |
| Historical Tropical Storms | `Historical_Tropical_Storm_Tracks.csv` | 8,000 | Wind, pressure, lat/lon, category | Storm |
| Atlantic Storms | `storms.csv` | 6,000 | Wind, pressure, lat/lon, category | Storm |
| Global Earthquakes | `earthquakes.csv` | ~2,500 | Depth, magnitude, gap, location | Earthquake |
| Iran Earthquakes | `iran earthquakes.csv` | 10,000 | Depth, magnitude, RMS, stations | Earthquake |
| Sri Lanka Floods | `sri_lanka_flood_risk_dataset_25000.csv` | 25,000 | Elevation, rainfall, NDVI, NDWI | Flood |

### Crisis Social Media Data

**CrisisMMD v2.0** — Multimodal Crisis Management Dataset

- **Size**: 13,500+ image-text pairs
- **Events**: 7 real-world disasters from 2017-2019
  - California Wildfires (2019)
  - Hurricane Harvey (2017)
  - Hurricane Irma (2017)
  - Hurricane Maria (2017)
  - Iraq-Iran Earthquake (2017)
  - Mexico Earthquake (2017)
  - Sri Lanka Floods (2017)
- **Tasks**: 3 classification tasks
  - Task 1: Informative vs Not-Informative (2 classes)
  - Task 2: Humanitarian Category (7 classes)
  - Task 3: Damage Severity (3 classes)
- **Split**: Train / Dev / Test provided in TSV files
- **Used in this project**: Task 2 (humanitarian), 5 classes (pipeline uses 5 of the 7)

---

## 11. Training Procedures

### Step 1: Train IoT Model

```bash
python IOT/train_iot.py
```

- Loads all 6 CSV datasets from `IOT/datasets/`
- Builds unified 32-dim feature vectors
- Trains AdaptiveIoTClassifier for 40 epochs
- Uses WeightedRandomSampler for class balance
- Saves best model (by macro-F1) to `IOT/models/iot_model.pth`

### Step 2: Train Crisis Model

The crisis model (`crisis/best_adaptive_model.pth`) is trained separately on CrisisMMD. The checkpoint includes the full AdaptiveFusionClassifier with frozen BLIP and XLM-RoBERTa weights.

### Step 3: Train Fusion Layer

```bash
python fusion/train_fusion.py
```

- Loads pre-trained IoT model from Step 1
- Reads CrisisMMD train split (13K+ samples)
- For each sample: generates synthetic sensor features based on event type, extracts real IoT embedding, generates synthetic crisis embedding
- Trains FusionLayer with 5 loss terms (severity + priority + disaster + population + resources)
- Saves best model (by val loss) to `fusion/fusion_model.pth`

### Step 4 (Optional): Evaluate IoT Model

```bash
python IOT/evaluate_iot.py
```

Generates 8 evaluation figures + `metrics.json` in `IOT/evaluation/`:
- Confusion matrix
- ROC curves (per-class)
- PR curves (per-class)
- Per-class F1 scores
- Severity regression scatter plot
- Sensor weight distribution
- Training history curves
- Classification report

### Training Hyperparameter Summary

| Parameter | IoT Model | Fusion Layer |
|-----------|-----------|-------------|
| Batch Size | 256 | 128 |
| Epochs | 40 | 40 |
| Optimizer | AdamW | AdamW |
| Learning Rate | 1e-3 | 1e-3 |
| Weight Decay | 1e-4 | 1e-4 |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR |
| Grad Clip | 1.0 | 1.0 |
| Train/Val Split | 85/15 | 80/20 |
| Loss | CE + MSE + MSE + MSE | MSE + CE + CE + MSE + MSE |
| Best Metric | Macro-F1 | Val Loss |

---

## 12. Quick Start

### Prerequisites

```bash
conda create -n crisismmd python=3.10
conda activate crisismmd
pip install -r requirements.txt
```

### Environment Variables (Optional)

```bash
export OPENAI_API_KEY=sk-...           # For GPT-4o briefings (XAI)
export ALLOWED_ORIGINS=http://localhost:8001  # CORS origins
export RATE_LIMIT_MAX=10               # Requests per window
export RATE_LIMIT_WINDOW=60            # Window in seconds
```

### Run the Server

```bash
python -m uvicorn fusion.server:app --host 0.0.0.0 --port 8001 --reload
```

On startup, the server loads all 3 models:
```
[Pipeline] Device: cpu
[Pipeline] Loading IoT model...
  [IoT] Model loaded -- val F1=0.9234  val acc=0.9456
[Pipeline] Loading BLIP + XLM-RoBERTa...
[Pipeline] Crisis model ready (with BLIP captioning).
[Pipeline] Loading FusionLayer...
[Pipeline] Fusion model loaded.
```

Open browser: http://localhost:8001

### (Re)Train Models

```bash
# Step 1: Train IoT model
python IOT/train_iot.py

# Step 2: Train Fusion layer (requires trained IoT model)
python fusion/train_fusion.py

# Step 3 (optional): Generate evaluation figures
python IOT/evaluate_iot.py
```

---

## 13. Directory Structure

```
Multimodal-diaster-management/
|
|-- IOT/                                 # IoT sensor module
|   |-- train_iot.py                     # Model definition + training script
|   |-- evaluate_iot.py                  # Evaluation + figure generation
|   |-- datasets/                        # 6 CSV datasets (~50K samples)
|   |   |-- CA_Weather_Fire_Dataset_1984-2025.csv
|   |   |-- Historical_Tropical_Storm_Tracks.csv
|   |   |-- storms.csv
|   |   |-- earthquakes.csv
|   |   |-- iran earthquakes.csv
|   |   |-- sri_lanka_flood_risk_dataset_25000.csv
|   |-- models/
|   |   |-- iot_model.pth                # Trained AdaptiveIoTClassifier (~180 KB)
|   |-- evaluation/                      # Generated figures + metrics.json
|
|-- crisis/                              # Crisis social media module
|   |-- server.py                        # AdaptiveFusionClassifier + standalone API
|   |-- best_adaptive_model.pth          # Trained crisis model (~450 MB)
|   |-- Dataset/CrisisMMD_v2.0/         # CrisisMMD dataset (13.5K+ samples)
|   |-- final_conference_package_crisis/ # Evaluation figures
|
|-- fusion/                              # Cross-modal fusion + unified server
|   |-- pipeline.py                      # DisasterPipeline: orchestrates all 3 models
|   |-- server.py                        # FastAPI unified server (all endpoints)
|   |-- fusion_layer.py                  # FusionLayer neural architecture
|   |-- train_fusion.py                  # Fusion layer training script
|   |-- fusion_model.pth                 # Trained FusionLayer (~2 MB)
|   |-- iot_predictor.py                 # IoT inference wrapper
|   |-- crisis_model.py                  # CrisisMMD dataset + model definitions
|   |-- xai.py                           # Grad-CAM + GPT-4o explanations
|   |-- static/
|       |-- index.html                   # Redirect to /overview
|       |-- css/
|       |   |-- base.css                 # Design system + component styles
|       |-- js/
|       |   |-- api.js                   # API client (fetch wrappers)
|       |   |-- app-shell.js             # Header, footer, job sync, health polling
|       |   |-- store.js                 # LocalStorage state management
|       |   |-- utils.js                 # Formatting utilities
|       |   |-- pages/
|       |       |-- overview.js          # Overview dashboard logic
|       |       |-- analysis.js          # Analysis form + result display
|       |       |-- incident.js          # Incident detail view
|       |       |-- iot-monitor.js       # IoT-only prediction UI
|       |       |-- reports.js           # History, search, export
|       |-- pages/
|           |-- overview.html
|           |-- analysis.html
|           |-- incident.html
|           |-- iot-monitor.html
|           |-- reports.html
|
|-- generate_arch.py                     # Architecture diagram generator
|-- requirements.txt                     # Python dependencies
|-- torun.txt                            # Quick start instructions
|-- .gitignore
|-- DOCUMENTATION.md                     # This file
```

---

## Appendix A: Novel Technical Contributions

1. **Adaptive Confidence-Weighted Sensor Fusion**: Per-sensor-group confidence estimation that automatically down-weights inactive sensor groups, preventing noise from irrelevant data streams.

2. **Cross-Group Multi-Head Attention for IoT**: Allows seismic signals to reinforce hydro signals (earthquake-triggered floods), capturing cascading disaster interactions that independent classifiers would miss.

3. **Gradient-Weighted Attention Rollout**: Combines ViT attention maps with gradient signals to produce accurate visual explanations. Solves the fundamental problem that standard Grad-CAM fails on ViT architectures where only the CLS token has downstream gradients.

4. **Dual-Direction Cross-Modal Attention**: Both the IoT and crisis embeddings attend to each other (in the crisis model), allowing bidirectional information flow between vision and text modalities.

5. **Multi-Task Fusion Learning**: A single shared representation simultaneously predicts severity, priority, disaster type, population impact, and resource needs, with each head providing complementary learning signals.

6. **Graceful Modality Degradation**: The pipeline handles missing IoT data seamlessly by falling back to crisis-only assessment with BLIP captioning for visual context and keyword matching for hazard inference.

---

## Appendix B: Key Hyperparameters and Thresholds

| Parameter | Value | Location | Purpose |
|-----------|-------|----------|---------|
| IMAGE_SIZE | 224 x 224 | pipeline.py, crisis_model.py | ViT input size |
| MAX_TOKEN_LENGTH | 128 | pipeline.py | XLM-RoBERTa token limit |
| TEMPERATURE | 2.5 | pipeline.py | Softmax temperature scaling |
| HIDDEN_DIM (IoT) | 128 | train_iot.py | Per-group encoder output |
| HIDDEN_DIM (Crisis) | 512 | crisis/server.py | Projection dimension |
| PROJ_DIM (Fusion) | 256 | fusion_layer.py | Shared projection space |
| CRISIS_DIM | 1024 | fusion_layer.py | 512 vision + 512 text |
| ATTENTION_HEADS (IoT) | 4 | train_iot.py | Cross-group attention |
| ATTENTION_HEADS (Crisis) | 8 | crisis/server.py | Cross-modal attention |
| ROLLOUT_LAYERS | 4 (last) | xai.py | Grad-CAM attention layers |
| THRESHOLD_PERCENTILE | 40th | xai.py | Attention suppression |
| HEATMAP_BLEND | 45%/55% | xai.py | Heatmap vs original image |
| GAUSSIAN_KERNEL | 15 x 15 | xai.py | Heatmap smoothing |
| ALERT_RED | >= 0.75 | pipeline.py | Alert threshold |
| ALERT_ORANGE | >= 0.50 | pipeline.py | Alert threshold |
| ALERT_YELLOW | >= 0.25 | pipeline.py | Alert threshold |
| MAX_HISTORY | 40 | store.js | LocalStorage record limit |
| JOB_POLL_INTERVAL | 3000 ms | app-shell.js | Background job polling |
| RATE_LIMIT | 10 / 60s | server.py | API rate limit per IP |
