# Tri-Fusion Training & Ablation Metrics

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | TriFusionLayer (pairwise cross-attention + adaptive gating) |
| Crisis dim | 1024 (512 vision + 512 text) |
| IoT dim | 128 (AdaptiveIoTClassifier) |
| Satellite dim | 640 (F_sat 512 + F_region 128) |
| Projection dim | 256 |
| Total parameters | 3,163,602 |
| Training samples | 13,608 (CrisisMMD humanitarian split) |
| Train/Val split | 80/20 (10,886 / 2,722) |
| Epochs | 40 |
| Learning rate | 1e-3 (AdamW, CosineAnnealing) |
| Batch size | 128 |
| Modality dropout | 30% (IoT and satellite independently) |
| Device | CPU |
| Best val loss | 0.0288 (epoch 15) |
| Final train loss | 0.1574 |
| IoT model val F1 | 0.9059 |

## Training Convergence

| Epoch | Train Loss | Val Loss | Priority Acc | Disaster Acc |
|-------|-----------|----------|-------------|-------------|
| 1 | 0.4855 | 0.0613 | 0.992 | 1.000 |
| 5 | 0.2077 | 0.0318 | 0.994 | 1.000 |
| 10 | 0.1909 | 0.0313 | 0.994 | 1.000 |
| 15 | 0.1914 | 0.0288 | 0.994 | 1.000 |
| 20 | 0.1856 | 0.0290 | 0.994 | 1.000 |
| 30 | 0.1793 | 0.0322 | 0.994 | 1.000 |
| 40 | 0.1574 | 0.0380 | 0.994 | 1.000 |

## Ablation Study Results (2,722 validation samples)

| Configuration | Severity MAE | Priority Acc | Disaster Acc | Population MAE | Resource MAE |
|---------------|-------------|-------------|-------------|---------------|-------------|
| Crisis only | 0.1165 | 68.96% | 100.00% | 0.0877 | 0.0468 |
| Crisis + IoT | 0.0989 | 72.37% | 100.00% | 0.0691 | 0.0393 |
| Crisis + Satellite | 0.0415 | 98.75% | 100.00% | 0.0274 | 0.0164 |
| Crisis + IoT + Satellite | **0.0398** | **99.41%** | **100.00%** | **0.0248** | **0.0161** |

## Key Findings for Paper

### 1. Satellite modality is the strongest contributor
- Adding satellite alone reduces Severity MAE by **64.4%** (0.1165 -> 0.0415)
- Adding satellite boosts Priority Accuracy by **+29.8pp** (68.96% -> 98.75%)
- Satellite provides direct physical damage evidence (segmentation-based features)

### 2. IoT provides incremental but meaningful improvement
- Adding IoT reduces Severity MAE by **15.1%** (0.1165 -> 0.0989)
- Adding IoT boosts Priority Accuracy by **+3.4pp** (68.96% -> 72.37%)
- IoT contributes environmental context (weather, seismic, hydrological signals)

### 3. Full tri-fusion achieves best results across all metrics
- Severity MAE: 0.0398 (best)
- Priority Accuracy: 99.41% (best)
- Population Impact MAE: 0.0248 (best)
- Resource Needs MAE: 0.0161 (best)
- Tri-fusion outperforms best single addition (crisis+satellite) by further **4.1%** Severity MAE reduction

### 4. Graceful degradation confirmed
- 30% modality dropout during training enables robust missing-modality handling
- Crisis-only mode still achieves 68.96% Priority Accuracy (usable baseline)
- Each added modality monotonically improves all metrics

### 5. Disaster type classification is saturated
- 100% accuracy across all configurations
- The 5-class disaster type (fire/storm/earthquake/flood/unknown) is well-separated in the embedding spaces
- This metric differentiates less between configs; severity and priority are more informative

## Relative Improvement Table (vs Crisis-Only Baseline)

| Configuration | Sev MAE Reduction | Pri Acc Gain | Pop MAE Reduction | Res MAE Reduction |
|---------------|-------------------|-------------|-------------------|-------------------|
| + IoT | -15.1% | +3.41pp | -21.2% | -16.0% |
| + Satellite | -64.4% | +29.79pp | -68.8% | -65.0% |
| + IoT + Satellite | -65.8% | +30.45pp | -71.7% | -65.6% |
