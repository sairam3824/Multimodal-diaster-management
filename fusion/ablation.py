"""
Ablation Study — Tri-Fusion Modality Contribution Analysis
============================================================
Compares 4 modality configurations:
  1. Crisis only
  2. Crisis + IoT
  3. Crisis + Satellite
  4. Crisis + IoT + Satellite

For each configuration, evaluates:
  - Severity MAE
  - Priority accuracy
  - Disaster type accuracy
  - Population impact MAE
  - Resource needs MAE

Also provides ablation-at-inference: modality contribution reporting
by dropping one modality at a time and measuring confidence shifts.

Usage:
    python fusion/ablation.py
"""

import os, sys, json, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from fusion.tri_fusion_layer import TriFusionLayer, TriFusionOutput
from fusion.fusion_layer import PRIORITY_LABELS, DISASTER_LABELS, RESOURCE_NAMES


OUTPUT_DIR = Path(os.path.dirname(__file__)) / ".." / "outputs" / "ablation"


def run_ablation_study(
    model: TriFusionLayer,
    crisis_X: torch.Tensor,
    iot_X: torch.Tensor,
    sat_X: torch.Tensor,
    sev_Y: torch.Tensor,
    pri_Y: torch.Tensor,
    dis_Y: torch.Tensor,
    pop_Y: torch.Tensor,
    res_Y: torch.Tensor,
    device: torch.device,
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Run full ablation study across all 4 modality configurations.

    Returns a dict with per-configuration metrics.
    """
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.to(device)

    configs = {
        "crisis_only":           {"crisis": True, "iot": False, "satellite": False},
        "crisis_iot":            {"crisis": True, "iot": True,  "satellite": False},
        "crisis_satellite":      {"crisis": True, "iot": False, "satellite": True},
        "crisis_iot_satellite":  {"crisis": True, "iot": True,  "satellite": True},
    }

    results = {}
    n = len(crisis_X)

    for config_name, flags in configs.items():
        print(f"\n  [{config_name}] Evaluating ({n} samples)...")

        sev_errors = []
        pop_errors = []
        res_errors = []
        pri_correct = 0
        dis_correct = 0

        batch_size = 128
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            c = crisis_X[start:end].to(device)
            i = iot_X[start:end].to(device) if flags["iot"] else None
            s = sat_X[start:end].to(device) if flags["satellite"] else None

            with torch.no_grad():
                out = model(c, i, s)

            sev_pred = out.severity_score.cpu()
            pri_pred = out.priority_logits.argmax(dim=1).cpu()
            dis_pred = out.disaster_logits.argmax(dim=1).cpu()
            pop_pred = out.population_impact.cpu()
            res_pred = out.resource_needs.cpu()

            sev_errors.append((sev_pred - sev_Y[start:end]).abs())
            pop_errors.append((pop_pred - pop_Y[start:end]).abs())
            res_errors.append((res_pred - res_Y[start:end]).abs().mean(dim=1))
            pri_correct += (pri_pred == pri_Y[start:end]).sum().item()
            dis_correct += (dis_pred == dis_Y[start:end]).sum().item()

        sev_mae = torch.cat(sev_errors).mean().item()
        pop_mae = torch.cat(pop_errors).mean().item()
        res_mae = torch.cat(res_errors).mean().item()
        pri_acc = pri_correct / n
        dis_acc = dis_correct / n

        results[config_name] = {
            "severity_mae": round(sev_mae, 4),
            "priority_accuracy": round(pri_acc, 4),
            "disaster_accuracy": round(dis_acc, 4),
            "population_mae": round(pop_mae, 4),
            "resource_mae": round(res_mae, 4),
            "active_modalities": [k for k, v in flags.items() if v],
        }

        print(f"    Severity MAE:      {sev_mae:.4f}")
        print(f"    Priority Accuracy: {pri_acc:.4f}")
        print(f"    Disaster Accuracy: {dis_acc:.4f}")
        print(f"    Population MAE:    {pop_mae:.4f}")
        print(f"    Resource MAE:      {res_mae:.4f}")

    # Save results
    report_path = out_dir / "ablation_results.json"
    with open(str(report_path), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAblation results saved -> {report_path}")

    # Print comparison table
    print("\n" + "=" * 80)
    print(f"{'Configuration':<25} {'Sev MAE':>8} {'Pri Acc':>8} {'Dis Acc':>8} {'Pop MAE':>8} {'Res MAE':>8}")
    print("-" * 80)
    for cfg, m in results.items():
        print(f"{cfg:<25} {m['severity_mae']:>8.4f} {m['priority_accuracy']:>8.4f} "
              f"{m['disaster_accuracy']:>8.4f} {m['population_mae']:>8.4f} {m['resource_mae']:>8.4f}")
    print("=" * 80)

    return results


def ablation_at_inference(
    model: TriFusionLayer,
    crisis_embedding: torch.Tensor,
    iot_embedding: Optional[torch.Tensor] = None,
    satellite_embedding: Optional[torch.Tensor] = None,
) -> Dict:
    """
    Ablation-at-inference: predict with all modalities, then drop one at a time.
    Reports confidence shifts to estimate per-modality contribution.

    Args:
        model: Trained TriFusionLayer
        crisis_embedding: [1, 1024] REQUIRED
        iot_embedding: [1, 128] or None
        satellite_embedding: [1, 640] or None

    Returns:
        Dict with full prediction and per-modality contribution estimates.
    """
    model.eval()

    def _predict(c, i, s):
        with torch.no_grad():
            out = model(c, i, s)
        return {
            "severity": float(out.severity_score.item()),
            "priority": PRIORITY_LABELS[int(out.priority_logits.argmax(dim=1).item())],
            "priority_confidence": float(F.softmax(out.priority_logits, dim=-1).max().item()),
            "disaster_type": DISASTER_LABELS[int(out.disaster_logits.argmax(dim=1).item())],
            "disaster_confidence": float(F.softmax(out.disaster_logits, dim=-1).max().item()),
            "population_impact": float(out.population_impact.item()),
            "modality_weights": {k: float(v.mean().item()) for k, v in out.modality_weights.items()},
        }

    full = _predict(crisis_embedding, iot_embedding, satellite_embedding)

    report = {
        "full_prediction": full,
        "contributions": {},
    }

    # Crisis-only baseline
    crisis_only = _predict(crisis_embedding, None, None)
    report["contributions"]["crisis_baseline"] = crisis_only

    # Drop IoT
    if iot_embedding is not None:
        no_iot = _predict(crisis_embedding, None, satellite_embedding)
        report["contributions"]["drop_iot"] = {
            "prediction_without_iot": no_iot,
            "severity_shift": round(full["severity"] - no_iot["severity"], 4),
            "priority_confidence_shift": round(
                full["priority_confidence"] - no_iot["priority_confidence"], 4
            ),
        }

    # Drop Satellite
    if satellite_embedding is not None:
        no_sat = _predict(crisis_embedding, iot_embedding, None)
        report["contributions"]["drop_satellite"] = {
            "prediction_without_satellite": no_sat,
            "severity_shift": round(full["severity"] - no_sat["severity"], 4),
            "priority_confidence_shift": round(
                full["priority_confidence"] - no_sat["priority_confidence"], 4
            ),
        }

    # Summary
    report["summary"] = {
        "crisis_gate_weight": full["modality_weights"].get("crisis", 0),
        "iot_gate_weight": full["modality_weights"].get("iot", 0),
        "satellite_gate_weight": full["modality_weights"].get("satellite", 0),
        "severity_gain_from_optional": round(
            full["severity"] - crisis_only["severity"], 4
        ),
    }

    return report


if __name__ == "__main__":
    # Quick demo with synthetic data
    import random

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = TriFusionLayer(
        crisis_dim=1024, iot_dim=128, satellite_dim=640, proj_dim=256
    ).to(device)

    n = 200
    crisis_X = torch.randn(n, 1024)
    iot_X = torch.randn(n, 128)
    sat_X = torch.randn(n, 640)
    sev_Y = torch.rand(n)
    pri_Y = torch.randint(0, 4, (n,))
    dis_Y = torch.randint(0, 5, (n,))
    pop_Y = torch.rand(n)
    res_Y = torch.rand(n, 4)

    print("Running ablation study with synthetic data...")
    results = run_ablation_study(
        model, crisis_X, iot_X, sat_X,
        sev_Y, pri_Y, dis_Y, pop_Y, res_Y,
        device=device,
    )

    print("\nRunning ablation-at-inference...")
    report = ablation_at_inference(
        model,
        crisis_X[:1].to(device),
        iot_X[:1].to(device),
        sat_X[:1].to(device),
    )
    print(json.dumps(report, indent=2))
