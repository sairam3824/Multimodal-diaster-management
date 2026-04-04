"""
Tri-Fusion Layer — Crisis x IoT x Satellite
==============================================
Extends the dual-fusion (IoT x Crisis) to a tri-modal architecture that
gracefully handles missing optional modalities.

Modality requirements:
  crisis_embedding    : MANDATORY  [B, 1024]
  iot_embedding       : OPTIONAL   [B, 128]
  satellite_embedding : OPTIONAL   [B, 640]

Design:
  1. Project each modality into a shared latent space (proj_dim=256)
  2. Binary modality masks determine which inputs are active
  3. Pairwise cross-attention between all active modality pairs
  4. Adaptive gating weights how much each modality contributes
  5. Concatenation + shared MLP -> output heads

The output shape is deterministic regardless of which modalities are present.
Missing modalities are replaced with learned default embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as torchF
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

from fusion.fusion_layer import RESOURCE_NAMES, PRIORITY_LABELS, DISASTER_LABELS


@dataclass
class TriFusionOutput:
    severity_score:      torch.Tensor   # [B] float 0-1
    priority_logits:     torch.Tensor   # [B, 4]
    disaster_logits:     torch.Tensor   # [B, 5]
    population_impact:   torch.Tensor   # [B] float 0-1
    resource_needs:      torch.Tensor   # [B, 4]  water/medical/rescue/shelter
    fused_embedding:     torch.Tensor   # [B, 256] final shared representation
    modality_weights:    Dict[str, torch.Tensor]  # per-modality contribution weights


class PairwiseCrossAttention(nn.Module):
    """Cross-attention: query attends to key/value from another modality."""

    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # query, context: [B, dim] -> unsqueeze to [B, 1, dim] for MHA
        q = query.unsqueeze(1)
        c = context.unsqueeze(1)
        attended, _ = self.attn(q, c, c)
        return self.norm(query + attended.squeeze(1))


class ModalityGate(nn.Module):
    """Learns adaptive weights for each modality based on content."""

    def __init__(self, proj_dim: int, n_modalities: int = 3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(proj_dim * n_modalities, 128),
            nn.ReLU(),
            nn.Linear(128, n_modalities),
        )

    def forward(
        self, projections: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            projections: [B, n_modalities, proj_dim] stacked modality projections
            mask: [B, n_modalities] binary mask (1=present, 0=absent)
        Returns:
            weights: [B, n_modalities] normalized gate weights
        """
        flat = projections.flatten(1)  # [B, n_modalities * proj_dim]
        raw = self.gate(flat)          # [B, n_modalities]
        # Mask out missing modalities before softmax
        raw = raw.masked_fill(mask == 0, float("-inf"))
        return torchF.softmax(raw, dim=-1)


class TriFusionLayer(nn.Module):
    """
    Tri-modal fusion: Crisis (required) + IoT (optional) + Satellite (optional).

    Missing modalities are replaced with learned default embeddings. An adaptive
    gate and pairwise cross-attention ensure the model leverages whatever
    information is available.

    Parameters
    ----------
    crisis_dim    : 1024 (512 vision + 512 text from AdaptiveFusionClassifier)
    iot_dim       : 128  (AdaptiveIoTClassifier hidden_dim)
    satellite_dim : 640  (F_sat 512 + F_region 128 from DeepLabV3Plus_xBD)
    proj_dim      : shared projection dimension (default 256)
    """

    def __init__(
        self,
        crisis_dim:    int = 1024,
        iot_dim:       int = 128,
        satellite_dim: int = 640,
        proj_dim:      int = 256,
        dropout:       float = 0.2,
    ):
        super().__init__()
        self.proj_dim = proj_dim

        # ── Modality projections ─────────────────────────────────────────
        self.crisis_proj = nn.Sequential(
            nn.Linear(crisis_dim, proj_dim * 2),
            nn.LayerNorm(proj_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        self.iot_proj = nn.Sequential(
            nn.Linear(iot_dim, proj_dim * 2),
            nn.LayerNorm(proj_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        self.satellite_proj = nn.Sequential(
            nn.Linear(satellite_dim, proj_dim * 2),
            nn.LayerNorm(proj_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        # ── Learned defaults for missing modalities ──────────────────────
        self.iot_default = nn.Parameter(torch.randn(1, proj_dim) * 0.01)
        self.satellite_default = nn.Parameter(torch.randn(1, proj_dim) * 0.01)

        # ── Pairwise cross-attention ─────────────────────────────────────
        # crisis <-> iot
        self.cross_crisis_iot = PairwiseCrossAttention(proj_dim)
        # crisis <-> satellite
        self.cross_crisis_sat = PairwiseCrossAttention(proj_dim)
        # iot <-> satellite
        self.cross_iot_sat = PairwiseCrossAttention(proj_dim)

        # ── Adaptive modality gate ───────────────────────────────────────
        self.gate = ModalityGate(proj_dim, n_modalities=3)

        # ── Shared MLP (from gated + cross-attended representations) ─────
        # Final fused = weighted sum of 3 projections (each proj_dim)
        # + 3 cross-attention outputs (each proj_dim)
        # = 6 * proj_dim concatenated
        fused_input_dim = proj_dim * 6
        self.shared_mlp = nn.Sequential(
            nn.Linear(fused_input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Output heads ─────────────────────────────────────────────────
        self.severity_head = nn.Linear(256, 1)
        self.priority_head = nn.Linear(256, 4)
        self.disaster_head = nn.Linear(256, 5)
        self.population_head = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
        self.resource_head = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, len(RESOURCE_NAMES)), nn.Sigmoid(),
        )

    def forward(
        self,
        crisis_embedding:    torch.Tensor,                 # [B, crisis_dim] REQUIRED
        iot_embedding:       Optional[torch.Tensor] = None,  # [B, iot_dim]
        satellite_embedding: Optional[torch.Tensor] = None,  # [B, satellite_dim]
    ) -> TriFusionOutput:
        B = crisis_embedding.size(0)
        device = crisis_embedding.device

        # ── Build modality mask [B, 3] ───────────────────────────────────
        crisis_mask = torch.ones(B, 1, device=device)
        iot_mask = torch.ones(B, 1, device=device) if iot_embedding is not None else torch.zeros(B, 1, device=device)
        sat_mask = torch.ones(B, 1, device=device) if satellite_embedding is not None else torch.zeros(B, 1, device=device)
        mask = torch.cat([crisis_mask, iot_mask, sat_mask], dim=1)  # [B, 3]

        # ── Project modalities ───────────────────────────────────────────
        crisis_p = self.crisis_proj(crisis_embedding)  # [B, proj_dim]

        if iot_embedding is not None:
            iot_p = self.iot_proj(iot_embedding)
        else:
            iot_p = self.iot_default.expand(B, -1)

        if satellite_embedding is not None:
            sat_p = self.satellite_proj(satellite_embedding)
        else:
            sat_p = self.satellite_default.expand(B, -1)

        # ── Pairwise cross-attention (skip when modality absent) ─────────
        if iot_embedding is not None:
            cross_ci = self.cross_crisis_iot(crisis_p, iot_p)
        else:
            cross_ci = crisis_p  # self-identity when IoT missing

        if satellite_embedding is not None:
            cross_cs = self.cross_crisis_sat(crisis_p, sat_p)
        else:
            cross_cs = crisis_p

        if iot_embedding is not None and satellite_embedding is not None:
            cross_is = self.cross_iot_sat(iot_p, sat_p)
        elif iot_embedding is not None:
            cross_is = iot_p
        elif satellite_embedding is not None:
            cross_is = sat_p
        else:
            cross_is = crisis_p  # crisis-only fallback

        # ── Adaptive gating ──────────────────────────────────────────────
        stacked = torch.stack([crisis_p, iot_p, sat_p], dim=1)  # [B, 3, proj_dim]
        weights = self.gate(stacked, mask)  # [B, 3]

        # Weighted sum of base projections
        gated = (
            weights[:, 0:1] * crisis_p
            + weights[:, 1:2] * iot_p
            + weights[:, 2:3] * sat_p
        )

        # ── Fuse everything ──────────────────────────────────────────────
        fused = torch.cat([
            crisis_p, gated, cross_ci, cross_cs, cross_is,
            sat_p if satellite_embedding is not None else iot_p if iot_embedding is not None else crisis_p,
        ], dim=1)  # [B, 6 * proj_dim]

        rep = self.shared_mlp(fused)  # [B, 256]

        # ── Output heads ─────────────────────────────────────────────────
        severity = torch.sigmoid(self.severity_head(rep)).squeeze(-1)
        priority = self.priority_head(rep)
        disaster = self.disaster_head(rep)
        population = self.population_head(rep).squeeze(-1)
        resources = self.resource_head(rep)

        modality_weights = {
            "crisis": weights[:, 0].detach(),
            "iot": weights[:, 1].detach(),
            "satellite": weights[:, 2].detach(),
        }

        return TriFusionOutput(
            severity_score=severity,
            priority_logits=priority,
            disaster_logits=disaster,
            population_impact=population,
            resource_needs=resources,
            fused_embedding=rep,
            modality_weights=modality_weights,
        )


# ──────────────────────────────────────────────
# Backward compatibility: wraps TriFusionLayer
# to accept dual-fusion calls (iot + crisis only)
# ──────────────────────────────────────────────

def upgrade_fusion_state_dict(old_state: dict) -> dict:
    """
    Map old FusionLayer state_dict keys to TriFusionLayer where possible.
    Returns a partial state dict suitable for strict=False loading.
    """
    mapping = {}
    for k, v in old_state.items():
        if k.startswith("iot_proj."):
            mapping[k] = v
        elif k.startswith("crisis_proj."):
            mapping[k] = v
        elif k.startswith("severity_head."):
            mapping[k] = v
        elif k.startswith("priority_head."):
            mapping[k] = v
        elif k.startswith("disaster_head."):
            mapping[k] = v
        elif k.startswith("population_head."):
            mapping[k] = v
        elif k.startswith("resource_head."):
            mapping[k] = v
    return mapping
