"""
Fusion Layer — IoT × Crisis
============================
Combines the AdaptiveIoTClassifier embedding (128-dim) with the
AdaptiveFusionClassifier embedding (1024-dim) from the crisis model,
using cross-modal attention for enriched disaster situational awareness.

Architecture:
  IoT embedding    [B, 128]  → IoTProjection    → iot_proj    [B, proj_dim]
  Crisis embedding [B, 1024] → CrisisProjection → crisis_proj [B, proj_dim]

  CrossModalAttention:
    Q = iot_proj,  K/V = crisis_proj
    → attended IoT representation enriched with social-media context

  Concat(iot_proj, attended, crisis_proj) → MLP → FusionOutput
    ├── severity_score  (0-1 regression)
    ├── priority_class  (Low / Medium / High / Critical)
    └── disaster_type   (fire / storm / earthquake / flood / unknown)
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


RESOURCE_NAMES = ["water", "medical", "rescue", "shelter"]


@dataclass
class FusionOutput:
    severity_score:    torch.Tensor   # [B] float 0-1
    priority_logits:   torch.Tensor   # [B, 4]
    disaster_logits:   torch.Tensor   # [B, 5]
    population_impact: torch.Tensor   # [B] float 0-1
    resource_needs:    torch.Tensor   # [B, 4]  water/medical/rescue/shelter
    fused_embedding:   torch.Tensor   # [B, 256]


PRIORITY_LABELS = ["Low", "Medium", "High", "Critical"]
DISASTER_LABELS = ["fire", "storm", "earthquake", "flood", "unknown"]


class CrossModalAttention(nn.Module):
    """
    IoT queries the crisis social-media context.
    Query = IoT (sensor signals), Key/Value = Crisis (social media).
    """
    def __init__(self, dim: int):
        super().__init__()
        self.q_proj   = nn.Linear(dim, dim)
        self.k_proj   = nn.Linear(dim, dim)
        self.v_proj   = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.scale    = dim ** -0.5

    def forward(self, iot_proj: torch.Tensor, crisis_proj: torch.Tensor):
        Q = self.q_proj(iot_proj).unsqueeze(1)       # [B, 1, dim]
        K = self.k_proj(crisis_proj).unsqueeze(1)    # [B, 1, dim]
        V = self.v_proj(crisis_proj).unsqueeze(1)    # [B, 1, dim]
        attn = torch.bmm(Q, K.transpose(1, 2)) * self.scale   # [B, 1, 1]
        attn = torch.softmax(attn, dim=-1)
        out  = torch.bmm(attn, V).squeeze(1)                   # [B, dim]
        return self.out_proj(out)


class FusionLayer(nn.Module):
    """
    Cross-modal fusion of IoT sensor embeddings and crisis social-media embeddings.

    Parameters
    ----------
    iot_dim    : AdaptiveIoTClassifier hidden_dim  (default 128)
    crisis_dim : AdaptiveFusionClassifier fused dim (default 1024 = 512+512)
    proj_dim   : shared projection space (default 256)
    """
    def __init__(
        self,
        iot_dim:    int = 128,
        crisis_dim: int = 1024,
        proj_dim:   int = 256,
        dropout:    float = 0.2,
    ):
        super().__init__()

        self.iot_proj = nn.Sequential(
            nn.Linear(iot_dim, proj_dim * 2),
            nn.LayerNorm(proj_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        self.crisis_proj = nn.Sequential(
            nn.Linear(crisis_dim, proj_dim * 2),
            nn.LayerNorm(proj_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        self.cross_attn = CrossModalAttention(proj_dim)

        fused_dim = proj_dim * 3   # concat(iot_proj, attended, crisis_proj)
        self.shared_mlp = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.severity_head   = nn.Linear(256, 1)
        self.priority_head   = nn.Linear(256, 4)
        self.disaster_head   = nn.Linear(256, 5)
        # ── New heads ─────────────────────────────────────────────────────
        self.population_head = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
        self.resource_head   = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, len(RESOURCE_NAMES)), nn.Sigmoid(),
        )

    def forward(
        self,
        iot_embedding:    torch.Tensor,   # [B, iot_dim]
        crisis_embedding: torch.Tensor,   # [B, crisis_dim]
        _: Optional[torch.Tensor] = None,
    ) -> FusionOutput:

        iot_p    = self.iot_proj(iot_embedding)       # [B, proj_dim]
        crisis_p = self.crisis_proj(crisis_embedding) # [B, proj_dim]
        attended = self.cross_attn(iot_p, crisis_p)   # [B, proj_dim]

        fused = torch.cat([iot_p, attended, crisis_p], dim=1)  # [B, 768]
        rep   = self.shared_mlp(fused)                          # [B, 256]

        severity   = torch.sigmoid(self.severity_head(rep)).squeeze(-1)
        priority   = self.priority_head(rep)
        disaster   = self.disaster_head(rep)
        population = self.population_head(rep).squeeze(-1)
        resources  = self.resource_head(rep)

        return FusionOutput(
            severity_score    = severity,
            priority_logits   = priority,
            disaster_logits   = disaster,
            population_impact = population,
            resource_needs    = resources,
            fused_embedding   = rep,
        )
