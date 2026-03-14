import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

# ── canvas ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 14))
ax.set_xlim(0, 20)
ax.set_ylim(0, 14)
ax.axis('off')
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# ── palette ──────────────────────────────────────────────────────────────────
C_BLUE      = '#4A90D9'
C_ORANGE    = '#E8954A'
C_CRISIS_BG = '#EBF5FF'
C_IOT_BG    = '#FFF3E8'
C_PURPLE    = '#7B2D8B'
C_GREEN     = '#27AE60'
C_RED       = '#C0392B'
C_ARROW     = '#555555'
C_WHITE     = 'white'
C_DARK      = '#1A1A2E'
C_GRAY_BOX  = '#F0F4F8'
C_GRAY_LINE = '#AAAAAA'

# ── helpers ──────────────────────────────────────────────────────────────────
def rbox(ax, x, y, w, h, text, facecolor, textcolor='white',
         fontsize=8.5, bold=False, alpha=1.0, lw=1.2,
         edgecolor=None, radius=0.25, va='center', linespacing=1.4):
    """Draw a rounded rectangle with centred text."""
    ec = edgecolor if edgecolor else facecolor
    # darken edge slightly
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=f"round,pad=0.0,rounding_size={radius}",
                         linewidth=lw, edgecolor=ec, facecolor=facecolor,
                         alpha=alpha, zorder=3)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va=va, fontsize=fontsize,
            color=textcolor, fontweight=weight, zorder=4,
            linespacing=linespacing,
            multialignment='center')

def shadow_rbox(ax, x, y, w, h, text, facecolor, textcolor='white',
                fontsize=8.5, bold=False, edgecolor=None, radius=0.25):
    """Rounded box with subtle drop shadow."""
    # shadow
    shadow = FancyBboxPatch((x - w/2 + 0.06, y - h/2 - 0.06), w, h,
                             boxstyle=f"round,pad=0.0,rounding_size={radius}",
                             linewidth=0, facecolor='#BBBBBB', alpha=0.4, zorder=2)
    ax.add_patch(shadow)
    rbox(ax, x, y, w, h, text, facecolor, textcolor, fontsize, bold,
         edgecolor=edgecolor, radius=radius)

def arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.6, style='->', mutation=12):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=f'->', color=color,
                                lw=lw, mutation_scale=mutation),
                zorder=5)

def section_bg(ax, x, y, w, h, facecolor, label, labelcolor, lw=1.2):
    """Rounded section background with label at top."""
    ec = facecolor  # use same but slightly darker
    bg = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.0,rounding_size=0.35",
                         linewidth=lw, edgecolor='#AAAAAA',
                         facecolor=facecolor, alpha=0.55, zorder=1)
    ax.add_patch(bg)
    ax.text(x + w/2, y + h - 0.18, label,
            ha='center', va='top', fontsize=8, fontweight='bold',
            color=labelcolor, zorder=4, style='italic')

# ════════════════════════════════════════════════════════════════════════════
# TITLE
# ════════════════════════════════════════════════════════════════════════════
ax.text(10, 13.6,
        'Adaptive IoT × Social Media Fusion Architecture for Real-Time Disaster Response',
        ha='center', va='center', fontsize=13, fontweight='bold',
        color=C_DARK, zorder=6)
ax.text(10, 13.25,
        'Unified Disaster Response System: IoT × Social Media Adaptive Fusion',
        ha='center', va='center', fontsize=9.5, color='#555577',
        style='italic', zorder=6)

# thin separator
ax.plot([0.5, 19.5], [13.05, 13.05], color=C_GRAY_LINE, lw=0.8, zorder=6)

# ════════════════════════════════════════════════════════════════════════════
# LAYER 1 — INPUTS  (y ≈ 12.25)
# ════════════════════════════════════════════════════════════════════════════
L1_Y = 12.2
IH   = 0.75   # input box height
IW   = 3.2    # input box width

# layer label
ax.text(0.45, L1_Y + 0.05, 'LAYER 1\nINPUTS', ha='center', va='center',
        fontsize=7, color='#888888', fontweight='bold', linespacing=1.3, zorder=6)

# box positions (x centre)
X_IMG  = 4.2
X_TEXT = 8.5
X_IOT  = 14.8

shadow_rbox(ax, X_IMG,  L1_Y, IW, IH,
            'Crisis Image\n(RGB Photo)',
            C_BLUE, fontsize=8.5, bold=False)

shadow_rbox(ax, X_TEXT, L1_Y, IW, IH,
            'Tweet / Social Media\nText',
            C_BLUE, fontsize=8.5)

shadow_rbox(ax, X_IOT,  L1_Y, 4.6, IH,
            'IoT Sensor Readings  (Optional — auto-detected)\n'
            'Weather · Storm · Seismic · Hydro',
            C_ORANGE, fontsize=8.2)

# ════════════════════════════════════════════════════════════════════════════
# LAYER 2a — CRISIS MODEL  (left panel, x: 1.0 … 11.0, y: 7.6 … 11.75)
# ════════════════════════════════════════════════════════════════════════════
CX0, CY0, CW, CH = 1.0, 7.5, 10.2, 4.2
section_bg(ax, CX0, CY0, CW, CH, C_CRISIS_BG,
           'CRISIS MODEL  (AdaptiveFusionClassifier)', '#1A5276', lw=1.4)

# ── encoders side by side ──
ENC_Y   = 10.9
ENC_W   = 3.0
ENC_H   = 0.80
X_VIS   = 3.8
X_TXT_E = 8.2

shadow_rbox(ax, X_VIS,   ENC_Y, ENC_W, ENC_H,
            'BLIP ViT\nVision Encoder\n[1, 768]',
            '#2471A3', fontsize=8.2)

shadow_rbox(ax, X_TXT_E, ENC_Y, ENC_W, ENC_H,
            'XLM-RoBERTa\nText Encoder\n[1, 768]',
            '#2471A3', fontsize=8.2)

# ── confidence estimators ──
CONF_Y = 9.9
shadow_rbox(ax, 6.0, CONF_Y, 6.0, 0.70,
            'Confidence Estimators   (Vision Weight  |  Text Weight)',
            '#1A6090', fontsize=8.2)

# ── cross-modal attention ──
ATTN_Y = 9.05
shadow_rbox(ax, 6.0, ATTN_Y, 6.0, 0.70,
            'Bidirectional Cross-Modal Attention\n(8 heads, 512-dim)',
            '#154360', fontsize=8.2)

# ── crisis classifier ──
CLS_Y = 8.1
shadow_rbox(ax, 6.0, CLS_Y, 8.5, 0.82,
            'Crisis Classifier  ·  5-class Softmax\n'
            '[affected  |  infra_damage  |  not_humanitarian  |  other_relevant  |  rescue_effort]',
            '#1B2631', fontsize=8.0)

# ── crisis model arrows ──
# img → vis enc
arrow(ax, X_IMG,  L1_Y - IH/2,  X_VIS,   ENC_Y + ENC_H/2)
# tweet → txt enc
arrow(ax, X_TEXT, L1_Y - IH/2,  X_TXT_E, ENC_Y + ENC_H/2)
# vis enc → conf
arrow(ax, X_VIS,   ENC_Y - ENC_H/2, 4.8, CONF_Y + 0.35)
# txt enc → conf
arrow(ax, X_TXT_E, ENC_Y - ENC_H/2, 7.2, CONF_Y + 0.35)
# conf → attn
arrow(ax, 6.0, CONF_Y - 0.35, 6.0, ATTN_Y + 0.35)
# attn → cls
arrow(ax, 6.0, ATTN_Y - 0.35, 6.0, CLS_Y + 0.41)

# ════════════════════════════════════════════════════════════════════════════
# LAYER 2b — IoT MODEL  (right panel, x: 11.8 … 19.2, y: 7.6 … 11.75)
# ════════════════════════════════════════════════════════════════════════════
IX0, IY0, IW2, IH2 = 11.6, 7.5, 7.8, 4.2
section_bg(ax, IX0, IY0, IW2, IH2, C_IOT_BG,
           'IoT MODEL  (AdaptiveIoTClassifier)', '#7D4800', lw=1.4)

IX_C = 15.5   # centre x for IoT column

# ── sensor encoders ──
SE_Y = 10.9
shadow_rbox(ax, IX_C, SE_Y, 6.6, 0.80,
            '4 Sensor Group Encoders\n(8 features × 4 groups = 32-dim input)',
            '#B7770D', fontsize=8.2)

# ── sensor confidence ──
SC_Y = 9.9
shadow_rbox(ax, IX_C, SC_Y, 6.6, 0.70,
            'SensorConfidenceEstimator\n(Adaptive Per-Group Weighting)',
            '#9C640C', fontsize=8.2)

# ── cross-group attention ──
CGA_Y = 9.05
shadow_rbox(ax, IX_C, CGA_Y, 6.6, 0.70,
            'CrossGroup MultiHead Attention\n(4 heads, 128-dim)',
            '#784212', fontsize=8.2)

# ── multi-task heads label ──
MTH_Y = 8.38
ax.text(IX_C, MTH_Y, 'Multi-Task Heads', ha='center', va='center',
        fontsize=8, fontweight='bold', color='#5D4037', zorder=5)

# 4 sub-boxes for multi-task heads
sub_texts = ['Disaster Type\n(5-class)',
             'Severity Score\n(regression)',
             'Risk Details\n(fire·storm·eq·flood)',
             'Casualty Risk\n(regression)']
sub_xs = [12.35, 13.85, 16.05, 18.05]
sub_w  = 1.35
sub_h  = 0.72
for tx, txt in zip(sub_xs, sub_texts):
    shadow_rbox(ax, tx, 7.88, sub_w, sub_h, txt,
                '#AF601A', fontsize=7.2)
    arrow(ax, IX_C, MTH_Y - 0.12, tx, 7.88 + sub_h/2, lw=1.2, mutation=9)

# IoT model arrows
arrow(ax, X_IOT, L1_Y - IH/2, IX_C, SE_Y + 0.40)
arrow(ax, IX_C, SE_Y - 0.40, IX_C, SC_Y + 0.35)
arrow(ax, IX_C, SC_Y - 0.35, IX_C, CGA_Y + 0.35)
arrow(ax, IX_C, CGA_Y - 0.35, IX_C, MTH_Y + 0.12)

# ════════════════════════════════════════════════════════════════════════════
# LAYER 3 — FUSION LAYER  (y ≈ 6.0)
# ════════════════════════════════════════════════════════════════════════════
FL_Y = 5.85
FL_H = 1.35
shadow_rbox(ax, 10.0, FL_Y, 18.0, FL_H,
            'FUSION LAYER   (CrossModalAttention + MLP)\n\n'
            'IoT Embedding [128]  ×  Crisis Embedding [1024]\n'
            'Cross-Modal Attention  →  Shared MLP [256-dim]\n\n'
            'Priority (4-class)  |  Disaster Type (5-class)  |  Fused Severity  |  '
            'Population Impact  |  Resource Needs (water · medical · rescue · shelter)',
            C_PURPLE, C_WHITE, fontsize=8.3, bold=False, radius=0.3)

# layer label
ax.text(0.45, FL_Y, 'LAYER 3\nFUSION', ha='center', va='center',
        fontsize=7, color='#888888', fontweight='bold', linespacing=1.3, zorder=6)

# arrows into fusion
# crisis cls → fusion
arrow(ax, 6.0, CLS_Y - 0.41, 6.0, FL_Y + FL_H/2)
# IoT multi-task → fusion (from centre bottom of IoT panel)
arrow(ax, IX_C, 7.5, IX_C, FL_Y + FL_H/2)

# ════════════════════════════════════════════════════════════════════════════
# LAYER 4 — XAI  (y ≈ 4.4)
# ════════════════════════════════════════════════════════════════════════════
XAI_Y  = 4.35
XAI_H  = 0.95
XAI_W  = 7.8

ax.text(0.45, XAI_Y, 'LAYER 4\nXAI', ha='center', va='center',
        fontsize=7, color='#888888', fontweight='bold', linespacing=1.3, zorder=6)

shadow_rbox(ax, 5.5, XAI_Y, XAI_W, XAI_H,
            'Grad-CAM\n(BLIP ViT last layer)\nVisual Attention Heatmap  (14×14 → 224×224)',
            C_GREEN, fontsize=8.2, radius=0.28)

shadow_rbox(ax, 14.5, XAI_Y, XAI_W, XAI_H,
            'GPT-4o Responder Briefing\n(SITUATION  ·  KEY RISKS\nRECOMMENDED ACTIONS  ·  WHY THIS ALERT)',
            C_GREEN, fontsize=8.2, radius=0.28)

# fusion → XAI arrows
arrow(ax, 7.0,  FL_Y - FL_H/2,  5.5,  XAI_Y + XAI_H/2)
arrow(ax, 13.0, FL_Y - FL_H/2, 14.5,  XAI_Y + XAI_H/2)

# ════════════════════════════════════════════════════════════════════════════
# LAYER 5 — OUTPUT  (y ≈ 2.9)
# ════════════════════════════════════════════════════════════════════════════
OUT_Y = 2.95
OUT_H = 1.30

ax.text(0.45, OUT_Y, 'LAYER 5\nOUTPUT', ha='center', va='center',
        fontsize=7, color='#888888', fontweight='bold', linespacing=1.3, zorder=6)

shadow_rbox(ax, 10.0, OUT_Y, 18.0, OUT_H,
            'UNIFIED DISASTER ASSESSMENT\n\n'
            'Alert Level (GREEN / YELLOW / ORANGE / RED)   |   Confirmed Disaster Type   |   '
            'Response Priority   |   Fused Severity Score\n'
            'IoT Risk Scores   |   Crisis Confidence   |   Population Impact   |   '
            'Resource Allocation',
            C_RED, C_WHITE, fontsize=8.5, bold=False, radius=0.3)

# XAI → output arrows
arrow(ax,  5.5, XAI_Y - XAI_H/2,  5.5, OUT_Y + OUT_H/2)
arrow(ax, 14.5, XAI_Y - XAI_H/2, 14.5, OUT_Y + OUT_H/2)

# ════════════════════════════════════════════════════════════════════════════
# LAYER LABELS (left side) — already added inline above
# ════════════════════════════════════════════════════════════════════════════
ax.text(0.45, L1_Y + 0.05, 'LAYER 1\nINPUTS', ha='center', va='center',
        fontsize=7, color='#888888', fontweight='bold', linespacing=1.3, zorder=6)
ax.text(0.45, (CY0 + CH/2 + 0.4), 'LAYER 2\nENCODERS', ha='center', va='center',
        fontsize=7, color='#888888', fontweight='bold', linespacing=1.3, zorder=6)

# ════════════════════════════════════════════════════════════════════════════
# LEGEND / decorative footer
# ════════════════════════════════════════════════════════════════════════════
ax.plot([0.5, 19.5], [2.05, 2.05], color=C_GRAY_LINE, lw=0.8, zorder=6)

legend_items = [
    (C_BLUE,   'Crisis Inputs'),
    (C_ORANGE, 'IoT Inputs'),
    ('#2471A3', 'Crisis Encoders'),
    ('#B7770D', 'IoT Encoders'),
    (C_PURPLE, 'Fusion Layer'),
    (C_GREEN,  'XAI / Explainability'),
    (C_RED,    'Final Output'),
]
lx = 1.2
for col, lbl in legend_items:
    patch = mpatches.Patch(facecolor=col, edgecolor='white', linewidth=0.5)
    ax.add_patch(FancyBboxPatch((lx - 0.22, 1.58), 0.44, 0.28,
                                boxstyle="round,pad=0.0,rounding_size=0.06",
                                facecolor=col, edgecolor='white', lw=0.5, zorder=6))
    ax.text(lx, 1.32, lbl, ha='center', va='top', fontsize=6.5,
            color='#444444', zorder=6)
    lx += 2.6

ax.text(10, 0.75,
        'Architecture designed for IEEE/ACM conference proceedings  —  '
        'Unified Disaster Response System v1.0',
        ha='center', va='center', fontsize=7, color='#AAAAAA',
        style='italic', zorder=6)

# ════════════════════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════════════════════
plt.tight_layout(pad=0.3)
plt.savefig('/Users/sairammaruri/Desktop/sdp/architecture.png',
            dpi=200, bbox_inches='tight', facecolor='white',
            metadata={'Title': 'Unified Disaster Response System Architecture'})
plt.close()
print("Saved: /Users/sairammaruri/Desktop/sdp/architecture.png")
