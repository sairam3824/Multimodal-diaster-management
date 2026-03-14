"""
AdaptiveIoTClassifier — Comprehensive Evaluation
==================================================
Generates publication-quality figures and a JSON metrics file for
conference paper submission.

Outputs (saved to IOT/evaluation/):
  confusion_matrix.png        — normalised heat-map
  per_class_metrics.png       — precision / recall / F1 bar chart
  roc_curves.png              — one-vs-rest ROC + AUC per class
  precision_recall_curves.png — PR curves per class
  severity_regression.png     — predicted vs actual severity scatter
  class_distribution.png      — dataset class balance
  sensor_group_weights.png    — mean adaptive group weights per class
  metrics.json                — all numbers for the paper

Run from repo root:
    python IOT/evaluate_iot.py
"""

import os, sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    mean_absolute_error, mean_squared_error, r2_score,
    cohen_kappa_score, matthews_corrcoef,
)
from sklearn.preprocessing import label_binarize

from IOT.train_iot import (
    AdaptiveIoTClassifier, build_dataset,
    DISASTER_TYPES, GROUP_SIZE, HIDDEN_DIM,
)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
CKPT_PATH   = os.path.join(os.path.dirname(__file__), "models", "iot_model.pth")
OUT_DIR     = os.path.join(os.path.dirname(__file__), "evaluation")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE  = 512
CLASS_NAMES = DISASTER_TYPES                        # fire storm earthquake flood unknown
CLASS_COLORS = ["#e74c3c", "#3498db", "#8e44ad", "#27ae60", "#95a5a6"]
FONT_SIZE   = 13
plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "font.size":     FONT_SIZE,
    "axes.titlesize": FONT_SIZE + 2,
    "axes.labelsize": FONT_SIZE,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
sns.set_palette("deep")


# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
def load_model():
    ckpt  = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    cfg   = ckpt["config"]
    model = AdaptiveIoTClassifier(
        group_size       = cfg["group_size"],
        hidden_dim       = cfg["hidden_dim"],
        n_disaster_types = cfg["n_disaster_types"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    print(f"[Eval] Model loaded  val_acc={ckpt['val_acc']:.4f}  val_f1={ckpt['val_f1']:.4f}")
    return model, ckpt


# ─────────────────────────────────────────────
# Run inference on entire dataset
# ─────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, X, Yt, Ys):
    all_preds, all_probs, all_sev_pred = [], [], []
    all_w_weather, all_w_storm, all_w_seismic, all_w_hydro = [], [], [], []

    for i in range(0, len(X), BATCH_SIZE):
        xb  = X[i:i+BATCH_SIZE].to(DEVICE)
        logits, sev, _, cas, emb, attn = model(xb, return_attention=True)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()

        all_preds.append(preds)
        all_probs.append(probs)
        all_sev_pred.append(sev.cpu().numpy())
        all_w_weather.append(attn["weather_weight"].cpu().numpy())
        all_w_storm.append(attn["storm_weight"].cpu().numpy())
        all_w_seismic.append(attn["seismic_weight"].cpu().numpy())
        all_w_hydro.append(attn["hydro_weight"].cpu().numpy())

    return (
        np.concatenate(all_preds),
        np.concatenate(all_probs),
        np.concatenate(all_sev_pred),
        np.concatenate(all_w_weather),
        np.concatenate(all_w_storm),
        np.concatenate(all_w_seismic),
        np.concatenate(all_w_hydro),
    )


# ─────────────────────────────────────────────
# Figure 1: Confusion Matrix
# ─────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Confusion Matrix (Counts)", "Confusion Matrix (Normalised)"],
        ["d", ".2f"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, linecolor="white", ax=ax,
            annot_kws={"size": 11},
        )
        ax.set_title(title, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return cm, cm_norm


# ─────────────────────────────────────────────
# Figure 2: Per-class Precision / Recall / F1
# ─────────────────────────────────────────────
def plot_per_class_metrics(y_true, y_pred):
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    prec = [report[c]["precision"] for c in CLASS_NAMES]
    rec  = [report[c]["recall"]    for c in CLASS_NAMES]
    f1   = [report[c]["f1-score"]  for c in CLASS_NAMES]
    sup  = [report[c]["support"]   for c in CLASS_NAMES]

    x    = np.arange(len(CLASS_NAMES))
    w    = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))

    b1 = ax.bar(x - w,   prec, w, label="Precision", color="#3498db", edgecolor="white")
    b2 = ax.bar(x,        rec,  w, label="Recall",    color="#2ecc71", edgecolor="white")
    b3 = ax.bar(x + w,   f1,   w, label="F1-Score",  color="#e74c3c", edgecolor="white")

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n(n={s:,})" for n, s in zip(CLASS_NAMES, sup)])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision / Recall / F1-Score", fontweight="bold")
    ax.legend(frameon=False)
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "per_class_metrics.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return report


# ─────────────────────────────────────────────
# Figure 3: ROC Curves (One-vs-Rest)
# ─────────────────────────────────────────────
def plot_roc_curves(y_true, y_probs):
    y_bin = label_binarize(y_true, classes=list(range(len(CLASS_NAMES))))
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    roc_aucs = {}
    for i, (cls, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc     = auc(fpr, tpr)
        roc_aucs[cls] = roc_auc

        ax = axes[i]
        ax.plot(fpr, tpr, color=color, lw=2, label=f"AUC = {roc_auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.fill_between(fpr, tpr, alpha=0.08, color=color)
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC — {cls.upper()}", fontweight="bold")
        ax.legend(loc="lower right", frameon=False)

    # Macro-average ROC
    all_fpr = np.unique(np.concatenate([roc_curve(y_bin[:, i], y_probs[:, i])[0]
                                        for i in range(len(CLASS_NAMES))]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(CLASS_NAMES)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= len(CLASS_NAMES)
    macro_auc = auc(all_fpr, mean_tpr)

    ax = axes[5]
    ax.plot(all_fpr, mean_tpr, color="navy", lw=2.5,
            label=f"Macro-avg AUC = {macro_auc:.4f}")
    for i, (cls, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        ax.plot(fpr, tpr, lw=1, alpha=0.55, color=color,
                label=f"{cls} ({roc_aucs[cls]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC — All Classes (Macro-Avg)", fontweight="bold")
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    plt.suptitle("One-vs-Rest ROC Curves — AdaptiveIoTClassifier",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "roc_curves.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    roc_aucs["macro_avg"] = macro_auc
    return roc_aucs


# ─────────────────────────────────────────────
# Figure 4: Precision-Recall Curves
# ─────────────────────────────────────────────
def plot_pr_curves(y_true, y_probs):
    y_bin = label_binarize(y_true, classes=list(range(len(CLASS_NAMES))))
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    ap_scores = {}
    for i, (cls, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_probs[:, i])
        ap           = average_precision_score(y_bin[:, i], y_probs[:, i])
        ap_scores[cls] = ap

        ax = axes[i]
        ax.plot(rec, prec, color=color, lw=2, label=f"AP = {ap:.4f}")
        ax.fill_between(rec, prec, alpha=0.08, color=color)
        baseline = y_bin[:, i].mean()
        ax.axhline(baseline, color="gray", linestyle="--", lw=1,
                   label=f"Baseline = {baseline:.3f}")
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.05])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PR Curve — {cls.upper()}", fontweight="bold")
        ax.legend(loc="upper right", frameon=False)

    # Summary in last panel
    ax = axes[5]
    for i, (cls, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_probs[:, i])
        ax.plot(rec, prec, lw=2, color=color,
                label=f"{cls} (AP={ap_scores[cls]:.3f})")
    macro_ap = np.mean(list(ap_scores.values()))
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"PR — All Classes  mAP={macro_ap:.4f}", fontweight="bold")
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    plt.suptitle("Precision-Recall Curves — AdaptiveIoTClassifier",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "precision_recall_curves.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    ap_scores["macro_mAP"] = macro_ap
    return ap_scores


# ─────────────────────────────────────────────
# Figure 5: Severity Regression
# ─────────────────────────────────────────────
def plot_severity(y_true_sev, y_pred_sev, y_true_cls):
    mae  = mean_absolute_error(y_true_sev, y_pred_sev)
    rmse = np.sqrt(mean_squared_error(y_true_sev, y_pred_sev))
    r2   = r2_score(y_true_sev, y_pred_sev)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter coloured by class
    ax = axes[0]
    for i, (cls, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        mask = (y_true_cls == i)
        ax.scatter(y_true_sev[mask], y_pred_sev[mask],
                   alpha=0.25, s=8, color=color, label=cls, rasterized=True)
    lo, hi = 0.0, 1.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="Perfect fit")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("True Severity"); ax.set_ylabel("Predicted Severity")
    ax.set_title(f"Severity Regression  R²={r2:.4f}", fontweight="bold")
    ax.legend(frameon=False, markerscale=3, fontsize=9)

    # Residual distribution
    ax = axes[1]
    residuals = y_pred_sev - y_true_sev
    ax.hist(residuals, bins=80, color="#3498db", edgecolor="none", alpha=0.8)
    ax.axvline(0, color="red", lw=1.5, linestyle="--")
    ax.axvline(residuals.mean(), color="orange", lw=1.5,
               linestyle="--", label=f"Mean={residuals.mean():.4f}")
    ax.set_xlabel("Residual (Predicted − True)")
    ax.set_ylabel("Count")
    ax.set_title(f"Severity Residuals  MAE={mae:.4f}  RMSE={rmse:.4f}", fontweight="bold")
    ax.legend(frameon=False)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "severity_regression.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


# ─────────────────────────────────────────────
# Figure 6: Dataset Class Distribution
# ─────────────────────────────────────────────
def plot_class_distribution(y_true):
    counts = np.bincount(y_true, minlength=len(CLASS_NAMES))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Bar
    ax = axes[0]
    bars = ax.bar(CLASS_NAMES, counts, color=CLASS_COLORS, edgecolor="white", width=0.6)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{cnt:,}", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Sample Count")
    ax.set_title("Dataset Class Distribution", fontweight="bold")

    # Pie
    ax = axes[1]
    wedges, texts, autotexts = ax.pie(
        counts, labels=CLASS_NAMES, colors=CLASS_COLORS,
        autopct="%1.1f%%", startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontsize(10)
    ax.set_title("Class Proportion", fontweight="bold")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "class_distribution.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# Figure 7: Mean Sensor-Group Weights per Class
# ─────────────────────────────────────────────
def plot_sensor_weights(y_true, w_weather, w_storm, w_seismic, w_hydro):
    GROUP_NAMES  = ["Weather", "Storm", "Seismic", "Hydro"]
    GROUP_COLORS = ["#f39c12", "#3498db", "#8e44ad", "#27ae60"]

    # Mean weight per class per group
    data = np.zeros((len(CLASS_NAMES), 4))
    for ci in range(len(CLASS_NAMES)):
        mask = (y_true == ci)
        if mask.sum() == 0:
            continue
        data[ci, 0] = w_weather[mask].mean()
        data[ci, 1] = w_storm[mask].mean()
        data[ci, 2] = w_seismic[mask].mean()
        data[ci, 3] = w_hydro[mask].mean()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Grouped bar
    ax = axes[0]
    x  = np.arange(len(CLASS_NAMES))
    w  = 0.18
    for gi, (gname, gc) in enumerate(zip(GROUP_NAMES, GROUP_COLORS)):
        bars = ax.bar(x + gi * w - 1.5 * w, data[:, gi], w,
                      label=gname, color=gc, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("Mean Adaptive Weight")
    ax.set_title("Mean Sensor-Group Weights per Disaster Class", fontweight="bold")
    ax.legend(frameon=False)

    # Heatmap
    ax = axes[1]
    sns.heatmap(
        data.T, annot=True, fmt=".3f", cmap="YlOrRd",
        xticklabels=CLASS_NAMES, yticklabels=GROUP_NAMES,
        linewidths=0.5, linecolor="white", ax=ax,
        annot_kws={"size": 11},
    )
    ax.set_title("Sensor Group Weight Heatmap (Mean per Class)", fontweight="bold")
    ax.set_xlabel("Disaster Class")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "sensor_group_weights.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return {
        CLASS_NAMES[ci]: {
            "weather": float(data[ci, 0]),
            "storm":   float(data[ci, 1]),
            "seismic": float(data[ci, 2]),
            "hydro":   float(data[ci, 3]),
        }
        for ci in range(len(CLASS_NAMES))
    }


# ─────────────────────────────────────────────
# Figure 8: Summary Dashboard
# ─────────────────────────────────────────────
def plot_summary_dashboard(y_true, y_pred, y_probs, roc_aucs, report):
    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # -- 1. Normalised confusion matrix
    ax1   = fig.add_subplot(gs[0, 0])
    cm_n  = confusion_matrix(y_true, y_pred, normalize="true")
    sns.heatmap(cm_n, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, linecolor="white", ax=ax1, annot_kws={"size": 9},
                cbar=False)
    ax1.set_title("Normalised Confusion Matrix", fontweight="bold")
    ax1.set_xlabel("Predicted"); ax1.set_ylabel("True")
    ax1.tick_params(axis="x", rotation=30, labelsize=9)

    # -- 2. F1 per class
    ax2  = fig.add_subplot(gs[0, 1])
    f1s  = [report[c]["f1-score"] for c in CLASS_NAMES]
    bars = ax2.barh(CLASS_NAMES, f1s, color=CLASS_COLORS, edgecolor="white")
    for bar, v in zip(bars, f1s):
        ax2.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{v:.3f}", va="center", fontsize=10)
    ax2.set_xlim(0, 1.08)
    ax2.axvline(0.9, color="gray", linestyle="--", lw=1, alpha=0.7)
    ax2.set_title("F1-Score per Class", fontweight="bold")

    # -- 3. ROC AUC per class
    ax3   = fig.add_subplot(gs[0, 2])
    aucs_ = [roc_aucs[c] for c in CLASS_NAMES]
    bars  = ax3.barh(CLASS_NAMES, aucs_, color=CLASS_COLORS, edgecolor="white")
    for bar, v in zip(bars, aucs_):
        ax3.text(v + 0.001, bar.get_y() + bar.get_height() / 2,
                 f"{v:.4f}", va="center", fontsize=10)
    ax3.set_xlim(0.9, 1.01)
    ax3.set_title("ROC-AUC per Class (OvR)", fontweight="bold")

    # -- 4. Metrics table
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis("off")
    overall = report["weighted avg"]
    macro   = report["macro avg"]
    rows = [
        ["Metric",          "Weighted Avg", "Macro Avg"],
        ["Precision",       f"{overall['precision']:.4f}", f"{macro['precision']:.4f}"],
        ["Recall",          f"{overall['recall']:.4f}",    f"{macro['recall']:.4f}"],
        ["F1-Score",        f"{overall['f1-score']:.4f}",  f"{macro['f1-score']:.4f}"],
        ["Accuracy",        f"{accuracy_score(y_true, y_pred):.4f}", "—"],
        ["Macro ROC-AUC",   f"{roc_aucs['macro_avg']:.4f}", "—"],
        ["Cohen's κ",       f"{cohen_kappa_score(y_true, y_pred):.4f}", "—"],
        ["MCC",             f"{matthews_corrcoef(y_true, y_pred):.4f}", "—"],
    ]
    tbl = ax4.table(cellText=rows[1:], colLabels=rows[0],
                    cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#ecf0f1")
        cell.set_edgecolor("white")
    ax4.set_title("Overall Classification Metrics", fontweight="bold", pad=10)

    # -- 5. Class support
    ax5  = fig.add_subplot(gs[1, 1])
    sup  = [report[c]["support"] for c in CLASS_NAMES]
    bars = ax5.bar(CLASS_NAMES, sup, color=CLASS_COLORS, edgecolor="white")
    for bar, v in zip(bars, sup):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                 f"{v:,}", ha="center", fontsize=9)
    ax5.set_ylabel("Support (samples)")
    ax5.set_title("Class Support", fontweight="bold")
    ax5.tick_params(axis="x", rotation=25)

    # -- 6. Precision vs Recall scatter
    ax6  = fig.add_subplot(gs[1, 2])
    prec = [report[c]["precision"] for c in CLASS_NAMES]
    rec  = [report[c]["recall"]    for c in CLASS_NAMES]
    for ci, (cls, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        ax6.scatter(rec[ci], prec[ci], s=200, color=color,
                    label=cls, zorder=3, edgecolors="white", linewidths=1.5)
        ax6.annotate(cls, (rec[ci], prec[ci]), textcoords="offset points",
                     xytext=(6, 4), fontsize=9)
    ax6.set_xlim(0, 1.05); ax6.set_ylim(0, 1.05)
    ax6.set_xlabel("Recall"); ax6.set_ylabel("Precision")
    ax6.set_title("Precision vs Recall", fontweight="bold")
    ax6.axhline(0.9, color="gray", linestyle="--", lw=0.8, alpha=0.6)
    ax6.axvline(0.9, color="gray", linestyle="--", lw=0.8, alpha=0.6)

    fig.suptitle(
        "AdaptiveIoTClassifier — Evaluation Dashboard",
        fontsize=16, fontweight="bold", y=1.01
    )
    path = os.path.join(OUT_DIR, "evaluation_dashboard.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# Save JSON
# ─────────────────────────────────────────────
def save_json(y_true, y_pred, y_probs, y_sev_true, y_sev_pred,
              report, roc_aucs, ap_scores, sev_metrics,
              weight_stats, ckpt):
    cm     = confusion_matrix(y_true, y_pred).tolist()
    cm_n   = (confusion_matrix(y_true, y_pred).astype(float) /
              confusion_matrix(y_true, y_pred).sum(axis=1, keepdims=True)).tolist()

    metrics = {
        "model": {
            "name":        "AdaptiveIoTClassifier",
            "architecture": {
                "input_dim":        32,
                "sensor_groups":    4,
                "group_size":       8,
                "hidden_dim":       128,
                "attention_heads":  4,
                "output_classes":   5,
                "multi_task_heads": ["disaster_type", "severity", "risk_details"],
                "novel_components": [
                    "SensorConfidenceEstimator (per-group adaptive weighting)",
                    "CrossGroupMultiheadAttention (sensor interaction)",
                    "Joint multi-task learning (type + severity + risk)",
                    "Automatic disaster-type detection (no user input required)",
                ],
            },
            "training": {
                "epochs":           40,
                "batch_size":       256,
                "optimizer":        "AdamW",
                "lr":               1e-3,
                "scheduler":        "CosineAnnealingLR",
                "sampler":          "WeightedRandomSampler",
                "train_val_split":  "85/15",
                "device":           str(DEVICE),
            },
            "checkpoint": {
                "val_accuracy": float(ckpt["val_acc"]),
                "val_macro_f1": float(ckpt["val_f1"]),
            },
        },
        "dataset": {
            "sources": [
                {"name": "CA Wildfire (FIRE_START_DAY)",           "class": "fire",       "rows": 14988},
                {"name": "Historical Tropical Storm Tracks",        "class": "storm",      "rows": 59228},
                {"name": "Atlantic Storms",                         "class": "storm",      "rows": 22705},
                {"name": "Global Earthquakes",                      "class": "earthquake", "rows": 8394},
                {"name": "Iran Earthquakes",                        "class": "earthquake", "rows": 52268},
                {"name": "Sri Lanka Flood Risk (25k)",              "class": "flood",      "rows": 25000},
            ],
            "class_support":   {CLASS_NAMES[i]: int(np.sum(y_true == i))
                                 for i in range(len(CLASS_NAMES))},
            "total_samples":   int(len(y_true)),
        },
        "classification": {
            "accuracy":         float(accuracy_score(y_true, y_pred)),
            "cohen_kappa":      float(cohen_kappa_score(y_true, y_pred)),
            "matthews_corrcoef":float(matthews_corrcoef(y_true, y_pred)),
            "macro_avg": {
                "precision": float(report["macro avg"]["precision"]),
                "recall":    float(report["macro avg"]["recall"]),
                "f1_score":  float(report["macro avg"]["f1-score"]),
            },
            "weighted_avg": {
                "precision": float(report["weighted avg"]["precision"]),
                "recall":    float(report["weighted avg"]["recall"]),
                "f1_score":  float(report["weighted avg"]["f1-score"]),
            },
            "per_class": {
                cls: {
                    "precision": float(report[cls]["precision"]),
                    "recall":    float(report[cls]["recall"]),
                    "f1_score":  float(report[cls]["f1-score"]),
                    "support":   int(report[cls]["support"]),
                    "roc_auc":   float(roc_aucs.get(cls, 0.0)),
                    "avg_precision": float(ap_scores.get(cls, 0.0)),
                }
                for cls in CLASS_NAMES
            },
            "roc_auc_macro":    float(roc_aucs.get("macro_avg", 0.0)),
            "mean_avg_precision": float(ap_scores.get("macro_mAP", 0.0)),
            "confusion_matrix":          cm,
            "confusion_matrix_normalised": cm_n,
        },
        "severity_regression": {
            "mae":  float(sev_metrics["mae"]),
            "rmse": float(sev_metrics["rmse"]),
            "r2":   float(sev_metrics["r2"]),
        },
        "sensor_group_weights": weight_stats,
    }

    path = os.path.join(OUT_DIR, "metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {path}")
    return metrics


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  AdaptiveIoTClassifier — Comprehensive Evaluation")
    print("=" * 60)

    model, ckpt = load_model()

    print("\nRebuilding full dataset (same as training)…")
    X, Yt, Ys, _, _ = build_dataset()
    print(f"  {len(X):,} total samples")

    print("\nRunning inference…")
    preds, probs, sev_pred, w_wea, w_sto, w_sei, w_hyd = run_inference(
        model, X, Yt, Ys
    )
    y_true     = Yt.numpy()
    y_sev_true = Ys.numpy()

    print("\nGenerating figures…")
    cm, cm_n      = plot_confusion_matrix(y_true, preds)
    report        = plot_per_class_metrics(y_true, preds)
    roc_aucs      = plot_roc_curves(y_true, probs)
    ap_scores     = plot_pr_curves(y_true, probs)
    sev_metrics   = plot_severity(y_sev_true, sev_pred, y_true)
    plot_class_distribution(y_true)
    weight_stats  = plot_sensor_weights(y_true, w_wea, w_sto, w_sei, w_hyd)
    plot_summary_dashboard(y_true, preds, probs, roc_aucs, report)

    print("\nSaving metrics.json…")
    metrics = save_json(
        y_true, preds, probs, y_sev_true, sev_pred,
        report, roc_aucs, ap_scores, sev_metrics,
        weight_stats, ckpt
    )

    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Accuracy         : {metrics['classification']['accuracy']:.4f}")
    print(f"  Macro F1         : {metrics['classification']['macro_avg']['f1_score']:.4f}")
    print(f"  Weighted F1      : {metrics['classification']['weighted_avg']['f1_score']:.4f}")
    print(f"  Macro ROC-AUC    : {metrics['classification']['roc_auc_macro']:.4f}")
    print(f"  Mean Avg Prec.   : {metrics['classification']['mean_avg_precision']:.4f}")
    print(f"  Cohen's κ        : {metrics['classification']['cohen_kappa']:.4f}")
    print(f"  MCC              : {metrics['classification']['matthews_corrcoef']:.4f}")
    print(f"  Severity MAE     : {metrics['severity_regression']['mae']:.4f}")
    print(f"  Severity R²      : {metrics['severity_regression']['r2']:.4f}")
    print(f"\n  All outputs saved to: {OUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
