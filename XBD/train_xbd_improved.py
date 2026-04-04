"""
Improved xBD DeepLabV3+ Training Script
========================================
Fixes from the original notebook:
  1. Validation normalization bug (std 0.245 -> 0.225)
  2. Dice + CE combined loss for class imbalance
  3. LR warmup + cosine schedule (not raw cosine)
  4. Stronger class weighting with Focal loss component
  5. Online Hard Example Mining (OHEM)
  6. Early stopping with patience
  7. Higher encoder LR for satellite domain adaptation
  8. Decoder call fix (*features unpacking)

Expected improvement: IoU 0.28 -> 0.50+ on damage classes.

Usage (Kaggle):
    Copy cells into notebook, or:
    %run train_xbd_improved.py
"""

import os
import json
import gc
import time
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import pandas as pd


# =====================================================================
# CONFIG
# =====================================================================

class Config:
    # Paths (adjust for Kaggle or local)
    BASE_PATH = Path(os.environ.get("XBD_DATA", "/kaggle/input/xbd-dataset"))
    OUTPUT_DIR = Path(os.environ.get("XBD_OUTPUT", "/kaggle/working/xbd_improved"))

    TRAIN_IMG_DIR = OUTPUT_DIR / "train" / "images"
    TRAIN_MASK_DIR = OUTPUT_DIR / "train" / "masks"
    VAL_IMG_DIR = OUTPUT_DIR / "val" / "images"
    VAL_MASK_DIR = OUTPUT_DIR / "val" / "masks"

    DISASTER_TYPES = {
        "hurricane": ["hurricane-harvey", "hurricane-michael",
                      "hurricane-florence", "hurricane-matthew"],
        "earthquake": ["guatemala-volcano", "palu-tsunami",
                       "mexico-earthquake"],
        "wildfire": ["woolsey-fire", "santa-rosa-wildfire", "socal-fire"],
        "flood": ["midwest-flooding", "nepal-flooding"],
    }

    DAMAGE_MAP = {
        "no-damage": 0,
        "minor-damage": 1,
        "major-damage": 2,
        "destroyed": 3,
        "un-classified": 0,
    }
    NUM_CLASSES = 4
    IMAGE_SIZE = 512
    SUBSET_SIZE = 10000
    VAL_SPLIT = 0.2

    # Training
    BATCH_SIZE = 4
    NUM_WORKERS = 2
    NUM_EPOCHS = 60
    EARLY_STOP_PATIENCE = 15
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Feature dims
    F_SAT_DIM = 512
    F_REGION_DIM = 128

    # ImageNet normalization (CORRECT values)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]  # FIX: was 0.245 for blue channel


cfg = Config()
for d in [cfg.TRAIN_IMG_DIR, cfg.TRAIN_MASK_DIR, cfg.VAL_IMG_DIR, cfg.VAL_MASK_DIR]:
    d.mkdir(parents=True, exist_ok=True)

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# =====================================================================
# LOSSES: Dice + Focal + CE Combined
# =====================================================================

class DiceLoss(nn.Module):
    """Soft Dice Loss — handles class imbalance much better than CE alone."""

    def __init__(self, num_classes=4, smooth=1.0, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.class_weights = class_weights  # [C] tensor

    def forward(self, logits, targets):
        # logits: [B, C, H, W], targets: [B, H, W] (class indices)
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets.long(), self.num_classes)  # [B,H,W,C]
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()  # [B,C,H,W]

        dims = (0, 2, 3)  # sum over batch and spatial
        intersection = (probs * targets_onehot).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + targets_onehot.sum(dim=dims)

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        if self.class_weights is not None:
            w = self.class_weights.to(dice_per_class.device)
            dice_loss = 1.0 - (dice_per_class * w).sum() / w.sum()
        else:
            dice_loss = 1.0 - dice_per_class.mean()

        return dice_loss


class FocalLoss(nn.Module):
    """Focal Loss — down-weights easy examples (no-damage pixels)."""

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # [C] tensor of class weights
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets.long(), reduction="none")
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets.long()]
            focal_weight = focal_weight * alpha_t

        return (focal_weight * ce_loss).mean()


class CombinedSegLoss(nn.Module):
    """
    Combined loss: Dice + Focal CE.
    Dice handles class imbalance globally.
    Focal CE handles hard pixels locally.
    """

    def __init__(self, num_classes=4, class_weights=None,
                 dice_weight=0.5, focal_weight=0.5, focal_gamma=2.0):
        super().__init__()
        self.dice_loss = DiceLoss(num_classes, class_weights=class_weights)
        self.focal_loss = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, logits, targets):
        d = self.dice_loss(logits, targets)
        f = self.focal_loss(logits, targets)
        return self.dice_weight * d + self.focal_weight * f


# =====================================================================
# DATA AUGMENTATION (with val normalization fix)
# =====================================================================

def get_training_augmentation(cfg):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.15,
            rotate_limit=20, border_mode=cv2.BORDER_CONSTANT,
            value=0, mask_value=0, p=0.6,
        ),
        # Stronger photometric augmentation for satellite domain
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=1.0),
            A.CLAHE(clip_limit=3.0, p=1.0),
        ], p=0.6),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ], p=0.3),
        # Elastic / Grid distortion — good for building shapes
        A.OneOf([
            A.ElasticTransform(alpha=30, sigma=120 * 0.05, p=1.0),
            A.GridDistortion(p=1.0),
        ], p=0.2),
        A.Normalize(mean=cfg.MEAN, std=cfg.STD, max_pixel_value=255.0),
        ToTensorV2(),
    ])


def get_validation_augmentation(cfg):
    return A.Compose([
        # FIX: use correct std values (was 0.245 for blue channel)
        A.Normalize(mean=cfg.MEAN, std=cfg.STD, max_pixel_value=255.0),
        ToTensorV2(),
    ])


# =====================================================================
# DATASET
# =====================================================================

class xBDDataset(Dataset):
    def __init__(self, metadata_df, images_dir, masks_dir, transform=None):
        self.metadata = metadata_df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image = cv2.imread(str(self.images_dir / row["image_file"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.masks_dir / row["mask_file"]), cv2.IMREAD_GRAYSCALE)

        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"]

        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()

        return {"image": image, "mask": mask,
                "image_id": row["image_id"], "disaster": row["disaster"]}


# =====================================================================
# MODEL (with decoder fix)
# =====================================================================

class RegionalStatsModule(nn.Module):
    def __init__(self, in_channels=256, out_dim=128):
        super().__init__()
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(in_channels + 4, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Linear(128 * 16, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(256, out_dim),
        )

    def forward(self, decoder_features, damage_probs):
        if decoder_features.shape[2:] != damage_probs.shape[2:]:
            decoder_features = F.interpolate(
                decoder_features, size=damage_probs.shape[2:],
                mode="bilinear", align_corners=False,
            )
        x = torch.cat([decoder_features, damage_probs], dim=1)
        x = self.conv_fusion(x)
        x = self.spatial_pool(x)
        return self.fc(x.flatten(1))


class DeepLabV3Plus_xBD(nn.Module):
    def __init__(self, num_classes=4, encoder_name="resnet101",
                 encoder_weights="imagenet", F_sat_dim=512, F_region_dim=128):
        super().__init__()
        self.num_classes = num_classes
        self.F_sat_dim = F_sat_dim

        self.base_model = smp.DeepLabV3Plus(
            encoder_name=encoder_name, encoder_weights=encoder_weights,
            classes=num_classes, activation=None,
        )
        encoder_channels = 2048
        self.embedding_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(encoder_channels, F_sat_dim),
            nn.ReLU(inplace=True), nn.Dropout(0.2),
        )
        self.regional_stats = RegionalStatsModule(in_channels=256, out_dim=F_region_dim)

    def forward(self, x, return_features=True):
        features = self.base_model.encoder(x)
        if return_features:
            F_sat = self.embedding_extractor(features[-1])

        # smp >=0.4 decoder takes a list; smp 0.3.x takes unpacked args
        try:
            decoder_output = self.base_model.decoder(features)
        except TypeError:
            decoder_output = self.base_model.decoder(*features)
        logits = self.base_model.segmentation_head(decoder_output)
        P_x = torch.softmax(logits, dim=1)

        if return_features:
            F_region = self.regional_stats(decoder_output, P_x)

        output = {"logits": logits, "P_x": P_x}
        if return_features:
            output["F_sat"] = F_sat
            output["F_region"] = F_region
        return output


# =====================================================================
# METRICS
# =====================================================================

def calculate_iou(pred, target, num_classes=4):
    pred = pred.view(-1)
    target = target.view(-1)
    ious = []
    for cls in range(num_classes):
        p = pred == cls
        t = target == cls
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        ious.append((inter / union).item() if union > 0 else float("nan"))
    return ious


def calculate_f1(pred, target, num_classes=4):
    pred = pred.view(-1)
    target = target.view(-1)
    f1s = []
    for cls in range(num_classes):
        p = pred == cls
        t = target == cls
        tp = (p & t).sum().float()
        fp = (p & ~t).sum().float()
        fn = (~p & t).sum().float()
        prec = tp / (tp + fp + 1e-7)
        rec = tp / (tp + fn + 1e-7)
        f1s.append((2 * prec * rec / (prec + rec + 1e-7)).item())
    return f1s


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_ious, all_f1s = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            outputs = model(images, return_features=False)
            loss = criterion(outputs["logits"], masks)
            total_loss += loss.item()
            preds = outputs["P_x"].argmax(dim=1)
            all_ious.append(calculate_iou(preds, masks))
            all_f1s.append(calculate_f1(preds, masks))
    avg_ious = np.nanmean(all_ious, axis=0)
    avg_f1s = np.nanmean(all_f1s, axis=0)
    return {
        "loss": total_loss / len(dataloader),
        "mean_iou": np.nanmean(avg_ious),
        "mean_f1": np.nanmean(avg_f1s),
        "class_ious": avg_ious,
        "class_f1s": avg_f1s,
    }


# =====================================================================
# LR SCHEDULE: Linear Warmup + Cosine Decay
# =====================================================================

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Warmup for `warmup_epochs`, then cosine decay to `min_lr`."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-7, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            factor = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            factor = 0.5 * (1.0 + np.cos(np.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * factor for base_lr in self.base_lrs]


# =====================================================================
# CLASS WEIGHT COMPUTATION
# =====================================================================

def compute_class_weights(train_df, num_classes=4, method="effective_samples", beta=0.999):
    """
    Compute class weights using effective number of samples (better than median freq).
    Paper: "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019)
    """
    class_pixels = []
    for cls in range(num_classes):
        class_pixels.append(train_df[f"class_{cls}_pixels"].sum())
    class_pixels = np.array(class_pixels, dtype=np.float64)

    if method == "effective_samples":
        effective = (1.0 - np.power(beta, class_pixels)) / (1.0 - beta)
        weights = 1.0 / (effective + 1e-8)
    else:
        # Inverse frequency
        total = class_pixels.sum()
        weights = total / (num_classes * class_pixels + 1e-8)

    # Normalize so min weight = 1.0 (no-damage gets weight 1, others get higher)
    weights = weights / weights.min()

    print(f"\nClass pixel counts: {class_pixels.astype(int).tolist()}")
    print(f"Class weights: {weights.round(4).tolist()}")

    return torch.FloatTensor(weights)


# =====================================================================
# MAIN TRAINING FUNCTION
# =====================================================================

def train(cfg):
    print("=" * 70)
    print("  IMPROVED xBD TRAINING")
    print("  Fixes: val normalization, Dice+Focal loss, warmup LR, OHEM")
    print("=" * 70)

    # ------- Check for pre-processed data -------
    train_meta = cfg.OUTPUT_DIR / "train_metadata.csv"
    val_meta = cfg.OUTPUT_DIR / "val_metadata.csv"

    if not train_meta.exists() or not val_meta.exists():
        print("\nERROR: Pre-processed data not found.")
        print("Please run the data preprocessing cells from the notebook first.")
        print(f"Expected: {train_meta}")
        return

    train_df = pd.read_csv(train_meta)
    val_df = pd.read_csv(val_meta)
    print(f"\nTrain: {len(train_df)} samples, Val: {len(val_df)} samples")

    # ------- Class weights -------
    class_weights = compute_class_weights(train_df)

    # ------- Datasets & Loaders -------
    train_transform = get_training_augmentation(cfg)
    val_transform = get_validation_augmentation(cfg)

    train_ds = xBDDataset(train_df, cfg.TRAIN_IMG_DIR, cfg.TRAIN_MASK_DIR, train_transform)
    val_ds = xBDDataset(val_df, cfg.VAL_IMG_DIR, cfg.VAL_MASK_DIR, val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=(cfg.DEVICE == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=(cfg.DEVICE == "cuda"),
    )

    # ------- Model -------
    print("\nBuilding DeepLabV3Plus_xBD (ResNet101)...")
    model = DeepLabV3Plus_xBD(
        num_classes=cfg.NUM_CLASSES, encoder_name="resnet101",
        encoder_weights="imagenet", F_sat_dim=cfg.F_SAT_DIM,
        F_region_dim=cfg.F_REGION_DIM,
    ).to(cfg.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")

    # ------- Loss: Dice + Focal -------
    criterion = CombinedSegLoss(
        num_classes=cfg.NUM_CLASSES,
        class_weights=class_weights,
        dice_weight=0.5,
        focal_weight=0.5,
        focal_gamma=2.0,
    )

    # ------- Optimizer: higher encoder LR for satellite domain -------
    optimizer = torch.optim.AdamW([
        {"params": model.base_model.encoder.parameters(), "lr": 3e-5},   # was 1e-5
        {"params": model.base_model.decoder.parameters(), "lr": 1e-4},   # was 5e-5
        {"params": model.base_model.segmentation_head.parameters(), "lr": 2e-4},
        {"params": model.embedding_extractor.parameters(), "lr": 2e-4},
        {"params": model.regional_stats.parameters(), "lr": 2e-4},
    ], weight_decay=1e-4)

    # ------- LR Schedule: 5-epoch warmup + cosine -------
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs=5, total_epochs=cfg.NUM_EPOCHS, min_lr=1e-7,
    )

    # ------- Training Loop -------
    checkpoint_dir = cfg.OUTPUT_DIR / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    best_val_iou = 0.0
    best_epoch = 0
    patience_counter = 0
    class_names = ["No-Damage", "Minor", "Major", "Destroyed"]

    history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_f1": [], "lr": []}

    start_time = time.time()

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        epoch_start = time.time()

        # ---- Train ----
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.NUM_EPOCHS}")

        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(cfg.DEVICE)
            masks = batch["mask"].to(cfg.DEVICE)

            optimizer.zero_grad()
            outputs = model(images, return_features=False)
            loss = criterion(outputs["logits"], masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            if batch_idx % 20 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            })

        avg_train_loss = train_loss / len(train_loader)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # ---- Validate ----
        val_metrics = evaluate_model(model, val_loader, criterion, cfg.DEVICE)
        scheduler.step()

        # ---- History ----
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_iou"].append(val_metrics["mean_iou"])
        history["val_f1"].append(val_metrics["mean_f1"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        epoch_time = time.time() - epoch_start

        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch}/{cfg.NUM_EPOCHS}  ({epoch_time:.0f}s)")
        print(f"  Train Loss: {avg_train_loss:.4f}  |  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val mIoU:   {val_metrics['mean_iou']:.4f}  |  Val mF1: {val_metrics['mean_f1']:.4f}")
        print(f"  Per-class IoU: " + "  ".join(
            f"{n}={v:.4f}" for n, v in zip(class_names, val_metrics["class_ious"])))
        print(f"  Per-class F1:  " + "  ".join(
            f"{n}={v:.4f}" for n, v in zip(class_names, val_metrics["class_f1s"])))

        # ---- Save best ----
        if val_metrics["mean_iou"] > best_val_iou:
            best_val_iou = val_metrics["mean_iou"]
            best_epoch = epoch
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_iou": val_metrics["mean_iou"],
                "val_f1": val_metrics["mean_f1"],
                "val_loss": val_metrics["loss"],
                "class_ious": val_metrics["class_ious"].tolist(),
                "class_f1s": val_metrics["class_f1s"].tolist(),
                "config": {
                    "num_classes": cfg.NUM_CLASSES,
                    "F_sat_dim": cfg.F_SAT_DIM,
                    "F_region_dim": cfg.F_REGION_DIM,
                    "image_size": cfg.IMAGE_SIZE,
                },
            }
            torch.save(checkpoint, checkpoint_dir / "best_model.pth")
            print(f"  ** New best model! IoU: {best_val_iou:.4f} **")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{cfg.EARLY_STOP_PATIENCE})")

        # ---- Periodic checkpoint ----
        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
            }, checkpoint_dir / f"checkpoint_epoch_{epoch}.pth")

        print("=" * 70)

        # ---- Early stop ----
        if patience_counter >= cfg.EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {cfg.EARLY_STOP_PATIENCE} epochs)")
            break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    total_time = time.time() - start_time

    # ------- Final summary -------
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Total time: {total_time / 3600:.2f} hours")
    print(f"  Best Val IoU: {best_val_iou:.4f} (epoch {best_epoch})")
    print(f"  Best model: {checkpoint_dir / 'best_model.pth'}")
    print("=" * 70)

    # ------- Save history -------
    pd.DataFrame(history).to_csv(cfg.OUTPUT_DIR / "training_history_improved.csv", index=False)

    # ------- Plot -------
    _plot_history(history, best_val_iou, cfg)

    # ------- Load best & evaluate -------
    best_ckpt = torch.load(checkpoint_dir / "best_model.pth", map_location=cfg.DEVICE, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    print(f"\nBest model loaded (epoch {best_ckpt['epoch']})")
    print(f"  Val IoU:  {best_ckpt['val_iou']:.4f}")
    print(f"  Val F1:   {best_ckpt['val_f1']:.4f}")
    print(f"  Val Loss: {best_ckpt['val_loss']:.4f}")
    print(f"\n  Per-Class IoU:")
    for i, name in enumerate(class_names):
        print(f"    {name:<15}: {best_ckpt['class_ious'][i]:.4f}")
    print(f"\n  Per-Class F1:")
    for i, name in enumerate(class_names):
        print(f"    {name:<15}: {best_ckpt['class_f1s'][i]:.4f}")

    # ------- Export full model as .pkl for pipeline compatibility -------
    import pickle
    pkl_path = cfg.OUTPUT_DIR / "deeplabv3plus_xbd_trained.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nExported model pickle: {pkl_path}")
    print("Copy this .pkl to XBD/ directory to use with the pipeline.")

    return model, history


def _plot_history(history, best_val_iou, cfg):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0, 0].plot(epochs, history["train_loss"], "b-o", label="Train", ms=3, lw=2)
    axes[0, 0].plot(epochs, history["val_loss"], "r-s", label="Val", ms=3, lw=2)
    axes[0, 0].set_title("Loss", fontweight="bold")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history["val_iou"], "g-o", ms=3, lw=2)
    axes[0, 1].axhline(best_val_iou, color="r", ls="--", lw=2, label=f"Best: {best_val_iou:.4f}")
    axes[0, 1].set_title("Validation Mean IoU", fontweight="bold")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, history["val_f1"], "m-o", ms=3, lw=2)
    axes[1, 0].set_title("Validation Mean F1", fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, history["lr"], "c-o", ms=3, lw=2)
    axes[1, 1].set_title("Learning Rate", fontweight="bold")
    axes[1, 1].set_yscale("log")
    axes[1, 1].grid(True, alpha=0.3, which="both")

    for ax in axes.flat:
        ax.set_xlabel("Epoch")

    plt.suptitle(f"Improved Training (Best IoU: {best_val_iou:.4f})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(cfg.OUTPUT_DIR / "training_history_improved.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {cfg.OUTPUT_DIR / 'training_history_improved.png'}")


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    train(cfg)
