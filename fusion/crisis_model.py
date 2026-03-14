"""
CrisisMMD Multimodal Model
===========================
A lightweight multimodal model for three CrisisMMD tasks:
  - Task 1: Informative vs Not-Informative
  - Task 2: Humanitarian Category
  - Task 3: Damage Severity

Architecture:
  Text  → DistilBERT → 768-d text embedding
  Image → ResNet18   → 512-d image embedding
  Both  → concat → MLP head → class logits
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import DistilBertTokenizer, DistilBertModel
from PIL import Image
import pandas as pd
import numpy as np

CRISIS_ROOT = os.path.join(os.path.dirname(__file__), "..", "crisis", "CrisisMMD_v2.0")
SPLITS_DIR  = os.path.join(CRISIS_ROOT, "crisismmd_datasplit_all", "crisismmd_datasplit_all")

TASK_CONFIGS = {
    "informative": {
        "train": "task_informative_text_img_train.tsv",
        "dev":   "task_informative_text_img_dev.tsv",
        "test":  "task_informative_text_img_test.tsv",
        "label_col": "label",
        "labels": ["informative", "not_informative"],
    },
    "humanitarian": {
        "train": "task_humanitarian_text_img_train.tsv",
        "dev":   "task_humanitarian_text_img_dev.tsv",
        "test":  "task_humanitarian_text_img_test.tsv",
        "label_col": "label",
        "labels": [
            "affected_individuals", "infrastructure_and_utility_damage",
            "not_humanitarian", "other_relevant_information",
            "rescue_volunteering_or_donation_effort",
            "vehicle_damage", "missing_or_found_people"
        ],
    },
    "damage": {
        "train": "task_damage_text_img_train.tsv",
        "dev":   "task_damage_text_img_dev.tsv",
        "test":  "task_damage_text_img_test.tsv",
        "label_col": "label",
        "labels": ["severe_damage", "mild_damage", "little_or_no_damage"],
    },
}


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class CrisisDataset(Dataset):
    IMG_TRANSFORMS = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, tsv_path: str, task: str, max_length: int = 128):
        cfg = TASK_CONFIGS[task]
        self.crisis_root = CRISIS_ROOT
        self.max_length  = max_length

        df = pd.read_csv(tsv_path, sep="\t")
        df = df.dropna(subset=[cfg["label_col"]])

        self.labels    = cfg["labels"]
        self.label2idx = {l: i for i, l in enumerate(self.labels)}

        # Keep only known labels
        df = df[df[cfg["label_col"]].isin(self.label2idx)]
        self.df = df.reset_index(drop=True)

        self.tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ── Text ──
        text = str(row.get("tweet_text", ""))
        enc  = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # ── Image ──
        img_rel_path = str(row.get("image", ""))
        img_path     = os.path.join(self.crisis_root, img_rel_path)
        try:
            img = Image.open(img_path).convert("RGB")
            img = self.IMG_TRANSFORMS(img)
        except Exception:
            img = torch.zeros(3, 224, 224)

        # ── Label ──
        label = self.label2idx[row[TASK_CONFIGS[self._task]["label_col"]]]

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "image":          img,
            "label":          torch.tensor(label, dtype=torch.long),
        }


def make_dataset(task: str, split: str = "train") -> CrisisDataset:
    cfg    = TASK_CONFIGS[task]
    tsv    = os.path.join(SPLITS_DIR, cfg[split])
    ds     = CrisisDataset(tsv, task)
    ds._task = task
    return ds


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
class TextEncoder(nn.Module):
    """DistilBERT [CLS] token as 768-d text embedding."""
    def __init__(self, freeze_base: bool = False):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        if freeze_base:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]   # [B, 768]


class ImageEncoder(nn.Module):
    """ResNet-18 feature extractor → 512-d image embedding."""
    def __init__(self, freeze_base: bool = False):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # drop fc
        if freeze_base:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        feat = self.backbone(x)   # [B, 512, 1, 1]
        return feat.squeeze(-1).squeeze(-1)   # [B, 512]


class CrisisMultimodalModel(nn.Module):
    """
    Multimodal model: text (768) + image (512) → concat (1280) → MLP → logits.
    Returns text_emb and image_emb as well so the fusion layer can use them.
    """
    def __init__(self, num_classes: int, dropout: float = 0.3,
                 freeze_text: bool = False, freeze_image: bool = False):
        super().__init__()
        self.text_enc  = TextEncoder(freeze_base=freeze_text)
        self.image_enc = ImageEncoder(freeze_base=freeze_image)

        self.classifier = nn.Sequential(
            nn.Linear(768 + 512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    @property
    def embed_dim(self):
        return 768 + 512   # 1280

    def forward(self, input_ids, attention_mask, image):
        text_emb  = self.text_enc(input_ids, attention_mask)   # [B, 768]
        image_emb = self.image_enc(image)                       # [B, 512]
        fused     = torch.cat([text_emb, image_emb], dim=1)    # [B, 1280]
        logits    = self.classifier(fused)
        return logits, text_emb, image_emb
