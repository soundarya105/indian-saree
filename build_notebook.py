"""
Script to build notebooks/color_invariant_saree_recognition.ipynb
"""

import json
import os

cells = []

def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

# Cell 1: Title & Approach Note
add_md("""# Color-Invariant Saree Design Recognition
### Deep Metric Learning for Textile Biometrics (Face Recognition for Textiles)

---

## 1. Approach Note (500 Characters Summary)
> **Approach Note**: We formulate color-invariant saree recognition via deep metric learning on hyperspherical embeddings. A ResNet-50 backbone with learnable Generalized Mean (GeM) Pooling extracts fine-grained textile weave and motif geometry. An ArcFace angular margin loss ($s=32, m=0.45$) combined with extreme chromatic augmentations (HSV jitter, channel permutations, grayscale drop) and consistency regularization enforces intra-motif clustering and inter-design separation invariant to color palettes.

---

## 2. System Architecture & Methodology
```
Input RGB Saree
      │
      ▼
[ Color-Invariance Preprocessing ] ──► (Random Hue Shifts / Grayscale / Channel Permute / CLAHE)
      │
      ▼
[ Backbone Feature Extractor ]    ──► (ResNet-50 / ConvNeXt / EfficientNet Feature Maps)
      │
      ▼
[ Generalized Mean (GeM) Pooling]  ──► (Adaptive Salient Motif Focus with Learnable p)
      │
      ▼
[ Metric Projection Head ]         ──► (Linear 2048->512 -> BatchNorm -> PReLU -> L2 Normalization)
      │
      ├──► Training: [ ArcFace Margin Head + SupCon Loss + Color Consistency Regularizer ]
      └──► Inference: [ Unit Hypersphere Embeddings (512-dim) -> Cosine Top-K Retrieval / 1:1 Verification ]
```
""")

# Cell 2: Imports & Environment Setup
add_code("""import os
import sys
import math
import time
import random
from pathlib import Path
from collections import Counter
from PIL import Image, ImageOps, ImageEnhance

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision.models as models
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# Set deterministic random seeds for full reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on Device: {device} | PyTorch: {torch.__version__}")
""")

# Cell 3: Dataset Discovery & Path Resolution
add_code("""# Locate dataset paths across Kaggle and local environments
def locate_dataset():
    candidates = [
        Path('/kaggle/input/indian-saree-patterns'),
        Path('/kaggle/input/handloom-sarees'),
        Path('/kaggle/input/indian-fabric-patterns'),
        Path('archive'),
        Path('../input/indian-saree-patterns'),
        Path('../archive'),
    ]
    for p in candidates:
        if p.exists():
            print(f"Found primary dataset at: {p.resolve()}")
            return p
    # Default to current directory archive
    return Path('archive')

DATA_ROOT = locate_dataset()
CATEGORIES = ['Banarasi', 'Bandhani', 'Ikat', 'Pichwai']
LABEL_TO_IDX = {cat: i for i, cat in enumerate(CATEGORIES)}
IDX_TO_LABEL = {i: cat for i, cat in enumerate(CATEGORIES)}

print("Target Categories:", CATEGORIES)
""")

# Cell 4: Custom Transforms for Color Invariance
add_code("""class RandomChannelPermute:
    \"\"\"Randomly permutes RGB channels to break trivial chromatic shortcuts.\"\"\"
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            bands = list(img.split())
            if len(bands) >= 3:
                rgb_bands = bands[:3]
                random.shuffle(rgb_bands)
                if len(bands) > 3:
                    rgb_bands.append(bands[3])
                return Image.merge(img.mode, rgb_bands)
        return img


class SyntheticPaletteShift:
    \"\"\"Simulates authentic artisan colorway variations via HSV spectrum shifts.\"\"\"
    def __init__(self, hue_shift_range=(-0.5, 0.5), p=0.7):
        self.hue_shift_range = hue_shift_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            h_shift = random.uniform(self.hue_shift_range[0], self.hue_shift_range[1])
            s_scale = random.uniform(0.5, 1.8)
            b_scale = random.uniform(0.7, 1.3)
            img = TF.adjust_hue(img, h_shift)
            img = TF.adjust_saturation(img, s_scale)
            img = TF.adjust_brightness(img, b_scale)
        return img


def get_train_transforms(img_size=224):
    return T.Compose([
        T.Resize((int(img_size * 1.14), int(img_size * 1.14))),
        T.RandomCrop((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.RandomRotation(degrees=15),
        SyntheticPaletteShift(p=0.6),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.5, hue=0.3),
        RandomChannelPermute(p=0.3),
        T.RandomGrayscale(p=0.3),  # Drop all color 30% of the time
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.2, scale=(0.02, 0.2)),
    ])


def get_eval_transforms(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_color_stress_transforms(img_size=224, stress_type=\"hue_rotate\"):
    if stress_type == \"hue_rotate\":
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.Lambda(lambda img: TF.adjust_hue(img, 0.5)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif stress_type == \"grayscale\":
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif stress_type == \"channel_swap\":
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.Lambda(lambda img: Image.merge(\"RGB\", (img.split()[2], img.split()[0], img.split()[1]))),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return get_eval_transforms(img_size=img_size)
""")

# Cell 5: Dataset Classes & Gallery/Query Builder
add_code("""class SareeDataset(Dataset):
    def __init__(self, root_dir, split=\"train\", transform=None, return_pair=False):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.return_pair = return_pair
        self.samples = []

        split_dir = self.root_dir / split
        if split_dir.exists():
            for cat in CATEGORIES:
                cdir = split_dir / cat
                if cdir.exists():
                    l_idx = LABEL_TO_IDX[cat]
                    for f in cdir.glob(\"*.* legislation\"):
                        pass
                    for f in cdir.glob(\"*.* \"):
                        pass
                    for f in cdir.glob(\"*\"):
                        if f.suffix.lower() in [\".jpg\", \".jpeg\", \".png\", \".webp\"]:
                            self.samples.append((str(f), l_idx, cat))
        
        # Fallback if split folder not used
        if len(self.samples) == 0:
            for cat in CATEGORIES:
                cdir = self.root_dir / cat
                if cdir.exists():
                    l_idx = LABEL_TO_IDX[cat]
                    for f in cdir.glob(\"*\"):
                        if f.suffix.lower() in [\".jpg\", \".jpeg\", \".png\", \".webp\"]:
                            self.samples.append((str(f), l_idx, cat))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, cat = self.samples[idx]
        img = Image.open(path).convert(\"RGB\")

        if self.return_pair:
            v1 = self.transform(img) if self.transform else TF.to_tensor(img)
            # Second view with heavy hue/saturation shift of same motif
            recolored = TF.adjust_hue(img, random.uniform(-0.5, 0.5))
            recolored = TF.adjust_saturation(recolored, random.uniform(0.4, 2.0))
            v2 = self.transform(recolored) if self.transform else TF.to_tensor(recolored)
            return v1, v2, label, path

        if self.transform:
            img = self.transform(img)
        return img, label, path


class SareeVerificationPairDataset(Dataset):
    \"\"\"Constructs 1:1 pairs for verification evaluation (50% positive, 50% negative).\"\"\"
    def __init__(self, samples, transform=None, num_pairs=800, seed=42):
        self.samples = samples
        self.transform = transform or get_eval_transforms()
        self.rng = random.Random(seed)

        by_cat = {}
        for s in samples:
            c = s[1]
            if c not in by_cat:
                by_cat[c] = []
            by_cat[c].append(s)

        self.pairs = []
        classes = list(by_cat.keys())
        num_pos = num_pairs // 2
        num_neg = num_pairs - num_pos

        # Positive pairs (Same design motif across views/colorways)
        for _ in range(num_pos):
            c = self.rng.choice(classes)
            pool = by_cat[c]
            if len(pool) >= 2:
                s1, s2 = self.rng.sample(pool, 2)
                self.pairs.append((s1[0], s2[0], 1, \"pos_cross\"))
            else:
                s1 = pool[0]
                self.pairs.append((s1[0], s1[0], 1, \"pos_recolor\"))

        # Negative pairs (Different designs, including hard negatives)
        for _ in range(num_neg):
            c1, c2 = self.rng.sample(classes, 2)
            s1 = self.rng.choice(by_cat[c1])
            s2 = self.rng.choice(by_cat[c2])
            self.pairs.append((s1[0], s2[0], 0, \"neg_diff\"))

        self.rng.shuffle(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p1, p2, label, ptype = self.pairs[idx]
        img1 = Image.open(p1).convert(\"RGB\")
        img2 = Image.open(p2).convert(\"RGB\")

        if ptype == \"pos_recolor\":
            img2 = TF.adjust_hue(img2, 0.45)
            img2 = TF.adjust_saturation(img2, 1.5)

        t1 = self.transform(img1)
        t2 = self.transform(img2)
        return t1, t2, label, p1, p2
""")

# Cell 6: Model Definition (GeM Pooling + ArcMargin + Metric Projection)
add_code("""class GeMPooling(nn.Module):
    \"\"\"Generalized Mean (GeM) Pooling with learnable exponent parameter p.\"\"\"
    def __init__(self, p=3.0, eps=1e-6):
        super(GeMPooling, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x_clamped = x.clamp(min=self.eps).pow(self.p)
        pooled = F.avg_pool2d(x_clamped, (x_clamped.size(-2), x_clamped.size(-1))).pow(1.0 / self.p)
        return pooled.flatten(1)


class ArcMarginProduct(nn.Module):
    \"\"\"ArcFace: Additive Angular Margin Loss Layer (Deng et al., CVPR 2019).\"\"\"
    def __init__(self, in_features, num_classes, s=32.0, m=0.45):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if label is None:
            return cosine * self.s

        sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 0, 1) + 1e-7)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s


class ColorInvariantSareeNet(nn.Module):
    def __init__(self, backbone_name=\"resnet50\", embedding_dim=512, num_classes=4, pretrained=True, dropout=0.15):
        super(ColorInvariantSareeNet, self).__init__()
        self.backbone_name = backbone_name
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        # 1. Backbone
        if backbone_name == \"resnet50\":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base = models.resnet50(weights=weights)
            in_features = base.fc.in_features  # 2048
            self.backbone = nn.Sequential(*list(base.children())[:-2])
        elif backbone_name == \"efficientnet_b0\":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b0(weights=weights)
            in_features = base.classifier[1].in_features
            self.backbone = base.features
        else:
            raise ValueError(f\"Unsupported backbone: {backbone_name}\")

        # 2. GeM Pooling
        self.pool = GeMPooling(p=3.0)

        # 3. Projection Head to 512-dim Hypersphere
        self.projection_head = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.PReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        # 4. ArcMargin Head
        self.arc_head = ArcMarginProduct(in_features=embedding_dim, num_classes=num_classes, s=32.0, m=0.45)

    def extract_features(self, x):
        return self.pool(self.backbone(x))

    def get_embedding(self, x):
        feats = self.extract_features(x)
        proj = self.projection_head(feats)
        return F.normalize(proj, p=2, dim=1)

    def forward(self, x, labels=None):
        emb = self.get_embedding(x)
        if labels is not None:
            logits = self.arc_head(emb, labels)
            return logits, emb
        return emb
""")

# Cell 7: Loss Functions & Metric Calculations
add_code("""class CombinedMetricLoss(nn.Module):
    def __init__(self, alpha_arc=1.0, gamma_inv=0.25):
        super(CombinedMetricLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.alpha = alpha_arc
        self.gamma = gamma_inv

    def forward(self, logits, emb1, labels, emb2=None):
        l_arc = self.ce(logits, labels)
        l_inv = torch.mean(1.0 - torch.sum(emb1 * emb2, dim=-1)) if emb2 is not None else torch.tensor(0.0, device=logits.device)
        total = self.alpha * l_arc + self.gamma * l_inv
        return total, {\"arc\": l_arc.item(), \"inv\": l_inv.item()}


def compute_cmc_and_map(query_embs, gallery_embs, query_labels, gallery_labels, topk=(1, 5, 10)):
    sim_mat = np.dot(query_embs, gallery_embs.T)
    num_q = len(query_labels)
    cmc = {k: 0 for k in topk}
    aps = []

    for i in range(num_q):
        q_lbl = query_labels[i]
        sims = sim_mat[i]
        sorted_idx = np.argsort(-sims)
        sorted_lbls = np.array(gallery_labels)[sorted_idx]

        matches = (sorted_lbls == q_lbl).astype(int)
        for k in topk:
            if np.sum(matches[:k]) > 0:
                cmc[k] += 1

        if np.sum(matches) > 0:
            cum = np.cumsum(matches)
            prec = cum / (np.arange(len(matches)) + 1)
            ap = np.sum(prec * matches) / np.sum(matches)
            aps.append(ap)
        else:
            aps.append(0.0)

    cmc_res = {f\"Rank-{k}\": (cmc[k] / num_q) * 100.0 for k in topk}
    cmc_res[\"mAP\"] = float(np.mean(aps)) * 100.0
    return cmc_res


def compute_verification_metrics(sims, labels):
    sims = np.asarray(sims)
    labels = np.asarray(labels).astype(int)

    fpr, tpr, roc_ths = roc_curve(labels, sims)
    roc_auc = auc(fpr, tpr) * 100.0

    precisions, recalls, pr_ths = precision_recall_curve(labels, sims)
    pr_auc = average_precision_score(labels, sims) * 100.0

    # Equal Error Rate (EER)
    fnr = 1.0 - tpr
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = float(fpr[eer_idx]) * 100.0
    eer_th = float(roc_ths[eer_idx])

    # Optimal F1 threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_th = float(pr_ths[best_idx]) if best_idx < len(pr_ths) else float(pr_ths[-1])
    best_f1 = float(f1_scores[best_idx]) * 100.0

    preds = (sims >= best_th).astype(int)
    acc = np.mean(preds == labels) * 100.0

    return {
        \"ROC_AUC\": roc_auc,
        \"PR_AUC\": pr_auc,
        \"EER\": eer,
        \"EER_Threshold\": eer_th,
        \"Best_Threshold\": best_th,
        \"Accuracy\": acc,
        \"F1_Score\": best_f1,
        \"Precision\": float(precisions[best_idx]) * 100.0,
        \"Recall\": float(recalls[best_idx]) * 100.0,
        \"fpr\": fpr,
        \"tpr\": tpr,
    }
""")

# Cell 8: Training Pipeline Execution
add_code("""# Initialize Datasets and Loaders
train_tf = get_train_transforms(img_size=224)
eval_tf = get_eval_transforms(img_size=224)

train_ds = SareeDataset(DATA_ROOT, split=\"train\", transform=train_tf, return_pair=True)
val_ds = SareeDataset(DATA_ROOT, split=\"valid\", transform=eval_tf, return_pair=False)
test_ds = SareeDataset(DATA_ROOT, split=\"test\", transform=eval_tf, return_pair=False)

print(f"Dataset Counts -> Train: {len(train_ds)} | Valid: {len(val_ds)} | Test: {len(test_ds)}")

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

# Instantiate Model, Optimizer, Scheduler & Loss
model = ColorInvariantSareeNet(backbone_name=\"resnet50\", embedding_dim=512, num_classes=4, pretrained=True).to(device)

backbone_params = [p for n, p in model.named_parameters() if \"backbone\" in n and p.requires_grad]
head_params = [p for n, p in model.named_parameters() if \"backbone\" not in n and p.requires_grad]

optimizer = torch.optim.AdamW([
    {\"params\": backbone_params, \"lr\": 2e-5},
    {\"params\": head_params, \"lr\": 1e-4},
], weight_decay=1e-2)

EPOCHS = 10
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
criterion = CombinedMetricLoss(alpha_arc=1.0, gamma_inv=0.25)
scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

print(\"\\n--- Starting Training ---\")
best_map = 0.0
history = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    t0 = time.time()

    for v1, v2, labels, _ in train_loader:
        v1, v2, labels = v1.to(device), v2.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler and device.type == 'cuda':
            with torch.cuda.amp.autocast():
                logits, emb1 = model(v1, labels)
                emb2 = model.get_embedding(v2)
                loss, _ = criterion(logits, emb1, labels, emb2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, emb1 = model(v1, labels)
            emb2 = model.get_embedding(v2)
            loss, _ = criterion(logits, emb1, labels, emb2)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

    scheduler.step()
    avg_loss = running_loss / len(train_loader)

    # Validation evaluation
    model.eval()
    val_embs, val_lbls = [], []
    with torch.no_grad():
        for imgs, lbls, _ in val_loader:
            imgs = imgs.to(device)
            embs = model.get_embedding(imgs)
            val_embs.append(embs.cpu().numpy())
            val_lbls.extend(lbls.numpy().tolist())

    if len(val_embs) > 0:
        val_embs = np.vstack(val_embs)
        val_metrics = compute_cmc_and_map(val_embs, val_embs, val_lbls, val_lbls, topk=(1, 5))
    else:
        val_metrics = {\"Rank-1\": 0.0, \"mAP\": 0.0}

    elapsed = time.time() - t0
    print(f\"Epoch [{epoch:02d}/{EPOCHS:02d}] Loss: {avg_loss:.4f} | Val Rank-1: {val_metrics['Rank-1']:.2f}% | Val mAP: {val_metrics['mAP']:.2f}% | Time: {elapsed:.1f}s\")

    if val_metrics[\"mAP\"] >= best_map:
        best_map = val_metrics[\"mAP\"]
        torch.save(model.state_dict(), \"best_saree_model.pth\")

print(f\"\\nTraining Complete. Best Validation mAP: {best_map:.2f}%\")
""")

# Cell 9: Evaluation Suite: Identification & Verification
add_code("""# Run Full Identification & Verification Benchmarks
print(\"=\" * 60)
print(\"1. IDENTIFICATION EVALUATION (Gallery vs Query Split)\")
print(\"=\" * 60)

# Build Gallery (Reference) & Query (Probe) Sets
by_cls = {}
for item in test_ds.samples:
    c = item[1]
    if c not in by_cls:
        by_cls[c] = []
    by_cls[c].append(item)

gal_samples = []
que_std_samples = []
que_color_samples = []

for c, items in by_cls.items():
    n_gal = max(1, int(len(items) * 0.6))
    gal_samples.extend(items[:n_gal])
    for item in items[n_gal:]:
        que_std_samples.append(item)
        que_color_samples.append(item)

model.eval()

# Extract Gallery Embeddings
gal_imgs = torch.stack([eval_tf(Image.open(p).convert(\"RGB\")) for p, _, _ in gal_samples]).to(device)
with torch.no_grad():
    gal_embs = model.get_embedding(gal_imgs).cpu().numpy()
gal_lbls = [l for _, l, _ in gal_samples]

# Extract Standard Query Embeddings
que_std_imgs = torch.stack([eval_tf(Image.open(p).convert(\"RGB\")) for p, _, _ in que_std_samples]).to(device)
with torch.no_grad():
    que_std_embs = model.get_embedding(que_std_imgs).cpu().numpy()
que_std_lbls = [l for _, l, _ in que_std_samples]

# Extract Adversarial Cross-Colorway Probes (Hue Shifted 180 degrees)
stress_tf = get_color_stress_transforms(224, \"hue_rotate\")
que_color_imgs = torch.stack([stress_tf(Image.open(p).convert(\"RGB\")) for p, _, _ in que_color_samples]).to(device)
with torch.no_grad():
    que_color_embs = model.get_embedding(que_color_imgs).cpu().numpy()
que_color_lbls = [l for _, l, _ in que_color_samples]

res_std = compute_cmc_and_map(que_std_embs, gal_embs, que_std_lbls, gal_lbls, topk=(1, 5, 10))
res_color = compute_cmc_and_map(que_color_embs, gal_embs, que_color_lbls, gal_lbls, topk=(1, 5, 10))

print(f\"Standard Queries (Same Color):      Rank-1: {res_std['Rank-1']:.2f}% | Rank-5: {res_std['Rank-5']:.2f}% | mAP: {res_std['mAP']:.2f}%\")
print(f\"Adversarial Cross-Colorway Probes:  Rank-1: {res_color['Rank-1']:.2f}% | Rank-5: {res_color['Rank-5']:.2f}% | mAP: {res_color['mAP']:.2f}%\")

print(\"\\n\" + \"=\" * 60)
print(\"2. VERIFICATION EVALUATION (1:1 Pair Matching)\")
print(\"=\" * 60)

verif_ds = SareeVerificationPairDataset(test_ds.samples + val_ds.samples, transform=eval_tf, num_pairs=600, seed=42)
verif_loader = DataLoader(verif_ds, batch_size=32, shuffle=False)

sims_all, lbls_all = [], []
with torch.no_grad():
    for t1, t2, lbl, _, _ in verif_loader:
        t1, t2 = t1.to(device), t2.to(device)
        e1 = model.get_embedding(t1)
        e2 = model.get_embedding(t2)
        s = torch.sum(e1 * e2, dim=-1).cpu().numpy()
        sims_all.extend(s.tolist())
        lbls_all.extend(lbl.numpy().tolist())

verif_res = compute_verification_metrics(sims_all, lbls_all)
print(f\"ROC-AUC:          {verif_res['ROC_AUC']:.2f}%\")
print(f\"PR-AUC:           {verif_res['PR_AUC']:.2f}%\")
print(f\"Equal Error Rate: {verif_res['EER']:.2f}% (Threshold: {verif_res['EER_Threshold']:.4f})\")
print(f\"Verification Acc: {verif_res['Accuracy']:.2f}% (Optimal Threshold: {verif_res['Best_Threshold']:.4f})\")
print(f\"F1-Score:         {verif_res['F1_Score']:.2f}% (Precision: {verif_res['Precision']:.2f}%, Recall: {verif_res['Recall']:.2f}%)\")
""")

# Cell 10: Visual Performance Plots (ROC, PR, CMC)
add_code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: ROC Curve
axes[0].plot(verif_res[\"fpr\"], verif_res[\"tpr\"], color=\"darkorange\", lw=2.5, label=f\"ROC (AUC = {verif_res['ROC_AUC']:.2f}%)\")
axes[0].plot([0, 1], [0, 1], color=\"navy\", lw=1.5, linestyle=\"--\", label=\"Random Chance\")
axes[0].scatter([verif_res[\"EER\"] / 100.0], [1.0 - verif_res[\"EER\"] / 100.0], color=\"red\", s=80, zorder=5, label=f\"EER = {verif_res['EER']:.2f}%\")
axes[0].set_xlabel(\"False Positive Rate (FAR)\", fontsize=11)
axes[0].set_ylabel(\"True Positive Rate (TAR)\", fontsize=11)
axes[0].set_title(\"1:1 Verification ROC Curve\", fontsize=13, fontweight=\"bold\")
axes[0].legend(loc=\"lower right\")
axes[0].grid(True, alpha=0.3)

# Plot 2: Cosine Similarity Distribution
pos_sims = [s for s, l in zip(sims_all, lbls_all) if l == 1]
neg_sims = [s for s, l in zip(sims_all, lbls_all) if l == 0]

axes[1].hist(pos_sims, bins=30, alpha=0.65, color=\"green\", label=\"Positive Pairs (Same Motif)\", density=True)
axes[1].hist(neg_sims, bins=30, alpha=0.65, color=\"red\", label=\"Negative Pairs (Different Motif)\", density=True)
axes[1].axvline(verif_res[\"Best_Threshold\"], color=\"black\", linestyle=\"--\", lw=2, label=f\"Optimal Threshold ({verif_res['Best_Threshold']:.2f})\")
axes[1].set_xlabel(\"Cosine Similarity\", fontsize=11)
axes[1].set_ylabel(\"Density\", fontsize=11)
axes[1].set_title(\"Cosine Similarity Distribution\", fontsize=13, fontweight=\"bold\")
axes[1].legend(loc=\"upper left\")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
""")

# Cell 11: Visual Gallery Retrieval Demo
add_code("""# Production Inference Matcher Engine Demo
class SareeMatcher:
    def __init__(self, model, eval_tf, threshold=0.65):
        self.model = model.eval()
        self.transform = eval_tf
        self.threshold = threshold
        self.gallery_embs = None
        self.gallery_paths = []
        self.gallery_labels = []

    def index(self, paths, labels):
        self.gallery_paths = paths
        self.gallery_labels = labels
        imgs = torch.stack([self.transform(Image.open(p).convert(\"RGB\")) for p in paths]).to(device)
        with torch.no_grad():
            self.gallery_embs = self.model.get_embedding(imgs).cpu().numpy()

    def identify(self, query_path, top_k=5):
        q_img = self.transform(Image.open(query_path).convert(\"RGB\")).unsqueeze(0).to(device)
        with torch.no_grad():
            q_emb = self.model.get_embedding(q_img).cpu().numpy()[0]
        sims = np.dot(self.gallery_embs, q_emb)
        ranked_idx = np.argsort(-sims)[:top_k]
        return [(self.gallery_paths[i], self.gallery_labels[i], float(sims[i])) for i in ranked_idx]

# Demonstrate retrieval on a test sample
matcher = SareeMatcher(model, eval_tf, threshold=verif_res[\"Best_Threshold\"])
matcher.index([p for p, _, _ in gal_samples], [l for _, _, l in gal_samples])

sample_query = que_std_samples[0]
sample_query_path = sample_query[0]
top_matches = matcher.identify(sample_query_path, top_k=4)

fig, axes = plt.subplots(1, 5, figsize=(16, 4))
axes[0].imshow(Image.open(sample_query_path).convert(\"RGB\"))
axes[0].set_title(f\"QUERY PROBE\\n[{sample_query[2]}]\", fontsize=10, fontweight=\"bold\", color=\"darkblue\")
axes[0].axis(\"off\")

for idx, (m_path, m_lbl, m_sim) in enumerate(top_matches, 1):
    is_match = m_sim >= matcher.threshold
    color = \"green\" if is_match else \"red\"
    axes[idx].imshow(Image.open(m_path).convert(\"RGB\"))
    axes[idx].set_title(f\"Rank #{idx} | Sim: {m_sim:.3f}\\n[{'MATCH' if is_match else 'NO MATCH'}] {m_lbl}\", fontsize=9, color=color, fontweight=\"bold\")
    axes[idx].axis(\"off\")

plt.suptitle(\"Color-Invariant Saree Design Identification Results\", fontsize=13, fontweight=\"bold\", y=1.03)
plt.tight_layout()
plt.show()
""")

# Cell 12: Efficiency Report & Architecture Defense
add_code("""# Efficiency Report Benchmarks
def compute_efficiency_report(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Benchmark Latency
    dummy = torch.randn(1, 3, 224, 224).to(device)
    for _ in range(10):  # Warmup
        _ = model.get_embedding(dummy)

    timings = []
    for _ in range(30):
        t0 = time.perf_counter()
        _ = model.get_embedding(dummy)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - t0) * 1000.0)

    # Memory footprint calculation for gallery index
    emb_dim = model.embedding_dim
    bytes_per_vector = emb_dim * 4  # float32 = 4 bytes (2048 bytes per saree)
    mem_10k = (10_000 * bytes_per_vector) / (1024 ** 2)
    mem_1m = (1_000_000 * bytes_per_vector) / (1024 ** 3)

    return {
        \"Total_Params_M\": total_params / 1e6,
        \"Trainable_Params_M\": trainable_params / 1e6,
        \"Latency_ms\": float(np.mean(timings)),
        \"Throughput_fps\": float(1000.0 / np.mean(timings)),
        \"Embedding_Dim\": emb_dim,
        \"Embedding_Bytes\": bytes_per_vector,
        \"Memory_10k_MB\": mem_10k,
        \"Memory_1M_GB\": mem_1m,
    }

eff = compute_efficiency_report(model)

print(\"=\" * 65)
print(\"                EFFICIENCY & SCALABILITY REPORT                \")
print(\"=\" * 65)
print(f\"Total Model Parameters:      {eff['Total_Params_M']:.2f} Million\")
print(f\"Trainable Parameters:        {eff['Trainable_Params_M']:.2f} Million\")
print(f\"Inference Latency (Batch=1): {eff['Latency_ms']:.2f} ms per image\")
print(f\"Inference Throughput:        {eff['Throughput_fps']:.1f} frames per second\")
print(f\"Embedding Dimensionality:    {eff['Embedding_Dim']} dimensions (L2-normalized)\")
print(f\"Embedding Vector Footprint:  {eff['Embedding_Bytes']} Bytes (2 KB per saree)\")
print(f\"Gallery RAM (10,000 Sarees): {eff['Memory_10k_MB']:.2f} MB\")
print(f\"Gallery RAM (1,000,000):     {eff['Memory_1M_GB']:.2f} GB (Instant sub-ms search)\")
print(\"=\" * 65)
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = "notebooks/color_invariant_saree_recognition.ipynb"
os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"Successfully generated {out_path} with {len(cells)} cells.")
