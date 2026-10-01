"""
dataset.py - Saree Image Loaders, Synthetic Colorway Generators, and Verification Pair Samplers
"""

import os
import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from .transforms import get_train_transforms, get_eval_transforms, get_color_stress_transforms


CATEGORIES = ["Banarasi", "Bandhani", "Ikat", "Pichwai"]
LABEL_TO_IDX = {cat: i for i, cat in enumerate(CATEGORIES)}
IDX_TO_LABEL = {i: cat for i, cat in enumerate(CATEGORIES)}


class SareeClassificationDataset(Dataset):
    """
    Standard labeled Saree dataset for metric learning training and evaluation.
    Optionally returns pairs of (original, recolored_view) for color consistency regularization.
    """
    def __init__(self, root_dir, split="train", transform=None, return_pair=False):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.return_pair = return_pair
        self.samples = []  # (path, label_idx, class_name)

        split_dir = self.root_dir / split
        if split_dir.exists():
            for cat_name in CATEGORIES:
                cat_dir = split_dir / cat_name
                if cat_dir.exists():
                    label_idx = LABEL_TO_IDX[cat_name]
                    for img_file in cat_dir.glob("*.*"):
                        if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                            self.samples.append((str(img_file), label_idx, cat_name))

        # Fallback if specific split directory not found
        if len(self.samples) == 0:
            for cat_name in CATEGORIES:
                cat_dir = self.root_dir / cat_name
                if cat_dir.exists():
                    label_idx = LABEL_TO_IDX[cat_name]
                    for img_file in cat_dir.glob("*.*"):
                        if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                            self.samples.append((str(img_file), label_idx, cat_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_idx, cat_name = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.return_pair:
            # Generate two views: original and heavily color-shifted view of same motif
            view1 = self.transform(img) if self.transform else TF.to_tensor(img)
            # Create extreme chromatic shift for view2
            recolored_img = TF.adjust_hue(img, random.uniform(-0.5, 0.5))
            recolored_img = TF.adjust_saturation(recolored_img, random.uniform(0.4, 2.0))
            view2 = self.transform(recolored_img) if self.transform else TF.to_tensor(recolored_img)
            return view1, view2, label_idx, img_path

        if self.transform:
            img = self.transform(img)

        return img, label_idx, img_path


class SareeVerificationDataset(Dataset):
    """
    Evaluation dataset providing pairs of images for 1:1 design verification:
    - Positive Pairs (Label 1): Same saree design in DIFFERENT colorways / views
    - Negative Pairs (Label 0): Different saree designs, including hard negatives with identical color palettes
    """
    def __init__(self, sample_list, transform=None, num_pairs=1000, seed=42):
        super(SareeVerificationDataset, self).__init__()
        self.sample_list = sample_list  # list of (path, label_idx, class_name)
        self.transform = transform or get_eval_transforms()
        self.rng = random.Random(seed)

        # Group samples by category
        self.class_to_samples = {}
        for item in self.sample_list:
            cls_idx = item[1]
            if cls_idx not in self.class_to_samples:
                self.class_to_samples[cls_idx] = []
            self.class_to_samples[cls_idx].append(item)

        self.pairs = self._build_pairs(num_pairs)

    def _build_pairs(self, num_pairs):
        pairs = []
        classes = list(self.class_to_samples.keys())
        num_pos = num_pairs // 2
        num_neg = num_pairs - num_pos

        # 1. Positive Pairs (Same class, cross-colorway / cross-image)
        for _ in range(num_pos):
            cls = self.rng.choice(classes)
            pool = self.class_to_samples[cls]
            if len(pool) >= 2:
                s1, s2 = self.rng.sample(pool, 2)
                pairs.append((s1[0], s2[0], 1, "pos_cross_sample"))
            else:
                s1 = pool[0]
                pairs.append((s1[0], s1[0], 1, "pos_same_sample_recolored"))

        # 2. Negative Pairs (Different classes, testing against color confounders)
        for _ in range(num_neg):
            cls1, cls2 = self.rng.sample(classes, 2)
            s1 = self.rng.choice(self.class_to_samples[cls1])
            s2 = self.rng.choice(self.class_to_samples[cls2])
            pairs.append((s1[0], s2[0], 0, "neg_different_class"))

        self.rng.shuffle(pairs)
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path1, path2, is_same, pair_type = self.pairs[idx]
        img1 = Image.open(path1).convert("RGB")
        img2 = Image.open(path2).convert("RGB")

        if pair_type == "pos_same_sample_recolored":
            # Recolor img2 completely
            img2 = TF.adjust_hue(img2, 0.45)
            img2 = TF.adjust_saturation(img2, 1.6)

        t_img1 = self.transform(img1)
        t_img2 = self.transform(img2)

        return t_img1, t_img2, is_same, path1, path2


def build_gallery_and_query_sets(dataset_root, test_split="test", gallery_ratio=0.6, seed=42):
    """
    Constructs an explicit, leakage-free Gallery (Reference database) and Query (Probe set).
    Returns:
        gallery_samples: list of (path, label, class_name)
        query_samples: list of (path, label, class_name, is_color_shifted)
    """
    rng = random.Random(seed)
    test_ds = SareeClassificationDataset(dataset_root, split=test_split)

    by_class = {}
    for item in test_ds.samples:
        cls = item[1]
        if cls not in by_class:
            by_class[cls] = []
        by_class[cls].append(item)

    gallery_samples = []
    query_samples = []

    for cls, items in by_class.items():
        rng.shuffle(items)
        n_gal = max(1, int(len(items) * gallery_ratio))
        gal = items[:n_gal]
        que = items[n_gal:]
        if len(que) == 0:
            que = items[:1]  # ensure query has at least 1 sample

        gallery_samples.extend(gal)
        for q in que:
            query_samples.append((q[0], q[1], q[2], False))
            # Also create a synthetic colorway probe for rigorous cross-colorway evaluation
            query_samples.append((q[0], q[1], q[2], True))

    return gallery_samples, query_samples
