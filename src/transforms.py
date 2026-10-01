"""
transforms.py - Color Invariance & Structural Preprocessing Pipeline for Saree Recognition
"""

import random
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps, ImageEnhance


class RandomChannelPermute:
    """Randomly permutes RGB color channels to break specific chromatic dependencies."""
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


class RandomSolarize:
    """Inverts all pixel values above a threshold with probability p."""
    def __init__(self, threshold=128, p=0.2):
        self.threshold = threshold
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return ImageOps.solarize(img, self.threshold)
        return img


class RandomAutoContrast:
    """Applies autocontrast with probability p."""
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return ImageOps.autocontrast(img)
        return img


class SyntheticPaletteShift:
    """Simulates realistic textile colorway changes by shifting HSV hue and saturation."""
    def __init__(self, hue_shift_range=(-0.5, 0.5), p=0.8):
        self.hue_shift_range = hue_shift_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            # Shift hue randomly across full spectrum
            hue_factor = random.uniform(self.hue_shift_range[0], self.hue_shift_range[1])
            sat_factor = random.uniform(0.5, 1.8)
            bright_factor = random.uniform(0.7, 1.3)
            img = TF.adjust_hue(img, hue_factor)
            img = TF.adjust_saturation(img, sat_factor)
            img = TF.adjust_brightness(img, bright_factor)
        return img


def get_train_transforms(img_size=224, aggressive_color=True):
    """
    Returns training transforms engineered for color-invariant feature learning.
    Combines spatial invariance (flips, crops, affine) with strong chromatic invariance
    (color jitter, hue shifts, channel permutations, grayscale conversions).
    """
    transform_list = [
        T.Resize((int(img_size * 1.14), int(img_size * 1.14))),
        T.RandomCrop((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.RandomRotation(degrees=15, interpolation=T.InterpolationMode.BILINEAR),
    ]

    if aggressive_color:
        transform_list.extend([
            SyntheticPaletteShift(hue_shift_range=(-0.5, 0.5), p=0.6),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.5, hue=0.3),
            RandomChannelPermute(p=0.3),
            RandomSolarize(threshold=128, p=0.15),
            RandomAutoContrast(p=0.25),
            T.RandomGrayscale(p=0.3),  # 30% of time drop all color entirely to force geometry/weave learning
        ])

    transform_list.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.2, scale=(0.02, 0.2), value="random"),
    ])

    return T.Compose(transform_list)


def get_eval_transforms(img_size=224):
    """Deterministic evaluation transforms for Gallery indexing and Query matching."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_color_stress_transforms(img_size=224, stress_type="hue_rotate"):
    """
    Stress-testing transformations to evaluate true color invariance on unseen sarees.
    Types: 'hue_rotate', 'grayscale', 'invert', 'extreme_jitter', 'channel_swap'.
    """
    if stress_type == "hue_rotate":
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.Lambda(lambda img: TF.adjust_hue(img, 0.5)),  # 180-degree inverted hue spectrum
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif stress_type == "grayscale":
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif stress_type == "invert":
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.Lambda(lambda img: ImageOps.invert(img.convert("RGB"))),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif stress_type == "channel_swap":
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.Lambda(lambda img: Image.merge("RGB", (img.split()[2], img.split()[0], img.split()[1]))),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return get_eval_transforms(img_size=img_size)
