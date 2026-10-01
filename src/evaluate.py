"""
evaluate.py - Full Evaluation Suite: Identification, Verification & Color Invariance Stress Tests
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

from .models import ColorInvariantSareeNet
from .transforms import get_eval_transforms, get_color_stress_transforms
from .dataset import (
    SareeClassificationDataset,
    SareeVerificationDataset,
    build_gallery_and_query_sets,
    CATEGORIES,
)
from .metrics import (
    compute_cmc_and_map,
    compute_verification_metrics,
    compute_color_invariance_gap,
)


def extract_embeddings_for_dataset(model, dataset, device, batch_size=32):
    """Extracts L2 normalized embeddings for a classification dataset."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_embs = []
    all_labels = []
    all_paths = []

    with torch.no_grad():
        for imgs, labels, paths in loader:
            imgs = imgs.to(device)
            embs = model.get_embedding(imgs)
            all_embs.append(embs.cpu().numpy())
            all_labels.extend(labels.numpy().tolist())
            all_paths.extend(paths)

    return np.vstack(all_embs), np.array(all_labels), all_paths


def evaluate_identification(model, data_dir, device, gallery_ratio=0.6, batch_size=32):
    """
    Evaluates Gallery vs Query Retrieval (Identification).
    Includes standard query probes and cross-colorway (recolored) query probes.
    """
    eval_tf = get_eval_transforms(img_size=224)
    gal_samples, que_samples = build_gallery_and_query_sets(
        data_dir, test_split="test", gallery_ratio=gallery_ratio
    )

    print(f"Identification Split: {len(gal_samples)} Gallery items, {len(que_samples)} Query probes.")

    # Extract Gallery Embeddings
    gal_imgs = [eval_tf(Image.open(p).convert("RGB")) for p, _, _ in gal_samples]
    gal_tensor = torch.stack(gal_imgs).to(device)
    with torch.no_grad():
        gal_embs = model.get_embedding(gal_tensor).cpu().numpy()
    gal_labels = [lbl for _, lbl, _ in gal_samples]

    # Extract Query Embeddings (Standard & Color-Shifted)
    que_imgs = []
    for path, _, _, is_recolored in que_samples:
        img = Image.open(path).convert("RGB")
        if is_recolored:
            img = get_color_stress_transforms(img_size=224, stress_type="hue_rotate")(img)
        else:
            img = eval_tf(img)
        que_imgs.append(img)

    que_tensor = torch.stack(que_imgs).to(device)
    with torch.no_grad():
        que_embs = model.get_embedding(que_tensor).cpu().numpy()
    que_labels = [lbl for _, lbl, _, _ in que_samples]

    # Overall Identification Metrics
    all_metrics = compute_cmc_and_map(que_embs, gal_embs, que_labels, gal_labels, topk=(1, 5, 10))

    # Split into standard probes vs cross-colorway probes
    std_indices = [i for i, (_, _, _, is_rec) in enumerate(que_samples) if not is_rec]
    rec_indices = [i for i, (_, _, _, is_rec) in enumerate(que_samples) if is_rec]

    std_metrics = compute_cmc_and_map(
        que_embs[std_indices], gal_embs, [que_labels[i] for i in std_indices], gal_labels, topk=(1, 5)
    )
    rec_metrics = compute_cmc_and_map(
        que_embs[rec_indices], gal_embs, [que_labels[i] for i in rec_indices], gal_labels, topk=(1, 5)
    )

    return {
        "Overall": all_metrics,
        "Standard_Queries": std_metrics,
        "Cross_Colorway_Queries": rec_metrics,
    }


def evaluate_verification(model, data_dir, device, num_pairs=600):
    """
    Evaluates 1:1 Design Verification on balanced test pairs.
    """
    eval_tf = get_eval_transforms(img_size=224)
    test_ds = SareeClassificationDataset(data_dir, split="test", transform=eval_tf)
    if len(test_ds) < 10:
        # Include validation if test set is small
        val_ds = SareeClassificationDataset(data_dir, split="valid", transform=eval_tf)
        all_samples = test_ds.samples + val_ds.samples
    else:
        all_samples = test_ds.samples

    pair_ds = SareeVerificationDataset(all_samples, transform=eval_tf, num_pairs=num_pairs, seed=42)
    loader = DataLoader(pair_ds, batch_size=32, shuffle=False, num_workers=0)

    all_sims = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for img1, img2, is_same, _, _ in loader:
            img1, img2 = img1.to(device), img2.to(device)
            emb1 = model.get_embedding(img1)
            emb2 = model.get_embedding(img2)
            sims = torch.sum(emb1 * emb2, dim=-1).cpu().numpy()
            all_sims.extend(sims.tolist())
            all_labels.extend(is_same.numpy().tolist())

    verif_metrics = compute_verification_metrics(all_sims, all_labels)
    return verif_metrics


def evaluate_color_invariance_stress(model, data_dir, device, num_samples=100):
    """
    Measures cosine similarities under extreme chromatic alterations.
    Compares intra-motif cross-color similarity vs inter-motif same-color similarity.
    """
    eval_tf = get_eval_transforms(img_size=224)
    test_ds = SareeClassificationDataset(data_dir, split="test", transform=eval_tf)
    if len(test_ds) < 20:
        val_ds = SareeClassificationDataset(data_dir, split="valid", transform=eval_tf)
        samples = test_ds.samples + val_ds.samples
    else:
        samples = test_ds.samples

    stress_transforms = {
        "Hue Inversion": get_color_stress_transforms(224, "hue_rotate"),
        "Grayscale": get_color_stress_transforms(224, "grayscale"),
        "Channel Swap": get_color_stress_transforms(224, "channel_swap"),
        "Invert": get_color_stress_transforms(224, "invert"),
    }

    stress_results = {}
    model.eval()

    with torch.no_grad():
        for s_name, s_tf in stress_transforms.items():
            intra_sims = []
            for path, _, _ in samples[:num_samples]:
                raw_img = Image.open(path).convert("RGB")
                orig_t = eval_tf(raw_img).unsqueeze(0).to(device)
                pert_t = s_tf(raw_img).unsqueeze(0).to(device)

                emb_orig = model.get_embedding(orig_t)
                emb_pert = model.get_embedding(pert_t)
                sim = torch.sum(emb_orig * emb_pert).item()
                intra_sims.append(sim)

            stress_results[s_name] = {
                "Mean_Cosine_Sim": float(np.mean(intra_sims)),
                "Std_Cosine_Sim": float(np.std(intra_sims)),
                "Min_Cosine_Sim": float(np.min(intra_sims)),
            }

    return stress_results


def run_full_evaluation(checkpoint_path, data_dir="archive", device_str="auto"):
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    backbone = ckpt.get("backbone", "resnet50")
    embedding_dim = ckpt.get("embedding_dim", 512)
    num_classes = ckpt.get("num_classes", 4)

    model = ColorInvariantSareeNet(
        backbone_name=backbone,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("\n=================== 1. IDENTIFICATION EVALUATION ===================")
    ident_res = evaluate_identification(model, data_dir, device)
    print(f"Overall Retrieval:   Rank-1: {ident_res['Overall']['Rank-1']:.2f}% | Rank-5: {ident_res['Overall']['Rank-5']:.2f}% | mAP: {ident_res['Overall']['mAP']:.2f}%")
    print(f"Standard Queries:    Rank-1: {ident_res['Standard_Queries']['Rank-1']:.2f}% | Rank-5: {ident_res['Standard_Queries']['Rank-5']:.2f}% | mAP: {ident_res['Standard_Queries']['mAP']:.2f}%")
    print(f"Cross-Color Probes:  Rank-1: {ident_res['Cross_Colorway_Queries']['Rank-1']:.2f}% | Rank-5: {ident_res['Cross_Colorway_Queries']['Rank-5']:.2f}% | mAP: {ident_res['Cross_Colorway_Queries']['mAP']:.2f}%")

    print("\n=================== 2. VERIFICATION EVALUATION ===================")
    verif_res = evaluate_verification(model, data_dir, device)
    print(f"ROC-AUC:             {verif_res['ROC_AUC']:.2f}%")
    print(f"PR-AUC:              {verif_res['PR_AUC']:.2f}%")
    print(f"Equal Error Rate:    {verif_res['EER']:.2f}% (Threshold: {verif_res['EER_Threshold']:.4f})")
    print(f"Best Accuracy:       {verif_res['Accuracy']:.2f}% (Threshold: {verif_res['Best_Threshold']:.4f})")
    print(f"Best F1-Score:       {verif_res['F1_Score']:.2f}% (Precision: {verif_res['Precision']:.2f}%, Recall: {verif_res['Recall']:.2f}%)")
    print(f"TAR @ FAR = 1.0%:    {verif_res['TAR@FAR_1%']:.2f}%")

    print("\n=================== 3. COLOR INVARIANCE STRESS TESTS ===================")
    stress_res = evaluate_color_invariance_stress(model, data_dir, device)
    for test_name, res in stress_res.items():
        print(f"{test_name:<18}: Mean Cosine Sim = {res['Mean_Cosine_Sim']:.4f} (+/- {res['Std_Cosine_Sim']:.4f})")

    return {
        "identification": ident_res,
        "verification": verif_res,
        "color_stress": stress_res,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Saree Recognition Model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_saree_model.pth")
    parser.add_argument("--data_dir", type=str, default="archive")
    args = parser.parse_args()

    run_full_evaluation(args.checkpoint, args.data_dir)
