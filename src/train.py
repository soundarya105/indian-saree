"""
train.py - Training Pipeline with ArcFace, SupCon, and Color-Invariance Regularization
"""

import os
import argparse
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .models import ColorInvariantSareeNet
from .losses import CombinedMetricLoss
from .transforms import get_train_transforms, get_eval_transforms
from .dataset import SareeClassificationDataset
from .metrics import compute_cmc_and_map, compute_verification_metrics, compute_cosine_similarity_matrix


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device, scaler=None):
    model.train()
    total_loss = 0.0
    loss_dict_total = {"arc": 0.0, "supcon": 0.0, "inv": 0.0}
    num_batches = len(dataloader)

    for i, (v1, v2, labels, _) in enumerate(dataloader):
        v1 = v1.to(device)
        v2 = v2.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if scaler and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits1, emb1 = model(v1, labels)
                emb2 = model.get_embedding(v2)
                loss, loss_dict = criterion(logits1, emb1, labels, recolored_embeddings=emb2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits1, emb1 = model(v1, labels)
            emb2 = model.get_embedding(v2)
            loss, loss_dict = criterion(logits1, emb1, labels, recolored_embeddings=emb2)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        for k in loss_dict_total:
            loss_dict_total[k] += loss_dict.get(k, 0.0)

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / max(1, num_batches)
    avg_losses = {k: v / max(1, num_batches) for k, v in loss_dict_total.items()}
    return avg_loss, avg_losses


def evaluate_val_set(model, dataloader, device):
    """Evaluates validation set retrieval (Rank-1, mAP)."""
    model.eval()
    all_embs = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels, _ in dataloader:
            imgs = imgs.to(device)
            embs = model.get_embedding(imgs)
            all_embs.append(embs.cpu())
            all_labels.extend(labels.numpy().tolist())

    if len(all_embs) == 0:
        return {"Rank-1": 0.0, "mAP": 0.0}

    all_embs = torch.cat(all_embs, dim=0).numpy()
    metrics = compute_cmc_and_map(
        query_embeddings=all_embs,
        gallery_embeddings=all_embs,
        query_labels=all_labels,
        gallery_labels=all_labels,
        topk=(1, 5),
    )
    return metrics


def run_training(
    data_dir="archive",
    backbone="resnet50",
    embedding_dim=512,
    num_classes=4,
    epochs=15,
    batch_size=32,
    lr=1e-4,
    weight_decay=1e-2,
    output_dir="checkpoints",
    device_str="auto"
):
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Training on device: {device} | Backbone: {backbone} | Epochs: {epochs} | Batch: {batch_size}")

    train_tf = get_train_transforms(img_size=224, aggressive_color=True)
    val_tf = get_eval_transforms(img_size=224)

    train_ds = SareeClassificationDataset(data_dir, split="train", transform=train_tf, return_pair=True)
    val_ds = SareeClassificationDataset(data_dir, split="valid", transform=val_tf, return_pair=False)

    print(f"Loaded {len(train_ds)} train samples, {len(val_ds)} validation samples.")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=(len(train_ds) > batch_size),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = ColorInvariantSareeNet(
        backbone_name=backbone,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        pretrained=True,
        gem_p=3.0,
        arc_margin=0.45,
        arc_scale=32.0,
    ).to(device)

    # Differential learning rate (smaller for pretrained backbone, higher for head)
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr * 0.2},
            {"params": head_params, "lr": lr},
        ],
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = CombinedMetricLoss(alpha_arc=1.0, beta_supcon=0.25, gamma_inv=0.25)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_map = 0.0
    best_checkpoint = os.path.join(output_dir, "best_saree_model.pth")
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, loss_breakdown = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, scaler
        )
        val_metrics = evaluate_val_set(model, val_loader, device)
        elapsed = time.time() - t0

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "loss_arc": loss_breakdown["arc"],
            "loss_supcon": loss_breakdown["supcon"],
            "loss_inv": loss_breakdown["inv"],
            "val_rank1": val_metrics["Rank-1"],
            "val_map": val_metrics["mAP"],
            "time_sec": elapsed,
        }
        history.append(record)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Loss: {train_loss:.4f} (Arc: {loss_breakdown['arc']:.3f}, Inv: {loss_breakdown['inv']:.3f}) | "
            f"Val Rank-1: {val_metrics['Rank-1']:.2f}% | Val mAP: {val_metrics['mAP']:.2f}% | "
            f"Time: {elapsed:.1f}s"
        )

        # Checkpointing
        if val_metrics["mAP"] >= best_map:
            best_map = val_metrics["mAP"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "backbone": backbone,
                "embedding_dim": embedding_dim,
                "num_classes": num_classes,
            }, best_checkpoint)

    # Save final model
    final_checkpoint = os.path.join(output_dir, "latest_saree_model.pth")
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "backbone": backbone,
        "embedding_dim": embedding_dim,
        "num_classes": num_classes,
    }, final_checkpoint)

    print(f"\nTraining Complete. Best Validation mAP: {best_map:.2f}%. Saved to {best_checkpoint}")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Color-Invariant Saree Recognition Model")
    parser.add_argument("--data_dir", type=str, default="archive")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "resnet34", "efficientnet_b0", "convnext_tiny"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    run_training(
        data_dir=args.data_dir,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
    )
