"""
metrics.py - Evaluation Protocols for Textile Identification, Verification, and Color Invariance
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


def compute_cosine_similarity_matrix(query_embeddings, gallery_embeddings):
    """
    Computes pairwise cosine similarity between query and gallery embeddings.
    Assumes embeddings are already L2 normalized.
    Returns: similarity matrix of shape (N_query, N_gallery)
    """
    if isinstance(query_embeddings, torch.Tensor):
        query_embeddings = query_embeddings.detach().cpu().numpy()
    if isinstance(gallery_embeddings, torch.Tensor):
        gallery_embeddings = gallery_embeddings.detach().cpu().numpy()

    return np.dot(query_embeddings, gallery_embeddings.T)


def compute_cmc_and_map(query_embeddings, gallery_embeddings, query_labels, gallery_labels, topk=(1, 5, 10)):
    """
    Computes Cumulative Matching Characteristics (Rank-1, Rank-5, Rank-10) and Mean Average Precision (mAP).
    """
    sim_mat = compute_cosine_similarity_matrix(query_embeddings, gallery_embeddings)
    num_queries = len(query_labels)
    cmc = {k: 0 for k in topk}
    aps = []
    mrrs = []

    for i in range(num_queries):
        q_label = query_labels[i]
        sims = sim_mat[i]
        sorted_indices = np.argsort(-sims)  # descending order
        sorted_labels = np.array(gallery_labels)[sorted_indices]

        # Match mask
        matches = (sorted_labels == q_label).astype(int)

        # CMC Rank-k
        for k in topk:
            if np.sum(matches[:k]) > 0:
                cmc[k] += 1

        # MRR
        first_match_idx = np.where(matches == 1)[0]
        if len(first_match_idx) > 0:
            mrrs.append(1.0 / (first_match_idx[0] + 1))
        else:
            mrrs.append(0.0)

        # Average Precision (AP)
        if np.sum(matches) > 0:
            cum_matches = np.cumsum(matches)
            precisions = cum_matches / (np.arange(len(matches)) + 1)
            ap = np.sum(precisions * matches) / np.sum(matches)
            aps.append(ap)
        else:
            aps.append(0.0)

    cmc_rates = {f"Rank-{k}": (cmc[k] / num_queries) * 100.0 for k in topk}
    mean_ap = float(np.mean(aps)) * 100.0
    mean_mrr = float(np.mean(mrrs))

    return {
        **cmc_rates,
        "mAP": mean_ap,
        "MRR": mean_mrr,
    }


def compute_verification_metrics(similarities, labels):
    """
    Evaluates 1:1 verification performance:
    - ROC-AUC
    - PR-AUC
    - Equal Error Rate (EER) and threshold
    - Best F1-score threshold, Accuracy, Precision, Recall
    - TAR at FAR=1.0% (0.01)
    """
    similarities = np.asarray(similarities)
    labels = np.asarray(labels).astype(int)

    # ROC curve
    fpr, tpr, roc_thresholds = roc_curve(labels, similarities)
    roc_auc = auc(fpr, tpr) * 100.0

    # PR curve
    precisions, recalls, pr_thresholds = precision_recall_curve(labels, similarities)
    pr_auc = average_precision_score(labels, similarities) * 100.0

    # Equal Error Rate (EER) where FPR == 1 - TPR (or FAR == FRR)
    fnr = 1.0 - tpr
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = float(fpr[eer_idx]) * 100.0
    eer_threshold = float(roc_thresholds[eer_idx])

    # TAR @ FAR = 1%
    far_target = 0.01
    idx_far_1 = np.where(fpr <= far_target)[0]
    tar_at_far_1 = float(tpr[idx_far_1[-1]]) * 100.0 if len(idx_far_1) > 0 else 0.0

    # Find optimal threshold by F1 score
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = float(pr_thresholds[best_f1_idx]) if best_f1_idx < len(pr_thresholds) else float(pr_thresholds[-1])
    best_f1 = float(f1_scores[best_f1_idx]) * 100.0
    best_prec = float(precisions[best_f1_idx]) * 100.0
    best_rec = float(recalls[best_f1_idx]) * 100.0

    # Verification Accuracy at optimal threshold
    predictions = (similarities >= best_threshold).astype(int)
    best_acc = np.mean(predictions == labels) * 100.0

    return {
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "EER": eer,
        "EER_Threshold": eer_threshold,
        "Best_Threshold": best_threshold,
        "Accuracy": best_acc,
        "F1_Score": best_f1,
        "Precision": best_prec,
        "Recall": best_rec,
        "TAR@FAR_1%": tar_at_far_1,
        "fpr": fpr,
        "tpr": tpr,
        "precisions": precisions,
        "recalls": recalls,
    }


def compute_color_invariance_gap(intra_cross_color_sims, inter_same_color_sims):
    """
    Computes the Color Invariance Margin:
    - Intra-Design Cross-Color Similarity: Similarity between the SAME design in DIFFERENT color palettes.
    - Inter-Design Same-Color Similarity: Similarity between DIFFERENT designs in the SAME color palette.
    - Color Invariance Gap = Intra_Cross_Color - Inter_Same_Color.
    A positive gap proves that design geometry dominates over color bias.
    """
    intra_mean = float(np.mean(intra_cross_color_sims))
    intra_std = float(np.std(intra_cross_color_sims))
    inter_mean = float(np.mean(inter_same_color_sims))
    inter_std = float(np.std(inter_same_color_sims))
    gap = intra_mean - inter_mean

    return {
        "Intra_Design_Cross_Color_Sim": (intra_mean, intra_std),
        "Inter_Design_Same_Color_Sim": (inter_mean, inter_std),
        "Color_Invariance_Gap": gap,
        "Invariance_Demonstrated": gap > 0,
    }
