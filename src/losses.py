"""
losses.py - Metric Learning Loss Functions: ArcFace, Supervised Contrastive, and Color Invariance Regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceCrossEntropyLoss(nn.Module):
    """Standard CrossEntropy loss over ArcMargin logits."""
    def __init__(self, label_smoothing=0.05):
        super(ArcFaceCrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits, labels):
        return self.criterion(logits, labels)


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Learning (SupCon) Loss.
    Pulls samples of the same design/motif together and repels different designs on hypersphere.
    Reference: Khosla et al., 'Supervised Contrastive Learning', NeurIPS 2020.
    """
    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, dim] or [bsz, dim].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                  has the same class as sample i. Can be asymmetric.
        """
        device = features.device

        if len(features.shape) < 3:
            # (bsz, dim) -> (bsz, 1, dim)
            features = features.unsqueeze(1)

        batch_size = features.shape[0]
        if labels is not None and mask is None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        elif mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown mode: {self.contrast_mode}")

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature,
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-7)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-7)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class ColorInvarianceConsistencyLoss(nn.Module):
    """
    Consistency Regularization Loss.
    Directly minimizes cosine distance between the embeddings of an original saree
    and its synthetically color-shifted / recolored colorway.
    """
    def __init__(self):
        super(ColorInvarianceConsistencyLoss, self).__init__()

    def forward(self, emb_orig, emb_recolored):
        # emb_orig, emb_recolored: (B, D) L2-normalized
        cos_sim = torch.sum(emb_orig * emb_recolored, dim=-1)
        loss = torch.mean(1.0 - cos_sim)
        return loss


class CombinedMetricLoss(nn.Module):
    """
    Multi-objective Loss combining:
    1. ArcFace Additive Angular Margin Loss (Classification & Hypersphere Margin)
    2. Supervised Contrastive Loss (Intra-class cluster compaction)
    3. Color Invariance Pairwise Regularizer (Invariance across chromatic shifts)
    """
    def __init__(self, alpha_arc=1.0, beta_supcon=0.2, gamma_inv=0.2):
        super(CombinedMetricLoss, self).__init__()
        self.arc_loss = ArcFaceCrossEntropyLoss()
        self.supcon_loss = SupervisedContrastiveLoss()
        self.inv_loss = ColorInvarianceConsistencyLoss()
        self.alpha = alpha_arc
        self.beta = beta_supcon
        self.gamma = gamma_inv

    def forward(self, logits, embeddings, labels, recolored_embeddings=None):
        l_arc = self.arc_loss(logits, labels)
        l_supcon = self.supcon_loss(embeddings, labels) if self.beta > 0 else torch.tensor(0.0, device=logits.device)
        l_inv = (
            self.inv_loss(embeddings, recolored_embeddings)
            if (self.gamma > 0 and recolored_embeddings is not None)
            else torch.tensor(0.0, device=logits.device)
        )
        total_loss = self.alpha * l_arc + self.beta * l_supcon + self.gamma * l_inv
        return total_loss, {"arc": l_arc.item(), "supcon": l_supcon.item(), "inv": l_inv.item()}
