"""
models.py - Deep Metric Learning Architectures with GeM Pooling & ArcMargin Head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class GeMPooling(nn.Module):
    """
    Generalized Mean (GeM) Pooling with learnable exponent parameter p.
    Captures salient motif patterns and localized textile textures better than standard GAP.
    Reference: Radenovic et al., 'Fine-tuning CNN Image Retrieval with No Human Annotation', TPAMI 2018.
    """
    def __init__(self, p=3.0, eps=1e-6, learnable=True):
        super(GeMPooling, self).__init__()
        if learnable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.register_buffer("p", torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3.0, eps=1e-6):
        # x: (B, C, H, W) -> (B, C, 1, 1) -> (B, C)
        x_clamped = x.clamp(min=eps).pow(p)
        pooled = F.avg_pool2d(x_clamped, (x_clamped.size(-2), x_clamped.size(-1))).pow(1.0 / p)
        return pooled.flatten(1)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class ArcMarginProduct(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss Layer.
    Forces compact intra-design clusters and maximizes inter-design angular separation on hypersphere.
    Reference: Deng et al., 'ArcFace: Additive Angular Margin Loss for Deep Face Recognition', CVPR 2019.
    """
    def __init__(self, in_features, num_classes, s=32.0, m=0.45, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # input: (B, in_features), weight: (num_classes, in_features)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if label is None:
            return cosine * self.s

        sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 0, 1) + 1e-7)
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class ColorInvariantSareeNet(nn.Module):
    """
    Color-Invariant Saree Design Recognition Network.
    Combines a deep spatial feature extractor, GeM pooling, metric projection head, and ArcMargin classifier.
    Produces L2-normalized 512-dim design embeddings invariant to colorway changes.
    """
    def __init__(
        self,
        backbone_name="resnet50",
        embedding_dim=512,
        num_classes=4,
        pretrained=True,
        gem_p=3.0,
        arc_margin=0.45,
        arc_scale=32.0,
        dropout=0.15
    ):
        super(ColorInvariantSareeNet, self).__init__()
        self.backbone_name = backbone_name
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        # 1. Backbone Feature Extractor
        if backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base = models.resnet50(weights=weights)
            in_features = base.fc.in_features  # 2048
            self.backbone = nn.Sequential(*list(base.children())[:-2])
        elif backbone_name == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            base = models.resnet34(weights=weights)
            in_features = base.fc.in_features  # 512
            self.backbone = nn.Sequential(*list(base.children())[:-2])
        elif backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b0(weights=weights)
            in_features = base.classifier[1].in_features  # 1280
            self.backbone = base.features
        elif backbone_name == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            base = models.convnext_tiny(weights=weights)
            in_features = base.classifier[2].in_features  # 768
            self.backbone = base.features
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # 2. Generalized Mean (GeM) Pooling
        self.pool = GeMPooling(p=gem_p, learnable=True)

        # 3. Metric Projection Head
        self.projection_head = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.PReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        # 4. ArcFace Metric Classification Head
        if num_classes is not None and num_classes > 0:
            self.arc_head = ArcMarginProduct(
                in_features=embedding_dim,
                num_classes=num_classes,
                s=arc_scale,
                m=arc_margin,
            )
        else:
            self.arc_head = None

    def extract_features(self, x):
        """Extracts spatial feature maps and applies GeM pooling."""
        feats = self.backbone(x)
        pooled = self.pool(feats)
        return pooled

    def get_embedding(self, x):
        """Returns L2-normalized 512-dim design embedding vector."""
        pooled = self.extract_features(x)
        proj = self.projection_head(pooled)
        normalized_emb = F.normalize(proj, p=2, dim=1)
        return normalized_emb

    def forward(self, x, labels=None):
        """
        Forward pass.
        If labels is provided, computes ArcFace margin logits for metric training.
        If labels is None, returns normalized embeddings for inference/verification.
        """
        emb = self.get_embedding(x)
        if labels is not None and self.arc_head is not None:
            logits = self.arc_head(emb, labels)
            return logits, emb
        return emb
