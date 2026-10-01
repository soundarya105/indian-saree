"""
inference.py - Production Inference Engine for Saree Identification & Pair Verification
"""

import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from .models import ColorInvariantSareeNet
from .transforms import get_eval_transforms


class SareeDesignMatcher:
    """
    Production-grade Inference Matcher:
    - Fast Gallery Indexing
    - Top-K Textile Motif Identification
    - 1:1 Design Verification with Confidence Scores
    - Visual Result Plotting
    """
    def __init__(self, checkpoint_path=None, backbone="resnet50", embedding_dim=512, num_classes=4, device=None, default_threshold=0.65):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.default_threshold = default_threshold
        self.transform = get_eval_transforms(img_size=224)

        self.model = ColorInvariantSareeNet(
            backbone_name=backbone,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            pretrained=False if checkpoint_path else True,
        ).to(self.device)

        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            if "model_state_dict" in ckpt:
                self.model.load_state_dict(ckpt["model_state_dict"])
            else:
                self.model.load_state_dict(ckpt)
            print(f"Loaded checkpoint from {checkpoint_path}")

        self.model.eval()

        # Gallery database
        self.gallery_embeddings = None  # (N, 512)
        self.gallery_metadata = []      # list of dicts

    def _prepare_image(self, img_input):
        """Converts file path, PIL Image, or NumPy array to transformed tensor."""
        if isinstance(img_input, str):
            img = Image.open(img_input).convert("RGB")
        elif isinstance(img_input, np.ndarray):
            img = Image.fromarray(img_input).convert("RGB")
        elif isinstance(img_input, Image.Image):
            img = img_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(img_input)}")

        tensor = self.transform(img).unsqueeze(0).to(self.device)
        return tensor, img

    def extract_embedding(self, img_input):
        """Extracts unit-normalized 512-dim embedding for a single image."""
        tensor, _ = self._prepare_image(img_input)
        with torch.no_grad():
            emb = self.model.get_embedding(tensor)
        return emb.squeeze(0).cpu().numpy()

    def build_gallery(self, image_paths, labels=None, metadata_list=None):
        """Indexes reference saree designs into the in-memory gallery."""
        embeddings = []
        self.gallery_metadata = []

        print(f"Indexing {len(image_paths)} saree designs into gallery...")
        for i, path in enumerate(image_paths):
            emb = self.extract_embedding(path)
            embeddings.append(emb)

            meta = {
                "id": i,
                "path": path,
                "label": labels[i] if labels is not None else None,
                "extra": metadata_list[i] if metadata_list is not None else {},
            }
            self.gallery_metadata.append(meta)

        self.gallery_embeddings = np.vstack(embeddings)  # (N, D)
        print(f"Gallery indexing complete. Index shape: {self.gallery_embeddings.shape}")

    def identify(self, query_img_input, top_k=5):
        """
        Ranks gallery designs by cosine similarity to the query image.
        Returns top-K matching records with similarities and decision flags.
        """
        if self.gallery_embeddings is None or len(self.gallery_embeddings) == 0:
            raise RuntimeError("Gallery is empty. Please run build_gallery() first.")

        q_emb = self.extract_embedding(query_img_input)  # (D,)
        # Cosine similarities
        sims = np.dot(self.gallery_embeddings, q_emb)    # (N,)

        sorted_indices = np.argsort(-sims)[:top_k]
        results = []

        for rank, idx in enumerate(sorted_indices, 1):
            score = float(sims[idx])
            meta = self.gallery_metadata[idx]
            results.append({
                "rank": rank,
                "gallery_id": meta["id"],
                "path": meta["path"],
                "label": meta["label"],
                "similarity": score,
                "is_match": bool(score >= self.default_threshold),
            })

        return results

    def verify(self, img_input_1, img_input_2, threshold=None):
        """
        Verifies if two saree images carry the SAME design motif, independent of color palette.
        """
        th = threshold if threshold is not None else self.default_threshold
        emb1 = self.extract_embedding(img_input_1)
        emb2 = self.extract_embedding(img_input_2)

        cosine_sim = float(np.dot(emb1, emb2))
        is_same = bool(cosine_sim >= th)
        confidence = float(abs(cosine_sim - th) / (1.0 - th + 1e-6))
        confidence = min(1.0, max(0.0, confidence))

        return {
            "is_same_design": is_same,
            "similarity": cosine_sim,
            "threshold": th,
            "confidence": confidence,
            "verdict": "MATCH (Same Design)" if is_same else "NO MATCH (Different Design)",
        }

    def plot_retrieval(self, query_img_input, results, save_path=None):
        """Visualizes Top-K gallery retrieval in a multi-card display."""
        _, q_pil = self._prepare_image(query_img_input)
        k = len(results)

        fig, axes = plt.subplots(1, k + 1, figsize=(3.5 * (k + 1), 4))
        # Query image
        axes[0].imshow(q_pil)
        axes[0].set_title("QUERY PROBE\n(Input Image)", fontsize=11, fontweight="bold", color="darkblue")
        axes[0].axis("off")

        for i, res in enumerate(results):
            match_pil = Image.open(res["path"]).convert("RGB")
            badge = "MATCH" if res["is_match"] else "DISTINCT"
            color = "green" if res["is_match"] else "red"

            axes[i + 1].imshow(match_pil)
            axes[i + 1].set_title(
                f"Rank #{res['rank']} | Sim: {res['similarity']:.3f}\n[{badge}] {res['label'] or ''}",
                fontsize=10,
                color=color,
                fontweight="bold",
            )
            axes[i + 1].axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved retrieval visualization to {save_path}")
        return fig
