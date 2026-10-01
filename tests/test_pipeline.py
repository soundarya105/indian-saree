import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
import torch
import numpy as np
from PIL import Image

from src.transforms import get_train_transforms, get_eval_transforms, get_color_stress_transforms
from src.models import ColorInvariantSareeNet, GeMPooling, ArcMarginProduct
from src.losses import CombinedMetricLoss, SupervisedContrastiveLoss, ArcFaceCrossEntropyLoss
from src.metrics import compute_cmc_and_map, compute_verification_metrics, compute_color_invariance_gap
from src.efficiency import count_parameters, estimate_flops, benchmark_latency
from src.inference import SareeDesignMatcher


class TestSareePipeline(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.dummy_img = Image.new("RGB", (256, 256), color=(200, 100, 50))
        self.dummy_tensor = torch.randn(4, 3, 224, 224)
        self.dummy_labels = torch.tensor([0, 1, 2, 3])

    def test_transforms(self):
        train_tf = get_train_transforms(224)
        eval_tf = get_eval_transforms(224)
        stress_tf = get_color_stress_transforms(224, "hue_rotate")

        t1 = train_tf(self.dummy_img)
        t2 = eval_tf(self.dummy_img)
        t3 = stress_tf(self.dummy_img)

        self.assertEqual(t1.shape, (3, 224, 224))
        self.assertEqual(t2.shape, (3, 224, 224))
        self.assertEqual(t3.shape, (3, 224, 224))

    def test_gem_pooling(self):
        gem = GeMPooling(p=3.0)
        feat = torch.randn(2, 64, 7, 7).abs() + 0.1
        pooled = gem(feat)
        self.assertEqual(pooled.shape, (2, 64))

    def test_model_forward(self):
        model = ColorInvariantSareeNet(
            backbone_name="resnet50",
            embedding_dim=512,
            num_classes=4,
            pretrained=False,
        )
        model.eval()
        emb = model.get_embedding(self.dummy_tensor)
        self.assertEqual(emb.shape, (4, 512))
        # Verify L2 normalization
        norms = torch.norm(emb, p=2, dim=1)
        np.testing.assert_allclose(norms.detach().numpy(), np.ones(4), atol=1e-5)

        logits, _ = model(self.dummy_tensor, self.dummy_labels)
        self.assertEqual(logits.shape, (4, 4))

    def test_losses(self):
        criterion = CombinedMetricLoss(alpha_arc=1.0, beta_supcon=0.2, gamma_inv=0.2)
        logits = torch.randn(4, 4)
        embs1 = torch.nn.functional.normalize(torch.randn(4, 512), dim=1)
        embs2 = torch.nn.functional.normalize(torch.randn(4, 512), dim=1)
        labels = torch.tensor([0, 1, 0, 1])

        loss, loss_dict = criterion(logits, embs1, labels, recolored_embeddings=embs2)
        self.assertGreater(loss.item(), 0.0)
        self.assertIn("arc", loss_dict)
        self.assertIn("supcon", loss_dict)
        self.assertIn("inv", loss_dict)

    def test_metrics(self):
        # Identification Metrics
        gal_embs = torch.nn.functional.normalize(torch.randn(20, 512), dim=1).numpy()
        que_embs = gal_embs[:5] + np.random.randn(5, 512) * 0.05
        que_embs = que_embs / np.linalg.norm(que_embs, axis=1, keepdims=True)
        gal_labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
        que_labels = [0, 0, 0, 0, 0]

        cmc = compute_cmc_and_map(que_embs, gal_embs, que_labels, gal_labels)
        self.assertIn("Rank-1", cmc)
        self.assertIn("mAP", cmc)
        self.assertGreaterEqual(cmc["Rank-1"], 0.0)

        # Verification Metrics
        sims = np.concatenate([np.random.uniform(0.7, 0.99, 50), np.random.uniform(0.1, 0.4, 50)])
        labels = np.concatenate([np.ones(50), np.zeros(50)])
        verif = compute_verification_metrics(sims, labels)
        self.assertGreater(verif["ROC_AUC"], 90.0)
        self.assertLess(verif["EER"], 10.0)

    def test_efficiency(self):
        model = ColorInvariantSareeNet(
            backbone_name="resnet50",
            embedding_dim=512,
            num_classes=4,
            pretrained=False,
        )
        params = count_parameters(model)
        self.assertGreater(params["Total_Parameters"], 20_000_000)

        flops = estimate_flops(model, input_size=(1, 3, 224, 224))
        self.assertGreater(flops["GFLOPs"], 1.0)

    def test_inference_engine(self):
        matcher = SareeDesignMatcher(
            checkpoint_path=None,
            backbone="resnet50",
            embedding_dim=512,
            num_classes=4,
            device="cpu",
        )
        dummy_paths = []
        # Create small dummy images on disk
        os.makedirs("tests/tmp", exist_ok=True)
        for i in range(5):
            p = f"tests/tmp/dummy_{i}.jpg"
            Image.new("RGB", (100, 100), color=(i * 40, 50, 100)).save(p)
            dummy_paths.append(p)

        matcher.build_gallery(dummy_paths, labels=["Banarasi", "Banarasi", "Bandhani", "Ikat", "Pichwai"])
        results = matcher.identify(dummy_paths[0], top_k=3)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["rank"], 1)

        v_res = matcher.verify(dummy_paths[0], dummy_paths[1])
        self.assertIn("is_same_design", v_res)
        self.assertIn("similarity", v_res)


if __name__ == "__main__":
    unittest.main()
