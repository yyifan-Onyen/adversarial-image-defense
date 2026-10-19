import numpy as np
import torch
from torch import nn

from aidefense.ablations import fit_ablation, score_ablation
from aidefense.attacks import cw_l2, fgsm, pgd
from aidefense.data import split_indices
from aidefense.detector import Detector, threshold_at_fpr
from aidefense.pipeline import DefensePipeline
from aidefense.squeezing import feature_tensor, squeeze


class TinyClassifier(nn.Module):
    def forward(self, x):
        mean = x.mean(dim=(1, 2, 3))
        return torch.stack([mean, 1 - mean], dim=1)


class Brighten(nn.Module):
    def forward(self, x):
        return (x + 0.1).clamp(0, 1)


def test_split_is_disjoint_and_reproducible():
    train, validation = split_indices(2026)
    train_again, validation_again = split_indices(2026)
    assert len(train) == 45000 and len(validation) == 5000
    assert len(set(train) | set(validation)) == 50000
    assert not (set(train) & set(validation))
    assert train == train_again and validation == validation_again


def test_linf_attacks_respect_bounds():
    model = TinyClassifier()
    x = torch.rand(3, 3, 32, 32)
    y = torch.tensor([0, 1, 0])
    for attacked in (fgsm(model, x, y, 4 / 255), pgd(model, x, y, 4 / 255, steps=3)):
        assert attacked.min() >= 0 and attacked.max() <= 1
        assert (attacked - x).abs().max() <= 4 / 255 + 1e-6


def test_squeezers_and_features_have_expected_shape():
    x = torch.rand(2, 3, 32, 32)
    for name in ("quantize_4bit", "median_3x3", "jpeg_q50"):
        transformed = squeeze(x, name)
        assert transformed.shape == x.shape
        assert transformed.min() >= 0 and transformed.max() <= 1
    logits, features = feature_tensor(TinyClassifier(), x)
    assert logits.shape == (2, 2)
    assert features.shape == (2, 4)
    assert torch.isfinite(features).all()


def test_calibration_does_not_exceed_target_false_positive_rate():
    clean = np.arange(100, dtype=float)
    threshold = threshold_at_fpr(clean, 0.05)
    assert np.mean(clean > threshold) <= 0.05


def test_exact_routing_uses_the_same_purifier_for_both_gates():
    model = TinyClassifier()
    purifier = Brighten()
    detector = Detector(np.zeros(4), 0.0, -1.0, -1.0)
    pipeline = DefensePipeline(model, purifier, detector)
    x = torch.full((2, 3, 32, 32), 0.5)
    always, _, _ = pipeline.infer(x, "always")
    for rule in ("max", "fusion"):
        routed, flags, _ = pipeline.infer(x, rule)
        assert flags.all()
        assert torch.allclose(routed, always)


def test_adaptive_surrogate_has_input_gradient_and_exact_verification_path():
    detector = Detector(np.ones(4), 0.0, 0.0, 0.0)
    pipeline = DefensePipeline(TinyClassifier(), Brighten(), detector)
    x = torch.full((1, 3, 32, 32), 0.4, requires_grad=True)
    gradient = torch.autograd.grad(pipeline.surrogate_logits(x)[:, 0].sum(), x)[0]
    assert gradient.shape == x.shape
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
    _, flags, _ = pipeline.infer(x.detach(), "fusion")
    assert flags.shape == (1,)


def test_cw_attack_returns_bounded_candidate_and_success_mask():
    model = TinyClassifier()
    x = torch.full((1, 3, 32, 32), 0.55)
    y = torch.tensor([0])
    attacked, success = cw_l2(model, x, y, steps=2, binary_search_steps=1)
    assert attacked.shape == x.shape and success.shape == (1,)
    assert attacked.min() >= 0 and attacked.max() <= 1


def test_squeezer_ablation_refits_and_calibrates_both_rules():
    rng = np.random.default_rng(2026)
    clean = rng.uniform(0, 0.2, size=(100, 4))
    attacked = rng.uniform(0.2, 0.8, size=(100, 4))
    train_features = np.vstack((clean, attacked))
    train_labels = np.r_[np.zeros(100), np.ones(100)]
    model, fusion_threshold, max_threshold = fit_ablation(
        train_features, train_labels, clean, included=[0, 2])
    scores = score_ablation(model, clean, attacked, [0, 2], fusion_threshold, max_threshold)
    assert scores["fusion"]["test_clean_fpr"] <= 0.05
    assert scores["max"]["test_clean_fpr"] <= 0.05
    assert scores["fusion"]["auroc"] > 0.5
