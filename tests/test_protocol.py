import numpy as np
import torch
from torch import nn

from aidefense.ablations import fit_ablation, score_ablation
from aidefense.attacks import cw_l2, fgsm, pgd
from aidefense.data import split_indices
from aidefense.detector import Detector, threshold_at_fpr
from aidefense.evaluation import attack_success_rates, oracle_routing_counts
from aidefense.pipeline import DefensePipeline
from aidefense.squeezing import feature_tensor, squeeze
from aidefense.training import detector_attack_spec


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


def test_mixed_detector_attack_suite_covers_both_attacks_and_budgets():
    specs = [detector_attack_spec(index, "mixed-budgets") for index in range(4)]
    assert specs == [
        ("fgsm", 4 / 255, 1),
        ("fgsm", 8 / 255, 1),
        ("pgd", 4 / 255, 10),
        ("pgd", 8 / 255, 10),
    ]


def test_oracle_routing_counts_all_prediction_outcomes():
    labels = np.array([0, 0, 0, 0])
    raw = np.array([0, 0, 1, 1])
    purified = np.array([0, 1, 0, 1])
    assert oracle_routing_counts(raw, purified, labels) == {
        "both_correct": 1,
        "raw_only_correct": 1,
        "purified_only_correct": 1,
        "both_wrong": 1,
    }


def test_spectrum_exchange_full_and_empty_masks():
    from aidefense.freqpure import spectrum_exchange
    torch.manual_seed(0)
    reference = torch.rand(2, 3, 32, 32)
    estimate = torch.rand(2, 3, 32, 32)
    full = spectrum_exchange(reference, estimate, amplitude_cut=100, phase_cut=100, delta=0.0)
    empty = spectrum_exchange(reference, estimate, amplitude_cut=-1, phase_cut=-1, delta=0.0)
    assert torch.allclose(full, reference, atol=1e-4)
    assert torch.allclose(empty, estimate, atol=1e-4)


def test_freqpure_schedule_and_output_range():
    from aidefense.freqpure import FreqPure, FreqPureConfig

    class ZeroEps(nn.Module):
        def forward(self, x, t):
            return torch.zeros_like(x)

    model = FreqPure(ZeroEps(), FreqPureConfig(max_timestep=100, num_denoising_steps=10))
    assert model.timesteps == list(range(9, 100, 10))
    out = model(torch.rand(2, 3, 32, 32))
    assert out.shape == (2, 3, 32, 32) and out.min() >= 0 and out.max() <= 1


def test_freqpure_reference_sources_and_call_counts():
    from aidefense.freqpure import FreqPure, FreqPureConfig

    class CountingZeroEps(nn.Module):
        calls = 0

        def forward(self, x, t):
            CountingZeroEps.calls += 1
            return torch.zeros_like(x)

    x = torch.rand(2, 3, 32, 32)
    for mode, expected in (("per_step", 20), ("once", 11), ("eps", 14)):
        CountingZeroEps.calls = 0
        model = FreqPure(CountingZeroEps(), FreqPureConfig(
            max_timestep=100, num_denoising_steps=10, reference=mode, reference_eps_steps=4))
        out = model(x)
        assert CountingZeroEps.calls == expected == model.unet_calls_per_image()
        assert out.shape == x.shape and out.min() >= 0 and out.max() <= 1
    # With a zero noise prediction the EPS reference averages rescaled noisy copies of x.
    model = FreqPure(CountingZeroEps(), FreqPureConfig(reference="eps", reference_eps_steps=3))
    torch.manual_seed(0)
    reference = model.eps_reference((x - 0.5) * 2)
    assert torch.allclose(reference, (x - 0.5) * 2, atol=0.05)
    # A precomputed reference from a shared detector skips the extra UNet calls.
    CountingZeroEps.calls = 0
    model = FreqPure(CountingZeroEps(), FreqPureConfig(max_timestep=100, num_denoising_steps=10))
    model(x, reference=(x - 0.5) * 2)
    assert CountingZeroEps.calls == 10


def test_majority_vote_purified_logits():
    from types import SimpleNamespace

    class Flip(nn.Module):
        config = SimpleNamespace(ensemble_runs=3)

        def forward(self, x):
            return 1 - x

    pipeline = DefensePipeline(TinyClassifier(), Flip(), detector=None)
    x = torch.full((2, 3, 4, 4), 0.9)
    logits = pipeline.purified_logits(x)
    assert logits.argmax(1).tolist() == [1, 1]
    assert torch.allclose(logits.floor(), torch.tensor([[0.0, 3.0], [0.0, 3.0]]))


def test_attack_success_rates_use_both_denominators():
    labels = np.array([0, 1, 2, 3])
    undefended_clean = np.array([0, 1, 2, 0])  # three correct
    variant_clean = np.array([0, 1, 0, 3])     # three correct
    attacked = np.array([1, 1, 2, 0])          # wrong on 0 and 3
    rates = attack_success_rates(variant_clean, attacked, labels, undefended_clean)
    assert rates["on_undefended_clean_correct"] == 1 / 3
    assert rates["on_variant_clean_correct"] == 2 / 3


def test_eps_matches_closed_form_for_zero_noise_model():
    from aidefense.epsad import ddpm_alpha_bar, expected_perturbation_score

    class ConstantEps(nn.Module):
        def forward(self, x, t):
            return torch.ones_like(x)

    x = torch.rand(2, 3, 8, 8)
    eps = expected_perturbation_score(ConstantEps(), x, steps=3)
    alpha_bar = ddpm_alpha_bar()
    expected = -sum(1 / (1 - alpha_bar[k]).sqrt() for k in range(3)) / 3
    assert torch.allclose(eps, torch.full_like(x, expected.item()), rtol=1e-5)


def test_epsad_kernel_mmd_and_scores_separate_shifted_data():
    from aidefense.epsad import DeepKernel, EPSADDetector, mmd_test_power

    torch.manual_seed(0)
    kernel = DeepKernel(img_size=32)
    clean = torch.randn(20, 3, 32, 32)
    shifted = torch.randn(20, 3, 32, 32) + 3
    assert mmd_test_power(kernel, clean, shifted).item() < mmd_test_power(kernel, clean, torch.randn(20, 3, 32, 32)).item()
    kernel.eval()
    detector = EPSADDetector(nn.Identity(), kernel, clean)
    scores_clean = detector.score_eps(torch.randn(5, 3, 32, 32))
    scores_shifted = detector.score_eps(torch.randn(5, 3, 32, 32) + 3)
    assert scores_shifted.min() > scores_clean.max()


def test_selective_pgd_switches_to_detector_once_fooled():
    from aidefense.adaptive import selective_pgd_evasion

    class ScoreByMean:
        threshold = -0.6

        def differentiable_score(self, x):
            return -x[:, 1:].mean(dim=(1, 2, 3))

    x = torch.full((2, 3, 4, 4), 0.45)
    y = torch.tensor([1, 1])  # TinyClassifier predicts 1 while mean < 0.5
    adv, found = selective_pgd_evasion(TinyClassifier(), ScoreByMean(), x, y, epsilon=0.2,
                                       steps=10, step_size=0.05, random_start=False)
    assert (adv - x).abs().max() <= 0.2 + 1e-6
    assert found.all()


def test_torch_median_matches_scipy_reference():
    from scipy.ndimage import median_filter
    from aidefense.squeezing import median_3x3

    x = torch.rand(3, 3, 32, 32)
    expected = median_filter(x.numpy(), size=(1, 1, 3, 3), mode="nearest")
    assert torch.equal(median_3x3(x), torch.from_numpy(expected))


def test_queued_purification_matches_per_batch_routing():
    from aidefense.latency import per_batch_stream_predictions, queued_stream_predictions

    detector = Detector(np.zeros(4), 0.0, -1.0, 0.005)  # max rule flags about half
    pipeline = DefensePipeline(TinyClassifier(), Brighten(), detector)
    torch.manual_seed(0)
    x = torch.rand(23, 3, 8, 8) * torch.linspace(0.2, 1, 23)[:, None, None, None]
    per_batch, purified_a = per_batch_stream_predictions(pipeline, x, "max", batch_size=5)
    queued, purified_b = queued_stream_predictions(pipeline, x, "max", batch_size=5)
    assert torch.equal(per_batch, queued)
    assert purified_a == purified_b and 0 < purified_a < 23
    _, always_purified = per_batch_stream_predictions(pipeline, x, "always", batch_size=5)
    assert always_purified == 23


def test_squeezing_score_has_bpda_gradient_and_matches_detector():
    from aidefense.adaptive import SqueezingScore
    from aidefense.squeezing import feature_tensor

    detector = Detector(np.array([1.0, -0.5, 2.0, 0.3]), 0.1, 0.0, 0.0)
    x = torch.rand(2, 3, 32, 32, requires_grad=True)
    for rule in ("max", "fusion"):
        score = SqueezingScore(TinyClassifier(), detector, rule)
        value = score.differentiable_score(x)
        _, features = feature_tensor(TinyClassifier(), x.detach())
        assert np.allclose(value.detach().numpy(), detector.score(features.numpy(), rule), atol=1e-5)
        gradient = torch.autograd.grad(value.sum(), x)[0]
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
