"""White-box adaptive attacks on the routed EPS-AD / feature-squeezing + FreqPure defenses.

Threat model: the attacker knows every component (classifier, squeezers,
fusion weights and thresholds, EPS-AD diffusion model, deep kernel, reference
set and threshold, FreqPure and its hyperparameters) and has an L-infinity
budget. Several complementary attacks are run on every image; an image counts
as broken for a defense variant if any attack's candidate is misclassified by
the exact, stochastic pipeline (worst case over attacks, per image).

* ``evade_<detector>``: Selective PGD (Bryniarski et al., ICLR 2022, "Evading
  adversarial example detectors using orthogonal projected gradient descent")
  against one detector (``epsad``, ``max`` or ``fusion``). Each step follows
  the classifier loss while the image is still classified correctly, and
  otherwise follows the negative detector score, so the candidate is both
  misclassified by the raw classifier and routed around the purifier. The
  EPS-AD score is differentiated exactly through the diffusion model (its
  random perturbations resampled every step, i.e. EOT); the squeezing scores
  use BPDA (exact forward, identity gradient through the squeezers).
* ``purify``: PGD with EOT through a differentiable FreqPure surrogate with
  ``surrogate_steps`` denoising steps and the deployed reference mode, full
  100-step purification for evaluation (Lee & Kim, ICCV 2023). It ignores the
  detectors: flagged images are purified, so the candidate must survive
  purification. A one-step surrogate markedly overstates robustness (validation:
  84.0% vs 31.6% purified accuracy with a ten-step surrogate), so the default
  is ten steps.
"""

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .data import cifar10_datasets, loader, seed_everything
from .detector import Detector
from .evaluation import evaluate_batch, fixed_subset, synchronize
from .freqpure import FreqPure
from .pipeline import DefensePipeline
from .squeezing import feature_tensor
from .training import load_classifier, load_purifier


def _project(adv, x, epsilon):
    return torch.min(torch.max(adv, x - epsilon), x + epsilon).clamp(0, 1)


def selective_pgd_evasion(classifier, epsad, x, y, epsilon, steps, step_size, random_start=True):
    """Return candidates and whether each was seen misclassified and undetected."""
    adv = x.clone()
    if random_start:
        adv = _project(adv + torch.empty_like(adv).uniform_(-epsilon, epsilon), x, epsilon)
    best = x.clone()
    found = torch.zeros(len(x), dtype=torch.bool, device=x.device)
    for _ in range(steps):
        adv = adv.detach().requires_grad_(True)
        with torch.enable_grad():
            logits = classifier(adv)
            ce_grad = torch.autograd.grad(F.cross_entropy(logits, y, reduction="sum"), adv)[0]
        fooled = logits.detach().argmax(1).ne(y)
        if hasattr(epsad, "score_and_input_gradient"):
            score, score_grad = epsad.score_and_input_gradient(adv.detach())
        else:
            with torch.enable_grad():
                score_var = epsad.differentiable_score(adv)
                score_grad = torch.autograd.grad(score_var.sum(), adv)[0]
            score = score_var.detach()
        # Per-sample selective objective: fool first, then hide from the detector.
        grad = torch.where(fooled[:, None, None, None], -score_grad, ce_grad)
        with torch.no_grad():
            success = fooled & (score <= epsad.threshold)
            new = success & ~found
            best[new] = adv.detach()[new]
            found |= success
            adv = _project(adv.detach() + step_size * grad.sign(), x, epsilon)
    best[~found] = adv[~found]
    return best.detach(), found


class SqueezingScore:
    """Max-rule or fusion score with BPDA gradients, for ``selective_pgd_evasion``."""

    def __init__(self, classifier, detector: Detector, rule: str):
        self.classifier, self.detector, self.rule = classifier, detector, rule
        self.threshold = detector.threshold(rule)

    def differentiable_score(self, x):
        _, features = feature_tensor(self.classifier, x, bpda=True)
        if self.rule == "max":
            return features[:, :3].max(dim=1).values
        coefficients = torch.as_tensor(self.detector.coefficients, dtype=x.dtype, device=x.device)
        return features @ coefficients + self.detector.intercept


def pgd_eot_through_purifier(classifier, surrogate, x, y, epsilon, steps, step_size, eot, random_start=True):
    adv = x.clone()
    if random_start:
        adv = _project(adv + torch.empty_like(adv).uniform_(-epsilon, epsilon), x, epsilon)
    for _ in range(steps):
        adv = adv.detach().requires_grad_(True)
        grad = torch.zeros_like(adv)
        with torch.enable_grad():
            for _ in range(eot):
                loss = F.cross_entropy(classifier(surrogate(adv)), y)
                grad += torch.autograd.grad(loss, adv)[0]
        adv = _project(adv.detach() + step_size * grad.sign(), x, epsilon)
    return adv.detach()


def run_adaptive_evaluation(
    data_dir: str, classifier_path: str, purifier_path: str, detector_path: str, epsad_path: str,
    output: str, device: torch.device, seed: int = 2026, batch_size: int = 50, workers: int = 0,
    split: str = "validation", examples: int = 256, epsilon: float = 8 / 255,
    steps: int = 100, step_size: float = 0.007, eot: int = 10, surrogate_steps: int = 10,
) -> dict:
    from .epsad import load_epsad

    seed_everything(seed)
    if split not in ("validation", "test"):
        raise ValueError("split must be validation or test")
    _, validation, test = cifar10_datasets(data_dir, seed, augment=False, download=True)
    dataset = fixed_subset(validation if split == "validation" else test, examples, seed + 300)
    classifier = load_classifier(classifier_path, device)
    purifier = load_purifier(purifier_path, device)
    if not isinstance(purifier, FreqPure):
        raise ValueError("adaptive evaluation expects a FreqPure purifier spec")
    epsad = load_epsad(epsad_path, device)
    pipeline = DefensePipeline(classifier, purifier, Detector.load(detector_path), epsad)
    surrogate = FreqPure(purifier.eps_model, dataclasses.replace(
        purifier.config, num_denoising_steps=surrogate_steps, ensemble_runs=1)).to(device)
    variants = pipeline.variants
    detectors = {"epsad": epsad, "max": SqueezingScore(classifier, pipeline.detector, "max"),
                 "fusion": SqueezingScore(classifier, pipeline.detector, "fusion")}
    attacks = tuple(f"evade_{name}" for name in detectors) + ("purify",)
    conditions = ("clean",) + attacks
    predictions = {c: {v: [] for v in variants} for c in conditions}
    flags = {c: {v: [] for v in pipeline.routed_variants} for c in conditions}
    labels = []
    evade_found = {name: [] for name in detectors}
    seconds = {attack: 0.0 for attack in attacks}
    for x, y in tqdm(loader(dataset, batch_size, False, seed, workers), desc="adaptive attacks"):
        x, y = x.to(device), y.to(device)
        labels.append(y.cpu().numpy())
        candidates = {"clean": x}
        for name, detector in detectors.items():
            synchronize(device)
            t0 = time.perf_counter()
            candidates[f"evade_{name}"], found = selective_pgd_evasion(
                classifier, detector, x, y, epsilon, steps, step_size)
            synchronize(device)
            seconds[f"evade_{name}"] += time.perf_counter() - t0
            evade_found[name].append(found.cpu().numpy())
        synchronize(device)
        t0 = time.perf_counter()
        candidates["purify"] = pgd_eot_through_purifier(
            classifier, surrogate, x, y, epsilon, steps, step_size, eot)
        synchronize(device)
        seconds["purify"] += time.perf_counter() - t0
        for condition in conditions:
            batch = evaluate_batch(pipeline, candidates[condition], y, device)
            for variant in variants:
                predictions[condition][variant].append(batch[variant]["predictions"])
            for variant in pipeline.routed_variants:
                flags[condition][variant].append(batch[variant]["flags"])
    labels = np.concatenate(labels)
    predictions = {c: {v: np.concatenate(p) for v, p in d.items()} for c, d in predictions.items()}
    flags = {c: {v: np.concatenate(f) for v, f in d.items()} for c, d in flags.items()}
    reference_correct = predictions["clean"]["none"] == labels
    results = {}
    for variant in variants:
        correct = {c: predictions[c][variant] == labels for c in conditions}
        worst = np.logical_and.reduce([correct[a] for a in attacks])
        broken = ~worst
        variant_clean_correct = correct["clean"]
        results[variant] = {
            "clean_accuracy": float(correct["clean"].mean()),
            **{f"{attack}_accuracy": float(correct[attack].mean()) for attack in attacks},
            "worst_case_accuracy": float(worst.mean()),
            "worst_case_attack_success_rate_on_undefended_clean_correct":
                float((reference_correct & broken).sum() / reference_correct.sum()),
            "worst_case_attack_success_rate_on_variant_clean_correct":
                float((variant_clean_correct & broken).sum() / max(variant_clean_correct.sum(), 1)),
        }
    report = {
        "protocol": {
            "split": split, "examples": int(len(labels)), "subset_seed": seed + 300, "seed": seed,
            "threat_model": "white-box, L-infinity", "epsilon_linf": epsilon, "steps": steps,
            "step_size": step_size, "eot_samples_purify_attack": eot,
            "freqpure_surrogate_denoising_steps": surrogate_steps,
            "attacks": list(attacks), "worst_case": "per image, over all attacks",
            "evaluation": "exact stochastic pipeline, one run per candidate",
            "purifier": getattr(purifier, "spec", None), "epsad_detector": getattr(epsad, "spec", None),
            "attack_seconds_per_image": {k: v / len(labels) for k, v in seconds.items()},
        },
        "flag_rate": {c: {v: float(f.mean()) for v, f in d.items()} for c, d in flags.items()},
        "evade_found_misclassified_and_undetected_during_search": {
            name: float(np.concatenate(found).mean()) for name, found in evade_found.items()},
        "variants": results,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    samples = {"labels": labels,
               **{f"{c}/{v}/flags": flags[c][v] for c in conditions for v in pipeline.routed_variants},
               **{f"{c}/{v}/predictions": predictions[c][v] for c in conditions for v in variants}}
    np.savez_compressed(path.with_suffix(".samples.npz"), **samples)
    return report
