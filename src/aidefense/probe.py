"""Quick validation-split probe of a purifier before any official-test run."""

import dataclasses
import json
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .attacks import fgsm, pgd
from .data import cifar10_datasets, loader, seed_everything
from .evaluation import fixed_subset, synchronize
from .pipeline import DefensePipeline
from .training import load_classifier, load_purifier

PROBE_CONDITIONS = (("clean", None), ("fgsm", 8 / 255), ("pgd", 4 / 255), ("pgd", 8 / 255))


def run_purifier_probe(
    data_dir: str, classifier_path: str, purifier_path: str, output: str, device: torch.device,
    seed: int = 2026, batch_size: int = 64, workers: int = 0, examples: int = 256,
    adaptive_examples: int = 0, adaptive_steps: int = 100, adaptive_eot: int = 10,
    adaptive_epsilon: float = 8 / 255, adaptive_step_size: float = 0.007,
    adaptive_surrogate_steps: int = 1,
) -> dict:
    """Raw versus always-purified accuracy under transfer FGSM/PGD.

    Uses a fixed subset of the validation split only, so purifier
    hyperparameters can be chosen without touching the official test set.
    With ``adaptive_examples > 0`` the first batches (up to that many images)
    are also attacked white-box with EOT-PGD through an
    ``adaptive_surrogate_steps``-denoising-step surrogate of the same FreqPure configuration (including its reference
    source), and the candidates are scored with the full purifier.
    """
    seed_everything(seed)
    _, validation, _ = cifar10_datasets(data_dir, seed, augment=False, download=True)
    subset = fixed_subset(validation, examples, seed + 100)
    classifier = load_classifier(classifier_path, device)
    purifier = load_purifier(purifier_path, device)
    pipeline = DefensePipeline(classifier, purifier, detector=None)
    warm = next(iter(loader(subset, min(batch_size, examples), False, seed, workers)))[0].to(device)
    pipeline.purified_logits(warm[:2])
    synchronize(device)
    totals = {f"{attack}_{0 if eps is None else round(eps * 255)}": {"raw": 0, "purified": 0, "seconds": 0.0}
              for attack, eps in PROBE_CONDITIONS}
    surrogate = None
    if adaptive_examples > 0:
        from .adaptive import pgd_eot_through_purifier
        from .freqpure import FreqPure
        if not isinstance(purifier, FreqPure):
            raise ValueError("adaptive probing expects a FreqPure purifier spec")
        surrogate = FreqPure(purifier.eps_model, dataclasses.replace(
            purifier.config, num_denoising_steps=adaptive_surrogate_steps, ensemble_runs=1)).to(device)
        adaptive_key = f"adaptive_eot_pgd_{round(adaptive_epsilon * 255)}"
        totals[adaptive_key] = {"raw": 0, "purified": 0, "seconds": 0.0, "count": 0, "attack_seconds": 0.0}
    count = 0
    for x, y in tqdm(loader(subset, batch_size, False, seed, workers), desc="purifier probe"):
        x, y = x.to(device), y.to(device)
        count += len(y)
        for (attack, eps), key in zip(PROBE_CONDITIONS, totals):
            if attack == "clean":
                inputs = x
            elif attack == "fgsm":
                inputs = fgsm(classifier, x, y, eps)
            else:
                inputs = pgd(classifier, x, y, eps, steps=20, step_size=eps / 8)
            with torch.no_grad():
                totals[key]["raw"] += classifier(inputs).argmax(1).eq(y).sum().item()
            synchronize(device)
            start = time.perf_counter()
            logits = pipeline.purified_logits(inputs)
            synchronize(device)
            totals[key]["seconds"] += time.perf_counter() - start
            totals[key]["purified"] += logits.argmax(1).eq(y).sum().item()
        if surrogate is not None and totals[adaptive_key]["count"] < adaptive_examples:
            entry = totals[adaptive_key]
            take = min(len(y), adaptive_examples - entry["count"])
            xa, ya = x[:take], y[:take]
            synchronize(device)
            start = time.perf_counter()
            adv = pgd_eot_through_purifier(classifier, surrogate, xa, ya, adaptive_epsilon,
                                           adaptive_steps, adaptive_step_size, adaptive_eot)
            synchronize(device)
            entry["attack_seconds"] += time.perf_counter() - start
            with torch.no_grad():
                entry["raw"] += classifier(adv).argmax(1).eq(ya).sum().item()
                entry["purified"] += pipeline.purified_logits(adv).argmax(1).eq(ya).sum().item()
            entry["count"] += take
    report = {
        "split": "validation",
        "examples": count,
        "subset_seed": seed + 100,
        "seed": seed,
        "batch_size": batch_size,
        "device": str(device),
        "purifier": getattr(purifier, "spec", {"checkpoint": str(purifier_path)}),
        "attacks": "transfer FGSM / 20-step PGD against the unprotected classifier",
        "unet_calls_per_purified_image": (purifier.unet_calls_per_image()
                                          if hasattr(purifier, "unet_calls_per_image") else None),
        "conditions": {
            key: {"raw_accuracy": value["raw"] / count,
                  "purified_accuracy": value["purified"] / count,
                  "purify_and_classify_ms_per_image": 1000 * value["seconds"] / count}
            for key, value in totals.items() if "count" not in value
        },
    }
    if surrogate is not None:
        entry = totals[adaptive_key]
        report["conditions"][adaptive_key] = {
            "examples": entry["count"],
            "raw_accuracy": entry["raw"] / entry["count"],
            "purified_accuracy": entry["purified"] / entry["count"],
            "attack": (f"white-box EOT-PGD, L-inf {round(adaptive_epsilon * 255)}/255, {adaptive_steps} steps, "
                       f"step {adaptive_step_size}, EOT {adaptive_eot}, {adaptive_surrogate_steps}-step "
                       "surrogate with the same reference source; evaluated with the full purifier"),
            "surrogate_denoising_steps": adaptive_surrogate_steps,
            "attack_seconds_per_image": entry["attack_seconds"] / entry["count"],
        }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report
