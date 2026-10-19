"""Matched clean, transfer, C&W, and adaptive defense evaluation."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import Subset
from tqdm.auto import tqdm

from .attacks import cw_l2, fgsm, pgd
from .data import cifar10_datasets, loader, seed_everything
from .detector import Detector
from .pipeline import DefensePipeline, VARIANTS
from .training import load_classifier, load_purifier


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def fixed_subset(dataset, count: int, seed: int):
    if count <= 0 or count > len(dataset):
        raise ValueError("subset count must be in the dataset range")
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:count].tolist()
    return Subset(dataset, indices)


def evaluate_batch(pipeline: DefensePipeline, x: torch.Tensor, y: torch.Tensor, device: torch.device):
    results = {}
    for variant in VARIANTS:
        synchronize(device)
        start = time.perf_counter()
        logits, flags, scores = pipeline.infer(x, variant)
        synchronize(device)
        duration = time.perf_counter() - start
        results[variant] = {
            "correct": logits.argmax(1).eq(y).sum().item(),
            "predictions": logits.argmax(1).detach().cpu().numpy(),
            "flagged": flags.sum().item(),
            "scores": scores,
            "seconds": duration,
        }
    return results


def evaluate_condition(
    pipeline: DefensePipeline, dataset, device: torch.device, seed: int, batch_size: int,
    attack: str, epsilon: float | None, workers: int,
    cw_steps: int, cw_binary_search_steps: int, adaptive_steps: int,
) -> dict:
    counts = {variant: {"correct": 0, "flagged": 0, "seconds": 0.0, "scores": []} for variant in VARIANTS}
    total = 0
    attack_successes = 0
    attack_attempts = 0
    adaptive_exact_successes = 0
    adaptive_exact_attempts = 0
    l2_distortions = []
    for x, y in tqdm(loader(dataset, batch_size, False, seed, workers), desc=f"evaluate {attack} {epsilon}"):
        x, y = x.to(device), y.to(device)
        if attack == "clean":
            adversarial = x
        elif attack == "fgsm":
            adversarial = fgsm(pipeline.classifier, x, y, float(epsilon))
        elif attack == "pgd":
            adversarial = pgd(pipeline.classifier, x, y, float(epsilon), steps=20, step_size=float(epsilon) / 8)
        elif attack == "cw_l2":
            adversarial, success = cw_l2(pipeline.classifier, x, y, steps=cw_steps,
                                         binary_search_steps=cw_binary_search_steps)
            with torch.no_grad():
                success &= pipeline.classifier(x).argmax(1).eq(y)
            distortions = (adversarial - x).flatten(1).norm(p=2, dim=1)
            l2_distortions.extend(distortions[success].detach().cpu().tolist())
        elif attack == "adaptive_pgd":
            adversarial = pgd(
                lambda z: pipeline.surrogate_logits(z, "fusion"), x, y, float(epsilon),
                steps=adaptive_steps, step_size=float(epsilon) / 8,
            )
        else:
            raise ValueError(f"unknown attack: {attack}")
        batch = evaluate_batch(pipeline, adversarial, y, device)
        if attack != "clean":
            with torch.no_grad():
                initially_correct = pipeline.classifier(x).argmax(1).eq(y)
            attacked_wrong = torch.as_tensor(batch["none"]["predictions"], device=device).ne(y)
            attack_successes += (initially_correct & attacked_wrong).sum().item()
            attack_attempts += initially_correct.sum().item()
            if attack == "adaptive_pgd":
                clean_fusion_logits, _, _ = pipeline.infer(x, "fusion")
                fusion_initially_correct = clean_fusion_logits.argmax(1).eq(y)
                fusion_attacked_wrong = torch.as_tensor(batch["fusion"]["predictions"], device=device).ne(y)
                adaptive_exact_successes += (fusion_initially_correct & fusion_attacked_wrong).sum().item()
                adaptive_exact_attempts += fusion_initially_correct.sum().item()
        total += len(y)
        for variant, result in batch.items():
            entry = counts[variant]
            entry["correct"] += result["correct"]
            entry["flagged"] += result["flagged"]
            entry["seconds"] += result["seconds"]
            if result["scores"] is not None:
                entry["scores"].extend(result["scores"].tolist())
    summary = {
        "attack": attack,
        "epsilon_linf": epsilon,
        "examples": total,
        "variants": {
            variant: {
                "accuracy": counts[variant]["correct"] / total,
                "flagged_fraction": counts[variant]["flagged"] / total if variant in ("max", "fusion") else None,
                "latency_ms_per_image": 1000 * counts[variant]["seconds"] / total,
            } for variant in VARIANTS
        },
        "scores": {variant: counts[variant]["scores"] for variant in ("max", "fusion")},
    }
    if attack != "clean":
        summary["baseline_attack_success_rate_on_initially_correct"] = (
            attack_successes / attack_attempts if attack_attempts else None
        )
        summary["initially_correct_examples"] = attack_attempts
    if attack == "cw_l2":
        summary["successful_cw_examples"] = len(l2_distortions)
        summary["successful_cw_l2_mean"] = float(np.mean(l2_distortions)) if l2_distortions else None
        summary["successful_cw_l2_median"] = float(np.median(l2_distortions)) if l2_distortions else None
    if attack == "adaptive_pgd":
        summary["exact_fusion_attack_success_rate_on_initially_correct"] = (
            adaptive_exact_successes / adaptive_exact_attempts if adaptive_exact_attempts else None
        )
        summary["exact_fusion_initially_correct_examples"] = adaptive_exact_attempts
    return summary


def run_evaluation(
    data_dir: str, classifier_path: str, purifier_path: str, detector_path: str,
    output: str, device: torch.device, seed: int = 2026, batch_size: int = 64,
    workers: int = 0, test_examples: int = 10000, expensive_subset: int = 1000,
    cw_steps: int = 100, cw_binary_search_steps: int = 3, adaptive_steps: int = 30,
) -> dict:
    seed_everything(seed)
    _, _, test = cifar10_datasets(data_dir, seed, augment=False, download=True)
    classifier = load_classifier(classifier_path, device)
    purifier = load_purifier(purifier_path, device)
    detector = Detector.load(detector_path)
    pipeline = DefensePipeline(classifier, purifier, detector)
    test = fixed_subset(test, test_examples, seed) if test_examples < len(test) else test
    expensive = fixed_subset(test, min(expensive_subset, len(test)), seed + 1)
    conditions = [("clean", None, test)]
    conditions += [(attack, epsilon, test) for attack in ("fgsm", "pgd") for epsilon in (4 / 255, 8 / 255)]
    conditions += [("cw_l2", None, expensive), ("adaptive_pgd", 8 / 255, expensive)]
    raw = [evaluate_condition(pipeline, subset, device, seed, batch_size, attack, epsilon, workers,
                              cw_steps, cw_binary_search_steps, adaptive_steps)
           for attack, epsilon, subset in conditions]
    clean_scores = raw[0]["scores"]
    for condition in raw[1:]:
        for variant in ("max", "fusion"):
            if condition["examples"] == raw[0]["examples"]:
                clean = np.asarray(clean_scores[variant])
                attacked = np.asarray(condition["scores"][variant])
                condition["variants"][variant]["detector_auroc"] = float(roc_auc_score(
                    np.r_[np.zeros(len(clean)), np.ones(len(attacked))], np.r_[clean, attacked]))
                condition["variants"][variant]["detection_rate_at_calibrated_threshold"] = (
                    condition["variants"][variant]["flagged_fraction"])
    for condition in raw:
        condition.pop("scores")
    report = {
        "protocol": {
            "dataset": "CIFAR-10 official test set",
            "seed": seed,
            "test_examples": test_examples,
            "expensive_subset": min(expensive_subset, len(test)),
            "batch_size": batch_size,
            "device": str(device),
            "detector_validation_target_fpr": detector.target_fpr,
            "cw_steps": cw_steps,
            "cw_binary_search_steps": cw_binary_search_steps,
            "cw_learning_rate": 0.01,
            "cw_initial_const": 0.1,
            "cw_confidence": 0.0,
            "adaptive_steps": adaptive_steps,
            "transfer_attacks_target": "unprotected classifier",
            "adaptive_candidate_target": "differentiable fusion-routing surrogate with BPDA",
            "adaptive_success_checked_on": "exact fusion pipeline",
        },
        "conditions": raw,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report
