"""Matched clean, transfer, C&W, and adaptive defense evaluation."""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
import sklearn
from sklearn.metrics import roc_auc_score
from torch.utils.data import Subset
from tqdm.auto import tqdm

from .attacks import cw_l2, fgsm, pgd
from .data import cifar10_datasets, loader, seed_everything
from .detector import Detector
from .pipeline import DefensePipeline
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


def oracle_routing_counts(raw_predictions, purified_predictions, labels) -> dict[str, int]:
    """Count label-aware routing outcomes for diagnosis, never deployment."""
    raw_correct = np.asarray(raw_predictions) == np.asarray(labels)
    purified_correct = np.asarray(purified_predictions) == np.asarray(labels)
    return {
        "both_correct": int(np.sum(raw_correct & purified_correct)),
        "raw_only_correct": int(np.sum(raw_correct & ~purified_correct)),
        "purified_only_correct": int(np.sum(~raw_correct & purified_correct)),
        "both_wrong": int(np.sum(~raw_correct & ~purified_correct)),
    }


def condition_key(attack: str, epsilon: float | None) -> str:
    return attack if epsilon is None else f"{attack}_{round(epsilon * 255)}_255"


def attack_success_rates(clean_predictions, attacked_predictions, labels, reference_clean_predictions):
    """ASR of one defense variant under one attack, with two denominators.

    ``on_undefended_clean_correct``: images the undefended classifier gets right
    on clean input (the same denominator for every variant).
    ``on_variant_clean_correct``: images this variant itself gets right on clean
    input.
    """
    labels = np.asarray(labels)
    attacked_wrong = np.asarray(attacked_predictions) != labels
    output = {}
    for name, reference in (("on_undefended_clean_correct", reference_clean_predictions),
                            ("on_variant_clean_correct", clean_predictions)):
        correct = np.asarray(reference) == labels
        output[name] = float(np.sum(correct & attacked_wrong) / np.sum(correct)) if correct.any() else None
    return output


def add_attack_success_rates(raw: list[dict]) -> None:
    """Attach per-variant ASR to attacked conditions evaluated on the clean inputs."""
    clean = next((condition for condition in raw if condition["attack"] == "clean"), None)
    if clean is None:
        return
    labels = clean["per_sample"]["labels"]
    clean_predictions = clean["per_sample"]["predictions"]
    for condition in raw:
        if condition is clean or condition["examples"] != clean["examples"]:
            continue
        if not np.array_equal(condition["per_sample"]["labels"], labels):
            continue
        for variant, predictions in condition["per_sample"]["predictions"].items():
            condition["variants"][variant]["attack_success_rate"] = attack_success_rates(
                clean_predictions[variant], predictions, labels, clean_predictions["none"])


def evaluate_batch(pipeline: DefensePipeline, x: torch.Tensor, y: torch.Tensor, device: torch.device):
    results = {}
    for variant in pipeline.variants:
        synchronize(device)
        start = time.perf_counter()
        logits, flags, scores = pipeline.infer(x, variant)
        synchronize(device)
        duration = time.perf_counter() - start
        results[variant] = {
            "correct": logits.argmax(1).eq(y).sum().item(),
            "predictions": logits.argmax(1).detach().cpu().numpy(),
            "flagged": flags.sum().item(),
            "flags": flags.detach().cpu().numpy(),
            "scores": scores,
            "seconds": duration,
        }
    return results


def evaluate_condition(
    pipeline: DefensePipeline, dataset, device: torch.device, seed: int, batch_size: int,
    attack: str, epsilon: float | None, workers: int,
    cw_steps: int, cw_binary_search_steps: int, adaptive_steps: int,
) -> dict:
    variants, routed = pipeline.variants, pipeline.routed_variants
    counts = {variant: {"correct": 0, "flagged": 0, "seconds": 0.0, "scores": []} for variant in variants}
    total = 0
    attack_successes = 0
    attack_attempts = 0
    adaptive_exact_successes = 0
    adaptive_exact_attempts = 0
    l2_distortions = []
    sample_labels = []
    sample_predictions = {variant: [] for variant in variants}
    sample_flags = {variant: [] for variant in routed}
    oracle_counts = {
        "both_correct": 0,
        "raw_only_correct": 0,
        "purified_only_correct": 0,
        "both_wrong": 0,
    }
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
        batch_oracle = oracle_routing_counts(
            batch["none"]["predictions"], batch["always"]["predictions"], y.detach().cpu().numpy())
        for key, value in batch_oracle.items():
            oracle_counts[key] += value
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
        sample_labels.append(y.detach().cpu().numpy())
        for variant, result in batch.items():
            sample_predictions[variant].append(result["predictions"])
            if variant in sample_flags:
                sample_flags[variant].append(result["flags"])
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
                "flagged_fraction": counts[variant]["flagged"] / total if variant in routed else None,
                "latency_ms_per_image": 1000 * counts[variant]["seconds"] / total,
            } for variant in variants
        },
        "scores": {variant: counts[variant]["scores"] for variant in routed},
        "per_sample": {
            "labels": np.concatenate(sample_labels),
            "predictions": {variant: np.concatenate(values) for variant, values in sample_predictions.items()},
            "flags": {variant: np.concatenate(values) for variant, values in sample_flags.items()},
        },
        "oracle_routing_diagnostic": {
            "accuracy": 1 - oracle_counts["both_wrong"] / total,
            "counts": oracle_counts,
            "fractions": {key: value / total for key, value in oracle_counts.items()},
            "purifier_rescue_fraction": oracle_counts["purified_only_correct"] / total,
            "purifier_harm_fraction": oracle_counts["raw_only_correct"] / total,
            "uses_ground_truth_labels": True,
            "deployable": False,
        },
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
    split: str = "test", include_expensive: bool = True, epsad_path: str | None = None,
) -> dict:
    seed_everything(seed)
    if split not in ("test", "validation"):
        raise ValueError("split must be test or validation")
    _, validation, official_test = cifar10_datasets(data_dir, seed, augment=False, download=True)
    classifier = load_classifier(classifier_path, device)
    purifier = load_purifier(purifier_path, device)
    detector = Detector.load(detector_path)
    epsad = None
    if epsad_path:
        from .epsad import load_epsad
        epsad = load_epsad(epsad_path, device)
    pipeline = DefensePipeline(classifier, purifier, detector, epsad)
    # The soft-routing surrogate would backpropagate through the full 100-step
    # FreqPure chain; FreqPure systems are attacked by ``adaptive-attack``
    # (aidefense.adaptive) instead, so only C&W runs here for them.
    in_loop_adaptive = include_expensive and getattr(purifier, "spec", {}).get("type") != "freqpure"
    test = validation if split == "validation" else official_test
    if test_examples > len(test):
        raise ValueError("test_examples exceeds the selected split size")
    test = fixed_subset(test, test_examples, seed) if test_examples < len(test) else test
    expensive = fixed_subset(test, min(expensive_subset, len(test)), seed + 1)
    conditions = [("clean", None, test)]
    conditions += [(attack, epsilon, test) for attack in ("fgsm", "pgd") for epsilon in (4 / 255, 8 / 255)]
    if include_expensive:
        conditions += [("cw_l2", None, expensive)]
    if in_loop_adaptive:
        conditions += [("adaptive_pgd", 8 / 255, expensive)]
    warm_x, _ = next(iter(loader(test, min(8, len(test)), False, seed, workers)))
    for variant in pipeline.variants:
        # Exclude one-time CUDA/cuDNN initialisation from every latency figure.
        pipeline.infer(warm_x.to(device), variant)
    synchronize(device)
    raw = [evaluate_condition(pipeline, subset, device, seed, batch_size, attack, epsilon, workers,
                              cw_steps, cw_binary_search_steps, adaptive_steps)
           for attack, epsilon, subset in conditions]
    clean_scores = raw[0]["scores"]
    for condition in raw[1:]:
        for variant in pipeline.routed_variants:
            if condition["examples"] == raw[0]["examples"]:
                clean = np.asarray(clean_scores[variant])
                attacked = np.asarray(condition["scores"][variant])
                condition["variants"][variant]["detector_auroc"] = float(roc_auc_score(
                    np.r_[np.zeros(len(clean)), np.ones(len(attacked))], np.r_[clean, attacked]))
                condition["variants"][variant]["detection_rate_at_calibrated_threshold"] = (
                    condition["variants"][variant]["flagged_fraction"])
    add_attack_success_rates(raw)
    samples = {}
    for condition in raw:
        scores = condition.pop("scores")
        per_sample = condition.pop("per_sample")
        key = condition_key(condition["attack"], condition["epsilon_linf"])
        samples[f"{key}/labels"] = per_sample["labels"]
        for variant, values in per_sample["predictions"].items():
            samples[f"{key}/{variant}/predictions"] = values
        for variant, values in per_sample["flags"].items():
            samples[f"{key}/{variant}/flags"] = values
            samples[f"{key}/{variant}/scores"] = np.asarray(scores[variant], dtype=np.float64)
    report = {
        "protocol": {
            "dataset": f"CIFAR-10 {split} split",
            "dataset_source": (
                "uoft-cs/cifar10@0b2714987fa478483af9968de7c934580d0bb9a2"
                if ((Path(data_dir) / "hf-train.parquet").exists() and
                    (Path(data_dir) / "hf-test.parquet").exists()) else "TorchVision CIFAR10 download"
            ),
            "seed": seed,
            "test_examples": test_examples,
            "expensive_subset": min(expensive_subset, len(test)),
            "expensive_subset_seed": seed + 1,
            "batch_size": batch_size,
            "device": str(device),
            "latency_warmup_excluded": True,
            "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                         "numpy": np.__version__, "scikit_learn": sklearn.__version__},
            "detector_validation_target_fpr": detector.target_fpr,
            "purifier": getattr(purifier, "spec", {"checkpoint": str(purifier_path)}),
            "epsad_detector": getattr(epsad, "spec", None),
            "cw_steps": cw_steps,
            "cw_binary_search_steps": cw_binary_search_steps,
            "cw_learning_rate": 0.01,
            "cw_initial_const": 0.1,
            "cw_confidence": 0.0,
            "adaptive_steps": adaptive_steps,
            "include_expensive_attacks": include_expensive,
            "transfer_attacks_target": "unprotected classifier",
            "adaptive_candidate_target": ("differentiable fusion-routing surrogate with BPDA"
                                          if in_loop_adaptive else "not run here; see adaptive-attack"),
            "adaptive_success_checked_on": "exact fusion pipeline" if in_loop_adaptive else None,
        },
        "conditions": raw,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(path.with_suffix(".samples.npz"), **samples)
    return report
