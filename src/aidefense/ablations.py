"""Matched squeezer ablations at the same clean-validation false-positive rate."""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from .attacks import pgd
from .data import cifar10_datasets, loader, seed_everything
from .detector import threshold_at_fpr
from .evaluation import fixed_subset
from .squeezing import SQUEEZER_NAMES, feature_tensor
from .training import load_classifier


def fit_ablation(train_features, train_labels, validation_features, included):
    columns = [*included, 3]
    model = LogisticRegression(max_iter=2000, random_state=0)
    model.fit(train_features[:, columns], train_labels)
    fusion_threshold = threshold_at_fpr(model.decision_function(validation_features[:, columns]))
    max_threshold = threshold_at_fpr(validation_features[:, included].max(axis=1))
    return model, fusion_threshold, max_threshold


def score_ablation(model, clean_features, attacked_features, included, fusion_threshold, max_threshold):
    columns = [*included, 3]
    output = {}
    for rule in ("fusion", "max"):
        if rule == "fusion":
            clean = model.decision_function(clean_features[:, columns])
            attacked = model.decision_function(attacked_features[:, columns])
            threshold = fusion_threshold
        else:
            clean = clean_features[:, included].max(axis=1)
            attacked = attacked_features[:, included].max(axis=1)
            threshold = max_threshold
        output[rule] = {
            "test_clean_fpr": float(np.mean(clean > threshold)),
            "attacked_detection_rate": float(np.mean(attacked > threshold)),
            "auroc": float(roc_auc_score(np.r_[np.zeros(len(clean)), np.ones(len(attacked))],
                                            np.r_[clean, attacked])),
        }
    return output


def run_squeezer_ablations(
    data_dir: str, classifier_path: str, feature_cache: str, output: str,
    device: torch.device, seed: int = 2026, batch_size: int = 64,
    workers: int = 0, test_examples: int = 1000,
) -> dict:
    seed_everything(seed)
    cache = np.load(feature_cache)
    train_features = cache["train_features"]
    train_labels = cache["train_labels"]
    validation_features = cache["clean_validation_features"]
    _, _, test = cifar10_datasets(data_dir, seed, augment=False, download=True)
    test = fixed_subset(test, test_examples, seed)
    classifier = load_classifier(classifier_path, device)
    clean_features = []
    attack_features = {4 / 255: [], 8 / 255: []}
    for x, y in tqdm(loader(test, batch_size, False, seed, workers), desc="ablation test features"):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            _, features = feature_tensor(classifier, x)
        clean_features.append(features.cpu().numpy())
        for epsilon in attack_features:
            attacked = pgd(classifier, x, y, epsilon, steps=20, step_size=epsilon / 8)
            with torch.no_grad():
                _, features = feature_tensor(classifier, attacked)
            attack_features[epsilon].append(features.cpu().numpy())
    clean_features = np.concatenate(clean_features)
    attack_features = {epsilon: np.concatenate(batches) for epsilon, batches in attack_features.items()}
    results = {}
    for excluded in (None, 0, 1, 2):
        included = [index for index in range(3) if index != excluded]
        name = "all" if excluded is None else f"without_{SQUEEZER_NAMES[excluded]}"
        model, fusion_threshold, max_threshold = fit_ablation(
            train_features, train_labels, validation_features, included)
        results[name] = {
            "included_squeezers": [SQUEEZER_NAMES[index] for index in included],
            "pgd_4_255": score_ablation(model, clean_features, attack_features[4 / 255], included,
                                        fusion_threshold, max_threshold),
            "pgd_8_255": score_ablation(model, clean_features, attack_features[8 / 255], included,
                                        fusion_threshold, max_threshold),
        }
    report = {"seed": seed, "test_examples": test_examples, "target_validation_fpr": 0.05,
              "attacks": "PGD transfer attacks on the unprotected classifier", "ablations": results}
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report
