"""Learned fusion and original max-rule detector with matched calibration."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from .squeezing import SQUEEZER_NAMES


def threshold_at_fpr(clean_scores: np.ndarray, target_fpr: float = 0.05) -> float:
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must be between zero and one")
    if len(clean_scores) == 0:
        raise ValueError("clean_scores cannot be empty")
    # A strict > comparison makes the empirical false-positive rate at most
    # the requested level, including scores tied at the threshold.
    return float(np.quantile(np.asarray(clean_scores), 1 - target_fpr, method="higher"))


@dataclass
class Detector:
    coefficients: np.ndarray
    intercept: float
    fusion_threshold: float
    max_threshold: float
    target_fpr: float = 0.05

    @classmethod
    def fit(cls, train_features: np.ndarray, train_labels: np.ndarray, clean_validation_features: np.ndarray):
        if train_features.shape[1] != 4 or clean_validation_features.shape[1] != 4:
            raise ValueError("features must have three squeezer distances and a margin")
        estimator = LogisticRegression(max_iter=2000, random_state=0)
        estimator.fit(train_features, train_labels)
        coefficients = estimator.coef_[0].astype(float)
        intercept = float(estimator.intercept_[0])
        fusion_clean = clean_validation_features @ coefficients + intercept
        max_clean = clean_validation_features[:, :3].max(axis=1)
        return cls(
            coefficients=coefficients,
            intercept=intercept,
            fusion_threshold=threshold_at_fpr(fusion_clean),
            max_threshold=threshold_at_fpr(max_clean),
        )

    def score(self, features: np.ndarray, rule: str) -> np.ndarray:
        if rule == "fusion":
            return features @ self.coefficients + self.intercept
        if rule == "max":
            return features[:, :3].max(axis=1)
        raise ValueError(f"unknown detector rule: {rule}")

    def threshold(self, rule: str) -> float:
        if rule == "fusion":
            return self.fusion_threshold
        if rule == "max":
            return self.max_threshold
        raise ValueError(f"unknown detector rule: {rule}")

    def flags(self, features: np.ndarray, rule: str) -> np.ndarray:
        if rule not in ("fusion", "max"):
            raise ValueError(f"unknown detector rule: {rule}")
        return self.score(features, rule) > self.threshold(rule)

    def save(self, path: str | Path) -> None:
        payload = {
            "feature_names": [*SQUEEZER_NAMES, "prediction_margin"],
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
            "fusion_threshold": self.fusion_threshold,
            "max_threshold": self.max_threshold,
            "target_fpr": self.target_fpr,
        }
        Path(path).write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path):
        payload = json.loads(Path(path).read_text())
        if payload["feature_names"] != [*SQUEEZER_NAMES, "prediction_margin"]:
            raise ValueError("detector feature order is incompatible")
        return cls(
            coefficients=np.asarray(payload["coefficients"], dtype=float),
            intercept=float(payload["intercept"]),
            fusion_threshold=float(payload["fusion_threshold"]),
            max_threshold=float(payload["max_threshold"]),
            target_fpr=float(payload["target_fpr"]),
        )
