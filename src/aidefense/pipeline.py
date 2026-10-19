"""Exact inference paths and a differentiable adaptive-attack surrogate."""

import numpy as np
import torch

from .detector import Detector
from .squeezing import feature_tensor

VARIANTS = ("none", "max", "always", "fusion")


class DefensePipeline:
    def __init__(self, classifier, purifier, detector: Detector):
        self.classifier = classifier.eval()
        self.purifier = purifier.eval()
        self.detector = detector

    @torch.no_grad()
    def infer(self, x: torch.Tensor, variant: str):
        """Return logits, detection flags, and detector scores for one batch."""
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant: {variant}")
        empty_flags = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        if variant == "none":
            return self.classifier(x), empty_flags, None
        if variant == "always":
            return self.classifier(self.purifier(x)), empty_flags, None
        original_logits, features = feature_tensor(self.classifier, x)
        scores_np = self.detector.score(features.cpu().numpy(), variant)
        flags = torch.from_numpy(self.detector.flags(features.cpu().numpy(), variant)).to(x.device)
        logits = original_logits.clone()
        if flags.any():
            logits[flags] = self.classifier(self.purifier(x[flags]))
        return logits, flags, scores_np

    def surrogate_logits(self, x: torch.Tensor, rule: str = "fusion", temperature: float = 0.1):
        """Soft routing with BPDA squeezers for adaptive PGD optimization.

        Attack success must always be checked with ``infer`` on this exact
        pipeline; the surrogate is used only to construct candidate inputs.
        """
        if rule not in ("fusion", "max") or temperature <= 0:
            raise ValueError("rule must be fusion/max and temperature positive")
        _, features = feature_tensor(self.classifier, x, bpda=True)
        if rule == "fusion":
            coefficients = torch.as_tensor(self.detector.coefficients, dtype=x.dtype, device=x.device)
            scores = features @ coefficients + self.detector.intercept
        else:
            scores = features[:, :3].max(dim=1).values
        gate = torch.sigmoid((scores - self.detector.threshold(rule)) / temperature)
        purified = self.purifier(x)
        routed = x + gate[:, None, None, None] * (purified - x)
        return self.classifier(routed)
