"""Exact inference paths and a differentiable adaptive-attack surrogate."""

import numpy as np
import torch

from .detector import Detector
from .squeezing import feature_tensor

VARIANTS = ("none", "max", "always", "fusion")
SQUEEZING_VARIANTS = ("max", "fusion")


class DefensePipeline:
    def __init__(self, classifier, purifier, detector: Detector | None, epsad=None):
        self.classifier = classifier.eval()
        self.purifier = purifier.eval()
        self.detector = detector
        self.epsad = epsad.eval() if epsad is not None else None

    @property
    def variants(self) -> tuple[str, ...]:
        return VARIANTS + (("epsad",) if self.epsad is not None else ())

    @property
    def routed_variants(self) -> tuple[str, ...]:
        return SQUEEZING_VARIANTS + (("epsad",) if self.epsad is not None else ())

    @torch.no_grad()
    def infer(self, x: torch.Tensor, variant: str):
        """Return logits, detection flags, and detector scores for one batch."""
        if variant not in self.variants:
            raise ValueError(f"unknown variant: {variant}")
        empty_flags = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        if variant == "none":
            return self.classifier(x), empty_flags, None
        if variant == "always":
            return self.purified_logits(x), empty_flags, None
        logits, flags, scores = self.detect(x, variant)
        if flags.any():
            logits[flags] = self.purified_logits(x[flags]).to(logits.dtype)
        return logits, flags, scores

    @torch.no_grad()
    def detect(self, x: torch.Tensor, variant: str):
        """Routing decision only: raw-classifier logits, flags, and scores.

        ``infer`` purifies the flagged rows of the same batch; a serving loop
        may instead queue them across batches (see ``aidefense.latency``).
        """
        if variant not in self.routed_variants:
            raise ValueError(f"not a routed variant: {variant}")
        if variant == "epsad":
            flags, scores = self.epsad.flags(x)
            return self.classifier(x), flags, scores.cpu().numpy()
        original_logits, features = feature_tensor(self.classifier, x)
        features_np = features.cpu().numpy()
        scores_np = self.detector.score(features_np, variant)
        flags = torch.from_numpy(self.detector.flags(features_np, variant)).to(x.device)
        return original_logits.clone(), flags, scores_np

    @torch.no_grad()
    def purified_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Classify purified inputs; stochastic purifiers may use majority voting.

        With ``ensemble_runs > 1`` (FreqPure's evaluation protocol) the output is
        the vote count per class plus half the mean softmax probability, so the
        argmax is the majority vote with ties broken by mean confidence.
        """
        runs = getattr(getattr(self.purifier, "config", None), "ensemble_runs", 1)
        if runs == 1:
            return self.classifier(self.purifier(x))
        votes = probabilities = None
        for _ in range(runs):
            logits = self.classifier(self.purifier(x))
            one_hot = torch.nn.functional.one_hot(logits.argmax(1), logits.shape[1]).to(logits.dtype)
            votes = one_hot if votes is None else votes + one_hot
            probability = logits.softmax(1)
            probabilities = probability if probabilities is None else probabilities + probability
        return votes + 0.5 * probabilities / runs

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
