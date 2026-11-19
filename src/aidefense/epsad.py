"""EPS-AD adversarial detection (Zhang et al., ICML 2023).

Reimplemented from the paper and the authors' code
(https://github.com/ZSHsh98/EPS-AD: ``eval_epsad.py``, ``train_D.py``,
``D_net.py``, ``utils_MMD.py``):

* Expected perturbation score (EPS): for k = 1..T* (T* = 20 on CIFAR-10) the
  input in [-1, 1] is diffused to step k, the diffusion model's score
  ``-eps(x_k, k) / sqrt(1 - alpha_bar_k)`` is evaluated, and the scores are
  averaged. One noise draw per step, as in the released code.
* A deep kernel (``Discriminator_cifar`` feature extractor plus Gaussian
  kernels on deep features and on raw EPS) is trained by maximising the MMD
  test power between EPS of natural images and EPS of FGSM / FGSM-L2
  adversarial images at epsilon = 1/255.
* The detection score of a test image is the biased MMD^2 between its EPS and
  the EPS of a fixed natural reference set (500 images); larger is more
  adversarial.

Differences from the authors' setup: the diffusion backbone is the public
``google/ddpm-cifar10-32`` epsilon-prediction UNet with the identical linear
1000-step schedule, evaluated at discrete step ``k - 1`` instead of the
continuous-time score_sde NCSN++ at ``t = k / 1000``; images come from this
project's training split; the threshold is calibrated on clean validation
images to the project's target false-positive rate.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from .attacks import fgsm
from .data import cifar10_datasets, loader, seed_everything
from .detector import threshold_at_fpr
from .evaluation import fixed_subset


def load_eps_model(unet_dir: str | Path, device: torch.device) -> nn.Module:
    from diffusers import UNet2DModel

    unet_dir = Path(unet_dir)
    if not unet_dir.is_absolute():
        unet_dir = Path.cwd() / unet_dir
    return UNet2DModel.from_pretrained(unet_dir).to(device).eval().requires_grad_(False)


def ddpm_alpha_bar(device=None) -> torch.Tensor:
    betas = torch.linspace(1e-4, 2e-2, 1000, dtype=torch.float64)
    return torch.cumprod(1 - betas, dim=0).float().to(device)


def perturbed_score(eps_model: nn.Module, x: torch.Tensor, index: int, noise: torch.Tensor,
                    alpha_bar: torch.Tensor) -> torch.Tensor:
    """Score of ``x`` diffused to discrete step ``index`` with the given noise."""
    a = alpha_bar[index]
    std = (1 - a).sqrt()
    perturbed = a.sqrt() * (2 * x - 1) + std * noise
    timesteps = torch.full((x.shape[0],), index, device=x.device, dtype=torch.long)
    out = eps_model(perturbed, timesteps)
    eps = out.sample if hasattr(out, "sample") else out
    return -eps / std


def expected_perturbation_score(eps_model: nn.Module, x: torch.Tensor, steps: int = 20,
                                alpha_bar: torch.Tensor | None = None) -> torch.Tensor:
    """EPS of images ``x`` in [0, 1]; returns a tensor shaped like ``x``."""
    if steps < 1:
        raise ValueError("steps must be positive")
    alpha_bar = ddpm_alpha_bar(x.device) if alpha_bar is None else alpha_bar
    total = torch.zeros_like(x)
    for index in range(steps):
        total = total + perturbed_score(eps_model, x, index, torch.randn_like(x), alpha_bar)
    return total / steps


class KernelFeatures(nn.Module):
    """``Discriminator_cifar`` feature extractor from the EPS-AD code."""

    def __init__(self, img_size: int = 32, feature_dim: int = 300) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(32, 0.8),
        )
        self.feature = nn.Linear(32 * (img_size // 4) ** 2, feature_dim)

    def forward(self, eps_score: torch.Tensor) -> torch.Tensor:
        return self.feature(self.model(eps_score).flatten(1))


def pairwise_sq_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    distance = x.square().sum(1)[:, None] + y.square().sum(1)[None, :] - 2.0 * x @ y.T
    return distance.clamp_min(0)


class DeepKernel(nn.Module):
    """k(a, b) = (1-e) exp(-|f(a)-f(b)|^2/s0 - |a-b|^2/s) + e exp(-|a-b|^2/s)."""

    def __init__(self, img_size: int = 32, feature_dim: int = 300, sigma0: float = 15.0,
                 sigma: float = 100.0, epsilon_exponent: float = 2.0, seed: int = 999) -> None:
        super().__init__()
        self.features = KernelFeatures(img_size, feature_dim)
        rng = np.random.default_rng(seed)
        self.epsilon_raw = nn.Parameter(torch.tensor([math.log(rng.random() * 10 ** (-epsilon_exponent))]))
        self.sigma_raw = nn.Parameter(torch.tensor([math.sqrt(2 * img_size * img_size * sigma)]))
        self.sigma0_raw = nn.Parameter(torch.tensor([math.sqrt(sigma0)]))

    def hyperparameters(self):
        return torch.sigmoid(self.epsilon_raw), self.sigma_raw.square(), self.sigma0_raw.square()

    def gram(self, feat_x, feat_y, raw_x, raw_y):
        epsilon, sigma, sigma0 = self.hyperparameters()
        d_feat = pairwise_sq_dist(feat_x, feat_y)
        d_raw = pairwise_sq_dist(raw_x, raw_y)
        return (1 - epsilon) * torch.exp(-d_feat / sigma0 - d_raw / sigma) + epsilon * torch.exp(-d_raw / sigma)


def mmd_test_power(kernel: DeepKernel, clean: torch.Tensor, adversarial: torch.Tensor) -> torch.Tensor:
    """Negative MMD^2 / std (``MMDu`` + ``h1_mean_var_gram``, unbiased, 1-sample U)."""
    n = clean.shape[0]
    if adversarial.shape[0] != n:
        raise ValueError("clean and adversarial batches must have equal size")
    features = kernel.features(torch.cat([clean, adversarial]))
    raw = torch.cat([clean, adversarial]).flatten(1)
    fx, fy, rx, ry = features[:n], features[n:], raw[:n], raw[n:]
    kx, ky, kxy = kernel.gram(fx, fx, rx, rx), kernel.gram(fy, fy, ry, ry), kernel.gram(fx, fy, rx, ry)
    xx = (kx.sum() - kx.diagonal().sum()) / (n * (n - 1))
    yy = (ky.sum() - ky.diagonal().sum()) / (n * (n - 1))
    xy = (kxy.sum() - kxy.diagonal().sum()) / (n * (n - 1))
    mmd2 = xx - 2 * xy + yy
    h = kx + ky - kxy - kxy.T
    v1 = torch.dot(h.sum(1) / n, h.sum(1) / n) / n
    v2 = h.sum() / n / n
    variance = 4 * (v1 - v2 ** 2)
    return -mmd2 / torch.sqrt(variance + 1e-8)


class EPSADDetector(nn.Module):
    """Scores images by EPS-based MMD to a natural reference set."""

    def __init__(self, eps_model: nn.Module, kernel: DeepKernel, reference_eps: torch.Tensor,
                 threshold: float = float("inf"), target_fpr: float = 0.05, steps: int = 20) -> None:
        super().__init__()
        self.eps_model = eps_model
        self.kernel = kernel.eval()
        self.register_buffer("reference_eps", reference_eps)
        self.threshold = threshold
        self.target_fpr = target_fpr
        self.steps = steps
        self._reference_cache = None

    def _reference(self):
        if self._reference_cache is None or self._reference_cache[0].device != self.reference_eps.device:
            with torch.no_grad():
                feat = self.kernel.features(self.reference_eps)
                raw = self.reference_eps.flatten(1)
                self_similarity = self.kernel.gram(feat, feat, raw, raw).mean()
            self._reference_cache = (feat, raw, self_similarity)
        return self._reference_cache

    def differentiable_score_eps(self, eps_score: torch.Tensor) -> torch.Tensor:
        """Biased MMD^2 between each EPS and the reference set (``MMD_batch2``)."""
        ref_feat, ref_raw, ref_self = self._reference()
        similarity = self.kernel.gram(self.kernel.features(eps_score), ref_feat,
                                      eps_score.flatten(1), ref_raw).mean(1)
        return ref_self - 2 * similarity + 1.0

    def differentiable_score(self, x: torch.Tensor) -> torch.Tensor:
        """Detection score with gradients w.r.t. ``x``, for adaptive attacks."""
        return self.differentiable_score_eps(expected_perturbation_score(self.eps_model, x, self.steps))

    def score_and_input_gradient(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact score gradient w.r.t. ``x`` with one diffusion graph alive at a time.

        EPS is a mean of per-step scores, so d score / dx is the sum over steps
        of the per-step vector-Jacobian product with d score / d EPS. The same
        noise draws are used for the forward value and every backward pass.
        """
        alpha_bar = ddpm_alpha_bar(x.device)
        noises = [torch.randn_like(x) for _ in range(self.steps)]
        with torch.no_grad():
            eps_score = sum(perturbed_score(self.eps_model, x, k, noises[k], alpha_bar)
                            for k in range(self.steps)) / self.steps
        with torch.enable_grad():
            eps_var = eps_score.detach().requires_grad_(True)
            score = self.differentiable_score_eps(eps_var)
            upstream = torch.autograd.grad(score.sum(), eps_var)[0] / self.steps
            gradient = torch.zeros_like(x)
            for k in range(self.steps):
                x_var = x.detach().requires_grad_(True)
                term = perturbed_score(self.eps_model, x_var, k, noises[k], alpha_bar)
                gradient += torch.autograd.grad((term * upstream).sum(), x_var)[0]
        return score.detach(), gradient

    @torch.no_grad()
    def score_eps(self, eps_score: torch.Tensor) -> torch.Tensor:
        return self.differentiable_score_eps(eps_score)

    @torch.no_grad()
    def score(self, x: torch.Tensor) -> torch.Tensor:
        return self.differentiable_score(x)

    @torch.no_grad()
    def flags(self, x: torch.Tensor):
        scores = self.score(x)
        return scores > self.threshold, scores


def save_epsad(detector: EPSADDetector, path: str | Path, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kernel": detector.kernel.state_dict(), "reference_eps": detector.reference_eps.cpu(),
                "threshold": detector.threshold, "target_fpr": detector.target_fpr,
                "steps": detector.steps, "metadata": metadata}, path)


def load_epsad(path: str | Path, device: torch.device) -> EPSADDetector:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    kernel = DeepKernel(feature_dim=metadata.get("feature_dim", 300))
    kernel.load_state_dict(checkpoint["kernel"])
    kernel.requires_grad_(False)
    detector = EPSADDetector(load_eps_model(metadata["unet"], device), kernel.to(device),
                             checkpoint["reference_eps"].to(device), float(checkpoint["threshold"]),
                             float(checkpoint["target_fpr"]), int(checkpoint["steps"]))
    detector.spec = {"type": "epsad", "checkpoint": str(path), "threshold": detector.threshold,
                     "target_fpr": detector.target_fpr, **metadata}
    return detector.to(device).eval()


def fgsm_l2(model, x: torch.Tensor, y: torch.Tensor, epsilon: float) -> torch.Tensor:
    """EPS-AD's FGSM-L2: one normalised-gradient step of L2 radius eps * sqrt(d)."""
    with torch.enable_grad():
        candidate = x.detach().requires_grad_(True)
        loss = nn.functional.cross_entropy(model(candidate), y)
        grad = torch.autograd.grad(loss, candidate)[0]
    norm = grad.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)
    radius = epsilon * math.sqrt(x[0].numel())
    return (x.detach() + radius * grad / norm).clamp(0, 1).detach()


def _eps_of(eps_model, batches, steps, device, desc):
    output = []
    for x in tqdm(batches, desc=desc):
        with torch.no_grad():
            output.append(expected_perturbation_score(eps_model, x.to(device), steps).cpu())
    return torch.cat(output)


def fit_epsad(
    data_dir: str, classifier_path: str, unet_dir: str, output: str, device: torch.device,
    seed: int = 2026, batch_size: int = 200, workers: int = 0, train_examples: int = 10000,
    reference_size: int = 500, steps: int = 20, attack_epsilon: float = 1 / 255,
    epochs: int = 200, learning_rate: float = 0.002, kernel_batch: int = 500,
    max_batches_per_epoch: int = 31, target_fpr: float = 0.05,
) -> dict:
    """Compute EPS, train the deep kernel, and calibrate on clean validation images."""
    from .training import load_classifier

    seed_everything(seed)
    train, validation, _ = cifar10_datasets(data_dir, seed, augment=False, download=True)
    classifier = load_classifier(classifier_path, device)
    eps_model = load_eps_model(unet_dir, device)
    subset = fixed_subset(train, train_examples, seed + 200)
    clean_x, fgsm_x, l2_x = [], [], []
    for x, y in tqdm(loader(subset, batch_size, False, seed, workers), desc="EPS-AD attacks"):
        x, y = x.to(device), y.to(device)
        clean_x.append(x.cpu())
        fgsm_x.append(fgsm(classifier, x, y, attack_epsilon).cpu())
        l2_x.append(fgsm_l2(classifier, x, y, attack_epsilon).cpu())
    split = lambda images: torch.cat(images).split(batch_size)
    clean_eps = _eps_of(eps_model, split(clean_x), steps, device, "EPS clean")
    fgsm_eps = _eps_of(eps_model, split(fgsm_x), steps, device, "EPS FGSM")
    l2_eps = _eps_of(eps_model, split(l2_x), steps, device, "EPS FGSM-L2")

    kernel = DeepKernel().to(device)
    optimizer = torch.optim.Adam(kernel.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    half = kernel_batch // 2
    history = []
    for epoch in range(epochs):
        kernel.train()
        order = [torch.randperm(len(clean_eps), generator=generator) for _ in range(3)]
        batches = min(max_batches_per_epoch, len(clean_eps) // kernel_batch)
        for b in range(batches):
            idx = slice(b * kernel_batch, (b + 1) * kernel_batch)
            clean = clean_eps[order[0][idx]].to(device)
            adversarial = torch.cat([l2_eps[order[1][idx]][:half], fgsm_eps[order[2][idx]][:half]]).to(device)
            loss = mmd_test_power(kernel, clean, adversarial)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        epsilon, sigma, sigma0 = (value.item() for value in kernel.hyperparameters())
        history.append({"epoch": epoch + 1, "negative_test_power": loss.item(),
                        "epsilon": epsilon, "sigma": sigma, "sigma0": sigma0})
        if (epoch + 1) % 20 == 0:
            print(history[-1], flush=True)
    kernel.eval()

    reference_eps = clean_eps[torch.randperm(len(clean_eps), generator=generator)[:reference_size]]
    detector = EPSADDetector(eps_model, kernel, reference_eps.to(device), target_fpr=target_fpr, steps=steps)
    validation_scores = []
    for x, _ in tqdm(loader(validation, batch_size, False, seed, workers), desc="EPS-AD calibration"):
        validation_scores.append(detector.score(x.to(device)).cpu())
    validation_scores = torch.cat(validation_scores).numpy()
    detector.threshold = threshold_at_fpr(validation_scores, target_fpr)

    metadata = {"unet": str(unet_dir), "seed": seed, "train_examples": train_examples,
                "reference_size": reference_size, "steps": steps,
                "attack_epsilon": attack_epsilon, "kernel_training_attacks": ["FGSM", "FGSM-L2"],
                "epochs": epochs, "learning_rate": learning_rate, "kernel_batch": kernel_batch,
                "feature_dim": 300, "calibration": "clean validation split (5,000 images)"}
    save_epsad(detector, output, metadata)
    np.save(Path(output).with_suffix(".validation_scores.npy"), validation_scores)
    return {**metadata, "threshold": detector.threshold, "target_fpr": target_fpr,
            "final_kernel": history[-1], "history": history}
