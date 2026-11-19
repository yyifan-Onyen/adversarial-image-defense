"""FreqPure diffusion purification (Pei et al., ICML 2025).

Reimplemented from the paper and the authors' released code
(https://github.com/GaozhengPei/FreqPure, ``purification.py``):

* Inputs are mapped to [-1, 1], diffused to ``max_timestep`` with the
  linear DDPM schedule (beta 1e-4 -> 2e-2 over 1000 steps), and denoised with
  ``num_denoising_steps`` evenly spaced DDPM (eta=1) or DDIM (eta=0) steps.
* At every reverse step the predicted clean image x0|t is corrected in the
  frequency domain: its low-frequency amplitude spectrum (radius
  ``amplitude_cut``) is replaced by the reference's, and its low-frequency
  phase (radius ``phase_cut``) is projected into ``[ref - delta, ref + delta]``.
* As in the released code, the reference is not the raw input but a one-step
  denoised estimate of the input diffused to ``reference_timestep``. The
  released code redraws it at every reverse step (``reference="per_step"``).
  Two cheaper variants are provided for the shared-backbone ablation:
  ``"once"`` draws it a single time per purification, and ``"eps"`` averages
  the one-step estimates x0 = (x_k - sqrt(1 - abar_k) eps) / sqrt(abar_k) over
  the ``reference_eps_steps`` small noise levels k = 0..T*-1 that EPS-AD
  already evaluates, so a detector sharing the backbone gets it for free.

The released code first overwrites the low-frequency phase with the reference
phase and then clips, which makes the projection exact replacement. Following
the paper, this module projects x0|t's own phase; ``delta=0`` reproduces the
released behaviour.

The diffusion backbone is any epsilon-prediction network ``eps(x_t, t)`` that
uses the same 1000-step linear schedule; this project uses the public
``google/ddpm-cifar10-32`` UNet (Ho et al., 2020) instead of the authors'
score_sde NCSN++ checkpoint.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


REFERENCE_MODES = ("per_step", "once", "eps")


@dataclass
class FreqPureConfig:
    max_timestep: int = 1000
    num_denoising_steps: int = 100
    amplitude_cut: float = 10.0
    phase_cut: float = 10.0
    delta: float = 0.3
    reference_timestep: int = 50
    sampling: str = "ddpm"
    ensemble_runs: int = 1
    reference: str = "per_step"
    reference_eps_steps: int = 20

    def validate(self) -> None:
        if not 1 <= self.max_timestep <= 1000:
            raise ValueError("max_timestep must be in [1, 1000]")
        if not 1 <= self.num_denoising_steps <= self.max_timestep:
            raise ValueError("num_denoising_steps must be in [1, max_timestep]")
        if self.max_timestep % self.num_denoising_steps:
            raise ValueError("max_timestep must be divisible by num_denoising_steps")
        if self.amplitude_cut < 0 or self.phase_cut < 0 or self.delta < 0:
            raise ValueError("frequency cutoffs and delta must be nonnegative")
        if not 0 <= self.reference_timestep < 1000:
            raise ValueError("reference_timestep must be in [0, 999]")
        if self.sampling not in ("ddpm", "ddim"):
            raise ValueError("sampling must be ddpm or ddim")
        if self.ensemble_runs < 1:
            raise ValueError("ensemble_runs must be positive")
        if self.reference not in REFERENCE_MODES:
            raise ValueError(f"reference must be one of {REFERENCE_MODES}")
        if self.reference_eps_steps < 1:
            raise ValueError("reference_eps_steps must be positive")


def low_frequency_mask(height: int, width: int, radius: float, device=None) -> torch.Tensor:
    """Boolean mask of fftshift-ed frequencies within ``radius`` of the centre."""
    v = torch.arange(-(height // 2), height - height // 2, device=device, dtype=torch.float32)
    u = torch.arange(-(width // 2), width - width // 2, device=device, dtype=torch.float32)
    return (v[:, None].square() + u[None, :].square()).sqrt() <= radius


def spectrum_exchange(reference: torch.Tensor, estimate: torch.Tensor,
                      amplitude_cut: float, phase_cut: float, delta: float) -> torch.Tensor:
    """Batched FreqPure amplitude exchange and phase projection.

    Both inputs are images in [0, 1] with shape (B, C, H, W); the result is in
    [0, 1]. Computation follows the released code on the 0-255 scale and
    reconstructs with the modulus of the inverse FFT.
    """
    height, width = reference.shape[-2:]
    ref = torch.fft.fftshift(torch.fft.fft2(reference.clamp(0, 1) * 255), dim=(-2, -1))
    est = torch.fft.fftshift(torch.fft.fft2(estimate.clamp(0, 1) * 255), dim=(-2, -1))
    amplitude_mask = low_frequency_mask(height, width, amplitude_cut, reference.device)
    phase_mask = low_frequency_mask(height, width, phase_cut, reference.device)
    amplitude = torch.where(amplitude_mask, ref.abs(), est.abs())
    ref_phase, est_phase = ref.angle(), est.angle()
    projected = torch.minimum(torch.maximum(est_phase, ref_phase - delta), ref_phase + delta)
    phase = torch.where(phase_mask, projected, est_phase)
    spectrum = torch.polar(amplitude, phase)
    image = torch.fft.ifft2(torch.fft.ifftshift(spectrum, dim=(-2, -1))).abs()
    return (image.clamp(0, 255) / 255).to(reference.dtype)


class FreqPure(nn.Module):
    """Stochastic purifier mapping [0, 1] images to purified [0, 1] images."""

    def __init__(self, eps_model: nn.Module, config: FreqPureConfig | None = None) -> None:
        super().__init__()
        self.eps_model = eps_model
        self.config = config or FreqPureConfig()
        self.config.validate()
        betas = torch.linspace(1e-4, 2e-2, 1000, dtype=torch.float64)
        alpha_bar = torch.cumprod(1 - betas, dim=0)
        # Index t + 1; index 0 (t = -1) is the noiseless endpoint with alpha_bar = 1.
        self.register_buffer("alpha_bar", torch.cat([torch.ones(1, dtype=torch.float64), alpha_bar]).float(),
                             persistent=False)
        step = self.config.max_timestep // self.config.num_denoising_steps
        self.timesteps = [t - 1 for t in range(step, self.config.max_timestep + 1, step)]

    def _alpha(self, t: int, batch: int) -> torch.Tensor:
        return self.alpha_bar[t + 1].expand(batch).view(-1, 1, 1, 1)

    def _eps(self, x: torch.Tensor, t: int) -> torch.Tensor:
        timesteps = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        out = self.eps_model(x, timesteps)
        return out.sample if hasattr(out, "sample") else out

    def _eps_low_memory(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """Noise prediction; under autograd keep one UNet graph alive at a time."""
        if torch.is_grad_enabled() and x.requires_grad:
            return checkpoint(self._eps, x, t, use_reentrant=False)
        return self._eps(x, t)

    def _diffuse(self, x: torch.Tensor, t: int) -> torch.Tensor:
        a = self._alpha(t, x.shape[0])
        return x * a.sqrt() + torch.randn_like(x) * (1 - a).sqrt()

    def _one_step_estimate(self, x_diff: torch.Tensor, t: int) -> torch.Tensor:
        noised = self._diffuse(x_diff, t)
        a = self._alpha(t, x_diff.shape[0])
        eps = self._eps_low_memory(noised, t)
        return (noised - eps * (1 - a).sqrt()) / a.sqrt()

    def _reference(self, x_diff: torch.Tensor) -> torch.Tensor:
        return self._one_step_estimate(x_diff, self.config.reference_timestep)

    def eps_reference(self, x_diff: torch.Tensor) -> torch.Tensor:
        """Mean one-step clean estimate over the EPS-AD noise levels (in [-1, 1])."""
        steps = self.config.reference_eps_steps
        return sum(self._one_step_estimate(x_diff, k) for k in range(steps)) / steps

    def unet_calls_per_image(self) -> int:
        cfg = self.config
        extra = {"per_step": cfg.num_denoising_steps, "once": 1, "eps": cfg.reference_eps_steps}
        return cfg.num_denoising_steps + extra[cfg.reference]

    def forward(self, x: torch.Tensor, reference: torch.Tensor | None = None) -> torch.Tensor:
        """Purify ``x`` in [0, 1]; ``reference`` (in [-1, 1]) may be precomputed by a shared detector."""
        cfg = self.config
        x_diff = (x - 0.5) * 2
        if reference is None and cfg.reference == "once":
            reference = self._reference(x_diff)
        elif reference is None and cfg.reference == "eps":
            reference = self.eps_reference(x_diff)
        x_t = self._diffuse(x_diff, cfg.max_timestep - 1)
        eta = 1.0 if cfg.sampling == "ddpm" else 0.0
        batch = x.shape[0]
        previous = [-1] + self.timesteps[:-1]
        for t, t_next in zip(reversed(self.timesteps), reversed(previous)):
            a_t, a_next = self._alpha(t, batch), self._alpha(t_next, batch)
            eps = self._eps_low_memory(x_t, t)
            x0 = (x_t - eps * (1 - a_t).sqrt()) / a_t.sqrt()
            step_reference = self._reference(x_diff) if reference is None else reference
            x0 = spectrum_exchange(step_reference / 2 + 0.5, x0 / 2 + 0.5,
                                   cfg.amplitude_cut, cfg.phase_cut, cfg.delta) * 2 - 1
            c1 = eta * ((1 - a_t / a_next) * (1 - a_next) / (1 - a_t)).sqrt()
            c2 = ((1 - a_next) - c1.square()).clamp_min(0).sqrt()
            x_t = a_next.sqrt() * x0 + c1 * torch.randn_like(x_t) + c2 * eps
        return (x_t / 2 + 0.5).clamp(0, 1)


def load_freqpure(config_path: str | Path, device: torch.device) -> FreqPure:
    """Load a FreqPure purifier from a JSON spec with ``unet`` and config fields."""
    from diffusers import UNet2DModel

    config_path = Path(config_path)
    spec = json.loads(config_path.read_text())
    if spec.get("type") != "freqpure":
        raise ValueError("purifier spec must have type 'freqpure'")
    unet_dir = Path(spec["unet"])
    if not unet_dir.is_absolute():
        unet_dir = Path.cwd() / unet_dir
    unet = UNet2DModel.from_pretrained(unet_dir).eval().requires_grad_(False)
    fields = set(FreqPureConfig.__dataclass_fields__)
    config = FreqPureConfig(**{key: value for key, value in spec.items() if key in fields})
    model = FreqPure(unet, config).to(device).eval()
    model.spec = {"type": "freqpure", "unet": str(spec["unet"]), **asdict(config)}
    return model
