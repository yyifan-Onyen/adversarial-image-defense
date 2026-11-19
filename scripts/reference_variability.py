"""How much does the FreqPure reference change between random draws?

FreqPure only uses the low-frequency amplitude and phase (radius
``amplitude_cut`` / ``phase_cut``) of its reference, a one-step denoised copy of
the input diffused to ``reference_timestep``. The released code redraws the
reference at every reverse step. This script draws it ``--draws`` times per
image and compares, separately for the low band FreqPure keeps and the high
band it discards:

* band SNR: energy of the mean spectrum / mean energy of draw-to-draw deviation;
* amplitude coefficient of variation across draws;
* circular standard deviation of the phase across draws (radians), and the
  fraction of low-band phases that move more than ``delta`` from their mean;
* for attacked inputs, the low-band shift of the mean reference caused by the
  perturbation, relative to the draw-to-draw deviation.

Validation split only. Usage:
    python scripts/reference_variability.py --examples 256 --draws 16
"""

import argparse
import json
from pathlib import Path

import torch

from aidefense.attacks import pgd
from aidefense.data import cifar10_datasets, loader, resolve_device, seed_everything
from aidefense.evaluation import fixed_subset
from aidefense.freqpure import load_freqpure, low_frequency_mask
from aidefense.training import load_classifier


def spectrum(images):
    """fftshift-ed spectrum on the 0-255 scale, as in FreqPure's exchange."""
    return torch.fft.fftshift(torch.fft.fft2(images.clamp(0, 1) * 255), dim=(-2, -1))


def band_statistics(draws, mask, delta):
    """draws: (R, B, C, H, W) complex spectra. Returns band sums over B*C*freq."""
    mean = draws.mean(0)
    deviation = draws - mean
    amplitude = draws.abs()
    unit = torch.polar(torch.ones_like(amplitude), draws.angle())
    resultant = unit.mean(0).abs().clamp(1e-12, 1)
    circular_std = (-2 * resultant.log()).sqrt()
    phase_offset = (unit * unit.mean(0).conj()).angle().abs()
    out = {}
    for name, band in (("low", mask), ("high", ~mask)):
        sel = band.expand_as(mean)
        out[name] = {
            "signal_energy": mean.abs().square()[sel].sum().item(),
            "deviation_energy": deviation.abs().square().mean(0)[sel].sum().item(),
            "amplitude_cv_sum": (amplitude.std(0) / amplitude.mean(0).clamp_min(1e-6))[sel].sum().item(),
            "phase_circular_std_sum": circular_std[sel].sum().item(),
            "phase_beyond_delta": (phase_offset > delta)[sel.expand_as(phase_offset)].float().sum().item(),
            "phase_samples": float(sel.expand_as(phase_offset).sum().item()),
            "count": float(sel.sum().item()),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--classifier", default="checkpoints/classifier.pt")
    parser.add_argument("--purifier", default="configs/freqpure.json")
    parser.add_argument("--examples", type=int, default=256)
    parser.add_argument("--draws", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="runs/reference-ablation/reference-variability.json")
    args = parser.parse_args()

    device = resolve_device(args.device)
    seed_everything(args.seed)
    _, validation, _ = cifar10_datasets(args.data_dir, args.seed, augment=False, download=True)
    subset = fixed_subset(validation, args.examples, args.seed + 100)
    classifier = load_classifier(args.classifier, device)
    purifier = load_freqpure(args.purifier, device)
    cfg = purifier.config
    mask = low_frequency_mask(32, 32, cfg.amplitude_cut, device)
    if cfg.phase_cut != cfg.amplitude_cut:
        raise ValueError("this script assumes equal amplitude and phase cutoffs")

    totals = {}
    shift = {"low": 0.0, "high": 0.0}
    for x, y in loader(subset, args.batch_size, False, args.seed, 0):
        x, y = x.to(device), y.to(device)
        adversarial = pgd(classifier, x, y, 8 / 255, steps=20, step_size=1 / 255)
        means = {}
        for condition, inputs in (("clean", x), ("pgd_8", adversarial)):
            with torch.no_grad():
                x_diff = (inputs - 0.5) * 2
                draws = torch.stack([spectrum(purifier._reference(x_diff) / 2 + 0.5)
                                     for _ in range(args.draws)])
            means[condition] = draws.mean(0)
            stats = band_statistics(draws, mask, cfg.delta)
            for band, values in stats.items():
                entry = totals.setdefault(condition, {}).setdefault(band, {})
                for key, value in values.items():
                    entry[key] = entry.get(key, 0.0) + value
        # Systematic low/high-band change of the mean reference caused by the attack.
        delta_mean = (means["pgd_8"] - means["clean"]).abs().square()
        for band, sel in (("low", mask), ("high", ~mask)):
            shift[band] += delta_mean[sel.expand_as(delta_mean)].sum().item()

    report = {
        "split": "validation", "examples": args.examples, "subset_seed": args.seed + 100,
        "draws_per_image": args.draws, "purifier": purifier.spec,
        "reference_noise_std": float((1 - purifier.alpha_bar[cfg.reference_timestep + 1]).sqrt()),
        "low_band_frequencies": int(mask.sum().item()), "total_frequencies": 32 * 32,
        "conditions": {},
    }
    for condition, bands in totals.items():
        report["conditions"][condition] = {
            band: {
                "snr_mean_over_draw_deviation": v["signal_energy"] / v["deviation_energy"],
                "share_of_draw_deviation_energy": None,
                "mean_amplitude_cv": v["amplitude_cv_sum"] / v["count"],
                "mean_phase_circular_std_rad": v["phase_circular_std_sum"] / v["count"],
                "fraction_phase_moves_beyond_delta": v["phase_beyond_delta"] / v["phase_samples"],
            } for band, v in bands.items()
        }
        total_dev = sum(v["deviation_energy"] for v in bands.values())
        for band, v in bands.items():
            report["conditions"][condition][band]["share_of_draw_deviation_energy"] = v["deviation_energy"] / total_dev
    clean_dev = totals["clean"]
    report["attack_shift_over_draw_deviation"] = {
        band: shift[band] / clean_dev[band]["deviation_energy"]
        for band in ("low", "high")
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
