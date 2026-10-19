"""Publication-ready figures generated only from measured evaluation JSON."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .pipeline import VARIANTS

LABELS = {
    "none": "No defense",
    "max": "Max-rule + purifier",
    "always": "Always purify",
    "fusion": "Learned fusion + purifier",
}
COLORS = {"none": "#6b7280", "max": "#d97706", "always": "#7c3aed", "fusion": "#0f766e"}


def plot_results(report_path: str, output_dir: str) -> list[str]:
    report = json.loads(Path(report_path).read_text())
    conditions = report["conditions"]
    clean = next(condition for condition in conditions if condition["attack"] == "clean")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    produced = []
    for attack in ("fgsm", "pgd"):
        relevant = sorted((condition for condition in conditions if condition["attack"] == attack),
                          key=lambda condition: condition["epsilon_linf"])
        if not relevant:
            continue
        figure, axis = plt.subplots(figsize=(7.2, 4.6), layout="constrained")
        x_values = [condition["epsilon_linf"] * 255 for condition in relevant]
        for variant in VARIANTS:
            y_values = [100 * condition["variants"][variant]["accuracy"] for condition in relevant]
            axis.plot(x_values, y_values, marker="o", linewidth=2, label=LABELS[variant], color=COLORS[variant])
        axis.set(xlabel=r"$L_\infty$ budget ($\epsilon$, pixel levels / 255)",
                 ylabel="Accuracy on attacked images (%)",
                 title=f"CIFAR-10 accuracy under {attack.upper()} transfer attack")
        axis.set_xticks(x_values)
        axis.set_ylim(0, 100)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
        for extension in ("png", "pdf"):
            path = output / f"{attack}_accuracy.{extension}"
            figure.savefig(path, dpi=200)
            produced.append(str(path))
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.6), layout="constrained")
    bars = axis.bar(
        [LABELS[variant] for variant in VARIANTS],
        [clean["variants"][variant]["latency_ms_per_image"] for variant in VARIANTS],
        color=[COLORS[variant] for variant in VARIANTS],
    )
    axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    axis.set(ylabel="End-to-end inference latency (ms/image)",
             title=f"Clean-input latency, batch size {report['protocol']['batch_size']}")
    axis.tick_params(axis="x", labelrotation=15)
    axis.grid(axis="y", alpha=0.2)
    for extension in ("png", "pdf"):
        path = output / f"clean_latency.{extension}"
        figure.savefig(path, dpi=200)
        produced.append(str(path))
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.6), layout="constrained")
    for variant in VARIANTS:
        axis.scatter(100 * clean["variants"][variant]["accuracy"],
                     100 * next(condition for condition in conditions if condition["attack"] == "pgd" and
                                abs(condition["epsilon_linf"] - 8 / 255) < 1e-9)["variants"][variant]["accuracy"],
                     s=85, label=LABELS[variant], color=COLORS[variant])
    axis.set(xlabel="Clean accuracy (%)", ylabel="PGD 8/255 accuracy (%)",
             title="Accuracy trade-off on CIFAR-10")
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 100)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8)
    for extension in ("png", "pdf"):
        path = output / f"accuracy_tradeoff.{extension}"
        figure.savefig(path, dpi=200)
        produced.append(str(path))
    plt.close(figure)
    return produced
