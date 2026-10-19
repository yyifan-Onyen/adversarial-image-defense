"""Build the English results report from completed, measured experiment JSON."""

import argparse
import json
from pathlib import Path


def condition(report, attack, epsilon=None):
    for entry in report["conditions"]:
        if entry["attack"] == attack and (
            (epsilon is None and entry["epsilon_linf"] is None)
            or (epsilon is not None and abs(entry["epsilon_linf"] - epsilon) < 1e-9)
        ):
            return entry
    raise ValueError(f"missing {attack} condition at {epsilon}")


def percent(value):
    return "n/a" if value is None else f"{100 * value:.1f}%"


def number(value, digits=2):
    return "n/a" if value is None else f"{value:.{digits}f}"


def build(evaluation, loss_ablation, squeezing, classifier, output):
    report = json.loads(Path(evaluation).read_text())
    no_classification = json.loads(Path(loss_ablation).read_text())
    squeeze = json.loads(Path(squeezing).read_text())
    classifier_metrics = json.loads(Path(classifier).read_text())
    protocol = report["protocol"]
    if protocol["dataset"] != "CIFAR-10 test split" or protocol["test_examples"] != 10000:
        raise ValueError("results report requires the full official test evaluation")
    if no_classification["protocol"]["dataset"] != protocol["dataset"] or no_classification["protocol"]["test_examples"] != protocol["test_examples"]:
        raise ValueError("loss ablation must use the same official test split")
    if squeeze["test_examples"] != 1000:
        raise ValueError("squeezer ablation requires its planned 1,000-image subset")

    variants = ("none", "max", "always", "fusion")
    titles = ("No defense", "Max rule + purifier", "Always purify", "Learned fusion + purifier")
    cases = (
        ("Clean", "clean", None),
        ("FGSM 4/255", "fgsm", 4 / 255),
        ("FGSM 8/255", "fgsm", 8 / 255),
        ("PGD 4/255", "pgd", 4 / 255),
        ("PGD 8/255", "pgd", 8 / 255),
        ("C&W L2*", "cw_l2", None),
        ("Adaptive PGD 8/255*", "adaptive_pgd", 8 / 255),
    )
    lines = [
        "# Experimental results",
        "",
        "This report is generated from the measured JSON outputs of the [experimental protocol](experimental-protocol.md). The [proposal](final-project-proposal.pdf) describes the research question and prior work. No smoke-test metrics appear here.",
        "",
        "## Run and controls",
        "",
        f"- Dataset: {protocol['dataset']}, {protocol['test_examples']:,} images; training data source: `{protocol['dataset_source']}`.",
        f"- Seed: `{protocol['seed']}`. Classifier: CIFAR-adapted ResNet-18, best validation accuracy {percent(classifier_metrics['best_validation_accuracy'])} after 20 epochs.",
        f"- Hardware: Apple M5 Pro (24 GB unified memory), PyTorch MPS; evaluation batch size `{protocol['batch_size']}`.",
        f"- Software: PyTorch `{protocol['versions']['torch']}`, TorchVision `{protocol['versions']['torchvision']}`, NumPy `{protocol['versions']['numpy']}`, scikit-learn `{protocol['versions']['scikit_learn']}`.",
        "- Both routing thresholds were calibrated on 5,000 clean validation images to a target false-positive rate of 5%. The measured validation FPR was 4.98% for each rule.",
        "",
        "## Classification accuracy",
        "",
        "| Input | n | " + " | ".join(titles) + " |",
        "|---|---:|" + "---:|" * len(variants),
    ]
    for label, attack, epsilon in cases:
        row = condition(report, attack, epsilon)
        lines.append(f"| {label} | {row['examples']:,} | " + " | ".join(percent(row["variants"][v]["accuracy"]) for v in variants) + " |")
    lines += [
        "",
        "*C&W L2 and adaptive PGD were evaluated on a fixed, seeded 1,000-image subset. Their accuracy rows are not directly comparable to the full-test rows by sample size. FGSM and PGD in the full-test rows target the unprotected classifier and transfer to the defended paths.*",
        "",
        "## Detection",
        "",
        "| Transfer attack | Max-rule AUROC | Fusion AUROC | Max-rule detection rate | Fusion detection rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, attack, epsilon in cases[1:5]:
        row = condition(report, attack, epsilon)
        max_results = row["variants"]["max"]
        fusion_results = row["variants"]["fusion"]
        lines.append(
            f"| {label} | {number(max_results['detector_auroc'], 3)} | {number(fusion_results['detector_auroc'], 3)} | "
            f"{percent(max_results['detection_rate_at_calibrated_threshold'])} | "
            f"{percent(fusion_results['detection_rate_at_calibrated_threshold'])} |"
        )
    lines += [
        "",
        "Detection rates are percentages of all attacked examples flagged at the validation-calibrated thresholds. AUROC compares clean test scores with scores on attacked test images.",
        "",
        "## Complete-pipeline inference latency",
        "",
        "| Path | Clean (ms/image) | PGD 8/255 (ms/image) |",
        "|---|---:|---:|",
    ]
    clean = condition(report, "clean")
    pgd8 = condition(report, "pgd", 8 / 255)
    for title, variant in zip(titles, variants):
        lines.append(f"| {title} | {number(clean['variants'][variant]['latency_ms_per_image'])} | {number(pgd8['variants'][variant]['latency_ms_per_image'])} |")
    lines += [
        "",
        "Timing includes feature squeezing, CPU/GPU transfers, detection, routing, purification, and classification, but excludes attack generation. Median filtering and JPEG are implemented on CPU, and timing depends on this machine and batch size.",
        "",
        "## C&W L2 and adaptive evaluation",
        "",
    ]
    cw = condition(report, "cw_l2")
    adaptive = condition(report, "adaptive_pgd", 8 / 255)
    lines += [
        f"- C&W-style L2 optimization: {percent(cw['baseline_attack_success_rate_on_initially_correct'])} attack success against the unprotected classifier among {cw['initially_correct_examples']} originally correct subset images. {cw['successful_cw_examples']} successful attacks had mean L2 distortion {number(cw['successful_cw_l2_mean'], 3)} and median {number(cw['successful_cw_l2_median'], 3)}. This implementation uses {protocol['cw_binary_search_steps']} constant-search rounds and {protocol['cw_steps']} optimization steps per round.",
        f"- Adaptive PGD: {percent(adaptive['exact_fusion_attack_success_rate_on_initially_correct'])} success on the **exact fusion pipeline** among {adaptive['exact_fusion_initially_correct_examples']} initially correct subset images. The candidates were generated with a soft-routing BPDA surrogate over {protocol['adaptive_steps']} PGD steps at 8/255.",
        "",
        "## Ablations",
        "",
        "### Squeezing features",
        "",
        "Each row retrains logistic fusion and recalibrates its threshold on the same clean validation features. Detection is measured on a separate fixed 1,000-image test subset.",
        "",
        "| Included features | PGD 4/255 AUROC | PGD 8/255 AUROC | PGD 8/255 detection rate |",
        "|---|---:|---:|---:|",
    ]
    labels = (("All three", "all"), ("Without quantization", "without_quantize_4bit"),
              ("Without median", "without_median_3x3"), ("Without JPEG", "without_jpeg_q50"))
    for label, key in labels:
        result = squeeze["ablations"][key]
        lines.append(f"| {label} | {number(result['pgd_4_255']['fusion']['auroc'], 3)} | {number(result['pgd_8_255']['fusion']['auroc'], 3)} | {percent(result['pgd_8_255']['fusion']['attacked_detection_rate'])} |")
    lines += [
        "",
        "### Purifier classification loss",
        "",
        "The ablation removes the classifier cross-entropy term while keeping the architecture, data split, attack settings, random seed, and training epochs matched.",
        "",
        "| Condition | Main purifier, fusion path | No-classification-loss purifier, fusion path |",
        "|---|---:|---:|",
    ]
    for label, attack, epsilon in cases[:5]:
        main = condition(report, attack, epsilon)
        ablated = condition(no_classification, attack, epsilon)
        lines.append(f"| {label} | {percent(main['variants']['fusion']['accuracy'])} | {percent(ablated['variants']['fusion']['accuracy'])} |")
    lines += [
        "",
        "## Interpretation and limits",
        "",
    ]
    max_pgd8 = pgd8["variants"]["max"]["accuracy"]
    fusion_pgd8 = pgd8["variants"]["fusion"]["accuracy"]
    comparison = "higher" if fusion_pgd8 > max_pgd8 else "lower" if fusion_pgd8 < max_pgd8 else "equal"
    lines += [
        f"At PGD 8/255, learned routing had {comparison} classification accuracy than max-rule routing ({percent(fusion_pgd8)} versus {percent(max_pgd8)}). This comparison uses the same purifier and matched validation false-positive target.",
        "",
        "These results measure one classifier, one training split, specified attack settings, and one hardware configuration. The C&W search is bounded, and the adaptive check uses one surrogate. No result establishes certified robustness or protection against every adaptive attack.",
        "",
        "## Reproduce",
        "",
        "Install the dependencies and run the commands in the [README](../README.md). The source JSON and generated figures are preserved in [`results/`](../results/). Model checkpoints are distributed separately as release assets because they are too large for the source repository.",
        "",
        "## Prior work and resources",
        "",
        "- Xu, Evans, and Qi. [Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks](https://www.ndss-symposium.org/wp-content/uploads/2018/02/ndss2018_03A-4_Xu_paper.pdf). NDSS 2018.",
        "- Carlini and Wagner. [Towards Evaluating the Robustness of Neural Networks](https://arxiv.org/abs/1608.04644). IEEE S&P 2017.",
        "- Athalye, Carlini, and Wagner. [Obfuscated Gradients Give a False Sense of Security](https://proceedings.mlr.press/v80/athalye18a.html). ICML 2018.",
        "- Tramèr et al. [On Adaptive Attacks to Adversarial Example Defenses](https://papers.neurips.cc/paper_files/paper/2020/hash/11f38f8ecd71867b42433548d1078e38-Abstract.html). NeurIPS 2020.",
        "- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) and [TorchVision ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).",
        "",
    ]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", default="runs/final/evaluation.json")
    parser.add_argument("--loss-ablation", default="runs/final/purifier_no_cls_evaluation.json")
    parser.add_argument("--squeezing", default="runs/final/squeezer_ablations.json")
    parser.add_argument("--classifier", default="checkpoints/classifier.metrics.json")
    parser.add_argument("--output", default="docs/results.md")
    args = parser.parse_args()
    build(args.evaluation, args.loss_ablation, args.squeezing, args.classifier, args.output)
