# Project progress snapshot

The team paused computation on October 19, 2026. This page records the completed work without treating interrupted runs as final results. The [proposal](final-project-proposal.pdf) and [experimental protocol](experimental-protocol.md) describe the full study.

## Completed

- **Classifier:** A CIFAR-adapted ResNet-18 completed 20 training epochs on the 45,000-image training split. The selected epoch-20 checkpoint reached **92.00% accuracy on the 5,000-image validation split**. The official test accuracy has not been measured in a full evaluation.
- **Detector:** Logistic fusion was fitted with 5,000 clean and 5,000 attacked training images; both fusion and the maximum-disagreement rule were calibrated on 5,000 clean validation images. Each rule flagged **4.98%** of clean validation images against the 5% target.
- **Purifier:** The three-term model completed **2 of 10 planned epochs**. Its best checkpoint is from epoch 2, with validation loss **0.9177**. This loss is an optimization criterion, not an accuracy or robustness result. The no-classification-loss ablation was interrupted before its first completed epoch, so it has no selected checkpoint.
- **Squeezer ablation:** A fixed, seeded 1,000-image test subset was evaluated under PGD at 4/255 and 8/255. Each feature subset was refitted and calibrated separately. See the measured [JSON](../results/preliminary/squeezer-ablations.json).
- **Software verification:** Eight protocol tests passed, and an end-to-end integration check ran on a small validation subset. Its metrics are excluded from this report.

## Preliminary detector results

These measurements come from the **1,000-image squeezer-ablation subset**, not the complete 10,000-image test set. Detection rates use the thresholds calibrated to at most 5% false positives on clean validation data.

| Features | PGD 4/255 fusion AUROC | PGD 8/255 fusion AUROC | PGD 8/255 fusion detection rate |
|---|---:|---:|---:|
| All three squeezers | 0.863 | 0.774 | 28.4% |
| Without 4-bit quantization | 0.858 | 0.765 | 26.6% |
| Without 3 × 3 median | 0.845 | 0.780 | 36.0% |
| Without JPEG quality 50 | 0.782 | 0.657 | 13.9% |

On this subset, the original max rule had AUROC **0.880** at PGD 4/255 and **0.797** at PGD 8/255, versus **0.863** and **0.774** for learned fusion. Thus the available detection evidence does not show a fusion advantage. The higher 8/255 detection rate without median filtering despite similar AUROC also shows why detection rate and AUROC should be reported separately.

## Pending experiments

The planned 10-epoch purifier training, full 10,000-image clean/FGSM/PGD accuracy and latency comparison, C&W L2 subset, exact adaptive PGD check, and purifier-loss ablation remain incomplete. No final robustness or end-to-end accuracy claim is made. The current model weights are a progress checkpoint for resuming work.

## Reproducibility

The classifier and epoch-2 purifier weights are available in the [preliminary GitHub release](https://github.com/yyifan-Onyen/adversarial-image-defense/releases/tag/v0.1.0-preliminary). Detector parameters and numeric records are in [`results/preliminary/`](../results/preliminary/). The pinned University of Toronto CIFAR-10 mirror is documented in the [README](../README.md); all training and calibration used seed `2026` on an Apple M5 Pro through PyTorch MPS.
