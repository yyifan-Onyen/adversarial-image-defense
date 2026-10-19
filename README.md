# Learned Fusion Detection and Selective Purification for Adversarial Images

An empirical study of input-side defenses for a clean-trained CIFAR-10 classifier. The project asks whether combining feature-squeezing signals improves attack detection, and whether routing only flagged images through a purifier improves classification outcomes at an acceptable runtime cost.

This repository is the workspace for Team 5's CSCI 8000 final project at the University of Georgia. The [final project proposal](docs/final-project-proposal.pdf) defines the agreed scope and evaluation plan.

## Research plan

- **Classifier:** Train a ResNet-18 on CIFAR-10, then freeze it for all defense comparisons.
- **Detection:** Measure prediction changes after 4-bit color quantization, 3 × 3 median smoothing, and JPEG compression at quality 50. Compare a logistic-regression fusion score with the original maximum-disagreement rule. Calibrate both thresholds to a 5% false-positive rate on clean validation images.
- **Purification:** Train a small convolutional autoencoder on adversarial-to-clean pairs using reconstruction, classification, and clean-image identity losses. Apply it only to inputs flagged by the detector.
- **Matched baselines:** Compare no defense, the maximum-disagreement detector routing to the same purifier, always-on purification, and learned routing.
- **Evaluation:** Report detector AUROC and detection rate, clean and attacked classification accuracy, and end-to-end latency. Evaluate FGSM and PGD at both 4/255 and 8/255; assess C&W L2 and one adaptive attack on a fixed, seeded subset. Report transfer and adaptive results separately.

The official CIFAR-10 test set is reserved for final evaluation. Experiments should record data splits, random seeds, attack settings, hardware, and batch sizes. Results will be added only after they are measured and verified.

## Status

The final proposal is archived here. Implementation and experimental results are pending.

## Team

Ruqing Liu, Yifan Zhou, and Xinliang Li.

## License

No license has been selected. Public visibility does not grant permission to reuse the code or project materials.
