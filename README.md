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

## Reproduce the study

Use Python 3.10 or newer. By default, TorchVision downloads CIFAR-10 into the ignored `data/` directory. If that server is slow, the pinned University of Toronto Hugging Face mirror can be downloaded with `scripts/download_cifar10_parquet.sh data` after installing `.[dev,hf-data]`; the loader automatically uses those two files and checks the train/test sizes. The default seed is `2026`; the 50,000 training images are split into 45,000 training and 5,000 validation images. The official 10,000-image test set is never used for model selection.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
adversarial-image-defense train-classifier --epochs 20
adversarial-image-defense train-purifier --classifier checkpoints/classifier.pt --epochs 10
adversarial-image-defense fit-detector --classifier checkpoints/classifier.pt
adversarial-image-defense ablate-squeezers --classifier checkpoints/classifier.pt \
  --feature-cache checkpoints/detector.features.npz
adversarial-image-defense evaluate --classifier checkpoints/classifier.pt \
  --purifier checkpoints/purifier.pt --detector checkpoints/detector.json \
  --output runs/final/evaluation.json
adversarial-image-defense plot-results --report runs/final/evaluation.json
pytest -q
```

For the purifier-loss ablation, repeat `train-purifier` with `--classification-weight 0 --output checkpoints/purifier_no_cls.pt`, then repeat `evaluate` with that purifier and a distinct output path. Compare it with the main three-term run at identical seeds, attacks, batch sizes, and hardware.

`--device auto` selects CUDA, then Apple MPS, then CPU. Training is computationally expensive; use a GPU when possible. Checkpoints, downloaded data, and experiment output are ignored by Git. The evaluation command records the measured protocol and metrics in JSON. Do not treat reduced `--max-train-batches` or `--test-examples` runs as final results.

## Status and limitations

The final proposal and an initial experimental pipeline implementation are present. Full training and evaluation are still pending; this repository does not claim experimental findings. The C&W implementation is a bounded optimization search, and the adaptive PGD check uses BPDA and soft routing to construct candidates, then measures their success on the exact pipeline. Neither establishes certified robustness. JPEG and median filtering currently run on CPU, so reported latency includes host/device transfer and depends on hardware and batch size.

Project members should verify course rules governing AI-assisted work before using this repository for a graded submission. External resources include the public CIFAR-10 dataset, the TorchVision ResNet-18 implementation, and the papers cited in the proposal.

## Team

Ruqing Liu, Yifan Zhou, and Xinliang Li.

## License

No license has been selected. Public visibility does not grant permission to reuse the code or project materials.
