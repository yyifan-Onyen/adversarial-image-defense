# Experimental protocol

This document records the implementation choices behind the [final proposal](final-project-proposal.pdf) and the [results report](progress-report.md). The official CIFAR-10 test set is held out from model selection; every design choice (purifier, reference mode, surrogate strength) was made on the validation split. Smoke tests are never reported as results.

## Data and classifier

Use the 50,000-image CIFAR-10 training set with a seeded random split of 45,000 training and 5,000 validation images (`seed=2026`). Training uses random 32 × 32 crops with four pixels of padding and random horizontal flips; validation and test images use only tensor conversion to RGB values in [0, 1]. The data loader accepts TorchVision's download or the pinned University of Toronto Hugging Face mirror ([`scripts/download_cifar10_parquet.sh`](../scripts/download_cifar10_parquet.sh), SHA-256 checked); each run records its source.

Train a CIFAR-adapted ResNet-18 (3 × 3 stride-1 stem, no max pool) from scratch for 20 epochs with SGD (`lr=0.1`, momentum 0.9, weight decay `5e-4`) and cosine decay. Select the epoch with the highest validation accuracy and freeze it for every detector, purifier, attack, and comparison.

## Detectors

All detectors use a strict `score > threshold` decision calibrated on the 5,000 clean validation images to an empirical false-positive rate of at most 5%.

**Feature squeezing (weak detectors).** Evaluate the classifier on the input and on three transformed copies: 4-bit quantization per channel, 3 × 3 median filtering with edge replication, and JPEG quality 50 (libjpeg through Pillow). Each feature is the L1 distance between the original and transformed softmax vectors; the fourth is the gap between the two highest original probabilities. The *max* rule scores an image by the largest of the three distances (Xu et al., NDSS 2018). The *fusion* rule is a logistic regression on all four features, fitted on 5,000 training-split images and their attacked versions alternating FGSM and 10-step PGD at 4/255 (the proposal suite).

**EPS-AD (strong detector).** Following Zhang et al. (ICML 2023), the expected perturbation score of an image is the mean diffusion score over noise levels t = 0..19 of the `google/ddpm-cifar10-32` model (20 UNet calls). A deep kernel is trained for 200 epochs to maximize MMD test power between the EPS of 10,000 clean training images and of their FGSM / FGSM-L2 versions at 1/255; an image's score is its MMD to the EPS of 500 clean reference images.

## Purifier: FreqPure with a once-computed reference

FreqPure (ICML 2025) diffuses the input to t = 999 and runs 100 DDPM reverse steps on the same diffusion model. At each step it computes the clean estimate x̂0, replaces the amplitude of its frequencies within radius 10 of the spectrum centre with that of a reference image, and projects their phase to within ±0.3 rad of the reference phase. The reference is the input diffused to t = 50 and denoised in one step. The released code redraws it at every step (`per_step`, 200 UNet calls per image); the deployed system computes it once before the loop (`once`, 101 calls). One purification run per image (no ensemble vote).

The reference mode was chosen on the validation split by a three-way ablation (`per_step`, `once`, and `eps`, which averages the one-step estimates at the EPS-AD noise levels so they could be shared with detection), evaluated on 1,000 images under transfer attacks and 256 images under adaptive attacks with one-step and ten-step surrogates. [`scripts/reference_variability.py`](../scripts/reference_variability.py) measures the reference's draw-to-draw variation in the low band FreqPure uses (256 validation images × 16 draws).

## Matched transfer evaluation

`evaluate` runs all five variants on identical inputs from the full 10,000-image test set. FGSM and 20-step PGD (uniform random start, step `epsilon / 8`) are generated against the undefended classifier at 4/255 and 8/255; these are transfer attacks against the defenses. Per condition and variant it reports accuracy, the fraction routed to the purifier, detector AUROC (clean vs attacked scores on the same images), and attack success rate (ASR) on images the undefended classifier classifies correctly, as well as on images the variant itself classifies correctly. A label-aware Oracle router is reported as a diagnostic upper bound only.

C&W L2 runs on a fixed 1,000-image test subset (`seed=2027`): untargeted, 100 Adam steps (learning rate 0.01) for each of 3 binary-search steps over the constant (initial 0.1), confidence 0. It records the L2 distortion of successful examples. The search budget is an implementation setting, not a claim of optimality.

## Adaptive evaluation

`adaptive-attack` evaluates every variant on a fixed 1,000-image test subset (`seed=2326`) under a white-box attacker who knows all components and parameters and has an L∞ budget of 8/255. Each image is attacked five ways, all with 100 steps of size 0.007 and a random start:

* `evade_epsad`, `evade_max`, `evade_fusion`: Selective PGD (Bryniarski et al., ICLR 2022). Each step follows the classifier loss while the image is classified correctly and the negative detector score once it is misclassified, so the candidate is both misclassified and routed around the purifier. EPS-AD gradients are exact through the diffusion model, with fresh noise each step; squeezing gradients use BPDA (exact forward, identity backward through each squeezer).
* `purify`: EOT-PGD (10 samples per step) through a differentiable FreqPure surrogate with 10 denoising steps and the deployed reference mode (Lee & Kim, ICCV 2023 protocol with a stronger surrogate). It ignores the detectors.

Every candidate is evaluated once by the exact stochastic pipeline. An image counts as correct for a variant only if all five candidates are classified correctly (worst case per image). Adaptive results are labelled separately from transfer results.

## Serving cost

`benchmark-latency` uses a fixed 1,024-image test subset (`seed=2426`) and its transfer PGD-8 versions. It reports batch-size-1 latency (mean, median, p95 over 100 images) for clean and attacked inputs, and end-to-end throughput for streams in which a fixed fraction (0, 5, 10, 25, 50, 75, 100%) of images is attacked, at batch size 64. Routed variants are timed with two serving policies: purifying flagged rows inside their own batch, and queueing flagged rows across batches and purifying them in full batches of 64. All timings synchronize the GPU and exclude warm-up and attack generation. Hardware: one NVIDIA GeForce RTX 5090; PyTorch 2.13.0+cu130.

## Scope

The experiments measure empirical behavior under the listed attacks, with one seed and one classifier; they make no certified-robustness claim. The proposal's purifier-loss ablation applied to the superseded autoencoder purifier and is replaced by the reference ablation; the squeezer ablation is detector-only and unaffected by the purifier change.
