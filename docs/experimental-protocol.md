# Experimental protocol

This document records the implementation choices behind the [final proposal](final-project-proposal.pdf). The official CIFAR-10 test set is held out from model selection. All reported numbers must come from completed runs; smoke tests are excluded.

## Data and classifier

Use the 50,000-image CIFAR-10 training set with a seeded random split of 45,000 training and 5,000 validation images (`seed=2026`). Training uses random 32 × 32 crops with four pixels of padding and random horizontal flips. Validation and test images use only tensor conversion to RGB values in [0, 1]. The 10,000-image official test set is used for final evaluation.

Train a CIFAR-adapted ResNet-18 from random initialization for 20 epochs using SGD (`lr=0.1`, momentum `0.9`, weight decay `5e-4`) and cosine decay. Select the checkpoint with highest validation accuracy, then freeze it for detector fitting, purifier training, attacks, and all comparisons. The first convolution uses a 3 × 3 kernel with stride 1 and the ImageNet stem's max pool is removed.

The data loader accepts TorchVision's CIFAR-10 download or the pinned University of Toronto Hugging Face mirror. The mirror is downloaded with [`scripts/download_cifar10_parquet.sh`](../scripts/download_cifar10_parquet.sh), whose SHA-256 checks guard against changed files. The run report records which source was used.

## Detector and routing

For each image, evaluate the frozen classifier on the original input and on each of three transformed inputs: 4-bit quantization per color channel, 3 × 3 median filtering, and JPEG quality 50. Each squeezing feature is the L1 distance between the original and transformed softmax vectors. The fourth feature is the gap between the two highest original softmax probabilities.

Fit logistic regression to clean and attacked examples from the training split. The attacked training examples alternate FGSM and 10-step PGD at 4/255. Calibrate a strict `score > threshold` decision on clean validation examples to an empirical false-positive rate no greater than 5%. Apply the same calibration to the original rule that takes the maximum of the three squeezing distances. The two routing variants use the **same** trained purifier; only their detection scores and thresholds differ.

## Purifier

The residual convolutional autoencoder has three convolutional encoder layers and two transposed-convolution decoder layers. Its output is the bounded input plus a learned residual. Training alternates FGSM and 7-step PGD examples at 4/255, using the frozen classifier to generate them. The main loss is

`MAE(D(x_adv), x) + 0.5 × CE(f(D(x_adv)), y) + MAE(D(x), x)`.

Both MAE terms average over images, channels, and pixels. The loss ablation sets the classification weight to zero and otherwise uses the same architecture, data split, attack settings, and seed. Each model is selected by its loss on the validation split. At inference, routed variants purify only flagged images, while the always-purify baseline processes every image.

## Evaluation

Compare no defense, max-rule routing, always-on purification, and learned-fusion routing on identical test inputs. FGSM and 20-step PGD are generated against the unprotected classifier at both 4/255 and 8/255. PGD starts uniformly within the allowed L-infinity ball and uses a step of `epsilon / 8`. These are transfer attacks against the defense. Report per-condition classification accuracy, baseline attack success on originally correct images, detector AUROC, detection rate at the validation-calibrated threshold, and complete-pipeline inference latency in milliseconds per image. Latency excludes attack generation and includes feature squeezing, host/device transfers, detection, routing, purification, and classification.

Run a separate C&W-style untargeted L2 optimization on a fixed 1,000-image test subset (`seed=2027`). Record attack success on originally correct images and the L2 distortion of successful attacks. Its bounded step count and constant search are implementation settings, not a claim of optimality. Run adaptive PGD at 8/255 on the same subset. The attack differentiates through a soft routing surrogate and uses BPDA for non-differentiable squeezers; report its success only after re-evaluating candidates with the exact discrete fusion pipeline. Label these subset results separately from full-test transfer results.

The experiment measures empirical behavior under these attacks. It makes no certified robustness claim. Inference latency depends on the hardware, batch size, and CPU implementations of JPEG and median filtering.

The CLI supports validation-split integration checks through `evaluate --split validation`; their metrics do not contribute numbers to the final test report.
