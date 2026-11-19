# Results report

**Snapshot date:** November 18, 2026

This report contains every measured result of the project. The research question, from the [proposal](final-project-proposal.pdf), is whether **detection gains translate into useful accuracy after accounting for latency and attack adaptation**. Official-test results are the final results; validation results are labelled and were used only for design decisions. The [experimental protocol](experimental-protocol.md) defines every setting. Raw records are committed under [`results/`](../results): `results/final/` holds the test-set evaluation, adaptive, and latency JSON with their run logs; `results/validation/` holds the validation runs behind the design decisions (reference ablation, EPS-AD fit, earlier FreqPure and autoencoder runs).

## Summary

| Question | Answer on the official test set |
|---|---|
| Does the proposal's learned fusion beat the max rule? | **Not consistently.** At the same 5% FPR it is better on FGSM (47.5% vs 34.2% accuracy at 4/255) and C&W (11.0% vs 3.5%) but worse on PGD (25.0% vs 40.1% at 8/255). Both squeezing detectors are weak (AUROC 0.76-0.87). |
| Does better detection give better accuracy under transfer attacks? | **Yes.** AUROC 0.76-0.87 → 1.00 lifts PGD-8 accuracy from 25.0-40.1% to 80.0%, equal to always-on purification (80.3%), with 5.1 points more clean accuracy (91.4% vs 86.3%). |
| Does selective purification save cost? | **Yes, with cross-batch queueing,** while attacked traffic stays below about 75%: 3.9× the throughput of always-on purification with no attacks, 2.7× at 10%. Purifying inside each batch saves little. |
| Does the detection gain hold for an unseen attack? | **No.** EPS-AD flags 56.8% of C&W L2 examples; routed accuracy is 54.1% vs 85.3% for always-on purification. |
| Does it hold against an adaptive attacker? | **No.** Worst-case accuracy is 0.0% (max, fusion), 4.7% (EPS-AD routing), and 23.8% (always-on purification). Routing lets the attacker bypass the purifier. |
| Can FreqPure be made cheaper? | **Yes.** Computing its reference once halves UNet calls (200 → 101) with no measured accuracy loss. |
| Is the standard robustness evaluation of diffusion purification reliable? | **No.** A one-step attack surrogate overstates accuracy by about 50 points compared with a ten-step surrogate. |

## Setup

| Item | Setting |
|---|---|
| Dataset | CIFAR-10, 45,000 train / 5,000 validation / 10,000 official test |
| Classifier | CIFAR ResNet-18, frozen; 92.00% validation, 91.71% test accuracy |
| Weak detectors | Feature squeezing (4-bit, 3 × 3 median, JPEG 50): max rule and logistic fusion |
| Strong detector | EPS-AD, 20 noise levels, deep-kernel MMD, threshold 1.00495. Added beyond the proposal so that detection quality can be varied from weak to near-perfect with the same purifier |
| Detector training data | Fusion: 5,000 training images and their FGSM / 10-step PGD versions at 4/255 (proposal setting). EPS-AD: 10,000 training images and their FGSM / FGSM-L2 versions at 1/255 (EPS-AD paper setting) |
| Calibration | 5% FPR on clean validation; measured test FPR 4.8% (max), 5.0% (fusion), 5.1% (EPS-AD) |
| Purifier | FreqPure on `google/ddpm-cifar10-32`, 100 DDPM steps, reference computed once (101 UNet calls). The public diffusion model is used as released; nothing is trained on adversarial examples |
| Attacks | FGSM, PGD, C&W L2, and the adaptive attacks are our own PyTorch implementations against the frozen classifier |
| Hardware | NVIDIA GeForce RTX 5090, PyTorch 2.13.0+cu130, batch size 64 unless stated |

## Transfer attacks on the full test set

FGSM and 20-step PGD are generated against the undefended classifier on all 10,000 test images; C&W L2 uses a fixed 1,000-image subset. Accuracy (%):

| Condition | No defense | Max + purify | Fusion + purify | EPS-AD + purify | Always purify | Oracle (diagnostic) |
|---|---:|---:|---:|---:|---:|---:|
| Clean | **91.7** | 90.9 | 90.8 | 91.4 | 86.3 | 94.4 |
| FGSM 4/255 | 14.5 | 34.2 | 47.5 | 81.9 | **82.0** | 82.1 |
| FGSM 8/255 | 8.3 | 22.8 | 39.4 | **76.4** | 76.1 | 76.4 |
| PGD 4/255 | 0.0 | 53.5 | 42.9 | 82.6 | **83.0** | 83.0 |
| PGD 8/255 | 0.0 | 40.1 | 25.0 | 80.0 | **80.3** | 80.3 |
| C&W L2 (1,000) | 0.0 | 3.5 | 11.0 | 54.1 | **85.3** | 85.3 |

The Oracle picks, with the true label, whichever of the raw and purified predictions is correct; it is an upper bound for any router over these two paths, not a defense.

Detection (fraction routed to the purifier, % / AUROC):

| Condition | Max rule | Fusion | EPS-AD |
|---|---:|---:|---:|
| Clean (false positives) | 4.8 | 5.0 | 5.1 |
| FGSM 4/255 | 22.2 / 0.771 | 36.8 / 0.812 | 100.0 / 1.000 |
| FGSM 8/255 | 16.8 / 0.761 | 37.2 / 0.814 | 100.0 / 1.000 |
| PGD 4/255 | 57.3 / 0.870 | 45.0 / 0.849 | 100.0 / 1.000 |
| PGD 8/255 | 43.0 / 0.785 | 26.4 / 0.756 | 100.0 / 1.000 |
| C&W L2 | 4.1 | 12.7 | 56.8 |

Attack success rate (%) among the 9,171 images the undefended classifier gets right:

| Attack | No defense | Max + purify | Fusion + purify | EPS-AD + purify | Always purify |
|---|---:|---:|---:|---:|---:|
| FGSM 4/255 | 84.2 | 63.0 | 48.5 | 12.8 | 12.7 |
| FGSM 8/255 | 91.0 | 75.5 | 57.3 | 18.0 | 18.5 |
| PGD 4/255 | 100.0 | 42.1 | 53.5 | 12.1 | 11.6 |
| PGD 8/255 | 100.0 | 56.6 | 72.9 | 14.5 | 14.3 |

C&W succeeds on all 914 subset images the undefended classifier gets right, with mean L2 distortion 0.145 (median 0.136).

**Reading.**

* With a weak detector, most attacked images are never purified, so accuracy follows the detection rate. With EPS-AD every FGSM/PGD image is purified and routed accuracy reaches the Oracle bound, while clean images keep the raw classifier's accuracy. Purification harms 811 clean test images and rescues 266, so always-on purification costs 5.4 clean points that routing avoids.
* Fusion beats max on FGSM but loses on PGD, especially at 8/255 (26.4% vs 43.0% detected). Fusion was fitted only on 4/255 attacks, so learning the combination does not generalize across attack types or budgets as well as the fixed max rule.
* C&W is unseen by every detector (EPS-AD's kernel was trained on FGSM and FGSM-L2 at 1/255). Its small perturbations evade 43% of EPS-AD decisions, and those images stay misclassified, although FreqPure would recover 85.3% of them.

## Serving cost

Per-image cost is measured on 1,024 test images and their transfer PGD-8 versions. In the matched evaluation above (batch 64, flagged rows purified inside their own batch), EPS-AD routing costs 23.0 ms per clean image and 34.1-34.7 ms per attacked image versus 28.6-29.5 ms for always-on purification. Squeezing detection itself costs about 0.33 ms per image and EPS-AD detection about 5.5 ms; the rest is purification. Because purification runs 100 sequential steps, the three false positives in a typical clean batch of 64 cost almost a full batch-purification pass.

Stream throughput, batch size 64, ms per image (accuracy in parentheses):

| Attacked fraction | Always purify | EPS-AD, per batch | **EPS-AD, queued** | Max, queued | Fusion, queued |
|---:|---:|---:|---:|---:|---:|
| 0% | 29.0 (86.3) | 23.7 | **7.5** (90.7) | 1.4 (90.6) | 1.8 (89.6) |
| 5% | 29.1 (85.9) | 23.8 | **8.7** (89.6) | 1.9 (88.3) | 1.9 (86.5) |
| 10% | 29.0 (86.2) | 23.7 | **10.7** (89.8) | 3.2 (85.0) | 3.1 (83.2) |
| 25% | 28.9 (84.3) | 24.2 | **14.1** (87.9) | 5.0 (77.1) | 3.3 (72.9) |
| 50% | 28.9 (83.3) | 25.8 | **22.3** (85.6) | 7.1 (65.8) | 5.3 (58.5) |
| 75% | 30.6 (83.2) | 30.0 | **29.1** (83.5) | 11.0 (54.0) | 7.0 (42.6) |
| 100% | 30.4 (80.2) | 34.6 | **34.7** (78.8) | 13.8 (41.4) | 9.0 (25.4) |

Batch-size-1 latency (ms, mean / median / p95 over 100 images):

| Variant | Clean | PGD 8/255 |
|---|---:|---:|
| No defense | 1.5 / 1.4 / 1.8 | 1.5 / 1.4 / 1.8 |
| Max + purify | 60 / 7.7 / 9.4 | 567 / 8.3 / 1,251 |
| Fusion + purify | 54 / 6.7 / 8.9 | 327 / 7.0 / 1,215 |
| EPS-AD + purify | 278 / 207 / 1,305 | 1,393 / 1,374 / 1,581 |
| Always purify | 1,169 / 1,159 / 1,299 | 1,197 / 1,173 / 1,345 |

**Reading.**

* Selective purification only pays off with a serving policy that collects flagged images across batches. Then EPS-AD routing is 3.9× faster than always-on purification on clean traffic and also more accurate at every attacked fraction up to 75%. Beyond about 75% attacked traffic its detection cost (about 5.5 ms per image) is pure overhead and accuracy is 1.4 points lower than always-on purification at 100%.
* The squeezing routers are cheaper still, but only because they let most attacked images through unpurified; their accuracy falls with the attacked fraction.
* At batch size 1 the diffusion model is launch-bound (about 11 ms per UNet call): EPS-AD detection alone takes about 0.2 s. EPS-AD routing is 4.2× faster than always-on purification for clean images and 16% slower for attacked ones.
* Queued images wait for a full purification batch; this benchmark measures throughput, not their tail latency.

## Adaptive attacks

White-box, L∞ 8/255, 1,000 test images, 100 steps of size 0.007. The proposal planned one adaptive PGD attack on the full pipeline; because a detect-then-purify system can be attacked through either component, we run one attack per component and take the worst case. Each image faces three Selective-PGD evasion attacks (one per detector) and one EOT-PGD attack through a ten-step FreqPure surrogate; an image counts only if every candidate is classified correctly by the exact pipeline. Accuracy (%), with the attack aimed at each variant in bold:

| Attack | No defense | Max + purify | Fusion + purify | EPS-AD + purify | Always purify |
|---|---:|---:|---:|---:|---:|
| Clean | 92.1 | 91.4 | 90.9 | 91.5 | 85.8 |
| Evade EPS-AD | 0.1 | 33.0 | 51.2 | **15.2** | 85.7 |
| Evade max | 0.0 | **0.0** | 47.6 | 85.2 | 84.4 |
| Evade fusion | 0.0 | 30.3 | **0.0** | 84.4 | 85.6 |
| Through purifier | 24.1 | 21.9 | 21.6 | 23.7 | **23.9** |
| **Worst case** | **0.0** | **0.0** | **0.0** | **4.7** | **23.8** |
| Worst-case ASR on undefended-correct | 100.0 | 100.0 | 100.0 | 94.9 | 74.2 |

| Attack | Seconds per image | Routed by max / fusion / EPS-AD (%) |
|---|---:|---:|
| Evade EPS-AD | 1.53 | 34.1 / 56.8 / 17.0 |
| Evade max | 0.03 | 0.0 / 51.8 / 100.0 |
| Evade fusion | 0.03 | 32.3 / 0.0 / 100.0 |
| Through purifier | 8.55 | 6.2 / 10.1 / 99.9 |

**Reading.**

* Every routed defense is broken by the attack that targets its own detector: the candidate is misclassified by the raw classifier and not flagged, so it never reaches the purifier. Selective PGD found misclassified-and-undetected EPS-AD candidates for 97.4% of images during the search; because EPS-AD is stochastic, 17% of them are flagged again at evaluation, which leaves 15.2%.
* The attack through the purifier works against every variant: it breaks always-on purification directly, is flagged by EPS-AD and then survives purification, and is mostly not flagged by squeezing, so the raw classifier sees it.
* Evasion attacks transfer poorly between detector families: images crafted to evade squeezing are all flagged by EPS-AD, and vice versa only partly. A defense that must be evaded by both families at once was not evaluated.
* Under adaptive attack, better detection gives no useful accuracy: EPS-AD routing (4.7%) is worse than always-on purification (23.8%).

## Reference ablation: once versus per step

Validation split; transfer conditions on 1,000 images, adaptive EOT-PGD (EOT 10, 100 steps, 8/255) on 256. Accuracy (%):

| Reference | UNet calls | Purify time | Clean | FGSM 8 | PGD 4 | PGD 8 | Adaptive, 1-step surrogate | Adaptive, 10-step surrogate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Per step (released) | 200 | 49.0 ms | 90.1 | 82.0 | 87.1 | 86.0 | 81.2 | 24.6 |
| Once (deployed) | 101 | 24.6 ms | 89.7 | 81.3 | 86.9 | 86.1 | 84.0 | 31.6 |
| EPS-AD noise levels | 120 | 29.0 ms | 90.4 | 70.6 | 84.0 | 77.1 | 70.3 | — |

The raw classifier gets 92.6% of the 1,000 clean images. Both adaptive columns use the same 256-image subset, on which the transfer accuracies are similar (PGD 8: 85.5% per step, 86.7% once).

* Computing the reference once halves the cost and is within noise of the released per-step reference under every condition (the 7-point ten-step difference, in favor of once, is about 1.8 standard errors if the two runs are treated as independent). Sharing EPS-AD's noise levels as the reference fails: detection needs low noise to preserve the perturbation, purification needs enough noise to remove it.
* Why redrawing is unnecessary: across 16 draws per image (256 validation images), the low band FreqPure uses has a signal-to-deviation energy ratio of 768, against 9 in the discarded high band. The redraws differ mainly in weak low-band components (23% of low-band phases move more than δ = 0.3 between draws, but they carry little energy). An adversarial perturbation shifts the mean low-band reference by only 0.26 of the draw-to-draw deviation energy.
* **Surrogate strength matters more than the reference.** The one-step surrogate used by the released FreqPure evaluation leaves 81-84% accuracy; ten steps leave 25-32%. An earlier 500-image validation run with one-step surrogates had reported 78.6% worst-case accuracy for always-on purification; the test-set adaptive results above use ten steps.

## Squeezer ablation

Detector-only, transfer PGD, fixed 1,000-image test subset; every feature subset refitted and recalibrated to 5% validation FPR. Detection rate / AUROC:

| Features | PGD 4 fusion | PGD 8 fusion | PGD 4 max | PGD 8 max |
|---|---:|---:|---:|---:|
| All three | 48.6% / 0.863 | 28.4% / 0.774 | 59.0% / 0.880 | 44.6% / 0.797 |
| Without quantization | 46.6% / 0.858 | 26.6% / 0.765 | 59.4% / 0.881 | 44.6% / 0.799 |
| Without median | 44.8% / 0.845 | 36.0% / 0.780 | 55.3% / 0.867 | 44.3% / 0.795 |
| Without JPEG | 27.8% / 0.782 | 13.9% / 0.657 | 43.0% / 0.810 | 24.2% / 0.668 |

JPEG is the most informative squeezer and quantization the least. This subset was inspected before the final evaluation, but no design choice was taken from it.

## Superseded autoencoder purifier

The proposal's original purifier, a residual autoencoder trained with reconstruction, classification, and identity losses (30 epochs, best at epoch 28), was evaluated on 1,000 validation images before the switch to FreqPure. Always-on purification reached 68.5% / 55.5% (FGSM 4 / 8) and 69.3% / 56.7% (PGD 4 / 8) while dropping clean accuracy from 93.8% to 80.5%; FreqPure gives 86-87% on the same kind of PGD inputs with a smaller clean cost. It was never run on the test set. The purifier-loss ablation belonged to this model and is replaced by the reference ablation.

## Answer to the research question

Detection gains translate into useful accuracy **only under the threat model the detector was built for**. The proposal's own detector improvement, learned fusion over the max rule, is too small and inconsistent to matter: both squeezing routers leave most attacked images unpurified. The question is answered by the larger step from squeezing to EPS-AD. Against transfer FGSM/PGD, a near-perfect detector lets selective purification match always-on purification under attack, keep the undefended classifier's clean accuracy, and, with queued purification, serve mostly clean traffic several times faster. Against an unseen attack (C&W) the gain shrinks to about half, and against an adaptive attacker it disappears: routing becomes the weakest point, and the routed defenses fall below always-on purification, which itself keeps only 23.8%. This is the negative result the proposal anticipated: simple detect-then-purify defenses improve average-case cost and accuracy, not worst-case robustness.

## Limitations

* One seed, one classifier, CIFAR-10 only; no confidence intervals beyond the binomial noise of the stated sample sizes (95% intervals of about ±1.0 points at 10,000 images and ±3.1 at 1,000 for accuracies near 50%).
* Adaptive results are upper bounds on robustness from specific strong attacks, not certified lower bounds. A combined attack that evades two detector families at once was not run.
* Latency and throughput are for one RTX 5090 and batch size 64 (plus batch size 1); the queued policy is measured by throughput only.
* The fusion detector was fitted on 4/255 attacks only, as in the proposal; a mixed-budget fusion detector fitted earlier did not improve validation accuracy.
