"""Command-line entrypoints for the reproducible experimental stages."""

import argparse
import json
from pathlib import Path

from .ablations import run_squeezer_ablations
from .data import resolve_device
from .evaluation import run_evaluation
from .training import DETECTOR_ATTACK_SUITES, fit_detector, train_classifier, train_purifier


def add_common(parser):
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    classifier = commands.add_parser("train-classifier", help="train the frozen CIFAR-10 ResNet-18")
    add_common(classifier)
    classifier.add_argument("--epochs", type=int, default=20)
    classifier.add_argument("--output", default="checkpoints/classifier.pt")
    classifier.add_argument("--max-train-batches", type=int, default=None, help="smoke testing only")

    purifier = commands.add_parser("train-purifier", help="fit the superseded three-term autoencoder purifier")
    add_common(purifier)
    purifier.add_argument("--classifier", default="checkpoints/classifier.pt")
    purifier.add_argument("--epochs", type=int, default=10)
    purifier.add_argument("--classification-weight", type=float, default=0.5)
    purifier.add_argument("--learning-rate", type=float, default=1e-3)
    purifier.add_argument("--lr-factor", type=float, default=0.5)
    purifier.add_argument("--lr-patience", type=int, default=2)
    purifier.add_argument("--min-learning-rate", type=float, default=1e-5)
    purifier.add_argument("--early-stopping-patience", type=int, default=6)
    purifier.add_argument("--min-delta", type=float, default=1e-4)
    purifier.add_argument("--output", default="checkpoints/purifier.pt")
    purifier.add_argument("--max-train-batches", type=int, default=None, help="smoke testing only")

    detector = commands.add_parser("fit-detector", help="fit learned fusion and calibrate both gates")
    add_common(detector)
    detector.add_argument("--classifier", default="checkpoints/classifier.pt")
    detector.add_argument("--train-examples", type=int, default=5000)
    detector.add_argument("--attack-suite", choices=DETECTOR_ATTACK_SUITES, default="proposal",
                          help="proposal: FGSM/PGD at 4/255; mixed-budgets: FGSM/PGD at 4/255 and 8/255")
    detector.add_argument("--output", default="checkpoints/detector.json")

    epsad = commands.add_parser("fit-epsad", help="compute EPS, train the EPS-AD deep kernel, calibrate")
    add_common(epsad)
    epsad.add_argument("--classifier", default="checkpoints/classifier.pt")
    epsad.add_argument("--unet", default="checkpoints/ddpm-cifar10-32")
    epsad.add_argument("--train-examples", type=int, default=10000)
    epsad.add_argument("--reference-size", type=int, default=500)
    epsad.add_argument("--steps", type=int, default=20, help="T*: number of EPS noise levels")
    epsad.add_argument("--epochs", type=int, default=200)
    epsad.add_argument("--learning-rate", type=float, default=0.002)
    epsad.add_argument("--target-fpr", type=float, default=0.05)
    epsad.add_argument("--output", default="checkpoints/epsad.pt")

    adaptive = commands.add_parser("adaptive-attack",
                                   help="white-box evasion and EOT purifier attacks on EPS-AD + FreqPure")
    add_common(adaptive)
    adaptive.add_argument("--classifier", default="checkpoints/classifier.pt")
    adaptive.add_argument("--purifier", default="configs/freqpure-ref-once.json")
    adaptive.add_argument("--detector", default="checkpoints/detector.json")
    adaptive.add_argument("--epsad", default="checkpoints/epsad.pt")
    adaptive.add_argument("--split", choices=("validation", "test"), default="validation")
    adaptive.add_argument("--examples", type=int, default=256)
    adaptive.add_argument("--epsilon", type=float, default=8.0, help="L-infinity budget in units of 1/255")
    adaptive.add_argument("--steps", type=int, default=100)
    adaptive.add_argument("--step-size", type=float, default=0.007)
    adaptive.add_argument("--eot", type=int, default=10)
    adaptive.add_argument("--surrogate-steps", type=int, default=10,
                          help="FreqPure surrogate denoising steps (1 overstates robustness)")
    adaptive.add_argument("--output", default="runs/adaptive/adaptive.json")

    probe = commands.add_parser("probe-purifier",
                                help="validation-only raw vs purified accuracy and purification latency")
    add_common(probe)
    probe.add_argument("--classifier", default="checkpoints/classifier.pt")
    probe.add_argument("--purifier", default="configs/freqpure.json",
                       help="trained purifier .pt checkpoint or FreqPure .json spec")
    probe.add_argument("--examples", type=int, default=256)
    probe.add_argument("--output", default="runs/probe/purifier-probe.json")
    probe.add_argument("--adaptive-examples", type=int, default=0,
                       help="also run white-box EOT-PGD through a 1-step FreqPure surrogate on this many images")
    probe.add_argument("--adaptive-steps", type=int, default=100)
    probe.add_argument("--adaptive-eot", type=int, default=10)
    probe.add_argument("--adaptive-surrogate-steps", type=int, default=1,
                       help="denoising steps in the differentiable FreqPure surrogate (must divide 1000)")

    ablations = commands.add_parser("ablate-squeezers", help="retrain fusion after dropping each squeezer")
    add_common(ablations)
    ablations.set_defaults(batch_size=64)
    ablations.add_argument("--classifier", default="checkpoints/classifier.pt")
    ablations.add_argument("--feature-cache", default="checkpoints/detector.features.npz")
    ablations.add_argument("--test-examples", type=int, default=1000)
    ablations.add_argument("--output", default="runs/final/squeezer_ablations.json")

    evaluation = commands.add_parser("evaluate", help="run matched defense and attack evaluation")
    add_common(evaluation)
    evaluation.set_defaults(batch_size=64)
    evaluation.add_argument("--classifier", default="checkpoints/classifier.pt")
    evaluation.add_argument("--purifier", default="configs/freqpure-ref-once.json",
                            help="FreqPure .json spec or trained autoencoder .pt checkpoint")
    evaluation.add_argument("--detector", default="checkpoints/detector.json")
    evaluation.add_argument("--epsad", default=None,
                            help="optional trained EPS-AD detector; adds the 'epsad' routing variant")
    evaluation.add_argument("--split", choices=("test", "validation"), default="test",
                            help="use validation for smoke checks; report final results on test")
    evaluation.add_argument("--test-examples", type=int, default=10000)
    evaluation.add_argument("--expensive-subset", type=int, default=1000)
    evaluation.add_argument("--cw-steps", type=int, default=100)
    evaluation.add_argument("--cw-binary-search-steps", type=int, default=3)
    evaluation.add_argument("--adaptive-steps", type=int, default=30)
    evaluation.add_argument("--skip-expensive", action="store_true",
                            help="omit C&W and adaptive checks for a matched loss ablation")
    evaluation.add_argument("--output", default="runs/final/evaluation.json")

    latency = commands.add_parser("benchmark-latency",
                                  help="batch-1 latency and per-batch vs queued throughput by attack fraction")
    add_common(latency)
    latency.set_defaults(batch_size=64)
    latency.add_argument("--classifier", default="checkpoints/classifier.pt")
    latency.add_argument("--purifier", default="configs/freqpure-ref-once.json")
    latency.add_argument("--detector", default="checkpoints/detector.json")
    latency.add_argument("--epsad", default="checkpoints/epsad.pt", help="pass '' to omit EPS-AD")
    latency.add_argument("--split", choices=("validation", "test"), default="test")
    latency.add_argument("--examples", type=int, default=1024)
    latency.add_argument("--batch1-examples", type=int, default=100)
    latency.add_argument("--fractions", type=float, nargs="+", default=None,
                         help="attack fractions of the stream (default 0 .05 .1 .25 .5 .75 1)")
    latency.add_argument("--output", default="runs/latency/latency.json")

    args = parser.parse_args(argv)
    device = resolve_device(args.device)
    shared = dict(data_dir=args.data_dir, device=device, seed=args.seed,
                  batch_size=args.batch_size, workers=args.workers)
    if args.command == "train-classifier":
        result = train_classifier(**shared, epochs=args.epochs, output=args.output,
                                  max_train_batches=args.max_train_batches)
    elif args.command == "train-purifier":
        result = train_purifier(**shared, classifier_path=args.classifier, epochs=args.epochs,
                                classification_weight=args.classification_weight, output=args.output,
                                learning_rate=args.learning_rate, lr_factor=args.lr_factor,
                                lr_patience=args.lr_patience, min_learning_rate=args.min_learning_rate,
                                early_stopping_patience=args.early_stopping_patience,
                                min_delta=args.min_delta,
                                max_train_batches=args.max_train_batches)
    elif args.command == "fit-detector":
        result = fit_detector(**shared, classifier_path=args.classifier,
                              train_examples=args.train_examples, attack_suite=args.attack_suite,
                              output=args.output)
    elif args.command == "fit-epsad":
        from .epsad import fit_epsad
        result = fit_epsad(**shared, classifier_path=args.classifier, unet_dir=args.unet,
                           train_examples=args.train_examples, reference_size=args.reference_size,
                           steps=args.steps, epochs=args.epochs, learning_rate=args.learning_rate,
                           target_fpr=args.target_fpr, output=args.output)
    elif args.command == "adaptive-attack":
        from .adaptive import run_adaptive_evaluation
        result = run_adaptive_evaluation(
            **shared, classifier_path=args.classifier, purifier_path=args.purifier,
            detector_path=args.detector, epsad_path=args.epsad, split=args.split,
            examples=args.examples, epsilon=args.epsilon / 255, steps=args.steps,
            step_size=args.step_size, eot=args.eot, surrogate_steps=args.surrogate_steps,
            output=args.output)
    elif args.command == "benchmark-latency":
        from .latency import DEFAULT_FRACTIONS, run_latency_benchmark
        result = run_latency_benchmark(
            **shared, classifier_path=args.classifier, purifier_path=args.purifier,
            detector_path=args.detector, epsad_path=args.epsad or None, split=args.split,
            examples=args.examples, batch1_examples=args.batch1_examples,
            fractions=tuple(args.fractions) if args.fractions else DEFAULT_FRACTIONS, output=args.output)
    elif args.command == "probe-purifier":
        from .probe import run_purifier_probe
        result = run_purifier_probe(**shared, classifier_path=args.classifier, purifier_path=args.purifier,
                                    examples=args.examples, adaptive_examples=args.adaptive_examples,
                                    adaptive_steps=args.adaptive_steps, adaptive_eot=args.adaptive_eot,
                                    adaptive_surrogate_steps=args.adaptive_surrogate_steps,
                                    output=args.output)
    elif args.command == "ablate-squeezers":
        result = run_squeezer_ablations(**shared, classifier_path=args.classifier,
                                        feature_cache=args.feature_cache,
                                        test_examples=args.test_examples, output=args.output)
    else:
        result = run_evaluation(**shared, classifier_path=args.classifier, purifier_path=args.purifier,
                                detector_path=args.detector, split=args.split,
                                test_examples=args.test_examples,
                                expensive_subset=args.expensive_subset, cw_steps=args.cw_steps,
                                cw_binary_search_steps=args.cw_binary_search_steps,
                                adaptive_steps=args.adaptive_steps,
                                include_expensive=not args.skip_expensive, epsad_path=args.epsad,
                                output=args.output)
    if args.command not in ("evaluate", "ablate-squeezers", "probe-purifier", "adaptive-attack",
                            "benchmark-latency"):
        output_path = Path(args.output).with_suffix(".metrics.json")
        output_path.write_text(json.dumps(result, indent=2) + "\n")
    summary = result["protocol"] if args.command in ("evaluate", "benchmark-latency") else result
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
