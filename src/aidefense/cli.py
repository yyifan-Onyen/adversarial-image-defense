"""Command-line entrypoints for the reproducible experimental stages."""

import argparse
import json
from pathlib import Path

from .data import resolve_device
from .evaluation import run_evaluation
from .training import fit_detector, train_classifier, train_purifier


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

    purifier = commands.add_parser("train-purifier", help="fit the three-term denoising autoencoder")
    add_common(purifier)
    purifier.add_argument("--classifier", default="checkpoints/classifier.pt")
    purifier.add_argument("--epochs", type=int, default=10)
    purifier.add_argument("--classification-weight", type=float, default=0.5)
    purifier.add_argument("--output", default="checkpoints/purifier.pt")
    purifier.add_argument("--max-train-batches", type=int, default=None, help="smoke testing only")

    detector = commands.add_parser("fit-detector", help="fit learned fusion and calibrate both gates")
    add_common(detector)
    detector.add_argument("--classifier", default="checkpoints/classifier.pt")
    detector.add_argument("--train-examples", type=int, default=5000)
    detector.add_argument("--output", default="checkpoints/detector.json")

    evaluation = commands.add_parser("evaluate", help="run matched defense and attack evaluation")
    add_common(evaluation)
    evaluation.set_defaults(batch_size=64)
    evaluation.add_argument("--classifier", default="checkpoints/classifier.pt")
    evaluation.add_argument("--purifier", default="checkpoints/purifier.pt")
    evaluation.add_argument("--detector", default="checkpoints/detector.json")
    evaluation.add_argument("--test-examples", type=int, default=10000)
    evaluation.add_argument("--expensive-subset", type=int, default=1000)
    evaluation.add_argument("--cw-steps", type=int, default=100)
    evaluation.add_argument("--cw-binary-search-steps", type=int, default=3)
    evaluation.add_argument("--adaptive-steps", type=int, default=30)
    evaluation.add_argument("--output", default="runs/final/evaluation.json")

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
                                max_train_batches=args.max_train_batches)
    elif args.command == "fit-detector":
        result = fit_detector(**shared, classifier_path=args.classifier,
                              train_examples=args.train_examples, output=args.output)
    else:
        result = run_evaluation(**shared, classifier_path=args.classifier, purifier_path=args.purifier,
                                detector_path=args.detector, test_examples=args.test_examples,
                                expensive_subset=args.expensive_subset, cw_steps=args.cw_steps,
                                cw_binary_search_steps=args.cw_binary_search_steps,
                                adaptive_steps=args.adaptive_steps, output=args.output)
    if args.command != "evaluate":
        output_path = Path(args.output).with_suffix(".metrics.json")
        output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["protocol"] if args.command == "evaluate" else result, indent=2), flush=True)


if __name__ == "__main__":
    main()
