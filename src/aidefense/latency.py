"""Serving-cost benchmark: batch-1 latency and throughput versus attack fraction.

The matched evaluation times each variant on whole batches that are either all
clean or all attacked, and purifies a batch's flagged rows inside that batch.
With a 100-step purifier this hides the benefit of selective routing: three
false positives in a batch of 64 still cost one full sequential purification
pass. This benchmark measures the serving cost directly:

* ``batch1``: per-image latency with batch size 1, for clean and transfer
  PGD-8 inputs separately (mean, median, p95).
* ``stream``: a fixed stream of images in which a given fraction is replaced
  by its transfer PGD-8 version, processed in batches of ``batch_size``. Two
  serving policies are timed for every routed variant:

  - ``per_batch``: flagged rows are purified inside their own batch (as in
    ``DefensePipeline.infer``);
  - ``queued``: flagged rows join a cross-batch queue that is purified in full
    batches of ``batch_size`` (remainder flushed at the end of the stream).
    Raw-path predictions are final immediately; queued images wait.

Throughput is measured end to end with device synchronisation, after warm-up.
Accuracy is reported only as a consistency check between the two policies.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from .attacks import pgd
from .data import cifar10_datasets, loader, seed_everything
from .detector import Detector
from .evaluation import fixed_subset, synchronize
from .pipeline import DefensePipeline
from .training import load_classifier, load_purifier

DEFAULT_FRACTIONS = (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)


@torch.no_grad()
def queued_stream_predictions(pipeline: DefensePipeline, inputs: torch.Tensor, variant: str,
                              batch_size: int) -> tuple[torch.Tensor, int]:
    """Serve ``inputs`` in arrival order with cross-batch queued purification."""
    predictions = torch.empty(len(inputs), dtype=torch.long, device=inputs.device)
    queue_x, queue_index = [], []
    queued = purified = 0

    def drain(count):
        nonlocal queue_x, queue_index, queued, purified
        x = torch.cat(queue_x)
        index = torch.cat(queue_index)
        predictions[index[:count]] = pipeline.purified_logits(x[:count]).argmax(1)
        queue_x, queue_index = [x[count:]], [index[count:]]
        queued -= count
        purified += count

    for start in range(0, len(inputs), batch_size):
        x = inputs[start:start + batch_size]
        logits, flags, _ = pipeline.detect(x, variant)
        positions = torch.arange(start, start + len(x), device=inputs.device)
        predictions[positions[~flags]] = logits[~flags].argmax(1)
        if flags.any():
            queue_x.append(x[flags])
            queue_index.append(positions[flags])
            queued += int(flags.sum())
        while queued >= batch_size:
            drain(batch_size)
    if queued:
        drain(queued)
    return predictions, purified


@torch.no_grad()
def per_batch_stream_predictions(pipeline: DefensePipeline, inputs: torch.Tensor, variant: str,
                                 batch_size: int) -> tuple[torch.Tensor, int]:
    predictions, purified = [], 0
    for start in range(0, len(inputs), batch_size):
        logits, flags, _ = pipeline.infer(inputs[start:start + batch_size], variant)
        predictions.append(logits.argmax(1))
        purified += int(flags.sum()) if variant in pipeline.routed_variants else (
            len(logits) if variant == "always" else 0)
    return torch.cat(predictions), purified


def _timed(device, function, *args):
    synchronize(device)
    start = time.perf_counter()
    result = function(*args)
    synchronize(device)
    return result, time.perf_counter() - start


def run_latency_benchmark(
    data_dir: str, classifier_path: str, purifier_path: str, detector_path: str, output: str,
    device: torch.device, epsad_path: str | None = None, seed: int = 2026, batch_size: int = 64,
    workers: int = 0, split: str = "test", examples: int = 1024, batch1_examples: int = 100,
    fractions=DEFAULT_FRACTIONS,
) -> dict:
    seed_everything(seed)
    if split not in ("validation", "test"):
        raise ValueError("split must be validation or test")
    if not 0 < batch1_examples <= examples:
        raise ValueError("batch1_examples must be in 1..examples")
    _, validation, test = cifar10_datasets(data_dir, seed, augment=False, download=True)
    dataset = fixed_subset(validation if split == "validation" else test, examples, seed + 400)
    classifier = load_classifier(classifier_path, device)
    purifier = load_purifier(purifier_path, device)
    epsad = None
    if epsad_path:
        from .epsad import load_epsad
        epsad = load_epsad(epsad_path, device)
    pipeline = DefensePipeline(classifier, purifier, Detector.load(detector_path), epsad)

    clean, attacked, labels = [], [], []
    for x, y in tqdm(loader(dataset, batch_size, False, seed, workers), desc="transfer PGD-8"):
        x, y = x.to(device), y.to(device)
        clean.append(x)
        attacked.append(pgd(classifier, x, y, 8 / 255, steps=20, step_size=1 / 255))
        labels.append(y)
    clean, attacked, labels = torch.cat(clean), torch.cat(attacked), torch.cat(labels)

    for variant in pipeline.variants:  # exclude one-time CUDA/cuDNN initialisation
        pipeline.infer(clean[:batch_size], variant)
        pipeline.infer(attacked[:batch_size], variant)
    synchronize(device)

    batch1 = {}
    for variant in tqdm(pipeline.variants, desc="batch-1 latency"):
        batch1[variant] = {}
        for condition, inputs in (("clean", clean), ("pgd_8", attacked)):
            times = []
            for i in range(batch1_examples):
                _, seconds = _timed(device, pipeline.infer, inputs[i:i + 1], variant)
                times.append(1000 * seconds)
            times = np.asarray(times)
            batch1[variant][condition] = {
                "mean_ms": float(times.mean()), "median_ms": float(np.median(times)),
                "p95_ms": float(np.percentile(times, 95)),
            }

    order = torch.randperm(examples, generator=torch.Generator().manual_seed(seed + 401))
    stream = []
    for fraction in tqdm(fractions, desc="stream throughput"):
        is_attacked = torch.zeros(examples, dtype=torch.bool)
        is_attacked[order[:round(fraction * examples)]] = True
        mask = is_attacked.to(device)[:, None, None, None]
        inputs = torch.where(mask, attacked, clean)
        entry = {"attack_fraction": fraction, "attacked_examples": int(is_attacked.sum()), "variants": {}}
        for variant in pipeline.variants:
            policies = {"per_batch": per_batch_stream_predictions}
            if variant in pipeline.routed_variants:
                policies["queued"] = queued_stream_predictions
            entry["variants"][variant] = {}
            for policy, function in policies.items():
                (predictions, purified), seconds = _timed(device, function, pipeline, inputs, variant, batch_size)
                entry["variants"][variant][policy] = {
                    "ms_per_image": 1000 * seconds / examples,
                    "images_per_second": examples / seconds,
                    "purified_fraction": purified / examples,
                    "accuracy": float(predictions.eq(labels).float().mean()),
                }
        stream.append(entry)

    report = {
        "protocol": {
            "split": split, "examples": examples, "subset_seed": seed + 400, "stream_order_seed": seed + 401,
            "batch_size": batch_size, "batch1_examples": batch1_examples,
            "attack": "transfer PGD-8 against the undefended classifier (20 steps, step 1/255)",
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "purifier": getattr(purifier, "spec", None), "epsad_detector": getattr(epsad, "spec", None),
            "warmup_excluded": True,
            "queued_policy": "flagged rows purified in full cross-batch batches; remainder flushed at end",
        },
        "batch1_latency": batch1,
        "stream": stream,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report
