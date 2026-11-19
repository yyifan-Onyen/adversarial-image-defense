"""Classifier, purifier, and detector fitting on the training split."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

from .attacks import fgsm, pgd
from .data import cifar10_datasets, loader, seed_everything
from .detector import Detector
from .models import Purifier, build_classifier
from .squeezing import feature_tensor


DETECTOR_ATTACK_SUITES = ("proposal", "mixed-budgets")


def detector_attack_spec(batch_index: int, attack_suite: str) -> tuple[str, float, int]:
    """Return the detector-training attack assigned to a batch."""
    if attack_suite == "proposal":
        return ("fgsm", 4 / 255, 1) if batch_index % 2 == 0 else ("pgd", 4 / 255, 10)
    if attack_suite == "mixed-budgets":
        return (
            ("fgsm", 4 / 255, 1),
            ("fgsm", 8 / 255, 1),
            ("pgd", 4 / 255, 10),
            ("pgd", 8 / 255, 10),
        )[batch_index % 4]
    raise ValueError(f"unknown detector attack suite: {attack_suite}")


def save_checkpoint(model: nn.Module, path: str | Path, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)


def load_classifier(path: str | Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = build_classifier()
    model.load_state_dict(checkpoint["state_dict"])
    model.requires_grad_(False)
    return model.to(device).eval()


def load_purifier(path: str | Path, device: torch.device) -> nn.Module:
    """Load a trained purifier checkpoint (.pt) or a FreqPure JSON spec."""
    if Path(path).suffix == ".json":
        from .freqpure import load_freqpure
        return load_freqpure(path, device)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = Purifier()
    model.load_state_dict(checkpoint["state_dict"])
    model.requires_grad_(False)
    return model.to(device).eval()


def accuracy(model, batches, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in batches:
            x, y = x.to(device), y.to(device)
            correct += model(x).argmax(1).eq(y).sum().item()
            total += len(y)
    return correct / total


def train_classifier(
    data_dir: str, output: str, device: torch.device, seed: int = 2026,
    epochs: int = 20, batch_size: int = 128, workers: int = 0,
    max_train_batches: int | None = None,
) -> dict:
    seed_everything(seed)
    train, validation, _ = cifar10_datasets(data_dir, seed, augment=True, download=True)
    train_loader = loader(train, batch_size, True, seed, workers)
    val_loader = loader(validation, batch_size, False, seed, workers)
    model = build_classifier().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_accuracy = -1.0
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = count = 0
        for batch_index, (x, y) in enumerate(tqdm(train_loader, desc=f"classifier epoch {epoch + 1}/{epochs}")):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            count += len(y)
        scheduler.step()
        val_accuracy = accuracy(model, val_loader, device)
        record = {"epoch": epoch + 1, "train_loss": total_loss / max(count, 1), "validation_accuracy": val_accuracy}
        history.append(record)
        print(record, flush=True)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_checkpoint(model, output, {"seed": seed, "epoch": epoch + 1, "validation_accuracy": val_accuracy,
                                            "validation_size": 5000, "architecture": "cifar_resnet18"})
    return {"best_validation_accuracy": best_accuracy, "history": history}


def train_purifier(
    data_dir: str, classifier_path: str, output: str, device: torch.device,
    seed: int = 2026, epochs: int = 10, batch_size: int = 128, workers: int = 0,
    classification_weight: float = 0.5, learning_rate: float = 1e-3,
    lr_factor: float = 0.5, lr_patience: int = 2, min_learning_rate: float = 1e-5,
    early_stopping_patience: int = 6, min_delta: float = 1e-4,
    max_train_batches: int | None = None,
) -> dict:
    """Train the proposal's autoencoder purifier (superseded by FreqPure).

    Loss: L1(D(x_adv), x) + classification_weight * CE(f(D(x_adv)), y) + L1(D(x), x),
    with x_adv alternating FGSM and 7-step PGD at 4/255. The checkpoint with the
    lowest validation loss is kept.
    """
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0 or min_learning_rate <= 0:
        raise ValueError("learning rates must be positive")
    if not 0 < lr_factor < 1:
        raise ValueError("lr_factor must be between 0 and 1")
    if lr_patience < 0 or early_stopping_patience < 1 or min_delta < 0:
        raise ValueError("patience values and min_delta must be nonnegative")
    seed_everything(seed)
    train, validation, _ = cifar10_datasets(data_dir, seed, augment=False, download=True)
    train_loader = loader(train, batch_size, True, seed, workers)
    val_loader = loader(validation, batch_size, False, seed, workers)
    classifier = load_classifier(classifier_path, device)
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    purifier = Purifier().to(device)
    optimizer = torch.optim.Adam(purifier.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience,
        threshold=min_delta, threshold_mode="abs", min_lr=min_learning_rate,
    )
    epsilon = 4 / 255
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
    history = []

    def batch_loss(x, y, batch_index):
        if batch_index % 2 == 0:
            adv = fgsm(classifier, x, y, epsilon)
        else:
            adv = pgd(classifier, x, y, epsilon, steps=7, step_size=2 / 255)
        purified_adv = purifier(adv)
        reconstruction = F.l1_loss(purified_adv, x)
        classification = (
            F.cross_entropy(classifier(purified_adv), y)
            if classification_weight else torch.zeros((), device=x.device)
        )
        identity = F.l1_loss(purifier(x), x)
        return reconstruction + classification_weight * classification + identity

    for epoch in range(epochs):
        purifier.train()
        total_loss = count = 0
        for batch_index, (x, y) in enumerate(tqdm(train_loader, desc=f"purifier epoch {epoch + 1}")):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(x, y, batch_index)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            count += len(y)
        purifier.eval()
        val_loss = val_count = 0
        for batch_index, (x, y) in enumerate(val_loader):
            x, y = x.to(device), y.to(device)
            with torch.enable_grad():
                loss = batch_loss(x, y, batch_index)
            val_loss += loss.item() * len(y)
            val_count += len(y)
        mean_val_loss = val_loss / max(val_count, 1)
        current_lr = optimizer.param_groups[0]["lr"]
        improved = mean_val_loss < best_loss - min_delta
        record = {"epoch": epoch + 1, "train_loss": total_loss / max(count, 1),
                  "validation_loss": mean_val_loss, "learning_rate": current_lr, "improved": improved}
        history.append(record)
        print(record, flush=True)
        if improved:
            best_loss = mean_val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            save_checkpoint(purifier, output, {
                "seed": seed, "epoch": epoch + 1, "classification_weight": classification_weight,
                "validation_loss": mean_val_loss, "epsilon": epsilon, "learning_rate": current_lr})
        else:
            epochs_without_improvement += 1
        scheduler.step(mean_val_loss)
        if epochs_without_improvement >= early_stopping_patience:
            stopped_early = True
            print({"early_stopping_epoch": epoch + 1, "best_epoch": best_epoch}, flush=True)
            break
    return {"epochs_requested": epochs, "epochs_completed": len(history),
            "stopped_early": stopped_early, "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "final_learning_rate": optimizer.param_groups[0]["lr"], "history": history}


def fit_detector(
    data_dir: str, classifier_path: str, output: str, device: torch.device,
    seed: int = 2026, batch_size: int = 128, workers: int = 0,
    train_examples: int = 5000, attack_suite: str = "proposal",
) -> dict:
    seed_everything(seed)
    if attack_suite not in DETECTOR_ATTACK_SUITES:
        raise ValueError(f"attack_suite must be one of {DETECTOR_ATTACK_SUITES}")
    train, validation, _ = cifar10_datasets(data_dir, seed, augment=False, download=True)
    if not 1 <= train_examples <= len(train):
        raise ValueError("train_examples must be within the 45,000-image training split")
    sampled = torch.utils.data.Subset(train, range(train_examples))
    classifier = load_classifier(classifier_path, device)
    features = []
    labels = []
    attack_counts = {}
    for batch_index, (x, y) in enumerate(tqdm(loader(sampled, batch_size, False, seed, workers), desc="detector training")):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            _, clean_features = feature_tensor(classifier, x)
        attack, epsilon, steps = detector_attack_spec(batch_index, attack_suite)
        if attack == "fgsm":
            adv = fgsm(classifier, x, y, epsilon)
        else:
            adv = pgd(classifier, x, y, epsilon, steps=steps, step_size=epsilon / 4)
        attack_name = f"{attack}_{round(epsilon * 255)}_255"
        attack_counts[attack_name] = attack_counts.get(attack_name, 0) + len(y)
        with torch.no_grad():
            _, attacked_features = feature_tensor(classifier, adv)
        features.extend((clean_features.cpu().numpy(), attacked_features.cpu().numpy()))
        labels.extend((np.zeros(len(y), dtype=int), np.ones(len(y), dtype=int)))
    clean_validation_features = []
    for x, _ in tqdm(loader(validation, batch_size, False, seed, workers), desc="detector calibration"):
        with torch.no_grad():
            _, batch_features = feature_tensor(classifier, x.to(device))
        clean_validation_features.append(batch_features.cpu().numpy())
    detector = Detector.fit(np.concatenate(features), np.concatenate(labels), np.concatenate(clean_validation_features))
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    detector.save(output_path)
    np.savez_compressed(
        output_path.with_suffix(".features.npz"),
        train_features=np.concatenate(features),
        train_labels=np.concatenate(labels),
        clean_validation_features=np.concatenate(clean_validation_features),
    )
    return {"train_examples": train_examples, "attack_suite": attack_suite,
            "attacked_training_examples": attack_counts, "target_fpr": detector.target_fpr,
            "fusion_threshold": detector.fusion_threshold, "max_threshold": detector.max_threshold,
            "feature_cache": str(output_path.with_suffix(".features.npz"))}
