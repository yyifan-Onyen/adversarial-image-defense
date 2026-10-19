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
    classification_weight: float = 0.5, max_train_batches: int | None = None,
) -> dict:
    seed_everything(seed)
    train, validation, _ = cifar10_datasets(data_dir, seed, augment=False, download=True)
    train_loader = loader(train, batch_size, True, seed, workers)
    val_loader = loader(validation, batch_size, False, seed, workers)
    classifier = load_classifier(classifier_path, device)
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    purifier = Purifier().to(device)
    optimizer = torch.optim.Adam(purifier.parameters(), lr=1e-3)
    best_loss = float("inf")
    history = []
    epsilon = 4 / 255

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
        return reconstruction + classification_weight * classification + identity, (reconstruction, classification, identity)

    for epoch in range(epochs):
        purifier.train()
        total_loss = count = 0
        for batch_index, (x, y) in enumerate(tqdm(train_loader, desc=f"purifier epoch {epoch + 1}/{epochs}")):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = batch_loss(x, y, batch_index)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            count += len(y)
        purifier.eval()
        val_loss = val_count = 0
        for batch_index, (x, y) in enumerate(val_loader):
            x, y = x.to(device), y.to(device)
            with torch.enable_grad():
                loss, _ = batch_loss(x, y, batch_index)
            val_loss += loss.item() * len(y)
            val_count += len(y)
        mean_val_loss = val_loss / max(val_count, 1)
        record = {"epoch": epoch + 1, "train_loss": total_loss / max(count, 1), "validation_loss": mean_val_loss}
        history.append(record)
        print(record, flush=True)
        if mean_val_loss < best_loss:
            best_loss = mean_val_loss
            save_checkpoint(purifier, output, {"seed": seed, "epoch": epoch + 1,
                                               "classification_weight": classification_weight,
                                               "validation_loss": mean_val_loss, "epsilon": epsilon})
    return {"best_validation_loss": best_loss, "history": history}


def fit_detector(
    data_dir: str, classifier_path: str, output: str, device: torch.device,
    seed: int = 2026, batch_size: int = 128, workers: int = 0,
    train_examples: int = 5000,
) -> dict:
    seed_everything(seed)
    train, validation, _ = cifar10_datasets(data_dir, seed, augment=False, download=True)
    if not 1 <= train_examples <= len(train):
        raise ValueError("train_examples must be within the 45,000-image training split")
    sampled = torch.utils.data.Subset(train, range(train_examples))
    classifier = load_classifier(classifier_path, device)
    features = []
    labels = []
    epsilon = 4 / 255
    for batch_index, (x, y) in enumerate(tqdm(loader(sampled, batch_size, False, seed, workers), desc="detector training")):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            _, clean_features = feature_tensor(classifier, x)
        adv = fgsm(classifier, x, y, epsilon) if batch_index % 2 == 0 else pgd(
            classifier, x, y, epsilon, steps=10, step_size=1 / 255)
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
    return {"train_examples": train_examples, "target_fpr": detector.target_fpr,
            "fusion_threshold": detector.fusion_threshold, "max_threshold": detector.max_threshold,
            "feature_cache": str(output_path.with_suffix(".features.npz"))}
