"""CIFAR-10 loading, fixed split, and device/reproducibility helpers."""

import random
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image


class ParquetCIFAR10(torch.utils.data.Dataset):
    """The UofT CIFAR-10 Hugging Face mirror, used when pinned files exist."""

    def __init__(self, path: Path, transform, expected_size: int):
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise RuntimeError("Parquet CIFAR-10 files found; install .[hf-data]") from exc
        table = parquet.read_table(path, columns=["img", "label"])
        if len(table) != expected_size:
            raise ValueError(f"expected {expected_size} images in {path}, found {len(table)}")
        self.images = [record["bytes"] for record in table["img"].to_pylist()]
        self.labels = table["label"].to_pylist()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        with Image.open(BytesIO(self.images[index])) as image:
            rgb = image.convert("RGB")
        return self.transform(rgb), int(self.labels[index])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        device = torch.device(name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def split_indices(seed: int = 2026, validation_size: int = 5000) -> tuple[list[int], list[int]]:
    if not 0 < validation_size < 50000:
        raise ValueError("validation_size must be between 1 and 49,999")
    indices = torch.randperm(50000, generator=torch.Generator().manual_seed(seed)).tolist()
    return indices[validation_size:], indices[:validation_size]


def cifar10_datasets(root: str | Path, seed: int = 2026, augment: bool = False, download: bool = False):
    root = Path(root)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]) if augment else transforms.ToTensor()
    parquet_train = root / "hf-train.parquet"
    parquet_test = root / "hf-test.parquet"
    if parquet_train.exists() and parquet_test.exists():
        raw_train = ParquetCIFAR10(parquet_train, train_transform, 50000)
        raw_validation = ParquetCIFAR10(parquet_train, transforms.ToTensor(), 50000)
        test = ParquetCIFAR10(parquet_test, transforms.ToTensor(), 10000)
    else:
        raw_train = datasets.CIFAR10(root=root, train=True, transform=train_transform, download=download)
        raw_validation = datasets.CIFAR10(root=root, train=True, transform=transforms.ToTensor(), download=download)
        test = datasets.CIFAR10(root=root, train=False, transform=transforms.ToTensor(), download=download)
    train_indices, validation_indices = split_indices(seed)
    return Subset(raw_train, train_indices), Subset(raw_validation, validation_indices), test


def loader(dataset, batch_size: int, shuffle: bool, seed: int, workers: int = 0) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed),
    )
