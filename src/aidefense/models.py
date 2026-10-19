"""Classifier and input purifier architectures."""

import torch
from torch import nn
from torchvision.models import resnet18


def build_classifier() -> nn.Module:
    """A randomly initialized ResNet-18 adapted to 32x32 CIFAR-10 inputs."""
    model = resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


class Purifier(nn.Module):
    """Compact residual convolutional autoencoder with bounded RGB output."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = 0.25 * torch.tanh(self.decoder(self.encoder(x)))
        return (x + residual).clamp(0.0, 1.0)
