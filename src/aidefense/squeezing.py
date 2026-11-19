"""Feature-squeezing transforms and classifier-disagreement features."""

from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SQUEEZER_NAMES = ("quantize_4bit", "median_3x3", "jpeg_q50")


def median_3x3(x: torch.Tensor) -> torch.Tensor:
    """Per-channel 3x3 median with edge replication, on the input's device.

    Bit-identical to ``scipy.ndimage.median_filter(size=(1, 1, 3, 3),
    mode="nearest")``, which the detector was originally fitted with.
    """
    padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
    return padded.unfold(2, 3, 1).unfold(3, 3, 1).reshape(*x.shape, 9).median(-1).values


def squeeze(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == "quantize_4bit":
        return (x * 15).round().div(15).clamp(0, 1)
    if name == "median_3x3":
        return median_3x3(x)
    # JPEG stays on libjpeg via PIL: measured 0.09 ms/image on CPU, while
    # torchvision's nvJPEG path was slower (0.46 ms/image at 32x32) and does not
    # reproduce libjpeg's output, which would invalidate the fitted detector.
    array = x.detach().cpu().numpy()
    if name == "jpeg_q50":
        images = []
        for rgb in array:
            pixels = (np.transpose(rgb, (1, 2, 0)).clip(0, 1) * 255).round().astype(np.uint8)
            buffer = BytesIO()
            Image.fromarray(pixels).save(buffer, format="JPEG", quality=50)
            buffer.seek(0)
            with Image.open(buffer) as image:
                images.append(np.transpose(np.asarray(image.convert("RGB"), dtype=np.float32) / 255, (2, 0, 1)))
        out = np.stack(images)
    else:
        raise ValueError(f"unknown squeezer: {name}")
    return torch.from_numpy(np.ascontiguousarray(out)).to(device=x.device, dtype=x.dtype)


def feature_tensor(model, x: torch.Tensor, bpda: bool = False):
    """Return raw logits and [three L1 probability distances, prediction margin].

    BPDA uses the exact forward value of each non-differentiable transform but
    propagates an identity gradient for the adaptive-attack surrogate.
    """
    logits = model(x)
    probabilities = logits.softmax(dim=1)
    distances = []
    for name in SQUEEZER_NAMES:
        transformed = squeeze(x, name)
        if bpda:
            transformed = x + (transformed - x).detach()
        changed = model(transformed).softmax(dim=1)
        distances.append((probabilities - changed).abs().sum(dim=1))
    top_two = probabilities.topk(2, dim=1).values
    margin = top_two[:, 0] - top_two[:, 1]
    return logits, torch.stack([*distances, margin], dim=1)
