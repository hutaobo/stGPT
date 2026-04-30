from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

_RESAMPLE_BILINEAR = Image.Resampling.BILINEAR


def load_image_tensor(path: str | Path | None, *, image_size: int, channels: int = 3) -> torch.Tensor:
    if path is None:
        return torch.zeros(channels, image_size, image_size, dtype=torch.float32)
    image_path = Path(path)
    if not image_path.exists():
        return torch.zeros(channels, image_size, image_size, dtype=torch.float32)
    image = Image.open(image_path).convert("RGB").resize((image_size, image_size), _RESAMPLE_BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    if channels == 1:
        tensor = tensor.mean(dim=0, keepdim=True)
    elif channels > 3:
        pad = torch.zeros(channels - 3, image_size, image_size, dtype=torch.float32)
        tensor = torch.cat([tensor, pad], dim=0)
    return tensor[:channels]


def write_synthetic_patch(path: str | Path, *, image_size: int, structure_id: int, intensity: float, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:image_size, 0:image_size]
    colors = np.array(
        [
            [185, 80, 145],
            [90, 135, 205],
            [220, 160, 80],
            [80, 170, 120],
            [170, 110, 210],
        ],
        dtype=np.float32,
    )
    base = colors[int(structure_id) % len(colors)]
    wave = 0.5 + 0.5 * np.sin((xx + yy) / max(4.0, image_size / 8.0) + float(structure_id))
    noise = rng.normal(0.0, 10.0, size=(image_size, image_size, 1))
    stain = base.reshape(1, 1, 3) * (0.55 + 0.35 * wave[..., None]) + 45.0 * float(intensity) + noise
    stain = np.clip(stain, 0, 255).astype(np.uint8)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(stain, mode="RGB").save(out)
    return out
