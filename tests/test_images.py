from __future__ import annotations

from pathlib import Path

from stgpt.images import load_image_tensor, write_synthetic_patch


def test_image_patch_tensor_loading(tmp_path: Path) -> None:
    image_path = write_synthetic_patch(tmp_path / "patch.png", image_size=32, structure_id=1, intensity=0.5, seed=4)
    tensor = load_image_tensor(image_path, image_size=32, channels=3)
    assert tensor.shape == (3, 32, 32)
    assert float(tensor.max()) <= 1.0
    assert float(tensor.min()) >= 0.0
