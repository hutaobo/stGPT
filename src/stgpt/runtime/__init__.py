from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import StGPTConfig
from ..evaluation import evaluate
from ..foundation import package_model
from ..spatho import run_spatho_export


def embed_cells(
    *,
    config: StGPTConfig | str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    batch_size: int = 32,
    device: str = "auto",
) -> dict[str, Any]:
    """Embed all cells in a case and write stGPT/spatho-compatible artifacts."""
    return run_spatho_export(
        config=config,
        checkpoint=checkpoint,
        output_dir=output_dir,
        batch_size=batch_size,
        device=device,
    ).to_dict()


def evaluate_checkpoint(
    *,
    checkpoint: str | Path,
    config: StGPTConfig | str | Path,
    splits: str | Path,
    output_dir: str | Path,
    batch_size: int = 32,
    device: str = "auto",
) -> dict[str, Any]:
    """Evaluate a checkpoint using an existing QC split file."""
    return evaluate(
        checkpoint=checkpoint,
        config=config,
        splits=splits,
        output_dir=output_dir,
        batch_size=batch_size,
        device=device,
    )


def export_spatho_artifacts(
    *,
    config: StGPTConfig | str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    batch_size: int = 32,
    device: str = "auto",
) -> dict[str, Any]:
    """Runtime tool alias for producing spatho-consumable stGPT evidence artifacts."""
    return embed_cells(
        config=config,
        checkpoint=checkpoint,
        output_dir=output_dir,
        batch_size=batch_size,
        device=device,
    )


__all__ = [
    "embed_cells",
    "evaluate_checkpoint",
    "export_spatho_artifacts",
    "package_model",
]
