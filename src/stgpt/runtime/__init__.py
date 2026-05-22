from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from ..annotation import annotate_regions as _annotate_regions
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
    """Deprecated compatibility wrapper for region-first embedding artifacts."""
    result = embed_regions(
        config=config,
        checkpoint=checkpoint,
        output_dir=output_dir,
        batch_size=batch_size,
        device=device,
    )
    result["deprecated"] = "embed_cells now returns region-first artifacts; use embed_regions instead."
    return result


def embed_regions(
    *,
    config: StGPTConfig | str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    batch_size: int = 32,
    device: str = "auto",
) -> dict[str, Any]:
    """Embed all contour/region units in a case and write stGPT/spatho-compatible artifacts."""
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
    return embed_regions(
        config=config,
        checkpoint=checkpoint,
        output_dir=output_dir,
        batch_size=batch_size,
        device=device,
    )


def annotate_regions(
    *,
    config: StGPTConfig | str | Path,
    checkpoint: str | Path,
    seed_labels: str | Path,
    output_dir: str | Path,
    region_ids: str | Path | None = None,
    include_no_image: bool = False,
    classifier: Literal["structure_head", "prototype_knn", "both"] = "both",
    abstain_prob: float = 0.5,
    write_probabilities: bool = False,
    seed_folds: int = 5,
    rng_seed: int = 42,
    batch_size: int = 32,
    device: str = "auto",
) -> dict[str, Any]:
    """Propagate sparse expert structure labels to unannotated regions."""
    return _annotate_regions(
        config=config,
        checkpoint=checkpoint,
        seed_labels=seed_labels,
        output_dir=output_dir,
        region_ids=region_ids,
        include_no_image=include_no_image,
        classifier=classifier,
        abstain_prob=abstain_prob,
        write_probabilities=write_probabilities,
        seed_folds=seed_folds,
        rng_seed=rng_seed,
        batch_size=batch_size,
        device=device,
    )


__all__ = [
    "annotate_regions",
    "embed_cells",
    "embed_regions",
    "evaluate_checkpoint",
    "export_spatho_artifacts",
    "package_model",
]
