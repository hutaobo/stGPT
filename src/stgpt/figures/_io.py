"""Artifact loading and provenance helpers for stGPT figures.

Figures are artifact-first: they read the parquet/csv that the evidence layer
already produced (carrying UMAP coordinates and checkpoint hashes) rather than
re-deriving geometry. This keeps a paper figure numerically identical to the
runtime evidence it is meant to document.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

# Columns that can act as a "batch"/platform axis for mixing, in priority order.
# `platform` is first so the cross-platform story uses it once the export carries
# it; until then the function falls back to the next column with >1 value.
BATCH_KEY_CANDIDATES: tuple[str, ...] = (
    "platform",
    "organ",
    "tissue",
    "scanner",
    "stain",
    "batch_id",
    "patient_id",
    "slide_id",
    "condition",
    "run_id",
)


def load_manifold_frame(manifold: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Load a projected latent-manifold table as a DataFrame.

    Accepts a DataFrame, a ``.csv``, or a ``.parquet`` path. The frame must
    already contain ``manifold_x``/``manifold_y`` projection columns; pass the
    output of ``stgpt latent-manifold`` (``latent_manifold.csv``). A region
    embedding table without projection columns is rejected with guidance, so
    coordinates always originate from the evidence layer.
    """
    if isinstance(manifold, pd.DataFrame):
        frame = manifold.copy()
        source = "<dataframe>"
    else:
        path = Path(manifold).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"manifold artifact not found: {path}")
        if path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(path)
        elif path.suffix.lower() in {".csv", ".tsv"}:
            frame = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
        else:
            raise ValueError(f"unsupported manifold format '{path.suffix}'; expected .csv or .parquet")
        source = str(path)
    if "manifold_x" not in frame.columns or "manifold_y" not in frame.columns:
        raise ValueError(
            "manifold frame is missing 'manifold_x'/'manifold_y'. Run "
            "`stgpt latent-manifold` first and pass its latent_manifold.csv so "
            "coordinates and provenance come from the evidence layer."
        )
    frame.attrs["source"] = source
    return frame


def resolve_batch_key(frame: pd.DataFrame, batch_key: str) -> tuple[str, list[str]]:
    """Resolve the mixing axis. ``"auto"`` picks the first candidate with >1 value.

    Returns the chosen column name and any warnings (e.g. no multi-value column
    was found, so the panel collapses to a single group).
    """
    warnings: list[str] = []
    if batch_key != "auto":
        if batch_key not in frame.columns:
            raise ValueError(f"batch_key '{batch_key}' is not a column in the manifold frame")
        return batch_key, warnings
    for candidate in BATCH_KEY_CANDIDATES:
        if candidate in frame.columns and frame[candidate].dropna().nunique() > 1:
            return candidate, warnings
    # Nothing varies: fall back to the first present candidate (single group).
    for candidate in BATCH_KEY_CANDIDATES:
        if candidate in frame.columns:
            warnings.append(f"no_multi_value_batch_key: '{candidate}' has a single value; mixing panel is trivial")
            return candidate, warnings
    warnings.append("no_batch_key_columns: manifold frame has no platform/tissue/batch columns")
    return "", warnings


def checkpoint_guardrail(frame: pd.DataFrame, run_id: str | None) -> tuple[list[str], list[str]]:
    """Mirror the manifold builder's cross-run guardrail.

    Cross-run geometry is only trustworthy when the points share a checkpoint.
    Returns ``(checkpoint_hashes, warnings)``.
    """
    warnings: list[str] = []
    if "checkpoint_hash" not in frame.columns:
        return [], ["missing_checkpoint_hashes: figure cannot assert reproducible geometry"]
    hashes = sorted(str(value) for value in frame["checkpoint_hash"].dropna().unique())
    if len(hashes) > 1 and run_id is None:
        warnings.append(
            "multiple_checkpoint_hashes: cross-run geometry is exploratory unless "
            "you filter to one run with run_id"
        )
    return hashes, warnings


def subsample(frame: pd.DataFrame, *, max_points: int, seed: int) -> tuple[pd.DataFrame, bool]:
    """Randomly subsample for render speed. Returns ``(frame, was_sampled)``."""
    if max_points <= 0 or len(frame) <= max_points:
        return frame, False
    return frame.sample(n=max_points, random_state=seed).sort_index(), True


def json_safe(value: Any) -> Any:
    """Recursively coerce numpy/pandas scalars into JSON-serialisable values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        import numpy as np

        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            number = float(value)
            return number if math.isfinite(number) else None
    except ImportError:
        pass
    return value
