from __future__ import annotations

import json
import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd
import torch

MODEL_MANIFEST = "stgpt_model_manifest.json"
MODEL_CARD = "README.md"
PACKAGED_CHECKPOINT = "checkpoint.pt"


def package_model(
    *,
    checkpoint: str | Path,
    evaluation: str | Path,
    output_dir: str | Path,
    model_name: str | None = None,
) -> dict[str, str]:
    """Package a trained checkpoint as a reusable stGPT model directory."""
    checkpoint_path = Path(checkpoint).resolve()
    evaluation_path = Path(evaluation).resolve()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu")
    cfg = payload.get("config", {})
    vocab = payload.get("vocab", {})
    metadata = _model_metadata(payload, checkpoint_path, evaluation_path, model_name=model_name)

    packaged_checkpoint = out / PACKAGED_CHECKPOINT
    packaged_eval = out / "evaluation_metrics.json"
    config_path = out / "config.json"
    vocab_path = out / "vocab.json"
    manifest_path = out / MODEL_MANIFEST
    model_card_path = out / MODEL_CARD

    shutil.copy2(checkpoint_path, packaged_checkpoint)
    shutil.copy2(evaluation_path, packaged_eval)
    config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    vocab_path.write_text(json.dumps(vocab, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    model_card_path.write_text(_model_card(metadata), encoding="utf-8")

    return {
        "checkpoint": str(packaged_checkpoint),
        "evaluation_metrics": str(packaged_eval),
        "manifest": str(manifest_path),
        "config": str(config_path),
        "vocab": str(vocab_path),
        "model_card": str(model_card_path),
    }


def resolve_model_checkpoint(path: str | Path) -> Path:
    """Resolve a checkpoint path from either a ``*.pt`` file or packaged model dir."""
    candidate = Path(path).resolve()
    if candidate.is_file():
        return candidate
    if not candidate.is_dir():
        raise FileNotFoundError(f"stGPT model path does not exist: {candidate}")

    manifest = candidate / MODEL_MANIFEST
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        checkpoint = payload.get("checkpoint") or payload.get("checkpoint_path") or PACKAGED_CHECKPOINT
        resolved = (candidate / checkpoint).resolve() if not Path(str(checkpoint)).is_absolute() else Path(str(checkpoint))
        if resolved.exists():
            return resolved

    checkpoint = candidate / PACKAGED_CHECKPOINT
    if checkpoint.exists():
        return checkpoint
    raise FileNotFoundError(f"No stGPT checkpoint found in packaged model dir: {candidate}")


def _model_metadata(
    payload: dict[str, Any],
    checkpoint_path: Path,
    evaluation_path: Path,
    *,
    model_name: str | None,
) -> dict[str, Any]:
    cfg = payload.get("config", {})
    data = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    training_provenance = _training_provenance(evaluation_path)
    return {
        "model_name": model_name or "stgpt-xenium-demo",
        "model_type": "Xenium-first morpho-molecular foundation-model backend in development",
        "model_version": payload.get("model_version", _stgpt_version()),
        "checkpoint": PACKAGED_CHECKPOINT,
        "source_checkpoint": str(checkpoint_path),
        "evaluation_metrics": "evaluation_metrics.json",
        "source_evaluation_metrics": str(evaluation_path),
        "case_name": cfg.get("case_name"),
        "data_mode": data.get("mode"),
        "training_unit": payload.get("training_unit", "region"),
        "n_regions": payload.get("n_regions"),
        "max_cells_per_region": payload.get("max_cells_per_region"),
        "panel_metadata": payload.get("panel_metadata", {}),
        "split_summary": payload.get("split_summary", {}),
        "training_provenance": training_provenance,
        "structure_names": payload.get("structure_names", []),
        "training_summary": payload.get("training_summary", {}),
        "intended_use": (
            "Generate auditable morpho-molecular embeddings for Xenium-centered spatial pathology workflows."
        ),
        "limitations": [
            "Not a diagnostic device.",
            "Do not treat imputed or reconstructed values as measured expression.",
            "Requires QC, split provenance, and case-level validation before biological claims.",
        ],
    }


def _model_card(metadata: dict[str, Any]) -> str:
    limitations = "\n".join(f"- {item}" for item in metadata.get("limitations", []))
    provenance = metadata.get("training_provenance", {})
    split = provenance.get("split", {}) if isinstance(provenance, dict) else {}
    return f"""# {metadata["model_name"]}

This packaged stGPT checkpoint is a Xenium-first morpho-molecular foundation-model backend in development.

## Intended Use

{metadata["intended_use"]}

## Scope

- Case: {metadata.get("case_name")}
- Data mode: {metadata.get("data_mode")}
- Training unit: {metadata.get("training_unit")}
- Regions: {metadata.get("n_regions")}
- Model version: {metadata.get("model_version")}

## Training Provenance

- Split strategy: {split.get("strategy")}
- Split group key: {split.get("group_key")}
- Split counts: {split.get("counts")}
- Holdout groups: {split.get("holdout_groups")}

## Limitations and Guardrails

{limitations}
"""


def _training_provenance(evaluation_path: Path) -> dict[str, Any]:
    if not evaluation_path.exists():
        return {}
    try:
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    provenance: dict[str, Any] = {"evaluation_metrics": str(evaluation_path)}
    splits_path = Path(str(evaluation.get("splits", ""))).expanduser()
    if splits_path.exists():
        splits = pd.read_csv(splits_path)
        provenance["splits_csv"] = str(splits_path)
        provenance["split"] = _split_provenance(splits)
        manifest_path = splits_path.parent / "case_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            provenance["case_manifest"] = str(manifest_path)
            provenance["case_name"] = manifest.get("case_name")
            provenance["case_metadata"] = manifest.get("case_metadata", {})
            provenance["domain_counts"] = manifest.get("domain_counts", {})
            provenance["panel"] = manifest.get("panel", {})
    return provenance


def _split_provenance(splits: pd.DataFrame) -> dict[str, Any]:
    split_counts = {str(key): int(value) for key, value in splits["split"].astype(str).value_counts().sort_index().items()}
    group_key = None
    if "split_group_key" in splits.columns:
        values = splits["split_group_key"].dropna().astype(str).unique().tolist()
        group_key = values[0] if values else None
    holdout = splits[splits["split"].astype(str) != "train"]
    holdout_groups = []
    if "block_id" in holdout.columns:
        holdout_groups = sorted(holdout["block_id"].dropna().astype(str).unique().tolist())
    return {
        "strategy": str(splits["split_strategy"].iloc[0]) if "split_strategy" in splits.columns and not splits.empty else None,
        "group_key": group_key,
        "counts": split_counts,
        "holdout_groups": holdout_groups[:100],
        "n_train": split_counts.get("train", 0),
        "n_val": split_counts.get("val", 0),
        "n_test": split_counts.get("test", 0),
    }


def _stgpt_version() -> str:
    try:
        return version("stgpt")
    except PackageNotFoundError:
        return "0.1.0"
