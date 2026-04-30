from __future__ import annotations

import json
import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

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
        "panel_metadata": payload.get("panel_metadata", {}),
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
    return f"""# {metadata["model_name"]}

This packaged stGPT checkpoint is a Xenium-first morpho-molecular foundation-model backend in development.

## Intended Use

{metadata["intended_use"]}

## Scope

- Case: {metadata.get("case_name")}
- Data mode: {metadata.get("data_mode")}
- Model version: {metadata.get("model_version")}

## Limitations and Guardrails

{limitations}
"""


def _stgpt_version() -> str:
    try:
        return version("stgpt")
    except PackageNotFoundError:
        return "0.1.0"
