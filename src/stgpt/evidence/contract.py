from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from ..config import StGPTConfig

ARCHITECTURE_FIELDS = (
    "d_model",
    "n_heads",
    "n_layers",
    "dim_feedforward",
    "max_genes",
    "n_expression_bins",
    "image_size",
    "image_channels",
    "image_encoder_backend",
    "image_encoder_preset",
    "image_encoder_name",
    "image_embedding_dim",
    "max_cells_per_region",
    "n_prototypes",
    "prototype_temperature",
    "use_expression_values",
    "use_image_context",
    "use_spatial_context",
    "use_structure_context",
    "use_cell_context",
)


def check_artifact_contract(
    *,
    checkpoint: str | Path,
    config: str | Path,
    run_dir: str | Path | None = None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Validate artifact compatibility before evidence synthesis.

    This checker is deliberately read-only. It verifies that the checkpoint,
    config, prototype head, and training metrics describe the same run lineage
    before downstream evidence tools treat a best-alignment checkpoint as a
    stable scientific artifact.
    """
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    config_path = Path(config).expanduser().resolve()
    cfg = StGPTConfig.from_file(config_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    warnings: list[str] = []
    errors: list[str] = []
    checks: dict[str, Any] = {}

    checkpoint_config = payload.get("config")
    if not isinstance(checkpoint_config, dict):
        errors.append("checkpoint_missing_config")
        checkpoint_model: dict[str, Any] = {}
    else:
        checkpoint_model = checkpoint_config.get("model", {})
        if not isinstance(checkpoint_model, dict):
            errors.append("checkpoint_missing_model_config")
            checkpoint_model = {}

    _check_architecture(cfg, checkpoint_model, errors, warnings, checks)
    _check_prototype_shape(cfg, payload, errors, warnings, checks)
    _check_training_state(payload, errors, warnings, checks)

    if run_dir is not None:
        _check_metrics_continuity(Path(run_dir).expanduser().resolve(), payload, errors, warnings, checks)

    status = "fail" if errors else "warning" if warnings else "pass"
    result = {
        "status": status,
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "run_dir": str(Path(run_dir).expanduser().resolve()) if run_dir is not None else None,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }
    if output is not None:
        output_path = Path(output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _check_architecture(
    cfg: StGPTConfig,
    checkpoint_model: dict[str, Any],
    errors: list[str],
    warnings: list[str],
    checks: dict[str, Any],
) -> None:
    config_model = cfg.model.model_dump()
    mismatches = []
    missing = []
    for field in ARCHITECTURE_FIELDS:
        if field not in checkpoint_model:
            missing.append(field)
            continue
        if checkpoint_model.get(field) != config_model.get(field):
            mismatches.append(
                {
                    "field": field,
                    "config": config_model.get(field),
                    "checkpoint": checkpoint_model.get(field),
                }
            )
    checks["architecture_mismatches"] = mismatches
    checks["architecture_missing_fields"] = missing
    if mismatches:
        errors.append("architecture_config_mismatch")
    if missing:
        warnings.append("checkpoint_architecture_fields_missing")


def _check_prototype_shape(
    cfg: StGPTConfig,
    payload: dict[str, Any],
    errors: list[str],
    warnings: list[str],
    checks: dict[str, Any],
) -> None:
    model_state = payload.get("model_state", {})
    if not isinstance(model_state, dict):
        errors.append("checkpoint_missing_model_state")
        return
    n_prototypes = int(cfg.model.n_prototypes)
    prototype_key = next((key for key in model_state if key.endswith("prototype_head.weight")), None)
    prototype_tensor = model_state.get(prototype_key) if prototype_key else None
    checks["prototype_key"] = prototype_key
    checks["config_n_prototypes"] = n_prototypes
    if isinstance(prototype_tensor, torch.Tensor):
        checks["checkpoint_n_prototypes"] = int(prototype_tensor.shape[0])
        checks["checkpoint_prototype_dim"] = int(prototype_tensor.shape[1]) if prototype_tensor.ndim > 1 else None
        if int(prototype_tensor.shape[0]) != n_prototypes:
            errors.append("prototype_count_mismatch")
    elif n_prototypes > 0:
        warnings.append("prototype_tensor_missing")
        checks["checkpoint_n_prototypes"] = None
    else:
        checks["checkpoint_n_prototypes"] = 0


def _check_training_state(
    payload: dict[str, Any],
    errors: list[str],
    warnings: list[str],
    checks: dict[str, Any],
) -> None:
    summary = payload.get("training_summary", {})
    if not isinstance(summary, dict):
        errors.append("checkpoint_missing_training_summary")
        summary = {}
    checkpoint_step = _safe_int(summary.get("steps"), default=len(payload.get("metrics", []) or []))
    metrics = payload.get("metrics", [])
    checks["checkpoint_step"] = checkpoint_step
    checks["checkpoint_metrics_rows"] = len(metrics) if isinstance(metrics, list) else 0
    checks["best_alignment_metric"] = _safe_float(summary.get("best_alignment_metric"))
    checks["best_metric"] = _safe_float(summary.get("best_metric"))
    if checkpoint_step > 0 and payload.get("optimizer_state") is None:
        warnings.append("optimizer_state_missing_for_trained_checkpoint")
    if checkpoint_step > 0 and isinstance(metrics, list) and metrics:
        last_step = _metric_step(metrics[-1], default=len(metrics))
        checks["checkpoint_metrics_last_step"] = last_step
        if last_step != checkpoint_step:
            warnings.append("checkpoint_metrics_step_mismatch")


def _check_metrics_continuity(
    run_dir: Path,
    payload: dict[str, Any],
    errors: list[str],
    warnings: list[str],
    checks: dict[str, Any],
) -> None:
    metrics_path = run_dir / "train" / "metrics.json"
    if not metrics_path.exists():
        metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        warnings.append("run_metrics_missing")
        checks["run_metrics_path"] = None
        return
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.append("run_metrics_unreadable")
        checks["run_metrics_error"] = str(exc)
        return
    if not isinstance(metrics, list) or not metrics:
        warnings.append("run_metrics_empty")
        checks["run_metrics_rows"] = 0
        return
    steps = [_metric_step(row, default=idx + 1) for idx, row in enumerate(metrics) if isinstance(row, dict)]
    checkpoint_summary = payload.get("training_summary", {}) if isinstance(payload.get("training_summary"), dict) else {}
    checkpoint_step = _safe_int(checkpoint_summary.get("steps"), default=len(payload.get("metrics", []) or []))
    discontinuities = [
        {"previous": int(prev), "current": int(curr)}
        for prev, curr in zip(steps, steps[1:], strict=False)
        if curr - prev != 1
    ][:25]
    checks["run_metrics_path"] = str(metrics_path)
    checks["run_metrics_rows"] = len(metrics)
    checks["run_metrics_max_step"] = int(max(steps)) if steps else 0
    checks["run_metrics_discontinuities"] = discontinuities
    if discontinuities:
        warnings.append("run_metrics_step_discontinuity")
    if steps and checkpoint_step > max(steps):
        errors.append("checkpoint_step_ahead_of_run_metrics")
    elif steps and checkpoint_step not in set(steps):
        warnings.append("checkpoint_step_not_present_in_run_metrics")


def _metric_step(row: dict[str, Any], *, default: int) -> int:
    return _safe_int(row.get("step") or row.get("global_step"), default=default)


def _safe_int(value: Any, *, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
