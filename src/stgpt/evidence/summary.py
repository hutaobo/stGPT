from __future__ import annotations

import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field

ImageSource = Literal["contour_store", "image_path", "zero_fallback"]
RunStatus = Literal["pass", "warning", "missing"]


class EvidenceRunSpec(BaseModel):
    """One artifact-first evidence run declared in an evidence suite."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    tissue: str
    condition: str
    config_path: str
    run_dir: str
    expected_image_source: ImageSource | None = None
    expected_prototypes: int | None = Field(default=None, ge=0)
    requires_training_metrics: bool = True
    requires_alignment_telemetry: bool = False
    checkpoint_role: Literal["last", "best_loss", "best_alignment", "random"] | None = None
    lambda_align: float | None = Field(default=None, ge=0.0)
    suite_stage: str | None = None


class EvidenceSuiteSpec(BaseModel):
    """Artifact-first evidence suite specification."""

    model_config = ConfigDict(extra="forbid")

    suite_name: str
    runs: list[EvidenceRunSpec] = Field(default_factory=list)


def load_evidence_suite(path: str | Path) -> EvidenceSuiteSpec:
    """Load and validate an evidence suite YAML/JSON file."""
    suite_path = Path(path).expanduser()
    payload = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Evidence suite must contain a mapping: {suite_path}")
    return EvidenceSuiteSpec.model_validate(payload)


def summarize_evidence_suite(
    suite: EvidenceSuiteSpec | str | Path,
    output_dir: str | Path,
    *,
    pointer_sample_size: int = 50,
) -> dict[str, Any]:
    """Summarize an artifact-first evidence suite without launching training or export.

    The harness reads existing configs, metrics, Spatho exports, and optional
    evaluation outputs. Missing artifacts are reported in ``run_status.json``;
    they are never produced by this function.
    """
    suite_path = Path(suite).expanduser() if isinstance(suite, (str, Path)) else None
    spec = load_evidence_suite(suite_path) if suite_path is not None else suite
    if pointer_sample_size < 0:
        raise ValueError("pointer_sample_size must be non-negative")

    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    pointer_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    learning_rows: list[dict[str, Any]] = []

    for run in spec.runs:
        run_dir = _resolve_suite_path(run.run_dir, suite_path)
        config_path = _resolve_suite_path(run.config_path, suite_path)
        run_summary, pointer_summary, status = _summarize_run(
            run,
            run_dir=run_dir,
            config_path=config_path,
            pointer_sample_size=pointer_sample_size,
        )
        summary_rows.append(run_summary)
        pointer_rows.append(pointer_summary)
        status_rows.append(status)
        learning_rows.extend(_learning_dynamics_rows(run, metrics_path=run_dir / "train" / "metrics.json"))

    summary_frame = pd.DataFrame(summary_rows)
    pointer_frame = pd.DataFrame(pointer_rows)
    status_payload = {
        "suite_name": spec.suite_name,
        "status": _suite_status(status_rows),
        "runs": status_rows,
    }

    summary_csv = out / "evidence_summary.csv"
    summary_json = out / "evidence_summary.json"
    summary_md = out / "evidence_summary.md"
    pointer_csv = out / "pointer_audit.csv"
    pointer_json = out / "pointer_audit.json"
    status_json = out / "run_status.json"
    paper_table = out / "paper_table.md"
    contour_attribution = out / "contour_attribution.md"
    learning_csv = out / "learning_dynamics.csv"
    learning_json = out / "learning_dynamics.json"
    learning_md = out / "learning_dynamics.md"
    pareto_csv = out / "pareto_frontier.csv"
    pareto_md = out / "pareto_frontier.md"

    summary_frame.to_csv(summary_csv, index=False)
    learning_frame = pd.DataFrame(learning_rows)
    pareto_frame = pd.DataFrame(_pareto_rows(summary_rows))
    summary_json.write_text(json.dumps(_json_safe(summary_rows), indent=2), encoding="utf-8")
    summary_md.write_text(_summary_markdown(spec.suite_name, summary_rows), encoding="utf-8")
    pointer_frame.to_csv(pointer_csv, index=False)
    pointer_json.write_text(json.dumps(_json_safe(pointer_rows), indent=2), encoding="utf-8")
    status_json.write_text(json.dumps(_json_safe(status_payload), indent=2), encoding="utf-8")
    paper_table.write_text(_paper_table_markdown(spec.suite_name, summary_rows), encoding="utf-8")
    contour_attribution.write_text(_contour_attribution_markdown(spec.suite_name, summary_rows), encoding="utf-8")
    learning_frame.to_csv(learning_csv, index=False)
    learning_json.write_text(json.dumps(_json_safe(learning_rows), indent=2), encoding="utf-8")
    learning_md.write_text(_learning_dynamics_markdown(spec.suite_name, learning_rows), encoding="utf-8")
    pareto_frame.to_csv(pareto_csv, index=False)
    pareto_md.write_text(_pareto_markdown(spec.suite_name, pareto_frame.to_dict(orient="records")), encoding="utf-8")

    return {
        "suite_name": spec.suite_name,
        "status": status_payload["status"],
        "n_runs": len(summary_rows),
        "artifacts": {
            "evidence_summary_csv": str(summary_csv),
            "evidence_summary_json": str(summary_json),
            "evidence_summary_md": str(summary_md),
            "pointer_audit_csv": str(pointer_csv),
            "pointer_audit_json": str(pointer_json),
            "run_status": str(status_json),
            "paper_table": str(paper_table),
            "contour_attribution": str(contour_attribution),
            "learning_dynamics_csv": str(learning_csv),
            "learning_dynamics_json": str(learning_json),
            "learning_dynamics_md": str(learning_md),
            "pareto_frontier_csv": str(pareto_csv),
            "pareto_frontier_md": str(pareto_md),
        },
    }


def audit_evidence_pointers(
    evidence_chain: str | Path,
    *,
    export_dir: str | Path | None = None,
    expected_image_source: ImageSource | None = None,
    sample_size: int = 50,
) -> dict[str, Any]:
    """Audit sampled pointer-only evidence chain records."""
    if sample_size < 0:
        raise ValueError("sample_size must be non-negative")
    path = Path(evidence_chain).expanduser()
    base = Path(export_dir).expanduser() if export_dir is not None else path.parent
    if not path.exists():
        return {
            "records_total": 0,
            "records_sampled": 0,
            "pointer_errors": 1,
            "image_source_counts": {},
            "expected_image_source_mismatches": 0,
            "zarr_pointer_errors": 0,
            "parquet_pointer_errors": 0,
            "missing_provenance_hashes": 0,
            "missing_paths": 0,
            "error_examples": [f"missing evidence chain: {path}"],
        }

    image_sources: Counter[str] = Counter()
    errors: list[str] = []
    zarr_errors = 0
    parquet_errors = 0
    missing_hashes = 0
    missing_paths = 0
    mismatches = 0
    total = 0
    sampled = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            total += 1
            if sampled >= sample_size:
                continue
            sampled += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_idx}: invalid json: {exc}")
                continue
            result = _audit_one_pointer_record(
                record,
                base,
                expected_image_source=expected_image_source,
                line_idx=line_idx,
            )
            image_sources.update(result["image_source_counts"])
            errors.extend(result["errors"])
            zarr_errors += int(result["zarr_pointer_errors"])
            parquet_errors += int(result["parquet_pointer_errors"])
            missing_hashes += int(result["missing_provenance_hashes"])
            missing_paths += int(result["missing_paths"])
            mismatches += int(result["expected_image_source_mismatches"])

    return {
        "records_total": int(total),
        "records_sampled": int(sampled),
        "pointer_errors": int(len(errors)),
        "image_source_counts": dict(sorted(image_sources.items())),
        "expected_image_source_mismatches": int(mismatches),
        "zarr_pointer_errors": int(zarr_errors),
        "parquet_pointer_errors": int(parquet_errors),
        "missing_provenance_hashes": int(missing_hashes),
        "missing_paths": int(missing_paths),
        "error_examples": errors[:10],
    }


def _summarize_run(
    run: EvidenceRunSpec,
    *,
    run_dir: Path,
    config_path: Path,
    pointer_sample_size: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    train_dir = run_dir / "train"
    export_dir = run_dir / "spatho_export"
    metrics_path = train_dir / "metrics.json"
    evidence_chain_path = export_dir / "contour_evidence_chains.jsonl"
    prototype_path = export_dir / "prototype_assignments.parquet"

    missing: list[str] = []
    warnings: list[str] = []
    if not config_path.exists():
        missing.append("config_path")
    if not run_dir.exists():
        missing.append("run_dir")
    if run.requires_training_metrics and not metrics_path.exists():
        missing.append("train/metrics.json")
    if not evidence_chain_path.exists():
        missing.append("spatho_export/contour_evidence_chains.jsonl")
    if (run.expected_prototypes or 0) > 0 and not prototype_path.exists():
        missing.append("spatho_export/prototype_assignments.parquet")

    metrics, first_metric, last_metric = _read_metrics(metrics_path)
    pointer_summary = audit_evidence_pointers(
        evidence_chain_path,
        export_dir=export_dir,
        expected_image_source=run.expected_image_source,
        sample_size=pointer_sample_size,
    )
    prototype_summary = _prototype_summary(prototype_path, expected_prototypes=run.expected_prototypes)
    evaluation_summary = _evaluation_summary(run_dir)
    wall_clock_sec, steps_per_sec = _wall_clock(train_dir, len(metrics))

    if pointer_summary["pointer_errors"]:
        warnings.append("pointer_audit_errors")
    if pointer_summary["expected_image_source_mismatches"]:
        warnings.append("expected_image_source_mismatch")
    if pointer_summary["missing_provenance_hashes"]:
        warnings.append("missing_provenance_hashes")
    sinkhorn_nonfinite = _safe_float(last_metric.get("sinkhorn_nonfinite_count"))
    if sinkhorn_nonfinite is not None and sinkhorn_nonfinite > 0:
        warnings.append("sinkhorn_nonfinite")
    if run.expected_prototypes is not None and prototype_summary["prototype_used_export_global"] is not None:
        if prototype_summary["prototype_used_export_global"] > run.expected_prototypes:
            warnings.append("prototype_usage_exceeds_expected")
    if run.requires_alignment_telemetry and metrics and not any(_safe_float(row.get("val_alignment_score")) is not None for row in metrics):
        warnings.append("missing_alignment_telemetry")

    status: RunStatus = "missing" if missing else "warning" if warnings else "pass"
    dominant_source = _dominant_source(pointer_summary["image_source_counts"])
    summary = {
        "run_id": run.run_id,
        "tissue": run.tissue,
        "condition": run.condition,
        "status": status,
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "steps": int(len(metrics)),
        "expected_image_source": run.expected_image_source,
        "image_source": dominant_source,
        "image_source_counts": json.dumps(pointer_summary["image_source_counts"], sort_keys=True),
        "expected_prototypes": run.expected_prototypes,
        "requires_training_metrics": bool(run.requires_training_metrics),
        "requires_alignment_telemetry": bool(run.requires_alignment_telemetry),
        "checkpoint_role": run.checkpoint_role,
        "lambda_align": run.lambda_align,
        "suite_stage": run.suite_stage,
        "loss_start": _safe_float(first_metric.get("loss")),
        "loss_final": _safe_float(last_metric.get("loss")),
        "val_gene_loss_start": _safe_float(first_metric.get("val_gene_loss")),
        "val_gene_loss_final": _safe_float(last_metric.get("val_gene_loss")),
        "gene_loss_final": _safe_float(last_metric.get("gene_loss")),
        "neighbor_loss_final": _safe_float(last_metric.get("neighbor_loss")),
        "image_gene_loss_final": _safe_float(last_metric.get("image_gene_loss")),
        "prototype_loss_final": _safe_float(last_metric.get("prototype_loss")),
        "prototype_entropy_normalized_final": _safe_float(last_metric.get("prototype_entropy_normalized")),
        "prototype_usage_batch_final": _safe_float(last_metric.get("prototype_usage_count")),
        "prototype_dead_batch_final": _safe_float(last_metric.get("prototype_dead_codes")),
        "prototype_usage_export_global": prototype_summary["prototype_used_export_global"],
        "prototype_dead_export_global": prototype_summary["prototype_dead_export_global"],
        "prototype_assignment_rows": prototype_summary["prototype_assignment_rows"],
        "prototype_mean_confidence": prototype_summary["prototype_mean_confidence"],
        "prototype_median_confidence": prototype_summary["prototype_median_confidence"],
        "sinkhorn_nonfinite_count": sinkhorn_nonfinite,
        "sinkhorn_row_residual_final": _safe_float(last_metric.get("sinkhorn_row_residual")),
        "sinkhorn_col_residual_final": _safe_float(last_metric.get("sinkhorn_col_residual")),
        "pointer_records_total": pointer_summary["records_total"],
        "pointer_records_sampled": pointer_summary["records_sampled"],
        "pointer_errors": pointer_summary["pointer_errors"],
        "missing_provenance_hashes": pointer_summary["missing_provenance_hashes"],
        "train_wall_clock_sec": wall_clock_sec,
        "steps_per_sec": steps_per_sec,
        "gpu_utilization": "N/A (CPU run)",
        **evaluation_summary,
    }
    pointer_row = {
        "run_id": run.run_id,
        "tissue": run.tissue,
        "condition": run.condition,
        "expected_image_source": run.expected_image_source,
        **pointer_summary,
        "image_source_counts": json.dumps(pointer_summary["image_source_counts"], sort_keys=True),
        "error_examples": json.dumps(pointer_summary["error_examples"], ensure_ascii=False),
    }
    status_row = {
        "run_id": run.run_id,
        "status": status,
        "missing_artifacts": missing,
        "warnings": sorted(set(warnings)),
        "config_path": str(config_path),
        "run_dir": str(run_dir),
    }
    return summary, pointer_row, status_row


def _audit_one_pointer_record(
    record: dict[str, Any],
    export_dir: Path,
    *,
    expected_image_source: ImageSource | None,
    line_idx: int,
) -> dict[str, Any]:
    errors: list[str] = []
    zarr_errors = 0
    parquet_errors = 0
    missing_paths = 0
    missing_hashes = 0
    mismatches = 0

    source = _string_or_none(record.get("qc_verdict", {}).get("image_source")) or "unknown"
    if expected_image_source is not None and source != expected_image_source:
        mismatches += 1
        errors.append(f"line {line_idx}: expected image_source={expected_image_source}, got {source}")

    measured = record.get("measured_evidence", {})
    model = record.get("model_derived_evidence", {})
    image_ref = measured.get("image_ref", {})
    geometry_ref = measured.get("geometry_ref", {})
    molecular_ref = measured.get("molecular_ref", {})
    embedding_ref = model.get("embedding_ref", {})
    prototype_ref = model.get("prototype_ref", {})

    if source == "contour_store":
        image_artifact = _string_or_none(image_ref.get("artifact"))
        if image_artifact is None or not image_artifact.endswith("contour_image_store.zarr"):
            zarr_errors += 1
            errors.append(f"line {line_idx}: contour_store image_ref does not point to contour_image_store.zarr")
        if image_ref.get("row_index") is None:
            zarr_errors += 1
            errors.append(f"line {line_idx}: contour_store image_ref missing row_index")
        if not isinstance(image_ref.get("arrays"), dict):
            zarr_errors += 1
            errors.append(f"line {line_idx}: contour_store image_ref missing arrays map")
        missing_paths += _path_missing_count(image_artifact, export_dir, errors, line_idx=line_idx, label="image_ref")

        geometry_artifact = _string_or_none(geometry_ref.get("artifact"))
        if geometry_artifact is None or not geometry_artifact.endswith("contour_image_manifest.parquet"):
            parquet_errors += 1
            errors.append(f"line {line_idx}: contour_store geometry_ref does not point to contour_image_manifest.parquet")
        if geometry_ref.get("row_index") is None:
            parquet_errors += 1
            errors.append(f"line {line_idx}: contour_store geometry_ref missing row_index")
        missing_paths += _path_missing_count(geometry_artifact, export_dir, errors, line_idx=line_idx, label="geometry_ref")
    elif source == "image_path":
        image_artifact = _string_or_none(image_ref.get("artifact"))
        if image_artifact is None:
            errors.append(f"line {line_idx}: image_path source missing image_ref artifact")
        missing_paths += _path_missing_count(image_artifact, export_dir, errors, line_idx=line_idx, label="image_ref")
    elif source == "zero_fallback":
        image_artifact = _string_or_none(image_ref.get("artifact"))
        if image_artifact is None:
            errors.append(f"line {line_idx}: zero_fallback source missing image_ref artifact")
        missing_paths += _path_missing_count(image_artifact, export_dir, errors, line_idx=line_idx, label="image_ref")
    else:
        errors.append(f"line {line_idx}: unknown image_source={source}")

    for label, ref in (
        ("molecular_ref", molecular_ref),
        ("embedding_ref", embedding_ref),
        ("prototype_ref", prototype_ref),
    ):
        artifact = _string_or_none(ref.get("artifact")) if isinstance(ref, dict) else None
        if artifact is None:
            errors.append(f"line {line_idx}: {label} missing artifact")
            continue
        if ref.get("row_index") is None:
            errors.append(f"line {line_idx}: {label} missing row_index")
        missing_paths += _path_missing_count(artifact, export_dir, errors, line_idx=line_idx, label=label)

    provenance = record.get("provenance", {})
    for hash_key in ("config_hash", "checkpoint_hash"):
        if not _string_or_none(provenance.get(hash_key)):
            missing_hashes += 1
    if source == "contour_store" and not _string_or_none(provenance.get("contour_manifest_hash")):
        missing_hashes += 1

    return {
        "errors": errors,
        "image_source_counts": {source: 1},
        "expected_image_source_mismatches": mismatches,
        "zarr_pointer_errors": zarr_errors,
        "parquet_pointer_errors": parquet_errors,
        "missing_provenance_hashes": missing_hashes,
        "missing_paths": missing_paths,
    }


def _read_metrics(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if not path.exists():
        return [], {}, {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return [], {}, {}
    rows = [row for row in payload if isinstance(row, dict)]
    return rows, rows[0] if rows else {}, rows[-1] if rows else {}


def _learning_dynamics_rows(run: EvidenceRunSpec, *, metrics_path: Path) -> list[dict[str, Any]]:
    metrics, _, _ = _read_metrics(metrics_path)
    rows: list[dict[str, Any]] = []
    for index, metric in enumerate(metrics, start=1):
        val_loss = _safe_float(metric.get("val_gene_loss"))
        image_to_gene = _safe_float(metric.get("val_image_to_gene_top5"))
        gene_to_image = _safe_float(metric.get("val_gene_to_image_top5"))
        alignment = _safe_float(metric.get("val_alignment_score"))
        if val_loss is None and image_to_gene is None and gene_to_image is None and alignment is None:
            continue
        rows.append(
            {
                "run_id": run.run_id,
                "tissue": run.tissue,
                "condition": run.condition,
                "suite_stage": run.suite_stage,
                "checkpoint_role": run.checkpoint_role,
                "lambda_align": run.lambda_align,
                "step": _safe_int(metric.get("step")) or index,
                "lr": _safe_float(metric.get("lr")),
                "val_gene_loss": val_loss,
                "image_to_gene_top5": image_to_gene,
                "gene_to_image_top5": gene_to_image,
                "alignment_score": alignment,
                "image_gene_loss_weight": _safe_float(metric.get("image_gene_loss_weight")),
                "neighborhood_loss_weight": _safe_float(metric.get("neighborhood_loss_weight")),
                "structure_loss_weight": _safe_float(metric.get("structure_loss_weight")),
                "prototype_loss_weight": _safe_float(metric.get("prototype_loss_weight")),
            }
        )
    return rows


def _prototype_summary(path: Path, *, expected_prototypes: int | None) -> dict[str, Any]:
    empty = {
        "prototype_used_export_global": None,
        "prototype_dead_export_global": None,
        "prototype_assignment_rows": None,
        "prototype_mean_confidence": None,
        "prototype_median_confidence": None,
    }
    if not path.exists():
        return empty
    frame = pd.read_parquet(path)
    if frame.empty or "prototype_id" not in frame.columns:
        empty["prototype_assignment_rows"] = int(len(frame))
        return empty
    ids = pd.to_numeric(frame["prototype_id"], errors="coerce")
    valid = ids[ids >= 0].dropna()
    used = int(valid.nunique()) if not valid.empty else 0
    expected = expected_prototypes if expected_prototypes is not None else None
    confidence = pd.to_numeric(frame.get("prototype_confidence"), errors="coerce") if "prototype_confidence" in frame else None
    confidence_values = confidence.dropna() if confidence is not None else None
    return {
        "prototype_used_export_global": used,
        "prototype_dead_export_global": max(0, int(expected) - used) if expected is not None else None,
        "prototype_assignment_rows": int(len(frame)),
        "prototype_mean_confidence": _safe_float(confidence_values.mean()) if confidence_values is not None and not confidence_values.empty else None,
        "prototype_median_confidence": _safe_float(confidence_values.median())
        if confidence_values is not None and not confidence_values.empty
        else None,
    }


def _evaluation_summary(run_dir: Path) -> dict[str, Any]:
    path = _find_evaluation_metrics(run_dir)
    if path is None:
        return {
            "evaluation_present": False,
            "eval_gene_mse": None,
            "eval_gene_correlation": None,
            "eval_image_to_gene_top1": None,
            "eval_gene_to_image_top1": None,
            "eval_image_to_gene_top5": None,
            "eval_gene_to_image_top5": None,
            "eval_label_retrieval_top1": None,
            "eval_label_retrieval_top5": None,
            "eval_silhouette_mean": None,
            "eval_batch_mixing_entropy_mean": None,
            "eval_failure_analysis_rows": None,
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    prediction = payload.get("overall_prediction", {})
    retrieval_top1 = _first_matching(payload.get("overall_retrieval", []), k=1)
    retrieval_top5 = _first_matching(payload.get("overall_retrieval", []), k=5)
    label_top1 = _first_matching(payload.get("overall_label_retrieval", []), k=1)
    label_top5 = _first_matching(payload.get("overall_label_retrieval", []), k=5)
    return {
        "evaluation_present": True,
        "eval_gene_mse": _safe_float(prediction.get("gene_mse")),
        "eval_gene_correlation": _safe_float(prediction.get("gene_correlation")),
        "eval_image_to_gene_top1": _safe_float(retrieval_top1.get("image_to_gene_topk")),
        "eval_gene_to_image_top1": _safe_float(retrieval_top1.get("gene_to_image_topk")),
        "eval_image_to_gene_top5": _safe_float(retrieval_top5.get("image_to_gene_topk")),
        "eval_gene_to_image_top5": _safe_float(retrieval_top5.get("gene_to_image_topk")),
        "eval_label_retrieval_top1": _safe_float(label_top1.get("same_label_recall")),
        "eval_label_retrieval_top5": _safe_float(label_top5.get("same_label_recall")),
        "eval_silhouette_mean": _mean_metric(payload.get("overall_embedding_qc", []), "silhouette"),
        "eval_batch_mixing_entropy_mean": _mean_metric(payload.get("overall_batch_mixing", []), "batch_mixing_entropy"),
        "eval_failure_analysis_rows": _safe_int(payload.get("n_failure_analysis_rows")),
    }


def _find_evaluation_metrics(run_dir: Path) -> Path | None:
    for relative in (
        Path("evaluation") / "evaluation_metrics.json",
        Path("eval") / "evaluation_metrics.json",
        Path("evaluate") / "evaluation_metrics.json",
    ):
        candidate = run_dir / relative
        if candidate.exists():
            return candidate
    return None


def _wall_clock(train_dir: Path, steps: int) -> tuple[float | None, float | None]:
    candidates = sorted(train_dir.glob("train_*.stdout.log"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not candidates or steps <= 0:
        return None, None
    stdout = candidates[0]
    start_name = stdout.name.replace(".stdout.log", ".started.txt")
    start_path = train_dir / start_name
    start = start_path.stat().st_ctime if start_path.exists() else stdout.stat().st_ctime
    end = stdout.stat().st_mtime
    seconds = max(0.0, float(end - start))
    return (seconds, float(steps / seconds)) if seconds > 0 else (None, None)


def _path_missing_count(artifact: str | None, export_dir: Path, errors: list[str], *, line_idx: int, label: str) -> int:
    if artifact is None:
        return 0
    path = _resolve_artifact(artifact, export_dir)
    if path.exists():
        return 0
    errors.append(f"line {line_idx}: {label} artifact missing on disk: {artifact}")
    return 1


def _resolve_artifact(artifact: str, export_dir: Path) -> Path:
    expanded = Path(os.path.expandvars(artifact)).expanduser()
    if expanded.is_absolute():
        return expanded
    return export_dir / expanded


def _resolve_suite_path(value: str, suite_path: Path | None) -> Path:
    path = Path(os.path.expandvars(value)).expanduser()
    if path.is_absolute():
        return path
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    if suite_path is not None:
        suite_candidate = (suite_path.parent / path).resolve()
        if suite_candidate.exists():
            return suite_candidate
        for parent in suite_path.resolve().parents:
            parent_candidate = (parent / path).resolve()
            if parent_candidate.exists():
                return parent_candidate
    return cwd_candidate


def _dominant_source(counts: dict[str, int]) -> str | None:
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _suite_status(rows: list[dict[str, Any]]) -> RunStatus:
    statuses = {str(row.get("status")) for row in rows}
    if "missing" in statuses:
        return "missing"
    if "warning" in statuses:
        return "warning"
    return "pass"


def _first_matching(rows: Any, *, k: int) -> dict[str, Any]:
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if isinstance(row, dict) and int(row.get("k", -1)) == k:
            return row
    return {}


def _mean_metric(rows: Any, key: str) -> float | None:
    if not isinstance(rows, list):
        return None
    values = [_safe_float(row.get(key)) for row in rows if isinstance(row, dict)]
    finite = [value for value in values if value is not None]
    return float(sum(finite) / len(finite)) if finite else None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value)
    return text if text and text.lower() not in {"nan", "none", "null"} else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _pareto_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("evaluation_present"):
            continue
        image_to_gene_top5 = _safe_float(row.get("eval_image_to_gene_top5"))
        gene_to_image_top5 = _safe_float(row.get("eval_gene_to_image_top5"))
        alignment_score = None
        if image_to_gene_top5 is not None and gene_to_image_top5 is not None:
            alignment_score = float((image_to_gene_top5 + gene_to_image_top5) / 2.0)
        out.append(
            {
                "run_id": row.get("run_id"),
                "tissue": row.get("tissue"),
                "condition": row.get("condition"),
                "suite_stage": row.get("suite_stage"),
                "checkpoint_role": row.get("checkpoint_role"),
                "lambda_align": row.get("lambda_align"),
                "image_source": row.get("image_source"),
                "val_gene_loss": row.get("val_gene_loss_final"),
                "gene_correlation": row.get("eval_gene_correlation"),
                "image_to_gene_top1": row.get("eval_image_to_gene_top1"),
                "image_to_gene_top5": image_to_gene_top5,
                "gene_to_image_top1": row.get("eval_gene_to_image_top1"),
                "gene_to_image_top5": gene_to_image_top5,
                "alignment_score": alignment_score,
                "label_retrieval_top1": row.get("eval_label_retrieval_top1"),
                "label_retrieval_top5": row.get("eval_label_retrieval_top5"),
                "silhouette": row.get("eval_silhouette_mean"),
                "prototype_usage": row.get("prototype_usage_export_global"),
                "expected_prototypes": row.get("expected_prototypes"),
            }
        )
    return out


def _summary_markdown(suite_name: str, rows: list[dict[str, Any]]) -> str:
    lines = [
        f"# Evidence Summary: {suite_name}",
        "",
        "| Run | Tissue | Condition | Status | Image source | Val gene loss | Prototype usage | Pointer errors |",
        "| :-- | :-- | :-- | :-- | :-- | :-- | :-- | --: |",
    ]
    for row in rows:
        lines.append(
            "| {run_id} | {tissue} | {condition} | {status} | {image_source} | {val_loss} | {prototype_usage} | {pointer_errors} |".format(
                run_id=row["run_id"],
                tissue=row["tissue"],
                condition=row["condition"],
                status=row["status"],
                image_source=row.get("image_source") or "missing",
                val_loss=_loss_transition(row.get("val_gene_loss_start"), row.get("val_gene_loss_final")),
                prototype_usage=_prototype_usage_text(row),
                pointer_errors=row.get("pointer_errors", 0),
            )
        )
    return "\n".join(lines) + "\n"


def _learning_dynamics_markdown(suite_name: str, rows: list[dict[str, Any]]) -> str:
    lines = [
        f"# Learning Dynamics: {suite_name}",
        "",
        "| Run | Tissue | Step | LR | Val gene loss | I->G@5 | G->I@5 | Alignment | Img-Gene w | Proto w |",
        "| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: |",
    ]
    if not rows:
        lines.append("| not recorded | - | 0 | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded |")
        return "\n".join(lines) + "\n"
    for row in rows:
        lines.append(
            "| {run_id} | {tissue} | {step} | {lr} | {val_loss} | {ig5} | {gi5} | {alignment} | {image_weight} | {prototype_weight} |".format(
                run_id=row.get("run_id"),
                tissue=row.get("tissue"),
                step=row.get("step"),
                lr=_format_number(row.get("lr"), digits=6),
                val_loss=_format_number(row.get("val_gene_loss")),
                ig5=_format_number(row.get("image_to_gene_top5")),
                gi5=_format_number(row.get("gene_to_image_top5")),
                alignment=_format_number(row.get("alignment_score")),
                image_weight=_format_number(row.get("image_gene_loss_weight")),
                prototype_weight=_format_number(row.get("prototype_loss_weight")),
            )
        )
    return "\n".join(lines) + "\n"


def _pareto_markdown(suite_name: str, rows: list[dict[str, Any]]) -> str:
    lines = [
        f"# Pareto Frontier: {suite_name}",
        "",
        "Contour-native alignment is summarized as mean(I->G@5, G->I@5). Lower val gene loss means stronger reconstruction; higher alignment means stronger morphology-molecular matching.",
        "",
        "| Tissue | Run | Stage | Role | Lambda | Val gene loss | Gene corr | I->G@5 | G->I@5 | Alignment | Prototypes |",
        "| :-- | :-- | :-- | :-- | --: | --: | --: | --: | --: | --: | :-- |",
    ]
    if not rows:
        lines.append("| not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | N/A |")
        return "\n".join(lines) + "\n"
    for row in rows:
        lines.append(
            "| {tissue} | {run_id} | {stage} | {role} | {lambda_align} | {val_loss} | {gene_corr} | {ig5} | {gi5} | {alignment} | {prototype_usage} |".format(
                tissue=row.get("tissue"),
                run_id=row.get("run_id"),
                stage=row.get("suite_stage") or "not recorded",
                role=row.get("checkpoint_role") or "not recorded",
                lambda_align=_format_number(row.get("lambda_align")),
                val_loss=_format_number(row.get("val_gene_loss")),
                gene_corr=_format_number(row.get("gene_correlation")),
                ig5=_format_number(row.get("image_to_gene_top5")),
                gi5=_format_number(row.get("gene_to_image_top5")),
                alignment=_format_number(row.get("alignment_score")),
                prototype_usage=_prototype_usage_pair(row),
            )
        )
    return "\n".join(lines) + "\n"


def _paper_table_markdown(suite_name: str, rows: list[dict[str, Any]]) -> str:
    lines = [
        f"# Paper Table: {suite_name}",
        "",
        "| Tissue | Config | Steps | Image source | Val gene loss | Gene corr | I→G@1 | I→G@5 | G→I@1 | G→I@5 | Label@1 | Label@5 | Silhouette | Prototype usage | Sinkhorn stable | Pointer errors | Throughput | GPU |",
        "| :-- | :-- | --: | :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: | :-- | :-- | --: | :-- | :-- |",
    ]
    for row in rows:
        lines.append(
            "| {tissue} | {condition} | {steps} | {image_source} | {val_loss} | {gene_corr} | {image_gene_top1} | {image_gene_top5} | {gene_image_top1} | {gene_image_top5} | {label_top1} | {label_top5} | {silhouette} | {prototype_usage} | {sinkhorn} | {pointer_errors} | {throughput} | {gpu} |".format(
                tissue=row["tissue"],
                condition=row["condition"],
                steps=row.get("steps", 0),
                image_source=row.get("image_source") or "missing",
                val_loss=_loss_transition(row.get("val_gene_loss_start"), row.get("val_gene_loss_final")),
                gene_corr=_format_number(row.get("eval_gene_correlation")),
                image_gene_top1=_format_number(row.get("eval_image_to_gene_top1")),
                image_gene_top5=_format_number(row.get("eval_image_to_gene_top5")),
                gene_image_top1=_format_number(row.get("eval_gene_to_image_top1")),
                gene_image_top5=_format_number(row.get("eval_gene_to_image_top5")),
                label_top1=_format_number(row.get("eval_label_retrieval_top1")),
                label_top5=_format_number(row.get("eval_label_retrieval_top5")),
                silhouette=_format_number(row.get("eval_silhouette_mean")),
                prototype_usage=_prototype_usage_text(row),
                sinkhorn=_sinkhorn_text(row),
                pointer_errors=row.get("pointer_errors", 0),
                throughput=_throughput_text(row.get("steps_per_sec"), row.get("train_wall_clock_sec")),
                gpu=row.get("gpu_utilization") or "N/A",
            )
        )
    return "\n".join(lines) + "\n"


def _contour_attribution_markdown(suite_name: str, rows: list[dict[str, Any]]) -> str:
    lines = [
        f"# Contour-Native Attribution: {suite_name}",
        "",
        "This report makes the contour-native claim explicit. The contour-region contract, row_index pointers, and packed contour store are held fixed; the comparison asks what is gained when the model can use contour H&E object/context/shape evidence instead of only gene and spatial tokens. A positive retrieval delta supports the contour-native evidence path; lower gene loss in the control should be interpreted as reconstruction specialization, not morphology alignment.",
        "",
        "## Within-Tissue Contour Lift",
        "",
        "| Tissue | Full M6 run | Control run | Δ I->G@1 | Δ Gene corr | Δ Val gene loss | Δ Silhouette | Interpretation |",
        "| :-- | :-- | :-- | --: | --: | --: | --: | :-- |",
    ]
    by_tissue: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_tissue.setdefault(str(row.get("tissue")), []).append(row)
    for tissue, tissue_rows in sorted(by_tissue.items()):
        full = _find_condition(tissue_rows, "Full M6 Zarr contour store")
        control = _find_condition(tissue_rows, "Contour-unit Gene+Spatial 500-step")
        if full is None or control is None:
            continue
        delta_retrieval = _delta(full.get("eval_image_to_gene_top1"), control.get("eval_image_to_gene_top1"))
        delta_gene_corr = _delta(full.get("eval_gene_correlation"), control.get("eval_gene_correlation"))
        delta_val_loss = _delta(full.get("val_gene_loss_final"), control.get("val_gene_loss_final"))
        delta_silhouette = _delta(full.get("eval_silhouette_mean"), control.get("eval_silhouette_mean"))
        interpretation = "contour H&E improves cross-modal retrieval" if delta_retrieval is not None and delta_retrieval > 0 else "no retrieval lift"
        lines.append(
            "| {tissue} | {full_run} | {control_run} | {delta_retrieval} | {delta_gene_corr} | {delta_val_loss} | {delta_silhouette} | {interpretation} |".format(
                tissue=tissue,
                full_run=full["run_id"],
                control_run=control["run_id"],
                delta_retrieval=_format_delta(delta_retrieval),
                delta_gene_corr=_format_delta(delta_gene_corr),
                delta_val_loss=_format_delta(delta_val_loss),
                delta_silhouette=_format_delta(delta_silhouette),
                interpretation=interpretation,
            )
        )
    transfer_lines = _cross_tissue_contour_transfer_lines(by_tissue)
    if transfer_lines:
        lines.extend(["", "## Cross-Tissue Contour Transfer", ""])
        lines.append(
            "These zero-shot rows evaluate a contour-trained checkpoint on a different tissue without retraining. Lift over random-init and contour-unit Gene+Spatial controls is the cleanest current test that contour-native embeddings transfer beyond one slide or tissue."
        )
        lines.extend(
            [
                "",
                "| Target tissue | Transfer run | I->G@1 | I->G@5 | G->I@1 | G->I@5 | Δ vs random I->G@1 | Δ vs control I->G@1 | Interpretation |",
                "| :-- | :-- | --: | --: | --: | --: | --: | --: | :-- |",
                *transfer_lines,
            ]
        )
    return "\n".join(lines) + "\n"


def _cross_tissue_contour_transfer_lines(by_tissue: dict[str, list[dict[str, Any]]]) -> list[str]:
    lines: list[str] = []
    for tissue, tissue_rows in sorted(by_tissue.items()):
        zero_shot_runs = [
            row
            for row in tissue_rows
            if str(row.get("condition", "")).lower().startswith("zero-shot")
        ]
        if not zero_shot_runs:
            continue
        random_floor = _find_condition(tissue_rows, "Full M6 contour-store random init")
        control = _find_condition(tissue_rows, "Contour-unit Gene+Spatial 500-step")
        for zero in zero_shot_runs:
            delta_random = _delta(zero.get("eval_image_to_gene_top1"), None if random_floor is None else random_floor.get("eval_image_to_gene_top1"))
            delta_control = _delta(zero.get("eval_image_to_gene_top1"), None if control is None else control.get("eval_image_to_gene_top1"))
            lines.append(
                "| {tissue} | {run_id} | {ig1} | {ig5} | {gi1} | {gi5} | {delta_random} | {delta_control} | {interpretation} |".format(
                    tissue=tissue,
                    run_id=zero["run_id"],
                    ig1=_format_number(zero.get("eval_image_to_gene_top1")),
                    ig5=_format_number(zero.get("eval_image_to_gene_top5")),
                    gi1=_format_number(zero.get("eval_gene_to_image_top1")),
                    gi5=_format_number(zero.get("eval_gene_to_image_top5")),
                    delta_random=_format_delta(delta_random),
                    delta_control=_format_delta(delta_control),
                    interpretation=_transfer_interpretation(delta_random, delta_control),
                )
            )
    return lines


def _transfer_interpretation(delta_random: float | None, delta_control: float | None) -> str:
    beats_random = delta_random is not None and delta_random > 0
    beats_control = delta_control is not None and delta_control > 0
    if beats_random and beats_control:
        return "cross-tissue contour signal transfers beyond random and gene+spatial control"
    if beats_random:
        return "cross-tissue contour signal beats random floor"
    return "no zero-shot contour transfer lift"


def _find_condition(rows: list[dict[str, Any]], condition: str) -> dict[str, Any] | None:
    for row in rows:
        if str(row.get("condition")) == condition:
            return row
    return None


def _delta(left: Any, right: Any) -> float | None:
    left_value = _safe_float(left)
    right_value = _safe_float(right)
    if left_value is None or right_value is None:
        return None
    return left_value - right_value


def _format_delta(value: float | None) -> str:
    return "not recorded" if value is None else f"{value:+.4f}"


def _loss_transition(start: Any, final: Any) -> str:
    left = _format_number(start)
    right = _format_number(final)
    return f"{left} -> {right}" if left != "not recorded" or right != "not recorded" else "not recorded"


def _prototype_usage_text(row: dict[str, Any]) -> str:
    expected = row.get("expected_prototypes")
    if not expected:
        return "N/A"
    batch = _format_number(row.get("prototype_usage_batch_final"), digits=0)
    global_used = _format_number(row.get("prototype_usage_export_global"), digits=0)
    return f"batch {batch}/{expected}; global {global_used}/{expected}"


def _prototype_usage_pair(row: dict[str, Any]) -> str:
    expected = row.get("expected_prototypes")
    if not expected:
        return "N/A"
    global_used = _format_number(row.get("prototype_usage"), digits=0)
    return f"{global_used}/{expected}"


def _sinkhorn_text(row: dict[str, Any]) -> str:
    if not row.get("expected_prototypes"):
        return "N/A"
    if not row.get("requires_training_metrics"):
        return "not recorded"
    nonfinite = row.get("sinkhorn_nonfinite_count")
    if nonfinite is None:
        return "not recorded"
    return "yes" if nonfinite in (0, 0.0) else "no"


def _throughput_text(steps_per_sec: Any, seconds: Any) -> str:
    rate = _safe_float(steps_per_sec)
    elapsed = _safe_float(seconds)
    if rate is None or elapsed is None:
        return "not recorded"
    return f"{rate:.3f} steps/sec ({elapsed:.1f}s)"


def _format_number(value: Any, *, digits: int = 4) -> str:
    number = _safe_float(value)
    if number is None:
        return "not recorded"
    if digits == 0:
        return str(int(round(number)))
    return f"{number:.{digits}f}"
