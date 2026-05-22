"""F4 figure: auditable evidence for the structure-context run."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import _io  # noqa: E402
from ._layout import place_panel_label  # noqa: E402
from .dynamics import DEFAULT_43CASE_RUN_IDS, DEFAULT_RUN_LABELS  # noqa: E402
from .export import save_figure  # noqa: E402
from .style import DOUBLE_COLUMN_IN, OKABE_ITO, apply_style  # noqa: E402

LABEL_METRICS: tuple[tuple[str, str], ...] = (
    ("eval_label_retrieval_top1", "Top-1"),
    ("eval_label_retrieval_top5", "Top-5"),
)


def plot_structure_context_evidence(
    evidence_summary: str | Path | pd.DataFrame,
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    pointer_audit: str | Path | pd.DataFrame | None = None,
    name: str = "f4_auditable_structure_context_evidence",
    run_ids: Sequence[str] = DEFAULT_43CASE_RUN_IDS,
    structure_run_id: str = "structure_context_m6_20k",
    formats: Sequence[str] = ("pdf", "png"),
    title: str | None = None,
) -> dict[str, Any]:
    """Render an auditable structure-context evidence panel.

    Reads summary metrics, the structure run's Spatho QC/manifest artifacts,
    prototype assignment table, and pointer audit rows. It never recomputes
    model metrics.
    """
    summary = _io.load_table(evidence_summary)
    summary_source = str(summary.attrs.get("source", "<dataframe>"))
    root = Path(run_dir).expanduser()
    export_dir = root / "spatho_export"
    qc_path = export_dir / "region_qc_report.json"
    manifest_path = export_dir / "evidence_manifest.json"
    prototype_path = export_dir / "prototype_assignments.parquet"

    warnings: list[str] = []
    for col in ("run_id", *[item[0] for item in LABEL_METRICS]):
        if col not in summary.columns:
            raise ValueError(f"evidence summary is missing required column '{col}'")
    selected = summary[summary["run_id"].astype(str).isin([str(item) for item in run_ids])].copy()
    if selected.empty:
        raise ValueError("no evidence-summary rows match the requested run_ids")
    if structure_run_id not in set(selected["run_id"].astype(str)):
        raise ValueError(f"structure_run_id '{structure_run_id}' is not present in the selected evidence summary")

    pointer_frame, pointer_source = _load_pointer_audit(pointer_audit, summary_source)
    pointer_row = _pointer_row(pointer_frame, structure_run_id)
    qc_payload = _read_json(qc_path)
    manifest_payload = _read_json(manifest_path)
    prototype_stats = _prototype_stats(prototype_path)
    structure_row = selected[selected["run_id"].astype(str) == structure_run_id].iloc[0].to_dict()

    apply_style()
    width = DOUBLE_COLUMN_IN * 0.78
    fig, axes = plt.subplots(2, 2, figsize=(width, width * 0.72), squeeze=False)
    flat_axes = [ax for row in axes for ax in row]
    _label_retrieval_panel(flat_axes[0], selected)
    _prototype_panel(flat_axes[1], selected, structure_run_id, prototype_stats)
    _audit_qc_panel(flat_axes[2], pointer_row, qc_payload, structure_row)
    _artifact_panel(flat_axes[3], manifest_payload, export_dir)
    for index, ax in enumerate(flat_axes):
        place_panel_label(ax, chr(ord("A") + index))
    if title:
        fig.suptitle(title, fontsize=9, fontweight="bold")

    artifact_paths = _manifest_artifact_paths(manifest_payload)
    checkpoint_hash = None
    provenance_payload = manifest_payload.get("provenance", {}) if isinstance(manifest_payload, dict) else {}
    if isinstance(provenance_payload, dict):
        checkpoint_hash = provenance_payload.get("checkpoint_hash")
    provenance = {
        "figure": "F4_structure_context_evidence",
        "source_evidence_summary": summary_source,
        "source_pointer_audit": pointer_source,
        "run_dir": str(root),
        "region_qc_report": str(qc_path),
        "evidence_manifest": str(manifest_path),
        "prototype_assignments": str(prototype_path),
        "run_ids": [str(item) for item in run_ids],
        "structure_run_id": structure_run_id,
        "metrics": [item[0] for item in LABEL_METRICS],
        "checkpoint_hashes": [checkpoint_hash] if checkpoint_hash else [],
        "export_artifacts": sorted(artifact_paths),
        "prototype_stats": prototype_stats,
        "pointer_errors": _number_or_none(pointer_row.get("pointer_errors")) if pointer_row else None,
        "image_coverage": _number_or_none(qc_payload.get("image_coverage")) if isinstance(qc_payload, dict) else None,
        "palette": "okabe_ito",
        "warnings": warnings,
        "status": "pass",
    }
    artifacts = save_figure(fig, output_dir, name, formats=formats, provenance=provenance)
    plt.close(fig)

    return {
        "status": "pass",
        "run_ids": [str(item) for item in run_ids],
        "structure_run_id": structure_run_id,
        "prototype_stats": prototype_stats,
        "warnings": warnings,
        "artifacts": artifacts,
    }


def _label_retrieval_panel(ax: plt.Axes, frame: pd.DataFrame) -> None:
    labels = [_short_label(run_id) for run_id in frame["run_id"].astype(str)]
    x = np.arange(len(labels), dtype=float)
    width = 0.34
    for index, (metric, label) in enumerate(LABEL_METRICS):
        values = pd.to_numeric(frame[metric], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        bars = ax.bar(x + (index - 0.5) * width, values, width=width, color=OKABE_ITO[index + 1], label=label)
        for bar, value in zip(bars, values, strict=False):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2g}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Structure label retrieval")
    ax.set_ylim(0.0, 1.12)
    ax.legend(loc="upper left")


def _prototype_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    structure_run_id: str,
    prototype_stats: dict[str, Any],
) -> None:
    rows = frame[frame["run_id"].astype(str).isin(("full_m6_contour_store_lambda_0_01_20k", structure_run_id))].copy()
    labels = [_short_label(run_id) for run_id in rows["run_id"].astype(str)]
    usage: list[float] = []
    confidence: list[float] = []
    for _, row in rows.iterrows():
        expected = _number_or_none(row.get("expected_prototypes")) or 0.0
        used = _number_or_none(row.get("prototype_usage_export_global")) or 0.0
        usage.append(float(used / expected) if expected else 0.0)
        confidence.append(float(_number_or_none(row.get("prototype_mean_confidence")) or 0.0))
    if rows["run_id"].astype(str).eq(structure_run_id).any() and prototype_stats.get("mean_confidence") is not None:
        index = rows["run_id"].astype(str).tolist().index(structure_run_id)
        confidence[index] = float(prototype_stats["mean_confidence"])

    x = np.arange(len(labels), dtype=float)
    width = 0.34
    ax.bar(x - width / 2, usage, width=width, color=OKABE_ITO[2], label="Prototype usage")
    ax.bar(x + width / 2, confidence, width=width, color=OKABE_ITO[4], label="Mean confidence")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Normalized value")
    ax.legend(loc="upper left")


def _audit_qc_panel(ax: plt.Axes, pointer_row: dict[str, Any], qc_payload: dict[str, Any], structure_row: dict[str, Any]) -> None:
    ax.set_axis_off()
    pointer_errors = int(_number_or_none(pointer_row.get("pointer_errors")) or 0) if pointer_row else 0
    sampled = int(_number_or_none(pointer_row.get("records_sampled")) or 0) if pointer_row else 0
    total = int(_number_or_none(pointer_row.get("records_total")) or 0) if pointer_row else 0
    image_coverage = _number_or_none(qc_payload.get("image_coverage")) if isinstance(qc_payload, dict) else None
    n_regions = int(_number_or_none(qc_payload.get("n_regions_total")) or _number_or_none(structure_row.get("prototype_assignment_rows")) or 0)
    n_cells = int(_number_or_none(qc_payload.get("n_cells_assigned")) or 0)
    lines = [
        "Audit and QC",
        f"Pointer errors: {pointer_errors}",
        f"Sampled pointers: {sampled:,} / {total:,}",
        f"Image coverage: {image_coverage:.1%}" if image_coverage is not None else "Image coverage: n/a",
        f"Regions exported: {n_regions:,}",
        f"Cells assigned: {n_cells:,}",
    ]
    ax.text(0.02, 0.95, "\n".join(lines), transform=ax.transAxes, va="top", ha="left", fontsize=7)


def _artifact_panel(ax: plt.Axes, manifest_payload: dict[str, Any], export_dir: Path) -> None:
    artifacts = _manifest_artifact_paths(manifest_payload)
    sizes: list[tuple[str, float]] = []
    for key, path in artifacts.items():
        resolved = Path(path)
        if not resolved.exists():
            resolved = export_dir / resolved.name
        if resolved.exists():
            sizes.append((key.replace("_", "\n"), resolved.stat().st_size / (1024.0 * 1024.0)))
    sizes = sorted(sizes, key=lambda item: item[1], reverse=True)[:6]
    if not sizes:
        ax.text(0.5, 0.5, "No artifact sizes", ha="center", va="center")
        ax.set_axis_off()
        return
    labels, values = zip(*sizes, strict=True)
    y = np.arange(len(labels), dtype=float)
    ax.barh(y, values, color=OKABE_ITO[0])
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Artifact size (MB)")


def _load_pointer_audit(pointer_audit: str | Path | pd.DataFrame | None, summary_source: str) -> tuple[pd.DataFrame, str | None]:
    if pointer_audit is not None:
        frame = _io.load_table(pointer_audit)
        return frame, str(frame.attrs.get("source", "<dataframe>"))
    if summary_source and summary_source != "<dataframe>":
        candidate = Path(summary_source).expanduser().with_name("pointer_audit.csv")
        if candidate.exists():
            frame = _io.load_table(candidate)
            return frame, str(candidate)
    return pd.DataFrame(), None


def _pointer_row(frame: pd.DataFrame, run_id: str) -> dict[str, Any]:
    if frame.empty or "run_id" not in frame.columns:
        return {}
    rows = frame[frame["run_id"].astype(str) == str(run_id)]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _prototype_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "present": False}
    try:
        frame = pd.read_parquet(path, columns=["prototype_id", "prototype_confidence"])
    except Exception:
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            return {"path": str(path), "present": True, "error": str(exc)}
    result: dict[str, Any] = {"path": str(path), "present": True, "rows": int(len(frame))}
    if "prototype_id" in frame.columns:
        result["unique_prototypes"] = int(frame["prototype_id"].dropna().nunique())
    if "prototype_confidence" in frame.columns:
        result["mean_confidence"] = float(pd.to_numeric(frame["prototype_confidence"], errors="coerce").mean())
    return result


def _manifest_artifact_paths(manifest_payload: dict[str, Any]) -> dict[str, str]:
    artifacts = manifest_payload.get("artifacts", {}) if isinstance(manifest_payload, dict) else {}
    if not isinstance(artifacts, dict):
        return {}
    return {str(key): str(value) for key, value in artifacts.items()}


def _short_label(run_id: str) -> str:
    return DEFAULT_RUN_LABELS.get(run_id, run_id.replace("_", " "))


def _number_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
