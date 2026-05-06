from __future__ import annotations

import json
import math
import os
from collections import Counter
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import zarr
from PIL import Image


def build_failure_gallery(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    max_items: int = 24,
    top_genes: int = 8,
    rare_prototype_fraction: float = 0.02,
) -> dict[str, Any]:
    """Build an artifact-first gallery of contour-level failure candidates.

    This triage layer never launches inference. It reads existing M7 evidence
    chains, prototype assignments, region metadata, and optional evaluation
    artifacts, then ranks contours for human review and later targeted ablation.
    """
    if max_items < 0:
        raise ValueError("max_items must be non-negative")
    if top_genes < 0:
        raise ValueError("top_genes must be non-negative")
    if rare_prototype_fraction < 0:
        raise ValueError("rare_prototype_fraction must be non-negative")

    run_path = Path(run_dir).expanduser()
    out = Path(output_dir).expanduser()
    tiles = out / "tiles"
    out.mkdir(parents=True, exist_ok=True)
    tiles.mkdir(parents=True, exist_ok=True)

    export_dir = run_path / "spatho_export"
    evidence_chain = export_dir / "contour_evidence_chains.jsonl"
    records = _read_jsonl(evidence_chain)
    prototype_frame = _read_optional_parquet(export_dir / "prototype_assignments.parquet")
    region_frame = _read_region_metadata(export_dir / "region_embeddings.parquet")
    failure_frame = _read_optional_csv(run_path / "evaluation" / "failure_analysis.csv")
    prototype_counts = _prototype_counts(records, prototype_frame)
    rare_threshold = max(1, int(math.ceil(max(1, sum(prototype_counts.values())) * rare_prototype_fraction)))

    rows = []
    for index, record in enumerate(records):
        rows.append(
            _failure_row(
                record,
                index=index,
                export_dir=export_dir,
                prototype_frame=prototype_frame,
                region_frame=region_frame,
                prototype_counts=prototype_counts,
                rare_threshold=rare_threshold,
            )
        )
    scored = _score_rows(rows)
    scored = sorted(scored, key=lambda row: (-float(row.get("failure_score") or 0.0), str(row.get("contour_id") or "")))
    for rank, row in enumerate(scored, start=1):
        row["failure_rank"] = rank
        row["export_for_ablation"] = bool(
            row.get("image_source") == "contour_store"
            and (
                rank <= max(1, min(max_items, 12))
                or "structure_blind" in str(row.get("failure_reasons") or "")
                or "hallucination_risk" in str(row.get("failure_reasons") or "")
            )
        )

    zarr_cache: dict[Path, zarr.Group] = {}
    gallery_rows = scored[:max_items]
    for row in gallery_rows:
        row["top_genes"] = json.dumps(
            _top_gene_rows(
                _dict(_dict(records[int(row["record_index"])].get("measured_evidence")).get("molecular_ref")),
                base_dir=export_dir,
                top_n=top_genes,
            )
        )
        _attach_tiles(row, records[int(row["record_index"])], export_dir=export_dir, tiles_dir=tiles, zarr_cache=zarr_cache)

    gallery_csv = out / "failure_gallery.csv"
    gallery_json = out / "failure_gallery.json"
    gallery_html = out / "failure_gallery.html"
    ablation_csv = out / "ablation_targets.csv"
    ablation_json = out / "ablation_targets.json"
    summary_json = out / "failure_summary.json"

    pd.DataFrame(scored).to_csv(gallery_csv, index=False)
    gallery_json.write_text(json.dumps(_json_safe(scored), indent=2), encoding="utf-8")
    ablation_rows = [row for row in scored if row.get("export_for_ablation")]
    pd.DataFrame(ablation_rows).to_csv(ablation_csv, index=False)
    ablation_json.write_text(json.dumps(_json_safe(ablation_rows), indent=2), encoding="utf-8")
    summary_payload = _summary_payload(scored, failure_frame=failure_frame, rare_threshold=rare_threshold)
    summary_json.write_text(json.dumps(_json_safe(summary_payload), indent=2), encoding="utf-8")
    gallery_html.write_text(_gallery_html(run_path, gallery_rows, summary_payload), encoding="utf-8")

    return {
        "run_dir": str(run_path),
        "n_records": len(records),
        "n_gallery_items": len(gallery_rows),
        "n_ablation_targets": len(ablation_rows),
        "artifacts": {
            "failure_gallery_csv": str(gallery_csv),
            "failure_gallery_json": str(gallery_json),
            "failure_gallery_html": str(gallery_html),
            "ablation_targets_csv": str(ablation_csv),
            "ablation_targets_json": str(ablation_json),
            "failure_summary": str(summary_json),
            "tiles_dir": str(tiles),
        },
    }


def _failure_row(
    record: dict[str, Any],
    *,
    index: int,
    export_dir: Path,
    prototype_frame: pd.DataFrame,
    region_frame: pd.DataFrame,
    prototype_counts: Counter[int],
    rare_threshold: int,
) -> dict[str, Any]:
    unit = _dict(record.get("unit"))
    measured = _dict(record.get("measured_evidence"))
    model = _dict(record.get("model_derived_evidence"))
    qc = _dict(record.get("qc_verdict"))
    provenance = _dict(record.get("provenance"))
    proto_ref = _dict(model.get("prototype_ref"))
    embedding_index = _safe_int(unit.get("embedding_row_index")) or _safe_int(proto_ref.get("row_index")) or index
    prototype_row = _frame_row(prototype_frame, embedding_index)
    region_row = _frame_row(region_frame, embedding_index)
    prototype_id = _safe_int(proto_ref.get("prototype_id"))
    if prototype_id is None:
        prototype_id = _safe_int(prototype_row.get("prototype_id"))
    confidence = _safe_float(proto_ref.get("confidence"))
    if confidence is None:
        confidence = _safe_float(prototype_row.get("prototype_confidence"))
    entropy = _safe_float(proto_ref.get("assignment_entropy"))
    if entropy is None:
        entropy = _safe_float(prototype_row.get("assignment_entropy"))
    area = _first_float(region_row, "area", "geometry_area")
    perimeter = _first_float(region_row, "perimeter", "geometry_perimeter")
    eccentricity = _first_float(region_row, "eccentricity", "geometry_eccentricity")
    complexity = _geometry_complexity(area, perimeter)
    image_source = str(qc.get("image_source") or region_row.get("image_source") or "unknown")
    qc_flag = str(qc.get("qc_flag") or region_row.get("qc_flag") or "unknown")
    gt_similarity = _first_float(prototype_row, "ground_truth_similarity", "matched_gene_similarity", "image_gene_gt_similarity")
    top1_similarity = _first_float(prototype_row, "top1_similarity", "image_to_gene_top1_similarity", "retrieved_gene_similarity")
    hallucination_index, hallucination_basis = _hallucination_index(
        confidence=confidence,
        entropy=entropy,
        gt_similarity=gt_similarity,
        top1_similarity=top1_similarity,
    )
    rare_count = int(prototype_counts.get(int(prototype_id), 0)) if prototype_id is not None else 0
    missing_hashes = [
        key
        for key in ("checkpoint_hash", "config_hash", "contour_manifest_hash")
        if not _string_or_none(provenance.get(key))
    ]
    return {
        "record_index": index,
        "evidence_id": record.get("evidence_id"),
        "contour_id": unit.get("contour_id") or unit.get("region_id"),
        "slide_id": unit.get("slide_id") or region_row.get("slide_id"),
        "row_index": unit.get("row_index") or region_row.get("row_index"),
        "embedding_row_index": embedding_index,
        "structure_label": region_row.get("structure_label") or unit.get("structure_label"),
        "image_source": image_source,
        "qc_flag": qc_flag,
        "prototype_id": prototype_id,
        "prototype_count": rare_count,
        "prototype_confidence": confidence,
        "assignment_entropy": entropy,
        "area": area,
        "perimeter": perimeter,
        "eccentricity": eccentricity,
        "geometry_complexity": complexity,
        "ground_truth_similarity": gt_similarity,
        "top1_similarity": top1_similarity,
        "hallucination_index": hallucination_index,
        "hallucination_basis": hallucination_basis,
        "rare_prototype_threshold": rare_threshold,
        "is_rare_prototype": bool(prototype_id is not None and rare_count <= rare_threshold),
        "missing_provenance_hashes": ",".join(missing_hashes),
        "top_genes": "[]",
        "image_ref": json.dumps(measured.get("image_ref", {}), sort_keys=True),
        "molecular_ref": json.dumps(measured.get("molecular_ref", {}), sort_keys=True),
    }


def _score_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    confidence = np.asarray([_safe_float(row.get("prototype_confidence")) or np.nan for row in rows], dtype=np.float64)
    entropy = np.asarray([_safe_float(row.get("assignment_entropy")) or np.nan for row in rows], dtype=np.float64)
    complexity = np.asarray([_safe_float(row.get("geometry_complexity")) or np.nan for row in rows], dtype=np.float64)
    hallucination = np.asarray([_safe_float(row.get("hallucination_index")) or np.nan for row in rows], dtype=np.float64)
    low_confidence = 1.0 - _minmax(confidence)
    high_entropy = _minmax(entropy)
    high_complexity = _minmax(complexity)
    hallucination_score = _minmax(hallucination)
    for idx, row in enumerate(rows):
        rare = 1.0 if row.get("is_rare_prototype") else 0.0
        image_issue = 1.0 if row.get("image_source") != "contour_store" or row.get("qc_flag") not in {"ok", "pass"} else 0.0
        provenance_issue = 1.0 if row.get("missing_provenance_hashes") else 0.0
        score = (
            0.24 * low_confidence[idx]
            + 0.24 * high_entropy[idx]
            + 0.16 * high_complexity[idx]
            + 0.14 * rare
            + 0.10 * image_issue
            + 0.08 * hallucination_score[idx]
            + 0.04 * provenance_issue
        )
        reasons = _failure_reasons(
            row,
            low_confidence=low_confidence[idx],
            high_entropy=high_entropy[idx],
            high_complexity=high_complexity[idx],
            hallucination_score=hallucination_score[idx],
        )
        row["failure_score"] = float(score)
        row["failure_reasons"] = ",".join(reasons) if reasons else "review_candidate"
    return rows


def _failure_reasons(
    row: dict[str, Any],
    *,
    low_confidence: float,
    high_entropy: float,
    high_complexity: float,
    hallucination_score: float,
) -> list[str]:
    reasons: list[str] = []
    if low_confidence >= 0.75:
        reasons.append("low_confidence")
    if high_entropy >= 0.75:
        reasons.append("high_entropy")
    if high_entropy >= 0.60 and high_complexity >= 0.60:
        reasons.append("structure_blind")
    if row.get("is_rare_prototype"):
        reasons.append("prototype_sinkhole")
    if row.get("image_source") != "contour_store" or row.get("qc_flag") not in {"ok", "pass"}:
        reasons.append("image_or_qc_issue")
    if hallucination_score >= 0.80:
        reasons.append("hallucination_risk")
    if row.get("missing_provenance_hashes"):
        reasons.append("provenance_gap")
    return reasons


def _attach_tiles(
    row: dict[str, Any],
    record: dict[str, Any],
    *,
    export_dir: Path,
    tiles_dir: Path,
    zarr_cache: dict[Path, zarr.Group],
) -> None:
    bundle = _load_image_bundle(record, base_dir=export_dir, zarr_cache=zarr_cache)
    prefix = f"failure_{int(row['failure_rank']):03d}_{_slug(str(row.get('contour_id') or row.get('evidence_id') or 'contour'))}"
    object_path = tiles_dir / f"{prefix}_object.png"
    context_path = tiles_dir / f"{prefix}_context.png"
    mask_path = tiles_dir / f"{prefix}_mask.png"
    overlay_path = tiles_dir / f"{prefix}_overlay.png"
    _save_rgb(bundle["object_rgb"], object_path)
    _save_rgb(bundle["context_rgb"], context_path)
    _save_mask(bundle["mask"], mask_path)
    _save_rgb(_overlay_mask(bundle["object_rgb"], bundle["mask"]), overlay_path)
    row["object_image"] = _relative_tile(object_path, tiles_dir.parent)
    row["context_image"] = _relative_tile(context_path, tiles_dir.parent)
    row["mask_image"] = _relative_tile(mask_path, tiles_dir.parent)
    row["overlay_image"] = _relative_tile(overlay_path, tiles_dir.parent)
    row["geometry_preview"] = json.dumps(bundle.get("geometry_preview", []))


def _summary_payload(rows: list[dict[str, Any]], *, failure_frame: pd.DataFrame, rare_threshold: int) -> dict[str, Any]:
    reason_counts: Counter[str] = Counter()
    for row in rows:
        for reason in str(row.get("failure_reasons") or "").split(","):
            if reason:
                reason_counts[reason] += 1
    return {
        "n_records": len(rows),
        "rare_prototype_threshold": rare_threshold,
        "reason_counts": dict(sorted(reason_counts.items())),
        "image_source_counts": dict(Counter(str(row.get("image_source")) for row in rows)),
        "evaluation_failure_analysis_rows": int(len(failure_frame)),
        "top_failure_score": rows[0]["failure_score"] if rows else None,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _read_optional_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _read_region_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        schema = pq.ParquetFile(path).schema_arrow
        wanted = [
            "region_id",
            "contour_id",
            "slide_id",
            "row_index",
            "structure_label",
            "qc_flag",
            "area",
            "perimeter",
            "eccentricity",
            "n_cells",
            "image_source",
        ]
        columns = [column for column in wanted if column in schema.names]
        frame = pd.read_parquet(path, columns=columns)
    except Exception:
        return pd.DataFrame()
    if "embedding_row_index" not in frame.columns:
        frame.insert(0, "embedding_row_index", range(len(frame)))
    return frame


def _prototype_counts(records: list[dict[str, Any]], frame: pd.DataFrame) -> Counter[int]:
    counts: Counter[int] = Counter()
    if not frame.empty and "prototype_id" in frame.columns:
        for value in pd.to_numeric(frame["prototype_id"], errors="coerce").dropna():
            counts[int(value)] += 1
        return counts
    for record in records:
        proto = _dict(_dict(record.get("model_derived_evidence")).get("prototype_ref"))
        prototype_id = _safe_int(proto.get("prototype_id"))
        if prototype_id is not None:
            counts[prototype_id] += 1
    return counts


def _frame_row(frame: pd.DataFrame, index: int) -> dict[str, Any]:
    if frame.empty or index < 0 or index >= len(frame):
        return {}
    return frame.iloc[index].to_dict()


def _geometry_complexity(area: float | None, perimeter: float | None) -> float | None:
    if area is None or perimeter is None or area <= 0.0 or perimeter <= 0.0:
        return None
    return float((perimeter * perimeter) / (4.0 * math.pi * area))


def _hallucination_index(
    *,
    confidence: float | None,
    entropy: float | None,
    gt_similarity: float | None,
    top1_similarity: float | None,
) -> tuple[float | None, str]:
    if confidence is None:
        return None, "unavailable"
    if gt_similarity is not None:
        bounded_gt = max(-1.0, min(1.0, gt_similarity))
        retrieved_bonus = max(0.0, top1_similarity or 0.0)
        return float(confidence * (1.0 - ((bounded_gt + 1.0) / 2.0)) * (1.0 + retrieved_bonus)), "direct:top1_conflict"
    if entropy is not None:
        return float(confidence * entropy), "proxy:confidence_entropy"
    return float(confidence), "proxy:confidence_only"


def _minmax(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float64)
    lo = float(finite.min())
    hi = float(finite.max())
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float64)
    filled = np.where(np.isfinite(values), values, lo)
    return (filled - lo) / (hi - lo)


def _top_gene_rows(molecular_ref: Any, *, base_dir: Path, top_n: int) -> list[dict[str, Any]]:
    if top_n <= 0 or not isinstance(molecular_ref, dict):
        return []
    artifact = _string_or_none(molecular_ref.get("artifact"))
    row_index = _safe_int(molecular_ref.get("row_index"))
    if artifact is None or row_index is None:
        return []
    path = _resolve_artifact(artifact, base_dir)
    row = _read_parquet_row(path, row_index)
    values: list[tuple[str, float]] = []
    for key, value in row.items():
        if str(key).startswith("gene_"):
            number = _safe_float(value)
            if number is not None:
                values.append((str(key)[5:], number))
    values.sort(key=lambda item: (-item[1], item[0]))
    return [{"gene": gene, "value": value} for gene, value in values[:top_n]]


def _read_parquet_row(path: Path, row_index: int) -> dict[str, Any]:
    if not path.exists() or row_index < 0:
        return {}
    try:
        parquet_file = pq.ParquetFile(path)
        offset = 0
        for group_idx in range(parquet_file.num_row_groups):
            group_rows = parquet_file.metadata.row_group(group_idx).num_rows
            if row_index < offset + group_rows:
                table = parquet_file.read_row_group(group_idx)
                data = table.slice(row_index - offset, 1).to_pydict()
                return {key: values[0] if values else None for key, values in data.items()}
            offset += group_rows
    except Exception:
        return {}
    return {}


def _load_image_bundle(record: dict[str, Any], *, base_dir: Path, zarr_cache: dict[Path, zarr.Group]) -> dict[str, Any]:
    measured = _dict(record.get("measured_evidence"))
    image_ref = _dict(measured.get("image_ref"))
    qc = _dict(record.get("qc_verdict"))
    source = str(qc.get("image_source") or "unknown")
    if source == "contour_store":
        artifact = _string_or_none(image_ref.get("artifact"))
        row_index = _safe_int(image_ref.get("row_index"))
        arrays = _dict(image_ref.get("arrays"))
        if artifact is not None and row_index is not None:
            try:
                store_path = _resolve_artifact(artifact, base_dir)
                root = zarr_cache.setdefault(store_path, zarr.open_group(str(store_path), mode="r"))
                object_rgb = _as_rgb(root[str(arrays.get("object_rgb") or "object_rgb")][row_index])
                context_rgb = _as_rgb(root[str(arrays.get("context_rgb") or "context_rgb")][row_index])
                mask = _as_mask(root[str(arrays.get("mask") or "soft_mask")][row_index])
                geometry = _read_zarr_geometry(root, row_index)
                return {"object_rgb": object_rgb, "context_rgb": context_rgb, "mask": mask, "geometry_preview": geometry}
            except Exception:
                pass
    return _placeholder_bundle()


def _read_zarr_geometry(root: zarr.Group, row_index: int) -> list[float]:
    if "geometry" not in root:
        return []
    try:
        values = np.asarray(root["geometry"][row_index], dtype=np.float32).reshape(-1)
    except Exception:
        return []
    return [float(value) for value in values[:8] if np.isfinite(value)]


def _placeholder_bundle(size: int = 64) -> dict[str, Any]:
    return {
        "object_rgb": np.full((size, size, 3), 230, dtype=np.uint8),
        "context_rgb": np.full((size, size, 3), 210, dtype=np.uint8),
        "mask": np.zeros((size, size, 1), dtype=np.uint8),
        "geometry_preview": [],
    }


def _save_rgb(array: np.ndarray, path: Path) -> None:
    Image.fromarray(_as_rgb(array)).save(path)


def _save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(_as_mask(mask)[:, :, 0]).save(path)


def _overlay_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image = _as_rgb(rgb).astype(np.float32)
    alpha = (_as_mask(mask).astype(np.float32) / 255.0) * 0.35
    red = np.zeros_like(image)
    red[:, :, 0] = 255.0
    return np.clip(image * (1.0 - alpha) + red * alpha, 0, 255).astype(np.uint8)


def _as_rgb(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3:
        return np.full((64, 64, 3), 230, dtype=np.uint8)
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    if array.shape[2] > 3:
        array = array[:, :, :3]
    return np.clip(array, 0, 255).astype(np.uint8) if array.dtype != np.uint8 else array


def _as_mask(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3:
        return np.zeros((64, 64, 1), dtype=np.uint8)
    if array.shape[2] != 1:
        array = array[:, :, :1]
    return np.clip(array, 0, 255).astype(np.uint8) if array.dtype != np.uint8 else array


def _gallery_html(run_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    cards = "\n".join(_gallery_card(row) for row in rows) if rows else "<p>No failure candidates were selected.</p>"
    reason_text = ", ".join(f"{key}: {value}" for key, value in summary.get("reason_counts", {}).items()) or "none"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Contour-Native Failure Gallery</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #172026; background: #f8fafb; }}
    h1 {{ margin-bottom: 4px; }}
    .subtitle {{ color: #50606c; margin-top: 0; max-width: 980px; }}
    .summary {{ background: #eef3f7; border: 1px solid #d7dee4; padding: 12px; border-radius: 8px; margin: 16px 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 18px; }}
    .card {{ background: white; border: 1px solid #d7dee4; border-radius: 8px; padding: 14px; }}
    .images {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 10px 0 12px; }}
    .images img {{ width: 100%; aspect-ratio: 1 / 1; object-fit: cover; border: 1px solid #d7dee4; }}
    figcaption {{ font-size: 11px; color: #50606c; margin-top: 4px; }}
    figure {{ margin: 0; }}
    .badge {{ display: inline-block; background: #fff2cc; padding: 2px 6px; border-radius: 4px; margin-right: 4px; font-size: 12px; }}
    code {{ font-size: 12px; }}
  </style>
</head>
<body>
  <h1>Contour-Native Failure Gallery</h1>
  <p class="subtitle">Run: {escape(str(run_dir))}. Artifact-first triage for finding contours that deserve human review or targeted ablation.</p>
  <div class="summary">Records: {summary.get("n_records", 0)}. Reason counts: {escape(reason_text)}.</div>
  <div class="grid">{cards}</div>
</body>
</html>
"""


def _gallery_card(row: dict[str, Any]) -> str:
    genes = _genes_html(row.get("top_genes"))
    reasons = "".join(f"<span class='badge'>{escape(reason)}</span>" for reason in str(row.get("failure_reasons") or "").split(",") if reason)
    return f"""<section class="card">
  <h2>#{escape(str(row.get("failure_rank")))} {escape(str(row.get("contour_id") or row.get("evidence_id") or "contour"))}</h2>
  <p>{reasons}</p>
  <div class="images">
    <figure><img src="{escape(str(row.get("object_image", "")))}" alt="Object RGB"><figcaption>Object</figcaption></figure>
    <figure><img src="{escape(str(row.get("context_image", "")))}" alt="Context RGB"><figcaption>Context</figcaption></figure>
    <figure><img src="{escape(str(row.get("mask_image", "")))}" alt="Mask"><figcaption>Mask</figcaption></figure>
    <figure><img src="{escape(str(row.get("overlay_image", "")))}" alt="Overlay"><figcaption>Overlay</figcaption></figure>
  </div>
  <p>Score={_fmt(row.get("failure_score"))} | prototype={escape(str(row.get("prototype_id")))} count={escape(str(row.get("prototype_count")))} | confidence={_fmt(row.get("prototype_confidence"))} | entropy={_fmt(row.get("assignment_entropy"))}</p>
  <p>Geometry complexity={_fmt(row.get("geometry_complexity"))} | eccentricity={_fmt(row.get("eccentricity"))} | hallucination={_fmt(row.get("hallucination_index"))} ({escape(str(row.get("hallucination_basis")))})</p>
  <p>Structure: {escape(str(row.get("structure_label")))} | image_source={escape(str(row.get("image_source")))} | export_for_ablation={escape(str(row.get("export_for_ablation")))}</p>
  <p>Top genes: {genes}</p>
</section>"""


def _genes_html(value: Any) -> str:
    try:
        rows = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        rows = []
    if not isinstance(rows, list) or not rows:
        return "N/A"
    parts = []
    for row in rows[:8]:
        if not isinstance(row, dict):
            continue
        number = _safe_float(row.get("value"))
        parts.append(f"{escape(str(row.get('gene')))}={number:.3g}" if number is not None else escape(str(row.get("gene"))))
    return ", ".join(parts) if parts else "N/A"


def _fmt(value: Any) -> str:
    number = _safe_float(value)
    return f"{number:.4g}" if number is not None else "N/A"


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_float(row: dict[str, Any] | pd.Series, *keys: str) -> float | None:
    for key in keys:
        if key in row:
            value = _safe_float(row.get(key))
            if value is not None:
                return value
    return None


def _resolve_artifact(artifact: str, base_dir: Path) -> Path:
    path = Path(os.path.expandvars(artifact)).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def _relative_tile(path: Path, output_dir: Path) -> str:
    try:
        return path.relative_to(output_dir).as_posix()
    except ValueError:
        return str(path)


def _slug(text: str) -> str:
    cleaned = [char if char.isalnum() or char in {"-", "_"} else "_" for char in text]
    return "".join(cleaned)[:80] or "contour"


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text and text.lower() not in {"nan", "none", "null"} else None


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


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value
