from __future__ import annotations

import json
import math
import os
from html import escape
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import numpy as np
import pandas as pd

from .summary import EvidenceSuiteSpec, load_evidence_suite

ReducerName = Literal["auto", "umap", "pca"]


def build_latent_manifold(
    suite: EvidenceSuiteSpec | str | Path,
    output_dir: str | Path,
    *,
    reducer: ReducerName = "auto",
    max_points_per_run: int = 0,
    max_html_points: int = 5000,
    seed: int = 0,
) -> dict[str, Any]:
    """Project existing region embeddings into one artifact-first manifold.

    The function reads Spatho export artifacts only. It never trains, exports, or
    mutates checkpoints. When UMAP is unavailable it falls back to PCA and records
    that choice in the summary so reports remain reproducible on lean servers.
    """
    if max_points_per_run < 0:
        raise ValueError("max_points_per_run must be non-negative")
    if max_html_points < 0:
        raise ValueError("max_html_points must be non-negative")
    suite_path = Path(suite).expanduser() if isinstance(suite, (str, Path)) else None
    spec = load_evidence_suite(suite_path) if suite_path is not None else suite
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    missing_runs: list[str] = []
    for run in spec.runs:
        run_dir = _resolve_suite_path(run.run_dir, suite_path)
        frame = _load_run_embeddings(run, run_dir=run_dir)
        if frame.empty:
            missing_runs.append(run.run_id)
            continue
        if max_points_per_run and len(frame) > max_points_per_run:
            frame = frame.sample(n=max_points_per_run, random_state=seed).sort_index().reset_index(drop=True)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    emb_cols = [col for col in combined.columns if str(col).startswith("emb_")]
    if combined.empty or not emb_cols:
        projected = combined.copy()
        summary = _summary_payload(
            spec.suite_name,
            reducer_requested=reducer,
            reducer_used=None,
            n_points=0,
            n_embedding_dims=0,
            missing_runs=missing_runs,
            frame=projected,
            diagnostics={},
        )
    else:
        matrix = combined[emb_cols].to_numpy(dtype=np.float32)
        coords, reducer_used, diagnostics = _project(matrix, reducer=reducer, seed=seed)
        projected = combined.drop(columns=emb_cols).copy()
        projected["manifold_x"] = coords[:, 0]
        projected["manifold_y"] = coords[:, 1]
        summary = _summary_payload(
            spec.suite_name,
            reducer_requested=reducer,
            reducer_used=reducer_used,
            n_points=len(projected),
            n_embedding_dims=len(emb_cols),
            missing_runs=missing_runs,
            frame=projected,
            diagnostics=diagnostics | _neighbor_diagnostics(matrix, projected, seed=seed),
        )

    html_frame, html_sampling = _sample_html_points(projected, max_html_points=max_html_points, seed=seed)
    summary.update(html_sampling)

    manifold_csv = out / "latent_manifold.csv"
    manifold_json = out / "latent_manifold.json"
    summary_json = out / "latent_manifold_summary.json"
    report_md = out / "latent_manifold.md"
    report_html = out / "latent_manifold.html"
    centroids_csv = out / "structure_centroids.csv"

    projected.to_csv(manifold_csv, index=False)
    manifold_json.write_text(json.dumps(_json_safe(projected.to_dict(orient="records")), indent=2), encoding="utf-8")
    summary_json.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    centroids = _centroid_frame(projected)
    centroids.to_csv(centroids_csv, index=False)
    report_md.write_text(_manifold_markdown(summary, centroids), encoding="utf-8")
    report_html.write_text(_manifold_html(summary, html_frame), encoding="utf-8")

    return {
        "suite_name": spec.suite_name,
        "status": summary["status"],
        "n_points": int(summary["n_points"]),
        "reducer_used": summary["reducer_used"],
        "artifacts": {
            "latent_manifold_csv": str(manifold_csv),
            "latent_manifold_json": str(manifold_json),
            "latent_manifold_summary": str(summary_json),
            "latent_manifold_md": str(report_md),
            "latent_manifold_html": str(report_html),
            "structure_centroids_csv": str(centroids_csv),
        },
    }


def _load_run_embeddings(run: Any, *, run_dir: Path) -> pd.DataFrame:
    export_dir = run_dir / "spatho_export"
    embedding_path = export_dir / "region_embeddings.parquet"
    if not embedding_path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_parquet(embedding_path)
    except Exception:
        return pd.DataFrame()
    emb_cols = [col for col in frame.columns if str(col).startswith("emb_")]
    if not emb_cols:
        return pd.DataFrame()
    keep = [
        col
        for col in (
            "region_id",
            "contour_id",
            "slide_id",
            "patient_id",
            "batch_id",
            "organ",
            "stain",
            "scanner",
            "structure_label",
            "n_cells",
            "qc_flag",
            "x",
            "y",
            "row_index",
        )
        if col in frame.columns
    ]
    result = frame[keep + emb_cols].copy()
    result.insert(0, "embedding_row_index", np.arange(len(result), dtype=np.int64))
    result.insert(0, "run_dir", str(run_dir))
    result.insert(0, "condition", run.condition)
    result.insert(0, "tissue", run.tissue)
    result.insert(0, "run_id", run.run_id)
    result["checkpoint_hash"] = _checkpoint_hash(export_dir)
    result = _merge_prototypes(result, export_dir / "prototype_assignments.parquet")
    return result


def _merge_prototypes(frame: pd.DataFrame, prototype_path: Path) -> pd.DataFrame:
    if not prototype_path.exists():
        return frame
    try:
        proto = pd.read_parquet(prototype_path)
    except Exception:
        return frame
    columns = [
        col
        for col in ("embedding_row_index", "prototype_id", "prototype_confidence", "assignment_entropy")
        if col in proto.columns
    ]
    if "embedding_row_index" not in columns:
        proto = proto.copy()
        proto.insert(0, "embedding_row_index", np.arange(len(proto), dtype=np.int64))
        columns = [
            col
            for col in ("embedding_row_index", "prototype_id", "prototype_confidence", "assignment_entropy")
            if col in proto.columns
        ]
    if len(columns) <= 1:
        return frame
    return frame.merge(proto[columns], on="embedding_row_index", how="left", sort=False)


def _checkpoint_hash(export_dir: Path) -> str | None:
    manifest = export_dir / "evidence_manifest.json"
    if not manifest.exists():
        chain = export_dir / "contour_evidence_chains.jsonl"
        if not chain.exists():
            return None
        try:
            with chain.open("r", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    provenance = record.get("provenance", {}) if isinstance(record, dict) else {}
                    if isinstance(provenance, dict):
                        return _string_or_none(provenance.get("checkpoint_hash"))
        except Exception:
            return None
        return None
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return None
    provenance = payload.get("provenance", {}) if isinstance(payload, dict) else {}
    return _string_or_none(provenance.get("checkpoint_hash")) if isinstance(provenance, dict) else None


def _project(matrix: np.ndarray, *, reducer: ReducerName, seed: int) -> tuple[np.ndarray, str, dict[str, Any]]:
    if matrix.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float32), "single_point", {}
    if reducer in {"auto", "umap"}:
        try:
            import umap  # type: ignore[import-untyped]

            n_neighbors = min(15, max(2, matrix.shape[0] - 1))
            reducer_model = umap.UMAP(
                n_components=2,
                metric="cosine",
                n_neighbors=n_neighbors,
                min_dist=0.1,
                random_state=seed,
            )
            coords = reducer_model.fit_transform(matrix).astype(np.float32)
            return coords, "umap", {"umap_n_neighbors": n_neighbors, "umap_metric": "cosine"}
        except Exception as exc:
            if reducer == "umap":
                return _pca_project(matrix), "pca_fallback", {"umap_error": str(exc)}
    return _pca_project(matrix), "pca", {}


def _pca_project(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    centered = matrix.astype(np.float32) - matrix.astype(np.float32).mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        coords = centered @ vt[:2].T
    except np.linalg.LinAlgError:
        coords = np.zeros((matrix.shape[0], 2), dtype=np.float32)
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(matrix.shape[0], dtype=np.float32)])
    return coords[:, :2].astype(np.float32)


def _neighbor_diagnostics(
    matrix: np.ndarray,
    frame: pd.DataFrame,
    *,
    max_points: int = 5000,
    seed: int = 0,
) -> dict[str, Any]:
    if matrix.shape[0] <= 1 or frame.empty:
        return {}
    diagnostics: dict[str, Any] = {
        "neighbor_diagnostic_points": int(matrix.shape[0]),
        "neighbor_diagnostic_total_points": int(matrix.shape[0]),
        "neighbor_diagnostic_sampling": "all",
    }
    if max_points > 0 and matrix.shape[0] > max_points:
        total_points = int(matrix.shape[0])
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(matrix.shape[0], size=max_points, replace=False))
        matrix = matrix[selected]
        frame = frame.iloc[selected].reset_index(drop=True)
        diagnostics.update(
            {
                "neighbor_diagnostic_points": int(max_points),
                "neighbor_diagnostic_total_points": total_points,
                "neighbor_diagnostic_sampling": "random_without_replacement",
            }
        )
    normalized = matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8, None)
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -np.inf)
    order = np.argsort(-similarity, axis=1, kind="mergesort")
    tissues = frame.get("tissue", pd.Series([""] * len(frame))).astype(str).to_numpy()
    top1 = order[:, 0]
    top5 = order[:, : min(5, order.shape[1])]
    cross_top1 = tissues[top1] != tissues
    cross_top5 = np.asarray([(tissues[neighbors] != tissues[idx]).any() for idx, neighbors in enumerate(top5)], dtype=bool)
    diagnostics.update(
        {
        "cross_tissue_top1_rate": float(cross_top1.mean()),
        "cross_tissue_top5_rate": float(cross_top5.mean()),
        }
    )
    return diagnostics


def _summary_payload(
    suite_name: str,
    *,
    reducer_requested: str,
    reducer_used: str | None,
    n_points: int,
    n_embedding_dims: int,
    missing_runs: list[str],
    frame: pd.DataFrame,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_counts = frame["checkpoint_hash"].fillna("missing").astype(str).value_counts().to_dict() if "checkpoint_hash" in frame else {}
    status = "pass" if n_points else "missing"
    warnings: list[str] = []
    if missing_runs:
        warnings.append(f"missing_embedding_runs={len(missing_runs)}")
    non_missing_hashes = {key for key in checkpoint_counts if key != "missing"}
    if len(non_missing_hashes) > 1:
        warnings.append("multiple_checkpoint_hashes: cross-run geometry is exploratory unless runs share a checkpoint")
        status = "warning" if status == "pass" else status
    elif "missing" in checkpoint_counts:
        warnings.append("missing_checkpoint_hashes")
        status = "warning" if status == "pass" else status
    return {
        "suite_name": suite_name,
        "status": status,
        "warnings": warnings,
        "reducer_requested": reducer_requested,
        "reducer_used": reducer_used,
        "n_points": int(n_points),
        "n_embedding_dims": int(n_embedding_dims),
        "n_runs": int(frame["run_id"].nunique()) if "run_id" in frame else 0,
        "tissue_counts": frame["tissue"].fillna("unknown").astype(str).value_counts().sort_index().to_dict()
        if "tissue" in frame
        else {},
        "run_counts": frame["run_id"].fillna("unknown").astype(str).value_counts().sort_index().to_dict()
        if "run_id" in frame
        else {},
        "checkpoint_hash_counts": checkpoint_counts,
        "missing_runs": missing_runs,
        **diagnostics,
    }


def _centroid_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "manifold_x" not in frame:
        return pd.DataFrame(columns=["run_id", "tissue", "structure_label", "n_points", "manifold_x", "manifold_y"])
    group_cols = [col for col in ("run_id", "tissue", "structure_label") if col in frame.columns]
    if not group_cols:
        return pd.DataFrame()
    return (
        frame.groupby(group_cols, dropna=False, sort=True)
        .agg(n_points=("manifold_x", "size"), manifold_x=("manifold_x", "mean"), manifold_y=("manifold_y", "mean"))
        .reset_index()
    )


def _manifold_markdown(summary: dict[str, Any], centroids: pd.DataFrame) -> str:
    centroid_table = _frame_to_markdown(centroids.head(25)) if not centroids.empty else "No centroids were produced."
    return f"""# Latent Manifold Projection

Suite: `{summary.get("suite_name")}`

Reducer: requested `{summary.get("reducer_requested")}`, used `{summary.get("reducer_used")}`

Status: `{summary.get("status")}`

Warnings: {", ".join(summary.get("warnings") or []) or "none"}

Points: {summary.get("n_points")}  
Embedding dims: {summary.get("n_embedding_dims")}  
Cross-tissue top-1 nearest-neighbor rate: {summary.get("cross_tissue_top1_rate", "N/A")}  
Cross-tissue top-5 nearest-neighbor rate: {summary.get("cross_tissue_top5_rate", "N/A")}

## Interpretation Guardrail

This artifact tests the universal morphology hypothesis only when compared runs
share a checkpoint hash. If multiple checkpoint hashes are present, the manifold
is useful for exploratory QC and figure layout, but not by itself proof of a
shared cross-organ latent space.

## Structure Centroids

{centroid_table}
"""


def _manifold_html(summary: dict[str, Any], frame: pd.DataFrame) -> str:
    points = _svg_points(frame) if not frame.empty and "manifold_x" in frame else "<p>No manifold points.</p>"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Latent Manifold Projection</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #172026; background: #f8fafb; }}
    .panel {{ background: white; border: 1px solid #d7dee4; border-radius: 8px; padding: 16px; max-width: 1120px; }}
    svg {{ width: 100%; height: auto; border: 1px solid #d7dee4; background: #fff; }}
    .legend span {{ display: inline-block; margin-right: 14px; font-size: 13px; }}
    code {{ font-size: 12px; }}
  </style>
</head>
<body>
  <div class="panel">
    <h1>Latent Manifold Projection</h1>
    <p>Reducer: <code>{escape(str(summary.get("reducer_used")))}</code>; points: {escape(str(summary.get("n_points")))}; HTML rendered: {escape(str(summary.get("html_points_rendered", "N/A")))}; status: <code>{escape(str(summary.get("status")))}</code></p>
    <p>Warnings: {escape(", ".join(summary.get("warnings") or []) or "none")}</p>
    <p>Point sampling: <code>{escape(str(summary.get("html_point_sampling", "N/A")))}</code>. Click visible points to open their evidence-chain pointer when available.</p>
    {points}
  </div>
</body>
</html>
"""


def _sample_html_points(
    frame: pd.DataFrame,
    *,
    max_html_points: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    total = len(frame)
    if frame.empty or "manifold_x" not in frame or "manifold_y" not in frame:
        return frame.copy(), {
            "html_points_rendered": 0,
            "html_point_sampling": "empty",
            "html_point_sampling_fraction": 0.0,
        }
    if max_html_points == 0 or total <= max_html_points:
        return frame.copy(), {
            "html_points_rendered": int(total),
            "html_point_sampling": "all",
            "html_point_sampling_fraction": 1.0 if total else 0.0,
        }
    finite = np.isfinite(pd.to_numeric(frame["manifold_x"], errors="coerce").to_numpy(dtype=float)) & np.isfinite(
        pd.to_numeric(frame["manifold_y"], errors="coerce").to_numpy(dtype=float)
    )
    if not finite.any():
        return frame.head(max_html_points).copy(), {
            "html_points_rendered": int(min(total, max_html_points)),
            "html_point_sampling": "head_nonfinite",
            "html_point_sampling_fraction": float(min(total, max_html_points) / total) if total else 0.0,
        }
    finite_frame = frame.loc[finite].copy()
    finite_frame["_source_index"] = finite_frame.index.to_numpy()
    finite_frame = _assign_density_cells(finite_frame)
    counts = finite_frame["_density_cell"].value_counts(sort=False)
    if len(counts) > max_html_points:
        selected = set(counts.sort_values(ascending=False, kind="mergesort").head(max_html_points).index)
        quotas = np.asarray([1 if cell in selected else 0 for cell in counts.index], dtype=int)
    else:
        weights = np.sqrt(counts.astype(float).to_numpy())
        raw = weights / weights.sum() * max_html_points
        quotas = np.floor(raw).astype(int)
        quotas = np.maximum(1, np.minimum(quotas, counts.to_numpy()))
        quotas = _rebalance_quotas(quotas, counts.to_numpy(), max_html_points)
    rng = np.random.default_rng(seed)
    sampled_indices: list[int] = []
    for cell, quota in zip(counts.index.to_list(), quotas, strict=False):
        if quota <= 0:
            continue
        cell_indices = finite_frame.loc[finite_frame["_density_cell"] == cell, "_source_index"].to_numpy(dtype=int)
        if len(cell_indices) <= quota:
            sampled_indices.extend(int(item) for item in cell_indices)
        else:
            sampled_indices.extend(int(item) for item in rng.choice(cell_indices, size=int(quota), replace=False))
    sampled = frame.loc[sorted(set(sampled_indices))].copy()
    return sampled, {
        "html_points_rendered": int(len(sampled)),
        "html_point_sampling": "density_grid_sqrt",
        "html_point_sampling_fraction": float(len(sampled) / total) if total else 0.0,
        "html_point_sampling_cap": int(max_html_points),
    }


def _assign_density_cells(frame: pd.DataFrame) -> pd.DataFrame:
    x = pd.to_numeric(frame["manifold_x"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(frame["manifold_y"], errors="coerce").to_numpy(dtype=float)
    bins = max(8, min(64, int(math.sqrt(max(len(frame), 1)))))
    xspan = float(np.nanmax(x) - np.nanmin(x)) or 1.0
    yspan = float(np.nanmax(y) - np.nanmin(y)) or 1.0
    xb = np.clip(((x - float(np.nanmin(x))) / xspan * bins).astype(int), 0, bins - 1)
    yb = np.clip(((y - float(np.nanmin(y))) / yspan * bins).astype(int), 0, bins - 1)
    result = frame.copy()
    result["_density_cell"] = [f"{int(ix)}:{int(iy)}" for ix, iy in zip(xb, yb, strict=False)]
    return result


def _rebalance_quotas(quotas: np.ndarray, capacities: np.ndarray, target: int) -> np.ndarray:
    adjusted = quotas.astype(int).copy()
    while int(adjusted.sum()) > target:
        candidates = np.where(adjusted > 1)[0]
        if len(candidates) == 0:
            break
        idx = int(candidates[np.argmax(adjusted[candidates])])
        adjusted[idx] -= 1
    while int(adjusted.sum()) < target:
        remaining = capacities - adjusted
        candidates = np.where(remaining > 0)[0]
        if len(candidates) == 0:
            break
        idx = int(candidates[np.argmax(remaining[candidates])])
        adjusted[idx] += 1
    return adjusted


def _svg_points(frame: pd.DataFrame) -> str:
    width, height, pad = 900, 620, 36
    x = pd.to_numeric(frame["manifold_x"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(frame["manifold_y"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        return "<p>No finite manifold coordinates.</p>"
    xmin, xmax = float(np.nanmin(x[finite])), float(np.nanmax(x[finite]))
    ymin, ymax = float(np.nanmin(y[finite])), float(np.nanmax(y[finite]))
    xrange = xmax - xmin if xmax > xmin else 1.0
    yrange = ymax - ymin if ymax > ymin else 1.0
    tissues = sorted(frame.get("tissue", pd.Series(["unknown"] * len(frame))).fillna("unknown").astype(str).unique())
    palette = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c", "#0891b2", "#4b5563"]
    color_map = {tissue: palette[idx % len(palette)] for idx, tissue in enumerate(tissues)}
    circles = []
    for idx, row in frame.iterrows():
        if not finite[idx]:
            continue
        sx = pad + ((float(x[idx]) - xmin) / xrange) * (width - 2 * pad)
        sy = height - pad - ((float(y[idx]) - ymin) / yrange) * (height - 2 * pad)
        tissue = str(row.get("tissue") or "unknown")
        label = _point_hover_label(row)
        href = _evidence_href(row)
        attrs = _point_data_attrs(row)
        circle = (
            f'<circle cx="{sx:.2f}" cy="{sy:.2f}" r="3.2" fill="{color_map.get(tissue, "#4b5563")}" '
            f'opacity="0.68"{attrs}><title>{escape(label)}</title></circle>'
        )
        if href:
            circle = f'<a href="{escape(href, quote=True)}" target="_blank" rel="noreferrer">{circle}</a>'
        circles.append(
            circle
        )
    legend = " ".join(
        f'<span><svg width="10" height="10"><circle cx="5" cy="5" r="5" fill="{color}"/></svg> {escape(tissue)}</span>'
        for tissue, color in color_map.items()
    )
    return f'<div class="legend">{legend}</div><svg viewBox="0 0 {width} {height}" role="img">{"".join(circles)}</svg>'


def _point_hover_label(row: pd.Series) -> str:
    fields = [
        ("run", row.get("run_id")),
        ("tissue", row.get("tissue")),
        ("structure", row.get("structure_label")),
        ("contour", row.get("contour_id")),
        ("region", row.get("region_id")),
        ("prototype", row.get("prototype_id")),
        ("confidence", _format_float(row.get("prototype_confidence"))),
        ("row_index", row.get("row_index")),
        ("evidence_link", _evidence_href(row)),
    ]
    return "\n".join(f"{name}: {value}" for name, value in fields if _string_or_none(value) is not None)


def _point_data_attrs(row: pd.Series) -> str:
    attrs = {
        "data-run-id": row.get("run_id"),
        "data-contour-id": row.get("contour_id"),
        "data-region-id": row.get("region_id"),
        "data-row-index": row.get("row_index"),
        "data-embedding-row-index": row.get("embedding_row_index"),
    }
    return "".join(
        f' {name}="{escape(str(value), quote=True)}"' for name, value in attrs.items() if _string_or_none(value) is not None
    )


def _evidence_href(row: pd.Series) -> str | None:
    run_dir = _string_or_none(row.get("run_dir"))
    if run_dir is None:
        return None
    evidence_chain = (Path(run_dir).expanduser() / "spatho_export" / "contour_evidence_chains.jsonl").resolve()
    if not evidence_chain.exists():
        return None
    fragment = _evidence_fragment(row)
    try:
        return f"{evidence_chain.as_uri()}{fragment}"
    except ValueError:
        path = quote(str(evidence_chain).replace("\\", "/"))
        return f"file:///{path}{fragment}"


def _evidence_fragment(row: pd.Series) -> str:
    region = _string_or_none(row.get("region_id"))
    contour = _string_or_none(row.get("contour_id"))
    embedding_row = _string_or_none(row.get("embedding_row_index"))
    if region:
        return f"#region-{quote(region)}"
    if contour:
        return f"#contour-{quote(contour)}"
    if embedding_row:
        return f"#row-{quote(embedding_row)}"
    return ""


def _format_float(value: Any) -> str | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return f"{number:.4g}"


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


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    columns = [str(column) for column in frame.columns]
    header = "| " + " | ".join(_escape_markdown_cell(column) for column in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(_escape_markdown_cell(row.get(column)) for column in frame.columns) + " |")
    return "\n".join([header, separator, *rows])


def _escape_markdown_cell(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).replace("|", "\\|").replace("\n", " ")


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
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value
