"""spatho adapter for stGPT: typed I/O contracts and end-to-end export pipeline.

This module defines the stable interface between stGPT's morpho-molecular embedding
model and downstream spatial-pathology tools such as spatho.  The key entry point is
:func:`run_spatho_export`, which accepts a config, a trained checkpoint, and an output
directory, and writes region-first evidence artifacts:

* ``region_embeddings.parquet`` – one row per contour/region with spatial centroid,
  structure label, cell count, QC flag, and the embedding vector.
* ``region_cell_membership.parquet`` – contour/region-to-cell membership table.
* ``region_molecular_summary.parquet`` – raw mean expression per contour/region.
* ``region_image_manifest.json`` – image patch and registration provenance.
* ``prototype_assignments.parquet`` – model-derived prototype IDs and confidence.
* ``contour_evidence_chains.jsonl`` – one pointer-only evidence chain per region.
* ``region_qc_report.json`` – operational QC summary for the export run.
* ``structure_summary.parquet`` – one row per structure with cell count and mean
  embedding vector.

Downstream tools should treat :data:`REGION_EMBEDDING_REQUIRED_COLUMNS` and
:data:`STRUCTURE_SUMMARY_REQUIRED_COLUMNS` as the stable region-first contract;
additional ``emb_*`` columns carry the actual embedding dimensions.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import StGPTConfig
from .foundation.packaging import resolve_model_checkpoint
from .inference import embed_region_outputs

#: Deprecated compatibility schema for the old cell-first export contract.
CELL_EMBEDDING_REQUIRED_COLUMNS: tuple[str, ...] = ("cell_id", "x", "y", "structure_label", "qc_flag")
REGION_EMBEDDING_REQUIRED_COLUMNS: tuple[str, ...] = ("region_id", "x", "y", "structure_label", "n_cells", "qc_flag")

#: Required non-embedding columns guaranteed to be present in ``structure_summary.parquet``.
STRUCTURE_SUMMARY_REQUIRED_COLUMNS: tuple[str, ...] = ("structure_label", "n_cells")


@dataclass(frozen=True)
class PatchManifestRow:
    """Schema for a single row in a spatho H&E patch manifest.

    This dataclass documents the expected column names for a spatho patch manifest
    (CSV or JSON list).  All fields are optional so that partial manifests produced
    at different spatho pipeline stages are accepted.

    Attributes:
        cell_id: Cell identifier matching ``AnnData.obs_names`` or
            ``AnnData.obs["cell_id"]``.
        contour_id: Spatho contour identifier for the patch bounding box.
        structure_id: Integer or string structure/region identifier that links the
            patch to a spatho structure annotation.
        structure_name: Human-readable structure label (e.g. ``"tumor"``,
            ``"stroma"``).
        image_path: Absolute or relative path to the extracted H&E patch image.
        x_px: Patch centre x-coordinate in slide pixel space.
        y_px: Patch centre y-coordinate in slide pixel space.
        patch_size_px: Side length of the square patch in pixels.
    """

    cell_id: str | None
    contour_id: str | None
    structure_id: str | int | None
    structure_name: str | None
    image_path: str | None
    x_px: float | None
    y_px: float | None
    patch_size_px: int | None


@dataclass(frozen=True)
class SpathoExportResult:
    """Paths and summary statistics for a completed :func:`run_spatho_export` run.

    Attributes:
        cell_embeddings: Deprecated alias for ``region_embeddings.parquet``.
        structure_summary: Path to ``structure_summary.parquet``.
        qc_report: Path to ``qc_report.json``.
        n_cells: Deprecated alias for total number of regions embedded.
        n_cells_with_image: Deprecated alias for regions whose H&E patch was found and loaded.
        embedding_dim: Dimensionality of the embedding vectors (number of ``emb_*``
            columns in ``cell_embeddings.parquet``).
    """

    cell_embeddings: Path
    structure_summary: Path
    qc_report: Path
    n_cells: int
    n_cells_with_image: int
    embedding_dim: int
    structure_embedding_summary: Path | None = None
    region_embeddings: Path | None = None
    region_cell_membership: Path | None = None
    region_molecular_summary: Path | None = None
    region_image_manifest: Path | None = None
    region_qc_report: Path | None = None
    evidence_manifest: Path | None = None
    prototype_assignments: Path | None = None
    contour_evidence_chains: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        payload = asdict(self)
        payload["cell_embeddings"] = str(self.cell_embeddings)
        payload["structure_summary"] = str(self.structure_summary)
        payload["qc_report"] = str(self.qc_report)
        if self.structure_embedding_summary is not None:
            payload["structure_embedding_summary"] = str(self.structure_embedding_summary)
        for key in (
            "region_embeddings",
            "region_cell_membership",
            "region_molecular_summary",
            "region_image_manifest",
            "region_qc_report",
            "evidence_manifest",
            "prototype_assignments",
            "contour_evidence_chains",
        ):
            value = getattr(self, key)
            if value is not None:
                payload[key] = str(value)
        return payload


def run_spatho_export(
    config: StGPTConfig | str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    batch_size: int = 32,
    device: str = "auto",
) -> SpathoExportResult:
    """Run the full spatho embedding export pipeline.

    Loads the Xenium/AnnData case described by *config*, embeds all contour/region
    units using the pretrained *checkpoint*, and writes region-first artifacts to
    *output_dir*.

    The stable output schema is:

    ``region_embeddings.parquet``
        Columns: ``region_id`` (str), ``x`` (float), ``y`` (float),
        ``structure_label`` (str), ``n_cells`` (int),
        ``qc_flag`` (str: ``"ok"`` | ``"no_image"``),
        ``emb_0`` … ``emb_{d-1}`` (float32).

    ``structure_summary.parquet``
        Columns: ``structure_label`` (str), ``n_cells`` (int; summed member cells),
        ``emb_0`` … ``emb_{d-1}`` (float32, mean over cells in structure).

    ``region_qc_report.json``
        Operational QC: region counts, image coverage, per-structure counts.

    Args:
        config: Path to a YAML/JSON stGPT config file or an already-parsed
            :class:`~stgpt.config.StGPTConfig` instance.
        checkpoint: Path to a ``*.pt`` checkpoint produced by ``stgpt train``.
        output_dir: Directory where output artifacts are written (created if absent).
        batch_size: Inference batch size (default 32).
        device: PyTorch device string: ``"auto"``, ``"cpu"``, or ``"cuda"``.

    Returns:
        :class:`SpathoExportResult` with paths to all written artifacts and summary
        statistics.

    Example::

        from stgpt.spatho import run_spatho_export

        result = run_spatho_export(
            "configs/atera_wta_breast.yaml",
            checkpoint="outputs/atera_wta_breast/train/checkpoints/last.pt",
            output_dir="outputs/atera_wta_breast/spatho_export",
        )
        print(result.region_embeddings)   # .../region_embeddings.parquet
    """
    cfg = StGPTConfig.from_file(config) if isinstance(config, (str, Path)) else config
    checkpoint_path = resolve_model_checkpoint(checkpoint)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inference_result = embed_region_outputs(
        cfg,
        checkpoint=checkpoint_path,
        batch_size=batch_size,
        device=device,
    )
    region_table = inference_result.region_table
    embeddings = inference_result.embeddings
    dataset = inference_result.dataset
    frame = _build_region_embedding_frame(region_table, embeddings)
    prototype_frame = _build_prototype_assignment_frame(region_table, inference_result.prototype_assignments)

    region_emb_path = out_dir / "region_embeddings.parquet"
    membership_path = out_dir / "region_cell_membership.parquet"
    molecular_path = out_dir / "region_molecular_summary.parquet"
    image_manifest_path = out_dir / "region_image_manifest.json"
    prototype_path = out_dir / "prototype_assignments.parquet"
    evidence_chain_path = out_dir / "contour_evidence_chains.jsonl"
    region_qc_path = out_dir / "region_qc_report.json"
    evidence_manifest_path = out_dir / "evidence_manifest.json"
    struct_sum_path = out_dir / "structure_summary.parquet"
    struct_sum_csv_path = out_dir / "structure_embedding_summary.csv"

    frame.to_parquet(region_emb_path, index=False)
    dataset.cell_membership.to_parquet(membership_path, index=False)
    _build_region_molecular_summary(dataset).to_parquet(molecular_path, index=False)
    prototype_frame.to_parquet(prototype_path, index=False)
    image_manifest_path.write_text(json.dumps(_build_region_image_manifest(region_table), indent=2), encoding="utf-8")
    summary = _build_structure_summary(frame)
    summary.to_parquet(struct_sum_path, index=False)
    summary.to_csv(struct_sum_csv_path, index=False)

    emb_cols = [col for col in frame.columns if str(col).startswith("emb_")]
    n_regions = int(len(frame))
    n_with_image = int((frame["qc_flag"] == "ok").sum()) if "qc_flag" in frame else 0
    qc_payload = _build_region_qc_report(cfg, checkpoint_path, frame, dataset)
    region_qc_path.write_text(json.dumps(qc_payload, indent=2), encoding="utf-8")
    _write_contour_evidence_chains(
        evidence_chain_path,
        frame,
        prototype_frame,
        cfg,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        device=device,
    )
    artifacts = {
        "region_embeddings": str(region_emb_path),
        "region_cell_membership": str(membership_path),
        "region_molecular_summary": str(molecular_path),
        "region_image_manifest": str(image_manifest_path),
        "prototype_assignments": str(prototype_path),
        "contour_evidence_chains": str(evidence_chain_path),
        "region_qc_report": str(region_qc_path),
    }
    evidence_manifest_path.write_text(
        json.dumps(
            {
                "case_name": cfg.case_name,
                "checkpoint": str(checkpoint_path),
                "training_unit": "region",
                "contract": "stgpt.spatho_evidence_bundle.v0.1",
                "rule": "json_stores_pointers_parquet_stores_matrices",
                "artifacts": artifacts,
                "provenance": {
                    "config_hash": _config_hash(cfg),
                    "checkpoint_hash": _sha256_path(checkpoint_path),
                    "contour_manifest_hash": _sha256_optional_path(cfg.data.path_or_none(cfg.data.contour_manifest)),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return SpathoExportResult(
        cell_embeddings=region_emb_path,
        structure_summary=struct_sum_path,
        qc_report=region_qc_path,
        n_cells=n_regions,
        n_cells_with_image=n_with_image,
        embedding_dim=len(emb_cols),
        structure_embedding_summary=struct_sum_csv_path,
        region_embeddings=region_emb_path,
        region_cell_membership=membership_path,
        region_molecular_summary=molecular_path,
        region_image_manifest=image_manifest_path,
        region_qc_report=region_qc_path,
        evidence_manifest=evidence_manifest_path,
        prototype_assignments=prototype_path,
        contour_evidence_chains=evidence_chain_path,
    )


def _build_region_embedding_frame(region_table: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    frame = region_table.copy().reset_index(drop=True)
    for dim_idx in range(embeddings.shape[1] if embeddings.ndim == 2 else 0):
        frame[f"emb_{dim_idx}"] = embeddings[:, dim_idx]
    for column in REGION_EMBEDDING_REQUIRED_COLUMNS:
        if column not in frame.columns:
            frame[column] = "unknown" if column in {"region_id", "structure_label", "qc_flag"} else 0
    return frame


def _build_region_molecular_summary(dataset) -> pd.DataFrame:
    genes = dataset.vocab.genes
    matrix = dataset.region_matrix.toarray().astype(np.float32)
    frame = pd.DataFrame(matrix, columns=[f"gene_{gene}" for gene in genes])
    frame.insert(0, "n_cells", dataset.region_table["n_cells"].to_numpy(dtype=np.int64))
    frame.insert(0, "region_id", dataset.region_table["region_id"].astype(str).to_numpy())
    return frame


def _build_prototype_assignment_frame(region_table: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    meta_columns = [
        column
        for column in ("region_id", "contour_id", "slide_id", "patient_id", "batch_id", "row_index", "structure_label", "qc_flag")
        if column in region_table.columns
    ]
    meta = region_table[meta_columns].copy().reset_index(drop=True) if meta_columns else pd.DataFrame(index=region_table.index)
    meta.insert(0, "embedding_row_index", np.arange(len(region_table), dtype=np.int64))
    if "region_id" not in meta.columns and "region_id" in region_table.columns:
        meta["region_id"] = region_table["region_id"].astype(str).to_numpy()

    assign = assignments.copy().reset_index(drop=True)
    if len(assign) != len(meta):
        assign = pd.DataFrame({"region_id": meta["region_id"].astype(str).to_numpy() if "region_id" in meta else []})
    duplicate_cols = [column for column in assign.columns if column in meta.columns]
    assign = assign.drop(columns=duplicate_cols, errors="ignore")
    frame = pd.concat([meta, assign], axis=1)
    if "prototype_id" not in frame.columns:
        frame["prototype_id"] = -1
    if "prototype_confidence" not in frame.columns:
        frame["prototype_confidence"] = np.nan
    if "assignment_entropy" not in frame.columns:
        frame["assignment_entropy"] = np.nan
    return frame


def _build_region_image_manifest(region_table: pd.DataFrame) -> dict[str, Any]:
    cols = [col for col in ("region_id", "image_path", "patch_x", "patch_y", "patch_size", "source_image", "registration_transform") if col in region_table]
    return {"regions": region_table[cols].to_dict(orient="records") if cols else []}


def _write_contour_evidence_chains(
    path: Path,
    frame: pd.DataFrame,
    prototype_frame: pd.DataFrame,
    cfg: StGPTConfig,
    *,
    checkpoint_path: Path,
    batch_size: int,
    device: str,
) -> None:
    config_hash = _config_hash(cfg)
    checkpoint_hash = _sha256_path(checkpoint_path)
    contour_manifest_hash = _sha256_optional_path(cfg.data.path_or_none(cfg.data.contour_manifest))
    emb_cols = [column for column in frame.columns if str(column).startswith("emb_")]
    with path.open("w", encoding="utf-8") as handle:
        for row_idx, row in frame.reset_index(drop=True).iterrows():
            proto_row = prototype_frame.iloc[row_idx] if row_idx < len(prototype_frame) else pd.Series(dtype=object)
            source_row_index = _nullable_int(row.get("row_index"))
            prototype_id = _nullable_int(proto_row.get("prototype_id"))
            record = {
                "schema_version": "stgpt.evidence_pointer.v0.1",
                "evidence_id": _evidence_id(cfg.case_name, row, row_idx),
                "unit": {
                    "type": "contour_region",
                    "region_id": _string_or_none(row.get("region_id")),
                    "contour_id": _string_or_none(row.get("contour_id")) or _string_or_none(row.get("region_id")),
                    "slide_id": _string_or_none(row.get("slide_id")),
                    "row_index": source_row_index,
                    "embedding_row_index": row_idx,
                },
                "measured_evidence": {
                    "molecular_ref": {
                        "artifact": "region_molecular_summary.parquet",
                        "row_index": row_idx,
                        "matrix": "region_mean_expression",
                    },
                    "image_ref": _image_pointer(row, cfg, row_idx=row_idx, source_row_index=source_row_index),
                    "geometry_ref": _geometry_pointer(cfg, row_idx=row_idx, source_row_index=source_row_index),
                    "spatial": {
                        "x": _nullable_float(row.get("x")),
                        "y": _nullable_float(row.get("y")),
                        "coordinate_space": "physical_or_registered_input",
                    },
                },
                "model_derived_evidence": {
                    "embedding_ref": {
                        "artifact": "region_embeddings.parquet",
                        "row_index": row_idx,
                        "vector_column_prefix": "emb_",
                        "embedding_dim": len(emb_cols),
                    },
                    "prototype_ref": {
                        "artifact": "prototype_assignments.parquet",
                        "row_index": row_idx,
                        "prototype_id": prototype_id,
                        "confidence": _nullable_float(proto_row.get("prototype_confidence")),
                        "assignment_entropy": _nullable_float(proto_row.get("assignment_entropy")),
                    },
                },
                "qc_verdict": {
                    "qc_flag": _string_or_none(row.get("qc_flag")) or "unknown",
                    "image_source": _image_source(row, cfg, source_row_index),
                    "registration_status": "unverified",
                },
                "provenance": {
                    "case_name": cfg.case_name,
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_hash": checkpoint_hash,
                    "config_hash": config_hash,
                    "contour_manifest_hash": contour_manifest_hash,
                    "tool_call": {
                        "tool": "stgpt.spatho.run_spatho_export",
                        "batch_size": int(batch_size),
                        "device": str(device),
                    },
                },
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _build_region_qc_report(cfg: StGPTConfig, checkpoint: Path, frame: pd.DataFrame, dataset) -> dict[str, Any]:
    n_regions = int(len(frame))
    n_with_image = int((frame["qc_flag"] == "ok").sum()) if "qc_flag" in frame else 0
    return {
        "case_name": cfg.case_name,
        "checkpoint": str(checkpoint),
        "training_unit": "region",
        "n_regions_total": n_regions,
        "n_regions_with_image": n_with_image,
        "n_regions_no_image": n_regions - n_with_image,
        "n_cells_assigned": int(dataset.cell_membership["cell_id"].nunique()) if not dataset.cell_membership.empty else 0,
        "image_coverage": round(n_with_image / max(1, n_regions), 4),
        "structure_counts": frame["structure_label"].value_counts(dropna=False).sort_index().astype(int).to_dict()
        if "structure_label" in frame
        else {},
    }


def _evidence_id(case_name: str, row: pd.Series, row_idx: int) -> str:
    region_id = _string_or_none(row.get("region_id")) or f"row-{row_idx}"
    slide_id = _string_or_none(row.get("slide_id")) or "slide-unknown"
    digest = hashlib.sha256(f"{case_name}|{slide_id}|{region_id}|{row_idx}".encode()).hexdigest()[:16]
    return f"ev_{digest}"


def _image_pointer(row: pd.Series, cfg: StGPTConfig, *, row_idx: int, source_row_index: int | None) -> dict[str, Any]:
    store = _string_or_none(row.get("image_store")) or cfg.data.contour_image_store
    if store and source_row_index is not None:
        return {
            "artifact": str(store),
            "row_index": source_row_index,
            "arrays": {
                "object_rgb": cfg.data.object_rgb_key,
                "context_rgb": cfg.data.context_rgb_key,
                "mask": cfg.data.mask_key,
            },
        }
    image_path = _string_or_none(row.get("image_path"))
    if image_path:
        return {"artifact": image_path, "row_index": None, "source": "image_path"}
    return {"artifact": "region_image_manifest.json", "row_index": row_idx, "source": "manifest_fallback"}


def _geometry_pointer(cfg: StGPTConfig, *, row_idx: int, source_row_index: int | None) -> dict[str, Any]:
    contour_manifest = cfg.data.contour_manifest
    if contour_manifest and source_row_index is not None:
        return {"artifact": str(contour_manifest), "row_index": source_row_index, "columns": "geometry"}
    return {"artifact": "region_image_manifest.json", "row_index": row_idx, "columns": "geometry_unavailable"}


def _image_source(row: pd.Series, cfg: StGPTConfig, source_row_index: int | None) -> str:
    if (_string_or_none(row.get("image_store")) or cfg.data.contour_image_store) and source_row_index is not None:
        return "contour_store"
    if _string_or_none(row.get("image_path")):
        return "image_path"
    return "zero_fallback"


def _config_hash(cfg: StGPTConfig) -> str:
    payload = json.dumps(cfg.model_dump(mode="json"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_optional_path(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    return _sha256_path(path)


def _sha256_path(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nullable_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _nullable_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _string_or_none(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value)
    return text if text and text.lower() not in {"nan", "none"} else None


def _compute_qc_flags(case, cfg: StGPTConfig) -> list[str]:
    """Compute per-cell QC flags based on patch manifest coverage."""
    patch_table = case.patch_table
    if "cell_id" in patch_table.columns and "image_path" in patch_table.columns:
        valid_rows = patch_table[patch_table["cell_id"].notna() & patch_table["image_path"].notna()].copy()
        covered: set[str] = {
            str(row.cell_id)
            for row in valid_rows.itertuples(index=False)
            if Path(str(row.image_path)).exists()
        }
    else:
        covered = set()

    cell_ids = (
        case.adata.obs["cell_id"].astype(str).tolist()
        if "cell_id" in case.adata.obs.columns
        else case.adata.obs_names.astype(str).tolist()
    )
    return ["ok" if cid in covered else "no_image" for cid in cell_ids]


def _build_cell_embedding_frame(embedded, cfg: StGPTConfig, qc_flags: list[str]) -> pd.DataFrame:
    """Build the cell_embeddings DataFrame with the required schema."""
    emb_matrix = np.asarray(embedded.obsm["X_stGPT"], dtype=np.float32)
    n_cells, emb_dim = emb_matrix.shape

    cell_ids = (
        embedded.obs["cell_id"].astype(str).tolist() if "cell_id" in embedded.obs.columns else embedded.obs_names.astype(str).tolist()
    )

    spatial_key = cfg.data.spatial_key
    if spatial_key in embedded.obsm:
        coords = np.asarray(embedded.obsm[spatial_key], dtype=np.float64)[:, :2]
    else:
        coords = np.full((n_cells, 2), np.nan)

    structure_col = cfg.data.structure_key
    structure_labels = (
        embedded.obs[structure_col].astype(str).tolist() if structure_col in embedded.obs.columns else ["unknown"] * n_cells
    )

    frame = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "x": coords[:, 0],
            "y": coords[:, 1],
            "structure_label": structure_labels,
            "qc_flag": qc_flags,
        }
    )
    for dim_idx in range(emb_dim):
        frame[f"emb_{dim_idx}"] = emb_matrix[:, dim_idx]
    return frame


def _build_structure_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate region embeddings to structure-level mean embeddings."""
    emb_cols = [col for col in frame.columns if str(col).startswith("emb_")]
    if not emb_cols:
        return pd.DataFrame(columns=list(STRUCTURE_SUMMARY_REQUIRED_COLUMNS))

    if "n_cells" in frame.columns:
        count_frame = frame.groupby("structure_label", sort=True)["n_cells"].sum().rename("n_cells").reset_index()
    else:
        count_frame = frame.groupby("structure_label", sort=True).size().rename("n_cells").reset_index()
    mean_frame = frame.groupby("structure_label", sort=True)[emb_cols].mean().reset_index()
    summary = count_frame.merge(mean_frame, on="structure_label", how="inner")
    ordered_cols = ["structure_label", "n_cells"] + emb_cols
    return summary[ordered_cols].reset_index(drop=True)


def _build_export_qc_report(
    cfg: StGPTConfig,
    checkpoint: Path,
    frame: pd.DataFrame,
    n_cells_with_image: int,
) -> dict[str, Any]:
    """Build the operational QC report for an export run."""
    n_cells = int(len(frame))
    structure_counts = frame["structure_label"].value_counts(dropna=False).sort_index()
    return {
        "case_name": cfg.case_name,
        "checkpoint": str(checkpoint),
        "n_cells_total": n_cells,
        "n_cells_with_image": n_cells_with_image,
        "n_cells_no_image": n_cells - n_cells_with_image,
        "image_coverage": round(n_cells_with_image / max(1, n_cells), 4),
        "structure_counts": {str(k): int(v) for k, v in structure_counts.items()},
    }
