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
* ``region_qc_report.json`` – operational QC summary for the export run.
* ``structure_summary.parquet`` – one row per structure with cell count and mean
  embedding vector.

Downstream tools should treat :data:`REGION_EMBEDDING_REQUIRED_COLUMNS` and
:data:`STRUCTURE_SUMMARY_REQUIRED_COLUMNS` as the stable region-first contract;
additional ``emb_*`` columns carry the actual embedding dimensions.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import StGPTConfig
from .data import build_training_case
from .foundation.packaging import resolve_model_checkpoint
from .inference import embed_regions

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

    region_table, embeddings, dataset = embed_regions(
        cfg,
        checkpoint=checkpoint_path,
        batch_size=batch_size,
        device=device,
    )
    frame = _build_region_embedding_frame(region_table, embeddings)

    region_emb_path = out_dir / "region_embeddings.parquet"
    membership_path = out_dir / "region_cell_membership.parquet"
    molecular_path = out_dir / "region_molecular_summary.parquet"
    image_manifest_path = out_dir / "region_image_manifest.json"
    region_qc_path = out_dir / "region_qc_report.json"
    evidence_manifest_path = out_dir / "evidence_manifest.json"
    struct_sum_path = out_dir / "structure_summary.parquet"
    struct_sum_csv_path = out_dir / "structure_embedding_summary.csv"

    frame.to_parquet(region_emb_path, index=False)
    dataset.cell_membership.to_parquet(membership_path, index=False)
    _build_region_molecular_summary(dataset).to_parquet(molecular_path, index=False)
    image_manifest_path.write_text(json.dumps(_build_region_image_manifest(region_table), indent=2), encoding="utf-8")
    summary = _build_structure_summary(frame)
    summary.to_parquet(struct_sum_path, index=False)
    summary.to_csv(struct_sum_csv_path, index=False)

    emb_cols = [col for col in frame.columns if str(col).startswith("emb_")]
    n_regions = int(len(frame))
    n_with_image = int((frame["qc_flag"] == "ok").sum()) if "qc_flag" in frame else 0
    qc_payload = _build_region_qc_report(cfg, checkpoint_path, frame, dataset)
    region_qc_path.write_text(json.dumps(qc_payload, indent=2), encoding="utf-8")
    evidence_manifest_path.write_text(
        json.dumps(
            {
                "case_name": cfg.case_name,
                "checkpoint": str(checkpoint_path),
                "training_unit": "region",
                "artifacts": {
                    "region_embeddings": str(region_emb_path),
                    "region_cell_membership": str(membership_path),
                    "region_molecular_summary": str(molecular_path),
                    "region_image_manifest": str(image_manifest_path),
                    "region_qc_report": str(region_qc_path),
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


def _build_region_image_manifest(region_table: pd.DataFrame) -> dict[str, Any]:
    cols = [col for col in ("region_id", "image_path", "patch_x", "patch_y", "patch_size", "source_image", "registration_transform") if col in region_table]
    return {"regions": region_table[cols].to_dict(orient="records") if cols else []}


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
