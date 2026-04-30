"""spatho adapter for stGPT: typed I/O contracts and end-to-end export pipeline.

This module defines the stable interface between stGPT's morpho-molecular embedding
model and downstream spatial-pathology tools such as spatho.  The key entry point is
:func:`run_spatho_export`, which accepts a config, a trained checkpoint, and an output
directory, and writes three versioned artifacts:

* ``cell_embeddings.parquet`` – one row per cell with spatial coordinates, structure
  label, QC flag, and the embedding vector.
* ``structure_summary.parquet`` – one row per structure with cell count and mean
  embedding vector.
* ``qc_report.json`` – operational QC summary for the export run.

Downstream tools should treat :data:`CELL_EMBEDDING_REQUIRED_COLUMNS` and
:data:`STRUCTURE_SUMMARY_REQUIRED_COLUMNS` as the stable contract; additional
``emb_*`` columns carry the actual embedding dimensions.
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
from .inference import embed_anndata

#: Required non-embedding columns guaranteed to be present in ``cell_embeddings.parquet``.
CELL_EMBEDDING_REQUIRED_COLUMNS: tuple[str, ...] = ("cell_id", "x", "y", "structure_label", "qc_flag")

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
        cell_embeddings: Path to ``cell_embeddings.parquet``.
        structure_summary: Path to ``structure_summary.parquet``.
        qc_report: Path to ``qc_report.json``.
        n_cells: Total number of cells embedded.
        n_cells_with_image: Number of cells whose H&E patch was found and loaded.
        embedding_dim: Dimensionality of the embedding vectors (number of ``emb_*``
            columns in ``cell_embeddings.parquet``).
    """

    cell_embeddings: Path
    structure_summary: Path
    qc_report: Path
    n_cells: int
    n_cells_with_image: int
    embedding_dim: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        payload = asdict(self)
        payload["cell_embeddings"] = str(self.cell_embeddings)
        payload["structure_summary"] = str(self.structure_summary)
        payload["qc_report"] = str(self.qc_report)
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

    Loads the Xenium/AnnData case described by *config*, embeds all cells using the
    pretrained *checkpoint*, and writes three output artifacts to *output_dir*.

    The stable output schema is:

    ``cell_embeddings.parquet``
        Columns: ``cell_id`` (str), ``x`` (float), ``y`` (float),
        ``structure_label`` (str), ``qc_flag`` (str: ``"ok"`` | ``"no_image"``),
        ``emb_0`` … ``emb_{d-1}`` (float32).

    ``structure_summary.parquet``
        Columns: ``structure_label`` (str), ``n_cells`` (int),
        ``emb_0`` … ``emb_{d-1}`` (float32, mean over cells in structure).

    ``qc_report.json``
        Operational QC: cell counts, image coverage, per-structure counts.

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
        print(result.cell_embeddings)   # .../cell_embeddings.parquet
    """
    cfg = StGPTConfig.from_file(config) if isinstance(config, (str, Path)) else config
    checkpoint_path = Path(checkpoint)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    case = build_training_case(cfg)
    embedded = embed_anndata(case.adata, checkpoint=checkpoint_path, batch_size=batch_size, device=device)

    qc_flags = _compute_qc_flags(case, cfg)
    frame = _build_cell_embedding_frame(embedded, cfg, qc_flags)

    cell_emb_path = out_dir / "cell_embeddings.parquet"
    struct_sum_path = out_dir / "structure_summary.parquet"
    qc_report_path = out_dir / "qc_report.json"

    frame.to_parquet(cell_emb_path, index=False)

    summary = _build_structure_summary(frame)
    summary.to_parquet(struct_sum_path, index=False)

    n_cells = int(len(frame))
    n_cells_with_image = int((frame["qc_flag"] == "ok").sum())
    emb_cols = [col for col in frame.columns if str(col).startswith("emb_")]
    embedding_dim = len(emb_cols)

    qc_payload = _build_export_qc_report(cfg, checkpoint_path, frame, n_cells_with_image)
    qc_report_path.write_text(json.dumps(qc_payload, indent=2), encoding="utf-8")

    return SpathoExportResult(
        cell_embeddings=cell_emb_path,
        structure_summary=struct_sum_path,
        qc_report=qc_report_path,
        n_cells=n_cells,
        n_cells_with_image=n_cells_with_image,
        embedding_dim=embedding_dim,
    )


def _compute_qc_flags(case, cfg: StGPTConfig) -> list[str]:
    """Compute per-cell QC flags based on patch manifest coverage."""
    patch_table = case.patch_table
    if "cell_id" in patch_table.columns and "image_path" in patch_table.columns:
        covered: set[str] = set()
        for _, row in patch_table.iterrows():
            if pd.notna(row.get("cell_id")) and pd.notna(row.get("image_path")) and Path(str(row["image_path"])).exists():
                covered.add(str(row["cell_id"]))
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
    """Aggregate cell embeddings to structure-level mean embeddings."""
    emb_cols = [col for col in frame.columns if str(col).startswith("emb_")]
    if not emb_cols:
        return pd.DataFrame(columns=list(STRUCTURE_SUMMARY_REQUIRED_COLUMNS))

    count_frame = frame.groupby("structure_label", sort=True)["cell_id"].count().rename("n_cells").reset_index()
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
