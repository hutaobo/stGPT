from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from .config import StGPTConfig
from .data import TrainingCase, build_training_case


def validate_data(config: StGPTConfig | str | Path, *, output_dir: str | Path | None = None) -> dict[str, Any]:
    cfg = StGPTConfig.from_file(config) if isinstance(config, (str, Path)) else config
    case = build_training_case(cfg)
    return validate_training_case(case, cfg, output_dir=output_dir)


def validate_training_case(
    case: TrainingCase,
    config: StGPTConfig,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    out = Path(output_dir).expanduser() if output_dir is not None else config.data.output_path / "qc"
    out.mkdir(parents=True, exist_ok=True)

    manifest = build_case_manifest(case, config, out)
    report = build_qc_report(case, config, manifest)
    splits = make_splits(case, config)

    case_manifest_path = out / "case_manifest.json"
    qc_report_json_path = out / "qc_report.json"
    qc_report_md_path = out / "qc_report.md"
    splits_path = out / "splits.csv"

    case_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    qc_report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    qc_report_md_path.write_text(render_qc_markdown(report, manifest), encoding="utf-8")
    splits.to_csv(splits_path, index=False)

    return {
        "status": report["status"],
        "case_manifest": str(case_manifest_path),
        "qc_report_json": str(qc_report_json_path),
        "qc_report_md": str(qc_report_md_path),
        "splits": str(splits_path),
        "fatal_errors": report["fatal_errors"],
        "warnings": report["warnings"],
        "n_cells": manifest["n_cells"],
        "n_genes": manifest["n_genes"],
    }


def build_case_manifest(case: TrainingCase, config: StGPTConfig, output_dir: Path | None = None) -> dict[str, Any]:
    adata = case.adata
    matrix = _matrix(adata)
    spatial = _spatial_array(case, config)
    patch_table = case.patch_table.copy()
    gene_names = _gene_names(case, config)
    panel_genes = _panel_genes(config)
    structure_counts = _structure_counts(case, config)
    patch_provenance = _patch_provenance(patch_table)
    return {
        "case_name": config.case_name,
        "mode": config.data.mode,
        "paths": {
            "dataset_root": _path_text(config.data.path_or_none(config.data.dataset_root)),
            "input_h5ad": _path_text(config.data.path_or_none(config.data.input_h5ad)),
            "spatho_run_root": _path_text(config.data.path_or_none(config.data.spatho_run_root)),
            "patch_manifest": _path_text(config.data.path_or_none(config.data.patch_manifest)),
            "structure_assignments_csv": _path_text(config.data.path_or_none(config.data.structure_assignments_csv)),
            "case_output_dir": str(case.output_dir),
            "qc_output_dir": str(output_dir) if output_dir is not None else None,
        },
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "gene_name_key": config.data.gene_name_key,
        "duplicate_gene_count": int(pd.Series(gene_names).duplicated().sum()),
        "panel": _panel_metrics(gene_names, panel_genes),
        "spatial_key": config.data.spatial_key,
        "has_spatial": spatial is not None,
        "spatial_bounds": _spatial_bounds(spatial),
        "matrix_density": _matrix_density(matrix, adata.n_obs, adata.n_vars),
        "patch_count": int(len(patch_table)),
        "patch_cell_count": int(patch_table["cell_id"].dropna().astype(str).nunique()) if "cell_id" in patch_table else 0,
        "patch_structure_count": int(patch_table["structure_id"].dropna().nunique()) if "structure_id" in patch_table else 0,
        "patch_provenance": patch_provenance,
        "structure_key": config.data.structure_key,
        "include_structure_context": bool(config.data.include_structure_context),
        "structure_counts": structure_counts,
        "split": {
            "strategy": config.split.strategy,
            "group_key": config.split.group_key,
            "train_fraction": float(config.split.train_fraction),
            "val_fraction": float(config.split.val_fraction),
            "test_fraction": float(config.split.test_fraction),
            "seed": int(_resolve_split_seed(config)),
        },
    }


def build_qc_report(case: TrainingCase, config: StGPTConfig, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest = manifest or build_case_manifest(case, config)
    adata = case.adata
    patch_table = case.patch_table.copy()
    fatal_errors: list[str] = []
    warnings: list[str] = []

    spatial = _spatial_array(case, config)
    missing_spatial_count = 0
    if spatial is None:
        fatal_errors.append(f"AnnData is missing obsm[{config.data.spatial_key!r}] spatial coordinates.")
        missing_spatial_count = int(adata.n_obs)
    else:
        finite_rows = np.isfinite(spatial[:, :2]).all(axis=1)
        missing_spatial_count = int((~finite_rows).sum())
        if missing_spatial_count:
            fatal_errors.append(f"{missing_spatial_count} cells have missing or non-finite spatial coordinates.")

    duplicate_gene_count = int(manifest["duplicate_gene_count"])
    if duplicate_gene_count:
        warnings.append(f"{duplicate_gene_count} duplicate gene names were found before vocabulary uniquification.")

    panel = manifest["panel"]
    if panel["configured"]:
        if panel["missing_from_data_count"]:
            warnings.append(f"{panel['missing_from_data_count']} configured panel genes are missing from the data matrix.")
        if panel["outside_panel_count"]:
            warnings.append(f"{panel['outside_panel_count']} data genes are outside the configured panel.")

    cell_ids = _cell_ids(case)
    patch_cell_ids = patch_table["cell_id"].dropna().astype(str) if "cell_id" in patch_table else pd.Series(dtype=str)
    patch_cell_coverage = float(patch_cell_ids[patch_cell_ids.isin(cell_ids)].nunique() / max(1, len(cell_ids)))
    patch_structure_coverage = _patch_structure_coverage(case, config, patch_table)
    if len(patch_table) == 0:
        warnings.append("No patch manifest rows were found; image inputs will fall back to zero tensors.")
    elif patch_cell_coverage < 1.0 and patch_structure_coverage < 1.0:
        warnings.append(
            f"Patch manifest covers {patch_cell_coverage:.1%} of cells by cell_id and {patch_structure_coverage:.1%} of structures."
        )

    missing_image_count = _missing_image_count(patch_table)
    if missing_image_count:
        warnings.append(f"{missing_image_count} patch rows point to missing image files.")
    patch_provenance = manifest["patch_provenance"]
    if len(patch_table) > 0 and not patch_provenance["has_coordinates"]:
        warnings.append("Patch manifest lacks explicit patch coordinate columns; registration traceability is limited.")
    if len(patch_table) > 0 and not patch_provenance["has_registration_metadata"]:
        warnings.append("Patch manifest lacks registration or transform metadata; H&E alignment assumptions should be documented.")

    structure_coverage = _structure_coverage(case, config)
    if config.data.include_structure_context and structure_coverage < 1.0:
        warnings.append(f"Structure context is enabled but covers {structure_coverage:.1%} of cells.")

    status = "fail" if fatal_errors else "pass"
    return {
        "case_name": config.case_name,
        "status": status,
        "fatal_errors": fatal_errors,
        "warnings": warnings,
        "metrics": {
            "patch_cell_coverage": patch_cell_coverage,
            "patch_structure_coverage": patch_structure_coverage,
            "missing_image_count": int(missing_image_count),
            "duplicate_gene_count": duplicate_gene_count,
            "missing_spatial_count": int(missing_spatial_count),
            "structure_assignment_coverage": structure_coverage,
            "matrix_density": manifest["matrix_density"],
            "panel_missing_from_data_count": int(panel["missing_from_data_count"]),
            "panel_outside_panel_count": int(panel["outside_panel_count"]),
        },
    }


def make_splits(case: TrainingCase, config: StGPTConfig) -> pd.DataFrame:
    adata = case.adata
    cell_ids = _cell_ids(case)
    n_cells = int(adata.n_obs)
    if n_cells == 0:
        return pd.DataFrame(columns=["cell_id", "split", "split_strategy", "block_id"])
    if config.split.strategy == "group_holdout":
        if not config.split.group_key or config.split.group_key not in adata.obs.columns:
            raise ValueError("split.strategy='group_holdout' requires split.group_key to name an AnnData obs column.")
        block_ids = adata.obs[config.split.group_key].fillna("missing").astype(str).to_numpy(dtype=object)
    elif config.split.strategy == "spatial_block":
        spatial = _spatial_array(case, config)
        if spatial is None:
            order = np.arange(n_cells)
            block_ids = np.asarray([f"idx_{idx:04d}" for idx in order], dtype=object)
        else:
            coords = np.asarray(spatial[:, :2], dtype=np.float64)
            coords[~np.isfinite(coords)] = 0.0
            n_bins = max(2, min(32, int(np.ceil(np.sqrt(max(4.0, n_cells / 4.0))))))
            x_bin = _rank_bins(coords[:, 0], n_bins)
            y_bin = _rank_bins(coords[:, 1], n_bins)
            block_ids = np.asarray([f"x{int(x)}_y{int(y)}" for x, y in zip(x_bin, y_bin, strict=False)], dtype=object)
    else:
        raise ValueError(f"Unsupported split strategy: {config.split.strategy}")
    assignment = _assign_blocks_to_splits(block_ids, config)
    split_values = [assignment[str(block)] for block in block_ids]
    return pd.DataFrame(
        {
            "cell_id": cell_ids,
            "split": split_values,
            "split_strategy": config.split.strategy,
            "block_id": block_ids,
        }
    )


def render_qc_markdown(report: dict[str, Any], manifest: dict[str, Any]) -> str:
    lines = [
        f"# stGPT QC Report: {report['case_name']}",
        "",
        f"Status: **{report['status']}**",
        "",
        "## Case Manifest",
        "",
        f"- Mode: `{manifest['mode']}`",
        f"- Cells: {manifest['n_cells']}",
        f"- Genes: {manifest['n_genes']}",
        f"- Matrix density: {manifest['matrix_density']:.6f}",
        f"- Patch rows: {manifest['patch_count']}",
        f"- Patch cell coverage: {report['metrics']['patch_cell_coverage']:.1%}",
        f"- Structure assignment coverage: {report['metrics']['structure_assignment_coverage']:.1%}",
        f"- Panel genes configured: {manifest['panel']['panel_gene_count']}",
        f"- Panel genes missing from data: {report['metrics']['panel_missing_from_data_count']}",
        f"- Patch coordinate columns: {', '.join(manifest['patch_provenance']['coordinate_columns']) or 'None'}",
        f"- Patch registration columns: {', '.join(manifest['patch_provenance']['registration_columns']) or 'None'}",
        "",
        "## Fatal Errors",
        "",
    ]
    lines.extend([f"- {item}" for item in report["fatal_errors"]] or ["- None"])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {item}" for item in report["warnings"]] or ["- None"])
    lines.extend(["", "## Split", ""])
    split = manifest["split"]
    lines.extend(
        [
            f"- Strategy: `{split['strategy']}`",
            f"- Group key: `{split['group_key']}`",
            f"- Fractions: train={split['train_fraction']:.2f}, val={split['val_fraction']:.2f}, test={split['test_fraction']:.2f}",
            f"- Seed: {split['seed']}",
            "",
        ]
    )
    return "\n".join(lines)


def _matrix(adata) -> sparse.csr_matrix:
    matrix = adata.layers["rna"] if "rna" in adata.layers else adata.X
    return matrix.tocsr() if sparse.issparse(matrix) else sparse.csr_matrix(np.asarray(matrix))


def _matrix_density(matrix: sparse.csr_matrix, n_obs: int, n_vars: int) -> float:
    denominator = int(n_obs) * int(n_vars)
    return 0.0 if denominator <= 0 else float(matrix.nnz / denominator)


def _spatial_array(case: TrainingCase, config: StGPTConfig) -> np.ndarray | None:
    if config.data.spatial_key not in case.adata.obsm:
        return None
    spatial = np.asarray(case.adata.obsm[config.data.spatial_key])
    if spatial.ndim != 2 or spatial.shape[1] < 2:
        return None
    return spatial


def _spatial_bounds(spatial: np.ndarray | None) -> dict[str, float | None]:
    if spatial is None or spatial.shape[0] == 0:
        return {"x_min": None, "x_max": None, "y_min": None, "y_max": None}
    coords = np.asarray(spatial[:, :2], dtype=np.float64)
    finite = np.isfinite(coords).all(axis=1)
    if not finite.any():
        return {"x_min": None, "x_max": None, "y_min": None, "y_max": None}
    valid = coords[finite]
    return {
        "x_min": float(valid[:, 0].min()),
        "x_max": float(valid[:, 0].max()),
        "y_min": float(valid[:, 1].min()),
        "y_max": float(valid[:, 1].max()),
    }


def _gene_names(case: TrainingCase, config: StGPTConfig) -> list[str]:
    if config.data.gene_name_key in case.adata.var.columns:
        return case.adata.var[config.data.gene_name_key].astype(str).tolist()
    return case.adata.var_names.astype(str).tolist()


def _cell_ids(case: TrainingCase) -> list[str]:
    if "cell_id" in case.adata.obs.columns:
        return case.adata.obs["cell_id"].astype(str).tolist()
    return case.adata.obs_names.astype(str).tolist()


def _structure_counts(case: TrainingCase, config: StGPTConfig) -> dict[str, int]:
    if config.data.structure_key not in case.adata.obs.columns:
        return {}
    counts = case.adata.obs[config.data.structure_key].astype(str).value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _panel_genes(config: StGPTConfig) -> list[str]:
    genes: list[str] = []
    if config.data.panel_genes:
        genes.extend(str(item) for item in config.data.panel_genes)
    panel_path = config.data.path_or_none(config.data.panel_gene_file)
    if panel_path is not None and panel_path.exists():
        if panel_path.suffix.lower() in {".csv", ".tsv"}:
            sep = "\t" if panel_path.suffix.lower() == ".tsv" else ","
            frame = pd.read_csv(panel_path, sep=sep)
            if not frame.empty:
                genes.extend(frame.iloc[:, 0].dropna().astype(str).tolist())
        else:
            genes.extend(line.strip() for line in panel_path.read_text(encoding="utf-8").splitlines())
    return _unique_nonempty(genes)


def _panel_metrics(gene_names: list[str], panel_genes: list[str]) -> dict[str, Any]:
    data_genes = set(_unique_nonempty(gene_names))
    panel_set = set(panel_genes)
    missing = sorted(panel_set.difference(data_genes))
    outside = sorted(data_genes.difference(panel_set)) if panel_set else []
    return {
        "configured": bool(panel_set),
        "panel_gene_count": int(len(panel_set)),
        "data_gene_count": int(len(data_genes)),
        "genes_in_panel_count": int(len(data_genes.intersection(panel_set))) if panel_set else 0,
        "missing_from_data_count": int(len(missing)),
        "outside_panel_count": int(len(outside)),
        "missing_from_data_preview": missing[:20],
        "outside_panel_preview": outside[:20],
    }


def _patch_provenance(patch_table: pd.DataFrame) -> dict[str, Any]:
    columns = [str(col) for col in patch_table.columns]
    lowered = {col: col.lower() for col in columns}
    coordinate_markers = {
        "x",
        "y",
        "patch_x",
        "patch_y",
        "center_x",
        "center_y",
        "x_centroid",
        "y_centroid",
        "pixel_x",
        "pixel_y",
    }
    coordinate_cols = [col for col, lower in lowered.items() if lower in coordinate_markers or lower.endswith(("_x", "_y"))]
    image_cols = [
        col
        for col, lower in lowered.items()
        if lower in {"image_path", "source_image", "wsi_path", "slide_path", "he_image_path"} or "image" in lower
    ]
    transform_cols = [
        col
        for col, lower in lowered.items()
        if "registration" in lower or "transform" in lower or "affine" in lower or "matrix" in lower
    ]
    parameter_cols = [
        col
        for col, lower in lowered.items()
        if lower in {"patch_size", "level", "magnification", "mpp", "scale", "stride"} or lower.startswith("patch_")
    ]
    return {
        "columns": columns,
        "coordinate_columns": coordinate_cols,
        "image_columns": image_cols,
        "registration_columns": transform_cols,
        "parameter_columns": parameter_cols,
        "has_coordinates": bool(coordinate_cols),
        "has_source_image": bool(image_cols),
        "has_registration_metadata": bool(transform_cols),
    }


def _unique_nonempty(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def _structure_coverage(case: TrainingCase, config: StGPTConfig) -> float:
    if config.data.structure_key not in case.adata.obs.columns:
        return 0.0
    values = case.adata.obs[config.data.structure_key]
    return float(values.notna().sum() / max(1, case.adata.n_obs))


def _patch_structure_coverage(case: TrainingCase, config: StGPTConfig, patch_table: pd.DataFrame) -> float:
    if "structure_id" not in patch_table or config.data.structure_key not in case.adata.obs.columns:
        return 0.0
    structures = case.adata.obs[config.data.structure_key].dropna().astype(str)
    if structures.empty:
        return 0.0
    patch_structures = patch_table["structure_id"].dropna().map(_normalize_structure_id).astype(str)
    return float(patch_structures[patch_structures.isin(set(structures))].nunique() / max(1, structures.nunique()))


def _missing_image_count(patch_table: pd.DataFrame) -> int:
    if "image_path" not in patch_table:
        return 0
    count = 0
    for value in patch_table["image_path"].dropna():
        if not Path(str(value)).exists():
            count += 1
    return int(count)


def _assign_blocks_to_splits(block_ids: np.ndarray, config: StGPTConfig) -> dict[str, str]:
    rng = np.random.default_rng(_resolve_split_seed(config))
    blocks = np.asarray(sorted(pd.unique(block_ids)), dtype=object)
    rng.shuffle(blocks)
    block_sizes = {block: int((block_ids == block).sum()) for block in blocks}
    n_cells = int(len(block_ids))
    assignment: dict[str, str] = {}
    cumulative = 0.0
    train_cut = n_cells * float(config.split.train_fraction)
    val_cut = n_cells * float(config.split.train_fraction + config.split.val_fraction)
    for block in blocks:
        size = block_sizes[block]
        midpoint = cumulative + size / 2.0
        if midpoint < train_cut:
            split = "train"
        elif midpoint < val_cut:
            split = "val"
        else:
            split = "test"
        assignment[str(block)] = split
        cumulative += size
    return assignment


def _normalize_structure_id(value: object) -> str:
    try:
        numeric = float(str(value))
    except ValueError:
        return str(value)
    if numeric.is_integer():
        return str(int(numeric))
    return str(value)


def _rank_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(values.shape[0])
    return np.minimum((ranks * n_bins) // max(1, values.shape[0]), n_bins - 1)


def _resolve_split_seed(config: StGPTConfig) -> int:
    if config.split.seed is not None:
        return int(config.split.seed)
    if int(config.training.seed) != 0:
        return int(config.training.seed)
    return int(config.data.seed)


def _path_text(path: Path | None) -> str | None:
    return str(path) if path is not None else None
