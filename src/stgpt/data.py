from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset

from .config import DataConfig, StGPTConfig
from .images import load_image_tensor, write_synthetic_patch
from .tokenization import ExpressionBinner, GeneVocab


@dataclass
class TrainingCase:
    adata: ad.AnnData
    patch_table: pd.DataFrame
    output_dir: Path
    region_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    cell_membership: pd.DataFrame = field(default_factory=pd.DataFrame)
    region_expression: sparse.csr_matrix = field(default_factory=lambda: sparse.csr_matrix((0, 0), dtype=np.float32))


def load_xenium_case(config: StGPTConfig | DataConfig | str | Path) -> ad.AnnData:
    cfg = _coerce_data_config(config)
    if cfg.mode == "synthetic":
        return make_synthetic_case(cfg).adata
    if cfg.mode == "anndata":
        input_h5ad = cfg.path_or_none(cfg.input_h5ad)
        if input_h5ad is None or not input_h5ad.exists():
            raise FileNotFoundError("data.input_h5ad must point to an existing .h5ad file.")
        adata = ad.read_h5ad(input_h5ad)
    elif cfg.mode == "xenium":
        dataset_root = cfg.path_or_none(cfg.dataset_root)
        if dataset_root is None or not dataset_root.exists():
            raise FileNotFoundError("data.dataset_root must point to an existing Xenium outs directory.")
        adata = _load_xenium_root(dataset_root)
    elif cfg.mode == "xenium_slide":
        adata = _load_xenium_slide(cfg)
    elif cfg.mode == "corpus":
        adata = _load_corpus(cfg)
    else:  # pragma: no cover - pydantic prevents this
        raise ValueError(f"Unsupported data mode: {cfg.mode}")
    _normalize_adata_contract(adata, cfg)
    _apply_case_metadata(adata, cfg)
    _merge_structure_assignments(adata, cfg)
    return adata


def build_training_case(config: StGPTConfig) -> TrainingCase:
    if config.data.mode == "synthetic":
        case = make_synthetic_case(config.data)
        return _build_region_training_case(case.adata, case.patch_table, config, output_dir=case.output_dir)
    adata = load_xenium_case(config)
    _merge_sibling_cell_to_contour(adata, config.data)
    patch_table = load_patch_table(config.data)
    if patch_table.empty and config.data.mode == "xenium_slide":
        patch_table = _load_xenium_slide_patch_table(config.data, adata)
    output_dir = config.data.output_path
    output_dir.mkdir(parents=True, exist_ok=True)
    return _build_region_training_case(adata, patch_table, config, output_dir=output_dir)


def ensure_region_training_case(case: TrainingCase, config: StGPTConfig) -> TrainingCase:
    """Return a TrainingCase with contour/region tables populated."""
    if not case.region_table.empty or case.adata.n_obs == 0:
        return case
    return _build_region_training_case(case.adata, case.patch_table, config, output_dir=case.output_dir)


def build_training_manifest(config: StGPTConfig | str | Path) -> dict[str, Any]:
    cfg = StGPTConfig.from_file(config) if isinstance(config, (str, Path)) else config
    case = build_training_case(cfg)
    out = case.output_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "case_name": cfg.case_name,
        "mode": cfg.data.mode,
        "case_metadata": _case_metadata(cfg.data),
        "n_cells": int(case.adata.n_obs),
        "n_regions": int(len(case.region_table)),
        "n_genes": int(case.adata.n_vars),
        "training_unit": "region",
        "region_id_key": cfg.data.region_id_key,
        "has_spatial": cfg.data.spatial_key in case.adata.obsm,
        "patch_count": int(len(case.patch_table)),
        "output_dir": str(out),
        "patch_table": str(out / "patch_table.csv"),
        "region_table": str(out / "region_table.csv"),
    }
    case.patch_table.to_csv(out / "patch_table.csv", index=False)
    case.region_table.to_csv(out / "region_table.csv", index=False)
    (out / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def make_synthetic_case(config: DataConfig) -> TrainingCase:
    rng = np.random.default_rng(config.seed)
    output_dir = config.output_path
    patch_dir = output_dir / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    n_cells = int(config.n_cells)
    n_genes = int(config.n_genes)
    n_structures = int(config.n_structures)
    n_regions = min(n_cells, max(8, n_structures * 4))
    region = np.arange(n_cells) % n_regions
    structure = region % n_structures
    coords = np.column_stack(
        [
            rng.normal(loc=structure * 120.0, scale=18.0, size=n_cells),
            rng.normal(loc=(structure % 2) * 80.0, scale=18.0, size=n_cells),
        ]
    ).astype(np.float32)
    programs = rng.gamma(shape=1.5, scale=1.0, size=(n_structures, n_genes)).astype(np.float32)
    for sid in range(n_structures):
        start = (sid * max(3, n_genes // n_structures)) % n_genes
        programs[sid, start : min(n_genes, start + max(3, n_genes // 8))] += 3.0
    expression = rng.poisson(programs[structure] + 0.2).astype(np.float32)
    obs = pd.DataFrame(
        {
            "cell_id": [f"cell_{idx:04d}" for idx in range(n_cells)],
            config.region_id_key: [f"contour_{rid:03d}" for rid in region],
            config.cluster_key: pd.Categorical([f"cluster_{sid}" for sid in structure]),
            config.structure_key: structure.astype(int),
            "structure_label": [f"structure_{sid}" for sid in structure],
            "x": coords[:, 0],
            "y": coords[:, 1],
        },
        index=[f"cell_{idx:04d}" for idx in range(n_cells)],
    )
    var = pd.DataFrame(
        {config.gene_name_key: [f"GENE{idx:03d}" for idx in range(n_genes)]},
        index=[f"GENE{idx:03d}" for idx in range(n_genes)],
    )
    adata = ad.AnnData(X=sparse.csr_matrix(expression), obs=obs, var=var)
    adata.layers["rna"] = adata.X.copy()
    adata.obsm[config.spatial_key] = coords
    rows: list[dict[str, Any]] = []
    for rid in range(n_regions):
        members = np.flatnonzero(region == rid)
        sid = int(rid % n_structures)
        center = coords[members].mean(axis=0) if members.size else np.asarray([0.0, 0.0], dtype=np.float32)
        mean_intensity = float(expression[members].mean() / max(1.0, expression.max())) if members.size else 0.0
        image_path = write_synthetic_patch(
            patch_dir / f"contour_{rid:03d}.png",
            image_size=config.image_size,
            structure_id=sid,
            intensity=mean_intensity,
            seed=config.seed + rid,
        )
        rows.append(
            {
                "cell_id": None,
                "contour_id": f"contour_{rid:03d}",
                "image_path": str(image_path),
                "patch_x": float(center[0]),
                "patch_y": float(center[1]),
                "patch_size": int(config.image_size),
                "source_image": "synthetic_he",
                "registration_transform": "identity",
                "structure_id": sid,
                "structure_label": f"structure_{sid}",
                "structure_name": f"structure_{sid}",
                "cluster_id": f"cluster_{sid}",
            }
        )
    patch_table = pd.DataFrame(rows)
    _normalize_adata_contract(adata, config)
    _apply_case_metadata(adata, config, source_name=config.slide_id or "synthetic_slide")
    return TrainingCase(adata=adata, patch_table=patch_table, output_dir=output_dir)


def load_patch_table(config: DataConfig) -> pd.DataFrame:
    path = config.path_or_none(config.patch_manifest)
    if path is None or not path.exists():
        return pd.DataFrame(columns=["cell_id", "structure_id", "image_path"])
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get("patches", payload.get("records", []))
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        patch_payload = row.get("patch")
        patch_meta: dict[str, Any] = patch_payload if isinstance(patch_payload, dict) else {}
        transform_payload = row.get("registration_transform") or row.get("transform") or patch_meta.get("registration_transform")
        normalized.append(
            {
                "cell_id": row.get("cell_id"),
                "contour_id": row.get("contour_id") or row.get("region_id"),
                "structure_id": row.get("structure_id"),
                "structure_label": row.get("structure_label"),
                "structure_name": row.get("structure_name"),
                "image_path": row.get("image_path") or patch_meta.get("image_path"),
                "patch_x": row.get("patch_x") or row.get("x") or patch_meta.get("patch_x") or patch_meta.get("x") or patch_meta.get("center_x"),
                "patch_y": row.get("patch_y") or row.get("y") or patch_meta.get("patch_y") or patch_meta.get("y") or patch_meta.get("center_y"),
                "patch_size": row.get("patch_size") or patch_meta.get("patch_size") or patch_meta.get("patch_size_px"),
                "source_image": row.get("source_image") or patch_meta.get("source_image"),
                "registration_transform": json.dumps(transform_payload, sort_keys=True) if isinstance(transform_payload, dict) else transform_payload,
                "bbox_level_xy": row.get("bbox_level_xy") or patch_meta.get("bbox_level_xy"),
                "bbox_level0_xy": row.get("bbox_level0_xy") or patch_meta.get("bbox_level0_xy"),
                "pyramid_level": row.get("pyramid_level") or patch_meta.get("pyramid_level"),
            }
        )
    return pd.DataFrame(normalized)


def _coerce_data_config(config: StGPTConfig | DataConfig | str | Path) -> DataConfig:
    if isinstance(config, DataConfig):
        return config
    if isinstance(config, StGPTConfig):
        return config.data
    return StGPTConfig.from_file(config).data


def _load_xenium_root(dataset_root: Path) -> ad.AnnData:
    try:
        from pyXenium.multimodal import load_rna_protein_anndata
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install stgpt[xenium] to load Xenium data through pyXenium.") from exc
    return load_rna_protein_anndata(str(dataset_root), read_morphology=False)


def _load_xenium_slide(config: DataConfig) -> ad.AnnData:
    slide_store = config.path_or_none(config.slide_store) or config.path_or_none(config.dataset_root)
    if slide_store is None or not slide_store.exists():
        raise FileNotFoundError("data.slide_store must point to an existing XeniumSlide zarr store.")
    try:
        from pyXenium.io import read_xenium_slide
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install pyXenium to load data.mode='xenium_slide'.") from exc
    slide = read_xenium_slide(slide_store)
    adata = slide.table.copy()
    adata.uns.setdefault("xenium_slide", {}).update(
        {
            "slide_store": str(slide_store),
            "metadata": slide.metadata,
            "component_summary": slide.component_summary(),
        }
    )
    return adata


def _load_xenium_slide_patch_table(config: DataConfig, adata: ad.AnnData) -> pd.DataFrame:
    slide_store = config.path_or_none(config.slide_store) or config.path_or_none(config.dataset_root)
    candidates: list[Path] = []
    slide_meta = adata.uns.get("xenium_slide", {})
    if isinstance(slide_meta, dict):
        contours = slide_meta.get("contours", {})
        if isinstance(contours, dict) and contours.get("contour_patches_manifest"):
            candidates.append(Path(str(contours["contour_patches_manifest"])).expanduser())
        metadata = slide_meta.get("metadata", {})
        if isinstance(metadata, dict):
            meta_contours = metadata.get("contours", {})
            if isinstance(meta_contours, dict) and meta_contours.get("contour_patches_manifest"):
                candidates.append(Path(str(meta_contours["contour_patches_manifest"])).expanduser())
    if slide_store is not None:
        candidates.append(slide_store.parent / "contour_patches_manifest.json")
    for candidate in candidates:
        if candidate.exists():
            payload = DataConfig(mode="anndata", patch_manifest=str(candidate), output_dir=config.output_dir)
            return load_patch_table(payload)
    return pd.DataFrame(columns=["contour_id", "structure_id", "image_path"])


def _load_corpus(config: DataConfig) -> ad.AnnData:
    inputs = config.paths_or_empty(config.input_h5ad_list)
    roots = config.paths_or_empty(config.dataset_roots)
    if not inputs and not roots:
        raise FileNotFoundError("data.mode='corpus' requires data.input_h5ad_list or data.dataset_roots.")

    adatas: list[ad.AnnData] = []
    keys: list[str] = []
    for idx, input_h5ad in enumerate(inputs):
        if not input_h5ad.exists():
            raise FileNotFoundError(f"Corpus AnnData file does not exist: {input_h5ad}")
        item = ad.read_h5ad(input_h5ad)
        _normalize_adata_contract(item, config)
        source_name = input_h5ad.stem or f"anndata_{idx}"
        _apply_case_metadata(item, config, source_name=source_name, source_index=idx)
        adatas.append(item)
        keys.append(source_name)

    offset = len(adatas)
    for idx, dataset_root in enumerate(roots):
        if not dataset_root.exists():
            raise FileNotFoundError(f"Corpus Xenium outs directory does not exist: {dataset_root}")
        item = _load_xenium_root(dataset_root)
        _normalize_adata_contract(item, config)
        source_name = dataset_root.name or f"xenium_{idx}"
        _apply_case_metadata(item, config, source_name=source_name, source_index=offset + idx)
        adatas.append(item)
        keys.append(source_name)

    if len(adatas) == 1:
        return adatas[0]
    merged = ad.concat(adatas, label="corpus_source", keys=keys, join="outer", fill_value=0.0, index_unique="-")
    if "rna" not in merged.layers:
        merged.layers["rna"] = merged.X.copy()
    if config.gene_name_key not in merged.var.columns:
        merged.var[config.gene_name_key] = merged.var_names.astype(str)
    return merged


def _normalize_adata_contract(adata: ad.AnnData, config: DataConfig) -> None:
    if "cell_id" not in adata.obs.columns:
        adata.obs["cell_id"] = adata.obs_names.astype(str)
    if config.spatial_key not in adata.obsm:
        for x_col, y_col in (("x", "y"), ("x_centroid", "y_centroid"), ("cell_x_centroid", "cell_y_centroid")):
            if x_col in adata.obs.columns and y_col in adata.obs.columns:
                adata.obsm[config.spatial_key] = adata.obs[[x_col, y_col]].to_numpy(dtype=np.float32)
                break
    if config.spatial_key not in adata.obsm:
        raise ValueError(f"AnnData must contain obsm[{config.spatial_key!r}] or recognized x/y columns.")
    if config.gene_name_key not in adata.var.columns:
        adata.var[config.gene_name_key] = adata.var_names.astype(str)


def _apply_case_metadata(
    adata: ad.AnnData,
    config: DataConfig,
    *,
    source_name: str | None = None,
    source_index: int = 0,
) -> None:
    source = source_name or config.slide_id or config.dataset_root or config.input_h5ad or f"case_{source_index}"
    metadata = {
        "slide_id": config.slide_id or (str(source) if config.mode == "corpus" else None),
        "patient_id": config.patient_id,
        "organ": config.organ,
        "batch_id": config.batch_id or (str(source) if config.mode == "corpus" else None),
        "stain": config.stain,
        "scanner": config.scanner,
    }
    for column, value in metadata.items():
        if value is not None and column not in adata.obs.columns:
            adata.obs[column] = str(value)


def _case_metadata(config: DataConfig) -> dict[str, str | None]:
    return {
        "slide_id": config.slide_id,
        "patient_id": config.patient_id,
        "organ": config.organ,
        "batch_id": config.batch_id,
        "stain": config.stain,
        "scanner": config.scanner,
    }


def _merge_structure_assignments(adata: ad.AnnData, config: DataConfig) -> None:
    path = config.path_or_none(config.structure_assignments_csv)
    if path is None or not path.exists() or config.structure_key in adata.obs.columns:
        return
    frame = pd.read_csv(path)
    cluster_candidates = [config.cluster_key, "cluster", "cluster_id", "graphclust"]
    structure_candidates = [config.structure_key, "structure_id", "structure", "structure_label"]
    cluster_col = next((col for col in cluster_candidates if col in frame.columns), None)
    structure_col = next((col for col in structure_candidates if col in frame.columns), None)
    if cluster_col is None or structure_col is None or config.cluster_key not in adata.obs.columns:
        return
    mapping = dict(zip(frame[cluster_col].astype(str), frame[structure_col], strict=False))
    adata.obs[config.structure_key] = adata.obs[config.cluster_key].astype(str).map(mapping).fillna(0).astype(int)


def _matrix(adata: ad.AnnData):
    matrix = adata.layers["rna"] if "rna" in adata.layers else adata.X
    return matrix.tocsr() if sparse.issparse(matrix) else sparse.csr_matrix(np.asarray(matrix))


def _build_region_training_case(
    adata: ad.AnnData,
    patch_table: pd.DataFrame,
    config: StGPTConfig,
    *,
    output_dir: Path,
) -> TrainingCase:
    matrix = _matrix(adata)
    region_key = config.data.region_id_key
    if region_key not in adata.obs.columns:
        if config.data.structure_key in adata.obs.columns:
            adata.obs[region_key] = adata.obs[config.data.structure_key].astype(str).map(lambda value: f"region_{value}")
        else:
            adata.obs[region_key] = adata.obs["cell_id"].astype(str) if "cell_id" in adata.obs.columns else adata.obs_names.astype(str)

    obs = adata.obs.copy()
    obs[region_key] = obs[region_key].astype(str)
    valid_obs = obs[obs[region_key].notna() & ~obs[region_key].isin(["", "nan", "None"])].copy()
    cell_ids = obs["cell_id"].astype(str) if "cell_id" in obs.columns else pd.Series(adata.obs_names.astype(str), index=obs.index)
    coords = np.asarray(adata.obsm[config.data.spatial_key], dtype=np.float32)[:, :2]
    coord_frame = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]}, index=obs.index)
    valid_obs["x"] = coord_frame.loc[valid_obs.index, "x"].to_numpy(dtype=np.float32)
    valid_obs["y"] = coord_frame.loc[valid_obs.index, "y"].to_numpy(dtype=np.float32)
    valid_obs["cell_id"] = cell_ids.loc[valid_obs.index].astype(str)

    grouped = valid_obs.groupby(region_key, sort=True, dropna=True)
    region_ids: list[str] = []
    region_expr_rows = []
    region_rows: list[dict[str, Any]] = []
    membership_rows: list[pd.DataFrame] = []
    patch_by_region = _patch_rows_by_region(patch_table)
    structure_lookup = _structure_lookup(valid_obs, config)

    for region_id, frame in grouped:
        if len(frame) < int(config.data.min_cells_per_region):
            continue
        indices = frame.index.map(adata.obs.index.get_loc).to_numpy(dtype=np.int64)
        expr = np.asarray(matrix[indices].mean(axis=0)).ravel().astype(np.float32)
        region_expr_rows.append(expr)
        patch_row = patch_by_region.get(str(region_id), {})
        raw_structure = patch_row.get("structure_id")
        if raw_structure is None:
            raw_structure = _first_non_null(frame[config.data.structure_key]) if config.data.structure_key in frame else None
        structure_label = patch_row.get("structure_label") or patch_row.get("structure_name")
        if structure_label is None:
            structure_label = structure_lookup.get(str(_lookup_key(raw_structure)), str(raw_structure if raw_structure is not None else "unknown"))
        region_row = {
            "region_id": str(region_id),
            "contour_id": str(region_id),
            "n_cells": int(len(frame)),
            "x": float(frame["x"].mean()),
            "y": float(frame["y"].mean()),
            "structure_id": raw_structure,
            "structure_label": str(structure_label),
            "image_path": patch_row.get("image_path"),
            "patch_x": patch_row.get("patch_x"),
            "patch_y": patch_row.get("patch_y"),
            "patch_size": patch_row.get("patch_size"),
            "source_image": patch_row.get("source_image"),
            "registration_transform": patch_row.get("registration_transform"),
            "bbox_level_xy": patch_row.get("bbox_level_xy"),
            "bbox_level0_xy": patch_row.get("bbox_level0_xy"),
            "pyramid_level": patch_row.get("pyramid_level"),
            "qc_flag": "ok" if patch_row.get("image_path") and Path(str(patch_row.get("image_path"))).exists() else "no_image",
        }
        if config.data.cluster_key in frame.columns:
            region_row[config.data.cluster_key] = _first_non_null(frame[config.data.cluster_key])
        for domain_col in ("slide_id", "patient_id", "organ", "batch_id", "stain", "scanner"):
            region_row[domain_col] = _first_non_null(frame[domain_col]) if domain_col in frame.columns else None
        region_rows.append(region_row)
        member = pd.DataFrame(
            {
                "region_id": str(region_id),
                "cell_id": frame["cell_id"].astype(str).to_numpy(),
                "x": frame["x"].to_numpy(dtype=np.float32),
                "y": frame["y"].to_numpy(dtype=np.float32),
                "membership": "inner",
                "weight": 1.0,
            }
        )
        membership_rows.append(member)
        region_ids.append(str(region_id))

    if region_expr_rows:
        region_expression = sparse.csr_matrix(np.vstack(region_expr_rows).astype(np.float32))
    else:
        region_expression = sparse.csr_matrix((0, adata.n_vars), dtype=np.float32)
    region_table = pd.DataFrame(region_rows)
    cell_membership = pd.concat(membership_rows, ignore_index=True) if membership_rows else pd.DataFrame(
        columns=["region_id", "cell_id", "x", "y", "membership", "weight"]
    )
    if not region_table.empty:
        region_table["region_id"] = pd.Categorical(region_table["region_id"], categories=region_ids, ordered=True).astype(str)
    return TrainingCase(
        adata=adata,
        patch_table=patch_table,
        output_dir=output_dir,
        region_table=region_table.reset_index(drop=True),
        cell_membership=cell_membership,
        region_expression=region_expression,
    )


def _merge_sibling_cell_to_contour(adata: ad.AnnData, config: DataConfig) -> None:
    existing_valid = pd.Series(False, index=adata.obs.index)
    if config.region_id_key in adata.obs.columns:
        existing = adata.obs[config.region_id_key]
        existing_valid = existing.notna() & ~existing.astype(str).isin(["", "nan", "None"])
    if bool(existing_valid.all()):
        return
    slide_store = config.path_or_none(config.slide_store) or config.path_or_none(config.dataset_root)
    if slide_store is None:
        return
    path = slide_store.parent / "cell_to_contour.parquet"
    if not path.exists():
        return
    frame = pd.read_parquet(path)
    if "cell_id" not in frame.columns or "contour_id" not in frame.columns:
        return
    cell_ids = adata.obs["cell_id"].astype(str) if "cell_id" in adata.obs.columns else pd.Series(adata.obs_names.astype(str), index=adata.obs.index)
    indexed = frame.drop_duplicates("cell_id").set_index("cell_id")
    mapped = cell_ids.map(indexed["contour_id"].astype(str))
    if config.region_id_key in adata.obs.columns:
        current = adata.obs[config.region_id_key].astype(object)
        adata.obs[config.region_id_key] = current.where(existing_valid, mapped).to_numpy()
    else:
        adata.obs[config.region_id_key] = mapped.to_numpy()
    for column in ("structure_id", "structure_label"):
        if column in frame.columns and column not in adata.obs.columns:
            adata.obs[column] = cell_ids.map(indexed[column]).to_numpy()


def _patch_rows_by_region(patch_table: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if patch_table.empty:
        return {}
    id_column = "contour_id" if "contour_id" in patch_table.columns else "region_id" if "region_id" in patch_table.columns else None
    if id_column is None:
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for _, row in patch_table.iterrows():
        value = row.get(id_column)
        if pd.isna(value):
            continue
        rows[str(value)] = {str(key): item for key, item in row.items()}
    return rows


def _structure_lookup(obs: pd.DataFrame, config: StGPTConfig) -> dict[str, str]:
    if config.data.structure_key not in obs.columns:
        return {}
    label_col = "structure_label" if "structure_label" in obs.columns else config.data.cluster_key
    if label_col not in obs.columns:
        return {}
    frame = obs[[config.data.structure_key, label_col]].dropna().drop_duplicates()
    return {_lookup_key(row[config.data.structure_key]): str(row[label_col]) for _, row in frame.iterrows()}


def _first_non_null(values) -> Any:
    for value in values:
        if pd.notna(value):
            return value
    return None


class RegionDataset(Dataset[dict[str, Any]]):
    def __init__(self, case: TrainingCase, config: StGPTConfig, *, for_inference: bool = False) -> None:
        if case.region_table.empty and case.adata.n_obs:
            case = ensure_region_training_case(case, config)
        self.case = case
        self.adata = case.adata
        self.region_table = case.region_table.reset_index(drop=True).copy()
        self.cell_membership = case.cell_membership.copy()
        self.config = config
        self.for_inference = bool(for_inference)
        self.vocab = GeneVocab.from_adata(self.adata, gene_name_key=config.data.gene_name_key)
        self.cell_matrix = _matrix(self.adata)
        self.region_matrix = case.region_expression.tocsr()
        self.neighbor_matrix = self._build_neighbor_matrix()
        coords = self.region_table[["x", "y"]].to_numpy(dtype=np.float32) if not self.region_table.empty else np.zeros((0, 2), dtype=np.float32)
        center = coords.mean(axis=0, keepdims=True) if coords.size else np.zeros((1, 2), dtype=np.float32)
        scale = coords.std(axis=0, keepdims=True) if coords.size else np.ones((1, 2), dtype=np.float32)
        scale[scale < 1e-6] = 1.0
        self.coords_norm = (coords - center) / scale
        self.structure_labels, self.structure_names = self._factorize_structures()
        self.cell_indices_by_region = self._cell_indices_by_region()
        self.binner = ExpressionBinner(config.model.n_expression_bins)
        self.rng = np.random.default_rng(config.training.seed)

    @property
    def n_structures(self) -> int:
        return max(1, len(self.structure_names))

    @property
    def region_ids(self) -> list[str]:
        return self.region_table["region_id"].astype(str).tolist() if "region_id" in self.region_table else []

    def __len__(self) -> int:
        return int(len(self.region_table))

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.region_table.iloc[index]
        region_id = str(row["region_id"])
        structure_label = int(self.structure_labels[index])
        return {
            "index": index,
            "region_id": region_id,
            "expression": self._dense_row(self.region_matrix, index),
            "neighbor_expression": self._dense_row(self.neighbor_matrix, index),
            "cell_indices": self._sample_cell_indices(region_id, index),
            "spatial": self.coords_norm[index].astype(np.float32),
            "structure_label": structure_label,
            "context_id": structure_label + 1 if self.config.data.include_structure_context else 0,
            "image_path": row.get("image_path"),
            "n_cells": int(row.get("n_cells", 0)),
        }

    def collate(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        max_genes = min(int(self.config.model.max_genes), self.adata.n_vars)
        max_cells = int(self.config.model.max_cells_per_region)
        batch_size = len(items)
        gene_ids = np.zeros((batch_size, max_genes), dtype=np.int64)
        expr_values = np.zeros((batch_size, max_genes), dtype=np.float32)
        target_values = np.zeros((batch_size, max_genes), dtype=np.float32)
        neighbor_values = np.zeros((batch_size, max_genes), dtype=np.float32)
        expr_bins = np.zeros((batch_size, max_genes), dtype=np.int64)
        cell_expr_values = np.zeros((batch_size, max_cells, max_genes), dtype=np.float32)
        cell_token_mask = np.ones((batch_size, max_cells), dtype=bool)
        mask = np.zeros((batch_size, max_genes), dtype=bool)
        padding = np.ones((batch_size, max_genes), dtype=bool)
        images = []
        spatial = []
        structure_labels = []
        context_ids = []
        region_indices = []
        region_ids: list[str] = []
        n_cells = []
        for row_idx, item in enumerate(items):
            expr = np.log1p(np.maximum(item["expression"], 0.0)).astype(np.float32)
            neigh = np.log1p(np.maximum(item["neighbor_expression"], 0.0)).astype(np.float32)
            positions = self._select_gene_positions(expr, max_genes=max_genes)
            n = len(positions)
            gene_ids[row_idx, :n] = self.vocab.ids_for_positions(positions)
            target_values[row_idx, :n] = expr[positions]
            neighbor_values[row_idx, :n] = neigh[positions]
            expr_bins[row_idx, :n] = self.binner.transform(expr[positions])
            padding[row_idx, :n] = False
            active_mask = self.rng.random(n) < float(self.config.training.mask_probability)
            if n > 0 and not active_mask.any():
                active_mask[int(self.rng.integers(0, n))] = True
            mask[row_idx, :n] = active_mask
            expr_values[row_idx, :n] = expr[positions]
            expr_values[row_idx, :n][active_mask] = 0.0
            for cell_pos, cell_index in enumerate(item["cell_indices"][:max_cells]):
                cell_expr = np.log1p(np.maximum(self._dense_row(self.cell_matrix, int(cell_index)), 0.0)).astype(np.float32)
                cell_expr_values[row_idx, cell_pos, :n] = cell_expr[positions]
                cell_token_mask[row_idx, cell_pos] = False
            images.append(
                load_image_tensor(
                    item.get("image_path"),
                    image_size=self.config.model.image_size,
                    channels=self.config.model.image_channels,
                )
            )
            spatial.append(item["spatial"])
            structure_labels.append(item["structure_label"])
            context_ids.append(item["context_id"])
            region_indices.append(item["index"])
            region_ids.append(item["region_id"])
            n_cells.append(item["n_cells"])
        return {
            "region_ids": region_ids,
            "region_indices": torch.tensor(region_indices, dtype=torch.long),
            "n_cells": torch.tensor(n_cells, dtype=torch.long),
            "gene_ids": torch.from_numpy(gene_ids),
            "expr_values": torch.from_numpy(expr_values),
            "target_values": torch.from_numpy(target_values),
            "neighbor_values": torch.from_numpy(neighbor_values),
            "expr_bins": torch.from_numpy(expr_bins),
            "cell_expr_values": torch.from_numpy(cell_expr_values),
            "cell_token_mask": torch.from_numpy(cell_token_mask),
            "mask": torch.from_numpy(mask),
            "gene_padding_mask": torch.from_numpy(padding),
            "image": torch.stack(images, dim=0),
            "spatial": torch.from_numpy(np.asarray(spatial, dtype=np.float32)),
            "structure_labels": torch.tensor(structure_labels, dtype=torch.long),
            "context_ids": torch.tensor(context_ids, dtype=torch.long),
        }

    def _select_gene_positions(self, expr: np.ndarray, *, max_genes: int) -> np.ndarray:
        nonzero = np.flatnonzero(expr > 0)
        if nonzero.size >= max_genes:
            return np.sort(self.rng.choice(nonzero, size=max_genes, replace=False))
        remaining = np.setdiff1d(np.arange(expr.shape[0]), nonzero, assume_unique=False)
        needed = max_genes - nonzero.size
        if needed > 0 and remaining.size > 0:
            fill = self.rng.choice(remaining, size=min(needed, remaining.size), replace=False)
            positions = np.concatenate([nonzero, fill])
        else:
            positions = nonzero
        return np.sort(positions.astype(np.int64))

    def _build_neighbor_matrix(self):
        if len(self.region_table) <= 1:
            return self.region_matrix.copy()
        coords = self.region_table[["x", "y"]].to_numpy(dtype=np.float32)
        k = min(int(self.config.training.neighborhood_k), len(self.region_table) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
        indices = nn.kneighbors(coords, return_distance=False)[:, 1:]
        rows = []
        for row in indices:
            rows.append(np.asarray(self.region_matrix[row].mean(axis=0)).ravel())
        return sparse.csr_matrix(np.vstack(rows).astype(np.float32))

    def _factorize_structures(self) -> tuple[np.ndarray, list[str]]:
        if "structure_label" in self.region_table.columns:
            labels, names = pd.factorize(self.region_table["structure_label"].fillna("unknown").astype(str), sort=True)
            return labels.astype(np.int64), [str(name) for name in names]
        if "structure_id" in self.region_table.columns:
            labels, names = pd.factorize(self.region_table["structure_id"].fillna("unknown").astype(str), sort=True)
            return labels.astype(np.int64), [str(name) for name in names]
        return np.zeros(len(self.region_table), dtype=np.int64), ["unknown"]

    def _cell_indices_by_region(self) -> dict[str, np.ndarray]:
        if self.cell_membership.empty:
            return {}
        cell_ids = self.adata.obs["cell_id"].astype(str) if "cell_id" in self.adata.obs.columns else pd.Series(self.adata.obs_names.astype(str), index=self.adata.obs.index)
        cell_index = pd.Series(np.arange(self.adata.n_obs, dtype=np.int64), index=cell_ids.to_numpy())
        frame = self.cell_membership.copy()
        frame["cell_index"] = frame["cell_id"].astype(str).map(cell_index)
        frame = frame[frame["cell_index"].notna()]
        return {
            str(region_id): group["cell_index"].to_numpy(dtype=np.int64)
            for region_id, group in frame.groupby("region_id", sort=True)
        }

    def _sample_cell_indices(self, region_id: str, index: int) -> np.ndarray:
        indices = self.cell_indices_by_region.get(str(region_id), np.zeros(0, dtype=np.int64))
        max_cells = int(self.config.model.max_cells_per_region)
        if len(indices) <= max_cells:
            return indices.astype(np.int64)
        rng = np.random.default_rng(int(self.config.training.seed) + int(index))
        return np.sort(rng.choice(indices, size=max_cells, replace=False).astype(np.int64))

    @staticmethod
    def _dense_row(matrix, index: int) -> np.ndarray:
        row = matrix[index]
        return np.asarray(row.toarray()).ravel().astype(np.float32) if sparse.issparse(row) else np.asarray(row).ravel().astype(np.float32)


# Compatibility name used by training/evaluation code paths.
ImageGeneDataset = RegionDataset


def _lookup_key(value: object) -> str:
    try:
        numeric = float(str(value))
    except (TypeError, ValueError):
        return str(value)
    if numeric.is_integer():
        return str(int(numeric))
    return str(value)
