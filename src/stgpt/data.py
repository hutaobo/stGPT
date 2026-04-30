from __future__ import annotations

import json
from dataclasses import dataclass
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
        try:
            from pyXenium.multimodal import load_rna_protein_anndata
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install stgpt[xenium] to load Xenium data through pyXenium.") from exc
        adata = load_rna_protein_anndata(str(dataset_root), read_morphology=False)
    else:  # pragma: no cover - pydantic prevents this
        raise ValueError(f"Unsupported data mode: {cfg.mode}")
    _normalize_adata_contract(adata, cfg)
    _merge_structure_assignments(adata, cfg)
    return adata


def build_training_case(config: StGPTConfig) -> TrainingCase:
    if config.data.mode == "synthetic":
        return make_synthetic_case(config.data)
    adata = load_xenium_case(config)
    patch_table = load_patch_table(config.data)
    output_dir = config.data.output_path
    output_dir.mkdir(parents=True, exist_ok=True)
    return TrainingCase(adata=adata, patch_table=patch_table, output_dir=output_dir)


def build_training_manifest(config: StGPTConfig | str | Path) -> dict[str, Any]:
    cfg = StGPTConfig.from_file(config) if isinstance(config, (str, Path)) else config
    case = build_training_case(cfg)
    out = case.output_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "case_name": cfg.case_name,
        "mode": cfg.data.mode,
        "n_cells": int(case.adata.n_obs),
        "n_genes": int(case.adata.n_vars),
        "has_spatial": cfg.data.spatial_key in case.adata.obsm,
        "patch_count": int(len(case.patch_table)),
        "output_dir": str(out),
        "patch_table": str(out / "patch_table.csv"),
    }
    case.patch_table.to_csv(out / "patch_table.csv", index=False)
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
    structure = np.arange(n_cells) % n_structures
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
            config.cluster_key: pd.Categorical([f"cluster_{sid}" for sid in structure]),
            config.structure_key: structure.astype(int),
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
    for idx, cell_id in enumerate(obs["cell_id"].astype(str)):
        image_path = write_synthetic_patch(
            patch_dir / f"{cell_id}.png",
            image_size=config.image_size,
            structure_id=int(structure[idx]),
            intensity=float(expression[idx].mean() / max(1.0, expression.max())),
            seed=config.seed + idx,
        )
        rows.append(
            {
                "cell_id": cell_id,
                "image_path": str(image_path),
                "patch_x": float(coords[idx, 0]),
                "patch_y": float(coords[idx, 1]),
                "patch_size": int(config.image_size),
                "source_image": "synthetic_he",
                "registration_transform": "identity",
                "structure_id": int(structure[idx]),
                "cluster_id": str(obs.iloc[idx][config.cluster_key]),
            }
        )
    patch_table = pd.DataFrame(rows)
    _normalize_adata_contract(adata, config)
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
        normalized.append(
            {
                "cell_id": row.get("cell_id"),
                "contour_id": row.get("contour_id"),
                "structure_id": row.get("structure_id"),
                "structure_name": row.get("structure_name"),
                "image_path": row.get("image_path") or patch_meta.get("image_path"),
                "patch_x": row.get("patch_x") or row.get("x") or patch_meta.get("patch_x") or patch_meta.get("x"),
                "patch_y": row.get("patch_y") or row.get("y") or patch_meta.get("patch_y") or patch_meta.get("y"),
                "patch_size": row.get("patch_size") or patch_meta.get("patch_size"),
                "source_image": row.get("source_image") or patch_meta.get("source_image"),
                "registration_transform": row.get("registration_transform") or row.get("transform") or patch_meta.get("registration_transform"),
            }
        )
    return pd.DataFrame(normalized)


def _coerce_data_config(config: StGPTConfig | DataConfig | str | Path) -> DataConfig:
    if isinstance(config, DataConfig):
        return config
    if isinstance(config, StGPTConfig):
        return config.data
    return StGPTConfig.from_file(config).data


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


class ImageGeneDataset(Dataset[dict[str, Any]]):
    def __init__(self, case: TrainingCase, config: StGPTConfig, *, for_inference: bool = False) -> None:
        self.adata = case.adata
        self.patch_table = case.patch_table.copy()
        self.config = config
        self.for_inference = bool(for_inference)
        self.vocab = GeneVocab.from_adata(self.adata, gene_name_key=config.data.gene_name_key)
        self.matrix = _matrix(self.adata)
        self.neighbor_matrix = self._build_neighbor_matrix()
        self.coords = np.asarray(self.adata.obsm[config.data.spatial_key], dtype=np.float32)[:, :2]
        center = self.coords.mean(axis=0, keepdims=True)
        scale = self.coords.std(axis=0, keepdims=True)
        scale[scale < 1e-6] = 1.0
        self.coords_norm = (self.coords - center) / scale
        self.structure_labels, self.structure_names = self._factorize_structures()
        self.image_by_cell, self.image_by_structure = self._build_image_maps()
        self.binner = ExpressionBinner(config.model.n_expression_bins)
        self.rng = np.random.default_rng(config.training.seed)

    @property
    def n_structures(self) -> int:
        return max(1, len(self.structure_names))

    def __len__(self) -> int:
        return int(self.adata.n_obs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        cell_id = str(self.adata.obs.iloc[index].get("cell_id", self.adata.obs_names[index]))
        structure_id = int(self.structure_labels[index])
        image_path = self.image_by_cell.get(cell_id) or self.image_by_structure.get(str(structure_id))
        return {
            "index": index,
            "cell_id": cell_id,
            "expression": self._dense_row(self.matrix, index),
            "neighbor_expression": self._dense_row(self.neighbor_matrix, index),
            "spatial": self.coords_norm[index].astype(np.float32),
            "structure_label": structure_id,
            "context_id": structure_id + 1 if self.config.data.include_structure_context else 0,
            "image_path": image_path,
        }

    def collate(self, items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_genes = min(int(self.config.model.max_genes), self.adata.n_vars)
        batch_size = len(items)
        gene_ids = np.zeros((batch_size, max_genes), dtype=np.int64)
        expr_values = np.zeros((batch_size, max_genes), dtype=np.float32)
        target_values = np.zeros((batch_size, max_genes), dtype=np.float32)
        neighbor_values = np.zeros((batch_size, max_genes), dtype=np.float32)
        expr_bins = np.zeros((batch_size, max_genes), dtype=np.int64)
        mask = np.zeros((batch_size, max_genes), dtype=bool)
        padding = np.ones((batch_size, max_genes), dtype=bool)
        images = []
        spatial = []
        structure_labels = []
        context_ids = []
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
        return {
            "gene_ids": torch.from_numpy(gene_ids),
            "expr_values": torch.from_numpy(expr_values),
            "target_values": torch.from_numpy(target_values),
            "neighbor_values": torch.from_numpy(neighbor_values),
            "expr_bins": torch.from_numpy(expr_bins),
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
        if self.adata.n_obs <= 1:
            return self.matrix.copy()
        coords = np.asarray(self.adata.obsm[self.config.data.spatial_key], dtype=np.float32)[:, :2]
        k = min(int(self.config.training.neighborhood_k), self.adata.n_obs - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
        indices = nn.kneighbors(coords, return_distance=False)[:, 1:]
        rows = []
        for row in indices:
            rows.append(np.asarray(self.matrix[row].mean(axis=0)).ravel())
        return sparse.csr_matrix(np.vstack(rows).astype(np.float32))

    def _factorize_structures(self) -> tuple[np.ndarray, list[str]]:
        key = self.config.data.structure_key
        if key in self.adata.obs.columns:
            labels, names = pd.factorize(self.adata.obs[key].astype(str), sort=True)
            return labels.astype(np.int64), [str(name) for name in names]
        return np.zeros(self.adata.n_obs, dtype=np.int64), ["unknown"]

    def _build_image_maps(self) -> tuple[dict[str, str], dict[str, str]]:
        by_cell: dict[str, str] = {}
        by_structure: dict[str, str] = {}
        for _, row in self.patch_table.iterrows():
            image_path = row.get("image_path")
            if not image_path:
                continue
            if pd.notna(row.get("cell_id")):
                by_cell[str(row.get("cell_id"))] = str(image_path)
            if pd.notna(row.get("structure_id")):
                by_structure[str(int(float(row.get("structure_id"))))] = str(image_path)
        return by_cell, by_structure

    @staticmethod
    def _dense_row(matrix, index: int) -> np.ndarray:
        row = matrix[index]
        return np.asarray(row.toarray()).ravel().astype(np.float32) if sparse.issparse(row) else np.asarray(row).ravel().astype(np.float32)
