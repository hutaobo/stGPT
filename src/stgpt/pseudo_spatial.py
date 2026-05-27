from __future__ import annotations

import json
import random
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, NamedTuple

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset

from .config import StGPTConfig
from .data import (
    TrainingCase,
    _apply_case_metadata,
    _build_region_training_case,
    _configured_dataset_roots,
    _merge_sibling_cell_to_contour,
    _merge_structure_assignments,
    _normalize_adata_contract,
    _prefix_training_case_ids,
    _resolve_processed_xenium_slide_root,
    _slide_corpus_item_config,
    build_training_case,
    ensure_region_training_case,
)
from .qc import make_splits
from .tokenization import GeneVocab


class PseudoSpatialPriorOutput(NamedTuple):
    structure_logits: Tensor
    x_bin_logits: Tensor
    y_bin_logits: Tensor
    niche_logits: Tensor
    embedding: Tensor


class PseudoSpatialPrior(nn.Module):
    """Expression-to-pseudo-space prior over tissue structure, bins, and niches."""

    def __init__(
        self,
        *,
        n_features: int,
        n_structures: int,
        n_x_bins: int,
        n_y_bins: int,
        n_niches: int,
        d_model: int = 256,
        hidden_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_layers = max(1, int(hidden_layers))
        blocks: list[nn.Module] = [
            nn.Linear(int(n_features), int(d_model)),
            nn.LayerNorm(int(d_model)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        ]
        for _ in range(hidden_layers - 1):
            blocks.extend(
                [
                    nn.Linear(int(d_model), int(d_model)),
                    nn.LayerNorm(int(d_model)),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                ]
            )
        self.encoder = nn.Sequential(*blocks)
        self.structure_head = nn.Linear(int(d_model), max(1, int(n_structures)))
        self.x_bin_head = nn.Linear(int(d_model), max(1, int(n_x_bins)))
        self.y_bin_head = nn.Linear(int(d_model), max(1, int(n_y_bins)))
        self.niche_head = nn.Linear(int(d_model), max(1, int(n_niches)))

    def forward(self, features: Tensor) -> PseudoSpatialPriorOutput:
        embedding = self.encoder(features.float())
        return PseudoSpatialPriorOutput(
            structure_logits=self.structure_head(embedding),
            x_bin_logits=self.x_bin_head(embedding),
            y_bin_logits=self.y_bin_head(embedding),
            niche_logits=self.niche_head(embedding),
            embedding=embedding,
        )


class _PseudoSpatialDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, features: np.ndarray, targets: pd.DataFrame | None = None) -> None:
        self.features = np.asarray(features, dtype=np.float32)
        self.targets = targets.reset_index(drop=True).copy() if targets is not None else None

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        item: dict[str, Tensor] = {"features": torch.from_numpy(self.features[index])}
        if self.targets is not None:
            row = self.targets.iloc[index]
            item.update(
                {
                    "structure": torch.tensor(int(row["structure_id"]), dtype=torch.long),
                    "x_bin": torch.tensor(int(row["x_bin"]), dtype=torch.long),
                    "y_bin": torch.tensor(int(row["y_bin"]), dtype=torch.long),
                    "niche": torch.tensor(int(row["niche_id"]), dtype=torch.long),
                }
            )
        return item


class _PseudoSpatialBlock(NamedTuple):
    region_table: pd.DataFrame
    expression: sparse.csr_matrix
    gene_names: list[str]


class _PseudoSpatialTrainingData(NamedTuple):
    features_raw: np.ndarray
    target_frame: pd.DataFrame
    target_meta: dict[str, Any]
    splits: pd.DataFrame
    selected_gene_indices: np.ndarray
    selected_genes: list[str]
    n_regions: int


def train_pseudo_spatial_prior(
    config: StGPTConfig | str | Path,
    *,
    output_dir: str | Path,
    preset: str | None = None,
    max_steps: int = 2000,
    n_spatial_bins: int = 32,
    n_niches: int = 32,
    max_genes: int = 512,
    d_model: int = 256,
    hidden_layers: int = 2,
    dropout: float = 0.1,
    batch_size: int = 512,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    device: str = "auto",
    num_workers: int = 0,
    seed: int | None = None,
    data_parallel: bool = True,
) -> dict[str, Any]:
    """Train the expression-only pseudo-spatial prior from region-level spatial transcriptomics."""

    cfg = StGPTConfig.from_file(config, preset=preset) if isinstance(config, (str, Path)) else config.apply_preset(preset)
    seed_value = int(cfg.training.seed if seed is None else seed)
    _seed_everything(seed_value)
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    training_data = _build_pseudo_spatial_training_data(
        cfg,
        max_genes=max_genes,
        n_spatial_bins=n_spatial_bins,
        n_niches=n_niches,
        seed=seed_value,
    )
    features_raw = training_data.features_raw
    feature_mean = features_raw.mean(axis=0, keepdims=True).astype(np.float32)
    feature_std = features_raw.std(axis=0, keepdims=True).astype(np.float32)
    feature_std[feature_std < 1e-6] = 1.0
    features = ((features_raw - feature_mean) / feature_std).astype(np.float32)

    target_frame = training_data.target_frame
    target_meta = training_data.target_meta
    splits = training_data.splits
    selected_indices = training_data.selected_gene_indices
    selected_genes = training_data.selected_genes
    split_values = splits["split"].astype(str).to_numpy() if "split" in splits.columns else np.asarray(["train"] * len(target_frame))
    train_indices = np.flatnonzero(split_values == "train").astype(int).tolist()
    val_indices = np.flatnonzero(split_values == "val").astype(int).tolist()
    if not train_indices:
        train_indices = list(range(features.shape[0]))

    dataset = _PseudoSpatialDataset(features, target_frame)
    target_device = _resolve_device(device if device != "auto" else cfg.training.device)
    loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=target_device.type == "cuda",
        drop_last=False,
    )
    val_loader = (
        DataLoader(
            Subset(dataset, val_indices),
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=min(int(num_workers), 4),
            pin_memory=target_device.type == "cuda",
            drop_last=False,
        )
        if val_indices
        else None
    )
    model = PseudoSpatialPrior(
        n_features=len(selected_indices),
        n_structures=len(target_meta["structure_names"]),
        n_x_bins=int(n_spatial_bins),
        n_y_bins=int(n_spatial_bins),
        n_niches=len(target_meta["niche_names"]),
        d_model=int(d_model),
        hidden_layers=int(hidden_layers),
        dropout=float(dropout),
    ).to(target_device)
    use_data_parallel = bool(data_parallel and target_device.type == "cuda" and torch.cuda.device_count() > 1)
    train_model: nn.Module = torch.nn.DataParallel(model) if use_data_parallel else model
    optimizer = torch.optim.AdamW(train_model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))

    metrics: list[dict[str, float]] = []
    best_loss = float("inf")
    best_checkpoint = out / "best.pt"
    step = 0
    train_model.train()
    while step < int(max_steps):
        for batch in loader:
            if step >= int(max_steps):
                break
            batch = _move_batch(batch, target_device)
            optimizer.zero_grad(set_to_none=True)
            output = train_model(batch["features"])
            losses = _pseudo_spatial_losses(output, batch)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(train_model.parameters(), max_norm=1.0)
            optimizer.step()
            step += 1
            if step == 1 or step == int(max_steps) or step % max(1, min(100, int(max_steps))) == 0:
                row = {key: float(value.detach().cpu()) for key, value in losses.items()}
                row["step"] = float(step)
                if val_loader is not None:
                    row.update(_evaluate_pseudo_spatial(train_model, val_loader, target_device))
                    if row.get("val_loss", float("inf")) < best_loss:
                        best_loss = float(row["val_loss"])
                        _save_pseudo_spatial_checkpoint(
                            best_checkpoint,
                            model=model,
                            optimizer=optimizer,
                            cfg=cfg,
                            model_config=_pseudo_model_config(
                                len(selected_indices),
                                target_meta,
                                d_model=d_model,
                                hidden_layers=hidden_layers,
                                dropout=dropout,
                            ),
                            target_meta=target_meta,
                            selected_indices=selected_indices,
                            selected_genes=selected_genes,
                            feature_mean=feature_mean,
                            feature_std=feature_std,
                            metrics=metrics + [row],
                            step=step,
                            best_loss=best_loss,
                            use_data_parallel=use_data_parallel,
                        )
                metrics.append(row)

    if not best_checkpoint.exists():
        best_loss = float(metrics[-1]["loss"]) if metrics else float("nan")
        _save_pseudo_spatial_checkpoint(
            best_checkpoint,
            model=model,
            optimizer=optimizer,
            cfg=cfg,
            model_config=_pseudo_model_config(
                len(selected_indices),
                target_meta,
                d_model=d_model,
                hidden_layers=hidden_layers,
                dropout=dropout,
            ),
            target_meta=target_meta,
            selected_indices=selected_indices,
            selected_genes=selected_genes,
            feature_mean=feature_mean,
            feature_std=feature_std,
            metrics=metrics,
            step=step,
            best_loss=best_loss,
            use_data_parallel=use_data_parallel,
        )
    last_checkpoint = out / "last.pt"
    _save_pseudo_spatial_checkpoint(
        last_checkpoint,
        model=model,
        optimizer=optimizer,
        cfg=cfg,
        model_config=_pseudo_model_config(
            len(selected_indices),
            target_meta,
            d_model=d_model,
            hidden_layers=hidden_layers,
            dropout=dropout,
        ),
        target_meta=target_meta,
        selected_indices=selected_indices,
        selected_genes=selected_genes,
        feature_mean=feature_mean,
        feature_std=feature_std,
        metrics=metrics,
        step=step,
        best_loss=best_loss,
        use_data_parallel=use_data_parallel,
    )
    metrics_json = out / "metrics.json"
    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metrics_csv = out / "metrics.csv"
    pd.DataFrame(metrics).to_csv(metrics_csv, index=False)
    splits.to_csv(out / "splits.csv", index=False)
    reference_path = _write_reference_regions(target_frame, out / "reference_regions.parquet")
    return {
        "checkpoint": str(last_checkpoint),
        "best_checkpoint": str(best_checkpoint),
        "metrics": str(metrics_json),
        "metrics_csv": str(metrics_csv),
        "reference_regions": str(reference_path),
        "steps": int(step),
        "device": str(target_device),
        "data_parallel": use_data_parallel,
        "n_regions": int(features.shape[0]),
        "n_genes": int(len(selected_indices)),
        "n_structures": int(len(target_meta["structure_names"])),
        "n_spatial_bins": int(n_spatial_bins),
        "n_niches": int(len(target_meta["niche_names"])),
    }


def predict_pseudo_spatial(
    checkpoint: str | Path,
    input_h5ad: str | Path,
    *,
    output: str | Path,
    reference_regions: str | Path | None = None,
    batch_size: int = 1024,
    device: str = "auto",
    full_probabilities: bool = True,
) -> dict[str, Any]:
    payload = torch.load(Path(checkpoint).expanduser(), map_location="cpu", weights_only=False)
    model_config = dict(payload["model_config"])
    target_device = _resolve_device(device)
    model = PseudoSpatialPrior(**model_config)
    model.load_state_dict(payload["model_state"])
    model.to(target_device)
    model.eval()

    adata = ad.read_h5ad(input_h5ad)
    features, missing_genes = _features_from_adata(adata, payload)
    dataset = _PseudoSpatialDataset(features)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)

    structure_probs: list[np.ndarray] = []
    x_probs: list[np.ndarray] = []
    y_probs: list[np.ndarray] = []
    niche_probs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, target_device)
            output_payload = model(batch["features"])
            structure_probs.append(torch.softmax(output_payload.structure_logits, dim=1).cpu().numpy())
            x_probs.append(torch.softmax(output_payload.x_bin_logits, dim=1).cpu().numpy())
            y_probs.append(torch.softmax(output_payload.y_bin_logits, dim=1).cpu().numpy())
            niche_probs.append(torch.softmax(output_payload.niche_logits, dim=1).cpu().numpy())

    s_prob = np.vstack(structure_probs).astype(np.float32)
    x_prob = np.vstack(x_probs).astype(np.float32)
    y_prob = np.vstack(y_probs).astype(np.float32)
    n_prob = np.vstack(niche_probs).astype(np.float32)
    predictions = _prediction_frame(adata, payload, s_prob, x_prob, y_prob, n_prob, full_probabilities=full_probabilities)
    projection_summary: dict[str, Any] | None = None
    if reference_regions is not None:
        reference = _prepare_reference_regions(_read_table(reference_regions), payload)
        projection = project_probabilities_to_reference(s_prob, x_prob, y_prob, n_prob, reference)
        predictions = pd.concat([predictions.reset_index(drop=True), projection.reset_index(drop=True)], axis=1)
        projection_summary = {
            "reference_regions": int(len(reference)),
            "projected_cells": int(len(projection)),
        }

    output_path = _write_table(predictions, output)
    provenance = {
        "checkpoint": str(Path(checkpoint).expanduser()),
        "input_h5ad": str(Path(input_h5ad).expanduser()),
        "output": str(output_path),
        "reference_regions": str(Path(reference_regions).expanduser()) if reference_regions is not None else None,
        "n_cells": int(adata.n_obs),
        "selected_gene_count": int(len(payload["selected_genes"])),
        "missing_selected_gene_count": int(len(missing_genes)),
        "missing_selected_genes_preview": missing_genes[:25],
        "full_probabilities": bool(full_probabilities),
        "projection": projection_summary,
    }
    sidecar = _sidecar_path(output_path)
    sidecar.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return {
        "predictions": str(output_path),
        "provenance": str(sidecar),
        "n_cells": int(adata.n_obs),
        "missing_selected_gene_count": int(len(missing_genes)),
        "projection": projection_summary,
    }


def _build_pseudo_spatial_training_data(
    cfg: StGPTConfig,
    *,
    max_genes: int,
    n_spatial_bins: int,
    n_niches: int,
    seed: int,
) -> _PseudoSpatialTrainingData:
    blocks = _processed_corpus_blocks(cfg)
    if blocks:
        return _training_data_from_blocks(
            cfg,
            blocks,
            max_genes=max_genes,
            n_spatial_bins=n_spatial_bins,
            n_niches=n_niches,
            seed=seed,
        )
    case = ensure_region_training_case(build_training_case(cfg), cfg)
    if case.region_table.empty or case.region_expression.shape[0] == 0:
        raise ValueError("No trainable region expression rows were found for pseudo-spatial prior training.")
    vocab = GeneVocab.from_adata(case.adata, gene_name_key=cfg.data.gene_name_key)
    selected_indices = _select_gene_indices(case.region_expression, max_genes=max_genes)
    selected_genes = [vocab.genes[int(idx)] for idx in selected_indices]
    target_frame, target_meta = build_pseudo_spatial_targets(
        case,
        n_spatial_bins=n_spatial_bins,
        n_niches=n_niches,
        seed=seed,
    )
    return _PseudoSpatialTrainingData(
        features_raw=_feature_matrix(case.region_expression, selected_indices),
        target_frame=target_frame,
        target_meta=target_meta,
        splits=make_splits(case, cfg),
        selected_gene_indices=selected_indices,
        selected_genes=selected_genes,
        n_regions=int(case.region_expression.shape[0]),
    )


def _processed_corpus_blocks(cfg: StGPTConfig) -> list[_PseudoSpatialBlock] | None:
    if cfg.data.mode != "corpus":
        return None
    roots = _configured_dataset_roots(cfg.data)
    if not roots:
        return None
    resolved_roots: list[tuple[Path, Path]] = []
    for root in roots:
        resolved = _resolve_processed_xenium_slide_root(root)
        if resolved is None:
            return None
        resolved_roots.append(resolved)
    blocks: list[_PseudoSpatialBlock] = []
    for idx, (case_root, slide_store) in enumerate(resolved_roots):
        slide_id = case_root.name or f"slide_{idx}"
        slide_cfg = _slide_corpus_item_config(cfg, case_root=case_root, slide_store=slide_store, slide_id=slide_id, index=idx)
        data = slide_cfg.data.model_copy(
            update={
                "patch_manifest": None,
                "contour_manifest": None,
                "contour_image_store": None,
            }
        )
        slide_cfg = slide_cfg.model_copy(update={"data": data}, deep=True)
        adata = _read_xenium_slide_cells(slide_store, slide_cfg, source_name=slide_id, source_index=idx)
        _merge_sibling_cell_to_contour(adata, slide_cfg.data)
        case = _build_region_training_case(
            adata,
            pd.DataFrame(columns=["contour_id", "structure_id", "structure_label", "image_path"]),
            slide_cfg,
            output_dir=slide_cfg.data.output_path,
        )
        case = _prefix_training_case_ids(case, slide_cfg.data, source_name=slide_id)
        if not case.region_table.empty and case.region_expression.shape[0] > 0:
            blocks.append(
                _PseudoSpatialBlock(
                    region_table=case.region_table.copy(),
                    expression=case.region_expression.tocsr(),
                    gene_names=_adata_gene_names(case.adata, slide_cfg.data.gene_name_key),
                )
            )
    if not blocks:
        raise ValueError("Processed XeniumSlide corpus contains no trainable pseudo-spatial regions.")
    return blocks


def _read_xenium_slide_cells(
    slide_store: Path,
    slide_cfg: StGPTConfig,
    *,
    source_name: str,
    source_index: int,
) -> ad.AnnData:
    cells_table = slide_store / "tables" / "cells"
    if not cells_table.exists():
        raise FileNotFoundError(f"XeniumSlide cells table is missing: {cells_table}")
    adata = ad.read_zarr(cells_table)
    _normalize_adata_contract(adata, slide_cfg.data)
    _apply_case_metadata(adata, slide_cfg.data, source_name=source_name, source_index=source_index)
    _merge_structure_assignments(adata, slide_cfg.data)
    return adata


def _training_data_from_blocks(
    cfg: StGPTConfig,
    blocks: list[_PseudoSpatialBlock],
    *,
    max_genes: int,
    n_spatial_bins: int,
    n_niches: int,
    seed: int,
) -> _PseudoSpatialTrainingData:
    selected_genes = _select_genes_from_blocks(blocks, max_genes=max_genes)
    features_raw = _features_from_blocks(blocks, selected_genes)
    region_table = pd.concat([block.region_table for block in blocks], ignore_index=True)
    case = TrainingCase(
        adata=ad.AnnData(X=sparse.csr_matrix((0, 0), dtype=np.float32)),
        patch_table=pd.DataFrame(),
        output_dir=cfg.data.output_path,
        region_table=region_table,
        region_expression=sparse.csr_matrix((len(region_table), 0), dtype=np.float32),
    )
    target_frame, target_meta = build_pseudo_spatial_targets(
        case,
        n_spatial_bins=n_spatial_bins,
        n_niches=n_niches,
        seed=seed,
    )
    return _PseudoSpatialTrainingData(
        features_raw=features_raw,
        target_frame=target_frame,
        target_meta=target_meta,
        splits=make_splits(case, cfg),
        selected_gene_indices=np.arange(len(selected_genes), dtype=np.int64),
        selected_genes=selected_genes,
        n_regions=int(features_raw.shape[0]),
    )


def _select_genes_from_blocks(blocks: list[_PseudoSpatialBlock], *, max_genes: int) -> list[str]:
    stats: dict[str, list[float]] = {}
    for block in blocks:
        matrix = block.expression.tocsr()
        sums = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64)
        squares = np.asarray(matrix.multiply(matrix).sum(axis=0)).ravel().astype(np.float64)
        n_rows = float(matrix.shape[0])
        for idx, gene in enumerate(block.gene_names):
            slot = stats.setdefault(str(gene), [0.0, 0.0, 0.0])
            slot[0] += float(sums[idx])
            slot[1] += float(squares[idx])
            slot[2] += n_rows
    scored: list[tuple[float, float, str]] = []
    for gene, (total, total_sq, count) in stats.items():
        denom = max(1.0, count)
        mean = total / denom
        variance = max(0.0, total_sq / denom - mean * mean)
        scored.append((variance, mean, gene))
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    keep = max(1, min(int(max_genes), len(scored)))
    return [gene for _, _, gene in scored[:keep]]


def _features_from_blocks(blocks: list[_PseudoSpatialBlock], selected_genes: list[str]) -> np.ndarray:
    frames: list[np.ndarray] = []
    for block in blocks:
        gene_to_index = {gene: idx for idx, gene in enumerate(block.gene_names)}
        present_pairs = [(out_idx, gene_to_index[gene]) for out_idx, gene in enumerate(selected_genes) if gene in gene_to_index]
        values = np.zeros((block.expression.shape[0], len(selected_genes)), dtype=np.float32)
        if present_pairs:
            out_indices = np.asarray([pair[0] for pair in present_pairs], dtype=np.int64)
            local_indices = np.asarray([pair[1] for pair in present_pairs], dtype=np.int64)
            values[:, out_indices] = block.expression[:, local_indices].toarray().astype(np.float32)
        frames.append(np.log1p(np.maximum(values, 0.0)).astype(np.float32))
    return np.vstack(frames).astype(np.float32) if frames else np.zeros((0, len(selected_genes)), dtype=np.float32)


def build_pseudo_spatial_targets(
    case: TrainingCase,
    *,
    n_spatial_bins: int,
    n_niches: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    regions = case.region_table.reset_index(drop=True).copy()
    if not {"x", "y"}.issubset(regions.columns):
        raise ValueError("Pseudo-spatial training requires region_table columns x and y.")
    structure_source = "structure_label" if "structure_label" in regions.columns else "structure_id"
    structure_values = regions[structure_source].fillna("unknown").astype(str)
    structure_ids, structure_names = pd.factorize(structure_values, sort=True)
    x_bin = _rank_bins_by_group(regions["x"].to_numpy(dtype=np.float64), _slide_groups(regions), int(n_spatial_bins))
    y_bin = _rank_bins_by_group(regions["y"].to_numpy(dtype=np.float64), _slide_groups(regions), int(n_spatial_bins))
    niche_features = _niche_features(regions, structure_ids, x_bin, y_bin, int(n_spatial_bins))
    n_clusters = max(1, min(int(n_niches), len(regions)))
    if n_clusters == 1:
        niche_ids = np.zeros(len(regions), dtype=np.int64)
        centers = np.zeros((1, niche_features.shape[1]), dtype=np.float32)
    else:
        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=int(seed),
            batch_size=min(8192, max(128, len(regions))),
            n_init="auto",
        )
        niche_ids = km.fit_predict(niche_features).astype(np.int64)
        centers = km.cluster_centers_.astype(np.float32)
    target_frame = regions.copy()
    target_frame["structure_id"] = structure_ids.astype(np.int64)
    target_frame["structure_name"] = [structure_names[int(idx)] for idx in structure_ids]
    target_frame["x_bin"] = x_bin.astype(np.int64)
    target_frame["y_bin"] = y_bin.astype(np.int64)
    target_frame["niche_id"] = niche_ids.astype(np.int64)
    target_frame["niche_name"] = [f"niche_{int(idx):02d}" for idx in niche_ids]
    meta = {
        "structure_names": [str(item) for item in structure_names],
        "n_spatial_bins": int(n_spatial_bins),
        "niche_names": [f"niche_{idx:02d}" for idx in range(n_clusters)],
        "niche_centers": centers.tolist(),
        "niche_feature_schema": ["x_bin_scaled", "y_bin_scaled", "structure_scaled", "log_n_cells_scaled"],
        "slide_group_key": _slide_group_key(regions),
    }
    return target_frame, meta


def project_probabilities_to_reference(
    structure_probabilities: np.ndarray,
    x_bin_probabilities: np.ndarray,
    y_bin_probabilities: np.ndarray,
    niche_probabilities: np.ndarray,
    reference_regions: pd.DataFrame,
) -> pd.DataFrame:
    required = {"region_id", "x", "y", "structure_id", "x_bin", "y_bin", "niche_id"}
    missing = required.difference(reference_regions.columns)
    if missing:
        raise ValueError(f"reference_regions is missing required columns: {sorted(missing)}")
    ref = reference_regions.dropna(subset=["structure_id", "x_bin", "y_bin", "niche_id"]).copy()
    ref["structure_id"] = ref["structure_id"].astype(int)
    ref["x_bin"] = ref["x_bin"].astype(int)
    ref["y_bin"] = ref["y_bin"].astype(int)
    ref["niche_id"] = ref["niche_id"].astype(int)
    grouped = (
        ref.sort_values(["n_cells", "region_id"], ascending=[False, True] if "n_cells" in ref.columns else [True, True])
        if "n_cells" in ref.columns
        else ref.sort_values("region_id")
    )
    grouped = grouped.drop_duplicates(["structure_id", "x_bin", "y_bin", "niche_id"], keep="first").reset_index(drop=True)
    s_idx = grouped["structure_id"].to_numpy(dtype=np.int64)
    x_idx = grouped["x_bin"].to_numpy(dtype=np.int64)
    y_idx = grouped["y_bin"].to_numpy(dtype=np.int64)
    n_idx = grouped["niche_id"].to_numpy(dtype=np.int64)
    valid = (
        (s_idx >= 0)
        & (s_idx < structure_probabilities.shape[1])
        & (x_idx >= 0)
        & (x_idx < x_bin_probabilities.shape[1])
        & (y_idx >= 0)
        & (y_idx < y_bin_probabilities.shape[1])
        & (n_idx >= 0)
        & (n_idx < niche_probabilities.shape[1])
    )
    if not bool(valid.any()):
        raise ValueError("No reference regions overlap the checkpoint pseudo-space token vocabulary.")
    grouped = grouped.loc[valid].reset_index(drop=True)
    s_idx = s_idx[valid]
    x_idx = x_idx[valid]
    y_idx = y_idx[valid]
    n_idx = n_idx[valid]
    log_s = np.log(np.clip(structure_probabilities, 1e-12, 1.0))
    log_x = np.log(np.clip(x_bin_probabilities, 1e-12, 1.0))
    log_y = np.log(np.clip(y_bin_probabilities, 1e-12, 1.0))
    log_n = np.log(np.clip(niche_probabilities, 1e-12, 1.0))
    projected_rows: list[pd.DataFrame] = []
    chunk = 1024
    for start in range(0, structure_probabilities.shape[0], chunk):
        stop = min(start + chunk, structure_probabilities.shape[0])
        scores = log_s[start:stop, :][:, s_idx] + log_x[start:stop, :][:, x_idx] + log_y[start:stop, :][:, y_idx] + log_n[start:stop, :][:, n_idx]
        best = scores.argmax(axis=1)
        chosen = grouped.iloc[best].reset_index(drop=True)
        projected_rows.append(
            pd.DataFrame(
                {
                    "projected_region_id": chosen["region_id"].astype(str).to_numpy(),
                    "projected_x": chosen["x"].to_numpy(dtype=np.float32),
                    "projected_y": chosen["y"].to_numpy(dtype=np.float32),
                    "projection_score": np.exp(scores[np.arange(stop - start), best]).astype(np.float32),
                }
            )
        )
    return pd.concat(projected_rows, ignore_index=True) if projected_rows else pd.DataFrame()


def _pseudo_spatial_losses(output: PseudoSpatialPriorOutput, batch: dict[str, Tensor]) -> dict[str, Tensor]:
    structure_loss = F.cross_entropy(output.structure_logits, batch["structure"])
    x_loss = F.cross_entropy(output.x_bin_logits, batch["x_bin"])
    y_loss = F.cross_entropy(output.y_bin_logits, batch["y_bin"])
    niche_loss = F.cross_entropy(output.niche_logits, batch["niche"])
    return {
        "loss": structure_loss + x_loss + y_loss + niche_loss,
        "structure_loss": structure_loss,
        "x_bin_loss": x_loss,
        "y_bin_loss": y_loss,
        "niche_loss": niche_loss,
    }


def _evaluate_pseudo_spatial(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    rows: list[dict[str, float]] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            output = model(batch["features"])
            losses = _pseudo_spatial_losses(output, batch)
            rows.append(
                {
                    **{f"val_{key}": float(value.detach().cpu()) for key, value in losses.items()},
                    "val_structure_acc": _accuracy(output.structure_logits, batch["structure"]),
                    "val_x_bin_acc": _accuracy(output.x_bin_logits, batch["x_bin"]),
                    "val_y_bin_acc": _accuracy(output.y_bin_logits, batch["y_bin"]),
                    "val_niche_acc": _accuracy(output.niche_logits, batch["niche"]),
                }
            )
    if was_training:
        model.train()
    if not rows:
        return {}
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def _accuracy(logits: Tensor, target: Tensor) -> float:
    if target.numel() == 0:
        return float("nan")
    return float((logits.argmax(dim=1) == target).float().mean().detach().cpu())


def _select_gene_indices(matrix: sparse.spmatrix, *, max_genes: int) -> np.ndarray:
    n_genes = int(matrix.shape[1])
    keep = min(int(max_genes), n_genes)
    if keep <= 0:
        raise ValueError("max_genes must select at least one gene.")
    mean = np.asarray(matrix.mean(axis=0)).ravel().astype(np.float64)
    mean_sq = np.asarray(matrix.power(2).mean(axis=0)).ravel().astype(np.float64) if sparse.issparse(matrix) else np.asarray(np.square(matrix).mean(axis=0)).ravel().astype(np.float64)
    score = np.maximum(mean_sq - np.square(mean), 0.0)
    if not np.isfinite(score).any() or float(score.max(initial=0.0)) <= 0.0:
        score = mean
    order = np.argsort(-score, kind="mergesort")[:keep]
    return np.sort(order.astype(np.int64))


def _feature_matrix(matrix: sparse.spmatrix, gene_indices: np.ndarray) -> np.ndarray:
    selected = matrix[:, gene_indices]
    dense = selected.toarray() if sparse.issparse(selected) else np.asarray(selected)
    return np.log1p(np.maximum(dense.astype(np.float32), 0.0))


def _features_from_adata(adata: ad.AnnData, payload: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    matrix = _adata_matrix(adata)
    names = _adata_gene_names(adata, str(payload.get("gene_name_key", "feature_name")))
    index = {name: pos for pos, name in enumerate(names)}
    selected_genes = [str(item) for item in payload["selected_genes"]]
    columns: list[np.ndarray] = []
    missing: list[str] = []
    for gene in selected_genes:
        pos = index.get(gene)
        if pos is None:
            columns.append(np.zeros((adata.n_obs, 1), dtype=np.float32))
            missing.append(gene)
        else:
            col = matrix[:, pos]
            values = col.toarray() if sparse.issparse(col) else np.asarray(col)
            columns.append(values.reshape(adata.n_obs, 1).astype(np.float32))
    features = np.log1p(np.maximum(np.hstack(columns), 0.0)).astype(np.float32)
    mean = np.asarray(payload["feature_mean"], dtype=np.float32).reshape(1, -1)
    std = np.asarray(payload["feature_std"], dtype=np.float32).reshape(1, -1)
    std[std < 1e-6] = 1.0
    return ((features - mean) / std).astype(np.float32), missing


def _prediction_frame(
    adata: ad.AnnData,
    payload: dict[str, Any],
    structure_prob: np.ndarray,
    x_prob: np.ndarray,
    y_prob: np.ndarray,
    niche_prob: np.ndarray,
    *,
    full_probabilities: bool,
) -> pd.DataFrame:
    structure_names = [str(item) for item in payload["structure_names"]]
    niche_names = [str(item) for item in payload["niche_names"]]
    cell_ids = adata.obs["cell_id"].astype(str).to_numpy() if "cell_id" in adata.obs.columns else adata.obs_names.astype(str).to_numpy()
    s_top = structure_prob.argmax(axis=1)
    x_top = x_prob.argmax(axis=1)
    y_top = y_prob.argmax(axis=1)
    n_top = niche_prob.argmax(axis=1)
    frame = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "structure_top1": [structure_names[int(idx)] for idx in s_top],
            "structure_probability": structure_prob[np.arange(len(s_top)), s_top],
            "x_bin_top1": x_top.astype(np.int64),
            "x_bin_probability": x_prob[np.arange(len(x_top)), x_top],
            "y_bin_top1": y_top.astype(np.int64),
            "y_bin_probability": y_prob[np.arange(len(y_top)), y_top],
            "niche_top1": [niche_names[int(idx)] for idx in n_top],
            "niche_probability": niche_prob[np.arange(len(n_top)), n_top],
        }
    )
    if full_probabilities:
        probability_frame = pd.concat(
            [
                pd.DataFrame(structure_prob, columns=[f"structure_prob_{idx}" for idx in range(structure_prob.shape[1])]),
                pd.DataFrame(x_prob, columns=[f"x_bin_prob_{idx}" for idx in range(x_prob.shape[1])]),
                pd.DataFrame(y_prob, columns=[f"y_bin_prob_{idx}" for idx in range(y_prob.shape[1])]),
                pd.DataFrame(niche_prob, columns=[f"niche_prob_{idx}" for idx in range(niche_prob.shape[1])]),
            ],
            axis=1,
        )
        frame = pd.concat([frame, probability_frame], axis=1)
    return frame


def _niche_features(
    regions: pd.DataFrame,
    structure_ids: np.ndarray,
    x_bin: np.ndarray,
    y_bin: np.ndarray,
    n_spatial_bins: int,
) -> np.ndarray:
    denom = max(1, int(n_spatial_bins) - 1)
    n_structures = max(1, int(np.max(structure_ids)) if len(structure_ids) else 0)
    n_cells = regions["n_cells"].to_numpy(dtype=np.float32) if "n_cells" in regions.columns else np.ones(len(regions), dtype=np.float32)
    log_cells = np.log1p(np.maximum(n_cells, 0.0))
    max_log = float(log_cells.max(initial=1.0))
    if max_log <= 0:
        max_log = 1.0
    return np.column_stack(
        [
            x_bin.astype(np.float32) / float(denom),
            y_bin.astype(np.float32) / float(denom),
            structure_ids.astype(np.float32) / float(max(1, n_structures)),
            log_cells / max_log,
        ]
    ).astype(np.float32)


def _rank_bins_by_group(values: np.ndarray, groups: np.ndarray, n_bins: int) -> np.ndarray:
    out = np.zeros(len(values), dtype=np.int64)
    for group in pd.Series(groups).drop_duplicates().tolist():
        mask = groups == group
        out[mask] = _rank_bins(values[mask], n_bins)
    return out


def _rank_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    n = len(values)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    clean = np.asarray(values, dtype=np.float64)
    clean[~np.isfinite(clean)] = np.nanmedian(clean[np.isfinite(clean)]) if np.isfinite(clean).any() else 0.0
    order = np.argsort(clean, kind="mergesort")
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(n, dtype=np.int64)
    return np.clip(np.floor(ranks * int(n_bins) / max(1, n)).astype(np.int64), 0, int(n_bins) - 1)


def _slide_group_key(regions: pd.DataFrame) -> str | None:
    for column in ("corpus_slide_id", "slide_id", "batch_id", "organ"):
        if column in regions.columns:
            return column
    return None


def _slide_groups(regions: pd.DataFrame) -> np.ndarray:
    key = _slide_group_key(regions)
    if key is None:
        return np.asarray(["all"] * len(regions), dtype=object)
    return regions[key].fillna("missing").astype(str).to_numpy(dtype=object)


def _prepare_reference_regions(frame: pd.DataFrame, payload: dict[str, Any]) -> pd.DataFrame:
    ref = frame.copy()
    if "region_id" not in ref.columns:
        ref["region_id"] = ref.index.astype(str)
    if not {"x", "y"}.issubset(ref.columns):
        raise ValueError("reference_regions must contain x and y columns.")
    structure_names = [str(item) for item in payload["structure_names"]]
    structure_lookup = {name: idx for idx, name in enumerate(structure_names)}
    if "structure_id" in ref.columns and pd.api.types.is_numeric_dtype(ref["structure_id"]):
        structure_id = ref["structure_id"].fillna(-1).astype(int).to_numpy()
    elif "structure_label" in ref.columns:
        structure_id = ref["structure_label"].fillna("unknown").astype(str).map(structure_lookup).fillna(-1).astype(int).to_numpy()
    else:
        structure_id = np.zeros(len(ref), dtype=np.int64)
    ref["structure_id"] = structure_id
    n_bins = int(payload["n_spatial_bins"])
    if "x_bin" not in ref.columns:
        ref["x_bin"] = _rank_bins_by_group(ref["x"].to_numpy(dtype=np.float64), _slide_groups(ref), n_bins)
    if "y_bin" not in ref.columns:
        ref["y_bin"] = _rank_bins_by_group(ref["y"].to_numpy(dtype=np.float64), _slide_groups(ref), n_bins)
    if "niche_id" not in ref.columns:
        centers = np.asarray(payload.get("niche_centers", [[0.0, 0.0, 0.0, 0.0]]), dtype=np.float32)
        features = _niche_features(ref, structure_id, ref["x_bin"].to_numpy(dtype=np.int64), ref["y_bin"].to_numpy(dtype=np.int64), n_bins)
        distance = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        ref["niche_id"] = distance.argmin(axis=1).astype(np.int64)
    return ref


def _save_pseudo_spatial_checkpoint(
    path: Path,
    *,
    model: PseudoSpatialPrior,
    optimizer: torch.optim.Optimizer,
    cfg: StGPTConfig,
    model_config: dict[str, Any],
    target_meta: dict[str, Any],
    selected_indices: np.ndarray,
    selected_genes: list[str],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    metrics: list[dict[str, float]],
    step: int,
    best_loss: float,
    use_data_parallel: bool,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": cfg.to_json_dict(),
            "model_config": model_config,
            "selected_gene_indices": [int(item) for item in selected_indices],
            "selected_genes": selected_genes,
            "feature_mean": feature_mean.reshape(-1).astype(float).tolist(),
            "feature_std": feature_std.reshape(-1).astype(float).tolist(),
            "structure_names": target_meta["structure_names"],
            "n_spatial_bins": target_meta["n_spatial_bins"],
            "niche_names": target_meta["niche_names"],
            "niche_centers": target_meta["niche_centers"],
            "niche_feature_schema": target_meta["niche_feature_schema"],
            "gene_name_key": cfg.data.gene_name_key,
            "metrics": metrics,
            "model_version": _stgpt_version(),
            "training_unit": "region",
            "task": "pseudo_spatial_prior",
            "training_summary": {
                "steps": int(step),
                "best_loss": float(best_loss),
                "last_metrics": metrics[-1] if metrics else {},
                "data_parallel": bool(use_data_parallel),
                "slide_group_key": target_meta.get("slide_group_key"),
            },
        },
        path,
    )


def _pseudo_model_config(
    n_features: int,
    target_meta: dict[str, Any],
    *,
    d_model: int,
    hidden_layers: int,
    dropout: float,
) -> dict[str, Any]:
    return {
        "n_features": int(n_features),
        "n_structures": int(len(target_meta["structure_names"])),
        "n_x_bins": int(target_meta["n_spatial_bins"]),
        "n_y_bins": int(target_meta["n_spatial_bins"]),
        "n_niches": int(len(target_meta["niche_names"])),
        "d_model": int(d_model),
        "hidden_layers": int(hidden_layers),
        "dropout": float(dropout),
    }


def _write_reference_regions(frame: pd.DataFrame, path: Path) -> Path:
    columns = [
        col
        for col in (
            "region_id",
            "contour_id",
            "corpus_slide_id",
            "slide_id",
            "organ",
            "x",
            "y",
            "n_cells",
            "structure_id",
            "structure_name",
            "structure_label",
            "x_bin",
            "y_bin",
            "niche_id",
            "niche_name",
        )
        if col in frame.columns
    ]
    return _write_table(frame[columns].copy(), path)


def _read_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path).expanduser()
    suffix = table_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(table_path)
    if suffix == ".csv":
        return pd.read_csv(table_path)
    if suffix == ".json":
        payload = json.loads(table_path.read_text(encoding="utf-8"))
        return pd.DataFrame(payload if isinstance(payload, list) else payload.get("records", payload))
    raise ValueError(f"Unsupported table format: {table_path}")


def _write_table(frame: pd.DataFrame, path: str | Path) -> Path:
    table_path = Path(path).expanduser()
    table_path.parent.mkdir(parents=True, exist_ok=True)
    if table_path.suffix.lower() == ".parquet":
        frame.to_parquet(table_path, index=False)
        return table_path
    if table_path.suffix.lower() == ".csv":
        frame.to_csv(table_path, index=False)
        return table_path
    if table_path.suffix:
        raise ValueError(f"Unsupported output table format: {table_path}")
    parquet_path = table_path.with_suffix(".parquet")
    frame.to_parquet(parquet_path, index=False)
    return parquet_path


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".provenance.json")


def _adata_matrix(adata: ad.AnnData):
    matrix = adata.layers["rna"] if "rna" in adata.layers else adata.X
    return matrix.tocsr() if sparse.issparse(matrix) else sparse.csr_matrix(np.asarray(matrix))


def _adata_gene_names(adata: ad.AnnData, gene_name_key: str) -> list[str]:
    if gene_name_key in adata.var.columns:
        names = adata.var[gene_name_key].astype(str).tolist()
    elif "gene_name" in adata.var.columns:
        names = adata.var["gene_name"].astype(str).tolist()
    elif "name" in adata.var.columns:
        names = adata.var["name"].astype(str).tolist()
    else:
        names = adata.var_names.astype(str).tolist()
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        count = seen.get(name, 0)
        seen[name] = count + 1
        out.append(name if count == 0 else f"{name}.{count}")
    return out


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _resolve_device(name: str) -> torch.device:
    normalized = str(name).lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(normalized)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _stgpt_version() -> str:
    try:
        return version("stgpt")
    except PackageNotFoundError:  # pragma: no cover
        return "0+unknown"
