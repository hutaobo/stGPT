from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import sparse
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset

from .config import StGPTConfig
from .data import (
    TrainingCase,
    _apply_case_metadata,
    _build_region_training_case,
    _configured_dataset_roots,
    _merge_sibling_cell_to_contour,
    _normalize_adata_contract,
    _prefix_training_case_ids,
    _resolve_processed_xenium_slide_root,
    _slide_corpus_item_config,
)
from .pseudo_spatial import (
    _accuracy,
    _adata_gene_names,
    _features_from_adata,
    _features_from_blocks,
    _move_batch,
    _PseudoSpatialBlock,
    _rank_bins_by_group,
    _read_table,
    _resolve_device,
    _seed_everything,
    _select_genes_from_blocks,
    _sidecar_path,
    _slide_group_key,
    _slide_groups,
    _stgpt_version,
    _write_table,
)
from .qc import make_splits

CURATED_STRUCTURE_FILENAME = "structure_assignments_v2_name.csv"
IGNORE_LABEL = "__ignore_review_needed__"
USABLE_CONFIDENCE_TIERS = {"high", "medium", "curated"}


class CuratedSpatialPriorOutput(NamedTuple):
    parent_logits: Tensor
    structure_logits: Tensor
    x_bin_logits: Tensor
    y_bin_logits: Tensor
    embedding: Tensor


class CuratedSpatialPrior(nn.Module):
    """Expression-to-curated-structure prior over parent/fine labels and spatial bins."""

    def __init__(
        self,
        *,
        n_features: int,
        n_parents: int,
        n_structures: int,
        n_x_bins: int,
        n_y_bins: int,
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
        self.parent_head = nn.Linear(int(d_model), max(1, int(n_parents)))
        self.structure_head = nn.Linear(int(d_model), max(1, int(n_structures)))
        self.x_bin_head = nn.Linear(int(d_model), max(1, int(n_x_bins)))
        self.y_bin_head = nn.Linear(int(d_model), max(1, int(n_y_bins)))

    def forward(self, features: Tensor) -> CuratedSpatialPriorOutput:
        embedding = self.encoder(features.float())
        return CuratedSpatialPriorOutput(
            parent_logits=self.parent_head(embedding),
            structure_logits=self.structure_head(embedding),
            x_bin_logits=self.x_bin_head(embedding),
            y_bin_logits=self.y_bin_head(embedding),
            embedding=embedding,
        )


class _CuratedSpatialDataset(Dataset[dict[str, Tensor]]):
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
                    "parent": torch.tensor(int(row["parent_id"]), dtype=torch.long),
                    "structure": torch.tensor(int(row["structure_id"]), dtype=torch.long),
                    "x_bin": torch.tensor(int(row["x_bin"]), dtype=torch.long),
                    "y_bin": torch.tensor(int(row["y_bin"]), dtype=torch.long),
                }
            )
        return item


class _CuratedTrainingData(NamedTuple):
    features_raw: np.ndarray
    target_frame: pd.DataFrame
    target_meta: dict[str, Any]
    splits: pd.DataFrame
    selected_gene_indices: np.ndarray
    selected_genes: list[str]
    n_regions: int


def audit_curated_structures(
    manifest: str | Path,
    *,
    output: str | Path,
    case_column: str = "case_leaf",
    root_base: str | Path | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest).expanduser()
    frame = pd.read_csv(manifest_path)
    if case_column not in frame.columns:
        raise ValueError(f"Manifest is missing case column {case_column!r}: {manifest_path}")
    base = Path(root_base).expanduser() if root_base is not None else manifest_path.parent
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        raw_case = str(row[case_column]).strip()
        case_root = Path(raw_case)
        if not case_root.is_absolute():
            case_root = base / case_root
        rows.append(_curated_case_inventory(case_root, manifest_row=row.to_dict()))

    inventory = pd.DataFrame(rows)
    out = Path(output).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "curated_structure_inventory.csv"
    json_path = out / "curated_structure_inventory.json"
    inventory.to_csv(csv_path, index=False)
    summary = _curated_inventory_summary(inventory)
    json_path.write_text(
        json.dumps({"summary": summary, "records": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        **summary,
        "inventory_csv": str(csv_path),
        "inventory_json": str(json_path),
    }


def train_curated_spatial_prior(
    config: StGPTConfig | str | Path,
    *,
    output_dir: str | Path,
    preset: str | None = None,
    max_steps: int = 2000,
    n_spatial_bins: int = 32,
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
    cfg = StGPTConfig.from_file(config, preset=preset) if isinstance(config, (str, Path)) else config.apply_preset(preset)
    seed_value = int(cfg.training.seed if seed is None else seed)
    _seed_everything(seed_value)
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    training_data = _build_curated_training_data(
        cfg,
        max_genes=max_genes,
        n_spatial_bins=n_spatial_bins,
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

    dataset = _CuratedSpatialDataset(features, target_frame)
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
    model = CuratedSpatialPrior(
        n_features=len(selected_indices),
        n_parents=len(target_meta["parent_names"]),
        n_structures=len(target_meta["structure_names"]),
        n_x_bins=int(n_spatial_bins),
        n_y_bins=int(n_spatial_bins),
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
            losses = _curated_losses(output, batch)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(train_model.parameters(), max_norm=1.0)
            optimizer.step()
            step += 1
            if step == 1 or step == int(max_steps) or step % max(1, min(100, int(max_steps))) == 0:
                row = {key: float(value.detach().cpu()) for key, value in losses.items()}
                row["step"] = float(step)
                if val_loader is not None:
                    row.update(_evaluate_curated(train_model, val_loader, target_device))
                    if row.get("val_loss", float("inf")) < best_loss:
                        best_loss = float(row["val_loss"])
                        _save_curated_checkpoint(
                            best_checkpoint,
                            model=model,
                            optimizer=optimizer,
                            cfg=cfg,
                            model_config=_curated_model_config(
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
        _save_curated_checkpoint(
            best_checkpoint,
            model=model,
            optimizer=optimizer,
            cfg=cfg,
            model_config=_curated_model_config(
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
    _save_curated_checkpoint(
        last_checkpoint,
        model=model,
        optimizer=optimizer,
        cfg=cfg,
        model_config=_curated_model_config(
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
    reference_path = _write_curated_reference_regions(target_frame, out / "reference_regions.parquet")
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
        "n_parents": int(len(target_meta["parent_names"])),
        "n_structures": int(len(target_meta["structure_names"])),
        "n_spatial_bins": int(n_spatial_bins),
    }


def predict_curated_spatial_prior(
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
    model = CuratedSpatialPrior(**model_config)
    model.load_state_dict(payload["model_state"])
    model.to(target_device)
    model.eval()

    adata = ad.read_h5ad(input_h5ad)
    features, missing_genes = _features_from_adata(adata, payload)
    dataset = _CuratedSpatialDataset(features)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)

    parent_probs: list[np.ndarray] = []
    structure_probs: list[np.ndarray] = []
    x_probs: list[np.ndarray] = []
    y_probs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, target_device)
            output_payload = model(batch["features"])
            parent_probs.append(torch.softmax(output_payload.parent_logits, dim=1).cpu().numpy())
            structure_probs.append(torch.softmax(output_payload.structure_logits, dim=1).cpu().numpy())
            x_probs.append(torch.softmax(output_payload.x_bin_logits, dim=1).cpu().numpy())
            y_probs.append(torch.softmax(output_payload.y_bin_logits, dim=1).cpu().numpy())

    p_prob = np.vstack(parent_probs).astype(np.float32)
    s_prob = np.vstack(structure_probs).astype(np.float32)
    x_prob = np.vstack(x_probs).astype(np.float32)
    y_prob = np.vstack(y_probs).astype(np.float32)
    predictions = _curated_prediction_frame(
        adata,
        payload,
        p_prob,
        s_prob,
        x_prob,
        y_prob,
        full_probabilities=full_probabilities,
    )
    projection_summary: dict[str, Any] | None = None
    if reference_regions is not None:
        reference = _prepare_curated_reference_regions(_read_table(reference_regions), payload)
        projection = project_curated_probabilities_to_reference(p_prob, s_prob, x_prob, y_prob, reference)
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
        "task": "curated_spatial_prior",
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


def build_curated_spatial_targets(
    region_table: pd.DataFrame,
    *,
    n_spatial_bins: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    regions = region_table.reset_index(drop=True).copy()
    required = {"x", "y", "standard_parent_class", "trainable_standard_name"}
    missing = required.difference(regions.columns)
    if missing:
        raise ValueError(f"Curated spatial targets require columns: {sorted(missing)}")
    mask = _usable_curated_mask(regions)
    regions = regions.loc[mask].reset_index(drop=True)
    if regions.empty:
        raise ValueError("No usable curated structure labels remain after filtering.")
    parent_values = regions["standard_parent_class"].fillna("unknown").astype(str)
    structure_values = regions["trainable_standard_name"].fillna("unknown").astype(str)
    parent_ids, parent_names = pd.factorize(parent_values, sort=True)
    structure_ids, structure_names = pd.factorize(structure_values, sort=True)
    x_bin = _rank_bins_by_group(regions["x"].to_numpy(dtype=np.float64), _slide_groups(regions), int(n_spatial_bins))
    y_bin = _rank_bins_by_group(regions["y"].to_numpy(dtype=np.float64), _slide_groups(regions), int(n_spatial_bins))
    target_frame = regions.copy()
    target_frame["parent_id"] = parent_ids.astype(np.int64)
    target_frame["parent_name"] = [parent_names[int(idx)] for idx in parent_ids]
    target_frame["structure_id"] = structure_ids.astype(np.int64)
    target_frame["structure_name"] = [structure_names[int(idx)] for idx in structure_ids]
    target_frame["x_bin"] = x_bin.astype(np.int64)
    target_frame["y_bin"] = y_bin.astype(np.int64)
    meta = {
        "parent_names": [str(item) for item in parent_names],
        "structure_names": [str(item) for item in structure_names],
        "n_spatial_bins": int(n_spatial_bins),
        "label_policy": _label_policy(),
        "slide_group_key": _slide_group_key(regions),
    }
    return target_frame, meta


def project_curated_probabilities_to_reference(
    parent_probabilities: np.ndarray,
    structure_probabilities: np.ndarray,
    x_bin_probabilities: np.ndarray,
    y_bin_probabilities: np.ndarray,
    reference_regions: pd.DataFrame,
) -> pd.DataFrame:
    required = {"region_id", "x", "y", "parent_id", "structure_id", "x_bin", "y_bin"}
    missing = required.difference(reference_regions.columns)
    if missing:
        raise ValueError(f"reference_regions is missing required columns: {sorted(missing)}")
    ref = reference_regions.dropna(subset=["parent_id", "structure_id", "x_bin", "y_bin"]).copy()
    ref["parent_id"] = ref["parent_id"].astype(int)
    ref["structure_id"] = ref["structure_id"].astype(int)
    ref["x_bin"] = ref["x_bin"].astype(int)
    ref["y_bin"] = ref["y_bin"].astype(int)
    grouped = (
        ref.sort_values(["n_cells", "region_id"], ascending=[False, True] if "n_cells" in ref.columns else [True, True])
        if "n_cells" in ref.columns
        else ref.sort_values("region_id")
    )
    grouped = grouped.drop_duplicates(["parent_id", "structure_id", "x_bin", "y_bin"], keep="first").reset_index(drop=True)
    p_idx = grouped["parent_id"].to_numpy(dtype=np.int64)
    s_idx = grouped["structure_id"].to_numpy(dtype=np.int64)
    x_idx = grouped["x_bin"].to_numpy(dtype=np.int64)
    y_idx = grouped["y_bin"].to_numpy(dtype=np.int64)
    valid = (
        (p_idx >= 0)
        & (p_idx < parent_probabilities.shape[1])
        & (s_idx >= 0)
        & (s_idx < structure_probabilities.shape[1])
        & (x_idx >= 0)
        & (x_idx < x_bin_probabilities.shape[1])
        & (y_idx >= 0)
        & (y_idx < y_bin_probabilities.shape[1])
    )
    if not bool(valid.any()):
        raise ValueError("No reference regions overlap the checkpoint curated structure vocabulary.")
    grouped = grouped.loc[valid].reset_index(drop=True)
    p_idx = p_idx[valid]
    s_idx = s_idx[valid]
    x_idx = x_idx[valid]
    y_idx = y_idx[valid]
    log_p = np.log(np.clip(parent_probabilities, 1e-12, 1.0))
    log_s = np.log(np.clip(structure_probabilities, 1e-12, 1.0))
    log_x = np.log(np.clip(x_bin_probabilities, 1e-12, 1.0))
    log_y = np.log(np.clip(y_bin_probabilities, 1e-12, 1.0))
    projected_rows: list[pd.DataFrame] = []
    chunk = 1024
    for start in range(0, parent_probabilities.shape[0], chunk):
        stop = min(start + chunk, parent_probabilities.shape[0])
        scores = log_p[start:stop, :][:, p_idx] + log_s[start:stop, :][:, s_idx] + log_x[start:stop, :][:, x_idx] + log_y[start:stop, :][:, y_idx]
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


def _build_curated_training_data(
    cfg: StGPTConfig,
    *,
    max_genes: int,
    n_spatial_bins: int,
) -> _CuratedTrainingData:
    blocks = _curated_processed_corpus_blocks(cfg)
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
    target_frame, target_meta = build_curated_spatial_targets(region_table, n_spatial_bins=n_spatial_bins)
    if len(target_frame) != features_raw.shape[0]:
        raise ValueError("Curated target filtering changed row count after feature construction.")
    return _CuratedTrainingData(
        features_raw=features_raw,
        target_frame=target_frame,
        target_meta=target_meta,
        splits=make_splits(case, cfg),
        selected_gene_indices=np.arange(len(selected_genes), dtype=np.int64),
        selected_genes=selected_genes,
        n_regions=int(features_raw.shape[0]),
    )


def _curated_processed_corpus_blocks(cfg: StGPTConfig) -> list[_PseudoSpatialBlock]:
    if cfg.data.mode != "corpus":
        raise ValueError("Curated spatial prior currently requires data.mode='corpus'.")
    roots = _configured_dataset_roots(cfg.data)
    if not roots:
        raise FileNotFoundError("data.mode='corpus' requires data.dataset_roots or data.dataset_manifest.")
    blocks: list[_PseudoSpatialBlock] = []
    skipped: list[str] = []
    for idx, root in enumerate(roots):
        resolved = _resolve_processed_xenium_slide_root(root)
        if resolved is None:
            raise FileNotFoundError(f"Processed XeniumSlide corpus root is missing xenium_slide.zarr: {root}")
        case_root, slide_store = resolved
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
        adata = _read_curated_xenium_slide_cells(slide_store, slide_cfg, source_name=slide_id, source_index=idx)
        _merge_sibling_cell_to_contour(adata, slide_cfg.data)
        case = _build_region_training_case(
            adata,
            pd.DataFrame(columns=["contour_id", "structure_id", "structure_label", "image_path"]),
            slide_cfg,
            output_dir=slide_cfg.data.output_path,
        )
        labels = _load_curated_assignments(case_root)
        region_table, keep_mask = _merge_curated_labels(case.region_table, labels)
        if region_table.empty:
            skipped.append(slide_id)
            continue
        expression = case.region_expression.tocsr()[keep_mask, :]
        filtered_case = TrainingCase(
            adata=case.adata,
            patch_table=case.patch_table,
            output_dir=case.output_dir,
            region_table=region_table.reset_index(drop=True),
            cell_membership=case.cell_membership,
            region_expression=expression.tocsr(),
        )
        prefixed = _prefix_training_case_ids(filtered_case, slide_cfg.data, source_name=slide_id)
        blocks.append(
            _PseudoSpatialBlock(
                region_table=prefixed.region_table.copy(),
                expression=prefixed.region_expression.tocsr(),
                gene_names=_adata_gene_names(prefixed.adata, slide_cfg.data.gene_name_key),
            )
        )
    if not blocks:
        detail = f" Skipped slides: {', '.join(skipped)}" if skipped else ""
        raise ValueError(f"Processed XeniumSlide corpus contains no curated trainable regions.{detail}")
    return blocks


def _read_curated_xenium_slide_cells(
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
    return adata


def _curated_case_inventory(case_root: Path, *, manifest_row: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest_row = manifest_row or {}
    slide_manifest = case_root / "slide_manifest.json"
    counts: dict[str, Any] = {}
    organ = None
    if slide_manifest.exists():
        try:
            payload = json.loads(slide_manifest.read_text(encoding="utf-8"))
            counts = payload.get("counts", {}) or {}
            organ = (payload.get("metadata", {}) or {}).get("organ")
        except json.JSONDecodeError:
            counts = {}
    path = _curated_assignment_path(case_root)
    row: dict[str, Any] = {
        "case_name": manifest_row.get("case_name"),
        "case_leaf": case_root.name,
        "case_root": str(case_root),
        "organ": organ,
        "cells": int(counts.get("cells") or 0),
        "assigned_cells": int(counts.get("assigned_cells") or 0),
        "contours": int(counts.get("contours") or 0),
        "genes": int(counts.get("genes") or 0),
        "has_curated_assignments": bool(path.exists()),
    }
    if not path.exists():
        row.update(
            {
                "regions": 0,
                "usable_structure_regions": 0,
                "usable_structure_classes": 0,
                "usable_parent_classes": 0,
                "review_needed_regions": 0,
            }
        )
        return row
    labels = _load_curated_assignments(case_root)
    usable = _usable_curated_mask(labels)
    row.update(
        {
            "regions": int(len(labels)),
            "usable_structure_regions": int(usable.sum()),
            "usable_structure_classes": int(labels.loc[usable, "trainable_standard_name"].nunique()),
            "usable_parent_classes": int(labels.loc[usable, "standard_parent_class"].nunique()),
            "review_needed_regions": int(_truthy_mask(labels.get("standard_needs_review", pd.Series(False, index=labels.index))).sum()),
            "confidence_counts": json.dumps(labels.get("standard_confidence_tier", pd.Series(dtype=object)).fillna("missing").value_counts().to_dict(), sort_keys=True),
            "top_trainable_names": json.dumps(labels.loc[usable, "trainable_standard_name"].value_counts().head(10).to_dict(), sort_keys=True),
            "top_parent_classes": json.dumps(labels.loc[usable, "standard_parent_class"].value_counts().head(10).to_dict(), sort_keys=True),
        }
    )
    return row


def _curated_inventory_summary(inventory: pd.DataFrame) -> dict[str, Any]:
    if inventory.empty:
        return {
            "n_cases": 0,
            "cases_with_curated_assignments": 0,
            "cases_with_any_usable_curated_regions": 0,
            "total_cells": 0,
            "total_assigned_cells": 0,
            "total_regions": 0,
            "total_usable_structure_regions": 0,
        }
    return {
        "n_cases": int(len(inventory)),
        "cases_with_curated_assignments": int(inventory["has_curated_assignments"].fillna(False).astype(bool).sum()),
        "cases_with_any_usable_curated_regions": int((inventory["usable_structure_regions"].fillna(0).astype(int) > 0).sum()),
        "total_cells": int(inventory["cells"].fillna(0).astype(int).sum()),
        "total_assigned_cells": int(inventory["assigned_cells"].fillna(0).astype(int).sum()),
        "total_regions": int(inventory["regions"].fillna(0).astype(int).sum()),
        "total_usable_structure_regions": int(inventory["usable_structure_regions"].fillna(0).astype(int).sum()),
    }


def _curated_assignment_path(case_root: Path) -> Path:
    return case_root / CURATED_STRUCTURE_FILENAME


def _load_curated_assignments(case_root: Path) -> pd.DataFrame:
    path = _curated_assignment_path(case_root)
    if not path.exists():
        raise FileNotFoundError(f"Curated structure sidecar is missing: {path}")
    frame = pd.read_csv(path)
    required = {"contour_id", "standard_parent_class", "trainable_standard_name", "standard_confidence_tier", "standard_needs_review"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Curated structure sidecar is missing columns {sorted(missing)}: {path}")
    frame = frame.copy()
    frame["contour_id"] = frame["contour_id"].astype(str)
    return frame


def _merge_curated_labels(region_table: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    if "contour_id" not in region_table.columns:
        raise ValueError("Region table must contain contour_id for curated label joins.")
    label_columns = [
        "contour_id",
        "standard_biological_name",
        "standard_parent_class",
        "standard_confidence_tier",
        "standard_needs_review",
        "standard_evidence_source",
        "trainable_standard_parent",
        "trainable_standard_name",
        "label_confidence",
        "label_source",
    ]
    labels = labels[[col for col in label_columns if col in labels.columns]].drop_duplicates("contour_id", keep="last")
    left = region_table.reset_index(drop=True).copy()
    left["__curated_join_key"] = left["contour_id"].astype(str).map(_unprefixed_id)
    right = labels.rename(columns={"contour_id": "__curated_join_key"}).copy()
    right["__curated_join_key"] = right["__curated_join_key"].astype(str).map(_unprefixed_id)
    merged = left.merge(right, on="__curated_join_key", how="left", sort=False).drop(columns=["__curated_join_key"])
    mask = _usable_curated_mask(merged).to_numpy(dtype=bool)
    return merged.loc[mask].reset_index(drop=True), mask


def _usable_curated_mask(frame: pd.DataFrame) -> pd.Series:
    index = frame.index
    if "trainable_standard_name" not in frame.columns:
        return pd.Series(False, index=index)
    trainable = frame["trainable_standard_name"].notna() & (frame["trainable_standard_name"].astype(str) != IGNORE_LABEL)
    if "standard_confidence_tier" in frame.columns:
        confidence = frame["standard_confidence_tier"].fillna("").astype(str).str.lower().isin(USABLE_CONFIDENCE_TIERS)
    else:
        confidence = pd.Series(True, index=index)
    if "standard_needs_review" in frame.columns:
        no_review = ~_truthy_mask(frame["standard_needs_review"])
    else:
        no_review = pd.Series(True, index=index)
    return trainable & confidence & no_review


def _truthy_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "t", "yes", "y"})


def _unprefixed_id(value: Any) -> str:
    text = str(value)
    return text.split("::", 1)[1] if "::" in text else text


def _label_policy() -> dict[str, Any]:
    return {
        "source": CURATED_STRUCTURE_FILENAME,
        "trainable_name_column": "trainable_standard_name",
        "parent_column": "standard_parent_class",
        "excluded_trainable_name": IGNORE_LABEL,
        "usable_confidence_tiers": sorted(USABLE_CONFIDENCE_TIERS),
        "standard_needs_review": False,
    }


def _curated_losses(output: CuratedSpatialPriorOutput, batch: dict[str, Tensor]) -> dict[str, Tensor]:
    parent_loss = F.cross_entropy(output.parent_logits, batch["parent"])
    structure_loss = F.cross_entropy(output.structure_logits, batch["structure"])
    x_loss = F.cross_entropy(output.x_bin_logits, batch["x_bin"])
    y_loss = F.cross_entropy(output.y_bin_logits, batch["y_bin"])
    return {
        "loss": parent_loss + structure_loss + x_loss + y_loss,
        "parent_loss": parent_loss,
        "structure_loss": structure_loss,
        "x_bin_loss": x_loss,
        "y_bin_loss": y_loss,
    }


def _evaluate_curated(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    rows: list[dict[str, float]] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            output = model(batch["features"])
            losses = _curated_losses(output, batch)
            rows.append(
                {
                    **{f"val_{key}": float(value.detach().cpu()) for key, value in losses.items()},
                    "val_parent_acc": _accuracy(output.parent_logits, batch["parent"]),
                    "val_structure_acc": _accuracy(output.structure_logits, batch["structure"]),
                    "val_x_bin_acc": _accuracy(output.x_bin_logits, batch["x_bin"]),
                    "val_y_bin_acc": _accuracy(output.y_bin_logits, batch["y_bin"]),
                }
            )
    if was_training:
        model.train()
    if not rows:
        return {}
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def _curated_prediction_frame(
    adata: ad.AnnData,
    payload: dict[str, Any],
    parent_prob: np.ndarray,
    structure_prob: np.ndarray,
    x_prob: np.ndarray,
    y_prob: np.ndarray,
    *,
    full_probabilities: bool,
) -> pd.DataFrame:
    parent_names = [str(item) for item in payload["parent_names"]]
    structure_names = [str(item) for item in payload["structure_names"]]
    cell_ids = adata.obs["cell_id"].astype(str).to_numpy() if "cell_id" in adata.obs.columns else adata.obs_names.astype(str).to_numpy()
    p_top = parent_prob.argmax(axis=1)
    s_top = structure_prob.argmax(axis=1)
    x_top = x_prob.argmax(axis=1)
    y_top = y_prob.argmax(axis=1)
    frame = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "parent_top1": [parent_names[int(idx)] for idx in p_top],
            "parent_probability": parent_prob[np.arange(len(p_top)), p_top],
            "structure_top1": [structure_names[int(idx)] for idx in s_top],
            "structure_probability": structure_prob[np.arange(len(s_top)), s_top],
            "x_bin_top1": x_top.astype(np.int64),
            "x_bin_probability": x_prob[np.arange(len(x_top)), x_top],
            "y_bin_top1": y_top.astype(np.int64),
            "y_bin_probability": y_prob[np.arange(len(y_top)), y_top],
        }
    )
    if full_probabilities:
        probability_frame = pd.concat(
            [
                pd.DataFrame(parent_prob, columns=[f"parent_prob_{idx}" for idx in range(parent_prob.shape[1])]),
                pd.DataFrame(structure_prob, columns=[f"structure_prob_{idx}" for idx in range(structure_prob.shape[1])]),
                pd.DataFrame(x_prob, columns=[f"x_bin_prob_{idx}" for idx in range(x_prob.shape[1])]),
                pd.DataFrame(y_prob, columns=[f"y_bin_prob_{idx}" for idx in range(y_prob.shape[1])]),
            ],
            axis=1,
        )
        frame = pd.concat([frame, probability_frame], axis=1)
    return frame


def _prepare_curated_reference_regions(frame: pd.DataFrame, payload: dict[str, Any]) -> pd.DataFrame:
    ref = frame.copy()
    if "region_id" not in ref.columns:
        ref["region_id"] = ref.index.astype(str)
    if not {"x", "y"}.issubset(ref.columns):
        raise ValueError("reference_regions must contain x and y columns.")
    parent_names = [str(item) for item in payload["parent_names"]]
    structure_names = [str(item) for item in payload["structure_names"]]
    parent_lookup = {name: idx for idx, name in enumerate(parent_names)}
    structure_lookup = {name: idx for idx, name in enumerate(structure_names)}
    if "parent_id" in ref.columns and pd.api.types.is_numeric_dtype(ref["parent_id"]):
        ref["parent_id"] = ref["parent_id"].fillna(-1).astype(int)
    elif "parent_name" in ref.columns:
        ref["parent_id"] = ref["parent_name"].fillna("unknown").astype(str).map(parent_lookup).fillna(-1).astype(int)
    elif "standard_parent_class" in ref.columns:
        ref["parent_id"] = ref["standard_parent_class"].fillna("unknown").astype(str).map(parent_lookup).fillna(-1).astype(int)
    else:
        ref["parent_id"] = -1
    if "structure_id" in ref.columns and pd.api.types.is_numeric_dtype(ref["structure_id"]):
        ref["structure_id"] = ref["structure_id"].fillna(-1).astype(int)
    elif "structure_name" in ref.columns:
        ref["structure_id"] = ref["structure_name"].fillna("unknown").astype(str).map(structure_lookup).fillna(-1).astype(int)
    elif "trainable_standard_name" in ref.columns:
        ref["structure_id"] = ref["trainable_standard_name"].fillna("unknown").astype(str).map(structure_lookup).fillna(-1).astype(int)
    else:
        ref["structure_id"] = -1
    n_bins = int(payload["n_spatial_bins"])
    if "x_bin" not in ref.columns:
        ref["x_bin"] = _rank_bins_by_group(ref["x"].to_numpy(dtype=np.float64), _slide_groups(ref), n_bins)
    if "y_bin" not in ref.columns:
        ref["y_bin"] = _rank_bins_by_group(ref["y"].to_numpy(dtype=np.float64), _slide_groups(ref), n_bins)
    return ref


def _save_curated_checkpoint(
    path: Path,
    *,
    model: CuratedSpatialPrior,
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
            "parent_names": target_meta["parent_names"],
            "structure_names": target_meta["structure_names"],
            "n_spatial_bins": target_meta["n_spatial_bins"],
            "label_policy": target_meta["label_policy"],
            "gene_name_key": cfg.data.gene_name_key,
            "metrics": metrics,
            "model_version": _stgpt_version(),
            "training_unit": "region",
            "task": "curated_spatial_prior",
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


def _curated_model_config(
    n_features: int,
    target_meta: dict[str, Any],
    *,
    d_model: int,
    hidden_layers: int,
    dropout: float,
) -> dict[str, Any]:
    return {
        "n_features": int(n_features),
        "n_parents": int(len(target_meta["parent_names"])),
        "n_structures": int(len(target_meta["structure_names"])),
        "n_x_bins": int(target_meta["n_spatial_bins"]),
        "n_y_bins": int(target_meta["n_spatial_bins"]),
        "d_model": int(d_model),
        "hidden_layers": int(hidden_layers),
        "dropout": float(dropout),
    }


def _write_curated_reference_regions(frame: pd.DataFrame, path: Path) -> Path:
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
            "parent_id",
            "parent_name",
            "structure_id",
            "structure_name",
            "standard_biological_name",
            "standard_parent_class",
            "trainable_standard_parent",
            "trainable_standard_name",
            "standard_confidence_tier",
            "standard_needs_review",
            "label_confidence",
            "label_source",
            "x_bin",
            "y_bin",
        )
        if col in frame.columns
    ]
    return _write_table(frame[columns].copy(), path)
