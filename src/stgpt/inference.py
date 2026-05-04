from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import StGPTConfig
from .data import RegionDataset, TrainingCase, build_training_case
from .models import ImageGeneSTGPT


@dataclass(frozen=True)
class RegionInferenceResult:
    """Region-level inference outputs used by spatho evidence export."""

    region_table: pd.DataFrame
    embeddings: np.ndarray
    dataset: RegionDataset
    prototype_assignments: pd.DataFrame


def embed_anndata(
    adata: ad.AnnData,
    *,
    checkpoint: str | Path,
    batch_size: int = 32,
    device: str = "auto",
) -> ad.AnnData:
    checkpoint_payload = torch.load(checkpoint, map_location="cpu")
    cfg = StGPTConfig.model_validate(checkpoint_payload["config"])
    if "spatial" not in adata.obsm and cfg.data.spatial_key not in adata.obsm:
        raise ValueError("AnnData must contain spatial coordinates for stGPT embedding.")
    case = TrainingCase(adata=adata.copy(), patch_table=pd.DataFrame(), output_dir=Path("."))
    payload = cfg.model_dump()
    payload["training"]["batch_size"] = int(batch_size)
    cfg = StGPTConfig.model_validate(payload)
    dataset = RegionDataset(case, cfg, for_inference=True)
    embeddings = _embed_dataset(dataset, checkpoint_payload, cfg, batch_size=batch_size, device=device)
    out = ad.AnnData(obs=dataset.region_table.set_index("region_id", drop=False).copy())
    out.obsm["X_stGPT"] = embeddings
    return out


def embed_regions(
    config: StGPTConfig | str | Path,
    *,
    checkpoint: str | Path,
    batch_size: int = 32,
    device: str = "auto",
) -> tuple[pd.DataFrame, np.ndarray, RegionDataset]:
    result = embed_region_outputs(config, checkpoint=checkpoint, batch_size=batch_size, device=device)
    return result.region_table, result.embeddings, result.dataset


def embed_region_outputs(
    config: StGPTConfig | str | Path,
    *,
    checkpoint: str | Path,
    batch_size: int = 32,
    device: str = "auto",
) -> RegionInferenceResult:
    cfg = StGPTConfig.from_file(config) if isinstance(config, (str, Path)) else config
    checkpoint_payload = torch.load(checkpoint, map_location="cpu")
    payload = cfg.model_dump()
    payload["training"]["batch_size"] = int(batch_size)
    cfg = StGPTConfig.model_validate(payload)
    case = build_training_case(cfg)
    dataset = RegionDataset(case, cfg, for_inference=True)
    checkpoint_genes = tuple(str(item) for item in checkpoint_payload.get("vocab", {}).get("genes", []))
    if checkpoint_genes and checkpoint_genes != dataset.vocab.genes:
        raise ValueError("Embedding data gene vocabulary does not match the checkpoint vocabulary.")
    embeddings, prototype_assignments = _embed_dataset_outputs(
        dataset,
        checkpoint_payload,
        cfg,
        batch_size=batch_size,
        device=device,
    )
    return RegionInferenceResult(
        region_table=dataset.region_table.copy(),
        embeddings=embeddings,
        dataset=dataset,
        prototype_assignments=prototype_assignments,
    )


def _embed_dataset(
    dataset: RegionDataset,
    checkpoint_payload: dict,
    cfg: StGPTConfig,
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    embeddings, _ = _embed_dataset_outputs(dataset, checkpoint_payload, cfg, batch_size=batch_size, device=device)
    return embeddings


def _embed_dataset_outputs(
    dataset: RegionDataset,
    checkpoint_payload: dict,
    cfg: StGPTConfig,
    *,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate, num_workers=0)
    target = _resolve_device(device)
    checkpoint_cfg = StGPTConfig.model_validate(checkpoint_payload.get("config", cfg.model_dump()))
    model = ImageGeneSTGPT(
        n_genes=dataset.vocab.size - 1,
        n_structures=int(checkpoint_payload.get("n_structures", dataset.n_structures)),
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        dim_feedforward=cfg.model.dim_feedforward,
        n_expression_bins=cfg.model.n_expression_bins,
        image_channels=cfg.model.image_channels,
        patch_scales=cfg.model.patch_scales,
        image_encoder_backend="precomputed" if cfg.data.image_embedding_store else cfg.model.image_encoder_backend,
        image_encoder_name=cfg.model.image_encoder_name,
        image_encoder_frozen=cfg.model.image_encoder_frozen,
        image_embedding_dim=cfg.model.image_embedding_dim or dataset.image_embedding_dim or None,
        n_prototypes=checkpoint_cfg.model.n_prototypes,
        prototype_temperature=checkpoint_cfg.model.prototype_temperature,
        use_expression_values=cfg.model.use_expression_values,
        use_image_context=cfg.model.use_image_context,
        use_spatial_context=cfg.model.use_spatial_context,
        use_structure_context=cfg.model.use_structure_context and cfg.data.include_structure_context,
        use_cell_context=cfg.model.use_cell_context,
        dropout=cfg.model.dropout,
    )
    model.load_state_dict(checkpoint_payload["model_state"], strict=False)
    model.to(target)
    model.eval()
    embeddings = []
    prototype_rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            raw_batch = batch
            batch = {key: value.to(target) if isinstance(value, torch.Tensor) else value for key, value in raw_batch.items()}
            output = model(
                gene_ids=batch["gene_ids"],
                expr_values=batch["expr_values"],
                expr_bins=batch["expr_bins"],
                image=batch["image"],
                spatial=batch["spatial"],
                context_ids=batch["context_ids"],
                gene_padding_mask=batch["gene_padding_mask"],
                cell_expr_values=batch["cell_expr_values"],
                cell_token_mask=batch["cell_token_mask"],
                object_image=batch.get("object_image"),
                context_image=batch.get("context_image"),
                contour_mask=batch.get("contour_mask"),
                contour_geometry=batch.get("contour_geometry"),
                precomputed_image_embedding=batch.get("precomputed_image_embedding"),
            )
            embeddings.append(output.region_emb.cpu().numpy())
            prototype_rows.extend(_prototype_assignment_rows(raw_batch, output))
    embedding_matrix = np.vstack(embeddings).astype(np.float32) if embeddings else np.zeros((0, cfg.model.d_model), dtype=np.float32)
    return embedding_matrix, pd.DataFrame(prototype_rows)


def _prototype_assignment_rows(batch: dict[str, object], output) -> list[dict[str, object]]:
    region_ids = [str(item) for item in batch.get("region_ids", [])]
    if not region_ids:
        return []
    region_indices = _tensor_to_numpy(batch.get("region_indices"), len(region_ids), fill=-1).astype(np.int64)
    row_indices = _tensor_to_numpy(batch.get("row_index"), len(region_ids), fill=-1).astype(np.int64)
    probs = output.prototype_probs.detach().cpu().numpy().astype(np.float32) if output.prototype_probs is not None else None
    ids = output.prototype_ids.detach().cpu().numpy().astype(np.int64) if output.prototype_ids is not None else np.full(len(region_ids), -1, dtype=np.int64)
    confidence = (
        output.prototype_confidence.detach().cpu().numpy().astype(np.float32)
        if output.prototype_confidence is not None
        else np.full(len(region_ids), np.nan, dtype=np.float32)
    )
    entropy = (
        -np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0)), axis=1).astype(np.float32)
        if probs is not None
        else np.full(len(region_ids), np.nan, dtype=np.float32)
    )
    records: list[dict[str, object]] = []
    for idx, region_id in enumerate(region_ids):
        record: dict[str, object] = {
            "region_id": region_id,
            "region_row_index": int(region_indices[idx]),
            "row_index": None if row_indices[idx] < 0 else int(row_indices[idx]),
            "prototype_id": int(ids[idx]),
            "prototype_confidence": float(confidence[idx]),
            "assignment_entropy": float(entropy[idx]),
        }
        if probs is not None:
            for proto_idx, value in enumerate(probs[idx]):
                record[f"prototype_prob_{proto_idx}"] = float(value)
        records.append(record)
    return records


def _tensor_to_numpy(value: object, length: int, *, fill: int) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.full(length, fill, dtype=np.int64)


def write_embeddings_table(adata: ad.AnnData, output: str | Path) -> Path:
    if "X_stGPT" not in adata.obsm:
        raise ValueError("AnnData is missing obsm['X_stGPT'].")
    frame = pd.DataFrame(adata.obsm["X_stGPT"], index=adata.obs_names)
    id_column = "region_id" if "region_id" in adata.obs.columns else "cell_id"
    frame.insert(0, id_column, adata.obs[id_column].astype(str).to_numpy() if id_column in adata.obs else adata.obs_names.astype(str))
    for column in ("cluster", "structure_id", "structure_label", "n_cells"):
        if column in adata.obs.columns:
            frame[column] = adata.obs[column].astype(str).to_numpy()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def export_spatho_summaries(embeddings: str | Path, output: str | Path) -> dict[str, str]:
    frame = pd.read_parquet(embeddings)
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for key in ("cluster", "structure_id"):
        if key not in frame.columns:
            continue
        numeric_cols = [col for col in frame.columns if str(col).isdigit() or str(col).startswith("emb_")]
        if not numeric_cols:
            numeric_cols = [col for col in frame.columns if col not in {"cell_id", "cluster", "structure_id"}]
        summary = frame.groupby(key)[numeric_cols].mean().reset_index()
        path = out_dir / f"{key}_embedding_summary.csv"
        summary.to_csv(path, index=False)
        outputs[f"{key}_summary"] = str(path)
    copied = out_dir / "cell_embeddings.parquet"
    frame.to_parquet(copied, index=False)
    outputs["cell_embeddings"] = str(copied)
    return outputs


def _resolve_device(name: str) -> torch.device:
    normalized = str(name).lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(normalized)
