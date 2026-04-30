from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import StGPTConfig
from .data import RegionDataset, TrainingCase, build_training_case
from .models import ImageGeneSTGPT


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
    embeddings = _embed_dataset(dataset, checkpoint_payload, cfg, batch_size=batch_size, device=device)
    return dataset.region_table.copy(), embeddings, dataset


def _embed_dataset(
    dataset: RegionDataset,
    checkpoint_payload: dict,
    cfg: StGPTConfig,
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate, num_workers=0)
    target = _resolve_device(device)
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
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(target) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
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
            )
            embeddings.append(output.region_emb.cpu().numpy())
    return np.vstack(embeddings).astype(np.float32) if embeddings else np.zeros((0, cfg.model.d_model), dtype=np.float32)


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
