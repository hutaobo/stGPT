from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader

from .config import StGPTConfig
from .data import ImageGeneDataset, build_training_case
from .models import ImageGeneSTGPT


def evaluate(
    *,
    checkpoint: str | Path,
    config: StGPTConfig | str | Path,
    splits: str | Path,
    output_dir: str | Path,
    batch_size: int = 32,
    device: str = "auto",
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint)
    splits_path = Path(splits)
    if not splits_path.exists():
        raise FileNotFoundError(f"splits file does not exist: {splits_path}")

    user_cfg = StGPTConfig.from_file(config) if isinstance(config, (str, Path)) else config
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_cfg = StGPTConfig.model_validate(checkpoint_payload["config"])
    eval_cfg = _merge_eval_config(checkpoint_cfg, user_cfg, batch_size=batch_size)

    case = build_training_case(eval_cfg)
    dataset = ImageGeneDataset(case, eval_cfg, for_inference=True)
    checkpoint_genes = tuple(str(item) for item in checkpoint_payload.get("vocab", {}).get("genes", []))
    if checkpoint_genes and checkpoint_genes != dataset.vocab.genes:
        raise ValueError("Evaluation data gene vocabulary does not match the checkpoint vocabulary.")

    split_frame = _load_splits(splits_path, _cell_ids(case))
    target = _resolve_device(device)
    model = _load_model(checkpoint_payload, eval_cfg, dataset).to(target)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate, num_workers=0)
    buffers = _new_prediction_buffers(split_frame["split"].unique().tolist())
    embeddings: list[np.ndarray] = []
    image_embeddings: list[np.ndarray] = []

    offset = 0
    with torch.no_grad():
        for batch in loader:
            batch_size_actual = int(batch["gene_ids"].shape[0])
            batch_splits = split_frame["split"].iloc[offset : offset + batch_size_actual].astype(str).tolist()
            batch = {key: value.to(target) for key, value in batch.items()}
            output = model(
                gene_ids=batch["gene_ids"],
                expr_values=batch["expr_values"],
                expr_bins=batch["expr_bins"],
                image=batch["image"],
                spatial=batch["spatial"],
                context_ids=batch["context_ids"],
                gene_padding_mask=batch["gene_padding_mask"],
            )
            _append_prediction_buffers(buffers, batch, output, batch_splits)
            embeddings.append(output.cell_emb.cpu().numpy())
            image_embeddings.append(output.image_emb.cpu().numpy())
            offset += batch_size_actual

    cell_emb = np.vstack(embeddings).astype(np.float32)
    image_emb = np.vstack(image_embeddings).astype(np.float32)
    prediction_summary = _prediction_summary(buffers)
    retrieval_metrics = _retrieval_metrics(cell_emb, image_emb, split_frame)
    embedding_qc = _embedding_qc(cell_emb, split_frame, case, eval_cfg)
    metrics = _metrics_payload(prediction_summary, retrieval_metrics, embedding_qc, checkpoint_path, splits_path, eval_cfg)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "evaluation_metrics.json"
    prediction_path = out / "prediction_summary.csv"
    retrieval_path = out / "retrieval_metrics.csv"
    embedding_qc_path = out / "embedding_qc.csv"

    prediction_summary.to_csv(prediction_path, index=False)
    retrieval_metrics.to_csv(retrieval_path, index=False)
    embedding_qc.to_csv(embedding_qc_path, index=False)
    metrics["artifacts"] = {
        "evaluation_metrics": str(metrics_path),
        "prediction_summary": str(prediction_path),
        "retrieval_metrics": str(retrieval_path),
        "embedding_qc": str(embedding_qc_path),
    }
    metrics = _json_safe(metrics)
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    return metrics


def _merge_eval_config(checkpoint_cfg: StGPTConfig, user_cfg: StGPTConfig, *, batch_size: int) -> StGPTConfig:
    payload = checkpoint_cfg.model_dump()
    payload["data"] = user_cfg.data.model_dump()
    payload["split"] = user_cfg.split.model_dump()
    payload["training"]["batch_size"] = int(batch_size)
    payload["training"]["num_workers"] = 0
    return StGPTConfig.model_validate(payload)


def _load_model(checkpoint_payload: dict[str, Any], config: StGPTConfig, dataset: ImageGeneDataset) -> ImageGeneSTGPT:
    model = ImageGeneSTGPT(
        n_genes=dataset.vocab.size - 1,
        n_structures=int(checkpoint_payload.get("n_structures", dataset.n_structures)),
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        dim_feedforward=config.model.dim_feedforward,
        n_expression_bins=config.model.n_expression_bins,
        image_channels=config.model.image_channels,
        dropout=config.model.dropout,
    )
    model.load_state_dict(checkpoint_payload["model_state"], strict=False)
    return model


def _load_splits(path: Path, cell_ids: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"cell_id", "split"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"splits file is missing required columns: {sorted(missing)}")
    frame = frame[["cell_id", "split"]].copy()
    frame["cell_id"] = frame["cell_id"].astype(str)
    frame["split"] = frame["split"].astype(str)
    duplicate_count = int(frame["cell_id"].duplicated().sum())
    if duplicate_count:
        raise ValueError(f"splits file contains {duplicate_count} duplicate cell_id values.")
    indexed = frame.set_index("cell_id")
    missing_cells = [cell_id for cell_id in cell_ids if cell_id not in indexed.index]
    if missing_cells:
        raise ValueError(f"splits file is missing {len(missing_cells)} cells from evaluation data.")
    return indexed.loc[cell_ids].reset_index()


def _new_prediction_buffers(splits: list[str]) -> dict[str, dict[str, list[np.ndarray]]]:
    buffers = {split: _empty_metric_lists() for split in sorted(set(splits))}
    buffers["overall"] = _empty_metric_lists()
    return buffers


def _empty_metric_lists() -> dict[str, list[np.ndarray]]:
    return {"gene_pred": [], "gene_target": [], "neighbor_pred": [], "neighbor_target": []}


def _append_prediction_buffers(buffers: dict[str, dict[str, list[np.ndarray]]], batch, output, batch_splits: list[str]) -> None:
    gene_pred = output.gene_pred.detach().cpu()
    neighbor_pred = output.neighbor_pred.detach().cpu()
    target_values = batch["target_values"].detach().cpu()
    neighbor_values = batch["neighbor_values"].detach().cpu()
    gene_mask = (batch["mask"] & ~batch["gene_padding_mask"]).detach().cpu()
    neighbor_mask = (~batch["gene_padding_mask"]).detach().cpu()
    for row_idx, split in enumerate(batch_splits):
        _append_one(buffers["overall"], gene_pred, target_values, neighbor_pred, neighbor_values, gene_mask, neighbor_mask, row_idx)
        _append_one(buffers[split], gene_pred, target_values, neighbor_pred, neighbor_values, gene_mask, neighbor_mask, row_idx)


def _append_one(buffer, gene_pred, target_values, neighbor_pred, neighbor_values, gene_mask, neighbor_mask, row_idx: int) -> None:
    active_gene = gene_mask[row_idx]
    active_neighbor = neighbor_mask[row_idx]
    buffer["gene_pred"].append(gene_pred[row_idx][active_gene].numpy())
    buffer["gene_target"].append(target_values[row_idx][active_gene].numpy())
    buffer["neighbor_pred"].append(neighbor_pred[row_idx][active_neighbor].numpy())
    buffer["neighbor_target"].append(neighbor_values[row_idx][active_neighbor].numpy())


def _prediction_summary(buffers: dict[str, dict[str, list[np.ndarray]]]) -> pd.DataFrame:
    rows = []
    for split, values in buffers.items():
        gene_pred = _concat(values["gene_pred"])
        gene_target = _concat(values["gene_target"])
        neighbor_pred = _concat(values["neighbor_pred"])
        neighbor_target = _concat(values["neighbor_target"])
        rows.append(
            {
                "split": split,
                "n_masked_gene_values": int(gene_pred.size),
                "gene_mse": _mse(gene_pred, gene_target),
                "gene_correlation": _correlation(gene_pred, gene_target),
                "n_neighbor_values": int(neighbor_pred.size),
                "neighbor_mse": _mse(neighbor_pred, neighbor_target),
                "neighbor_correlation": _correlation(neighbor_pred, neighbor_target),
            }
        )
    return pd.DataFrame(rows).sort_values("split").reset_index(drop=True)


def _retrieval_metrics(cell_emb: np.ndarray, image_emb: np.ndarray, split_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, indices in _split_indices(split_frame).items():
        if len(indices) == 0:
            continue
        sim = image_emb[indices] @ cell_emb[indices].T
        for k in (1, 5):
            effective_k = min(k, len(indices))
            rows.append(
                {
                    "split": split,
                    "k": int(k),
                    "effective_k": int(effective_k),
                    "n_cells": int(len(indices)),
                    "image_to_gene_topk": _topk_accuracy(sim, effective_k),
                    "gene_to_image_topk": _topk_accuracy(sim.T, effective_k),
                }
            )
    return pd.DataFrame(rows)


def _embedding_qc(cell_emb: np.ndarray, split_frame: pd.DataFrame, case, config: StGPTConfig) -> pd.DataFrame:
    label_columns = [col for col in (config.data.cluster_key, config.data.structure_key) if col in case.adata.obs.columns]
    rows = []
    for label_column in label_columns:
        labels = case.adata.obs[label_column].astype(str).to_numpy()
        for split, indices in _split_indices(split_frame).items():
            split_labels = labels[indices]
            rows.append(
                {
                    "split": split,
                    "label_column": label_column,
                    "n_cells": int(len(indices)),
                    "n_labels": int(pd.Series(split_labels).nunique()),
                    "silhouette": _silhouette(cell_emb[indices], split_labels),
                }
            )
    return pd.DataFrame(rows, columns=["split", "label_column", "n_cells", "n_labels", "silhouette"])


def _metrics_payload(
    prediction_summary: pd.DataFrame,
    retrieval_metrics: pd.DataFrame,
    embedding_qc: pd.DataFrame,
    checkpoint: Path,
    splits: Path,
    config: StGPTConfig,
) -> dict[str, Any]:
    overall_prediction = prediction_summary[prediction_summary["split"] == "overall"].to_dict(orient="records")
    overall_retrieval = retrieval_metrics[retrieval_metrics["split"] == "overall"].to_dict(orient="records")
    overall_embedding = embedding_qc[embedding_qc["split"] == "overall"].to_dict(orient="records")
    return {
        "case_name": config.case_name,
        "checkpoint": str(checkpoint),
        "splits": str(splits),
        "n_prediction_rows": int(len(prediction_summary)),
        "n_retrieval_rows": int(len(retrieval_metrics)),
        "n_embedding_qc_rows": int(len(embedding_qc)),
        "overall_prediction": overall_prediction[0] if overall_prediction else {},
        "overall_retrieval": overall_retrieval,
        "overall_embedding_qc": overall_embedding,
    }


def _split_indices(split_frame: pd.DataFrame) -> dict[str, np.ndarray]:
    split_values = split_frame["split"].astype(str).to_numpy()
    indices = {"overall": np.arange(len(split_frame))}
    for split in sorted(pd.unique(split_values)):
        indices[str(split)] = np.flatnonzero(split_values == split)
    return indices


def _topk_accuracy(similarity: np.ndarray, k: int) -> float:
    if similarity.size == 0:
        return float("nan")
    topk = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    labels = np.arange(similarity.shape[0])[:, None]
    return float((topk == labels).any(axis=1).mean())


def _silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float:
    n_labels = int(pd.Series(labels).nunique())
    if embeddings.shape[0] < 3 or n_labels < 2 or n_labels >= embeddings.shape[0]:
        return float("nan")
    try:
        return float(silhouette_score(embeddings, labels, metric="euclidean"))
    except ValueError:
        return float("nan")


def _concat(values: list[np.ndarray]) -> np.ndarray:
    non_empty = [value for value in values if value.size]
    return np.concatenate(non_empty).astype(np.float32) if non_empty else np.zeros(0, dtype=np.float32)


def _mse(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.size == 0:
        return float("nan")
    return float(np.mean((pred - target) ** 2))


def _correlation(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.size < 2 or float(np.std(pred)) <= 1e-12 or float(np.std(target)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(pred, target)[0, 1])


def _cell_ids(case) -> list[str]:
    if "cell_id" in case.adata.obs.columns:
        return case.adata.obs["cell_id"].astype(str).tolist()
    return case.adata.obs_names.astype(str).tolist()


def _resolve_device(name: str) -> torch.device:
    normalized = str(name).lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(normalized)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value
