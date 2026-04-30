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
from .data import RegionDataset, build_training_case
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
    dataset = RegionDataset(case, eval_cfg, for_inference=True)
    checkpoint_genes = tuple(str(item) for item in checkpoint_payload.get("vocab", {}).get("genes", []))
    if checkpoint_genes and checkpoint_genes != dataset.vocab.genes:
        raise ValueError("Evaluation data gene vocabulary does not match the checkpoint vocabulary.")

    split_frame = _load_splits(splits_path, dataset.region_ids)
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
            _append_prediction_buffers(buffers, batch, output, batch_splits)
            embeddings.append(output.region_emb.cpu().numpy())
            image_embeddings.append(output.image_emb.cpu().numpy())
            offset += batch_size_actual

    region_emb = np.vstack(embeddings).astype(np.float32)
    image_emb = np.vstack(image_embeddings).astype(np.float32)
    prediction_summary = _prediction_summary(buffers)
    retrieval_metrics = _retrieval_metrics(region_emb, image_emb, split_frame)
    embedding_qc = _embedding_qc(region_emb, split_frame, case, eval_cfg)
    label_retrieval = _label_retrieval_metrics(region_emb, split_frame, case, eval_cfg)
    batch_mixing = _batch_mixing_metrics(region_emb, split_frame, case)
    failure_analysis = _failure_analysis(case, split_frame, eval_cfg)
    metrics = _metrics_payload(
        prediction_summary,
        retrieval_metrics,
        embedding_qc,
        label_retrieval,
        batch_mixing,
        failure_analysis,
        checkpoint_path,
        splits_path,
        eval_cfg,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "evaluation_metrics.json"
    prediction_path = out / "prediction_summary.csv"
    retrieval_path = out / "retrieval_metrics.csv"
    embedding_qc_path = out / "embedding_qc.csv"
    label_retrieval_path = out / "label_retrieval_metrics.csv"
    batch_mixing_path = out / "batch_mixing_metrics.csv"
    failure_analysis_path = out / "failure_analysis.csv"

    prediction_summary.to_csv(prediction_path, index=False)
    retrieval_metrics.to_csv(retrieval_path, index=False)
    embedding_qc.to_csv(embedding_qc_path, index=False)
    label_retrieval.to_csv(label_retrieval_path, index=False)
    batch_mixing.to_csv(batch_mixing_path, index=False)
    failure_analysis.to_csv(failure_analysis_path, index=False)
    metrics["artifacts"] = {
        "evaluation_metrics": str(metrics_path),
        "prediction_summary": str(prediction_path),
        "retrieval_metrics": str(retrieval_path),
        "embedding_qc": str(embedding_qc_path),
        "label_retrieval_metrics": str(label_retrieval_path),
        "batch_mixing_metrics": str(batch_mixing_path),
        "failure_analysis": str(failure_analysis_path),
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


def _load_model(checkpoint_payload: dict[str, Any], config: StGPTConfig, dataset: RegionDataset) -> ImageGeneSTGPT:
    model = ImageGeneSTGPT(
        n_genes=dataset.vocab.size - 1,
        n_structures=int(checkpoint_payload.get("n_structures", dataset.n_structures)),
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        dim_feedforward=config.model.dim_feedforward,
        n_expression_bins=config.model.n_expression_bins,
        image_channels=config.model.image_channels,
        patch_scales=config.model.patch_scales,
        use_expression_values=config.model.use_expression_values,
        use_image_context=config.model.use_image_context,
        use_spatial_context=config.model.use_spatial_context,
        use_structure_context=config.model.use_structure_context and config.data.include_structure_context,
        use_cell_context=config.model.use_cell_context,
        dropout=config.model.dropout,
    )
    model.load_state_dict(checkpoint_payload["model_state"], strict=False)
    return model


def _load_splits(path: Path, region_ids: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "region_id" not in frame.columns and "cell_id" in frame.columns:
        frame = frame.rename(columns={"cell_id": "region_id"})
    required = {"region_id", "split"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"splits file is missing required columns: {sorted(missing)}")
    frame = frame[["region_id", "split"]].copy()
    frame["region_id"] = frame["region_id"].astype(str)
    frame["split"] = frame["split"].astype(str)
    duplicate_count = int(frame["region_id"].duplicated().sum())
    if duplicate_count:
        raise ValueError(f"splits file contains {duplicate_count} duplicate region_id values.")
    indexed = frame.set_index("region_id")
    missing_regions = [region_id for region_id in region_ids if region_id not in indexed.index]
    if missing_regions:
        raise ValueError(f"splits file is missing {len(missing_regions)} regions from evaluation data.")
    return indexed.loc[region_ids].reset_index()


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


def _retrieval_metrics(region_emb: np.ndarray, image_emb: np.ndarray, split_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, indices in _split_indices(split_frame).items():
        if len(indices) == 0:
            continue
        sim = image_emb[indices] @ region_emb[indices].T
        for k in (1, 5):
            effective_k = min(k, len(indices))
            rows.append(
                {
                    "split": split,
                    "k": int(k),
                    "effective_k": int(effective_k),
                    "n_regions": int(len(indices)),
                    "image_to_gene_topk": _topk_accuracy(sim, effective_k),
                    "gene_to_image_topk": _topk_accuracy(sim.T, effective_k),
                }
            )
    return pd.DataFrame(rows)


def _embedding_qc(region_emb: np.ndarray, split_frame: pd.DataFrame, case, config: StGPTConfig) -> pd.DataFrame:
    label_columns = _meaningful_label_columns(case, config)
    rows = []
    for label_column in label_columns:
        labels = case.region_table[label_column].astype(str).to_numpy()
        for split, indices in _split_indices(split_frame).items():
            split_labels = labels[indices]
            rows.append(
                {
                    "split": split,
                    "label_column": label_column,
                    "n_regions": int(len(indices)),
                    "n_labels": int(pd.Series(split_labels).nunique()),
                    "silhouette": _silhouette(region_emb[indices], split_labels),
                }
            )
    return pd.DataFrame(rows, columns=["split", "label_column", "n_regions", "n_labels", "silhouette"])


def _label_retrieval_metrics(region_emb: np.ndarray, split_frame: pd.DataFrame, case, config: StGPTConfig) -> pd.DataFrame:
    label_columns = _meaningful_label_columns(case, config)
    rows = []
    for label_column in label_columns:
        labels = case.region_table[label_column].astype(str).to_numpy()
        for split, indices in _split_indices(split_frame).items():
            split_labels = labels[indices]
            for k in (1, 5):
                rows.append(
                    {
                        "split": split,
                        "label_column": label_column,
                        "k": int(k),
                        "effective_k": int(min(k, max(0, len(indices) - 1))),
                        "n_regions": int(len(indices)),
                        "same_label_recall": _same_label_recall(region_emb[indices], split_labels, k),
                    }
                )
    return pd.DataFrame(rows, columns=["split", "label_column", "k", "effective_k", "n_regions", "same_label_recall"])


def _meaningful_label_columns(case, config: StGPTConfig) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for column in ("structure_label", "structure_id", config.data.cluster_key):
        if column in seen or column not in case.region_table.columns:
            continue
        seen.add(column)
        labels = case.region_table[column].astype(str).replace({"nan": "", "None": "", "unknown": ""})
        if labels[labels != ""].nunique() > 1:
            columns.append(column)
    return columns


def _batch_mixing_metrics(region_emb: np.ndarray, split_frame: pd.DataFrame, case) -> pd.DataFrame:
    rows = []
    for label_column in ("batch", "batch_id", "case_id", "slide_id", "patient_id", "stain", "scanner", "platform"):
        if label_column not in case.region_table.columns:
            continue
        labels = case.region_table[label_column].astype(str).to_numpy()
        for split, indices in _split_indices(split_frame).items():
            for k in (5,):
                rows.append(
                    {
                        "split": split,
                        "batch_column": label_column,
                        "k": int(k),
                        "effective_k": int(min(k, max(0, len(indices) - 1))),
                        "n_regions": int(len(indices)),
                        "batch_mixing_entropy": _neighbor_entropy(region_emb[indices], labels[indices], k),
                    }
                )
    return pd.DataFrame(rows, columns=["split", "batch_column", "k", "effective_k", "n_regions", "batch_mixing_entropy"])


def _metrics_payload(
    prediction_summary: pd.DataFrame,
    retrieval_metrics: pd.DataFrame,
    embedding_qc: pd.DataFrame,
    label_retrieval: pd.DataFrame,
    batch_mixing: pd.DataFrame,
    failure_analysis: pd.DataFrame,
    checkpoint: Path,
    splits: Path,
    config: StGPTConfig,
) -> dict[str, Any]:
    overall_prediction = prediction_summary[prediction_summary["split"] == "overall"].to_dict(orient="records")
    overall_retrieval = retrieval_metrics[retrieval_metrics["split"] == "overall"].to_dict(orient="records")
    overall_embedding = embedding_qc[embedding_qc["split"] == "overall"].to_dict(orient="records")
    overall_label_retrieval = label_retrieval[label_retrieval["split"] == "overall"].to_dict(orient="records")
    overall_batch_mixing = batch_mixing[batch_mixing["split"] == "overall"].to_dict(orient="records")
    return {
        "case_name": config.case_name,
        "checkpoint": str(checkpoint),
        "splits": str(splits),
        "training_unit": "region",
        "n_prediction_rows": int(len(prediction_summary)),
        "n_retrieval_rows": int(len(retrieval_metrics)),
        "n_embedding_qc_rows": int(len(embedding_qc)),
        "n_label_retrieval_rows": int(len(label_retrieval)),
        "n_batch_mixing_rows": int(len(batch_mixing)),
        "n_failure_analysis_rows": int(len(failure_analysis)),
        "ablation_mode": config.training.ablation_mode,
        "model_modalities": {
            "use_expression_values": bool(config.model.use_expression_values),
            "use_image_context": bool(config.model.use_image_context),
            "use_spatial_context": bool(config.model.use_spatial_context),
            "use_structure_context": bool(config.model.use_structure_context and config.data.include_structure_context),
            "use_cell_context": bool(config.model.use_cell_context),
            "patch_scales": list(config.model.patch_scales),
            "max_cells_per_region": int(config.model.max_cells_per_region),
        },
        "overall_prediction": overall_prediction[0] if overall_prediction else {},
        "overall_retrieval": overall_retrieval,
        "overall_embedding_qc": overall_embedding,
        "overall_label_retrieval": overall_label_retrieval,
        "overall_batch_mixing": overall_batch_mixing,
    }


def _split_indices(split_frame: pd.DataFrame) -> dict[str, np.ndarray]:
    split_values = split_frame["split"].astype(str).to_numpy()
    indices = {"overall": np.arange(len(split_frame))}
    for split in sorted(pd.unique(split_values)):
        indices[str(split)] = np.flatnonzero(split_values == split)
    return indices


def _failure_analysis(case, split_frame: pd.DataFrame, config: StGPTConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    split_indices = _split_indices(split_frame)
    for split, indices in split_indices.items():
        rows.append(
            {
                "split": split,
                "category": "split",
                "metric": "n_regions",
                "value": float(len(indices)),
                "detail": config.split.strategy,
            }
        )

    patch_table = case.patch_table.copy()
    region_ids = case.region_table["region_id"].astype(str).tolist() if "region_id" in case.region_table else []
    patch_region_ids = patch_table["contour_id"].dropna().astype(str) if "contour_id" in patch_table else pd.Series(dtype=str)
    region_image_coverage = float(patch_region_ids[patch_region_ids.isin(region_ids)].nunique() / max(1, len(region_ids)))
    rows.append(_failure_row("overall", "patch", "patch_rows", float(len(patch_table)), "Rows in patch manifest."))
    rows.append(_failure_row("overall", "patch", "region_image_coverage", region_image_coverage, "Region-level patch coverage."))
    rows.append(_failure_row("overall", "patch", "missing_image_count", float(_missing_image_count(patch_table)), "Patch image files missing on disk."))
    if "n_cells" in case.region_table:
        rows.append(_failure_row("overall", "region", "low_cell_region_count", float((case.region_table["n_cells"] <= 0).sum()), "Regions without member cells."))

    provenance = _patch_provenance(patch_table)
    rows.append(
        _failure_row(
            "overall",
            "registration",
            "has_patch_coordinates",
            float(provenance["has_coordinates"]),
            ",".join(provenance["coordinate_columns"]) or "no coordinate columns",
        )
    )
    rows.append(
        _failure_row(
            "overall",
            "registration",
            "has_registration_metadata",
            float(provenance["has_registration_metadata"]),
            ",".join(provenance["registration_columns"]) or "no registration columns",
        )
    )

    panel_genes = _configured_panel_genes(config)
    if panel_genes:
        data_genes = _gene_names(case, config)
        panel_set = set(panel_genes)
        data_set = set(data_genes)
        rows.append(_failure_row("overall", "panel", "panel_gene_count", float(len(panel_set)), "Configured panel size."))
        rows.append(
            _failure_row(
                "overall",
                "panel",
                "missing_from_data_count",
                float(len(panel_set.difference(data_set))),
                "Configured panel genes absent from AnnData.",
            )
        )
        rows.append(
            _failure_row(
                "overall",
                "panel",
                "outside_panel_count",
                float(len(data_set.difference(panel_set))),
                "AnnData genes absent from configured panel.",
            )
        )

    for key in ("batch", "batch_id", "case_id", "slide_id", "donor_id", "stain_id", "platform"):
        if key not in case.region_table.columns:
            continue
        values = case.region_table[key].astype(str).to_numpy()
        for split, indices in split_indices.items():
            rows.append(
                _failure_row(
                    split,
                    "domain",
                    f"{key}_unique_values",
                    float(pd.Series(values[indices]).nunique()),
                    key,
                )
            )
    return pd.DataFrame(rows, columns=["split", "category", "metric", "value", "detail"])


def _failure_row(split: str, category: str, metric: str, value: float, detail: str) -> dict[str, Any]:
    return {"split": split, "category": category, "metric": metric, "value": value, "detail": detail}


def _topk_accuracy(similarity: np.ndarray, k: int) -> float:
    if similarity.size == 0:
        return float("nan")
    topk = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    labels = np.arange(similarity.shape[0])[:, None]
    return float((topk == labels).any(axis=1).mean())


def _same_label_recall(embeddings: np.ndarray, labels: np.ndarray, k: int) -> float:
    if embeddings.shape[0] <= 1:
        return float("nan")
    effective_k = min(k, embeddings.shape[0] - 1)
    if effective_k <= 0:
        return float("nan")
    sim = embeddings @ embeddings.T
    np.fill_diagonal(sim, -np.inf)
    neighbors = np.argpartition(-sim, kth=effective_k - 1, axis=1)[:, :effective_k]
    same = labels[neighbors] == labels[:, None]
    return float(same.any(axis=1).mean())


def _neighbor_entropy(embeddings: np.ndarray, labels: np.ndarray, k: int) -> float:
    if embeddings.shape[0] <= 1:
        return float("nan")
    n_labels = int(pd.Series(labels).nunique())
    if n_labels < 2:
        return float("nan")
    effective_k = min(k, embeddings.shape[0] - 1)
    if effective_k <= 0:
        return float("nan")
    sim = embeddings @ embeddings.T
    np.fill_diagonal(sim, -np.inf)
    neighbors = np.argpartition(-sim, kth=effective_k - 1, axis=1)[:, :effective_k]
    entropies = []
    for row in neighbors:
        counts = pd.Series(labels[row]).value_counts(normalize=True).to_numpy(dtype=np.float64)
        entropy = -float(np.sum(counts * np.log(counts + 1e-12)))
        entropies.append(entropy / max(1e-12, float(np.log(n_labels))))
    return float(np.mean(entropies))


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


def _gene_names(case, config: StGPTConfig) -> list[str]:
    if config.data.gene_name_key in case.adata.var.columns:
        return case.adata.var[config.data.gene_name_key].astype(str).tolist()
    return case.adata.var_names.astype(str).tolist()


def _configured_panel_genes(config: StGPTConfig) -> list[str]:
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
    seen: set[str] = set()
    out: list[str] = []
    for gene in genes:
        normalized = str(gene).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return out


def _missing_image_count(patch_table: pd.DataFrame) -> int:
    if "image_path" not in patch_table:
        return 0
    count = 0
    for value in patch_table["image_path"].dropna():
        if not Path(str(value)).exists():
            count += 1
    return int(count)


def _patch_provenance(patch_table: pd.DataFrame) -> dict[str, Any]:
    columns = [str(col) for col in patch_table.columns]
    lowered = {col: col.lower() for col in columns}
    coordinate_cols = [
        col
        for col, lower in lowered.items()
        if lower in {"x", "y", "patch_x", "patch_y", "center_x", "center_y", "pixel_x", "pixel_y"}
        or lower.endswith(("_x", "_y"))
    ]
    transform_cols = [
        col
        for col, lower in lowered.items()
        if "registration" in lower or "transform" in lower or "affine" in lower or "matrix" in lower
    ]
    return {
        "coordinate_columns": coordinate_cols,
        "registration_columns": transform_cols,
        "has_coordinates": bool(coordinate_cols),
        "has_registration_metadata": bool(transform_cols),
    }


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
