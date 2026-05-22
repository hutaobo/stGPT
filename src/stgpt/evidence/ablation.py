from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from ..config import StGPTConfig
from ..data import RegionDataset, build_training_case
from ..evaluation import _load_model, _merge_eval_config

AblationMode = Literal["baseline", "drop_object", "drop_context", "drop_shape"]
ABLATION_MODES: tuple[AblationMode, ...] = ("baseline", "drop_object", "drop_context", "drop_shape")


def run_contour_ablation(
    *,
    checkpoint: str | Path,
    config: StGPTConfig | str | Path,
    targets: str | Path,
    output_dir: str | Path,
    batch_size: int = 32,
    device: str = "auto",
) -> dict[str, Any]:
    """Run targeted contour-native ablation over existing failure-gallery targets.

    This is an inference-only E5.2 harness. It embeds the whole case once to build
    a stable gene/region reference bank, then reruns only selected target contours
    with object, context, and shape evidence ablated. The retrieval gallery remains
    the unablated baseline region embedding bank so deltas quantify how much the
    contour evidence changes image-to-gene alignment.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    checkpoint_path = Path(checkpoint).expanduser()
    target_path = Path(targets).expanduser()
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    user_cfg = StGPTConfig.from_file(config) if isinstance(config, (str, Path)) else config
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_cfg = StGPTConfig.model_validate(checkpoint_payload["config"])
    eval_cfg = _merge_eval_config(checkpoint_cfg, user_cfg, batch_size=batch_size)

    case = build_training_case(eval_cfg)
    dataset = RegionDataset(case, eval_cfg, for_inference=True)
    checkpoint_genes = tuple(str(item) for item in checkpoint_payload.get("vocab", {}).get("genes", []))
    if checkpoint_genes and checkpoint_genes != dataset.vocab.genes:
        raise ValueError("Ablation data gene vocabulary does not match the checkpoint vocabulary.")

    target_frame = _load_targets(target_path)
    target_indices = _resolve_target_indices(target_frame, dataset)
    target_frame = target_frame.assign(dataset_index=target_indices)
    target_frame = target_frame[target_frame["dataset_index"].notna()].copy()
    target_frame["dataset_index"] = target_frame["dataset_index"].astype(int)
    target_frame = target_frame.drop_duplicates(subset=["dataset_index"], keep="first").reset_index(drop=True)

    target_device = _resolve_device(device)
    model = _load_model(checkpoint_payload, eval_cfg, dataset).to(target_device)
    model.eval()

    reference = _embed_dataset(model, dataset, batch_size=batch_size, device=target_device)
    rows: list[dict[str, Any]] = []
    if not target_frame.empty:
        for mode in ABLATION_MODES:
            mode_rows = _embed_targets(
                model,
                dataset,
                target_frame,
                reference_region_emb=reference["region_emb"],
                mode=mode,
                batch_size=batch_size,
                device=target_device,
            )
            rows.extend(mode_rows)

    result_frame = pd.DataFrame(rows)
    if not result_frame.empty:
        result_frame = _attach_baseline_deltas(result_frame)
    summary_frame = _summary_frame(result_frame)
    gallery_frame = _gallery_with_ablation(target_frame, result_frame)
    status_payload = _status_payload(
        checkpoint_path=checkpoint_path,
        target_path=target_path,
        config=eval_cfg,
        n_loaded_targets=len(target_frame),
        n_reference_regions=int(reference["region_emb"].shape[0]),
        result_frame=result_frame,
    )

    result_csv = out / "ablation_results.csv"
    result_json = out / "ablation_results.json"
    summary_csv = out / "ablation_summary.csv"
    summary_json = out / "ablation_summary.json"
    gallery_csv = out / "failure_gallery_with_ablation.csv"
    gallery_json = out / "failure_gallery_with_ablation.json"
    anatomy_md = out / "anatomy_of_failure.md"

    result_frame.to_csv(result_csv, index=False)
    result_json.write_text(json.dumps(_json_safe(result_frame.to_dict(orient="records")), indent=2), encoding="utf-8")
    summary_frame.to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(_json_safe(status_payload), indent=2), encoding="utf-8")
    gallery_frame.to_csv(gallery_csv, index=False)
    gallery_json.write_text(json.dumps(_json_safe(gallery_frame.to_dict(orient="records")), indent=2), encoding="utf-8")
    anatomy_md.write_text(_anatomy_markdown(status_payload, summary_frame, gallery_frame), encoding="utf-8")

    return {
        "status": status_payload["status"],
        "n_targets": int(len(target_frame)),
        "n_reference_regions": int(reference["region_emb"].shape[0]),
        "artifacts": {
            "ablation_results_csv": str(result_csv),
            "ablation_results_json": str(result_json),
            "ablation_summary_csv": str(summary_csv),
            "ablation_summary_json": str(summary_json),
            "failure_gallery_with_ablation_csv": str(gallery_csv),
            "failure_gallery_with_ablation_json": str(gallery_json),
            "anatomy_of_failure": str(anatomy_md),
        },
    }


def _load_targets(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"ablation targets file does not exist: {path}")
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("targets", payload.get("ablation_targets", []))
        frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["embedding_row_index", "row_index", "contour_id", "evidence_id"])
    return frame


def _resolve_target_indices(targets: pd.DataFrame, dataset: RegionDataset) -> list[int | None]:
    region_table = dataset.region_table.reset_index(drop=True)
    by_region = _string_index(region_table, "region_id")
    by_contour = _string_index(region_table, "contour_id")
    by_row_index: dict[int, int] = {}
    if "row_index" in region_table.columns:
        for idx, value in enumerate(region_table["row_index"].tolist()):
            row_index = _safe_int(value)
            if row_index is not None and row_index not in by_row_index:
                by_row_index[row_index] = idx
    resolved: list[int | None] = []
    for _, row in targets.iterrows():
        idx = _safe_int(row.get("embedding_row_index"))
        if idx is not None and 0 <= idx < len(dataset):
            resolved.append(idx)
            continue
        for key, mapping in (("contour_id", by_contour), ("region_id", by_region)):
            value = _string_or_none(row.get(key))
            if value is not None and value in mapping:
                resolved.append(mapping[value])
                break
        else:
            row_index = _safe_int(row.get("row_index"))
            resolved.append(by_row_index.get(row_index) if row_index is not None else None)
    return resolved


def _string_index(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    result: dict[str, int] = {}
    for idx, value in enumerate(frame[column].tolist()):
        key = _string_or_none(value)
        if key is not None and key not in result:
            result[key] = idx
    return result


def _embed_dataset(model, dataset: RegionDataset, *, batch_size: int, device: torch.device) -> dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate, num_workers=4)
    region_emb: list[np.ndarray] = []
    image_emb: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            output = _forward_batch(model, batch)
            region_emb.append(output.region_emb.detach().cpu().numpy())
            image_emb.append(output.image_emb.detach().cpu().numpy())
    return {
        "region_emb": np.vstack(region_emb).astype(np.float32) if region_emb else np.zeros((0, 0), dtype=np.float32),
        "image_emb": np.vstack(image_emb).astype(np.float32) if image_emb else np.zeros((0, 0), dtype=np.float32),
    }


def _embed_targets(
    model,
    dataset: RegionDataset,
    target_frame: pd.DataFrame,
    *,
    reference_region_emb: np.ndarray,
    mode: AblationMode,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    indices = target_frame["dataset_index"].astype(int).tolist()
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate, num_workers=4)
    rows: list[dict[str, Any]] = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            batch_count = int(batch["gene_ids"].shape[0])
            batch = _move_batch(batch, device)
            ablated = _apply_ablation(batch, mode)
            output = _forward_batch(model, ablated)
            image_emb = output.image_emb.detach().cpu().numpy().astype(np.float32)
            for local_idx in range(batch_count):
                target_row = target_frame.iloc[offset + local_idx].to_dict()
                dataset_index = int(target_row["dataset_index"])
                score = _retrieval_score(image_emb[local_idx], reference_region_emb, dataset_index)
                rows.append(
                    {
                        "evidence_id": target_row.get("evidence_id"),
                        "contour_id": target_row.get("contour_id") or target_row.get("region_id"),
                        "slide_id": target_row.get("slide_id"),
                        "structure_label": target_row.get("structure_label"),
                        "image_source": target_row.get("image_source"),
                        "failure_rank": target_row.get("failure_rank"),
                        "failure_score": target_row.get("failure_score"),
                        "failure_reasons": target_row.get("failure_reasons"),
                        "prototype_id": target_row.get("prototype_id"),
                        "prototype_confidence": target_row.get("prototype_confidence"),
                        "assignment_entropy": target_row.get("assignment_entropy"),
                        "row_index": target_row.get("row_index"),
                        "embedding_row_index": target_row.get("embedding_row_index"),
                        "dataset_index": dataset_index,
                        "ablation_mode": mode,
                        "ablation_operation": _ablation_operation(mode),
                        **score,
                    }
                )
            offset += batch_count
    return rows


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _forward_batch(model, batch: dict[str, Any]):
    return model(
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


def _apply_ablation(batch: dict[str, Any], mode: AblationMode) -> dict[str, Any]:
    ablated = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    if mode == "baseline":
        return ablated
    if mode == "drop_object":
        if "object_image" in ablated:
            ablated["object_image"] = torch.zeros_like(ablated["object_image"])
        if "image" in ablated:
            ablated["image"] = torch.zeros_like(ablated["image"])
    elif mode == "drop_context":
        if "context_image" in ablated:
            ablated["context_image"] = torch.zeros_like(ablated["context_image"])
    elif mode == "drop_shape":
        if "contour_geometry" in ablated:
            ablated["contour_geometry"] = _equal_area_circle_geometry(ablated["contour_geometry"])
    else:  # pragma: no cover - Literal prevents this
        raise ValueError(f"Unsupported ablation mode: {mode}")
    return ablated


def _equal_area_circle_geometry(geometry: torch.Tensor) -> torch.Tensor:
    values = torch.zeros_like(geometry)
    if geometry.numel() == 0:
        return values
    if geometry.ndim == 1:
        geometry = geometry.unsqueeze(0)
        values = values.unsqueeze(0)
    area = torch.clamp(geometry[:, 0], min=0.0)
    values[:, 0] = area
    if values.shape[1] > 1:
        values[:, 1] = 2.0 * torch.sqrt(torch.clamp(math.pi * area, min=0.0))
    if values.shape[1] > 2:
        values[:, 2] = 0.0
    return values.reshape_as(geometry)


def _ablation_operation(mode: AblationMode) -> str:
    if mode == "baseline":
        return "unaltered_contour_evidence"
    if mode == "drop_object":
        return "zero_object_rgb"
    if mode == "drop_context":
        return "zero_context_rgb"
    return "equal_area_circle_geometry"


def _retrieval_score(query: np.ndarray, reference_region_emb: np.ndarray, target_index: int) -> dict[str, Any]:
    if reference_region_emb.size == 0 or target_index < 0 or target_index >= len(reference_region_emb):
        return {
            "i_to_g_rank": None,
            "i_to_g_at_1": False,
            "i_to_g_at_5": False,
            "matched_similarity": None,
            "top1_similarity": None,
            "top1_dataset_index": None,
            "reciprocal_rank": None,
        }
    similarities = reference_region_emb @ query.astype(np.float32)
    order = np.argsort(-similarities, kind="mergesort")
    matches = np.flatnonzero(order == int(target_index))
    rank = int(matches[0]) + 1 if matches.size else len(order) + 1
    return {
        "i_to_g_rank": rank,
        "i_to_g_at_1": bool(rank <= 1),
        "i_to_g_at_5": bool(rank <= 5),
        "matched_similarity": float(similarities[int(target_index)]),
        "top1_similarity": float(similarities[int(order[0])]) if len(order) else None,
        "top1_dataset_index": int(order[0]) if len(order) else None,
        "reciprocal_rank": float(1.0 / rank) if rank > 0 else None,
    }


def _attach_baseline_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    baseline = frame[frame["ablation_mode"] == "baseline"].set_index("dataset_index")
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        payload = row.to_dict()
        base = baseline.loc[payload["dataset_index"]] if payload["dataset_index"] in baseline.index else pd.Series(dtype=object)
        base_similarity = _safe_float(base.get("matched_similarity"))
        similarity = _safe_float(payload.get("matched_similarity"))
        base_rank = _safe_int(base.get("i_to_g_rank"))
        rank = _safe_int(payload.get("i_to_g_rank"))
        payload["baseline_matched_similarity"] = base_similarity
        payload["matched_similarity_drop"] = (
            float(base_similarity - similarity) if base_similarity is not None and similarity is not None else None
        )
        payload["rank_delta_vs_baseline"] = int(rank - base_rank) if rank is not None and base_rank is not None else None
        payload["i_to_g_at5_drop"] = int(bool(base.get("i_to_g_at_5")) and not bool(payload.get("i_to_g_at_5")))
        rows.append(payload)
    return pd.DataFrame(rows)


def _summary_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "ablation_mode",
                "n_targets",
                "i_to_g_at_1",
                "i_to_g_at_5",
                "mean_rank",
                "mean_matched_similarity",
                "mean_matched_similarity_drop",
                "mean_rank_delta_vs_baseline",
            ]
        )
    rows = []
    for mode, group in frame.groupby("ablation_mode", sort=False):
        rows.append(
            {
                "ablation_mode": mode,
                "n_targets": int(len(group)),
                "i_to_g_at_1": float(group["i_to_g_at_1"].astype(float).mean()),
                "i_to_g_at_5": float(group["i_to_g_at_5"].astype(float).mean()),
                "mean_rank": _mean(group["i_to_g_rank"]),
                "mean_matched_similarity": _mean(group["matched_similarity"]),
                "mean_matched_similarity_drop": _mean(group.get("matched_similarity_drop", pd.Series(dtype=float))),
                "mean_rank_delta_vs_baseline": _mean(group.get("rank_delta_vs_baseline", pd.Series(dtype=float))),
            }
        )
    return pd.DataFrame(rows)


def _gallery_with_ablation(target_frame: pd.DataFrame, result_frame: pd.DataFrame) -> pd.DataFrame:
    if target_frame.empty:
        return target_frame.copy()
    gallery = target_frame.copy()
    if result_frame.empty:
        return gallery
    metrics = result_frame.pivot_table(
        index="dataset_index",
        columns="ablation_mode",
        values=["i_to_g_rank", "i_to_g_at_5", "matched_similarity", "matched_similarity_drop", "rank_delta_vs_baseline"],
        aggfunc="first",
    )
    metrics.columns = [f"{metric}_{mode}" for metric, mode in metrics.columns]
    metrics = metrics.reset_index()
    return gallery.merge(metrics, on="dataset_index", how="left", sort=False)


def _status_payload(
    *,
    checkpoint_path: Path,
    target_path: Path,
    config: StGPTConfig,
    n_loaded_targets: int,
    n_reference_regions: int,
    result_frame: pd.DataFrame,
) -> dict[str, Any]:
    status = "pass" if n_loaded_targets and not result_frame.empty else "warning"
    return {
        "status": status,
        "protocol": "E5.2 mask-aware contour ablation",
        "checkpoint": str(checkpoint_path),
        "targets": str(target_path),
        "case_name": config.case_name,
        "n_targets": int(n_loaded_targets),
        "n_reference_regions": int(n_reference_regions),
        "ablation_modes": list(ABLATION_MODES),
        "retrieval_gallery": "baseline region_emb from the same checkpoint and config",
        "shape_ablation": "equal-area circle geometry; object/context pixels and mask are kept unchanged",
    }


def _anatomy_markdown(status: dict[str, Any], summary: pd.DataFrame, gallery: pd.DataFrame) -> str:
    summary_table = _frame_to_markdown(summary) if not summary.empty else "No ablation rows were produced."
    columns = [
        col
        for col in (
            "evidence_id",
            "contour_id",
            "structure_label",
            "failure_reasons",
            "prototype_id",
            "i_to_g_rank_baseline",
            "i_to_g_rank_drop_object",
            "i_to_g_rank_drop_context",
            "i_to_g_rank_drop_shape",
            "matched_similarity_drop_drop_object",
            "matched_similarity_drop_drop_context",
            "matched_similarity_drop_drop_shape",
        )
        if col in gallery.columns
    ]
    gallery_table = _frame_to_markdown(gallery[columns].head(24)) if columns else "No target-level rows were produced."
    return f"""# Anatomy of a Failure

Protocol: {status.get("protocol")}

Checkpoint: `{status.get("checkpoint")}`

Targets: `{status.get("targets")}`

This report is an inference-only E5.2 smoke/analysis artifact. JSON stores only
pointers and scalar summaries; the retrieval gallery is the unablated baseline
`region_emb` bank from the same checkpoint.

## Summary

{summary_table}

## Target Deltas

{gallery_table}

## Interpretation Notes

- `drop_object` zeroes Object RGB and tests whether contour-internal morphology drives image-to-gene retrieval.
- `drop_context` zeroes Context RGB and tests whether surrounding niche morphology drives retrieval.
- `drop_shape` replaces explicit geometry with an equal-area circle prior, keeping pixels unchanged, so it isolates the shape token rather than pixel evidence.
- Positive `matched_similarity_drop` means the ablation weakened alignment relative to the unaltered baseline.
"""


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    columns = [str(column) for column in frame.columns]
    header = "| " + " | ".join(_escape_markdown_cell(column) for column in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(_escape_markdown_cell(row.get(column)) for column in frame.columns) + " |")
    return "\n".join([header, separator, *rows])


def _escape_markdown_cell(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _resolve_device(device: str) -> torch.device:
    normalized = str(device).strip()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(normalized)


def _mean(values: Any) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    if len(numeric) == 0:
        return None
    return float(numeric.mean())


def _safe_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value)
    return text if text and text.lower() not in {"nan", "none", "null"} else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if pd.isna(value):
        return None
    return value
