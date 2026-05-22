"""Region auto-annotation: propagate sparse expert structure labels to the rest.

This module implements RFC 0002.  Given an annotated subset of regions on a
case (the "seed set"), it produces per-region predicted structure labels for
the remaining regions ("the pool"), with calibrated probability, entropy,
nearest-seed distance, and an explicit abstain flag.

Two classifier paths are exposed and can be emitted together for comparison:

* ``structure_head`` – reuse the trained classification head from the
  checkpoint, with a single-scalar temperature fit on the seed labels.
* ``prototype_knn`` – class prototypes (or k-NN for very small seed counts)
  computed on the region embedding.  Independent of the head, so the same
  user can supply labels that were not in the training vocabulary.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import StGPTConfig
from .data import RegionDataset, build_training_case
from .foundation.packaging import resolve_model_checkpoint
from .models import ImageGeneSTGPT

ABSTAIN_LABEL = "__abstain__"
SCHEMA_VERSION = "stgpt.region_auto_annotation.v0.1"

ClassifierKind = Literal["structure_head", "prototype_knn", "both"]


@dataclass(frozen=True)
class AutoAnnotationResult:
    predictions: Path
    report: Path
    probabilities: Path | None
    path_agreement: Path | None
    n_seed_regions: int
    n_pool_regions: int
    n_abstain: int

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "predictions": str(self.predictions),
            "report": str(self.report),
            "n_seed_regions": int(self.n_seed_regions),
            "n_pool_regions": int(self.n_pool_regions),
            "n_abstain": int(self.n_abstain),
        }
        if self.probabilities is not None:
            payload["probabilities"] = str(self.probabilities)
        if self.path_agreement is not None:
            payload["path_agreement"] = str(self.path_agreement)
        return payload


def annotate_regions(
    *,
    config: StGPTConfig | str | Path,
    checkpoint: str | Path,
    seed_labels: str | Path,
    output_dir: str | Path,
    region_ids: str | Path | None = None,
    include_no_image: bool = False,
    classifier: ClassifierKind = "both",
    abstain_prob: float = 0.5,
    write_probabilities: bool = False,
    seed_folds: int = 5,
    rng_seed: int = 42,
    batch_size: int = 32,
    device: str = "auto",
) -> dict[str, Any]:
    """Propagate sparse expert structure labels to unannotated regions.

    Writes ``region_predictions.parquet`` and ``auto_annotation_report.json``
    under *output_dir*.  See RFC 0002 for the field-by-field contract.
    """
    cfg = StGPTConfig.from_file(config) if isinstance(config, (str, Path)) else config
    checkpoint_path = resolve_model_checkpoint(checkpoint)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inference = _run_inference(cfg, checkpoint_path, batch_size=batch_size, device=device)
    label_vocab = inference.label_vocab
    region_df = inference.region_table.copy().reset_index(drop=True)
    region_ids_all = region_df["region_id"].astype(str).to_numpy()
    embeddings = inference.embeddings
    logits = inference.structure_logits  # may be None

    seeds_path = Path(seed_labels)
    seed_frame = _read_seed_labels(seeds_path, region_ids_all)
    label_vocab, mode = _resolve_label_vocabulary(seed_frame, label_vocab, classifier)

    seed_index_map = _build_index_map(region_ids_all)
    seed_indices = np.array([seed_index_map[rid] for rid in seed_frame["region_id"].to_numpy()], dtype=np.int64)
    seed_labels_int = np.array([label_vocab.index(lbl) for lbl in seed_frame["structure_label"].to_numpy()], dtype=np.int64)
    seed_weights = seed_frame["confidence"].astype(np.float64).to_numpy()

    pool_mask = np.ones(len(region_df), dtype=bool)
    pool_mask[seed_indices] = False
    if not include_no_image and "qc_flag" in region_df.columns:
        pool_mask &= region_df["qc_flag"].astype(str).to_numpy() == "ok"
    if region_ids is not None:
        restrict = _read_region_id_list(Path(region_ids))
        pool_mask &= np.isin(region_ids_all, list(restrict))

    pool_indices = np.where(pool_mask)[0]
    n_classes = len(label_vocab)
    if n_classes < 2:
        raise ValueError("Need at least 2 distinct seed structure labels to propagate.")

    selected_paths: list[str] = []
    if classifier in {"structure_head", "both"}:
        if logits is None or mode != "checkpoint_vocab":
            if classifier == "structure_head":
                raise ValueError(
                    "Path A (structure_head) is unavailable: checkpoint vocabulary does not match seed labels, "
                    "or the checkpoint has no structure_head."
                )
        else:
            selected_paths.append("structure_head")
    if classifier in {"prototype_knn", "both"}:
        selected_paths.append("prototype_knn")
    if not selected_paths:
        raise ValueError("No classifier path is available for this run.")

    rng = np.random.default_rng(rng_seed)

    path_predictions: dict[str, _PathPrediction] = {}
    if "structure_head" in selected_paths:
        path_predictions["structure_head"] = _fit_predict_structure_head(
            logits=logits,
            seed_indices=seed_indices,
            seed_labels=seed_labels_int,
            seed_weights=seed_weights,
            pool_indices=pool_indices,
            n_classes=n_classes,
            rng=rng,
            seed_folds=seed_folds,
        )
    if "prototype_knn" in selected_paths:
        path_predictions["prototype_knn"] = _fit_predict_prototype(
            embeddings=embeddings,
            seed_indices=seed_indices,
            seed_labels=seed_labels_int,
            seed_weights=seed_weights,
            pool_indices=pool_indices,
            n_classes=n_classes,
            rng=rng,
            seed_folds=seed_folds,
        )

    primary_path = "structure_head" if "structure_head" in path_predictions else "prototype_knn"
    primary = path_predictions[primary_path]
    seed_emb_distances = _seed_to_seed_distances(embeddings, seed_indices)
    distance_cutoff = float(np.quantile(seed_emb_distances, 0.99)) if seed_emb_distances.size else float("inf")
    entropy_cutoff = float(np.log(max(n_classes, 2)) - 0.5)

    nearest_seed_idx, nearest_seed_dist = _nearest_seed(embeddings, seed_indices, pool_indices)

    predictions_table = _build_predictions_table(
        region_ids_all=region_ids_all,
        region_df=region_df,
        seed_indices=seed_indices,
        seed_labels_int=seed_labels_int,
        pool_indices=pool_indices,
        primary=primary,
        primary_path=primary_path,
        nearest_seed_idx=nearest_seed_idx,
        nearest_seed_dist=nearest_seed_dist,
        label_vocab=label_vocab,
        abstain_prob=float(abstain_prob),
        entropy_cutoff=entropy_cutoff,
        distance_cutoff=distance_cutoff,
        evidence_prefix=_evidence_prefix(cfg, checkpoint_path, seeds_path),
    )
    predictions_path = out_dir / "region_predictions.parquet"
    predictions_table.to_parquet(predictions_path, index=False)

    probabilities_path: Path | None = None
    if write_probabilities:
        probabilities_path = out_dir / "region_predictions_per_class.parquet"
        _build_probabilities_table(
            region_ids_all=region_ids_all,
            pool_indices=pool_indices,
            primary=primary,
            label_vocab=label_vocab,
        ).to_parquet(probabilities_path, index=False)

    path_agreement_path: Path | None = None
    if classifier == "both" and len(path_predictions) == 2:
        path_agreement_path = out_dir / "path_agreement.csv"
        _build_path_agreement_table(
            region_ids_all=region_ids_all,
            pool_indices=pool_indices,
            paths=path_predictions,
            label_vocab=label_vocab,
        ).to_csv(path_agreement_path, index=False)

    seed_cv = _seed_cross_validation(
        path=primary_path,
        embeddings=embeddings,
        logits=logits,
        seed_indices=seed_indices,
        seed_labels=seed_labels_int,
        seed_weights=seed_weights,
        n_classes=n_classes,
        seed_folds=seed_folds,
        rng=rng,
    )
    n_abstain = int((predictions_table["predicted_label"] == ABSTAIN_LABEL).sum())
    report = {
        "schema_version": SCHEMA_VERSION,
        "case_name": cfg.case_name,
        "checkpoint": str(checkpoint_path),
        "config_fingerprint": _config_hash(cfg),
        "checkpoint_fingerprint": _sha256_path(checkpoint_path),
        "seed_fingerprint": _sha256_path(seeds_path),
        "label_vocab": list(label_vocab),
        "label_vocab_mode": mode,
        "classifier_requested": classifier,
        "classifiers_used": list(path_predictions.keys()),
        "primary_classifier": primary_path,
        "abstain_rule": {
            "abstain_prob": float(abstain_prob),
            "entropy_cutoff": entropy_cutoff,
            "distance_cutoff": distance_cutoff,
        },
        "seed_counts_per_class": _label_counts(seed_labels_int, label_vocab),
        "n_seed_regions": int(seed_indices.size),
        "n_pool_regions": int(pool_indices.size),
        "n_abstain": n_abstain,
        "abstain_rate": float(n_abstain / max(1, pool_indices.size)),
        "warnings": _collect_warnings(seed_labels_int, label_vocab, n_abstain, pool_indices.size, seed_cv),
        "seed_cross_validation": seed_cv,
        "rng_seed": int(rng_seed),
    }
    report_path = out_dir / "auto_annotation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    result = AutoAnnotationResult(
        predictions=predictions_path,
        report=report_path,
        probabilities=probabilities_path,
        path_agreement=path_agreement_path,
        n_seed_regions=int(seed_indices.size),
        n_pool_regions=int(pool_indices.size),
        n_abstain=n_abstain,
    )
    return result.to_dict()


# ---------------------------------------------------------------------------
# Inference: collect embeddings AND structure_logits in one forward pass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _InferenceOutputs:
    region_table: pd.DataFrame
    embeddings: np.ndarray
    structure_logits: np.ndarray | None
    label_vocab: tuple[str, ...]


def _run_inference(
    cfg: StGPTConfig,
    checkpoint_path: Path,
    *,
    batch_size: int,
    device: str,
) -> _InferenceOutputs:
    payload: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu")
    inner_cfg = cfg.model_copy(deep=True)
    inner_cfg = StGPTConfig.model_validate({**inner_cfg.model_dump(), "training": {**inner_cfg.training.model_dump(), "batch_size": int(batch_size)}})
    case = build_training_case(inner_cfg)
    dataset = RegionDataset(case, inner_cfg, for_inference=True)
    checkpoint_genes = tuple(str(item) for item in payload.get("vocab", {}).get("genes", []))
    if checkpoint_genes and checkpoint_genes != dataset.vocab.genes:
        raise ValueError("Embedding data gene vocabulary does not match the checkpoint vocabulary.")

    target = _resolve_device(device)
    checkpoint_cfg = StGPTConfig.model_validate(payload.get("config", inner_cfg.model_dump()))
    n_structures = int(payload.get("n_structures", dataset.n_structures))
    label_vocab = tuple(payload.get("structure_names") or dataset.structure_names)
    if len(label_vocab) < n_structures:
        # Defensive: align vocab length to head width by padding with placeholders.
        label_vocab = tuple(list(label_vocab) + [f"__unknown_{i}__" for i in range(n_structures - len(label_vocab))])
    model = ImageGeneSTGPT(
        n_genes=dataset.vocab.size - 1,
        n_structures=n_structures,
        d_model=inner_cfg.model.d_model,
        n_heads=inner_cfg.model.n_heads,
        n_layers=inner_cfg.model.n_layers,
        dim_feedforward=inner_cfg.model.dim_feedforward,
        n_expression_bins=inner_cfg.model.n_expression_bins,
        image_channels=inner_cfg.model.image_channels,
        patch_scales=inner_cfg.model.patch_scales,
        image_encoder_backend="precomputed" if inner_cfg.data.image_embedding_store else inner_cfg.model.image_encoder_backend,
        image_encoder_preset=inner_cfg.model.image_encoder_preset,
        image_encoder_name=inner_cfg.model.image_encoder_name,
        image_encoder_frozen=inner_cfg.model.image_encoder_frozen,
        image_embedding_dim=inner_cfg.model.image_embedding_dim or dataset.image_embedding_dim or None,
        n_prototypes=checkpoint_cfg.model.n_prototypes,
        prototype_temperature=checkpoint_cfg.model.prototype_temperature,
        use_expression_values=inner_cfg.model.use_expression_values,
        use_image_context=inner_cfg.model.use_image_context,
        use_spatial_context=inner_cfg.model.use_spatial_context,
        use_structure_context=inner_cfg.model.use_structure_context and inner_cfg.data.include_structure_context,
        use_cell_context=inner_cfg.model.use_cell_context,
        dropout=inner_cfg.model.dropout,
    )
    model.load_state_dict(payload["model_state"], strict=False)
    model.to(target)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate, num_workers=0)
    embeddings_chunks: list[np.ndarray] = []
    logits_chunks: list[np.ndarray] = []
    have_logits = True
    with torch.no_grad():
        for batch in loader:
            batch_on_device = {k: v.to(target) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            output = model(
                gene_ids=batch_on_device["gene_ids"],
                expr_values=batch_on_device["expr_values"],
                expr_bins=batch_on_device["expr_bins"],
                image=batch_on_device["image"],
                spatial=batch_on_device["spatial"],
                context_ids=batch_on_device["context_ids"],
                gene_padding_mask=batch_on_device["gene_padding_mask"],
                cell_expr_values=batch_on_device["cell_expr_values"],
                cell_token_mask=batch_on_device["cell_token_mask"],
                object_image=batch_on_device.get("object_image"),
                context_image=batch_on_device.get("context_image"),
                contour_mask=batch_on_device.get("contour_mask"),
                contour_geometry=batch_on_device.get("contour_geometry"),
                precomputed_image_embedding=batch_on_device.get("precomputed_image_embedding"),
            )
            embeddings_chunks.append(output.region_emb.cpu().numpy().astype(np.float32))
            if output.structure_logits is None:
                have_logits = False
            else:
                logits_chunks.append(output.structure_logits.cpu().numpy().astype(np.float32))
    embeddings = np.vstack(embeddings_chunks) if embeddings_chunks else np.zeros((0, inner_cfg.model.d_model), dtype=np.float32)
    logits = np.vstack(logits_chunks) if (have_logits and logits_chunks) else None
    return _InferenceOutputs(
        region_table=dataset.region_table.copy(),
        embeddings=embeddings,
        structure_logits=logits,
        label_vocab=label_vocab,
    )


# ---------------------------------------------------------------------------
# Seed CSV parsing and validation
# ---------------------------------------------------------------------------


def _read_seed_labels(path: Path, all_region_ids: np.ndarray) -> pd.DataFrame:
    frame = pd.read_csv(path) if path.suffix.lower() in {".csv", ".tsv"} else pd.read_parquet(path)
    required = {"region_id", "structure_label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Seed labels file is missing required columns: {sorted(missing)}")
    frame = frame[list(required) + (["confidence"] if "confidence" in frame.columns else [])].copy()
    frame["region_id"] = frame["region_id"].astype(str)
    frame["structure_label"] = frame["structure_label"].astype(str)
    if "confidence" not in frame.columns:
        frame["confidence"] = 1.0
    else:
        frame["confidence"] = pd.to_numeric(frame["confidence"], errors="coerce").fillna(1.0).clip(lower=0.0)
    frame = frame.drop_duplicates(subset=["region_id"], keep="first")
    region_set = set(all_region_ids.tolist())
    unknown = sorted(set(frame["region_id"].tolist()) - region_set)
    if unknown:
        raise ValueError(f"Seed labels reference unknown region_ids: {unknown[:5]}{'...' if len(unknown) > 5 else ''}")
    return frame.reset_index(drop=True)


def _read_region_id_list(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        column = "region_id" if "region_id" in frame.columns else frame.columns[0]
        return [str(item) for item in frame[column].tolist()]
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve_label_vocabulary(
    seed_frame: pd.DataFrame,
    checkpoint_vocab: tuple[str, ...],
    classifier: ClassifierKind,
) -> tuple[tuple[str, ...], str]:
    seed_set = sorted(set(seed_frame["structure_label"].tolist()))
    if all(label in checkpoint_vocab for label in seed_set):
        return checkpoint_vocab, "checkpoint_vocab"
    if classifier == "structure_head":
        unknown = [label for label in seed_set if label not in checkpoint_vocab]
        raise ValueError(
            f"Path A (structure_head) was requested but seed labels include "
            f"classes not in the checkpoint vocabulary: {unknown}"
        )
    return tuple(seed_set), "seed_vocab"


def _build_index_map(region_ids: np.ndarray) -> dict[str, int]:
    return {str(rid): idx for idx, rid in enumerate(region_ids)}


# ---------------------------------------------------------------------------
# Path A: structure_head with temperature scaling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PathPrediction:
    pool_probs: np.ndarray  # (n_pool, n_classes)
    pool_top1_idx: np.ndarray  # (n_pool,)
    pool_top1_prob: np.ndarray  # (n_pool,)
    pool_entropy: np.ndarray  # (n_pool,)
    temperature: float


def _fit_predict_structure_head(
    *,
    logits: np.ndarray,
    seed_indices: np.ndarray,
    seed_labels: np.ndarray,
    seed_weights: np.ndarray,
    pool_indices: np.ndarray,
    n_classes: int,
    rng: np.random.Generator,
    seed_folds: int,
) -> _PathPrediction:
    head_width = logits.shape[1]
    if n_classes != head_width:
        # Pad logits with -inf for any class we did not train on; should not happen
        # because checkpoint_vocab mode aligns the two.
        raise ValueError("Structure head width does not match label vocabulary size.")
    seed_logits = logits[seed_indices]
    temperature = _fit_temperature(seed_logits, seed_labels, seed_weights)
    pool_logits = logits[pool_indices] / temperature
    pool_probs = _softmax(pool_logits)
    return _summarize(pool_probs, temperature)


def _fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> float:
    if logits.size == 0:
        return 1.0
    logits_t = torch.from_numpy(logits.astype(np.float64))
    labels_t = torch.from_numpy(labels.astype(np.int64))
    weights_t = torch.from_numpy(weights.astype(np.float64))
    log_temp = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(log_temp).clamp(min=1e-3, max=1e3)
        scaled = logits_t / temperature
        log_probs = torch.log_softmax(scaled, dim=1)
        nll = -log_probs.gather(1, labels_t.unsqueeze(1)).squeeze(1)
        loss = (nll * weights_t).sum() / weights_t.sum().clamp(min=1e-6)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except RuntimeError:
        # LBFGS occasionally fails on degenerate inputs; fall back to T=1.0.
        return 1.0
    temperature = float(torch.exp(log_temp).clamp(min=1e-3, max=1e3).item())
    if not np.isfinite(temperature) or temperature <= 0:
        return 1.0
    return temperature


# ---------------------------------------------------------------------------
# Path B: per-class prototypes on the region embedding
# ---------------------------------------------------------------------------


def _fit_predict_prototype(
    *,
    embeddings: np.ndarray,
    seed_indices: np.ndarray,
    seed_labels: np.ndarray,
    seed_weights: np.ndarray,
    pool_indices: np.ndarray,
    n_classes: int,
    rng: np.random.Generator,
    seed_folds: int,
) -> _PathPrediction:
    normalized = _l2_normalize(embeddings)
    prototypes = _class_prototypes(normalized, seed_indices, seed_labels, seed_weights, n_classes)
    seed_logits = normalized[seed_indices] @ prototypes.T  # cosine similarities
    temperature = _fit_temperature(seed_logits, seed_labels, seed_weights)
    pool_logits = (normalized[pool_indices] @ prototypes.T) / temperature
    pool_probs = _softmax(pool_logits)
    return _summarize(pool_probs, temperature)


def _class_prototypes(
    normalized: np.ndarray,
    seed_indices: np.ndarray,
    seed_labels: np.ndarray,
    seed_weights: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    d_model = normalized.shape[1]
    prototypes = np.zeros((n_classes, d_model), dtype=np.float32)
    for cls in range(n_classes):
        mask = seed_labels == cls
        if not mask.any():
            prototypes[cls] = 0.0
            continue
        weights = seed_weights[mask].astype(np.float64)
        weights = weights / max(weights.sum(), 1e-6)
        proto = (normalized[seed_indices[mask]].astype(np.float64) * weights[:, None]).sum(axis=0)
        norm = float(np.linalg.norm(proto))
        prototypes[cls] = (proto / norm).astype(np.float32) if norm > 1e-6 else proto.astype(np.float32)
    return prototypes


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    shifted = x - x.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float64)


def _summarize(pool_probs: np.ndarray, temperature: float) -> _PathPrediction:
    if pool_probs.size == 0:
        return _PathPrediction(
            pool_probs=pool_probs,
            pool_top1_idx=np.zeros(0, dtype=np.int64),
            pool_top1_prob=np.zeros(0, dtype=np.float64),
            pool_entropy=np.zeros(0, dtype=np.float64),
            temperature=temperature,
        )
    top1 = pool_probs.argmax(axis=1)
    top1_prob = pool_probs[np.arange(pool_probs.shape[0]), top1]
    entropy = -np.sum(pool_probs * np.log(np.clip(pool_probs, 1e-12, 1.0)), axis=1)
    return _PathPrediction(
        pool_probs=pool_probs,
        pool_top1_idx=top1.astype(np.int64),
        pool_top1_prob=top1_prob.astype(np.float64),
        pool_entropy=entropy.astype(np.float64),
        temperature=temperature,
    )


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return (matrix / norms).astype(np.float32)


def _nearest_seed(
    embeddings: np.ndarray,
    seed_indices: np.ndarray,
    pool_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if pool_indices.size == 0 or seed_indices.size == 0:
        return np.zeros(pool_indices.size, dtype=np.int64), np.zeros(pool_indices.size, dtype=np.float64)
    normalized = _l2_normalize(embeddings)
    seed_emb = normalized[seed_indices]
    pool_emb = normalized[pool_indices]
    sim = pool_emb @ seed_emb.T
    nearest = sim.argmax(axis=1)
    cosine_dist = 1.0 - sim[np.arange(sim.shape[0]), nearest]
    return seed_indices[nearest].astype(np.int64), np.clip(cosine_dist.astype(np.float64), 0.0, 2.0)


def _seed_to_seed_distances(embeddings: np.ndarray, seed_indices: np.ndarray) -> np.ndarray:
    if seed_indices.size < 2:
        return np.zeros(0, dtype=np.float64)
    normalized = _l2_normalize(embeddings)
    sub = normalized[seed_indices]
    sim = sub @ sub.T
    np.fill_diagonal(sim, -np.inf)
    nearest_sim = sim.max(axis=1)
    return np.clip(1.0 - nearest_sim, 0.0, 2.0).astype(np.float64)


def _build_predictions_table(
    *,
    region_ids_all: np.ndarray,
    region_df: pd.DataFrame,
    seed_indices: np.ndarray,
    seed_labels_int: np.ndarray,
    pool_indices: np.ndarray,
    primary: _PathPrediction,
    primary_path: str,
    nearest_seed_idx: np.ndarray,
    nearest_seed_dist: np.ndarray,
    label_vocab: tuple[str, ...],
    abstain_prob: float,
    entropy_cutoff: float,
    distance_cutoff: float,
    evidence_prefix: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    qc_col = region_df["qc_flag"].astype(str).to_numpy() if "qc_flag" in region_df.columns else np.array(["unknown"] * len(region_df))
    for rank, idx in enumerate(seed_indices.tolist()):
        rid = str(region_ids_all[idx])
        rows.append({
            "region_id": rid,
            "predicted_label": label_vocab[int(seed_labels_int[rank])],
            "predicted_prob": 1.0,
            "entropy": 0.0,
            "nearest_seed_region_id": rid,
            "nearest_seed_distance": 0.0,
            "qc_flag": "seed",
            "classifier": primary_path,
            "evidence_id": _evidence_id(evidence_prefix, rid),
            "propagation_kind": "same_slide",
            "expression_present": True,
            "source_case_id": evidence_prefix.split("|", 1)[0],
        })
    for pool_rank, idx in enumerate(pool_indices.tolist()):
        rid = str(region_ids_all[idx])
        top_idx = int(primary.pool_top1_idx[pool_rank])
        prob = float(primary.pool_top1_prob[pool_rank])
        entropy = float(primary.pool_entropy[pool_rank])
        nearest_idx = int(nearest_seed_idx[pool_rank])
        nearest_distance = float(nearest_seed_dist[pool_rank])
        abstain = (
            prob < float(abstain_prob)
            or entropy > float(entropy_cutoff)
            or nearest_distance > float(distance_cutoff)
        )
        rows.append({
            "region_id": rid,
            "predicted_label": ABSTAIN_LABEL if abstain else label_vocab[top_idx],
            "predicted_prob": prob,
            "entropy": entropy,
            "nearest_seed_region_id": str(region_ids_all[nearest_idx]) if nearest_seed_idx.size else "",
            "nearest_seed_distance": nearest_distance,
            "qc_flag": str(qc_col[idx]),
            "classifier": primary_path,
            "evidence_id": _evidence_id(evidence_prefix, rid),
            "propagation_kind": "same_slide",
            "expression_present": True,
            "source_case_id": evidence_prefix.split("|", 1)[0],
        })
    return pd.DataFrame(rows)


def _build_probabilities_table(
    *,
    region_ids_all: np.ndarray,
    pool_indices: np.ndarray,
    primary: _PathPrediction,
    label_vocab: tuple[str, ...],
) -> pd.DataFrame:
    if pool_indices.size == 0 or primary.pool_probs.size == 0:
        return pd.DataFrame(columns=["region_id", "structure_label", "probability"])
    rows: list[dict[str, Any]] = []
    for pool_rank, idx in enumerate(pool_indices.tolist()):
        rid = str(region_ids_all[idx])
        for cls_idx, label in enumerate(label_vocab):
            rows.append({
                "region_id": rid,
                "structure_label": label,
                "probability": float(primary.pool_probs[pool_rank, cls_idx]),
            })
    return pd.DataFrame(rows)


def _build_path_agreement_table(
    *,
    region_ids_all: np.ndarray,
    pool_indices: np.ndarray,
    paths: dict[str, _PathPrediction],
    label_vocab: tuple[str, ...],
) -> pd.DataFrame:
    head = paths["structure_head"]
    proto = paths["prototype_knn"]
    rows: list[dict[str, Any]] = []
    for pool_rank, idx in enumerate(pool_indices.tolist()):
        head_top = label_vocab[int(head.pool_top1_idx[pool_rank])]
        proto_top = label_vocab[int(proto.pool_top1_idx[pool_rank])]
        rows.append({
            "region_id": str(region_ids_all[idx]),
            "structure_head_top1": head_top,
            "structure_head_prob": float(head.pool_top1_prob[pool_rank]),
            "prototype_knn_top1": proto_top,
            "prototype_knn_prob": float(proto.pool_top1_prob[pool_rank]),
            "agree": bool(head_top == proto_top),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Seed cross-validation QC
# ---------------------------------------------------------------------------


def _seed_cross_validation(
    *,
    path: str,
    embeddings: np.ndarray,
    logits: np.ndarray | None,
    seed_indices: np.ndarray,
    seed_labels: np.ndarray,
    seed_weights: np.ndarray,
    n_classes: int,
    seed_folds: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    n_seeds = seed_indices.size
    per_class_counts = [int((seed_labels == c).sum()) for c in range(n_classes)]
    min_per_class = min(per_class_counts) if per_class_counts else 0
    effective_folds = max(2, min(int(seed_folds), n_seeds, min_per_class)) if min_per_class >= 2 else 0
    if effective_folds < 2 or n_seeds < 2:
        return {
            "status": "skipped",
            "reason": "insufficient_seeds_for_cv",
            "n_seeds": int(n_seeds),
            "n_classes": int(n_classes),
            "per_class_counts": per_class_counts,
        }
    folds = _stratified_fold_indices(seed_labels, n_folds=effective_folds, rng=rng)
    oof_pred = np.full(n_seeds, -1, dtype=np.int64)
    for fold_idx in range(effective_folds):
        train_mask = folds != fold_idx
        valid_mask = ~train_mask
        if path == "structure_head" and logits is not None:
            train_logits = logits[seed_indices[train_mask]]
            valid_logits = logits[seed_indices[valid_mask]]
            temperature = _fit_temperature(train_logits, seed_labels[train_mask], seed_weights[train_mask])
            probs = _softmax(valid_logits / temperature)
        else:
            normalized = _l2_normalize(embeddings)
            prototypes = _class_prototypes(
                normalized,
                seed_indices[train_mask],
                seed_labels[train_mask],
                seed_weights[train_mask],
                n_classes,
            )
            valid_sim = normalized[seed_indices[valid_mask]] @ prototypes.T
            temperature = _fit_temperature(
                normalized[seed_indices[train_mask]] @ prototypes.T,
                seed_labels[train_mask],
                seed_weights[train_mask],
            )
            probs = _softmax(valid_sim / temperature)
        oof_pred[valid_mask] = probs.argmax(axis=1)
    return _classification_report(seed_labels, oof_pred, n_classes=n_classes, effective_folds=int(effective_folds))


def _stratified_fold_indices(labels: np.ndarray, *, n_folds: int, rng: np.random.Generator) -> np.ndarray:
    folds = np.full(labels.size, -1, dtype=np.int64)
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        for j, pos in enumerate(idx):
            folds[pos] = j % n_folds
    return folds


def _classification_report(true: np.ndarray, pred: np.ndarray, *, n_classes: int, effective_folds: int) -> dict[str, Any]:
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(true.tolist(), pred.tolist()):
        if p < 0:
            continue
        confusion[int(t), int(p)] += 1
    per_class: list[dict[str, float]] = []
    f1_scores: list[float] = []
    for cls in range(n_classes):
        tp = int(confusion[cls, cls])
        fp = int(confusion[:, cls].sum() - tp)
        fn = int(confusion[cls, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class.append({
            "class_index": cls,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(confusion[cls, :].sum()),
        })
        f1_scores.append(float(f1))
    return {
        "status": "ok",
        "n_folds": int(effective_folds),
        "macro_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "accuracy": float((true == pred).mean()) if pred.size else 0.0,
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


def _label_counts(seed_labels: np.ndarray, label_vocab: tuple[str, ...]) -> dict[str, int]:
    counts: dict[str, int] = {label: 0 for label in label_vocab}
    for cls_idx in seed_labels.tolist():
        counts[label_vocab[int(cls_idx)]] += 1
    return counts


def _collect_warnings(
    seed_labels: np.ndarray,
    label_vocab: tuple[str, ...],
    n_abstain: int,
    n_pool: int,
    seed_cv: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    counts = _label_counts(seed_labels, label_vocab)
    low = [label for label, count in counts.items() if count < 5]
    if low:
        warnings.append(f"low_seed_count: classes with fewer than 5 seeds: {low}")
    if n_pool > 0 and (n_abstain / n_pool) > 0.5:
        warnings.append(f"high_abstain_rate: {n_abstain}/{n_pool} regions abstained")
    if seed_cv.get("status") == "ok" and seed_cv.get("macro_f1", 0.0) < 0.6:
        warnings.append(f"low_seed_macro_f1: {seed_cv['macro_f1']:.3f}")
    return warnings


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def _evidence_prefix(cfg: StGPTConfig, checkpoint_path: Path, seeds_path: Path) -> str:
    return f"{cfg.case_name}|{_config_hash(cfg)}|{_sha256_path(checkpoint_path) or 'no-checkpoint'}|{_sha256_path(seeds_path) or 'no-seeds'}"


def _evidence_id(prefix: str, region_id: str) -> str:
    return "ann_" + hashlib.sha256(f"{prefix}|{region_id}".encode("utf-8")).hexdigest()[:16]


def _config_hash(cfg: StGPTConfig) -> str:
    payload = json.dumps(cfg.model_dump(mode="json"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_device(name: str) -> torch.device:
    normalized = str(name).lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(normalized)


__all__ = [
    "ABSTAIN_LABEL",
    "AutoAnnotationResult",
    "SCHEMA_VERSION",
    "annotate_regions",
]
