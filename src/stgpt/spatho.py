from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import StGPTConfig
from .data import ImageGeneDataset, TrainingCase, build_training_case
from .models import ImageGeneSTGPT
from .qc import validate_training_case

MODEL_MANIFEST = "stgpt_model_manifest.json"
MODEL_CARD = "README.md"
PACKAGED_CHECKPOINT = "checkpoint.pt"


@dataclass
class SpathoStGPTModel:
    checkpoint_path: Path
    checkpoint_payload: dict[str, Any]
    config: StGPTConfig
    device: torch.device

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, *, device: str = "auto") -> SpathoStGPTModel:
        checkpoint_path = resolve_checkpoint_path(checkpoint)
        payload = torch.load(checkpoint_path, map_location="cpu")
        cfg = StGPTConfig.model_validate(payload["config"])
        target = _resolve_device(device)
        return cls(checkpoint_path=checkpoint_path, checkpoint_payload=payload, config=cfg, device=target)

    def embed_case(
        self,
        config: StGPTConfig | str | Path,
        *,
        batch_size: int = 32,
    ) -> tuple[ad.AnnData, pd.DataFrame, TrainingCase]:
        user_cfg = StGPTConfig.from_file(config) if isinstance(config, (str, Path)) else config
        eval_cfg = _merge_runtime_config(self.config, user_cfg, batch_size=batch_size)
        case = build_training_case(eval_cfg)
        dataset = ImageGeneDataset(case, eval_cfg, for_inference=True)
        _validate_vocab(self.checkpoint_payload, dataset)
        model = _load_model(self.checkpoint_payload, eval_cfg, dataset).to(self.device)
        model.eval()

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate, num_workers=0)
        cell_embeddings: list[np.ndarray] = []
        image_embeddings: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                output = model(
                    gene_ids=batch["gene_ids"],
                    expr_values=batch["expr_values"],
                    expr_bins=batch["expr_bins"],
                    image=batch["image"],
                    spatial=batch["spatial"],
                    context_ids=batch["context_ids"],
                    gene_padding_mask=batch["gene_padding_mask"],
                )
                cell_embeddings.append(output.cell_emb.cpu().numpy())
                image_embeddings.append(output.image_emb.cpu().numpy())

        adata = case.adata.copy()
        adata.obsm["X_stGPT"] = np.vstack(cell_embeddings).astype(np.float32)
        frame = _embedding_frame(adata, np.vstack(cell_embeddings), np.vstack(image_embeddings), case.patch_table, eval_cfg)
        return adata, frame, case


def embed_spatho_case(
    config: StGPTConfig | str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    batch_size: int = 32,
    device: str = "auto",
) -> dict[str, str]:
    model = SpathoStGPTModel.from_checkpoint(checkpoint, device=device)
    adata, embeddings, case = model.embed_case(config, batch_size=batch_size)
    outputs = write_spatho_artifacts(adata, embeddings, output_dir)
    qc_result = validate_training_case(case, StGPTConfig.from_file(config) if isinstance(config, (str, Path)) else config, output_dir=output_dir)
    manifest_path = _write_spatho_manifest(
        output_dir=Path(output_dir),
        model=model,
        config=config,
        embeddings=embeddings,
        outputs=outputs,
        qc_report=qc_result["qc_report_json"],
    )
    outputs["stgpt_spatho_manifest"] = str(manifest_path)
    outputs["qc_report"] = qc_result["qc_report_json"]
    return outputs


def write_spatho_artifacts(adata: ad.AnnData, embeddings: pd.DataFrame, output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame = embeddings.copy()
    if "cell_id" not in frame.columns:
        frame.insert(0, "cell_id", adata.obs["cell_id"].astype(str).to_numpy() if "cell_id" in adata.obs else adata.obs_names.astype(str))
    for column in ("cluster", "structure_id"):
        if column not in frame.columns and column in adata.obs.columns:
            frame[column] = adata.obs[column].astype(str).to_numpy()

    outputs: dict[str, str] = {}
    cell_path = out / "cell_embeddings.parquet"
    frame.to_parquet(cell_path, index=False)
    outputs["cell_embeddings"] = str(cell_path)

    emb_cols = [col for col in frame.columns if str(col).startswith("emb_")]
    if "structure_id" in frame.columns and emb_cols:
        structure = frame.groupby("structure_id", dropna=False)[emb_cols].mean().reset_index()
        structure.insert(1, "n_cells", frame.groupby("structure_id", dropna=False).size().to_numpy())
        path = out / "structure_embedding_summary.csv"
        structure.to_csv(path, index=False)
        outputs["structure_embedding_summary"] = str(path)

    patch_col = _first_existing_column(frame, ("patch_id", "contour_id", "image_path"))
    if patch_col is not None and emb_cols:
        patch = frame.groupby(patch_col, dropna=False)[emb_cols].mean().reset_index()
        patch.insert(1, "n_cells", frame.groupby(patch_col, dropna=False).size().to_numpy())
        path = out / "patch_embedding_summary.csv"
        patch.to_csv(path, index=False)
        outputs["patch_embedding_summary"] = str(path)
    return outputs


def package_model(
    *,
    checkpoint: str | Path,
    evaluation: str | Path,
    output_dir: str | Path,
    model_name: str | None = None,
) -> dict[str, str]:
    checkpoint_path = resolve_checkpoint_path(checkpoint)
    evaluation_path = Path(evaluation)
    if not evaluation_path.exists():
        raise FileNotFoundError(f"evaluation metrics file does not exist: {evaluation_path}")
    payload = torch.load(checkpoint_path, map_location="cpu")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    packaged_checkpoint = out / PACKAGED_CHECKPOINT
    shutil.copy2(checkpoint_path, packaged_checkpoint)
    evaluation_copy = out / "evaluation_metrics.json"
    shutil.copy2(evaluation_path, evaluation_copy)

    cfg = StGPTConfig.model_validate(payload["config"])
    metadata = _model_metadata(payload, cfg, model_name=model_name, checkpoint_path=packaged_checkpoint, evaluation_path=evaluation_copy)
    manifest_path = out / MODEL_MANIFEST
    config_path = out / "config.json"
    vocab_path = out / "vocab.json"
    card_path = out / MODEL_CARD
    manifest_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    config_path.write_text(json.dumps(cfg.to_json_dict(), indent=2), encoding="utf-8")
    vocab_path.write_text(json.dumps(payload.get("vocab", {}), indent=2), encoding="utf-8")
    card_path.write_text(_model_card(metadata), encoding="utf-8")
    return {
        "model_dir": str(out),
        "checkpoint": str(packaged_checkpoint),
        "evaluation_metrics": str(evaluation_copy),
        "manifest": str(manifest_path),
        "config": str(config_path),
        "vocab": str(vocab_path),
        "model_card": str(card_path),
    }


def resolve_checkpoint_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    if not candidate.exists():
        raise FileNotFoundError(f"model path does not exist: {candidate}")
    for name in (PACKAGED_CHECKPOINT, "last.pt", "checkpoints/last.pt"):
        nested = candidate / name
        if nested.exists():
            return nested
    manifest = candidate / MODEL_MANIFEST
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        checkpoint = payload.get("checkpoint")
        if checkpoint:
            resolved = candidate / checkpoint
            if resolved.exists():
                return resolved
    raise FileNotFoundError(f"No stGPT checkpoint found under model directory: {candidate}")


def _embedding_frame(
    adata: ad.AnnData,
    cell_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    patch_table: pd.DataFrame,
    config: StGPTConfig,
) -> pd.DataFrame:
    frame = pd.DataFrame(cell_embeddings.astype(np.float32), columns=[f"emb_{idx:04d}" for idx in range(cell_embeddings.shape[1])])
    frame.insert(0, "cell_id", adata.obs["cell_id"].astype(str).to_numpy() if "cell_id" in adata.obs else adata.obs_names.astype(str))
    for column in (config.data.cluster_key, config.data.structure_key):
        if column in adata.obs.columns:
            out_col = "cluster" if column == config.data.cluster_key else "structure_id"
            frame[out_col] = adata.obs[column].astype(str).to_numpy()
    frame["stgpt_embedding_source"] = "cell_emb"
    frame["stgpt_model_version"] = _stgpt_version()
    frame["stgpt_case_name"] = config.case_name
    for idx in range(image_embeddings.shape[1]):
        frame[f"image_emb_{idx:04d}"] = image_embeddings[:, idx].astype(np.float32)
    return _merge_patch_provenance(frame, patch_table)


def _merge_patch_provenance(frame: pd.DataFrame, patch_table: pd.DataFrame) -> pd.DataFrame:
    if patch_table.empty or "cell_id" not in patch_table.columns:
        return frame
    columns = [col for col in ("cell_id", "contour_id", "patch_id", "image_path", "structure_name") if col in patch_table.columns]
    if len(columns) <= 1:
        return frame
    provenance = patch_table[columns].copy()
    provenance["cell_id"] = provenance["cell_id"].astype(str)
    if "patch_id" not in provenance.columns:
        if "contour_id" in provenance.columns:
            provenance["patch_id"] = provenance["contour_id"].astype(str)
        elif "image_path" in provenance.columns:
            provenance["patch_id"] = provenance["image_path"].astype(str)
    return frame.merge(provenance.drop_duplicates("cell_id"), on="cell_id", how="left")


def _merge_runtime_config(checkpoint_cfg: StGPTConfig, user_cfg: StGPTConfig, *, batch_size: int) -> StGPTConfig:
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
        patch_scales=config.model.patch_scales,
        use_expression_values=config.model.use_expression_values,
        use_image_context=config.model.use_image_context,
        use_spatial_context=config.model.use_spatial_context,
        use_structure_context=config.model.use_structure_context and config.data.include_structure_context,
        dropout=config.model.dropout,
    )
    model.load_state_dict(checkpoint_payload["model_state"], strict=False)
    return model


def _validate_vocab(checkpoint_payload: dict[str, Any], dataset: ImageGeneDataset) -> None:
    checkpoint_genes = tuple(str(item) for item in checkpoint_payload.get("vocab", {}).get("genes", []))
    if checkpoint_genes and checkpoint_genes != dataset.vocab.genes:
        raise ValueError("spatho data gene vocabulary does not match the stGPT checkpoint vocabulary.")


def _write_spatho_manifest(
    *,
    output_dir: Path,
    model: SpathoStGPTModel,
    config: StGPTConfig | str | Path,
    embeddings: pd.DataFrame,
    outputs: dict[str, str],
    qc_report: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = StGPTConfig.from_file(config) if isinstance(config, (str, Path)) else config
    payload = {
        "case_name": cfg.case_name,
        "model_checkpoint": str(model.checkpoint_path),
        "model_version": _stgpt_version(),
        "n_cells": int(len(embeddings)),
        "embedding_dim": int(len([col for col in embeddings.columns if str(col).startswith("emb_")])),
        "outputs": outputs,
        "qc_report": qc_report,
        "spatho_runtime_dependency": "optional",
    }
    path = output_dir / "stgpt_spatho_manifest.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _model_metadata(
    payload: dict[str, Any],
    config: StGPTConfig,
    *,
    model_name: str | None,
    checkpoint_path: Path,
    evaluation_path: Path,
) -> dict[str, Any]:
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    vocab = payload.get("vocab", {})
    metadata = {
        "model_name": model_name or config.case_name,
        "model_type": "stGPT morpho-molecular embedding backend",
        "checkpoint": checkpoint_path.name,
        "evaluation_metrics": evaluation_path.name,
        "stgpt_version": payload.get("model_version", _stgpt_version()),
        "case_name": config.case_name,
        "n_genes": len(vocab.get("genes", [])),
        "n_structures": int(payload.get("n_structures", 1)),
        "structure_names": payload.get("structure_names", []),
        "panel_metadata": payload.get("panel_metadata", _panel_metadata(config)),
        "training_summary": payload.get("training_summary", _training_summary(payload)),
        "evaluation_summary": {
            "case_name": evaluation.get("case_name"),
            "overall_prediction": evaluation.get("overall_prediction", {}),
            "overall_retrieval": evaluation.get("overall_retrieval", []),
            "overall_embedding_qc": evaluation.get("overall_embedding_qc", []),
            "artifacts": evaluation.get("artifacts", {}),
        },
        "intended_use": "Generate Xenium-centered morpho-molecular embeddings for spatho-compatible spatial pathology workflows.",
        "limitations": [
            "The first packaged models are expected to be dataset- and panel-specific.",
            "Outputs are research-use embeddings, not clinical diagnostic predictions.",
            "H&E registration and panel mismatch must be inspected through QC/failure-analysis artifacts.",
        ],
    }
    return metadata


def _model_card(metadata: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"# {metadata['model_name']}",
            "",
            "## Model Summary",
            "",
            f"- Type: {metadata['model_type']}",
            f"- stGPT version: {metadata['stgpt_version']}",
            f"- Case: {metadata['case_name']}",
            f"- Genes: {metadata['n_genes']}",
            f"- Structures: {metadata['n_structures']}",
            "",
            "## Intended Use",
            "",
            metadata["intended_use"],
            "",
            "## Panel and Data Scope",
            "",
            f"```json\n{json.dumps(metadata['panel_metadata'], indent=2)}\n```",
            "",
            "## Evaluation Summary",
            "",
            f"```json\n{json.dumps(metadata['evaluation_summary'], indent=2)}\n```",
            "",
            "## Limitations",
            "",
            *[f"- {item}" for item in metadata["limitations"]],
            "",
        ]
    )


def _panel_metadata(config: StGPTConfig) -> dict[str, Any]:
    panel_genes = config.data.panel_genes or []
    return {
        "panel_gene_count": len(panel_genes),
        "panel_gene_file": config.data.panel_gene_file,
        "gene_name_key": config.data.gene_name_key,
    }


def _training_summary(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics", [])
    return {
        "steps": len(metrics),
        "last_metrics": metrics[-1] if metrics else {},
    }


def _first_existing_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    return next((column for column in candidates if column in frame.columns), None)


def _stgpt_version() -> str:
    try:
        return version("stgpt")
    except PackageNotFoundError:
        return "0.1.0"


def _resolve_device(name: str) -> torch.device:
    normalized = str(name).lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(normalized)
