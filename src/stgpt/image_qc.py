from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import StGPTConfig
from .data import RegionDataset, build_training_case
from .models import ContourEvidenceEncoder, resolve_image_encoder_spec


def inspect_images(
    config: StGPTConfig | str | Path,
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Inspect contour H&E evidence and write figure-ready image QC artifacts."""
    cfg = StGPTConfig.from_file(config) if isinstance(config, (str, Path)) else config
    dataset = RegionDataset(build_training_case(cfg), cfg, for_inference=True)
    return inspect_dataset_images(dataset, cfg, output_dir=output_dir)


def inspect_dataset_images(
    dataset: RegionDataset,
    config: StGPTConfig,
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [_inspect_item(dataset, index) for index in range(len(dataset))]
    frame = pd.DataFrame(rows)
    summary = _image_qc_summary(frame, config)
    csv_path = out / "image_qc_summary.csv"
    json_path = out / "image_qc_summary.json"
    frame.to_csv(csv_path, index=False)
    payload = {
        "summary": summary,
        "artifacts": {
            "image_qc_summary_csv": str(csv_path),
            "image_qc_summary_json": str(json_path),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return payload


def precompute_image_embeddings(
    config: StGPTConfig | str | Path,
    *,
    output: str | Path,
    encoder_backend: str | None = None,
    encoder_preset: str | None = None,
    encoder_name: str | None = None,
    batch_size: int = 32,
    device: str = "auto",
) -> dict[str, Any]:
    """Precompute contour H&E embeddings from object/context/mask evidence."""
    cfg = StGPTConfig.from_file(config) if isinstance(config, (str, Path)) else config
    dataset = RegionDataset(build_training_case(cfg), cfg, for_inference=True)
    preset = encoder_preset if encoder_preset is not None else cfg.model.image_encoder_preset
    backend = str(encoder_backend or ("timm" if preset else cfg.model.image_encoder_backend))
    if backend == "precomputed":
        raise ValueError("precompute-images needs a live image encoder backend, not 'precomputed'.")
    spec = resolve_image_encoder_spec(
        backend=backend,
        name=encoder_name if encoder_name is not None else cfg.model.image_encoder_name,
        preset=preset,
    )
    backend = spec.backend
    name = spec.name
    target = _resolve_device(device)
    encoder = ContourEvidenceEncoder(
        cfg.model.image_channels,
        cfg.model.d_model,
        scales=cfg.model.patch_scales,
        image_encoder_backend=backend,  # type: ignore[arg-type]
        image_encoder_preset=preset,  # type: ignore[arg-type]
        image_encoder_name=name,
        image_encoder_frozen=cfg.model.image_encoder_frozen,
        image_embedding_dim=cfg.model.image_embedding_dim,
    ).to(target)
    encoder.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate, num_workers=4)
    embeddings: list[np.ndarray] = []
    region_ids: list[str] = []
    image_sources: list[int] = []
    with torch.no_grad():
        for batch in loader:
            moved = {key: value.to(target) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            _, image_emb = encoder(
                object_image=moved["object_image"],
                context_image=moved.get("context_image"),
                contour_mask=moved.get("contour_mask"),
                contour_geometry=moved.get("contour_geometry"),
            )
            embeddings.append(image_emb.detach().cpu().numpy().astype(np.float32))
            region_ids.extend(str(item) for item in batch["region_ids"])
            image_sources.extend([int(item) for item in batch["image_source"].detach().cpu().numpy().tolist()])
    matrix = np.vstack(embeddings).astype(np.float32) if embeddings else np.zeros((0, cfg.model.d_model), dtype=np.float32)
    frame = pd.DataFrame(matrix, columns=[f"emb_{idx}" for idx in range(matrix.shape[1])])
    frame.insert(0, "image_source", image_sources)
    frame.insert(0, "region_id", region_ids)
    frame["encoder_backend"] = backend
    frame["encoder_preset"] = preset
    frame["encoder_name"] = name
    frame["encoder_frozen"] = bool(cfg.model.image_encoder_frozen)
    frame["image_embedding_dim"] = int(matrix.shape[1])
    frame["image_size"] = spec.image_size or cfg.model.image_size
    frame["input_mode"] = spec.input_mode
    frame["normalization_source"] = spec.normalization_source
    frame["embedding_strategy"] = spec.embedding_strategy
    frame["gated_access"] = spec.gated_access

    output_path = Path(output)
    if output_path.suffix.lower() != ".parquet":
        output_path.mkdir(parents=True, exist_ok=True)
        store_path = output_path / "image_embeddings.parquet"
        manifest_path = output_path / "image_embedding_manifest.csv"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        store_path = output_path
        manifest_path = output_path.with_name("image_embedding_manifest.csv")
    frame.to_parquet(store_path, index=False)
    _embedding_manifest(frame, store_path, cfg).to_csv(manifest_path, index=False)
    summary = {
        "case_name": cfg.case_name,
        "n_regions": int(len(frame)),
        "embedding_dim": int(matrix.shape[1]),
        "encoder_backend": backend,
        "encoder_preset": preset,
        "encoder_name": name,
        "normalization_source": spec.normalization_source,
        "store": str(store_path),
        "manifest": str(manifest_path),
    }
    return summary


def _inspect_item(dataset: RegionDataset, index: int) -> dict[str, Any]:
    item = dataset[index]
    evidence = item["image_evidence"]
    object_image = evidence["object_image"].detach().cpu()
    mask = evidence["mask"].detach().cpu()
    stats = _tensor_stats(object_image)
    row = dataset.region_table.iloc[index]
    return {
        "region_id": item["region_id"],
        "image_path": row.get("image_path"),
        "image_source": evidence.get("source"),
        "image_source_id": int(evidence.get("source_id", 0)),
        "has_image": int(evidence.get("source_id", 0)) > 0,
        "has_precomputed_embedding": bool(evidence.get("has_precomputed_embedding", False)),
        "width": int(object_image.shape[-1]),
        "height": int(object_image.shape[-2]),
        "channels": int(object_image.shape[0]),
        "blankness": float(stats["blankness"]),
        "is_blank": bool(stats["blankness"] > 0.98),
        "tissue_fraction": float(stats["tissue_fraction"]),
        "mask_coverage": float(mask.float().mean().item()) if mask.numel() else float("nan"),
        "rgb_mean_r": float(stats["mean"][0]),
        "rgb_mean_g": float(stats["mean"][1]),
        "rgb_mean_b": float(stats["mean"][2]),
        "rgb_std_r": float(stats["std"][0]),
        "rgb_std_g": float(stats["std"][1]),
        "rgb_std_b": float(stats["std"][2]),
        "qc_flag": row.get("qc_flag", "unknown"),
    }


def _tensor_stats(image: torch.Tensor) -> dict[str, Any]:
    tensor = image.float().clamp(0.0, 1.0)
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    elif tensor.shape[0] < 3:
        pad = torch.zeros(3 - tensor.shape[0], tensor.shape[1], tensor.shape[2], dtype=tensor.dtype)
        tensor = torch.cat([tensor, pad], dim=0)
    tensor = tensor[:3]
    channel_mean = tensor.mean(dim=(1, 2)).numpy().astype(float)
    channel_std = tensor.std(dim=(1, 2)).numpy().astype(float)
    intensity = tensor.mean(dim=0)
    saturation = (tensor.max(dim=0).values - tensor.min(dim=0).values)
    tissue = ((intensity < 0.92) & (intensity > 0.05) & (saturation > 0.04)).float().mean().item()
    blankness = 1.0 - min(float(tensor.std().item()) / 0.12, 1.0)
    return {"mean": channel_mean.tolist(), "std": channel_std.tolist(), "tissue_fraction": tissue, "blankness": blankness}


def _image_qc_summary(frame: pd.DataFrame, config: StGPTConfig) -> dict[str, Any]:
    missing = int((~frame["has_image"]).sum()) if "has_image" in frame else 0
    blank = int(frame["is_blank"].sum()) if "is_blank" in frame else 0
    low_tissue = int((frame["tissue_fraction"] < 0.01).sum()) if "tissue_fraction" in frame else 0
    fatal_errors: list[str] = []
    warnings: list[str] = []
    if missing:
        fatal_errors.append(f"{missing} regions have no readable H&E image evidence.")
    if blank:
        warnings.append(f"{blank} regions look blank after resizing.")
    if low_tissue:
        warnings.append(f"{low_tissue} regions have very low tissue-fraction estimates.")
    return {
        "case_name": config.case_name,
        "n_regions": int(len(frame)),
        "status": "fail" if fatal_errors else "pass",
        "fatal_errors": fatal_errors,
        "warnings": warnings,
        "missing_image_count": missing,
        "blank_patch_count": blank,
        "low_tissue_patch_count": low_tissue,
        "mean_tissue_fraction": float(frame["tissue_fraction"].mean()) if len(frame) else float("nan"),
        "mean_mask_coverage": float(frame["mask_coverage"].mean()) if len(frame) else float("nan"),
        "stain_normalization": config.data.image_stain_normalization,
    }


def _embedding_manifest(frame: pd.DataFrame, store_path: Path, config: StGPTConfig) -> pd.DataFrame:
    rows = []
    for backend, group in frame.groupby("encoder_backend", dropna=False):
        rows.append(
            {
                "case_name": config.case_name,
                "store": str(store_path),
                "encoder_backend": backend,
                "encoder_preset": group["encoder_preset"].iloc[0] if "encoder_preset" in group and not group.empty else None,
                "encoder_name": group["encoder_name"].iloc[0] if "encoder_name" in group and not group.empty else config.model.image_encoder_name,
                "n_regions": int(len(group)),
                "embedding_dim": int(group["image_embedding_dim"].iloc[0]) if not group.empty else 0,
                "image_size": int(group["image_size"].iloc[0]) if "image_size" in group and not group.empty else config.model.image_size,
                "input_mode": group["input_mode"].iloc[0] if "input_mode" in group and not group.empty else "RGB",
                "normalization_source": group["normalization_source"].iloc[0] if "normalization_source" in group and not group.empty else None,
                "embedding_strategy": group["embedding_strategy"].iloc[0] if "embedding_strategy" in group and not group.empty else None,
                "gated_access": bool(group["gated_access"].iloc[0]) if "gated_access" in group and not group.empty else False,
                "uses_object_context_mask_geometry": True,
            }
        )
    return pd.DataFrame(rows)


def _resolve_device(name: str) -> torch.device:
    normalized = str(name).lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(normalized)
