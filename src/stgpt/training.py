from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import AblationMode, StGPTConfig
from .data import ImageGeneDataset, build_training_case
from .losses import compute_losses
from .models import ImageGeneSTGPT


def train(
    config: StGPTConfig | str | Path,
    *,
    preset: str | None = None,
    max_steps: int | None = None,
    ablation: AblationMode | str | None = None,
) -> dict[str, Any]:
    cfg = StGPTConfig.from_file(config, preset=preset) if isinstance(config, (str, Path)) else config.apply_preset(preset)
    cfg = cfg.apply_ablation(ablation or cfg.training.ablation_mode)
    if max_steps is not None:
        payload = cfg.model_dump()
        payload["training"]["max_steps"] = int(max_steps)
        cfg = StGPTConfig.model_validate(payload)
    _seed_everything(cfg.training.seed)
    device = _resolve_device(cfg.training.device)
    case = build_training_case(cfg)
    dataset = ImageGeneDataset(case, cfg)
    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=dataset.collate,
        num_workers=cfg.training.num_workers,
        drop_last=False,
    )
    model = ImageGeneSTGPT(
        n_genes=dataset.vocab.size - 1,
        n_structures=dataset.n_structures,
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
        dropout=cfg.model.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    output_dir = cfg.training.output_path
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, float]] = []
    step = 0
    model.train()
    while step < cfg.training.max_steps:
        for batch in loader:
            if step >= cfg.training.max_steps:
                break
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                gene_ids=batch["gene_ids"],
                expr_values=batch["expr_values"],
                expr_bins=batch["expr_bins"],
                image=batch["image"],
                spatial=batch["spatial"],
                context_ids=batch["context_ids"],
                gene_padding_mask=batch["gene_padding_mask"],
            )
            losses = compute_losses(
                output,
                batch,
                image_gene_weight=cfg.training.image_gene_loss_weight,
                neighborhood_weight=cfg.training.neighborhood_loss_weight,
                structure_weight=cfg.training.structure_loss_weight,
            )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            metrics.append({key: float(value.detach().cpu()) for key, value in losses.items()})
            step += 1
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "last.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg.to_json_dict(),
            "vocab": dataset.vocab.to_dict(),
            "n_structures": dataset.n_structures,
            "structure_names": dataset.structure_names,
            "metrics": metrics,
        },
        checkpoint_path,
    )
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {
        "checkpoint": str(checkpoint_path),
        "metrics": metrics,
        "metrics_path": str(metrics_path),
        "steps": step,
        "device": str(device),
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_device(name: str) -> torch.device:
    normalized = str(name).lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(normalized)


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}
