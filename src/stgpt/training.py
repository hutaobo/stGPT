from __future__ import annotations

import json
import math
import random
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .config import AblationMode, StGPTConfig
from .data import RegionDataset, build_training_case
from .losses import compute_losses
from .models import ImageGeneSTGPT
from .qc import make_splits


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
    dataset = RegionDataset(case, cfg)
    if len(dataset) == 0:
        raise ValueError("No trainable contour/region rows were found. Check contour membership and data.min_cells_per_region.")
    splits = make_splits(case, cfg)
    split_values = splits["split"].astype(str).to_numpy()
    train_indices = np.flatnonzero(split_values == "train").astype(int).tolist()
    val_indices = np.flatnonzero(split_values == "val").astype(int).tolist()
    if not train_indices:
        train_indices = list(range(len(dataset)))
    loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=dataset.collate,
        num_workers=cfg.training.num_workers,
        drop_last=False,
    )
    val_loader = (
        DataLoader(
            Subset(dataset, val_indices),
            batch_size=cfg.training.batch_size,
            shuffle=False,
            collate_fn=dataset.collate,
            num_workers=0,
            drop_last=False,
        )
        if val_indices
        else None
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
        use_cell_context=cfg.model.use_cell_context,
        dropout=cfg.model.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    scheduler = _make_scheduler(optimizer, cfg)
    output_dir = cfg.training.output_path
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, float]] = []
    best_metric = float("inf")
    best_checkpoint_path = checkpoint_dir / "best.pt"
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
                cell_expr_values=batch["cell_expr_values"],
                cell_token_mask=batch["cell_token_mask"],
            )
            weights = _scheduled_loss_weights(cfg, step + 1)
            losses = compute_losses(
                output,
                batch,
                image_gene_weight=weights["image_gene_loss_weight"],
                neighborhood_weight=weights["neighborhood_loss_weight"],
                structure_weight=weights["structure_loss_weight"],
            )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            step += 1
            metric_row = {key: float(value.detach().cpu()) for key, value in losses.items()}
            metric_row["lr"] = float(optimizer.param_groups[0]["lr"])
            metric_row.update({key: float(value) for key, value in weights.items()})

            if _should_validate(step, cfg):
                val_metrics = _evaluate_validation(model, val_loader, device, cfg, step)
                if val_metrics:
                    metric_row.update(val_metrics)
                    if val_metrics["val_loss"] < best_metric:
                        best_metric = val_metrics["val_loss"]
                        _save_checkpoint(
                            best_checkpoint_path,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            cfg=cfg,
                            dataset=dataset,
                            metrics=metrics + [metric_row],
                            step=step,
                            best_metric=best_metric,
                            split_summary=_split_summary(splits),
                        )

            metrics.append(metric_row)
            if cfg.training.save_every_n_steps and step % int(cfg.training.save_every_n_steps) == 0:
                _save_checkpoint(
                    checkpoint_dir / f"step_{step:06d}.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    cfg=cfg,
                    dataset=dataset,
                    metrics=metrics,
                    step=step,
                    best_metric=best_metric,
                    split_summary=_split_summary(splits),
                )

    checkpoint_path = checkpoint_dir / "last.pt"
    if not best_checkpoint_path.exists():
        best_metric = float(metrics[-1]["loss"]) if metrics else float("nan")
        _save_checkpoint(
            best_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            dataset=dataset,
            metrics=metrics,
            step=step,
            best_metric=best_metric,
            split_summary=_split_summary(splits),
        )
    _save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        dataset=dataset,
        metrics=metrics,
        step=step,
        best_metric=best_metric,
        split_summary=_split_summary(splits),
    )
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {
        "checkpoint": str(checkpoint_path),
        "best_checkpoint": str(best_checkpoint_path),
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


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _make_scheduler(optimizer: torch.optim.Optimizer, config: StGPTConfig):
    schedule = config.training.lr_schedule
    if schedule == "none":
        return None
    if schedule == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.training.learning_rate,
            total_steps=config.training.max_steps,
            pct_start=min(0.95, max(0.01, config.training.warmup_steps / max(1, config.training.max_steps))),
        )
    if schedule == "cosine":
        warmup = int(config.training.warmup_steps)
        total = int(config.training.max_steps)

        def lr_lambda(step: int) -> float:
            current = step + 1
            if warmup > 0 and current <= warmup:
                return max(1e-8, current / warmup)
            if total <= warmup:
                return 1.0
            progress = min(1.0, max(0.0, (current - warmup) / (total - warmup)))
            return max(1e-8, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    raise ValueError("training.lr_schedule must be one of: none, cosine, onecycle")


def _scheduled_loss_weights(config: StGPTConfig, step: int) -> dict[str, float]:
    return {
        "image_gene_loss_weight": _warmup_value(
            config.training.image_gene_loss_weight,
            step,
            config.training.image_gene_loss_warmup_steps,
        ),
        "neighborhood_loss_weight": _warmup_value(
            config.training.neighborhood_loss_weight,
            step,
            config.training.neighborhood_loss_warmup_steps,
        ),
        "structure_loss_weight": _warmup_value(
            config.training.structure_loss_weight,
            step,
            config.training.structure_loss_warmup_steps,
        ),
    }


def _warmup_value(target: float, step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return float(target)
    return float(target) * min(1.0, max(0.0, step / warmup_steps))


def _should_validate(step: int, config: StGPTConfig) -> bool:
    if step == 1 or step >= config.training.max_steps:
        return True
    if config.training.save_every_n_steps:
        return step % int(config.training.save_every_n_steps) == 0
    return False


def _evaluate_validation(
    model: ImageGeneSTGPT,
    val_loader: DataLoader | None,
    device: torch.device,
    config: StGPTConfig,
    step: int,
) -> dict[str, float]:
    if val_loader is None:
        return {}
    weights = _scheduled_loss_weights(config, step)
    rows: list[dict[str, float]] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch = _move_batch(batch, device)
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
            losses = compute_losses(
                output,
                batch,
                image_gene_weight=weights["image_gene_loss_weight"],
                neighborhood_weight=weights["neighborhood_loss_weight"],
                structure_weight=weights["structure_loss_weight"],
            )
            rows.append({key: float(value.detach().cpu()) for key, value in losses.items()})
    if was_training:
        model.train()
    if not rows:
        return {}
    keys = sorted(rows[0])
    return {f"val_{key}": float(np.mean([row[key] for row in rows])) for key in keys}


def _save_checkpoint(
    path: Path,
    *,
    model: ImageGeneSTGPT,
    optimizer: torch.optim.Optimizer,
    scheduler,
    cfg: StGPTConfig,
    dataset: RegionDataset,
    metrics: list[dict[str, float]],
    step: int,
    best_metric: float,
    split_summary: dict[str, Any],
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "config": cfg.to_json_dict(),
            "vocab": dataset.vocab.to_dict(),
            "n_structures": dataset.n_structures,
            "structure_names": dataset.structure_names,
            "training_unit": "region",
            "n_regions": len(dataset),
            "max_cells_per_region": cfg.model.max_cells_per_region,
            "metrics": metrics,
            "model_version": _stgpt_version(),
            "panel_metadata": _panel_metadata(cfg),
            "split_summary": split_summary,
            "training_summary": {
                "steps": step,
                "last_metrics": metrics[-1] if metrics else {},
                "best_metric": best_metric,
                "ablation_mode": cfg.training.ablation_mode,
                "lr_schedule": cfg.training.lr_schedule,
                "training_unit": "region",
            },
        },
        path,
    )


def _split_summary(splits) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "counts": {str(key): int(value) for key, value in splits["split"].value_counts().sort_index().items()},
    }
    if "split_group_key" in splits.columns:
        group_key = next((str(value) for value in splits["split_group_key"].dropna().unique()), None)
        summary["group_key"] = group_key
    if "block_id" in splits.columns:
        holdout = splits[splits["split"].astype(str) != "train"]
        summary["holdout_blocks"] = sorted(holdout["block_id"].dropna().astype(str).unique().tolist())[:100]
    return summary


def _panel_metadata(config: StGPTConfig) -> dict[str, Any]:
    panel_genes = config.data.panel_genes or []
    if config.data.panel_gene_file:
        path = config.data.path_or_none(config.data.panel_gene_file)
        if path is not None and path.exists():
            panel_genes = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {
        "panel_gene_count": len(panel_genes),
        "panel_gene_file": config.data.panel_gene_file,
        "gene_name_key": config.data.gene_name_key,
        "slide_id": config.data.slide_id,
        "patient_id": config.data.patient_id,
        "organ": config.data.organ,
        "batch_id": config.data.batch_id,
        "stain": config.data.stain,
        "scanner": config.data.scanner,
    }


def _stgpt_version() -> str:
    try:
        return version("stgpt")
    except PackageNotFoundError:
        return "0.1.0"
