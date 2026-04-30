from __future__ import annotations

from pathlib import Path

import torch

from stgpt.config import DataConfig, ModelConfig, StGPTConfig, TrainingConfig
from stgpt.data import ImageGeneDataset, build_training_manifest, make_synthetic_case


def _config(tmp_path: Path) -> StGPTConfig:
    return StGPTConfig(
        case_name="test",
        data=DataConfig(
            mode="synthetic",
            output_dir=str(tmp_path / "case"),
            n_cells=12,
            n_genes=20,
            n_structures=3,
            image_size=32,
            seed=2,
        ),
        model=ModelConfig(d_model=32, n_heads=4, n_layers=1, max_genes=12, image_size=32, n_expression_bins=8),
        training=TrainingConfig(batch_size=4, max_steps=1, output_dir=str(tmp_path / "train"), device="cpu"),
    )


def test_synthetic_case_and_manifest(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    case = make_synthetic_case(cfg.data)
    assert case.adata.n_obs == 12
    assert case.adata.n_vars == 20
    assert "spatial" in case.adata.obsm
    assert len(case.patch_table) == 12
    manifest = build_training_manifest(cfg)
    assert manifest["n_cells"] == 12
    assert Path(manifest["patch_table"]).exists()


def test_dataset_collate_masks_and_images(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    dataset = ImageGeneDataset(make_synthetic_case(cfg.data), cfg)
    batch = dataset.collate([dataset[0], dataset[1], dataset[2]])
    assert batch["gene_ids"].shape == (3, 12)
    assert batch["image"].shape == (3, 3, 32, 32)
    assert batch["mask"].any()
    assert batch["gene_padding_mask"].dtype == torch.bool
