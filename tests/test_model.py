from __future__ import annotations

from pathlib import Path

import torch

from stgpt.config import DataConfig, ModelConfig, StGPTConfig, TrainingConfig
from stgpt.data import ImageGeneDataset, make_synthetic_case
from stgpt.losses import compute_losses
from stgpt.models import ImageGeneSTGPT


def test_model_forward_and_optimizer_step(tmp_path: Path) -> None:
    cfg = StGPTConfig(
        case_name="model",
        data=DataConfig(mode="synthetic", output_dir=str(tmp_path / "case"), n_cells=10, n_genes=18, image_size=32),
        model=ModelConfig(d_model=32, n_heads=4, n_layers=1, max_genes=10, image_size=32, n_expression_bins=8),
        training=TrainingConfig(batch_size=5, max_steps=1, output_dir=str(tmp_path / "train"), device="cpu"),
    )
    dataset = ImageGeneDataset(make_synthetic_case(cfg.data), cfg)
    batch = dataset.collate([dataset[i] for i in range(5)])
    model = ImageGeneSTGPT(
        n_genes=dataset.vocab.size - 1,
        n_structures=dataset.n_structures,
        d_model=32,
        n_heads=4,
        n_layers=1,
        n_expression_bins=8,
    )
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
    assert output.gene_pred.shape == batch["target_values"].shape
    assert output.region_emb.shape == (5, 32)
    assert torch.equal(output.cell_emb, output.region_emb)
    losses = compute_losses(output, batch, image_gene_weight=0.1, neighborhood_weight=0.25, structure_weight=0.1)
    assert torch.isfinite(losses["loss"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    losses["loss"].backward()
    optimizer.step()


def test_model_multiscale_and_disabled_modalities(tmp_path: Path) -> None:
    cfg = StGPTConfig(
        case_name="model_ablation",
        data=DataConfig(mode="synthetic", output_dir=str(tmp_path / "case"), n_cells=8, n_genes=16, image_size=32),
        model=ModelConfig(
            d_model=32,
            n_heads=4,
            n_layers=1,
            max_genes=8,
            image_size=32,
            n_expression_bins=8,
            patch_scales=[1, 2],
            use_expression_values=False,
            use_image_context=False,
            use_spatial_context=False,
            use_structure_context=False,
        ),
        training=TrainingConfig(batch_size=4, max_steps=1, output_dir=str(tmp_path / "train"), device="cpu"),
    )
    dataset = ImageGeneDataset(make_synthetic_case(cfg.data), cfg)
    batch = dataset.collate([dataset[i] for i in range(4)])
    model = ImageGeneSTGPT(
        n_genes=dataset.vocab.size - 1,
        n_structures=dataset.n_structures,
        d_model=32,
        n_heads=4,
        n_layers=1,
        n_expression_bins=8,
        patch_scales=[1, 2],
        use_expression_values=False,
        use_image_context=False,
        use_spatial_context=False,
        use_structure_context=False,
    )
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
    assert output.gene_pred.shape == batch["target_values"].shape
    assert torch.allclose(output.image_emb, torch.zeros_like(output.image_emb))
