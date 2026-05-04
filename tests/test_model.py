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
        object_image=batch["object_image"],
        context_image=batch["context_image"],
        contour_mask=batch["contour_mask"],
        contour_geometry=batch["contour_geometry"],
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
        object_image=batch["object_image"],
        context_image=batch["context_image"],
        contour_mask=batch["contour_mask"],
        contour_geometry=batch["contour_geometry"],
    )
    assert output.gene_pred.shape == batch["target_values"].shape
    assert torch.allclose(output.image_emb, torch.zeros_like(output.image_emb))


def test_model_accepts_contour_evidence_tokens() -> None:
    model = ImageGeneSTGPT(n_genes=12, n_structures=2, d_model=32, n_heads=4, n_layers=1, n_expression_bins=8)
    batch_size = 3
    seq_len = 6
    gene_ids = torch.randint(1, 12, (batch_size, seq_len))
    expr_values = torch.rand(batch_size, seq_len)
    expr_bins = torch.randint(0, 8, (batch_size, seq_len))
    image = torch.rand(batch_size, 3, 32, 32)
    context = torch.rand(batch_size, 3, 32, 32)
    mask = torch.ones(batch_size, 1, 32, 32)
    geometry = torch.rand(batch_size, 5)

    output = model(
        gene_ids=gene_ids,
        expr_values=expr_values,
        expr_bins=expr_bins,
        image=image,
        object_image=image,
        context_image=context,
        contour_mask=mask,
        contour_geometry=geometry,
        spatial=torch.rand(batch_size, 2),
        context_ids=torch.ones(batch_size, dtype=torch.long),
    )

    assert output.gene_pred.shape == (batch_size, seq_len)
    assert output.neighbor_pred.shape == (batch_size, seq_len)
    assert output.region_emb.shape == (batch_size, 32)
    assert output.image_emb.shape == (batch_size, 32)
    assert output.structure_logits is not None


def test_model_outputs_prototype_assignments_when_enabled() -> None:
    model = ImageGeneSTGPT(n_genes=12, d_model=32, n_heads=4, n_layers=1, n_expression_bins=8, n_prototypes=4)
    batch_size = 3
    seq_len = 6

    output = model(
        gene_ids=torch.randint(1, 12, (batch_size, seq_len)),
        expr_values=torch.rand(batch_size, seq_len),
        expr_bins=torch.randint(0, 8, (batch_size, seq_len)),
        image=torch.rand(batch_size, 3, 32, 32),
        spatial=torch.rand(batch_size, 2),
    )

    assert output.prototype_logits is not None and output.prototype_logits.shape == (batch_size, 4)
    assert output.prototype_probs is not None and output.prototype_probs.shape == (batch_size, 4)
    assert output.prototype_ids is not None and output.prototype_ids.shape == (batch_size,)
    assert output.prototype_confidence is not None and output.prototype_confidence.shape == (batch_size,)
    assert torch.allclose(output.prototype_probs.sum(dim=1), torch.ones(batch_size), atol=1e-6)


def test_model_image_only_fallback_still_works() -> None:
    model = ImageGeneSTGPT(n_genes=10, d_model=32, n_heads=4, n_layers=1, n_expression_bins=8)
    output = model(
        gene_ids=torch.randint(1, 10, (2, 5)),
        expr_values=torch.rand(2, 5),
        expr_bins=torch.randint(0, 8, (2, 5)),
        image=torch.rand(2, 3, 32, 32),
        spatial=torch.rand(2, 2),
    )

    assert output.gene_pred.shape == (2, 5)
    assert output.image_emb.shape == (2, 32)


def test_gated_midfusion_is_zero_initialized() -> None:
    model = ImageGeneSTGPT(n_genes=10, d_model=32, n_heads=4, n_layers=2, n_expression_bins=8)
    model.eval()
    assert torch.equal(model.gated_fusion.attn_gate.detach(), torch.zeros_like(model.gated_fusion.attn_gate))
    assert torch.equal(model.gated_fusion.ffn_gate.detach(), torch.zeros_like(model.gated_fusion.ffn_gate))

    gene_ids = torch.randint(1, 10, (2, 5))
    expr_values = torch.rand(2, 5)
    expr_bins = torch.randint(0, 8, (2, 5))
    image = torch.zeros(2, 3, 32, 32)
    spatial = torch.rand(2, 2)
    context_ids = torch.ones(2, dtype=torch.long)

    with torch.no_grad():
        first = model(
            gene_ids=gene_ids,
            expr_values=expr_values,
            expr_bins=expr_bins,
            image=image,
            object_image=torch.zeros_like(image),
            context_image=torch.zeros_like(image),
            contour_mask=torch.ones(2, 1, 32, 32),
            contour_geometry=torch.zeros(2, 4),
            spatial=spatial,
            context_ids=context_ids,
        )
        second = model(
            gene_ids=gene_ids,
            expr_values=expr_values,
            expr_bins=expr_bins,
            image=image,
            object_image=torch.ones_like(image),
            context_image=torch.ones_like(image),
            contour_mask=torch.ones(2, 1, 32, 32),
            contour_geometry=torch.ones(2, 4),
            spatial=spatial,
            context_ids=context_ids,
        )

    assert torch.allclose(first.gene_pred, second.gene_pred, atol=1e-6)
    assert torch.allclose(first.region_emb, second.region_emb, atol=1e-6)
