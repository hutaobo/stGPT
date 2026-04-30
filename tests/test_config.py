from __future__ import annotations

from pathlib import Path

from stgpt.config import StGPTConfig


def test_config_env_expansion(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("STGPT_TMP", str(tmp_path))
    config = tmp_path / "config.yaml"
    config.write_text(
        """
case_name: env_case
data:
  mode: synthetic
  output_dir: ${STGPT_TMP}/case
  n_cells: 8
  n_genes: 12
model:
  d_model: 32
  n_heads: 4
  n_layers: 1
  max_genes: 8
  image_size: 32
training:
  batch_size: 4
  max_steps: 2
  output_dir: ${STGPT_TMP}/train
  device: cpu
""",
        encoding="utf-8",
    )
    cfg = StGPTConfig.from_file(config, preset="smoke")
    assert cfg.case_name == "env_case"
    assert str(tmp_path) in cfg.data.output_dir
    assert cfg.training.device == "cpu"
    assert cfg.training.max_steps == 2
    assert cfg.split.strategy == "spatial_block"
    assert cfg.split.train_fraction == 0.70


def test_apply_ablation_sets_modalities_and_losses() -> None:
    cfg = StGPTConfig().apply_ablation("image_gene_spatial")
    assert cfg.training.ablation_mode == "image_gene_spatial"
    assert cfg.model.use_expression_values
    assert cfg.model.use_image_context
    assert cfg.model.use_spatial_context
    assert not cfg.model.use_structure_context
    assert cfg.training.structure_loss_weight == 0.0

    gene_only = StGPTConfig().apply_ablation("gene_only")
    assert gene_only.model.use_expression_values
    assert not gene_only.model.use_image_context
    assert not gene_only.model.use_spatial_context
    assert gene_only.training.image_gene_loss_weight == 0.0
    assert gene_only.training.neighborhood_loss_weight == 0.0
