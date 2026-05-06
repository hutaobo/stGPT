from __future__ import annotations

from pathlib import Path

import torch
from typer.testing import CliRunner

from stgpt.cli import app
from stgpt.training import train


def _write_config(tmp_path: Path) -> Path:
    config = tmp_path / "smoke.yaml"
    config.write_text(
        f"""
case_name: cli_smoke
data:
  mode: synthetic
  output_dir: {tmp_path.as_posix()}/case
  n_cells: 12
  n_genes: 20
  n_structures: 2
  image_size: 32
model:
  d_model: 32
  n_heads: 4
  n_layers: 1
  max_genes: 12
  n_expression_bins: 8
  image_size: 32
training:
  batch_size: 4
  learning_rate: 0.001
  lr_schedule: cosine
  warmup_steps: 1
  max_steps: 2
  save_every_n_steps: 1
  output_dir: {tmp_path.as_posix()}/train
  device: cpu
  num_workers: 0
""",
        encoding="utf-8",
    )
    return config


def test_train_smoke(tmp_path: Path) -> None:
    result = train(_write_config(tmp_path), preset="smoke", max_steps=1)
    assert result["steps"] == 1
    assert Path(result["checkpoint"]).exists()
    assert Path(result["best_checkpoint"]).exists()
    assert "best_alignment_checkpoint" in result
    assert result["metrics"]
    assert "lr" in result["metrics"][-1]
    payload = torch.load(result["checkpoint"], map_location="cpu")
    assert payload["training_unit"] == "region"
    assert payload["n_regions"] > 0
    assert payload["max_cells_per_region"] > 0


def test_cli_doctor_and_train(tmp_path: Path) -> None:
    runner = CliRunner()
    doctor = runner.invoke(app, ["doctor"])
    assert doctor.exit_code == 0
    config = _write_config(tmp_path)
    result = runner.invoke(app, ["train", "--config", str(config), "--preset", "smoke", "--max-steps", "1"])
    assert result.exit_code == 0, result.output
    assert "checkpoint" in result.output


def test_train_ablation_records_config(tmp_path: Path) -> None:
    result = train(_write_config(tmp_path), preset="smoke", max_steps=1, ablation="gene_only")
    checkpoint = Path(result["checkpoint"])
    payload = torch.load(checkpoint, map_location="cpu")
    assert payload["config"]["training"]["ablation_mode"] == "gene_only"
    assert not payload["config"]["model"]["use_image_context"]
    assert payload["config"]["training"]["image_gene_loss_weight"] == 0.0
    assert payload["training_summary"]["lr_schedule"] == "cosine"


def test_train_with_prototype_queue_records_metrics_and_buffers(tmp_path: Path) -> None:
    config = tmp_path / "prototype.yaml"
    config.write_text(
        f"""
case_name: prototype_smoke
data:
  mode: synthetic
  output_dir: {tmp_path.as_posix()}/case_proto
  n_cells: 16
  n_genes: 20
  n_structures: 2
  image_size: 32
model:
  d_model: 32
  n_heads: 4
  n_layers: 1
  max_genes: 12
  n_expression_bins: 8
  image_size: 32
  n_prototypes: 3
  prototype_temperature: 0.1
training:
  batch_size: 4
  learning_rate: 0.001
  max_steps: 2
  output_dir: {tmp_path.as_posix()}/train_proto
  device: cpu
  num_workers: 0
  prototype_loss_weight: 0.1
  prototype_queue_size: 8
  prototype_sinkhorn_iterations: 3
""",
        encoding="utf-8",
    )

    result = train(config, preset="smoke", max_steps=2)
    metrics = result["metrics"]
    assert "prototype_loss" in metrics[-1]
    assert "prototype_entropy" in metrics[-1]
    assert "sinkhorn_row_residual" in metrics[-1]
    checkpoint = torch.load(result["checkpoint"], map_location="cpu")
    state = checkpoint["model_state"]
    assert "prototype_queue.queue" in state
    assert int(state["prototype_queue.queue_filled"]) > 0


def test_train_best_alignment_checkpoint(tmp_path: Path) -> None:
    config = tmp_path / "alignment.yaml"
    config.write_text(
        f"""
case_name: alignment_smoke
data:
  mode: synthetic
  output_dir: {tmp_path.as_posix()}/case_align
  n_cells: 20
  n_genes: 20
  n_structures: 2
  image_size: 32
model:
  d_model: 32
  n_heads: 4
  n_layers: 1
  max_genes: 12
  n_expression_bins: 8
  image_size: 32
training:
  batch_size: 4
  learning_rate: 0.001
  max_steps: 2
  save_every_n_steps: 1
  image_gene_loss_weight: 0.1
  output_dir: {tmp_path.as_posix()}/train_align
  device: cpu
  num_workers: 0
split:
  strategy: spatial_block
  train_fraction: 0.70
  val_fraction: 0.15
  test_fraction: 0.15
""",
        encoding="utf-8",
    )
    result = train(config, preset="smoke", max_steps=2)
    assert "best_alignment_checkpoint" in result
    alignment_path = Path(result["best_alignment_checkpoint"])
    assert alignment_path.exists(), "best_alignment.pt should be saved when val_image_gene_loss is tracked"
    payload = torch.load(alignment_path, map_location="cpu")
    assert "model_state" in payload
    assert "config" in payload
