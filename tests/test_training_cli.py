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
  max_steps: 2
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
    assert result["metrics"]


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
