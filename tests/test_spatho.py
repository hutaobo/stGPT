from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from stgpt.cli import app
from stgpt.config import DataConfig, ModelConfig, SplitConfig, StGPTConfig, TrainingConfig
from stgpt.evaluation import evaluate
from stgpt.qc import validate_data
from stgpt.spatho import SpathoStGPTModel, embed_spatho_case, package_model
from stgpt.training import train


def _config(tmp_path: Path) -> StGPTConfig:
    return StGPTConfig(
        case_name="spatho_case",
        data=DataConfig(
            mode="synthetic",
            output_dir=str(tmp_path / "case"),
            n_cells=16,
            n_genes=22,
            n_structures=2,
            image_size=32,
            seed=12,
        ),
        model=ModelConfig(d_model=32, n_heads=4, n_layers=1, max_genes=12, image_size=32, n_expression_bins=8, patch_scales=[1, 2]),
        training=TrainingConfig(batch_size=4, max_steps=1, output_dir=str(tmp_path / "train"), device="cpu", seed=6),
        split=SplitConfig(seed=8),
    )


def _write_config(path: Path, cfg: StGPTConfig) -> Path:
    path.write_text(
        f"""
case_name: {cfg.case_name}
data:
  mode: synthetic
  output_dir: {cfg.data.output_dir}
  n_cells: {cfg.data.n_cells}
  n_genes: {cfg.data.n_genes}
  n_structures: {cfg.data.n_structures}
  image_size: {cfg.data.image_size}
  seed: {cfg.data.seed}
model:
  d_model: {cfg.model.d_model}
  n_heads: {cfg.model.n_heads}
  n_layers: {cfg.model.n_layers}
  max_genes: {cfg.model.max_genes}
  n_expression_bins: {cfg.model.n_expression_bins}
  image_size: {cfg.model.image_size}
  patch_scales: [1, 2]
training:
  batch_size: {cfg.training.batch_size}
  max_steps: {cfg.training.max_steps}
  output_dir: {cfg.training.output_dir}
  device: cpu
  seed: {cfg.training.seed}
split:
  strategy: spatial_block
  train_fraction: {cfg.split.train_fraction}
  val_fraction: {cfg.split.val_fraction}
  test_fraction: {cfg.split.test_fraction}
  seed: {cfg.split.seed}
""",
        encoding="utf-8",
    )
    return path


def _trained_model(tmp_path: Path) -> tuple[StGPTConfig, Path, Path]:
    cfg = _config(tmp_path)
    checkpoint = Path(train(cfg, preset="smoke", max_steps=1)["checkpoint"])
    qc = validate_data(cfg, output_dir=tmp_path / "qc")
    evaluation = evaluate(
        checkpoint=checkpoint,
        config=cfg,
        splits=qc["splits"],
        output_dir=tmp_path / "eval",
        batch_size=4,
        device="cpu",
    )
    return cfg, checkpoint, Path(evaluation["artifacts"]["evaluation_metrics"])


def test_spatho_module_does_not_require_spatho_package() -> None:
    assert SpathoStGPTModel is not None


def test_embed_spatho_case_writes_standard_artifacts(tmp_path: Path) -> None:
    cfg, checkpoint, _ = _trained_model(tmp_path)
    outputs = embed_spatho_case(cfg, checkpoint, tmp_path / "spatho_out", batch_size=4, device="cpu")
    for key in ("cell_embeddings", "structure_embedding_summary", "patch_embedding_summary", "stgpt_spatho_manifest", "qc_report"):
        assert key in outputs
        assert Path(outputs[key]).exists()
    frame = pd.read_parquet(outputs["cell_embeddings"])
    assert "cell_id" in frame.columns
    assert "cluster" in frame.columns
    assert "structure_id" in frame.columns
    assert "patch_id" in frame.columns
    assert "stgpt_embedding_source" in frame.columns
    assert any(str(col).startswith("emb_") for col in frame.columns)
    manifest = json.loads(Path(outputs["stgpt_spatho_manifest"]).read_text(encoding="utf-8"))
    assert manifest["n_cells"] == len(frame)
    assert manifest["spatho_runtime_dependency"] == "optional"


def test_package_model_and_load_from_model_dir(tmp_path: Path) -> None:
    cfg, checkpoint, evaluation = _trained_model(tmp_path)
    model_dir = tmp_path / "model_package"
    package = package_model(checkpoint=checkpoint, evaluation=evaluation, output_dir=model_dir, model_name="synthetic-stgpt")
    for key in ("checkpoint", "evaluation_metrics", "manifest", "config", "vocab", "model_card"):
        assert Path(package[key]).exists()
    direct = SpathoStGPTModel.from_checkpoint(checkpoint, device="cpu")
    packaged = SpathoStGPTModel.from_checkpoint(model_dir, device="cpu")
    assert direct.config.case_name == cfg.case_name
    assert packaged.config.case_name == cfg.case_name


def test_cli_package_model_and_spatho_embed(tmp_path: Path) -> None:
    cfg, checkpoint, evaluation = _trained_model(tmp_path)
    config_path = _write_config(tmp_path / "spatho.yaml", cfg)
    runner = CliRunner()
    model_dir = tmp_path / "cli_model"
    packaged = runner.invoke(
        app,
        [
            "package-model",
            "--checkpoint",
            str(checkpoint),
            "--eval",
            str(evaluation),
            "--output",
            str(model_dir),
            "--model-name",
            "cli-synthetic-stgpt",
        ],
    )
    assert packaged.exit_code == 0, packaged.output
    embedded = runner.invoke(
        app,
        [
            "spatho-embed",
            "--model",
            str(model_dir),
            "--config",
            str(config_path),
            "--output",
            str(tmp_path / "cli_spatho"),
            "--batch-size",
            "4",
            "--device",
            "cpu",
        ],
    )
    assert embedded.exit_code == 0, embedded.output
    payload = json.loads(embedded.output)
    assert Path(payload["cell_embeddings"]).exists()
