from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from stgpt.cli import app
from stgpt.config import DataConfig, ModelConfig, SplitConfig, StGPTConfig, TrainingConfig
from stgpt.data import make_synthetic_case
from stgpt.evaluation import evaluate
from stgpt.qc import validate_data
from stgpt.training import train


def _config(tmp_path: Path, *, mode: str = "synthetic", input_h5ad: Path | None = None) -> StGPTConfig:
    return StGPTConfig(
        case_name="eval_case",
        data=DataConfig(
            mode=mode,
            input_h5ad=str(input_h5ad) if input_h5ad is not None else None,
            output_dir=str(tmp_path / f"{mode}_case"),
            n_cells=18,
            n_genes=24,
            n_structures=3,
            image_size=32,
            seed=9,
        ),
        model=ModelConfig(d_model=32, n_heads=4, n_layers=1, max_genes=12, image_size=32, n_expression_bins=8),
        training=TrainingConfig(batch_size=6, max_steps=1, output_dir=str(tmp_path / "train"), device="cpu", seed=3),
        split=SplitConfig(seed=4),
    )


def _write_config(path: Path, cfg: StGPTConfig) -> Path:
    path.write_text(
        f"""
case_name: {cfg.case_name}
data:
  mode: {cfg.data.mode}
  output_dir: {cfg.data.output_dir}
  input_h5ad: {cfg.data.input_h5ad or ""}
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


def _checkpoint_and_splits(tmp_path: Path) -> tuple[StGPTConfig, Path, Path]:
    cfg = _config(tmp_path)
    checkpoint = Path(train(cfg, preset="smoke", max_steps=1)["checkpoint"])
    qc = validate_data(cfg, output_dir=tmp_path / "qc")
    return cfg, checkpoint, Path(qc["splits"])


def test_evaluate_synthetic_checkpoint_writes_artifacts(tmp_path: Path) -> None:
    cfg, checkpoint, splits = _checkpoint_and_splits(tmp_path)
    result = evaluate(checkpoint=checkpoint, config=cfg, splits=splits, output_dir=tmp_path / "eval", batch_size=6, device="cpu")
    for key in ("evaluation_metrics", "prediction_summary", "retrieval_metrics", "embedding_qc", "failure_analysis"):
        assert Path(result["artifacts"][key]).exists()
    prediction = pd.read_csv(result["artifacts"]["prediction_summary"])
    retrieval = pd.read_csv(result["artifacts"]["retrieval_metrics"])
    embedding_qc = pd.read_csv(result["artifacts"]["embedding_qc"])
    failure_analysis = pd.read_csv(result["artifacts"]["failure_analysis"])
    assert {"overall", "train", "val", "test"}.issubset(set(prediction["split"]))
    assert {"image_to_gene_topk", "gene_to_image_topk"}.issubset(retrieval.columns)
    assert {"cluster", "structure_id"}.issubset(set(embedding_qc["label_column"]))
    assert {"patch", "registration", "split"}.issubset(set(failure_analysis["category"]))
    metrics = json.loads(Path(result["artifacts"]["evaluation_metrics"]).read_text(encoding="utf-8"))
    assert metrics["overall_prediction"]["n_masked_gene_values"] > 0
    assert metrics["n_failure_analysis_rows"] > 0


def test_cli_evaluate_writes_artifacts(tmp_path: Path) -> None:
    cfg, checkpoint, splits = _checkpoint_and_splits(tmp_path)
    config_path = _write_config(tmp_path / "eval.yaml", cfg)
    output_dir = tmp_path / "cli_eval"
    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--checkpoint",
            str(checkpoint),
            "--config",
            str(config_path),
            "--splits",
            str(splits),
            "--output",
            str(output_dir),
            "--batch-size",
            "6",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert Path(payload["artifacts"]["evaluation_metrics"]).exists()


def test_evaluate_requires_existing_splits(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    checkpoint = Path(train(cfg, preset="smoke", max_steps=1)["checkpoint"])
    with pytest.raises(FileNotFoundError):
        evaluate(checkpoint=checkpoint, config=cfg, splits=tmp_path / "missing_splits.csv", output_dir=tmp_path / "eval", device="cpu")


def test_evaluate_without_labels_writes_empty_embedding_qc(tmp_path: Path) -> None:
    cfg, checkpoint, _ = _checkpoint_and_splits(tmp_path)
    case = make_synthetic_case(cfg.data)
    for column in (cfg.data.cluster_key, cfg.data.structure_key):
        if column in case.adata.obs:
            del case.adata.obs[column]
    input_h5ad = tmp_path / "no_labels.h5ad"
    case.adata.write_h5ad(input_h5ad)

    eval_cfg = _config(tmp_path, mode="anndata", input_h5ad=input_h5ad)
    qc = validate_data(eval_cfg, output_dir=tmp_path / "no_labels_qc")
    result = evaluate(
        checkpoint=checkpoint,
        config=eval_cfg,
        splits=qc["splits"],
        output_dir=tmp_path / "no_labels_eval",
        batch_size=6,
        device="cpu",
    )
    embedding_qc = pd.read_csv(result["artifacts"]["embedding_qc"])
    assert embedding_qc.empty
