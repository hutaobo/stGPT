from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from stgpt.cli import app
from stgpt.config import DataConfig, ModelConfig, SplitConfig, StGPTConfig, TrainingConfig
from stgpt.data import make_synthetic_case
from stgpt.qc import make_splits, validate_data, validate_training_case


def _config(tmp_path: Path, *, include_structure_context: bool = False) -> StGPTConfig:
    return StGPTConfig(
        case_name="qc_case",
        data=DataConfig(
            mode="synthetic",
            output_dir=str(tmp_path / "case"),
            n_cells=18,
            n_genes=24,
            n_structures=3,
            image_size=32,
            include_structure_context=include_structure_context,
            seed=7,
        ),
        model=ModelConfig(d_model=32, n_heads=4, n_layers=1, max_genes=12, image_size=32, n_expression_bins=8),
        training=TrainingConfig(batch_size=4, max_steps=1, output_dir=str(tmp_path / "train"), device="cpu", seed=11),
        split=SplitConfig(seed=13),
    )


def _write_config(tmp_path: Path) -> Path:
    config = tmp_path / "qc.yaml"
    config.write_text(
        f"""
case_name: qc_cli
data:
  mode: synthetic
  output_dir: {tmp_path.as_posix()}/case
  n_cells: 14
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
  max_steps: 1
  output_dir: {tmp_path.as_posix()}/train
  device: cpu
split:
  strategy: spatial_block
  train_fraction: 0.70
  val_fraction: 0.15
  test_fraction: 0.15
  seed: 5
""",
        encoding="utf-8",
    )
    return config


def test_validate_data_writes_qc_artifacts(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    result = validate_data(cfg, output_dir=tmp_path / "qc")
    assert result["status"] == "pass"
    for key in ("case_manifest", "qc_report_json", "qc_report_md", "splits"):
        assert Path(result[key]).exists()
    manifest = json.loads(Path(result["case_manifest"]).read_text(encoding="utf-8"))
    report = json.loads(Path(result["qc_report_json"]).read_text(encoding="utf-8"))
    splits = pd.read_csv(result["splits"])
    assert manifest["n_cells"] == 18
    assert manifest["patch_count"] == 18
    assert report["metrics"]["patch_cell_coverage"] == 1.0
    assert len(splits) == 18
    assert set(splits["split"]) == {"train", "val", "test"}


def test_missing_patch_paths_are_reported(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    case = make_synthetic_case(cfg.data)
    case.patch_table.loc[0, "image_path"] = str(tmp_path / "missing.png")
    result = validate_training_case(case, cfg, output_dir=tmp_path / "qc_missing_patch")
    report = json.loads(Path(result["qc_report_json"]).read_text(encoding="utf-8"))
    assert result["status"] == "pass"
    assert report["metrics"]["missing_image_count"] == 1
    assert any("missing image" in item for item in report["warnings"])


def test_duplicate_gene_names_are_reported(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    case = make_synthetic_case(cfg.data)
    gene_key = cfg.data.gene_name_key
    case.adata.var.loc[case.adata.var.index[1], gene_key] = str(case.adata.var.iloc[0][gene_key])
    result = validate_training_case(case, cfg, output_dir=tmp_path / "qc_duplicate_genes")
    report = json.loads(Path(result["qc_report_json"]).read_text(encoding="utf-8"))
    assert report["metrics"]["duplicate_gene_count"] == 1
    assert any("duplicate gene" in item for item in report["warnings"])


def test_structure_context_coverage_is_reported(tmp_path: Path) -> None:
    cfg = _config(tmp_path, include_structure_context=True)
    case = make_synthetic_case(cfg.data)
    del case.adata.obs[cfg.data.structure_key]
    result = validate_training_case(case, cfg, output_dir=tmp_path / "qc_structure_context")
    report = json.loads(Path(result["qc_report_json"]).read_text(encoding="utf-8"))
    assert report["metrics"]["structure_assignment_coverage"] == 0.0
    assert any("Structure context is enabled" in item for item in report["warnings"])


def test_spatial_block_splits_are_deterministic(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    case = make_synthetic_case(cfg.data)
    first = make_splits(case, cfg)
    second = make_splits(case, cfg)
    pd.testing.assert_frame_equal(first, second)


def test_cli_validate_data_writes_artifacts(tmp_path: Path) -> None:
    runner = CliRunner()
    output_dir = tmp_path / "qc_cli"
    result = runner.invoke(app, ["validate-data", "--config", str(_write_config(tmp_path)), "--output", str(output_dir)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "pass"
    assert Path(payload["case_manifest"]).exists()
    assert Path(payload["qc_report_json"]).exists()
    assert Path(payload["qc_report_md"]).exists()
    assert Path(payload["splits"]).exists()
