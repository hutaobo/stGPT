"""Tests for the stgpt.spatho adapter module.

Covers:
- I/O contract dataclasses (PatchManifestRow, SpathoExportResult)
- run_spatho_export(): artifact creation and schema validation
- CLI export-spatho command: correct invocation and output parsing
- ImageGeneSTGPT.from_pretrained() and load_checkpoint() classmethods
- Full Phase-1 pipeline integration: validate-data → train → evaluate → export-spatho
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

import stgpt
from stgpt.cli import app
from stgpt.config import DataConfig, ModelConfig, SplitConfig, StGPTConfig, TrainingConfig
from stgpt.models import ImageGeneSTGPT
from stgpt.qc import validate_data
from stgpt.spatho import (
    CELL_EMBEDDING_REQUIRED_COLUMNS,
    REGION_EMBEDDING_REQUIRED_COLUMNS,
    STRUCTURE_SUMMARY_REQUIRED_COLUMNS,
    PatchManifestRow,
    SpathoExportResult,
    run_spatho_export,
)
from stgpt.training import train

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _small_config(tmp_path: Path) -> StGPTConfig:
    """Return a minimal synthetic config suitable for fast unit tests."""
    return StGPTConfig(
        case_name="spatho_test",
        data=DataConfig(
            mode="synthetic",
            output_dir=str(tmp_path / "case"),
            n_cells=16,
            n_genes=20,
            n_structures=2,
            image_size=32,
            seed=42,
        ),
        model=ModelConfig(d_model=32, n_heads=4, n_layers=1, max_genes=12, image_size=32, n_expression_bins=8),
        training=TrainingConfig(batch_size=4, max_steps=1, output_dir=str(tmp_path / "train"), device="cpu", seed=7),
        split=SplitConfig(seed=3),
    )


def _write_config(path: Path, cfg: StGPTConfig) -> Path:
    path.write_text(
        f"""
case_name: {cfg.case_name}
data:
  mode: {cfg.data.mode}
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


def _checkpoint(tmp_path: Path, cfg: StGPTConfig) -> Path:
    return Path(train(cfg, preset="smoke", max_steps=1)["checkpoint"])


# ---------------------------------------------------------------------------
# Phase 1 – end-to-end pipeline integration test
# ---------------------------------------------------------------------------


def test_full_pipeline_validate_train_evaluate_export(tmp_path: Path) -> None:
    """End-to-end: validate-data → train → evaluate → export-spatho all succeed."""
    cfg = _small_config(tmp_path)

    # Step 1 – validate-data
    qc = validate_data(cfg, output_dir=tmp_path / "qc")
    assert qc["status"] == "pass"
    for artifact in ("case_manifest", "qc_report_json", "qc_report_md", "splits"):
        assert Path(qc[artifact]).exists(), f"Missing QC artifact: {artifact}"

    # Step 2 – train
    checkpoint = _checkpoint(tmp_path, cfg)
    assert checkpoint.exists()

    # Step 3 – evaluate
    from stgpt.evaluation import evaluate

    eval_result = evaluate(
        checkpoint=checkpoint,
        config=cfg,
        splits=qc["splits"],
        output_dir=tmp_path / "eval",
        batch_size=4,
        device="cpu",
    )
    for artifact in ("evaluation_metrics", "prediction_summary", "retrieval_metrics", "embedding_qc"):
        assert Path(eval_result["artifacts"][artifact]).exists(), f"Missing eval artifact: {artifact}"

    # Step 4 – export-spatho
    export_result = run_spatho_export(cfg, checkpoint=checkpoint, output_dir=tmp_path / "spatho_out", batch_size=4, device="cpu")
    assert export_result.cell_embeddings.exists()
    assert export_result.region_embeddings is not None and export_result.region_embeddings.exists()
    assert export_result.region_cell_membership is not None and export_result.region_cell_membership.exists()
    assert export_result.region_molecular_summary is not None and export_result.region_molecular_summary.exists()
    assert export_result.region_image_manifest is not None and export_result.region_image_manifest.exists()
    assert export_result.region_qc_report is not None and export_result.region_qc_report.exists()
    assert export_result.evidence_manifest is not None and export_result.evidence_manifest.exists()
    assert export_result.structure_summary.exists()
    assert export_result.qc_report.exists()


# ---------------------------------------------------------------------------
# Phase 2 – I/O contract dataclasses
# ---------------------------------------------------------------------------


def test_patch_manifest_row_is_immutable() -> None:
    row = PatchManifestRow(
        cell_id="c001",
        contour_id="k42",
        structure_id=1,
        structure_name="tumor",
        image_path="/data/patches/c001.png",
        x_px=1024.5,
        y_px=2048.0,
        patch_size_px=224,
    )
    assert row.cell_id == "c001"
    assert row.patch_size_px == 224
    with pytest.raises((AttributeError, TypeError)):  # frozen dataclass raises on attribute assignment
        row.cell_id = "other"  # type: ignore[misc]


def test_patch_manifest_row_all_none() -> None:
    row = PatchManifestRow(
        cell_id=None, contour_id=None, structure_id=None, structure_name=None, image_path=None, x_px=None, y_px=None, patch_size_px=None
    )
    assert row.cell_id is None


def test_spatho_export_result_to_dict(tmp_path: Path) -> None:
    result = SpathoExportResult(
        cell_embeddings=tmp_path / "cell_embeddings.parquet",
        structure_summary=tmp_path / "structure_summary.parquet",
        qc_report=tmp_path / "qc_report.json",
        n_cells=100,
        n_cells_with_image=80,
        embedding_dim=32,
    )
    payload = result.to_dict()
    assert payload["n_cells"] == 100
    assert payload["embedding_dim"] == 32
    assert str(tmp_path) in payload["cell_embeddings"]


def test_cell_embedding_required_columns_constant() -> None:
    assert "cell_id" in CELL_EMBEDDING_REQUIRED_COLUMNS
    assert "x" in CELL_EMBEDDING_REQUIRED_COLUMNS
    assert "y" in CELL_EMBEDDING_REQUIRED_COLUMNS
    assert "structure_label" in CELL_EMBEDDING_REQUIRED_COLUMNS
    assert "qc_flag" in CELL_EMBEDDING_REQUIRED_COLUMNS


def test_region_embedding_required_columns_constant() -> None:
    assert "region_id" in REGION_EMBEDDING_REQUIRED_COLUMNS
    assert "structure_label" in REGION_EMBEDDING_REQUIRED_COLUMNS
    assert "n_cells" in REGION_EMBEDDING_REQUIRED_COLUMNS
    assert "qc_flag" in REGION_EMBEDDING_REQUIRED_COLUMNS


def test_structure_summary_required_columns_constant() -> None:
    assert "structure_label" in STRUCTURE_SUMMARY_REQUIRED_COLUMNS
    assert "n_cells" in STRUCTURE_SUMMARY_REQUIRED_COLUMNS


# ---------------------------------------------------------------------------
# Phase 3 – run_spatho_export() and CLI
# ---------------------------------------------------------------------------


def test_run_spatho_export_writes_all_artifacts(tmp_path: Path) -> None:
    cfg = _small_config(tmp_path)
    checkpoint = _checkpoint(tmp_path, cfg)
    result = run_spatho_export(cfg, checkpoint=checkpoint, output_dir=tmp_path / "out", batch_size=4, device="cpu")

    assert result.cell_embeddings.exists()
    assert result.region_embeddings is not None and result.region_embeddings.exists()
    assert result.region_cell_membership is not None and result.region_cell_membership.exists()
    assert result.region_molecular_summary is not None and result.region_molecular_summary.exists()
    assert result.region_image_manifest is not None and result.region_image_manifest.exists()
    assert result.evidence_manifest is not None and result.evidence_manifest.exists()
    assert result.structure_summary.exists()
    assert result.qc_report.exists()


def test_run_spatho_export_cell_embeddings_schema(tmp_path: Path) -> None:
    cfg = _small_config(tmp_path)
    checkpoint = _checkpoint(tmp_path, cfg)
    result = run_spatho_export(cfg, checkpoint=checkpoint, output_dir=tmp_path / "out", batch_size=4, device="cpu")

    frame = pd.read_parquet(result.cell_embeddings)
    for col in REGION_EMBEDDING_REQUIRED_COLUMNS:
        assert col in frame.columns, f"Required column '{col}' missing from region_embeddings.parquet"
    emb_cols = [col for col in frame.columns if str(col).startswith("emb_")]
    assert len(emb_cols) > 0, "No emb_* columns found in region_embeddings.parquet"
    assert len(frame) == result.n_cells


def test_run_spatho_export_structure_summary_schema(tmp_path: Path) -> None:
    cfg = _small_config(tmp_path)
    checkpoint = _checkpoint(tmp_path, cfg)
    result = run_spatho_export(cfg, checkpoint=checkpoint, output_dir=tmp_path / "out", batch_size=4, device="cpu")

    summary = pd.read_parquet(result.structure_summary)
    for col in STRUCTURE_SUMMARY_REQUIRED_COLUMNS:
        assert col in summary.columns, f"Required column '{col}' missing from structure_summary.parquet"
    emb_cols = [col for col in summary.columns if str(col).startswith("emb_")]
    assert len(emb_cols) > 0, "No emb_* columns in structure_summary.parquet"
    assert summary["n_cells"].sum() == cfg.data.n_cells


def test_run_spatho_export_qc_report_content(tmp_path: Path) -> None:
    cfg = _small_config(tmp_path)
    checkpoint = _checkpoint(tmp_path, cfg)
    result = run_spatho_export(cfg, checkpoint=checkpoint, output_dir=tmp_path / "out", batch_size=4, device="cpu")

    payload = json.loads(result.qc_report.read_text(encoding="utf-8"))
    assert payload["training_unit"] == "region"
    assert payload["n_regions_total"] == result.n_cells
    assert payload["n_regions_with_image"] + payload["n_regions_no_image"] == payload["n_regions_total"]
    assert "image_coverage" in payload
    assert "structure_counts" in payload
    evidence = json.loads(result.evidence_manifest.read_text(encoding="utf-8")) if result.evidence_manifest else {}
    assert evidence["training_unit"] == "region"
    assert "region_embeddings" in evidence["artifacts"]


def test_run_spatho_export_result_statistics(tmp_path: Path) -> None:
    cfg = _small_config(tmp_path)
    checkpoint = _checkpoint(tmp_path, cfg)
    result = run_spatho_export(cfg, checkpoint=checkpoint, output_dir=tmp_path / "out", batch_size=4, device="cpu")

    assert result.n_cells > 0
    assert result.n_cells_with_image <= result.n_cells
    assert result.embedding_dim > 0


def test_run_spatho_export_accepts_config_path(tmp_path: Path) -> None:
    cfg = _small_config(tmp_path)
    checkpoint = _checkpoint(tmp_path, cfg)
    config_path = _write_config(tmp_path / "cfg.yaml", cfg)

    result = run_spatho_export(config_path, checkpoint=checkpoint, output_dir=tmp_path / "out", batch_size=4, device="cpu")
    assert result.cell_embeddings.exists()


def test_run_spatho_export_qc_flags_synthetic_all_ok(tmp_path: Path) -> None:
    """Synthetic case writes real patch images, so all cells should be flagged 'ok'."""
    cfg = _small_config(tmp_path)
    checkpoint = _checkpoint(tmp_path, cfg)
    result = run_spatho_export(cfg, checkpoint=checkpoint, output_dir=tmp_path / "out", batch_size=4, device="cpu")

    frame = pd.read_parquet(result.cell_embeddings)
    assert set(frame["qc_flag"].unique()) == {"ok"}, "All synthetic patches should be flagged 'ok'"


def test_cli_export_spatho_writes_artifacts(tmp_path: Path) -> None:
    cfg = _small_config(tmp_path)
    checkpoint = _checkpoint(tmp_path, cfg)
    config_path = _write_config(tmp_path / "cfg.yaml", cfg)
    output_dir = tmp_path / "cli_spatho"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "export-spatho",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output_dir),
            "--batch-size",
            "4",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert Path(payload["cell_embeddings"]).exists()
    assert Path(payload["region_embeddings"]).exists()
    assert Path(payload["region_cell_membership"]).exists()
    assert Path(payload["region_molecular_summary"]).exists()
    assert Path(payload["structure_summary"]).exists()
    assert Path(payload["qc_report"]).exists()
    assert payload["n_cells"] > 0
    assert Path(payload["region_embeddings"]).exists()


# ---------------------------------------------------------------------------
# Phase 4 – ImageGeneSTGPT.from_pretrained() and load_checkpoint()
# ---------------------------------------------------------------------------


def test_from_pretrained_returns_eval_model(tmp_path: Path) -> None:
    cfg = _small_config(tmp_path)
    checkpoint = _checkpoint(tmp_path, cfg)

    model = ImageGeneSTGPT.from_pretrained(checkpoint, device="cpu")

    assert isinstance(model, ImageGeneSTGPT)
    assert not model.training, "from_pretrained() must return the model in eval mode"


def test_from_pretrained_model_dimensions(tmp_path: Path) -> None:
    cfg = _small_config(tmp_path)
    checkpoint = _checkpoint(tmp_path, cfg)

    model = ImageGeneSTGPT.from_pretrained(checkpoint, device="cpu")
    assert model.d_model == cfg.model.d_model


def test_from_pretrained_is_on_cpu(tmp_path: Path) -> None:
    cfg = _small_config(tmp_path)
    checkpoint = _checkpoint(tmp_path, cfg)

    model = ImageGeneSTGPT.from_pretrained(checkpoint, device="cpu")
    param = next(model.parameters())
    assert param.device.type == "cpu"


def test_load_checkpoint_returns_raw_dict(tmp_path: Path) -> None:
    cfg = _small_config(tmp_path)
    checkpoint = _checkpoint(tmp_path, cfg)

    payload = ImageGeneSTGPT.load_checkpoint(checkpoint)
    assert isinstance(payload, dict)
    assert "model_state" in payload
    assert "config" in payload
    assert "vocab" in payload


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


def test_top_level_exports_include_spatho_symbols() -> None:
    assert hasattr(stgpt, "run_spatho_export")
    assert hasattr(stgpt, "SpathoExportResult")
    assert hasattr(stgpt, "PatchManifestRow")
    assert hasattr(stgpt, "CELL_EMBEDDING_REQUIRED_COLUMNS")
    assert hasattr(stgpt, "REGION_EMBEDDING_REQUIRED_COLUMNS")
    assert hasattr(stgpt, "STRUCTURE_SUMMARY_REQUIRED_COLUMNS")
