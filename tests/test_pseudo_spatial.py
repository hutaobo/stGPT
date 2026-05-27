from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from stgpt.cli import app
from stgpt.config import DataConfig, ModelConfig, StGPTConfig, TrainingConfig
from stgpt.data import build_training_case
from stgpt.pseudo_spatial import build_pseudo_spatial_targets, predict_pseudo_spatial, train_pseudo_spatial_prior


def _config(tmp_path: Path) -> StGPTConfig:
    return StGPTConfig(
        case_name="pseudo_spatial_test",
        data=DataConfig(
            mode="synthetic",
            output_dir=str(tmp_path / "case"),
            n_cells=24,
            n_genes=24,
            n_structures=3,
            image_size=32,
            seed=5,
        ),
        model=ModelConfig(d_model=32, n_heads=4, n_layers=1, max_genes=12, image_size=32, n_expression_bins=8),
        training=TrainingConfig(batch_size=4, max_steps=1, output_dir=str(tmp_path / "train"), device="cpu", seed=5),
    )


def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "pseudo.yaml"
    path.write_text(_config(tmp_path).model_dump_json(indent=2), encoding="utf-8")
    return path


def test_build_pseudo_spatial_targets(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    case = build_training_case(cfg)

    targets, meta = build_pseudo_spatial_targets(case, n_spatial_bins=4, n_niches=3, seed=0)

    assert {"structure_id", "x_bin", "y_bin", "niche_id"}.issubset(targets.columns)
    assert targets["x_bin"].between(0, 3).all()
    assert targets["y_bin"].between(0, 3).all()
    assert len(meta["structure_names"]) == 3
    assert len(meta["niche_names"]) == 3


def test_train_and_predict_pseudo_spatial_prior(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    result = train_pseudo_spatial_prior(
        cfg,
        output_dir=tmp_path / "pseudo_train",
        max_steps=2,
        n_spatial_bins=4,
        n_niches=3,
        max_genes=10,
        d_model=32,
        batch_size=4,
        device="cpu",
    )

    assert Path(result["checkpoint"]).exists()
    assert Path(result["best_checkpoint"]).exists()
    assert Path(result["reference_regions"]).exists()

    adata = build_training_case(cfg).adata
    h5ad = tmp_path / "input_cells.h5ad"
    adata.write_h5ad(h5ad)
    prediction = predict_pseudo_spatial(
        result["checkpoint"],
        h5ad,
        output=tmp_path / "predictions.csv",
        reference_regions=result["reference_regions"],
        batch_size=5,
        device="cpu",
    )

    frame = pd.read_csv(prediction["predictions"])
    assert len(frame) == adata.n_obs
    assert {
        "cell_id",
        "structure_top1",
        "x_bin_top1",
        "y_bin_top1",
        "niche_top1",
        "projected_region_id",
        "projected_x",
        "projected_y",
    }.issubset(frame.columns)
    assert prediction["missing_selected_gene_count"] == 0


def test_cli_train_pseudo_spatial_smoke(tmp_path: Path) -> None:
    runner = CliRunner()
    config = _write_config(tmp_path)

    result = runner.invoke(
        app,
        [
            "train-pseudo-spatial",
            "--config",
            str(config),
            "--output",
            str(tmp_path / "cli_pseudo"),
            "--max-steps",
            "1",
            "--n-spatial-bins",
            "4",
            "--n-niches",
            "2",
            "--max-genes",
            "8",
            "--d-model",
            "32",
            "--batch-size",
            "4",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "checkpoint" in result.output
