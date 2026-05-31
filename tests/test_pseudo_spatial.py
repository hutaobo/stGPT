from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from typer.testing import CliRunner

from stgpt.cli import app
from stgpt.config import DataConfig, ModelConfig, SplitConfig, StGPTConfig, TrainingConfig
from stgpt.data import build_training_case
from stgpt.pseudo_spatial import (
    _resolve_device,
    build_pseudo_spatial_targets,
    predict_pseudo_spatial,
    train_pseudo_spatial_prior,
)


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


def test_resolve_device_strips_cli_whitespace() -> None:
    assert _resolve_device(" cpu\r\n").type == "cpu"


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


def test_train_pseudo_spatial_uses_processed_corpus_streaming_path(tmp_path: Path) -> None:
    roots: list[Path] = []
    for slide_idx in range(2):
        root = tmp_path / f"slide_{slide_idx}_outs"
        cells = root / "xenium_slide.zarr" / "tables" / "cells"
        cells.parent.mkdir(parents=True)
        obs = pd.DataFrame(
            {
                "cell_id": [f"s{slide_idx}_c{i}" for i in range(8)],
                "contour_id": [f"r{i % 4}" for i in range(8)],
                "structure_id": [i % 2 for i in range(8)],
                "structure_label": [f"structure_{i % 2}" for i in range(8)],
                "x": np.arange(8, dtype=np.float32) + slide_idx * 100,
                "y": np.arange(8, dtype=np.float32),
            },
            index=[f"s{slide_idx}_c{i}" for i in range(8)],
        )
        var = pd.DataFrame({"feature_name": [f"GENE{i}" for i in range(6)]}, index=[f"GENE{i}" for i in range(6)])
        values = np.arange(48, dtype=np.float32).reshape(8, 6)
        adata = ad.AnnData(X=sparse.csr_matrix(values), obs=obs, var=var)
        adata.write_zarr(cells)
        roots.append(root)

    cfg = StGPTConfig(
        case_name="processed_corpus_pseudo",
        data=DataConfig(mode="corpus", dataset_roots=[str(root) for root in roots], output_dir=str(tmp_path / "case")),
        model=ModelConfig(
            d_model=32,
            n_heads=4,
            n_layers=1,
            max_genes=4,
            image_size=32,
            n_expression_bins=8,
            use_image_context=False,
        ),
        training=TrainingConfig(batch_size=4, max_steps=1, output_dir=str(tmp_path / "train"), device="cpu", seed=3),
        split=SplitConfig(
            strategy="slide_holdout",
            group_key="corpus_slide_id",
            train_fraction=0.5,
            val_fraction=0.25,
            test_fraction=0.25,
            seed=3,
        ),
    )

    result = train_pseudo_spatial_prior(
        cfg,
        output_dir=tmp_path / "processed_train",
        max_steps=1,
        n_spatial_bins=4,
        n_niches=2,
        max_genes=4,
        d_model=32,
        batch_size=4,
        device="cpu",
    )

    reference = pd.read_parquet(result["reference_regions"])
    assert result["n_regions"] == 8
    assert result["n_genes"] == 4
    assert set(reference["corpus_slide_id"].astype(str)) == {"slide_0_outs", "slide_1_outs"}
