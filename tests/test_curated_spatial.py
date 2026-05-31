from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from typer.testing import CliRunner

from stgpt.cli import app
from stgpt.config import DataConfig, ModelConfig, SplitConfig, StGPTConfig, TrainingConfig
from stgpt.curated_spatial import (
    IGNORE_LABEL,
    audit_curated_structures,
    build_curated_spatial_targets,
    predict_curated_spatial_prior,
    train_curated_spatial_prior,
)


def _write_processed_slide(root: Path, *, slide_idx: int) -> ad.AnnData:
    cells = root / "xenium_slide.zarr" / "tables" / "cells"
    cells.parent.mkdir(parents=True)
    obs = pd.DataFrame(
        {
            "cell_id": [f"s{slide_idx}_c{i}" for i in range(8)],
            "contour_id": [f"r{i % 4}" for i in range(8)],
            "x": np.arange(8, dtype=np.float32) + slide_idx * 100,
            "y": np.arange(8, dtype=np.float32),
        },
        index=[f"s{slide_idx}_c{i}" for i in range(8)],
    )
    var = pd.DataFrame({"feature_name": [f"GENE{i}" for i in range(6)]}, index=[f"GENE{i}" for i in range(6)])
    values = (np.arange(48, dtype=np.float32).reshape(8, 6) + slide_idx).astype(np.float32)
    adata = ad.AnnData(X=sparse.csr_matrix(values), obs=obs, var=var)
    adata.obsm["spatial"] = obs[["x", "y"]].to_numpy(dtype=np.float32)
    adata.write_zarr(cells)
    labels = pd.DataFrame(
        {
            "contour_id": ["r0", "r1", "r2", "r3"],
            "standard_biological_name": [
                "Breast Tumor/Epithelial Region",
                "Endothelial/Vascular Region",
                "Myeloid/Macrophage-Rich Region",
                "Needs Review",
            ],
            "standard_parent_class": [
                "Tumor/Epithelial",
                "Endothelial/Vascular",
                "Myeloid/Macrophage",
                "Neural/Brain",
            ],
            "standard_confidence_tier": ["high", "medium", "curated", "low_review"],
            "standard_needs_review": [False, False, False, True],
            "standard_evidence_source": ["test"] * 4,
            "trainable_standard_parent": [
                "Tumor/Epithelial",
                "Endothelial/Vascular",
                "Myeloid/Macrophage",
                IGNORE_LABEL,
            ],
            "trainable_standard_name": [
                "Breast Tumor/Epithelial Region",
                "Endothelial/Vascular Region",
                "Myeloid/Macrophage-Rich Region",
                IGNORE_LABEL,
            ],
            "label_confidence": [0.95, 0.8, 0.9, 0.1],
            "label_source": ["test"] * 4,
        }
    )
    labels.to_csv(root / "structure_assignments_v2_name.csv", index=False)
    (root / "slide_manifest.json").write_text(
        json.dumps(
            {
                "counts": {"cells": 8, "assigned_cells": 8, "contours": 4, "genes": 6},
                "metadata": {"organ": "test"},
            }
        ),
        encoding="utf-8",
    )
    return adata


def _corpus_config(tmp_path: Path, roots: list[Path]) -> StGPTConfig:
    return StGPTConfig(
        case_name="curated_spatial_test",
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


def test_build_curated_spatial_targets_filters_review_labels() -> None:
    regions = pd.DataFrame(
        {
            "region_id": ["r0", "r1", "r2"],
            "contour_id": ["r0", "r1", "r2"],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 1.0, 2.0],
            "standard_parent_class": ["A", "B", "C"],
            "trainable_standard_name": ["A region", IGNORE_LABEL, "C region"],
            "standard_confidence_tier": ["high", "high", "low_review"],
            "standard_needs_review": [False, False, True],
        }
    )

    targets, meta = build_curated_spatial_targets(regions, n_spatial_bins=4)

    assert len(targets) == 1
    assert targets["parent_name"].tolist() == ["A"]
    assert targets["structure_name"].tolist() == ["A region"]
    assert meta["label_policy"]["excluded_trainable_name"] == IGNORE_LABEL


def test_audit_curated_structures_counts_usable_regions(tmp_path: Path) -> None:
    root = tmp_path / "slide_a"
    _write_processed_slide(root, slide_idx=0)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame({"case_name": ["slide_a"], "case_leaf": ["slide_a"]}).to_csv(manifest, index=False)

    result = audit_curated_structures(manifest, output=tmp_path / "audit")

    assert result["n_cases"] == 1
    assert result["cases_with_curated_assignments"] == 1
    assert result["total_regions"] == 4
    assert result["total_usable_structure_regions"] == 3
    assert Path(result["inventory_csv"]).exists()


def test_train_and_predict_curated_spatial_prior(tmp_path: Path) -> None:
    roots = [tmp_path / f"slide_{idx}" for idx in range(2)]
    adatas = [_write_processed_slide(root, slide_idx=idx) for idx, root in enumerate(roots)]
    cfg = _corpus_config(tmp_path, roots)

    result = train_curated_spatial_prior(
        cfg,
        output_dir=tmp_path / "curated_train",
        max_steps=2,
        n_spatial_bins=4,
        max_genes=4,
        d_model=32,
        batch_size=4,
        device="cpu",
    )

    assert Path(result["checkpoint"]).exists()
    assert result["n_regions"] == 6
    reference = pd.read_parquet(result["reference_regions"])
    assert IGNORE_LABEL not in set(reference["structure_name"].astype(str))
    assert {"parent_id", "structure_id", "x_bin", "y_bin"}.issubset(reference.columns)

    h5ad = tmp_path / "input_cells.h5ad"
    adatas[0].write_h5ad(h5ad)
    prediction = predict_curated_spatial_prior(
        result["checkpoint"],
        h5ad,
        output=tmp_path / "predictions.csv",
        reference_regions=result["reference_regions"],
        batch_size=5,
        device="cpu",
    )

    frame = pd.read_csv(prediction["predictions"])
    assert len(frame) == adatas[0].n_obs
    assert {
        "cell_id",
        "parent_top1",
        "structure_top1",
        "x_bin_top1",
        "y_bin_top1",
        "projected_region_id",
        "projected_x",
        "projected_y",
    }.issubset(frame.columns)
    assert prediction["missing_selected_gene_count"] == 0


def test_cli_audit_curated_structures_smoke(tmp_path: Path) -> None:
    root = tmp_path / "slide_cli"
    _write_processed_slide(root, slide_idx=0)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame({"case_leaf": ["slide_cli"]}).to_csv(manifest, index=False)

    result = CliRunner().invoke(
        app,
        [
            "audit-curated-structures",
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "audit_cli"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "total_usable_structure_regions" in result.output
