from __future__ import annotations

import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from scipy import sparse

from stgpt.config import DataConfig, ModelConfig, StGPTConfig, TrainingConfig
from stgpt.data import (
    ImageGeneDataset,
    build_training_case,
    build_training_manifest,
    load_xenium_case,
    make_synthetic_case,
)
from stgpt.images import write_synthetic_patch
from stgpt.qc import validate_data


def _config(tmp_path: Path) -> StGPTConfig:
    return StGPTConfig(
        case_name="test",
        data=DataConfig(
            mode="synthetic",
            output_dir=str(tmp_path / "case"),
            n_cells=12,
            n_genes=20,
            n_structures=3,
            image_size=32,
            seed=2,
        ),
        model=ModelConfig(d_model=32, n_heads=4, n_layers=1, max_genes=12, image_size=32, n_expression_bins=8),
        training=TrainingConfig(batch_size=4, max_steps=1, output_dir=str(tmp_path / "train"), device="cpu"),
    )


def test_synthetic_case_and_manifest(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    case = make_synthetic_case(cfg.data)
    assert case.adata.n_obs == 12
    assert case.adata.n_vars == 20
    assert "spatial" in case.adata.obsm
    assert len(case.patch_table) == 12
    built = build_training_case(cfg)
    assert len(built.region_table) == 12
    assert built.cell_membership["region_id"].nunique() == 12
    manifest = build_training_manifest(cfg)
    assert manifest["n_cells"] == 12
    assert manifest["training_unit"] == "region"
    assert manifest["n_regions"] == 12
    assert Path(manifest["patch_table"]).exists()
    assert Path(manifest["region_table"]).exists()


def test_dataset_collate_masks_and_images(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    dataset = ImageGeneDataset(make_synthetic_case(cfg.data), cfg)
    batch = dataset.collate([dataset[0], dataset[1], dataset[2]])
    assert batch["gene_ids"].shape == (3, 12)
    assert len(batch["region_ids"]) == 3
    assert batch["cell_expr_values"].shape[:2] == (3, cfg.model.max_cells_per_region)
    assert batch["cell_token_mask"].dtype == torch.bool
    assert batch["image"].shape == (3, 3, 32, 32)
    assert batch["mask"].any()
    assert batch["gene_padding_mask"].dtype == torch.bool
    assert {"region_indices", "n_cells", "spatial", "structure_labels"}.issubset(batch)


def test_corpus_mode_concatenates_h5ad_and_assigns_slide_metadata(tmp_path: Path) -> None:
    first_cfg = DataConfig(mode="synthetic", output_dir=str(tmp_path / "case_a"), n_cells=6, n_genes=10, image_size=32, seed=1)
    second_cfg = DataConfig(mode="synthetic", output_dir=str(tmp_path / "case_b"), n_cells=7, n_genes=10, image_size=32, seed=2)
    first = make_synthetic_case(first_cfg).adata
    second = make_synthetic_case(second_cfg).adata
    first_path = tmp_path / "slide_a.h5ad"
    second_path = tmp_path / "slide_b.h5ad"
    first.write_h5ad(first_path)
    second.write_h5ad(second_path)

    cfg = DataConfig(
        mode="corpus",
        input_h5ad_list=[str(first_path), str(second_path)],
        output_dir=str(tmp_path / "corpus"),
    )
    adata = load_xenium_case(cfg)
    assert adata.n_obs == 13
    assert "slide_id" in adata.obs.columns
    assert set(adata.obs["slide_id"].astype(str)) == {"slide_a", "slide_b"}
    assert "batch_id" in adata.obs.columns


def test_xenium_slide_mode_uses_contour_patch_context_and_validates(tmp_path: Path) -> None:
    sibling_pyxenium = Path(__file__).resolve().parents[2] / "pyXenium" / "src"
    if sibling_pyxenium.exists() and str(sibling_pyxenium) not in sys.path:
        sys.path.insert(0, str(sibling_pyxenium))
    pyxenium_io = pytest.importorskip("pyXenium.io")

    patch_dir = tmp_path / "contour_patches"
    patch_path = write_synthetic_patch(
        patch_dir / "contour_a.png",
        image_size=32,
        structure_id=1,
        intensity=0.8,
        seed=11,
    )
    patch_manifest = tmp_path / "contour_patches_manifest.json"
    patch_manifest.write_text(
        json.dumps(
            [
                {
                    "contour_id": "contour_a",
                    "structure_id": 7,
                    "structure_label": "tumor_region",
                    "structure_name": "tumor_region",
                    "image_path": str(patch_path),
                    "patch": {
                        "bbox_level_xy": [0, 0, 32, 32],
                        "bbox_level0_xy": [0, 0, 32, 32],
                        "pyramid_level": 0,
                    },
                    "transform": {"transform_direction": "image_pixel_xy_to_xenium_pixel_xy"},
                }
            ]
        ),
        encoding="utf-8",
    )
    adata = ad.AnnData(
        X=sparse.csr_matrix(np.asarray([[1.0, 0.0, 2.0], [0.0, 3.0, 1.0]], dtype=np.float32)),
        obs=pd.DataFrame(
            {
                "cell_id": ["cell_a", "cell_b"],
                "contour_id": ["contour_a", "contour_a"],
                "structure_id": [7, 7],
                "structure_label": ["tumor_region", "tumor_region"],
            },
            index=["cell_a", "cell_b"],
        ),
        var=pd.DataFrame({"feature_name": ["GeneA", "GeneB", "GeneC"]}, index=["GeneA", "GeneB", "GeneC"]),
    )
    adata.layers["rna"] = adata.X.copy()
    adata.obsm["spatial"] = np.asarray([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32)
    slide = pyxenium_io.XeniumSlide(
        table=adata,
        metadata={"contours": {"contour_patches_manifest": str(patch_manifest)}},
    )
    slide_store = tmp_path / "xenium_slide.zarr"
    pyxenium_io.write_xenium_slide(slide, slide_store)

    cfg = StGPTConfig(
        case_name="slide_case",
        data=DataConfig(
            mode="xenium_slide",
            slide_store=str(slide_store),
            output_dir=str(tmp_path / "case"),
            include_structure_context=True,
        ),
        model=ModelConfig(d_model=32, n_heads=4, n_layers=1, max_genes=4, image_size=32, n_expression_bins=8),
        training=TrainingConfig(batch_size=2, max_steps=1, output_dir=str(tmp_path / "train"), device="cpu"),
    )
    case = build_training_case(cfg)
    dataset = ImageGeneDataset(case, cfg)
    item = dataset[0]
    batch = dataset.collate([item])
    qc = validate_data(cfg, output_dir=tmp_path / "qc")

    assert case.adata.n_obs == 2
    assert len(case.patch_table) == 1
    assert len(case.region_table) == 1
    assert case.cell_membership["cell_id"].tolist() == ["cell_a", "cell_b"]
    assert item["image_path"] == str(patch_path)
    assert batch["region_ids"] == ["contour_a"]
    assert float(batch["image"].sum()) > 0.0
    assert qc["status"] == "pass"
