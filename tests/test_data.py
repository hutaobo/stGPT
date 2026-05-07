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
from torch.utils.data import DataLoader

import stgpt.data as data_module
from stgpt.config import DataConfig, ModelConfig, StGPTConfig, TrainingConfig
from stgpt.contour_store import (
    ContourStoreSpec,
    NeighborGraphConfig,
    SlideNeighborGraphBuilder,
    create_contour_image_store,
    write_contour_manifest,
)
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


def _write_mock_packed_contour_inputs(
    tmp_path: Path,
    *,
    n_contours: int,
    image_size: int,
    max_neighbors: int,
) -> tuple[Path, Path]:
    store_path = tmp_path / "contour_image_store.zarr"
    manifest_path = tmp_path / "contour_image_manifest.parquet"
    spec = ContourStoreSpec(
        n_contours=n_contours,
        image_size=image_size,
        geometry_size=5,
        max_neighbors=max_neighbors,
        chunk_size=4,
    )
    root = create_contour_image_store(store_path, spec=spec)
    for idx in range(n_contours):
        root["object_rgb"][idx] = np.full((image_size, image_size, 3), 32 + idx, dtype=np.uint8)
        root["context_rgb"][idx] = np.full((image_size, image_size, 3), 96 + idx, dtype=np.uint8)
        root["soft_mask"][idx] = np.full((image_size, image_size, 1), 255, dtype=np.uint8)
        root["geometry"][idx] = np.asarray([idx, idx + 1, idx + 2, idx + 3, idx + 4], dtype=np.float32)
        root["contour_ids"][idx] = f"contour_{idx:03d}".encode()
    builder = SlideNeighborGraphBuilder(
        NeighborGraphConfig(max_neighbors=max_neighbors, mode="knn", chunk_size=4)
    )
    table = builder.build(
        contour_ids=[f"contour_{idx:03d}" for idx in range(n_contours)],
        slide_ids=["mock_slide"] * n_contours,
        centroids=[[float(idx), 0.0] for idx in range(n_contours)],
    )
    write_contour_manifest(table, manifest_path, max_neighbors=max_neighbors)
    return store_path, manifest_path


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


def test_dataset_reads_packed_contour_store_by_row_index(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    store_path, manifest_path = _write_mock_packed_contour_inputs(tmp_path, n_contours=12, image_size=32, max_neighbors=3)
    payload = cfg.model_dump()
    payload["data"]["contour_image_store"] = str(store_path)
    payload["data"]["contour_manifest"] = str(manifest_path)
    cfg = StGPTConfig.model_validate(payload)

    case = build_training_case(cfg)
    dataset = ImageGeneDataset(case, cfg)
    item = dataset[0]
    batch = dataset.collate([item])

    assert item["row_index"] == 0
    assert item["image_evidence"]["source"] == "contour_store"
    assert batch["image_source"].tolist() == [2]
    assert batch["image"].shape == (1, 3, 32, 32)
    assert batch["object_image"].shape == (1, 3, 32, 32)
    assert batch["context_image"].shape == (1, 3, 32, 32)
    assert batch["contour_mask"].shape == (1, 1, 32, 32)
    assert batch["contour_geometry"].shape == (1, 5)
    assert batch["neighbor_row_indices"].shape == (1, 3)
    assert batch["neighbor_offsets_xy"].shape == (1, 3, 2)
    assert batch["neighbor_valid_mask"].dtype == torch.bool
    assert torch.isclose(batch["image"].mean(), torch.tensor(32.0 / 255.0), atol=1e-5)


def test_dataset_reads_packed_contour_store_with_dataloader_workers(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    store_path, manifest_path = _write_mock_packed_contour_inputs(tmp_path, n_contours=12, image_size=32, max_neighbors=2)
    payload = cfg.model_dump()
    payload["data"]["contour_image_store"] = str(store_path)
    payload["data"]["contour_manifest"] = str(manifest_path)
    payload["training"]["num_workers"] = 2
    cfg = StGPTConfig.model_validate(payload)

    dataset = ImageGeneDataset(build_training_case(cfg), cfg)
    loader = DataLoader(dataset, batch_size=3, shuffle=False, collate_fn=dataset.collate, num_workers=2)
    batch = next(iter(loader))

    assert batch["image_source"].tolist() == [2, 2, 2]
    assert batch["image"].shape == (3, 3, 32, 32)
    assert batch["neighbor_row_indices"].shape == (3, 2)
    assert batch["neighbor_valid_mask"].dtype == torch.bool


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


def test_processed_xenium_slide_corpus_preserves_per_slide_contour_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    slide_roots = [tmp_path / "WTA_Preview_FFPE_Breast_Cancer_outs", tmp_path / "WTA_Preview_FFPE_Cervical_Cancer_outs"]
    for root in slide_roots:
        root.mkdir()
        (root / "xenium_slide.zarr").mkdir()
        _write_mock_packed_contour_inputs(root, n_contours=2, image_size=32, max_neighbors=1)

    def fake_load_xenium_slide(data_config: DataConfig) -> ad.AnnData:
        slide_name = Path(str(data_config.slide_store)).parent.name
        obs = pd.DataFrame(
            {
                "cell_id": [f"{slide_name}_cell_a", f"{slide_name}_cell_b", f"{slide_name}_cell_c"],
                "contour_id": ["contour_000", "contour_000", "contour_001"],
                "structure_id": [1, 1, 2],
                "structure_label": ["region_a", "region_a", "region_b"],
            },
            index=[f"{slide_name}_cell_a", f"{slide_name}_cell_b", f"{slide_name}_cell_c"],
        )
        var = pd.DataFrame(
            {"feature_name": ["GeneA", "GeneB", f"{slide_name}_GeneC"]},
            index=["GeneA", "GeneB", f"{slide_name}_GeneC"],
        )
        values = np.asarray([[1.0, 0.0, 2.0], [2.0, 1.0, 0.0], [0.0, 3.0, 1.0]], dtype=np.float32)
        adata = ad.AnnData(X=sparse.csr_matrix(values), obs=obs, var=var)
        adata.layers["rna"] = adata.X.copy()
        adata.obsm["spatial"] = np.asarray([[0.0, 0.0], [1.0, 1.0], [10.0, 10.0]], dtype=np.float32)
        return adata

    monkeypatch.setattr(data_module, "_load_xenium_slide", fake_load_xenium_slide)
    cfg = StGPTConfig(
        case_name="multi_slide",
        data=DataConfig(
            mode="corpus",
            dataset_roots=[str(root) for root in slide_roots],
            output_dir=str(tmp_path / "case"),
            min_cells_per_region=1,
        ),
        model=ModelConfig(d_model=32, n_heads=4, n_layers=1, max_genes=4, image_size=32, n_expression_bins=8),
        training=TrainingConfig(batch_size=2, max_steps=1, output_dir=str(tmp_path / "train"), device="cpu"),
    )

    case = build_training_case(cfg)
    dataset = ImageGeneDataset(case, cfg)
    batch = dataset.collate([dataset[0], dataset[2]])

    assert case.adata.n_obs == 6
    assert len(case.region_table) == 4
    assert case.region_table["region_id"].is_unique
    assert set(case.region_table["corpus_slide_id"]) == {root.name for root in slide_roots}
    assert set(case.region_table["image_store"].map(lambda value: Path(str(value)).parent.name)) == {root.name for root in slide_roots}
    assert all("::" in value for value in case.region_table["region_id"].astype(str))
    assert case.region_expression.shape == (4, 4)
    assert batch["image_source"].tolist() == [2, 2]
    assert batch["image"].shape == (2, 3, 32, 32)
    assert batch["row_index"].tolist() == [0, 0]


def test_legacy_qc_codex_flags_downgrade_region_qc(tmp_path: Path) -> None:
    patch_dir = tmp_path / "patches"
    patch_a = write_synthetic_patch(patch_dir / "contour_a.png", image_size=32, structure_id=1, intensity=0.7, seed=1)
    patch_b = write_synthetic_patch(patch_dir / "contour_b.png", image_size=32, structure_id=2, intensity=0.5, seed=2)
    patch_manifest = tmp_path / "contour_patches_manifest.json"
    patch_manifest.write_text(
        json.dumps(
            [
                {"contour_id": "contour_a", "structure_id": 1, "structure_label": "ok", "image_path": str(patch_a)},
                {"contour_id": "contour_b", "structure_id": 2, "structure_label": "fold", "image_path": str(patch_b)},
            ]
        ),
        encoding="utf-8",
    )
    qc_dir = tmp_path / "stgpt_qc_codex"
    qc_dir.mkdir()
    pd.DataFrame(
        {
            "contour_id": ["contour_a", "contour_b"],
            "mask_quality": ["pass", "folded"],
        }
    ).to_csv(qc_dir / "mask_quality_report.csv", index=False)
    adata = ad.AnnData(
        X=sparse.csr_matrix(np.asarray([[1, 0], [0, 2], [3, 1]], dtype=np.float32)),
        obs=pd.DataFrame(
            {
                "cell_id": ["cell_a", "cell_b", "cell_c"],
                "contour_id": ["contour_a", "contour_b", "contour_b"],
            },
            index=["cell_a", "cell_b", "cell_c"],
        ),
        var=pd.DataFrame({"feature_name": ["GeneA", "GeneB"]}, index=["GeneA", "GeneB"]),
    )
    adata.obsm["spatial"] = np.asarray([[0, 0], [1, 1], [2, 2]], dtype=np.float32)
    h5ad = tmp_path / "cells.h5ad"
    adata.write_h5ad(h5ad)
    cfg = StGPTConfig(
        case_name="legacy_qc",
        data=DataConfig(
            mode="anndata",
            input_h5ad=str(h5ad),
            patch_manifest=str(patch_manifest),
            output_dir=str(tmp_path / "case"),
        ),
        model=ModelConfig(d_model=32, n_heads=4, n_layers=1, max_genes=4, image_size=32, n_expression_bins=8),
        training=TrainingConfig(batch_size=2, max_steps=1, output_dir=str(tmp_path / "train"), device="cpu"),
    )

    case = build_training_case(cfg)
    flags = dict(zip(case.region_table["contour_id"], case.region_table["qc_flag"], strict=False))

    assert flags["contour_a"] == "ok"
    assert flags["contour_b"] == "legacy_qc_fail"
    assert "legacy_qc_source" in case.region_table.columns


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
