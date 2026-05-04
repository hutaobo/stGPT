from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pytest
import zarr

from stgpt.contour_store import (
    CONTEXT_RGB_KEY,
    OBJECT_RGB_KEY,
    ContourStoreSpec,
    NeighborGraphConfig,
    SlideNeighborGraphBuilder,
    build_mock_contour_manifest,
    build_mock_contour_store,
    contour_manifest_schema,
    create_contour_image_store,
    pack_contour_patches,
    read_contour_manifest,
    validate_contour_image_store,
    validate_contour_manifest,
)


def test_manifest_schema_uses_fixed_size_neighbor_fields() -> None:
    schema = contour_manifest_schema(max_neighbors=4)

    assert schema.field("neighbor_row_indices").type == pa.list_(pa.int32(), 4)
    assert schema.field("neighbor_distances").type == pa.list_(pa.float32(), 4)
    assert schema.field("neighbor_offsets_xy").type == pa.list_(pa.float32(), 8)
    assert schema.field("neighbor_valid_mask").type == pa.list_(pa.bool_(), 4)
    assert schema.field("bbox_level0_xy").type == pa.list_(pa.float32(), 4)


def test_mock_contour_store_writes_valid_zarr_and_manifest(tmp_path: Path) -> None:
    spec = ContourStoreSpec(n_contours=8, image_size=16, geometry_size=5, max_neighbors=3, chunk_size=4)
    result = build_mock_contour_store(
        tmp_path / "contour_image_store.zarr",
        tmp_path / "contour_image_manifest.parquet",
        spec=spec,
        slide_ids=["slide_a", "slide_b"],
        seed=7,
    )

    assert result["store_validation"]["n_contours"] == 8
    assert result["manifest_validation"]["slide_count"] == 2

    root = zarr.open_group(result["store_path"], mode="r")
    assert root.attrs["contract_version"] == "rfc-0001"
    assert root.attrs["is_spatially_sorted"] is True
    assert root[OBJECT_RGB_KEY].shape == (8, 16, 16, 3)
    assert root[OBJECT_RGB_KEY].chunks == (4, 16, 16, 3)
    assert root[CONTEXT_RGB_KEY].shape == (8, 16, 16, 3)

    table = read_contour_manifest(result["manifest_path"])
    assert table.num_rows == 8
    assert table.column("row_index").to_pylist() == list(range(8))
    assert validate_contour_manifest(table, max_neighbors=3)["schema"] == "ok"


def test_pack_contour_patches_writes_store_and_manifest(tmp_path: Path) -> None:
    patch_dir = tmp_path / "patches"
    patch_dir.mkdir()
    image_path = patch_dir / "contour_a.png"
    from PIL import Image

    Image.new("RGB", (12, 8), color=(120, 80, 40)).save(image_path)
    patch_manifest = tmp_path / "contour_patches_manifest.json"
    patch_manifest.write_text(
        json.dumps(
            [
                {
                    "contour_id": "contour_a",
                    "image_path": image_path.as_posix(),
                    "bbox": {"x0": 10, "y0": 20, "x1": 30, "y1": 40},
                    "patch": {"saved_width": 12, "saved_height": 8},
                },
                {
                    "contour_id": "contour_b",
                    "image_path": image_path.as_posix(),
                    "bbox": {"x0": 40, "y0": 20, "x1": 50, "y1": 40},
                    "patch": {"saved_width": 12, "saved_height": 8},
                },
            ]
        ),
        encoding="utf-8",
    )

    result = pack_contour_patches(
        patch_manifest,
        tmp_path / "contour_image_store.zarr",
        tmp_path / "contour_image_manifest.parquet",
        slide_id="slide_a",
        image_size=16,
        max_neighbors=1,
        chunk_size=2,
    )

    assert result["n_contours"] == 2
    assert result["store_validation"]["n_contours"] == 2
    table = read_contour_manifest(result["manifest_path"])
    assert table.column("contour_id").to_pylist() == ["contour_a", "contour_b"]
    assert validate_contour_manifest(table, max_neighbors=1)["schema"] == "ok"


def test_manifest_validation_rejects_cross_slide_neighbors() -> None:
    spec = ContourStoreSpec(n_contours=4, image_size=16, max_neighbors=2, chunk_size=2)
    table = build_mock_contour_manifest(spec, slide_ids=["slide_a", "slide_a", "slide_b", "slide_b"])
    data = table.to_pydict()
    data["neighbor_row_indices"][0] = [2, -1]
    data["neighbor_valid_mask"][0] = [True, False]
    broken = pa.Table.from_pydict(data, schema=contour_manifest_schema(max_neighbors=2))

    with pytest.raises(ValueError, match="crosses slide boundary"):
        validate_contour_manifest(broken, max_neighbors=2)


def test_manifest_validation_rejects_variable_length_neighbor_schema() -> None:
    schema = pa.schema(
        [
            pa.field("contour_id", pa.string(), nullable=False),
            pa.field("row_index", pa.int32(), nullable=False),
            pa.field("slide_id", pa.string(), nullable=False),
            pa.field("spatial_sort_key", pa.uint64(), nullable=False),
            pa.field("chunk_id", pa.int32(), nullable=False),
            pa.field("centroid_x", pa.float32(), nullable=False),
            pa.field("centroid_y", pa.float32(), nullable=False),
            pa.field("bbox_level0_xy", pa.list_(pa.float32(), 4), nullable=False),
            pa.field("neighbor_row_indices", pa.list_(pa.int32()), nullable=False),
            pa.field("neighbor_distances", pa.list_(pa.float32(), 2), nullable=False),
            pa.field("neighbor_offsets_xy", pa.list_(pa.float32(), 4), nullable=False),
            pa.field("neighbor_valid_mask", pa.list_(pa.bool_(), 2), nullable=False),
            pa.field("area", pa.float32(), nullable=False),
            pa.field("perimeter", pa.float32(), nullable=False),
            pa.field("eccentricity", pa.float32(), nullable=False),
            pa.field("qc_flag", pa.string(), nullable=False),
            pa.field("transform_fingerprint", pa.string(), nullable=False),
        ]
    )
    table = pa.Table.from_pydict(
        {
            "contour_id": ["a"],
            "row_index": [0],
            "slide_id": ["slide"],
            "spatial_sort_key": [0],
            "chunk_id": [0],
            "centroid_x": [0.0],
            "centroid_y": [0.0],
            "bbox_level0_xy": [[0.0, 0.0, 1.0, 1.0]],
            "neighbor_row_indices": [[-1]],
            "neighbor_distances": [[0.0, 0.0]],
            "neighbor_offsets_xy": [[0.0, 0.0, 0.0, 0.0]],
            "neighbor_valid_mask": [[False, False]],
            "area": [1.0],
            "perimeter": [1.0],
            "eccentricity": [0.0],
            "qc_flag": ["ok"],
            "transform_fingerprint": ["mock"],
        },
        schema=schema,
    )

    with pytest.raises(ValueError, match="neighbor_row_indices"):
        validate_contour_manifest(table, max_neighbors=2)


def test_contour_store_validation_rejects_wrong_chunk_size(tmp_path: Path) -> None:
    create_contour_image_store(
        tmp_path / "store.zarr",
        spec=ContourStoreSpec(n_contours=4, image_size=16, chunk_size=2),
    )

    with pytest.raises(ValueError, match="chunk_size"):
        validate_contour_image_store(
            tmp_path / "store.zarr",
            spec=ContourStoreSpec(n_contours=4, image_size=16, chunk_size=3),
        )


def test_neighbor_builder_sorts_by_slide_and_morton_order() -> None:
    builder = SlideNeighborGraphBuilder(NeighborGraphConfig(max_neighbors=1, chunk_size=2, sort_bits=4))

    table = builder.build(
        contour_ids=["b_top", "a_far", "b_origin", "a_origin"],
        slide_ids=["b", "a", "b", "a"],
        centroids=[[10.0, 10.0], [10.0, 10.0], [0.0, 0.0], [0.0, 0.0]],
    )
    data = table.to_pydict()

    assert data["row_index"] == [0, 1, 2, 3]
    assert data["slide_id"] == ["a", "a", "b", "b"]
    assert data["contour_id"] == ["a_origin", "a_far", "b_origin", "b_top"]
    assert data["neighbor_row_indices"] == [[1], [0], [3], [2]]
    assert validate_contour_manifest(table, max_neighbors=1)["slide_count"] == 2


def test_radius_neighbor_builder_pads_sparse_rows() -> None:
    builder = SlideNeighborGraphBuilder(
        NeighborGraphConfig(max_neighbors=2, mode="radius", radius=1.5, chunk_size=8)
    )

    table = builder.build(
        contour_ids=["near_a", "near_b", "far"],
        slide_ids=["slide", "slide", "slide"],
        centroids=[[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]],
    )
    data = table.to_pydict()
    near_a = data["contour_id"].index("near_a")
    far = data["contour_id"].index("far")

    assert data["neighbor_row_indices"][near_a][0] == data["contour_id"].index("near_b")
    assert data["neighbor_valid_mask"][near_a] == [True, False]
    assert data["neighbor_offsets_xy"][near_a] == [1.0, 0.0, 0.0, 0.0]
    assert data["neighbor_row_indices"][far] == [-1, -1]
    assert data["neighbor_valid_mask"][far] == [False, False]


def test_angular_neighbor_sampling_preserves_directional_coverage() -> None:
    builder = SlideNeighborGraphBuilder(
        NeighborGraphConfig(max_neighbors=4, mode="radius", radius=10.0, sampling="angular", chunk_size=8)
    )

    table = builder.build(
        contour_ids=["anchor", "east1", "east2", "east3", "east4", "north", "west", "south"],
        slide_ids=["slide"] * 8,
        centroids=[
            [0.0, 0.0],
            [1.0, 0.0],
            [1.1, 0.0],
            [1.2, 0.0],
            [1.3, 0.0],
            [0.0, 5.0],
            [-5.0, 0.0],
            [0.0, -5.0],
        ],
    )
    data = table.to_pydict()
    anchor = data["contour_id"].index("anchor")
    neighbor_ids = {data["contour_id"][idx] for idx, valid in zip(data["neighbor_row_indices"][anchor], data["neighbor_valid_mask"][anchor], strict=True) if valid}

    assert neighbor_ids == {"east1", "north", "west", "south"}


def test_neighbor_builder_never_links_across_slides() -> None:
    builder = SlideNeighborGraphBuilder(NeighborGraphConfig(max_neighbors=2, mode="knn", chunk_size=8))

    table = builder.build(
        contour_ids=["a0", "b0", "a1", "b1"],
        slide_ids=["a", "b", "a", "b"],
        centroids=[[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
    )
    data = table.to_pydict()

    for row_idx, slide_id in enumerate(data["slide_id"]):
        for neighbor_idx, valid in zip(data["neighbor_row_indices"][row_idx], data["neighbor_valid_mask"][row_idx], strict=True):
            if valid:
                assert data["slide_id"][neighbor_idx] == slide_id
