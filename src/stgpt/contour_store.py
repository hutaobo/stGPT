from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from PIL import Image
from scipy.spatial import cKDTree

CONTOUR_STORE_CONTRACT_VERSION = "rfc-0001"
CONTOUR_ID_DTYPE = "S64"
DEFAULT_IMAGE_SIZE = 224
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_GEOMETRY_SIZE = 8
DEFAULT_MAX_NEIGHBORS = 16

OBJECT_RGB_KEY = "object_rgb"
CONTEXT_RGB_KEY = "context_rgb"
SOFT_MASK_KEY = "soft_mask"
GEOMETRY_KEY = "geometry"
CONTOUR_IDS_KEY = "contour_ids"

MANIFEST_FIXED_FIELDS = {
    "bbox_level0_xy": 4,
    "neighbor_row_indices": "max_neighbors",
    "neighbor_distances": "max_neighbors",
    "neighbor_offsets_xy": "max_neighbors*2",
    "neighbor_valid_mask": "max_neighbors",
}


@dataclass(frozen=True)
class ContourStoreSpec:
    n_contours: int = 0
    image_size: int = DEFAULT_IMAGE_SIZE
    image_channels: int = 3
    geometry_size: int = DEFAULT_GEOMETRY_SIZE
    max_neighbors: int = DEFAULT_MAX_NEIGHBORS
    chunk_size: int = DEFAULT_CHUNK_SIZE
    contract_version: str = CONTOUR_STORE_CONTRACT_VERSION
    zarr_format: int = 2
    fill_policy: str = "slide_tissue_mean"
    is_spatially_sorted: bool = True

    def __post_init__(self) -> None:
        checks = {
            "n_contours": self.n_contours,
            "image_size": self.image_size,
            "image_channels": self.image_channels,
            "geometry_size": self.geometry_size,
            "max_neighbors": self.max_neighbors,
            "chunk_size": self.chunk_size,
        }
        for name, value in checks.items():
            minimum = 0 if name == "n_contours" else 1
            if int(value) < minimum:
                raise ValueError(f"{name} must be >= {minimum}")
        if self.image_channels != 3:
            raise ValueError("Contour image stores currently require RGB image_channels=3")
        if self.zarr_format != 2:
            raise ValueError("Contour image stores currently use zarr_format=2 for stable fixed-width string arrays")


@dataclass(frozen=True)
class NeighborGraphConfig:
    max_neighbors: int = DEFAULT_MAX_NEIGHBORS
    mode: Literal["knn", "radius"] = "knn"
    radius: float | None = None
    sampling: Literal["nearest", "angular"] = "nearest"
    chunk_size: int = DEFAULT_CHUNK_SIZE
    sort_bits: int = 21

    def __post_init__(self) -> None:
        _validate_max_neighbors(self.max_neighbors)
        if self.mode not in {"knn", "radius"}:
            raise ValueError("mode must be one of: knn, radius")
        if self.sampling not in {"nearest", "angular"}:
            raise ValueError("sampling must be one of: nearest, angular")
        if self.mode == "radius" and (self.radius is None or float(self.radius) <= 0.0):
            raise ValueError("radius mode requires radius > 0")
        if int(self.chunk_size) < 1:
            raise ValueError("chunk_size must be >= 1")
        if int(self.sort_bits) < 1 or int(self.sort_bits) > 31:
            raise ValueError("sort_bits must be between 1 and 31")


class SlideNeighborGraphBuilder:
    """Build fixed-shape, slide-local physical neighbor manifests from contour centroids."""

    def __init__(self, config: NeighborGraphConfig | None = None) -> None:
        self.config = config or NeighborGraphConfig()

    def build(
        self,
        *,
        contour_ids: Sequence[str],
        slide_ids: Sequence[str],
        centroids: Sequence[Sequence[float]] | np.ndarray,
        bbox_level0_xy: Sequence[Sequence[float]] | np.ndarray | None = None,
        area: Sequence[float] | np.ndarray | None = None,
        perimeter: Sequence[float] | np.ndarray | None = None,
        eccentricity: Sequence[float] | np.ndarray | None = None,
        qc_flag: Sequence[str] | None = None,
        transform_fingerprint: Sequence[str] | None = None,
    ) -> pa.Table:
        ids, slides, coords = _validate_builder_inputs(contour_ids, slide_ids, centroids)
        order, spatial_keys = _spatial_sort_order(slides, coords, bits=self.config.sort_bits)
        sorted_ids = [ids[idx] for idx in order]
        sorted_slides = [slides[idx] for idx in order]
        sorted_coords = coords[order]
        sorted_bbox = _ordered_bbox(coords, order, bbox_level0_xy)
        sorted_area = _ordered_float_values(area, order, default=0.0)
        sorted_perimeter = _ordered_float_values(perimeter, order, default=0.0)
        sorted_eccentricity = _ordered_float_values(eccentricity, order, default=0.0)
        sorted_qc = _ordered_str_values(qc_flag, order, default="ok")
        sorted_transform = _ordered_str_values(transform_fingerprint, order, default="unknown_transform")

        neighbors = self._build_neighbors(sorted_coords, sorted_slides)
        data = {
            "contour_id": sorted_ids,
            "row_index": list(range(len(sorted_ids))),
            "slide_id": sorted_slides,
            "spatial_sort_key": [int(spatial_keys[idx]) for idx in order],
            "chunk_id": [idx // int(self.config.chunk_size) for idx in range(len(sorted_ids))],
            "centroid_x": [float(value) for value in sorted_coords[:, 0]],
            "centroid_y": [float(value) for value in sorted_coords[:, 1]],
            "bbox_level0_xy": sorted_bbox,
            "neighbor_row_indices": neighbors["row_indices"],
            "neighbor_distances": neighbors["distances"],
            "neighbor_offsets_xy": neighbors["offsets_xy"],
            "neighbor_valid_mask": neighbors["valid_mask"],
            "area": sorted_area,
            "perimeter": sorted_perimeter,
            "eccentricity": sorted_eccentricity,
            "qc_flag": sorted_qc,
            "transform_fingerprint": sorted_transform,
        }
        table = pa.Table.from_pydict(data, schema=contour_manifest_schema(self.config.max_neighbors))
        validate_contour_manifest(table, max_neighbors=self.config.max_neighbors)
        return table

    def _build_neighbors(self, coords: np.ndarray, slide_ids: Sequence[str]) -> dict[str, list[list[Any]]]:
        n_rows = len(slide_ids)
        row_indices = [[-1] * self.config.max_neighbors for _ in range(n_rows)]
        distances = [[0.0] * self.config.max_neighbors for _ in range(n_rows)]
        offsets_xy = [[0.0] * (self.config.max_neighbors * 2) for _ in range(n_rows)]
        valid_mask = [[False] * self.config.max_neighbors for _ in range(n_rows)]

        for slide_id in _unique_in_order(slide_ids):
            slide_rows = np.asarray([idx for idx, value in enumerate(slide_ids) if value == slide_id], dtype=np.int64)
            if slide_rows.size <= 1:
                continue
            slide_coords = coords[slide_rows]
            tree = cKDTree(slide_coords)
            candidates = self._query_slide(tree, slide_coords)
            for local_anchor, candidate_pairs in enumerate(candidates):
                selected = _select_neighbor_candidates(
                    anchor=slide_coords[local_anchor],
                    coords=slide_coords,
                    candidate_pairs=candidate_pairs,
                    max_neighbors=self.config.max_neighbors,
                    sampling=self.config.sampling,
                )
                global_anchor = int(slide_rows[local_anchor])
                _write_padded_neighbors(
                    global_anchor=global_anchor,
                    local_anchor=local_anchor,
                    slide_rows=slide_rows,
                    slide_coords=slide_coords,
                    selected=selected,
                    row_indices=row_indices,
                    distances=distances,
                    offsets_xy=offsets_xy,
                    valid_mask=valid_mask,
                )
        return {
            "row_indices": row_indices,
            "distances": distances,
            "offsets_xy": offsets_xy,
            "valid_mask": valid_mask,
        }

    def _query_slide(self, tree: cKDTree, coords: np.ndarray) -> list[list[tuple[int, float]]]:
        if self.config.mode == "knn":
            query_k = min(coords.shape[0], int(self.config.max_neighbors) + 1)
            query_distances, query_indices = tree.query(coords, k=query_k)
            query_distances = np.asarray(query_distances)
            query_indices = np.asarray(query_indices)
            if query_indices.ndim == 1:
                query_indices = query_indices[:, None]
                query_distances = query_distances[:, None]
            rows: list[list[tuple[int, float]]] = []
            for anchor_idx, (indices, distances) in enumerate(zip(query_indices, query_distances, strict=True)):
                pairs = [
                    (int(idx), float(distance))
                    for idx, distance in zip(indices, distances, strict=True)
                    if int(idx) != anchor_idx and np.isfinite(distance)
                ]
                rows.append(pairs)
            return rows

        radius = float(self.config.radius or 0.0)
        radius_indices = tree.query_ball_point(coords, r=radius, return_sorted=True)
        rows = []
        for anchor_idx, indices in enumerate(radius_indices):
            pairs = []
            for idx in indices:
                idx = int(idx)
                if idx == anchor_idx:
                    continue
                distance = float(np.linalg.norm(coords[idx] - coords[anchor_idx]))
                pairs.append((idx, distance))
            rows.append(pairs)
        return rows


def contour_manifest_schema(max_neighbors: int = DEFAULT_MAX_NEIGHBORS) -> pa.Schema:
    _validate_max_neighbors(max_neighbors)
    return pa.schema(
        [
            pa.field("contour_id", pa.string(), nullable=False),
            pa.field("row_index", pa.int32(), nullable=False),
            pa.field("slide_id", pa.string(), nullable=False),
            pa.field("spatial_sort_key", pa.uint64(), nullable=False),
            pa.field("chunk_id", pa.int32(), nullable=False),
            pa.field("centroid_x", pa.float32(), nullable=False),
            pa.field("centroid_y", pa.float32(), nullable=False),
            pa.field("bbox_level0_xy", pa.list_(pa.float32(), 4), nullable=False),
            pa.field("neighbor_row_indices", pa.list_(pa.int32(), max_neighbors), nullable=False),
            pa.field("neighbor_distances", pa.list_(pa.float32(), max_neighbors), nullable=False),
            pa.field("neighbor_offsets_xy", pa.list_(pa.float32(), max_neighbors * 2), nullable=False),
            pa.field("neighbor_valid_mask", pa.list_(pa.bool_(), max_neighbors), nullable=False),
            pa.field("area", pa.float32(), nullable=False),
            pa.field("perimeter", pa.float32(), nullable=False),
            pa.field("eccentricity", pa.float32(), nullable=False),
            pa.field("qc_flag", pa.string(), nullable=False),
            pa.field("transform_fingerprint", pa.string(), nullable=False),
        ]
    )


def create_contour_image_store(
    store_path: str | Path,
    *,
    spec: ContourStoreSpec | None = None,
    overwrite: bool = True,
) -> zarr.Group:
    resolved = Path(store_path)
    cfg = spec or ContourStoreSpec()
    mode = "w" if overwrite else "w-"
    root = zarr.open_group(str(resolved), mode=mode, zarr_format=cfg.zarr_format)
    root.attrs.update(
        {
            "contract_version": cfg.contract_version,
            "n_contours": int(cfg.n_contours),
            "image_size": int(cfg.image_size),
            "image_channels": int(cfg.image_channels),
            "geometry_size": int(cfg.geometry_size),
            "max_neighbors": int(cfg.max_neighbors),
            "chunk_size": int(cfg.chunk_size),
            "fill_policy": str(cfg.fill_policy),
            "is_spatially_sorted": bool(cfg.is_spatially_sorted),
            "zarr_format": int(cfg.zarr_format),
        }
    )
    for name, array_spec in _array_contract(cfg).items():
        root.create_array(
            name,
            shape=array_spec["shape"],
            chunks=array_spec["chunks"],
            dtype=array_spec["dtype"],
            overwrite=True,
        )
    return root


def validate_contour_image_store(store_path: str | Path, *, spec: ContourStoreSpec | None = None) -> dict[str, Any]:
    root = zarr.open_group(str(store_path), mode="r")
    cfg = spec or _spec_from_attrs(root.attrs)
    if root.attrs.get("contract_version") != cfg.contract_version:
        raise ValueError(
            f"Contour store contract_version={root.attrs.get('contract_version')!r} "
            f"does not match expected {cfg.contract_version!r}"
        )
    if bool(root.attrs.get("is_spatially_sorted")) is not True:
        raise ValueError("Contour store must declare is_spatially_sorted=True")
    for attr in ("n_contours", "image_size", "image_channels", "geometry_size", "chunk_size"):
        actual = int(root.attrs.get(attr, -1))
        expected = int(getattr(cfg, attr))
        if actual != expected:
            raise ValueError(f"Contour store attr {attr}={actual} does not match expected {expected}")
    for name, array_spec in _array_contract(cfg).items():
        if name not in root:
            raise ValueError(f"Contour store is missing required array {name!r}")
        array = root[name]
        actual_shape = tuple(int(item) for item in array.shape)
        actual_chunks = tuple(int(item) for item in array.chunks)
        expected_shape = tuple(int(item) for item in array_spec["shape"])
        expected_chunks = tuple(int(item) for item in array_spec["chunks"])
        if actual_shape != expected_shape:
            raise ValueError(f"Array {name!r} shape {actual_shape} does not match expected {expected_shape}")
        if actual_chunks != expected_chunks:
            raise ValueError(f"Array {name!r} chunks {actual_chunks} does not match expected {expected_chunks}")
        if not _dtype_matches(array.dtype, array_spec["dtype"]):
            raise ValueError(f"Array {name!r} dtype {array.dtype} does not match expected {array_spec['dtype']}")
    return {
        "store_path": str(store_path),
        "contract_version": cfg.contract_version,
        "n_contours": cfg.n_contours,
        "arrays": sorted(_array_contract(cfg)),
    }


def read_contour_manifest(path: str | Path) -> pa.Table:
    return pq.read_table(path)


def write_contour_manifest(table: pa.Table, path: str | Path, *, max_neighbors: int | None = None) -> Path:
    validate_contour_manifest(table, max_neighbors=max_neighbors)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output)
    return output


def validate_contour_manifest(table_or_path: pa.Table | str | Path, *, max_neighbors: int | None = None) -> dict[str, Any]:
    table = read_contour_manifest(table_or_path) if isinstance(table_or_path, (str, Path)) else table_or_path
    expected_neighbors = int(max_neighbors or _infer_max_neighbors(table.schema))
    expected_schema = contour_manifest_schema(expected_neighbors)
    _validate_manifest_schema(table.schema, expected_schema, expected_neighbors)

    n_rows = table.num_rows
    required = [field.name for field in expected_schema]
    for name in required:
        if table.column(name).null_count:
            raise ValueError(f"Manifest field {name!r} contains null values")

    row_indices = np.asarray(table.column("row_index").to_pylist(), dtype=np.int64)
    expected_rows = np.arange(n_rows, dtype=np.int64)
    if not np.array_equal(row_indices, expected_rows):
        raise ValueError("Manifest row_index must be contiguous, unique, and sorted from 0 to n_rows - 1")

    chunk_ids = np.asarray(table.column("chunk_id").to_pylist(), dtype=np.int64)
    if np.any(chunk_ids < 0):
        raise ValueError("Manifest chunk_id values must be non-negative")

    slide_ids = [str(item) for item in table.column("slide_id").to_pylist()]
    neighbor_rows = table.column("neighbor_row_indices").to_pylist()
    neighbor_masks = table.column("neighbor_valid_mask").to_pylist()
    neighbor_distances = table.column("neighbor_distances").to_pylist()
    neighbor_offsets = table.column("neighbor_offsets_xy").to_pylist()
    _validate_neighbors(
        slide_ids=slide_ids,
        neighbor_rows=neighbor_rows,
        neighbor_masks=neighbor_masks,
        neighbor_distances=neighbor_distances,
        neighbor_offsets=neighbor_offsets,
        max_neighbors=expected_neighbors,
    )
    return {
        "n_rows": n_rows,
        "max_neighbors": expected_neighbors,
        "slide_count": len(set(slide_ids)),
        "schema": "ok",
    }


def build_mock_contour_store(
    store_path: str | Path,
    manifest_path: str | Path,
    *,
    spec: ContourStoreSpec,
    slide_ids: Sequence[str] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    root = create_contour_image_store(store_path, spec=spec, overwrite=True)
    rng = np.random.default_rng(seed)
    root[OBJECT_RGB_KEY][:] = rng.integers(
        0,
        256,
        size=(spec.n_contours, spec.image_size, spec.image_size, spec.image_channels),
        dtype=np.uint8,
    )
    root[CONTEXT_RGB_KEY][:] = rng.integers(
        0,
        256,
        size=(spec.n_contours, spec.image_size, spec.image_size, spec.image_channels),
        dtype=np.uint8,
    )
    root[SOFT_MASK_KEY][:] = _mock_masks(spec)
    root[GEOMETRY_KEY][:] = _mock_geometry(spec)
    contour_ids = np.asarray([f"contour_{idx:06d}".encode() for idx in range(spec.n_contours)], dtype=CONTOUR_ID_DTYPE)
    root[CONTOUR_IDS_KEY][:] = contour_ids

    table = build_mock_contour_manifest(spec, slide_ids=slide_ids)
    output_manifest = write_contour_manifest(table, manifest_path, max_neighbors=spec.max_neighbors)
    store_validation = validate_contour_image_store(store_path, spec=spec)
    manifest_validation = validate_contour_manifest(output_manifest, max_neighbors=spec.max_neighbors)
    return {
        "store_path": str(store_path),
        "manifest_path": str(output_manifest),
        "store_validation": store_validation,
        "manifest_validation": manifest_validation,
    }


def build_mock_contour_manifest(
    spec: ContourStoreSpec,
    *,
    slide_ids: Sequence[str] | None = None,
) -> pa.Table:
    row_slides = _mock_slide_ids(spec.n_contours, slide_ids)
    centroids = _mock_centroids(row_slides)
    bbox = [[float(x - 5.0), float(y - 5.0), float(x + 5.0), float(y + 5.0)] for x, y in centroids]
    builder = SlideNeighborGraphBuilder(
        NeighborGraphConfig(max_neighbors=spec.max_neighbors, mode="knn", chunk_size=spec.chunk_size)
    )
    return builder.build(
        contour_ids=[f"contour_{idx:06d}" for idx in range(spec.n_contours)],
        slide_ids=row_slides,
        centroids=centroids,
        bbox_level0_xy=bbox,
        area=[100.0 + float(idx % 7) for idx in range(spec.n_contours)],
        perimeter=[40.0 + float(idx % 5) for idx in range(spec.n_contours)],
        eccentricity=[float((idx % 10) / 10.0) for idx in range(spec.n_contours)],
        qc_flag=["ok"] * spec.n_contours,
        transform_fingerprint=["mock_transform"] * spec.n_contours,
    )


def pack_contour_patches(
    patch_manifest: str | Path,
    store_path: str | Path,
    manifest_path: str | Path,
    *,
    slide_id: str,
    image_size: int = DEFAULT_IMAGE_SIZE,
    max_neighbors: int = DEFAULT_MAX_NEIGHBORS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overwrite: bool = True,
) -> dict[str, Any]:
    rows = _read_patch_manifest_rows(patch_manifest)
    if not rows:
        raise ValueError(f"Patch manifest contains no rows: {patch_manifest}")
    spec = ContourStoreSpec(
        n_contours=len(rows),
        image_size=image_size,
        geometry_size=DEFAULT_GEOMETRY_SIZE,
        max_neighbors=max_neighbors,
        chunk_size=chunk_size,
    )
    object_rgb_array = np.zeros((len(rows), image_size, image_size, 3), dtype=np.uint8)
    mask_array = np.zeros((len(rows), image_size, image_size, 1), dtype=np.uint8)
    geometry_array = np.zeros((len(rows), DEFAULT_GEOMETRY_SIZE), dtype=np.float32)
    contour_id_array = np.zeros((len(rows),), dtype=CONTOUR_ID_DTYPE)
    contour_ids: list[str] = []
    centroids: list[list[float]] = []
    bbox_rows: list[list[float]] = []
    areas: list[float] = []
    perimeters: list[float] = []
    eccentricities: list[float] = []
    qc_flags: list[str] = []
    transforms: list[str] = []
    for row_idx, row in enumerate(rows):
        contour_id = str(row.get("contour_id") or row.get("region_id") or f"contour_{row_idx:06d}")
        image_path = _patch_image_path(row)
        object_rgb, mask = _load_patch_rgb_and_mask(image_path, image_size=image_size)
        geometry, bbox = _geometry_from_patch_row(row, object_rgb=object_rgb, mask=mask)
        object_rgb_array[row_idx] = object_rgb
        mask_array[row_idx] = mask
        geometry_array[row_idx] = geometry
        contour_id_array[row_idx] = contour_id.encode()[: np.dtype(CONTOUR_ID_DTYPE).itemsize]
        contour_ids.append(contour_id)
        centroids.append([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])
        bbox_rows.append(bbox)
        areas.append(float(geometry[0]))
        perimeters.append(float(geometry[1]))
        eccentricities.append(float(geometry[2]))
        qc_flags.append("ok" if image_path.exists() else "missing_image")
        transforms.append(str(row.get("registration_transform") or row.get("transform") or row.get("source_geojson") or "unknown_transform"))

    root = create_contour_image_store(store_path, spec=spec, overwrite=overwrite)
    root[OBJECT_RGB_KEY][:] = object_rgb_array
    root[CONTEXT_RGB_KEY][:] = object_rgb_array
    root[SOFT_MASK_KEY][:] = mask_array
    root[GEOMETRY_KEY][:] = geometry_array
    root[CONTOUR_IDS_KEY][:] = contour_id_array

    builder = SlideNeighborGraphBuilder(
        NeighborGraphConfig(max_neighbors=max_neighbors, mode="knn", chunk_size=chunk_size)
    )
    table = builder.build(
        contour_ids=contour_ids,
        slide_ids=[slide_id] * len(contour_ids),
        centroids=centroids,
        bbox_level0_xy=bbox_rows,
        area=areas,
        perimeter=perimeters,
        eccentricity=eccentricities,
        qc_flag=qc_flags,
        transform_fingerprint=transforms,
    )
    output_manifest = write_contour_manifest(table, manifest_path, max_neighbors=max_neighbors)
    return {
        "store_path": str(store_path),
        "manifest_path": str(output_manifest),
        "n_contours": len(contour_ids),
        "image_size": int(image_size),
        "max_neighbors": int(max_neighbors),
        "store_validation": validate_contour_image_store(store_path, spec=spec),
        "manifest_validation": validate_contour_manifest(output_manifest, max_neighbors=max_neighbors),
    }


def _array_contract(spec: ContourStoreSpec) -> dict[str, dict[str, Any]]:
    n = int(spec.n_contours)
    h = int(spec.image_size)
    c = int(spec.image_channels)
    chunk = int(spec.chunk_size)
    return {
        OBJECT_RGB_KEY: {"shape": (n, h, h, c), "chunks": (chunk, h, h, c), "dtype": "uint8"},
        CONTEXT_RGB_KEY: {"shape": (n, h, h, c), "chunks": (chunk, h, h, c), "dtype": "uint8"},
        SOFT_MASK_KEY: {"shape": (n, h, h, 1), "chunks": (chunk, h, h, 1), "dtype": "uint8"},
        GEOMETRY_KEY: {"shape": (n, int(spec.geometry_size)), "chunks": (chunk, int(spec.geometry_size)), "dtype": "float32"},
        CONTOUR_IDS_KEY: {"shape": (n,), "chunks": (chunk,), "dtype": CONTOUR_ID_DTYPE},
    }


def _read_patch_manifest_rows(path: str | Path) -> list[dict[str, Any]]:
    payload = Path(path).read_text(encoding="utf-8")
    data = json.loads(payload)
    rows = data if isinstance(data, list) else data.get("patches", data.get("records", [])) if isinstance(data, dict) else []
    return [dict(row) for row in rows if isinstance(row, dict)]


def _patch_image_path(row: dict[str, Any]) -> Path:
    patch = row.get("patch") if isinstance(row.get("patch"), dict) else {}
    value = row.get("image_path") or patch.get("path") or patch.get("image_path")
    return Path(str(value)).expanduser() if value else Path("__missing_patch_image__")


def _load_patch_rgb_and_mask(path: Path, *, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        rgb = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    else:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR), dtype=np.uint8)
    mask = (rgb.max(axis=2, keepdims=True) > 0).astype(np.uint8) * 255
    if not mask.any():
        mask = np.ones((image_size, image_size, 1), dtype=np.uint8) * 255
    return rgb, mask


def _geometry_from_patch_row(row: dict[str, Any], *, object_rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, list[float]]:
    bbox = _bbox_from_patch_row(row)
    width = max(1.0, float(bbox[2] - bbox[0]))
    height = max(1.0, float(bbox[3] - bbox[1]))
    area = float(width * height)
    perimeter = float(2.0 * (width + height))
    major = max(width, height)
    minor = min(width, height)
    eccentricity = float(np.sqrt(max(0.0, 1.0 - (minor / max(major, 1e-6)) ** 2)))
    nonzero_fraction = float((mask > 0).mean())
    mean_intensity = float(object_rgb.mean() / 255.0)
    geometry = np.asarray(
        [area, perimeter, eccentricity, width, height, nonzero_fraction, mean_intensity, 1.0],
        dtype=np.float32,
    )
    return geometry, [float(value) for value in bbox]


def _bbox_from_patch_row(row: dict[str, Any]) -> list[float]:
    patch = row.get("patch") if isinstance(row.get("patch"), dict) else {}
    bbox = row.get("bbox_level0_xy") or patch.get("bbox_level0_xy") or patch.get("bbox_level_xy")
    if isinstance(bbox, list | tuple) and len(bbox) >= 4:
        return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
    nested = row.get("bbox")
    if isinstance(nested, dict) and all(key in nested for key in ("x0", "y0", "x1", "y1")):
        return [float(nested["x0"]), float(nested["y0"]), float(nested["x1"]), float(nested["y1"])]
    width = float(patch.get("saved_width") or patch.get("original_width") or 1.0)
    height = float(patch.get("saved_height") or patch.get("original_height") or 1.0)
    return [0.0, 0.0, width, height]


def _spec_from_attrs(attrs) -> ContourStoreSpec:
    return ContourStoreSpec(
        n_contours=int(attrs["n_contours"]),
        image_size=int(attrs["image_size"]),
        image_channels=int(attrs["image_channels"]),
        geometry_size=int(attrs["geometry_size"]),
        max_neighbors=int(attrs.get("max_neighbors", DEFAULT_MAX_NEIGHBORS)),
        chunk_size=int(attrs["chunk_size"]),
        contract_version=str(attrs["contract_version"]),
        zarr_format=int(attrs.get("zarr_format", 2)),
        fill_policy=str(attrs.get("fill_policy", "slide_tissue_mean")),
        is_spatially_sorted=bool(attrs.get("is_spatially_sorted", True)),
    )


def _validate_max_neighbors(max_neighbors: int) -> None:
    if int(max_neighbors) < 1:
        raise ValueError("max_neighbors must be >= 1")


def _validate_manifest_schema(schema: pa.Schema, expected_schema: pa.Schema, max_neighbors: int) -> None:
    field_names = set(schema.names)
    for expected in expected_schema:
        if expected.name not in field_names:
            raise ValueError(f"Manifest is missing required field {expected.name!r}")
        actual = schema.field(expected.name)
        if actual.type != expected.type:
            raise ValueError(f"Manifest field {expected.name!r} has type {actual.type}, expected {expected.type}")
    for name, size in {
        "bbox_level0_xy": 4,
        "neighbor_row_indices": max_neighbors,
        "neighbor_distances": max_neighbors,
        "neighbor_offsets_xy": max_neighbors * 2,
        "neighbor_valid_mask": max_neighbors,
    }.items():
        actual_type = schema.field(name).type
        if not pa.types.is_fixed_size_list(actual_type) or actual_type.list_size != size:
            raise ValueError(f"Manifest field {name!r} must be fixed_size_list with length {size}")


def _infer_max_neighbors(schema: pa.Schema) -> int:
    if "neighbor_row_indices" not in schema.names:
        raise ValueError("Manifest is missing neighbor_row_indices; cannot infer max_neighbors")
    field_type = schema.field("neighbor_row_indices").type
    if not pa.types.is_fixed_size_list(field_type):
        raise ValueError("neighbor_row_indices must be a fixed_size_list field")
    return int(field_type.list_size)


def _validate_neighbors(
    *,
    slide_ids: list[str],
    neighbor_rows: list[list[int]],
    neighbor_masks: list[list[bool]],
    neighbor_distances: list[list[float]],
    neighbor_offsets: list[list[float]],
    max_neighbors: int,
) -> None:
    n_rows = len(slide_ids)
    for row_idx, (indices, masks, distances, offsets) in enumerate(
        zip(neighbor_rows, neighbor_masks, neighbor_distances, neighbor_offsets, strict=True)
    ):
        if len(indices) != max_neighbors or len(masks) != max_neighbors or len(distances) != max_neighbors:
            raise ValueError(f"Neighbor arrays at row {row_idx} do not match max_neighbors={max_neighbors}")
        if len(offsets) != max_neighbors * 2:
            raise ValueError(f"neighbor_offsets_xy at row {row_idx} must have length {max_neighbors * 2}")
        for pos, (neighbor_idx, valid) in enumerate(zip(indices, masks, strict=True)):
            neighbor_idx = int(neighbor_idx)
            if valid:
                if neighbor_idx < 0 or neighbor_idx >= n_rows:
                    raise ValueError(f"Invalid neighbor row index {neighbor_idx} at row {row_idx}, position {pos}")
                if slide_ids[neighbor_idx] != slide_ids[row_idx]:
                    raise ValueError(f"Neighbor row {neighbor_idx} crosses slide boundary at row {row_idx}")
            elif neighbor_idx != -1:
                raise ValueError(f"Invalid padded neighbor value {neighbor_idx}; expected -1 at row {row_idx}, position {pos}")


def _dtype_matches(actual: np.dtype, expected: str) -> bool:
    actual_dtype = np.dtype(actual)
    expected_dtype = np.dtype(expected)
    if expected == CONTOUR_ID_DTYPE:
        return actual_dtype.kind in {"S", "U"} and actual_dtype.itemsize >= expected_dtype.itemsize
    return actual_dtype == expected_dtype


def _validate_builder_inputs(
    contour_ids: Sequence[str],
    slide_ids: Sequence[str],
    centroids: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[list[str], list[str], np.ndarray]:
    ids = [str(item) for item in contour_ids]
    slides = [str(item) for item in slide_ids]
    coords = np.asarray(centroids, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("centroids must have shape [n_contours, 2]")
    if len(ids) != coords.shape[0] or len(slides) != coords.shape[0]:
        raise ValueError("contour_ids, slide_ids, and centroids must have the same length")
    if len(set(ids)) != len(ids):
        raise ValueError("contour_ids must be unique")
    if not np.isfinite(coords).all():
        raise ValueError("centroids must contain only finite coordinates")
    return ids, slides, coords


def morton_sort_keys(coords: Sequence[Sequence[float]] | np.ndarray, *, bits: int = 21) -> np.ndarray:
    """Return deterministic 2D Morton Z-order keys for physical coordinates."""
    if int(bits) < 1 or int(bits) > 31:
        raise ValueError("bits must be between 1 and 31")
    values = np.asarray(coords, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("coords must have shape [n, 2]")
    if values.shape[0] == 0:
        return np.zeros(0, dtype=np.uint64)
    if not np.isfinite(values).all():
        raise ValueError("coords must contain only finite values")
    mins = values.min(axis=0)
    spans = values.max(axis=0) - mins
    spans[spans <= 0.0] = 1.0
    max_value = (1 << int(bits)) - 1
    quantized = np.floor((values - mins) / spans * max_value).astype(np.uint64)
    quantized = np.clip(quantized, 0, max_value).astype(np.uint64)
    keys = np.zeros(values.shape[0], dtype=np.uint64)
    for idx, (x_value, y_value) in enumerate(quantized):
        keys[idx] = _interleave_2d_bits(int(x_value), int(y_value), int(bits))
    return keys


def _interleave_2d_bits(x_value: int, y_value: int, bits: int) -> int:
    key = 0
    for bit in range(bits):
        key |= ((x_value >> bit) & 1) << (2 * bit)
        key |= ((y_value >> bit) & 1) << (2 * bit + 1)
    return int(key)


def _spatial_sort_order(slide_ids: Sequence[str], coords: np.ndarray, *, bits: int) -> tuple[np.ndarray, np.ndarray]:
    spatial_keys = np.zeros(len(slide_ids), dtype=np.uint64)
    ordered: list[int] = []
    ids_array = np.asarray(slide_ids, dtype=object)
    for slide_id in sorted(set(slide_ids)):
        indices = np.flatnonzero(ids_array == slide_id)
        keys = morton_sort_keys(coords[indices], bits=bits)
        spatial_keys[indices] = keys
        local_contour_order = np.lexsort((indices, keys))
        ordered.extend(indices[local_contour_order].astype(int).tolist())
    return np.asarray(ordered, dtype=np.int64), spatial_keys


def _ordered_bbox(
    coords: np.ndarray,
    order: np.ndarray,
    bbox_level0_xy: Sequence[Sequence[float]] | np.ndarray | None,
) -> list[list[float]]:
    if bbox_level0_xy is None:
        return [[float(x - 0.5), float(y - 0.5), float(x + 0.5), float(y + 0.5)] for x, y in coords[order]]
    values = np.asarray(bbox_level0_xy, dtype=np.float32)
    if values.shape != (coords.shape[0], 4):
        raise ValueError("bbox_level0_xy must have shape [n_contours, 4]")
    if not np.isfinite(values).all():
        raise ValueError("bbox_level0_xy must contain only finite values")
    return [[float(item) for item in row] for row in values[order]]


def _ordered_float_values(values: Sequence[float] | np.ndarray | None, order: np.ndarray, *, default: float) -> list[float]:
    if values is None:
        return [float(default)] * len(order)
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (len(order),):
        raise ValueError("metadata float arrays must have shape [n_contours]")
    if not np.isfinite(array).all():
        raise ValueError("metadata float arrays must contain only finite values")
    return [float(item) for item in array[order]]


def _ordered_str_values(values: Sequence[str] | None, order: np.ndarray, *, default: str) -> list[str]:
    if values is None:
        return [str(default)] * len(order)
    items = [str(item) for item in values]
    if len(items) != len(order):
        raise ValueError("metadata string arrays must have length n_contours")
    return [items[idx] for idx in order]


def _unique_in_order(values: Sequence[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if value not in seen:
            ordered.append(str(value))
            seen.add(value)
    return ordered


def _select_neighbor_candidates(
    *,
    anchor: np.ndarray,
    coords: np.ndarray,
    candidate_pairs: Sequence[tuple[int, float]],
    max_neighbors: int,
    sampling: Literal["nearest", "angular"],
) -> list[tuple[int, float]]:
    unique: dict[int, float] = {}
    for idx, distance in candidate_pairs:
        if idx not in unique or distance < unique[idx]:
            unique[int(idx)] = float(distance)
    pairs = sorted(unique.items(), key=lambda item: (item[1], item[0]))
    if sampling == "nearest" or len(pairs) <= max_neighbors:
        return pairs[:max_neighbors]
    return _angular_select(anchor=anchor, coords=coords, pairs=pairs, max_neighbors=max_neighbors)


def _angular_select(
    *,
    anchor: np.ndarray,
    coords: np.ndarray,
    pairs: Sequence[tuple[int, float]],
    max_neighbors: int,
) -> list[tuple[int, float]]:
    selected: list[tuple[int, float]] = []
    used: set[int] = set()
    bins: list[list[tuple[int, float]]] = [[] for _ in range(max_neighbors)]
    for idx, distance in pairs:
        delta = coords[idx] - anchor
        angle = float(np.arctan2(delta[1], delta[0]))
        if angle < 0:
            angle += float(2.0 * np.pi)
        bin_position = angle / (2.0 * np.pi) * max_neighbors
        nearest_boundary = round(bin_position)
        if abs(bin_position - nearest_boundary) < 1e-6:
            bin_position = float(nearest_boundary)
        bin_idx = min(max_neighbors - 1, int(np.floor(bin_position)))
        bins[bin_idx].append((idx, distance))
    for bucket in bins:
        if not bucket:
            continue
        idx, distance = sorted(bucket, key=lambda item: (item[1], item[0]))[0]
        selected.append((idx, distance))
        used.add(idx)
    for idx, distance in pairs:
        if len(selected) >= max_neighbors:
            break
        if idx in used:
            continue
        selected.append((idx, distance))
        used.add(idx)
    return sorted(selected[:max_neighbors], key=lambda item: (item[1], item[0]))


def _write_padded_neighbors(
    *,
    global_anchor: int,
    local_anchor: int,
    slide_rows: np.ndarray,
    slide_coords: np.ndarray,
    selected: Sequence[tuple[int, float]],
    row_indices: list[list[int]],
    distances: list[list[float]],
    offsets_xy: list[list[float]],
    valid_mask: list[list[bool]],
) -> None:
    for pos, (local_neighbor, distance) in enumerate(selected):
        global_neighbor = int(slide_rows[int(local_neighbor)])
        offset = slide_coords[int(local_neighbor)] - slide_coords[int(local_anchor)]
        row_indices[global_anchor][pos] = global_neighbor
        distances[global_anchor][pos] = float(distance)
        offsets_xy[global_anchor][2 * pos] = float(offset[0])
        offsets_xy[global_anchor][2 * pos + 1] = float(offset[1])
        valid_mask[global_anchor][pos] = True


def _mock_masks(spec: ContourStoreSpec) -> np.ndarray:
    h = int(spec.image_size)
    yy, xx = np.ogrid[:h, :h]
    center = (h - 1) / 2.0
    radius = max(1.0, h / 3.0)
    mask = (((xx - center) ** 2 + (yy - center) ** 2) <= radius**2).astype(np.uint8) * 255
    return np.broadcast_to(mask[None, :, :, None], (spec.n_contours, h, h, 1)).copy()


def _mock_geometry(spec: ContourStoreSpec) -> np.ndarray:
    values = np.zeros((spec.n_contours, spec.geometry_size), dtype=np.float32)
    if spec.n_contours == 0:
        return values
    row = np.arange(spec.n_contours, dtype=np.float32)
    for col in range(spec.geometry_size):
        values[:, col] = (row + col) / max(1, spec.n_contours)
    return values


def _mock_slide_ids(n_contours: int, slide_ids: Sequence[str] | None) -> list[str]:
    if n_contours == 0:
        return []
    if slide_ids is None:
        return ["mock_slide"] * n_contours
    items = [str(item) for item in slide_ids]
    if len(items) == n_contours:
        return items
    if not items:
        raise ValueError("slide_ids must not be empty")
    return [items[min(len(items) - 1, idx * len(items) // n_contours)] for idx in range(n_contours)]


def _mock_centroids(slide_ids: Sequence[str]) -> np.ndarray:
    coords = np.zeros((len(slide_ids), 2), dtype=np.float32)
    slide_to_rows: dict[str, list[int]] = {}
    for idx, slide_id in enumerate(slide_ids):
        slide_to_rows.setdefault(str(slide_id), []).append(idx)
    for slide_idx, (_slide_id, rows) in enumerate(sorted(slide_to_rows.items())):
        side = int(np.ceil(np.sqrt(max(1, len(rows)))))
        for local_idx, row_idx in enumerate(rows):
            coords[row_idx, 0] = float((local_idx % side) * 10.0 + slide_idx * 10_000.0)
            coords[row_idx, 1] = float((local_idx // side) * 10.0)
    return coords
