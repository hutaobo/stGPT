from __future__ import annotations

import json
import math
import os
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import zarr
from PIL import Image


def build_contour_panel(
    evidence_chain: str | Path,
    output_dir: str | Path,
    *,
    sample_size: int = 12,
    sort_by: str = "low_confidence",
    top_genes: int = 8,
) -> dict[str, Any]:
    """Build an artifact-first contour-native visual evidence panel.

    The panel follows M7 evidence pointers into Zarr/Parquet artifacts and writes
    only lightweight thumbnails plus manifest metadata. It does not run training
    or inference and it keeps matrices out of JSON.
    """
    if sample_size < 0:
        raise ValueError("sample_size must be non-negative")
    if top_genes < 0:
        raise ValueError("top_genes must be non-negative")
    chain_path = Path(evidence_chain).expanduser()
    out = Path(output_dir).expanduser()
    tiles = out / "tiles"
    out.mkdir(parents=True, exist_ok=True)
    tiles.mkdir(parents=True, exist_ok=True)

    records = _select_records(_read_jsonl(chain_path), sample_size=sample_size, sort_by=sort_by)
    zarr_cache: dict[Path, zarr.Group] = {}
    manifest_rows: list[dict[str, Any]] = []
    for panel_index, record in enumerate(records):
        row = _build_panel_row(
            record,
            panel_index=panel_index,
            base_dir=chain_path.parent,
            tiles_dir=tiles,
            zarr_cache=zarr_cache,
            top_genes=top_genes,
        )
        manifest_rows.append(row)

    manifest_csv = out / "contour_panel_manifest.csv"
    manifest_json = out / "contour_panel_manifest.json"
    panel_html = out / "contour_panel.html"
    pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False)
    manifest_json.write_text(json.dumps(_json_safe(manifest_rows), indent=2), encoding="utf-8")
    panel_html.write_text(_panel_html(chain_path, manifest_rows), encoding="utf-8")

    return {
        "evidence_chain": str(chain_path),
        "n_records": len(records),
        "artifacts": {
            "contour_panel_html": str(panel_html),
            "contour_panel_manifest_csv": str(manifest_csv),
            "contour_panel_manifest_json": str(manifest_json),
            "tiles_dir": str(tiles),
        },
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _select_records(records: list[dict[str, Any]], *, sample_size: int, sort_by: str) -> list[dict[str, Any]]:
    if sample_size == 0:
        return []
    if sort_by == "first":
        selected = records
    elif sort_by == "high_confidence":
        selected = sorted(records, key=lambda record: (_prototype_confidence(record) is None, -(_prototype_confidence(record) or -1.0)))
    elif sort_by == "low_confidence":
        selected = sorted(records, key=lambda record: (_prototype_confidence(record) is None, _prototype_confidence(record) or math.inf))
    else:
        raise ValueError("sort_by must be one of: first, low_confidence, high_confidence")
    return selected[:sample_size]


def _build_panel_row(
    record: dict[str, Any],
    *,
    panel_index: int,
    base_dir: Path,
    tiles_dir: Path,
    zarr_cache: dict[Path, zarr.Group],
    top_genes: int,
) -> dict[str, Any]:
    unit = record.get("unit", {}) if isinstance(record.get("unit"), dict) else {}
    measured = record.get("measured_evidence", {}) if isinstance(record.get("measured_evidence"), dict) else {}
    model = record.get("model_derived_evidence", {}) if isinstance(record.get("model_derived_evidence"), dict) else {}
    qc = record.get("qc_verdict", {}) if isinstance(record.get("qc_verdict"), dict) else {}
    provenance = record.get("provenance", {}) if isinstance(record.get("provenance"), dict) else {}
    proto = model.get("prototype_ref", {}) if isinstance(model.get("prototype_ref"), dict) else {}

    image_bundle = _load_image_bundle(record, base_dir=base_dir, zarr_cache=zarr_cache)
    prefix = f"panel_{panel_index:03d}_{_slug(str(unit.get('contour_id') or unit.get('region_id') or panel_index))}"
    object_path = tiles_dir / f"{prefix}_object.png"
    context_path = tiles_dir / f"{prefix}_context.png"
    mask_path = tiles_dir / f"{prefix}_mask.png"
    overlay_path = tiles_dir / f"{prefix}_overlay.png"
    _save_rgb(image_bundle["object_rgb"], object_path)
    _save_rgb(image_bundle["context_rgb"], context_path)
    _save_mask(image_bundle["mask"], mask_path)
    _save_rgb(_overlay_mask(image_bundle["object_rgb"], image_bundle["mask"]), overlay_path)

    top_gene_rows = _top_gene_rows(measured.get("molecular_ref"), base_dir=base_dir, top_n=top_genes)
    spatial = measured.get("spatial", {}) if isinstance(measured.get("spatial"), dict) else {}
    row = {
        "panel_index": panel_index,
        "evidence_id": record.get("evidence_id"),
        "contour_id": unit.get("contour_id") or unit.get("region_id"),
        "slide_id": unit.get("slide_id"),
        "row_index": unit.get("row_index"),
        "embedding_row_index": unit.get("embedding_row_index"),
        "x": spatial.get("x"),
        "y": spatial.get("y"),
        "image_source": qc.get("image_source"),
        "qc_flag": qc.get("qc_flag"),
        "object_image": _relative_tile(object_path, tiles_dir.parent),
        "context_image": _relative_tile(context_path, tiles_dir.parent),
        "mask_image": _relative_tile(mask_path, tiles_dir.parent),
        "overlay_image": _relative_tile(overlay_path, tiles_dir.parent),
        "geometry_preview": json.dumps(image_bundle.get("geometry_preview", [])),
        "top_genes": json.dumps(top_gene_rows),
        "prototype_id": proto.get("prototype_id"),
        "prototype_confidence": proto.get("confidence"),
        "assignment_entropy": proto.get("assignment_entropy"),
        "embedding_ref": json.dumps(model.get("embedding_ref", {}), sort_keys=True),
        "molecular_ref": json.dumps(measured.get("molecular_ref", {}), sort_keys=True),
        "image_ref": json.dumps(measured.get("image_ref", {}), sort_keys=True),
        "checkpoint_hash": provenance.get("checkpoint_hash"),
        "config_hash": provenance.get("config_hash"),
        "contour_manifest_hash": provenance.get("contour_manifest_hash"),
    }
    return row


def _load_image_bundle(record: dict[str, Any], *, base_dir: Path, zarr_cache: dict[Path, zarr.Group]) -> dict[str, Any]:
    measured = record.get("measured_evidence", {}) if isinstance(record.get("measured_evidence"), dict) else {}
    image_ref = measured.get("image_ref", {}) if isinstance(measured.get("image_ref"), dict) else {}
    qc = record.get("qc_verdict", {}) if isinstance(record.get("qc_verdict"), dict) else {}
    source = str(qc.get("image_source") or "unknown")
    if source == "contour_store":
        artifact = _string_or_none(image_ref.get("artifact"))
        row_index = _safe_int(image_ref.get("row_index"))
        arrays = image_ref.get("arrays", {}) if isinstance(image_ref.get("arrays"), dict) else {}
        if artifact is not None and row_index is not None:
            store_path = _resolve_artifact(artifact, base_dir)
            try:
                root = zarr_cache.setdefault(store_path, zarr.open_group(str(store_path), mode="r"))
                object_rgb = _as_rgb(root[str(arrays.get("object_rgb") or "object_rgb")][row_index])
                context_rgb = _as_rgb(root[str(arrays.get("context_rgb") or "context_rgb")][row_index])
                mask = _as_mask(root[str(arrays.get("mask") or "soft_mask")][row_index])
                geometry = _read_zarr_geometry(root, row_index)
                return {
                    "object_rgb": object_rgb,
                    "context_rgb": context_rgb,
                    "mask": mask,
                    "geometry_preview": geometry,
                }
            except Exception:
                pass
    if source == "image_path":
        artifact = _string_or_none(image_ref.get("artifact"))
        if artifact is not None:
            path = _resolve_artifact(artifact, base_dir)
            try:
                with Image.open(path) as image:
                    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
                mask = np.ones(rgb.shape[:2] + (1,), dtype=np.uint8) * 255
                return {"object_rgb": rgb, "context_rgb": rgb, "mask": mask, "geometry_preview": []}
            except Exception:
                pass
    return _placeholder_bundle()


def _read_zarr_geometry(root: zarr.Group, row_index: int) -> list[float]:
    if "geometry" not in root:
        return []
    try:
        values = np.asarray(root["geometry"][row_index], dtype=np.float32).reshape(-1)
    except Exception:
        return []
    return [float(value) for value in values[:8] if np.isfinite(value)]


def _top_gene_rows(molecular_ref: Any, *, base_dir: Path, top_n: int) -> list[dict[str, Any]]:
    if top_n <= 0 or not isinstance(molecular_ref, dict):
        return []
    artifact = _string_or_none(molecular_ref.get("artifact"))
    row_index = _safe_int(molecular_ref.get("row_index"))
    if artifact is None or row_index is None:
        return []
    path = _resolve_artifact(artifact, base_dir)
    row = _read_parquet_row(path, row_index)
    gene_values: list[tuple[str, float]] = []
    for key, value in row.items():
        if not str(key).startswith("gene_"):
            continue
        number = _safe_float(value)
        if number is not None:
            gene_values.append((str(key)[5:], number))
    gene_values.sort(key=lambda item: (-item[1], item[0]))
    return [{"gene": gene, "value": value} for gene, value in gene_values[:top_n]]


def _read_parquet_row(path: Path, row_index: int) -> dict[str, Any]:
    if not path.exists() or row_index < 0:
        return {}
    try:
        parquet_file = pq.ParquetFile(path)
        offset = 0
        for group_idx in range(parquet_file.num_row_groups):
            group_rows = parquet_file.metadata.row_group(group_idx).num_rows
            if row_index < offset + group_rows:
                table = parquet_file.read_row_group(group_idx)
                local = row_index - offset
                data = table.slice(local, 1).to_pydict()
                return {key: values[0] if values else None for key, values in data.items()}
            offset += group_rows
    except Exception:
        return {}
    return {}


def _save_rgb(array: np.ndarray, path: Path) -> None:
    Image.fromarray(_as_rgb(array)).save(path)


def _save_mask(mask: np.ndarray, path: Path) -> None:
    value = _as_mask(mask)
    Image.fromarray(value[:, :, 0]).save(path)


def _overlay_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image = _as_rgb(rgb).astype(np.float32)
    alpha = (_as_mask(mask).astype(np.float32) / 255.0) * 0.35
    red = np.zeros_like(image)
    red[:, :, 0] = 255.0
    return np.clip(image * (1.0 - alpha) + red * alpha, 0, 255).astype(np.uint8)


def _placeholder_bundle(size: int = 64) -> dict[str, Any]:
    object_rgb = np.full((size, size, 3), 230, dtype=np.uint8)
    context_rgb = np.full((size, size, 3), 210, dtype=np.uint8)
    mask = np.zeros((size, size, 1), dtype=np.uint8)
    return {"object_rgb": object_rgb, "context_rgb": context_rgb, "mask": mask, "geometry_preview": []}


def _as_rgb(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3:
        return np.full((64, 64, 3), 230, dtype=np.uint8)
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    if array.shape[2] > 3:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _as_mask(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3:
        return np.zeros((64, 64, 1), dtype=np.uint8)
    if array.shape[2] != 1:
        array = array[:, :, :1]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _prototype_confidence(record: dict[str, Any]) -> float | None:
    model = record.get("model_derived_evidence", {}) if isinstance(record.get("model_derived_evidence"), dict) else {}
    proto = model.get("prototype_ref", {}) if isinstance(model.get("prototype_ref"), dict) else {}
    return _safe_float(proto.get("confidence"))


def _resolve_artifact(artifact: str, base_dir: Path) -> Path:
    path = Path(os.path.expandvars(artifact)).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def _relative_tile(path: Path, output_dir: Path) -> str:
    try:
        return path.relative_to(output_dir).as_posix()
    except ValueError:
        return str(path)


def _slug(text: str) -> str:
    cleaned = [char if char.isalnum() or char in {"-", "_"} else "_" for char in text]
    return "".join(cleaned)[:80] or "contour"


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text and text.lower() not in {"nan", "none", "null"} else None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _panel_html(evidence_chain: Path, rows: list[dict[str, Any]]) -> str:
    cards = "\n".join(_panel_card(row) for row in rows) if rows else "<p>No evidence records were selected.</p>"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Contour-Native Visual Evidence Panel</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #172026; background: #f8fafb; }}
    h1 {{ margin-bottom: 4px; }}
    .subtitle {{ color: #50606c; margin-top: 0; max-width: 980px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; }}
    .card {{ background: white; border: 1px solid #d7dee4; border-radius: 8px; padding: 14px; }}
    .images {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 10px 0 12px; }}
    .images figure {{ margin: 0; }}
    .images img {{ width: 100%; aspect-ratio: 1 / 1; object-fit: cover; border: 1px solid #d7dee4; }}
    figcaption {{ font-size: 11px; color: #50606c; margin-top: 4px; }}
    .facts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
    .facts h3 {{ margin: 4px 0; font-size: 13px; }}
    .facts p {{ margin: 4px 0; font-size: 12px; line-height: 1.4; }}
    code {{ font-size: 12px; }}
  </style>
</head>
<body>
  <h1>Contour-Native Visual Evidence Panel</h1>
  <p class="subtitle">Source: {escape(str(evidence_chain))}. This artifact-first panel separates measured evidence from model-derived evidence and follows JSON pointers into Zarr/Parquet stores.</p>
  <div class="grid">
    {cards}
  </div>
</body>
</html>
"""


def _panel_card(row: dict[str, Any]) -> str:
    genes = _genes_html(row.get("top_genes"))
    return f"""<section class="card">
  <h2>{escape(str(row.get("contour_id") or row.get("evidence_id") or "contour"))}</h2>
  <p><code>{escape(str(row.get("slide_id") or "slide unknown"))}</code> | row {escape(str(row.get("row_index")))} | image_source={escape(str(row.get("image_source")))}</p>
  <div class="images">
    <figure><img src="{escape(str(row.get("object_image")))}" alt="Object RGB"><figcaption>Object RGB</figcaption></figure>
    <figure><img src="{escape(str(row.get("context_image")))}" alt="Context RGB"><figcaption>Context RGB</figcaption></figure>
    <figure><img src="{escape(str(row.get("mask_image")))}" alt="Contour mask"><figcaption>Mask</figcaption></figure>
    <figure><img src="{escape(str(row.get("overlay_image")))}" alt="Mask overlay"><figcaption>Overlay</figcaption></figure>
  </div>
  <div class="facts">
    <div>
      <h3>Measured evidence</h3>
      <p>Spatial: x={escape(str(row.get("x")))} y={escape(str(row.get("y")))}</p>
      <p>Top genes: {genes}</p>
      <p>Geometry: <code>{escape(str(row.get("geometry_preview")))}</code></p>
    </div>
    <div>
      <h3>Model-derived evidence</h3>
      <p>Prototype: {escape(str(row.get("prototype_id")))} | confidence={escape(str(row.get("prototype_confidence")))}</p>
      <p>Entropy: {escape(str(row.get("assignment_entropy")))}</p>
      <p>Checkpoint hash: <code>{escape(str(row.get("checkpoint_hash")))}</code></p>
    </div>
  </div>
</section>"""


def _genes_html(value: Any) -> str:
    try:
        rows = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        rows = []
    if not isinstance(rows, list) or not rows:
        return "N/A"
    parts = []
    for row in rows[:8]:
        if not isinstance(row, dict):
            continue
        gene = escape(str(row.get("gene")))
        number = _safe_float(row.get("value"))
        parts.append(f"{gene}={number:.3g}" if number is not None else gene)
    return ", ".join(parts) if parts else "N/A"
