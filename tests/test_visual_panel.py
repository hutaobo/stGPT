from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from typer.testing import CliRunner

from stgpt.cli import app
from stgpt.contour_store import ContourStoreSpec, create_contour_image_store
from stgpt.visual import build_contour_panel


def _write_panel_artifacts(tmp_path: Path) -> Path:
    export_dir = tmp_path / "spatho_export"
    export_dir.mkdir(parents=True, exist_ok=True)
    store = tmp_path / "contour_image_store.zarr"
    root = create_contour_image_store(
        store,
        spec=ContourStoreSpec(n_contours=2, image_size=16, geometry_size=8, chunk_size=2),
    )
    root["object_rgb"][0] = np.full((16, 16, 3), [120, 80, 200], dtype=np.uint8)
    root["context_rgb"][0] = np.full((16, 16, 3), [30, 140, 90], dtype=np.uint8)
    root["soft_mask"][0] = np.ones((16, 16, 1), dtype=np.uint8) * 255
    root["geometry"][0] = np.arange(8, dtype=np.float32)
    root["contour_ids"][0] = b"contour_a"
    pd.DataFrame(
        {
            "region_id": ["contour_a", "contour_b"],
            "n_cells": [3, 2],
            "gene_A": [4.0, 1.0],
            "gene_B": [2.0, 3.0],
        }
    ).to_parquet(export_dir / "region_molecular_summary.parquet", index=False)
    record = {
        "schema_version": "stgpt.evidence_pointer.v0.1",
        "evidence_id": "ev_mock",
        "unit": {
            "type": "contour_region",
            "region_id": "contour_a",
            "contour_id": "contour_a",
            "slide_id": "slide_a",
            "row_index": 0,
            "embedding_row_index": 0,
        },
        "measured_evidence": {
            "molecular_ref": {"artifact": "region_molecular_summary.parquet", "row_index": 0},
            "image_ref": {
                "artifact": str(store),
                "row_index": 0,
                "arrays": {"object_rgb": "object_rgb", "context_rgb": "context_rgb", "mask": "soft_mask"},
            },
            "geometry_ref": {"artifact": str(tmp_path / "contour_image_manifest.parquet"), "row_index": 0},
            "spatial": {"x": 12.5, "y": 22.5},
        },
        "model_derived_evidence": {
            "embedding_ref": {"artifact": "region_embeddings.parquet", "row_index": 0},
            "prototype_ref": {
                "artifact": "prototype_assignments.parquet",
                "row_index": 0,
                "prototype_id": 7,
                "confidence": 0.42,
                "assignment_entropy": 0.8,
            },
        },
        "qc_verdict": {"qc_flag": "ok", "image_source": "contour_store"},
        "provenance": {"checkpoint_hash": "ckpt", "config_hash": "cfg", "contour_manifest_hash": "manifest"},
    }
    evidence_chain = export_dir / "contour_evidence_chains.jsonl"
    evidence_chain.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return evidence_chain


def test_build_contour_panel_follows_zarr_and_parquet_pointers(tmp_path: Path) -> None:
    evidence_chain = _write_panel_artifacts(tmp_path)

    result = build_contour_panel(evidence_chain, tmp_path / "panel", sample_size=1, top_genes=2)

    html = Path(result["artifacts"]["contour_panel_html"])
    manifest = pd.read_csv(result["artifacts"]["contour_panel_manifest_csv"])
    assert html.exists()
    assert len(manifest) == 1
    assert manifest["contour_id"].iloc[0] == "contour_a"
    assert "gene_A" not in manifest["top_genes"].iloc[0]
    assert '"gene": "A"' in manifest["top_genes"].iloc[0]
    assert "Measured evidence" in html.read_text(encoding="utf-8")
    for column in ("object_image", "context_image", "mask_image", "overlay_image"):
        tile = html.parent / manifest[column].iloc[0]
        assert tile.exists()
        with Image.open(tile) as image:
            assert image.size == (16, 16)


def test_contour_panel_cli_writes_outputs(tmp_path: Path) -> None:
    evidence_chain = _write_panel_artifacts(tmp_path)

    result = CliRunner().invoke(
        app,
        ["contour-panel", "--evidence-chain", str(evidence_chain), "--output", str(tmp_path / "panel")],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert Path(payload["artifacts"]["contour_panel_html"]).exists()
