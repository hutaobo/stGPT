from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from typer.testing import CliRunner

from stgpt.cli import app
from stgpt.contour_store import ContourStoreSpec, create_contour_image_store
from stgpt.evidence import build_failure_gallery


def _write_failure_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    export_dir = run_dir / "spatho_export"
    eval_dir = run_dir / "evaluation"
    export_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    store = tmp_path / "contour_image_store.zarr"
    root = create_contour_image_store(
        store,
        spec=ContourStoreSpec(n_contours=3, image_size=16, geometry_size=8, chunk_size=3),
    )
    for idx, color in enumerate(([220, 40, 80], [80, 180, 90], [50, 90, 220])):
        root["object_rgb"][idx] = np.full((16, 16, 3), color, dtype=np.uint8)
        root["context_rgb"][idx] = np.full((16, 16, 3), np.asarray(color, dtype=np.uint8) // 2, dtype=np.uint8)
        root["soft_mask"][idx] = np.ones((16, 16, 1), dtype=np.uint8) * 255
        root["geometry"][idx] = np.asarray([10 + idx, 50 + idx, 0.5, 1, 2, 3, 4, 5], dtype=np.float32)
        root["contour_ids"][idx] = f"contour_{idx}".encode()
    pd.DataFrame(
        {
            "region_id": ["contour_0", "contour_1", "contour_2"],
            "n_cells": [5, 3, 1],
            "gene_A": [10.0, 1.0, 3.0],
            "gene_B": [1.0, 9.0, 2.0],
        }
    ).to_parquet(export_dir / "region_molecular_summary.parquet", index=False)
    pd.DataFrame(
        {
            "region_id": ["contour_0", "contour_1", "contour_2"],
            "contour_id": ["contour_0", "contour_1", "contour_2"],
            "slide_id": ["slide_a", "slide_a", "slide_a"],
            "row_index": [0, 1, 2],
            "structure_label": ["tumor", "stroma", "rare"],
            "qc_flag": ["ok", "ok", "ok"],
            "area": [10.0, 100.0, 80.0],
            "perimeter": [200.0, 40.0, 30.0],
            "eccentricity": [0.95, 0.2, 0.4],
        }
    ).to_parquet(export_dir / "region_embeddings.parquet", index=False)
    pd.DataFrame(
        {
            "embedding_row_index": [0, 1, 2],
            "prototype_id": [1, 1, 2],
            "prototype_confidence": [0.95, 0.05, 0.4],
            "assignment_entropy": [0.99, 0.2, 0.6],
        }
    ).to_parquet(export_dir / "prototype_assignments.parquet", index=False)
    pd.DataFrame(
        [{"split": "overall", "category": "patch", "metric": "missing_image_count", "value": 0.0, "detail": "mock"}]
    ).to_csv(eval_dir / "failure_analysis.csv", index=False)
    with (export_dir / "contour_evidence_chains.jsonl").open("w", encoding="utf-8") as handle:
        for idx in range(3):
            record = {
                "evidence_id": f"ev_{idx}",
                "unit": {
                    "contour_id": f"contour_{idx}",
                    "region_id": f"contour_{idx}",
                    "slide_id": "slide_a",
                    "row_index": idx,
                    "embedding_row_index": idx,
                },
                "measured_evidence": {
                    "molecular_ref": {"artifact": "region_molecular_summary.parquet", "row_index": idx},
                    "image_ref": {
                        "artifact": str(store),
                        "row_index": idx,
                        "arrays": {"object_rgb": "object_rgb", "context_rgb": "context_rgb", "mask": "soft_mask"},
                    },
                    "spatial": {"x": float(idx), "y": float(idx + 1)},
                },
                "model_derived_evidence": {
                    "prototype_ref": {
                        "artifact": "prototype_assignments.parquet",
                        "row_index": idx,
                        "prototype_id": [1, 1, 2][idx],
                        "confidence": [0.95, 0.05, 0.4][idx],
                        "assignment_entropy": [0.99, 0.2, 0.6][idx],
                    },
                    "embedding_ref": {"artifact": "region_embeddings.parquet", "row_index": idx},
                },
                "qc_verdict": {"image_source": "contour_store", "qc_flag": "ok"},
                "provenance": {"checkpoint_hash": "ckpt", "config_hash": "cfg", "contour_manifest_hash": "manifest"},
            }
            handle.write(json.dumps(record) + "\n")
    return run_dir


def test_build_failure_gallery_scores_and_exports_targets(tmp_path: Path) -> None:
    run_dir = _write_failure_run(tmp_path)

    result = build_failure_gallery(run_dir, tmp_path / "gallery", max_items=2, top_genes=2, rare_prototype_fraction=0.34)

    assert result["n_records"] == 3
    html = Path(result["artifacts"]["failure_gallery_html"])
    gallery = pd.read_csv(result["artifacts"]["failure_gallery_csv"])
    targets = pd.read_csv(result["artifacts"]["ablation_targets_csv"])
    assert html.exists()
    assert "Contour-Native Failure Gallery" in html.read_text(encoding="utf-8")
    assert "structure_blind" in ",".join(gallery["failure_reasons"].astype(str))
    assert "hallucination_risk" in ",".join(gallery["failure_reasons"].astype(str))
    assert len(targets) >= 1
    for column in ("object_image", "context_image", "mask_image", "overlay_image"):
        tile = html.parent / str(gallery[column].dropna().iloc[0])
        assert tile.exists()
        with Image.open(tile) as image:
            assert image.size == (16, 16)


def test_failure_gallery_cli_writes_outputs(tmp_path: Path) -> None:
    run_dir = _write_failure_run(tmp_path)

    result = CliRunner().invoke(
        app,
        ["failure-gallery", "--run-dir", str(run_dir), "--output", str(tmp_path / "gallery"), "--max-items", "2"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert Path(payload["artifacts"]["failure_summary"]).exists()
