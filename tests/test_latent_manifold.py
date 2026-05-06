from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from stgpt.cli import app
from stgpt.evidence import build_latent_manifold


def _write_run(root: Path, run_id: str, *, tissue: str, checkpoint_hash: str, offset: float) -> Path:
    run_dir = root / run_id
    export = run_dir / "spatho_export"
    export.mkdir(parents=True)
    rows = []
    for idx in range(4):
        rows.append(
            {
                "region_id": f"{run_id}_region_{idx}",
                "contour_id": f"{run_id}_contour_{idx}",
                "slide_id": f"{tissue}_slide",
                "structure_label": "tumor" if idx % 2 == 0 else "stroma",
                "n_cells": idx + 1,
                "qc_flag": "ok",
                "x": float(idx),
                "y": float(idx + 1),
                "row_index": idx,
                "emb_0": float(offset + idx),
                "emb_1": float(offset + idx * 0.5),
                "emb_2": float(1.0 - idx * 0.1),
            }
        )
    pd.DataFrame(rows).to_parquet(export / "region_embeddings.parquet", index=False)
    pd.DataFrame(
        {
            "embedding_row_index": [0, 1, 2, 3],
            "prototype_id": [1, 1, 2, 2],
            "prototype_confidence": [0.9, 0.8, 0.7, 0.6],
            "assignment_entropy": [0.1, 0.2, 0.3, 0.4],
        }
    ).to_parquet(export / "prototype_assignments.parquet", index=False)
    (export / "evidence_manifest.json").write_text(
        json.dumps({"provenance": {"checkpoint_hash": checkpoint_hash}}),
        encoding="utf-8",
    )
    (export / "contour_evidence_chains.jsonl").write_text(
        "\n".join(json.dumps({"region_id": f"{run_id}_region_{idx}"}) for idx in range(4)),
        encoding="utf-8",
    )
    return run_dir


def _write_suite(path: Path, breast: Path, cervical: Path) -> Path:
    payload = {
        "suite_name": "mock_manifold",
        "runs": [
            {
                "run_id": "breast",
                "tissue": "breast",
                "condition": "Breast full",
                "config_path": str(path.parent / "config.yaml"),
                "run_dir": str(breast),
                "expected_image_source": "contour_store",
                "expected_prototypes": 2,
            },
            {
                "run_id": "cervical",
                "tissue": "cervical",
                "condition": "Cervical full",
                "config_path": str(path.parent / "config.yaml"),
                "run_dir": str(cervical),
                "expected_image_source": "contour_store",
                "expected_prototypes": 2,
            },
        ],
    }
    path.parent.joinpath("config.yaml").write_text("case_name: mock\n", encoding="utf-8")
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_build_latent_manifold_writes_projection_and_guardrail(tmp_path: Path) -> None:
    breast = _write_run(tmp_path, "breast_run", tissue="breast", checkpoint_hash="hash_a", offset=0.0)
    cervical = _write_run(tmp_path, "cervical_run", tissue="cervical", checkpoint_hash="hash_b", offset=2.0)
    suite = _write_suite(tmp_path / "suite.json", breast, cervical)

    result = build_latent_manifold(suite, tmp_path / "manifold", reducer="pca", max_points_per_run=3)

    assert result["status"] == "warning"
    frame = pd.read_csv(result["artifacts"]["latent_manifold_csv"])
    summary = json.loads(Path(result["artifacts"]["latent_manifold_summary"]).read_text(encoding="utf-8"))
    assert len(frame) == 6
    assert {"manifold_x", "manifold_y", "prototype_id"}.issubset(frame.columns)
    assert "multiple_checkpoint_hashes" in ",".join(summary["warnings"])
    assert summary["cross_tissue_top5_rate"] >= 0.0
    assert summary["html_point_sampling"] == "all"
    html = Path(result["artifacts"]["latent_manifold_html"])
    assert html.exists()
    text = html.read_text(encoding="utf-8")
    assert "contour:" in text
    assert "structure:" in text
    assert "evidence_link:" in text
    assert "data-contour-id" in text


def test_latent_manifold_html_density_samples_points(tmp_path: Path) -> None:
    breast = _write_run(tmp_path, "breast_run", tissue="breast", checkpoint_hash="hash_a", offset=0.0)
    cervical = _write_run(tmp_path, "cervical_run", tissue="cervical", checkpoint_hash="hash_a", offset=2.0)
    suite = _write_suite(tmp_path / "suite.json", breast, cervical)

    result = build_latent_manifold(
        suite,
        tmp_path / "manifold",
        reducer="pca",
        max_html_points=3,
    )

    summary = json.loads(Path(result["artifacts"]["latent_manifold_summary"]).read_text(encoding="utf-8"))
    assert summary["n_points"] == 8
    assert summary["html_points_rendered"] == 3
    assert summary["html_point_sampling"] == "density_grid_sqrt"
    html = Path(result["artifacts"]["latent_manifold_html"]).read_text(encoding="utf-8")
    assert html.count("data-contour-id") == 3


def test_latent_manifold_cli_writes_outputs(tmp_path: Path) -> None:
    breast = _write_run(tmp_path, "breast_run", tissue="breast", checkpoint_hash="hash_a", offset=0.0)
    cervical = _write_run(tmp_path, "cervical_run", tissue="cervical", checkpoint_hash="hash_a", offset=2.0)
    suite = _write_suite(tmp_path / "suite.json", breast, cervical)

    result = CliRunner().invoke(
        app,
        [
            "latent-manifold",
            "--suite",
            str(suite),
            "--output",
            str(tmp_path / "manifold"),
            "--reducer",
            "pca",
            "--max-html-points",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "pass"
    assert Path(payload["artifacts"]["latent_manifold_md"]).exists()
