from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from stgpt.inspect_cli import app as inspect_app
from stgpt.inspection import inspect_registry


def test_inspect_registry_reports_normalized_paths(tmp_path: Path) -> None:
    case_dir = tmp_path / "Xenium_Prime_Test_outs"
    case_dir.mkdir()
    slide_store = case_dir / "xenium_slide.zarr"
    slide_store.mkdir()
    patch = case_dir / "patch.png"
    patch.write_bytes(b"png")
    patch_manifest = case_dir / "contour_patches_manifest.json"
    patch_manifest.write_text(json.dumps([{"contour_id": "c1", "image_path": str(patch)}]), encoding="utf-8")
    cell_to_contour = case_dir / "cell_to_contour.parquet"
    pd.DataFrame({"cell_id": ["cell"], "contour_id": ["c1"]}).to_parquet(cell_to_contour, index=False)
    structures = case_dir / "structure_assignments.csv"
    structures.write_text("contour_id,structure_id\nc1,1\n", encoding="utf-8")
    qc = case_dir / "qc_report.json"
    qc.write_text("{}", encoding="utf-8")
    metadata = case_dir / "metadata_10x.json"
    metadata.write_text("{}", encoding="utf-8")
    contour_source = case_dir / "contour_source_manifest.json"
    contour_source.write_text("{}", encoding="utf-8")
    slide_manifest = case_dir / "slide_manifest.json"
    slide_manifest.write_text(
        json.dumps(
            {
                "output_dir": str(case_dir),
                "slide_store": str(slide_store),
                "artifacts": {
                    "cell_to_contour": str(cell_to_contour),
                    "structure_assignments": str(structures),
                    "contour_patches_manifest": str(patch_manifest),
                },
            }
        ),
        encoding="utf-8",
    )
    registry = tmp_path / "dataset_registry.csv"
    pd.DataFrame(
        [
            {
                "case_name": "test_case",
                "selected_for_build": True,
                "output_dir": str(case_dir),
                "slide_manifest": str(slide_manifest),
                "contour_source_manifest": str(contour_source),
                "qc_report": str(qc),
                "metadata_10x": str(metadata),
            }
        ]
    ).to_csv(registry, index=False)

    result = inspect_registry(registry, root=tmp_path, sample_images=1)

    assert result["summary"]["selected_records"] == 1
    assert result["summary"]["cases_with_errors"] == 0
    assert result["summary"]["cases_with_contour_patches"] == 1


def test_stgpt_inspect_cli_writes_report(tmp_path: Path) -> None:
    registry = tmp_path / "dataset_registry.csv"
    pd.DataFrame([{"case_name": "empty", "selected_for_build": False}]).to_csv(registry, index=False)
    output = tmp_path / "inspection.json"

    result = CliRunner().invoke(inspect_app, ["--registry", str(registry), "--output", str(output)])

    assert result.exit_code == 0, result.output
    assert output.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["summary"]["records"] == 1
