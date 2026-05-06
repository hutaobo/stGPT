from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from stgpt.cli import app
from stgpt.evidence import audit_evidence_pointers, load_evidence_suite, summarize_evidence_suite


def _write_config(path: Path) -> Path:
    path.write_text("case_name: mock\n", encoding="utf-8")
    return path


def _write_metrics(run_dir: Path, *, with_prototypes: bool = True, with_alignment: bool = False) -> None:
    train_dir = run_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        {
            "loss": 0.5,
            "step": 1,
            "lr": 0.001,
            "gene_loss": 0.2,
            "val_gene_loss": 0.3,
            "val_image_to_gene_top5": 0.2 if with_alignment else None,
            "val_gene_to_image_top5": 0.4 if with_alignment else None,
            "val_alignment_score": 0.3 if with_alignment else None,
            "image_gene_loss_weight": 0.1 if with_alignment else None,
            "prototype_loss_weight": 0.1 if with_alignment else None,
            "prototype_usage_count": 2.0 if with_prototypes else None,
            "sinkhorn_nonfinite_count": 0.0 if with_prototypes else None,
        },
        {
            "loss": 0.2,
            "step": 2,
            "lr": 0.0005,
            "gene_loss": 0.1,
            "val_gene_loss": 0.12,
            "val_image_to_gene_top5": 0.5 if with_alignment else None,
            "val_gene_to_image_top5": 0.7 if with_alignment else None,
            "val_alignment_score": 0.6 if with_alignment else None,
            "image_gene_loss_weight": 0.1 if with_alignment else None,
            "prototype_loss_weight": 0.1 if with_alignment else None,
            "prototype_loss": 1.5 if with_prototypes else None,
            "prototype_entropy_normalized": 0.9 if with_prototypes else None,
            "prototype_usage_count": 3.0 if with_prototypes else None,
            "prototype_dead_codes": 1.0 if with_prototypes else None,
            "sinkhorn_nonfinite_count": 0.0 if with_prototypes else None,
            "sinkhorn_row_residual": 0.0 if with_prototypes else None,
            "sinkhorn_col_residual": 0.001 if with_prototypes else None,
        },
    ]
    (train_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


def _write_spatho_export(
    run_dir: Path,
    *,
    image_source: str,
    n_records: int = 3,
    n_prototypes: int = 4,
) -> None:
    export = run_dir / "spatho_export"
    export.mkdir(parents=True, exist_ok=True)
    for name in ("region_embeddings.parquet", "region_molecular_summary.parquet", "region_image_manifest.json"):
        path = export / name
        if path.suffix == ".json":
            path.write_text("{}", encoding="utf-8")
        else:
            path.write_bytes(b"placeholder")
    pd.DataFrame(
        {
            "prototype_id": [idx % n_prototypes if n_prototypes > 0 else -1 for idx in range(n_records)],
            "prototype_confidence": [0.8, 0.7, 0.6][:n_records],
            "assignment_entropy": [0.2, 0.3, 0.4][:n_records],
        }
    ).to_parquet(export / "prototype_assignments.parquet", index=False)

    image_file = export / "patch.png"
    image_file.write_bytes(b"png")
    contour_store = run_dir / "contour_image_store.zarr"
    contour_store.mkdir(exist_ok=True)
    contour_manifest = run_dir / "contour_image_manifest.parquet"
    contour_manifest.write_bytes(b"manifest")

    with (export / "contour_evidence_chains.jsonl").open("w", encoding="utf-8") as handle:
        for idx in range(n_records):
            if image_source == "contour_store":
                image_ref = {
                    "artifact": str(contour_store),
                    "row_index": idx,
                    "arrays": {"object_rgb": "object_rgb", "context_rgb": "context_rgb", "mask": "soft_mask"},
                }
                geometry_ref = {"artifact": str(contour_manifest), "row_index": idx, "columns": "geometry"}
                contour_hash = "def"
            elif image_source == "image_path":
                image_ref = {"artifact": str(image_file), "row_index": None, "source": "image_path"}
                geometry_ref = {"artifact": "region_image_manifest.json", "row_index": idx, "columns": "geometry_unavailable"}
                contour_hash = None
            else:
                image_ref = {"artifact": "region_image_manifest.json", "row_index": idx, "source": "manifest_fallback"}
                geometry_ref = {"artifact": "region_image_manifest.json", "row_index": idx, "columns": "geometry_unavailable"}
                contour_hash = None
            record = {
                "measured_evidence": {
                    "image_ref": image_ref,
                    "geometry_ref": geometry_ref,
                    "molecular_ref": {"artifact": "region_molecular_summary.parquet", "row_index": idx},
                },
                "model_derived_evidence": {
                    "embedding_ref": {"artifact": "region_embeddings.parquet", "row_index": idx},
                    "prototype_ref": {"artifact": "prototype_assignments.parquet", "row_index": idx, "prototype_id": idx},
                },
                "provenance": {
                    "config_hash": "abc",
                    "checkpoint_hash": "123",
                    "contour_manifest_hash": contour_hash,
                },
                "qc_verdict": {"image_source": image_source, "qc_flag": "ok"},
            }
            handle.write(json.dumps(record) + "\n")


def _write_suite(path: Path, runs: list[dict[str, object]]) -> Path:
    payload = {"suite_name": "mock_suite", "runs": runs}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_summarize_evidence_suite_parses_mock_runs(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "config.yaml")
    contour_run = tmp_path / "contour_run"
    image_run = tmp_path / "image_run"
    _write_metrics(contour_run)
    _write_spatho_export(contour_run, image_source="contour_store", n_prototypes=4)
    _write_metrics(image_run, with_prototypes=False)
    _write_spatho_export(image_run, image_source="image_path", n_prototypes=0)
    suite = _write_suite(
        tmp_path / "suite.json",
        [
            {
                "run_id": "contour",
                "tissue": "breast",
                "condition": "Full M6 Zarr",
                "config_path": str(config),
                "run_dir": str(contour_run),
                "expected_image_source": "contour_store",
                "expected_prototypes": 4,
            },
            {
                "run_id": "image",
                "tissue": "breast",
                "condition": "PNG fallback",
                "config_path": str(config),
                "run_dir": str(image_run),
                "expected_image_source": "image_path",
                "expected_prototypes": 0,
            },
        ],
    )

    result = summarize_evidence_suite(suite, tmp_path / "out", pointer_sample_size=2)

    assert result["status"] == "pass"
    summary = pd.read_csv(result["artifacts"]["evidence_summary_csv"])
    assert len(summary) == 2
    assert set(summary["image_source"]) == {"contour_store", "image_path"}
    assert float(summary.loc[summary["run_id"] == "contour", "val_gene_loss_final"].iloc[0]) == pytest.approx(0.12)
    assert Path(result["artifacts"]["pareto_frontier_csv"]).exists()


def test_missing_artifacts_are_reported_without_crashing(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "config.yaml")
    suite = _write_suite(
        tmp_path / "suite.json",
        [
            {
                "run_id": "missing",
                "tissue": "breast",
                "condition": "missing",
                "config_path": str(config),
                "run_dir": str(tmp_path / "missing_run"),
                "expected_image_source": "contour_store",
                "expected_prototypes": 4,
            }
        ],
    )

    result = summarize_evidence_suite(suite, tmp_path / "out")

    assert result["status"] == "missing"
    status = json.loads(Path(result["artifacts"]["run_status"]).read_text(encoding="utf-8"))
    assert status["runs"][0]["status"] == "missing"
    assert "train/metrics.json" in status["runs"][0]["missing_artifacts"]


def test_audit_evidence_pointers_accepts_all_image_sources(tmp_path: Path) -> None:
    for image_source in ("contour_store", "image_path", "zero_fallback"):
        run_dir = tmp_path / image_source
        _write_spatho_export(run_dir, image_source=image_source)
        result = audit_evidence_pointers(
            run_dir / "spatho_export" / "contour_evidence_chains.jsonl",
            export_dir=run_dir / "spatho_export",
            expected_image_source=image_source,  # type: ignore[arg-type]
            sample_size=3,
        )
        assert result["pointer_errors"] == 0
        assert result["image_source_counts"] == {image_source: 3}


def test_evidence_summary_cli_writes_outputs(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "config.yaml")
    run_dir = tmp_path / "run"
    _write_metrics(run_dir)
    _write_spatho_export(run_dir, image_source="contour_store")
    suite = _write_suite(
        tmp_path / "suite.json",
        [
            {
                "run_id": "contour",
                "tissue": "breast",
                "condition": "Full M6 Zarr",
                "config_path": str(config),
                "run_dir": str(run_dir),
                "expected_image_source": "contour_store",
                "expected_prototypes": 4,
            }
        ],
    )

    result = CliRunner().invoke(app, ["evidence-summary", "--suite", str(suite), "--output", str(tmp_path / "out")])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert Path(payload["artifacts"]["paper_table"]).exists()


def test_learning_dynamics_and_missing_alignment_warning(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "config.yaml")
    aligned_run = tmp_path / "aligned"
    missing_run = tmp_path / "missing_alignment"
    _write_metrics(aligned_run, with_alignment=True)
    _write_spatho_export(aligned_run, image_source="contour_store")
    _write_metrics(missing_run, with_alignment=False)
    _write_spatho_export(missing_run, image_source="contour_store")
    suite = _write_suite(
        tmp_path / "suite.json",
        [
            {
                "run_id": "aligned",
                "tissue": "breast",
                "condition": "Full M6 Pareto lambda=0.1",
                "config_path": str(config),
                "run_dir": str(aligned_run),
                "expected_image_source": "contour_store",
                "expected_prototypes": 4,
                "requires_alignment_telemetry": True,
                "checkpoint_role": "best_alignment",
                "lambda_align": 0.1,
                "suite_stage": "pareto_grid",
            },
            {
                "run_id": "missing_alignment",
                "tissue": "breast",
                "condition": "Full M6 Pareto lambda=0.5",
                "config_path": str(config),
                "run_dir": str(missing_run),
                "expected_image_source": "contour_store",
                "expected_prototypes": 4,
                "requires_alignment_telemetry": True,
                "checkpoint_role": "best_alignment",
                "lambda_align": 0.5,
                "suite_stage": "pareto_grid",
            },
        ],
    )

    result = summarize_evidence_suite(suite, tmp_path / "out")

    assert result["status"] == "warning"
    dynamics = pd.read_csv(result["artifacts"]["learning_dynamics_csv"])
    assert set(dynamics["run_id"]) == {"aligned", "missing_alignment"}
    assert float(dynamics.loc[dynamics["run_id"] == "aligned", "alignment_score"].max()) == pytest.approx(0.6)
    status = json.loads(Path(result["artifacts"]["run_status"]).read_text(encoding="utf-8"))
    missing = next(row for row in status["runs"] if row["run_id"] == "missing_alignment")
    assert "missing_alignment_telemetry" in missing["warnings"]


def test_atera_wta_evidence_artifacts_regression(tmp_path: Path) -> None:
    suite = Path("configs/evidence/atera_wta_v1.yaml")
    metrics = Path("outputs/pilot_runs/atera_wta_v1/breast_full_m6_contour_store/train/metrics.json")
    if not suite.exists() or not metrics.exists():
        pytest.skip("Atera WTA proof artifacts are not available in this checkout.")

    result = summarize_evidence_suite(suite, tmp_path / "atera_evidence", pointer_sample_size=50)

    assert result["n_runs"] >= 6
    summary = pd.read_csv(result["artifacts"]["evidence_summary_csv"])
    breast_zarr = summary[summary["run_id"] == "breast_full_m6_contour_store"].iloc[0]
    assert breast_zarr["image_source"] == "contour_store"
    assert float(breast_zarr["val_gene_loss_final"]) == pytest.approx(0.0255, rel=1e-2)


def test_load_evidence_suite_validates_schema(tmp_path: Path) -> None:
    suite = _write_suite(
        tmp_path / "suite.json",
        [
            {
                "run_id": "contour",
                "tissue": "breast",
                "condition": "Full M6 Zarr",
                "config_path": "config.yaml",
                "run_dir": "run",
                "expected_image_source": "contour_store",
                "expected_prototypes": 4,
            }
        ],
    )

    spec = load_evidence_suite(suite)

    assert spec.suite_name == "mock_suite"
    assert spec.runs[0].run_id == "contour"
