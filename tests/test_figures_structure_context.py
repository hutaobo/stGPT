from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from stgpt.figures import plot_structure_context_evidence  # noqa: E402


def _summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": "gene_spatial_contour_unit_20k",
                "expected_prototypes": 0,
                "eval_label_retrieval_top1": 0.12,
                "eval_label_retrieval_top5": 0.39,
                "prototype_usage_export_global": 0,
                "prototype_mean_confidence": None,
                "prototype_assignment_rows": 100,
            },
            {
                "run_id": "full_m6_contour_store_lambda_0_01_20k",
                "expected_prototypes": 128,
                "eval_label_retrieval_top1": 0.06,
                "eval_label_retrieval_top5": 0.26,
                "prototype_usage_export_global": 127,
                "prototype_mean_confidence": 0.02,
                "prototype_assignment_rows": 100,
            },
            {
                "run_id": "structure_context_m6_20k",
                "expected_prototypes": 128,
                "eval_label_retrieval_top1": 1.0,
                "eval_label_retrieval_top5": 1.0,
                "prototype_usage_export_global": 128,
                "prototype_mean_confidence": 0.05,
                "prototype_assignment_rows": 100,
            },
        ]
    )


def _run_dir(tmp_path: Path) -> Path:
    export = tmp_path / "run" / "spatho_export"
    export.mkdir(parents=True)
    (export / "region_qc_report.json").write_text(
        json.dumps(
            {
                "n_regions_total": 100,
                "n_regions_with_image": 98,
                "n_cells_assigned": 500,
                "image_coverage": 0.98,
            }
        ),
        encoding="utf-8",
    )
    artifact = export / "contour_evidence_chains.jsonl"
    artifact.write_text("{}\n", encoding="utf-8")
    (export / "evidence_manifest.json").write_text(
        json.dumps(
            {
                "artifacts": {"contour_evidence_chains": str(artifact)},
                "provenance": {"checkpoint_hash": "abc"},
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame({"prototype_id": [1, 2, 2], "prototype_confidence": [0.1, 0.2, 0.3]}).to_parquet(
        export / "prototype_assignments.parquet"
    )
    return tmp_path / "run"


def test_f4_writes_figure_and_provenance(tmp_path: Path) -> None:
    pointer = pd.DataFrame(
        {
            "run_id": ["structure_context_m6_20k"],
            "records_total": [100],
            "records_sampled": [50],
            "pointer_errors": [0],
        }
    )

    result = plot_structure_context_evidence(_summary_frame(), _run_dir(tmp_path), tmp_path, pointer_audit=pointer, formats=("png",))

    assert result["status"] == "pass"
    assert Path(result["artifacts"]["png"]).exists()
    payload = json.loads(Path(result["artifacts"]["provenance"]).read_text(encoding="utf-8"))
    assert payload["figure"] == "F4_structure_context_evidence"
    assert payload["checkpoint_hashes"] == ["abc"]
    assert payload["prototype_stats"]["unique_prototypes"] == 2
    assert payload["pointer_errors"] == 0


def test_f4_rejects_missing_structure_run(tmp_path: Path) -> None:
    frame = _summary_frame()
    frame = frame[frame["run_id"] != "structure_context_m6_20k"]

    with pytest.raises(ValueError, match="structure_run_id"):
        plot_structure_context_evidence(frame, _run_dir(tmp_path), tmp_path, formats=("png",))
