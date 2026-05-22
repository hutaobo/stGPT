from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from stgpt.figures import plot_learning_dynamics  # noqa: E402


def _learning_frame() -> pd.DataFrame:
    rows = []
    for run_id in (
        "gene_spatial_contour_unit_20k",
        "full_m6_contour_store_lambda_0_01_20k",
        "structure_context_m6_20k",
        "smoke_5case_full_m6_lambda_0_01_500",
    ):
        for step in (1, 100, 200):
            rows.append(
                {
                    "run_id": run_id,
                    "step": step,
                    "val_gene_loss": 1.0 / step,
                    "alignment_score": step / 200.0,
                    "image_to_gene_top5": step / 400.0,
                    "gene_to_image_top5": step / 300.0,
                }
            )
    return pd.DataFrame(rows)


def test_f3_writes_figure_and_provenance(tmp_path: Path) -> None:
    result = plot_learning_dynamics(_learning_frame(), tmp_path, formats=("png",))

    assert result["status"] == "pass"
    assert Path(result["artifacts"]["png"]).exists()
    sidecar = Path(result["artifacts"]["provenance"])
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["figure"] == "F3_learning_dynamics"
    assert payload["run_ids"] == [
        "gene_spatial_contour_unit_20k",
        "full_m6_contour_store_lambda_0_01_20k",
        "structure_context_m6_20k",
    ]
    assert payload["step_max"] == 200


def test_f3_rejects_missing_required_columns(tmp_path: Path) -> None:
    frame = pd.DataFrame({"run_id": ["structure_context_m6_20k"], "val_gene_loss": [0.1]})

    with pytest.raises(ValueError, match="required columns"):
        plot_learning_dynamics(frame, tmp_path, formats=("png",))


def test_f3_warns_about_missing_metric(tmp_path: Path) -> None:
    frame = _learning_frame().drop(columns=["alignment_score"])

    result = plot_learning_dynamics(frame, tmp_path, formats=("png",))

    assert result["status"] == "warning"
    assert any("missing_metrics" in warning for warning in result["warnings"])
