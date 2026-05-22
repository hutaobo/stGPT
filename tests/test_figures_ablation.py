from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from stgpt.figures import plot_ablation_comparison  # noqa: E402


def _summary_frame() -> pd.DataFrame:
    rows = []
    conditions = [
        "Full M6 Zarr contour store",
        "Contour-unit Gene+Spatial 500-step",
        "Full M6 contour-store random init",
    ]
    base = {
        "eval_gene_correlation": [0.83, 0.84],
        "eval_label_retrieval_top5": [0.74, 0.68],
        "eval_silhouette_mean": [-0.11, -0.18],
        "eval_image_to_gene_top5": [0.08, 0.04],
    }
    for tissue_index, tissue in enumerate(["breast", "cervical"]):
        for condition in conditions:
            rows.append(
                {
                    "tissue": tissue,
                    "condition": condition,
                    "eval_gene_correlation": base["eval_gene_correlation"][tissue_index],
                    "eval_label_retrieval_top5": base["eval_label_retrieval_top5"][tissue_index],
                    "eval_silhouette_mean": base["eval_silhouette_mean"][tissue_index],
                    "eval_image_to_gene_top5": base["eval_image_to_gene_top5"][tissue_index],
                }
            )
    # A baseline row with no eval metrics — should be dropped, not plotted.
    rows.append(
        {
            "tissue": "breast",
            "condition": "Gene-only baseline",
            "eval_gene_correlation": None,
            "eval_label_retrieval_top5": None,
            "eval_silhouette_mean": None,
            "eval_image_to_gene_top5": None,
        }
    )
    return pd.DataFrame(rows)


def test_f2_writes_figure_and_provenance(tmp_path: Path) -> None:
    frame = _summary_frame()

    result = plot_ablation_comparison(frame, tmp_path, formats=("png",))

    assert result["status"] == "pass"
    assert Path(result["artifacts"]["png"]).exists()
    sidecar = Path(result["artifacts"]["provenance"])
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["figure"] == "F2_ablation_comparison"
    assert payload["palette"] == "okabe_ito"
    assert payload["groups"] == ["breast", "cervical"]
    assert "stgpt_version" in payload


def test_f2_drops_rows_without_eval_metrics(tmp_path: Path) -> None:
    frame = _summary_frame()

    result = plot_ablation_comparison(frame, tmp_path, formats=("png",))

    # The Gene-only baseline row has no eval metrics and must be dropped.
    assert any(warning.startswith("dropped_") for warning in result["warnings"])
    assert "Gene-only" not in result["conditions"]
    assert "Full M6" in result["conditions"]


def test_f2_orders_conditions_floor_to_transfer(tmp_path: Path) -> None:
    frame = _summary_frame()
    frame = pd.concat(
        [
            frame,
            pd.DataFrame(
                [
                    {
                        "tissue": "breast",
                        "condition": "Zero-shot Cervical→Breast Full M6",
                        "eval_gene_correlation": 0.82,
                        "eval_label_retrieval_top5": 0.74,
                        "eval_silhouette_mean": -0.13,
                        "eval_image_to_gene_top5": 0.03,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    result = plot_ablation_comparison(frame, tmp_path, formats=("png",))

    # Canonical order: Random init -> Gene+spatial -> Full M6 -> Zero-shot.
    assert result["conditions"] == ["Random init", "Gene+spatial", "Full M6", "Zero-shot"]


def test_f2_flags_missing_metrics(tmp_path: Path) -> None:
    frame = _summary_frame()

    result = plot_ablation_comparison(
        frame,
        tmp_path,
        metrics=[("eval_gene_correlation", "Gene correlation"), ("eval_made_up_metric", "Nope")],
        formats=("png",),
    )

    assert any("missing_metrics" in warning for warning in result["warnings"])
    assert result["metrics"] == ["eval_gene_correlation"]


def test_f2_rejects_when_no_metric_columns(tmp_path: Path) -> None:
    frame = pd.DataFrame({"tissue": ["breast"], "condition": ["Full M6 Zarr contour store"]})

    with pytest.raises(ValueError, match="none of the requested metric columns"):
        plot_ablation_comparison(frame, tmp_path, formats=("png",))


def test_f2_pdf_uses_editable_truetype_fonts(tmp_path: Path) -> None:
    frame = _summary_frame()

    result = plot_ablation_comparison(frame, tmp_path, formats=("pdf",))

    pdf_bytes = Path(result["artifacts"]["pdf"]).read_bytes()
    # The matplotlib default Type 3 path font is not editable as text in
    # Illustrator. Type 42/CIDFontType2 remains editable and is portable across
    # machines even when Arial is unavailable and DejaVu Sans is used instead.
    assert b"/Subtype /Type3" not in pdf_bytes
    assert b"CIDFontType2" in pdf_bytes or b"/FontFile2" in pdf_bytes
