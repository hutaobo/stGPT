from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from stgpt.figures import plot_cross_platform_manifold  # noqa: E402


def _manifold_frame(*, checkpoint_hashes: list[str], n_per_run: int = 6) -> pd.DataFrame:
    rows = []
    tissues = ["breast", "cervical"]
    structures = ["tumor", "stroma", "immune"]
    for run_index, checkpoint in enumerate(checkpoint_hashes):
        for idx in range(n_per_run):
            rows.append(
                {
                    "run_id": f"run_{run_index}",
                    "tissue": tissues[idx % len(tissues)],
                    "structure_label": structures[idx % len(structures)],
                    "checkpoint_hash": checkpoint,
                    "manifold_x": float(run_index + idx * 0.1),
                    "manifold_y": float(idx - run_index * 0.2),
                }
            )
    return pd.DataFrame(rows)


def test_f1_writes_figure_and_provenance(tmp_path: Path) -> None:
    frame = _manifold_frame(checkpoint_hashes=["hash_a"])

    result = plot_cross_platform_manifold(frame, tmp_path, formats=("png",))

    assert result["status"] == "pass"
    assert result["batch_key"] == "tissue"  # first multi-value candidate
    assert Path(result["artifacts"]["png"]).exists()
    sidecar = Path(result["artifacts"]["provenance"])
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["figure"] == "F1_cross_platform_manifold"
    assert payload["checkpoint_hashes"] == ["hash_a"]
    assert payload["palette"] == "okabe_ito"
    assert "stgpt_version" in payload


def test_f1_flags_multiple_checkpoint_hashes(tmp_path: Path) -> None:
    frame = _manifold_frame(checkpoint_hashes=["hash_a", "hash_b"])

    result = plot_cross_platform_manifold(frame, tmp_path, formats=("png",))

    assert result["status"] == "warning"
    assert any("multiple_checkpoint_hashes" in warning for warning in result["warnings"])


def test_f1_run_id_filter_silences_guardrail(tmp_path: Path) -> None:
    frame = _manifold_frame(checkpoint_hashes=["hash_a", "hash_b"])

    result = plot_cross_platform_manifold(frame, tmp_path, run_id="run_0", formats=("png",))

    assert result["status"] == "pass"
    assert result["checkpoint_hashes"] == ["hash_a"]


def test_f1_panel_c_reads_metric_csvs(tmp_path: Path) -> None:
    frame = _manifold_frame(checkpoint_hashes=["hash_a"])
    mixing = tmp_path / "batch_mixing_metrics.csv"
    qc = tmp_path / "embedding_qc.csv"
    pd.DataFrame(
        {"split": ["test"], "batch_column": ["tissue"], "k": [15], "n_regions": [12], "batch_mixing_entropy": [0.42]}
    ).to_csv(mixing, index=False)
    pd.DataFrame(
        {"split": ["test"], "label_column": ["structure_label"], "n_regions": [12], "n_labels": [3], "silhouette": [0.18]}
    ).to_csv(qc, index=False)

    result = plot_cross_platform_manifold(
        frame, tmp_path, batch_mixing_csv=mixing, embedding_qc_csv=qc, formats=("png",)
    )

    assert result["status"] == "pass"
    assert Path(result["artifacts"]["png"]).exists()


def test_f1_rejects_frame_without_projection(tmp_path: Path) -> None:
    frame = pd.DataFrame({"emb_0": [0.1, 0.2], "structure_label": ["tumor", "stroma"]})

    with pytest.raises(ValueError, match="manifold_x"):
        plot_cross_platform_manifold(frame, tmp_path, formats=("png",))
