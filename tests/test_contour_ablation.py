from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from stgpt.cli import app
from stgpt.evidence import run_contour_ablation
from stgpt.training import train


def _write_ablation_config(tmp_path: Path) -> Path:
    config = tmp_path / "ablation_smoke.yaml"
    config.write_text(
        f"""
case_name: ablation_smoke
data:
  mode: synthetic
  output_dir: {tmp_path.as_posix()}/case
  n_cells: 16
  n_genes: 20
  n_structures: 2
  image_size: 32
model:
  d_model: 32
  n_heads: 4
  n_layers: 1
  max_genes: 12
  n_expression_bins: 8
  image_size: 32
  n_prototypes: 4
training:
  batch_size: 4
  learning_rate: 0.001
  max_steps: 1
  output_dir: {tmp_path.as_posix()}/train
  device: cpu
  num_workers: 0
  prototype_loss_weight: 0.1
  prototype_queue_size: 8
""",
        encoding="utf-8",
    )
    return config


def _write_targets(path: Path) -> Path:
    payload = [
        {
            "evidence_id": "ev_0",
            "contour_id": "contour_000",
            "embedding_row_index": 0,
            "row_index": 0,
            "image_source": "image_path",
            "failure_rank": 1,
            "failure_score": 0.7,
            "failure_reasons": "low_confidence",
        },
        {
            "evidence_id": "ev_1",
            "contour_id": "contour_001",
            "embedding_row_index": 1,
            "row_index": 1,
            "image_source": "image_path",
            "failure_rank": 2,
            "failure_score": 0.6,
            "failure_reasons": "high_entropy",
        },
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_run_contour_ablation_writes_anatomy_report(tmp_path: Path) -> None:
    config = _write_ablation_config(tmp_path)
    checkpoint = Path(train(config, preset="smoke", max_steps=1)["checkpoint"])
    targets = _write_targets(tmp_path / "ablation_targets.json")

    result = run_contour_ablation(
        checkpoint=checkpoint,
        config=config,
        targets=targets,
        output_dir=tmp_path / "ablation",
        batch_size=2,
        device="cpu",
    )

    assert result["status"] == "pass"
    rows = pd.read_csv(result["artifacts"]["ablation_results_csv"])
    assert len(rows) == 8
    assert set(rows["ablation_mode"]) == {"baseline", "drop_object", "drop_context", "drop_shape"}
    assert "matched_similarity_drop" in rows.columns
    report = Path(result["artifacts"]["anatomy_of_failure"]).read_text(encoding="utf-8")
    assert "Anatomy of a Failure" in report
    assert "equal-area circle" in report


def test_ablate_cli_writes_outputs(tmp_path: Path) -> None:
    config = _write_ablation_config(tmp_path)
    checkpoint = Path(train(config, preset="smoke", max_steps=1)["checkpoint"])
    targets = _write_targets(tmp_path / "ablation_targets.json")

    result = CliRunner().invoke(
        app,
        [
            "ablate",
            "--checkpoint",
            str(checkpoint),
            "--config",
            str(config),
            "--targets",
            str(targets),
            "--output",
            str(tmp_path / "ablation_cli"),
            "--batch-size",
            "2",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert Path(payload["artifacts"]["failure_gallery_with_ablation_csv"]).exists()
