from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from stgpt.cli import app
from stgpt.evidence.watchtower import generate_watchtower_report


def _write_watchtower_suite(path: Path, *, config: Path, run_dir: Path) -> Path:
    payload = {
        "suite_name": "mock_watchtower",
        "runs": [
            {
                "run_id": "breast_full_m6_lambda_0_1",
                "tissue": "breast",
                "condition": "Full M6 Pareto lambda=0.1",
                "config_path": str(config),
                "run_dir": str(run_dir),
                "expected_image_source": "contour_store",
                "expected_prototypes": 32,
                "requires_alignment_telemetry": True,
                "checkpoint_role": "best_alignment",
                "lambda_align": 0.1,
                "suite_stage": "pareto_grid",
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_watchtower_metrics(run_dir: Path) -> None:
    train_dir = run_dir / "train"
    ckpt_dir = train_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        {
            "step": 1,
            "lr": 0.0003,
            "loss": 0.8,
            "val_gene_loss": 0.4,
            "val_image_to_gene_top5": 0.1,
            "val_gene_to_image_top5": 0.1,
            "val_alignment_score": 0.1,
            "prototype_usage_count": 12,
            "prototype_dead_codes": 20,
        },
        {
            "step": 250,
            "lr": 0.0002,
            "loss": 0.4,
            "val_gene_loss": 0.2,
            "val_image_to_gene_top5": 0.3,
            "val_gene_to_image_top5": 0.5,
            "val_alignment_score": 0.4,
            "prototype_usage_count": 20,
            "prototype_dead_codes": 12,
        },
        {
            "step": 500,
            "lr": 0.0001,
            "loss": 0.3,
            "val_gene_loss": 0.18,
            "val_image_to_gene_top5": 0.3,
            "val_gene_to_image_top5": 0.4,
            "val_alignment_score": 0.35,
            "prototype_usage_count": 22,
            "prototype_dead_codes": 10,
            "sinkhorn_nonfinite_count": 0,
        },
    ]
    (train_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (ckpt_dir / "step_000500.pt").write_bytes(b"checkpoint")


def test_watchtower_report_reads_alignment_bursts(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("training:\n  max_steps: 1000\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    _write_watchtower_metrics(run_dir)
    suite = _write_watchtower_suite(tmp_path / "suite.json", config=config, run_dir=run_dir)

    result = generate_watchtower_report(suite, tmp_path / "watchtower")

    assert result["n_runs"] == 1
    summary = pd.read_csv(result["artifacts"]["watchtower_summary_csv"])
    row = summary.iloc[0]
    assert int(row["latest_step"]) == 500
    assert int(row["best_alignment_step"]) == 250
    assert float(row["alignment_burst_delta"]) == 0.3
    html = Path(result["artifacts"]["watchtower_report_html"]).read_text(encoding="utf-8")
    assert "Contour-Native Watchtower" in html


def test_watchtower_cli_writes_outputs(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("training:\n  max_steps: 1000\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    _write_watchtower_metrics(run_dir)
    suite = _write_watchtower_suite(tmp_path / "suite.json", config=config, run_dir=run_dir)

    result = CliRunner().invoke(app, ["watchtower", "--suite", str(suite), "--output", str(tmp_path / "out")])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert Path(payload["artifacts"]["watchtower_status"]).exists()
