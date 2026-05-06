from __future__ import annotations

import json
from pathlib import Path

import torch
from typer.testing import CliRunner

from stgpt.cli import app
from stgpt.config import StGPTConfig
from stgpt.evidence import check_artifact_contract


def _write_config(path: Path, *, n_prototypes: int = 3) -> dict:
    payload = StGPTConfig(case_name="contract_mock").model_dump()
    payload["model"]["n_prototypes"] = n_prototypes
    payload["model"]["d_model"] = 32
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _write_checkpoint(path: Path, config_payload: dict, *, n_prototypes: int = 3, step: int = 2) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": {"prototype_head.weight": torch.zeros(n_prototypes, config_payload["model"]["d_model"])},
            "config": config_payload,
            "optimizer_state": {"state": {}, "param_groups": []},
            "scheduler_state": None,
            "metrics": [{"step": idx, "val_loss": 1.0 / idx} for idx in range(1, step + 1)],
            "training_summary": {
                "steps": step,
                "best_metric": 0.5,
                "best_alignment_metric": 0.8,
            },
        },
        path,
    )
    return path


def test_check_artifact_contract_passes_matching_checkpoint(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_payload = _write_config(config_path, n_prototypes=3)
    checkpoint = _write_checkpoint(tmp_path / "best_alignment.pt", config_payload, n_prototypes=3, step=2)
    run_dir = tmp_path / "run"
    (run_dir / "train").mkdir(parents=True)
    (run_dir / "train" / "metrics.json").write_text(
        json.dumps([{"step": 1, "val_loss": 0.7}, {"step": 2, "val_loss": 0.5}]),
        encoding="utf-8",
    )

    result = check_artifact_contract(checkpoint=checkpoint, config=config_path, run_dir=run_dir)

    assert result["status"] == "pass"
    assert result["checks"]["checkpoint_n_prototypes"] == 3
    assert result["checks"]["run_metrics_max_step"] == 2


def test_check_artifact_contract_fails_prototype_mismatch(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_payload = _write_config(config_path, n_prototypes=3)
    checkpoint = _write_checkpoint(tmp_path / "best_alignment.pt", config_payload, n_prototypes=2, step=2)

    result = check_artifact_contract(checkpoint=checkpoint, config=config_path)

    assert result["status"] == "fail"
    assert "prototype_count_mismatch" in result["errors"]


def test_check_artifact_contract_warns_on_run_metric_gap(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_payload = _write_config(config_path, n_prototypes=3)
    checkpoint = _write_checkpoint(tmp_path / "best_alignment.pt", config_payload, n_prototypes=3, step=4)
    run_dir = tmp_path / "run"
    (run_dir / "train").mkdir(parents=True)
    (run_dir / "train" / "metrics.json").write_text(
        json.dumps([{"step": 1}, {"step": 3}, {"step": 4}]),
        encoding="utf-8",
    )

    result = check_artifact_contract(
        checkpoint=checkpoint,
        config=config_path,
        run_dir=run_dir,
        output=tmp_path / "contract.json",
    )

    assert result["status"] == "warning"
    assert "run_metrics_step_discontinuity" in result["warnings"]
    assert (tmp_path / "contract.json").exists()


def test_check_contract_cli(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_payload = _write_config(config_path, n_prototypes=3)
    checkpoint = _write_checkpoint(tmp_path / "best_alignment.pt", config_payload, n_prototypes=3, step=1)

    result = CliRunner().invoke(app, ["check-contract", "--checkpoint", str(checkpoint), "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "pass"
