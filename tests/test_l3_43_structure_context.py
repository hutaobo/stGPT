from __future__ import annotations

from pathlib import Path

import yaml

from stgpt.config import StGPTConfig


def test_structure_context_m6_config_conditions_on_structure_without_leak(monkeypatch) -> None:
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("STGPT_OUTPUT_ROOT", str(repo / "outputs"))
    monkeypatch.setenv("STGPT_XENIUM_SLIDES", str(repo / "outputs" / "xenium_slides"))

    cfg = StGPTConfig.from_file(repo / "configs" / "pilots" / "l3_43" / "structure_context_m6_20k.yaml")

    assert cfg.case_name == "l3_20260507_43case_structure_context_m6_20k"
    assert cfg.data.mode == "corpus"
    assert cfg.data.include_structure_context is True
    assert cfg.model.use_structure_context is True
    assert cfg.model.use_image_context is True
    assert cfg.model.use_spatial_context is True
    # Structure is fed as an input context token, so it must NOT also be a
    # prediction target (that would let the structure head copy the input).
    assert cfg.training.structure_loss_weight == 0.0
    assert cfg.training.image_gene_loss_weight == 0.01
    assert cfg.split.strategy == "slide_holdout"


def test_l3_43_evidence_suite_includes_structure_context_run() -> None:
    suite_path = Path(__file__).resolve().parents[1] / "configs" / "evidence" / "l3_43.yaml"
    payload = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    runs = {run["run_id"]: run for run in payload["runs"]}

    run = runs["structure_context_m6_20k"]
    assert run["checkpoint_role"] == "best_loss"
    assert run["suite_stage"] == "l3_43_structure_context"
    assert run["requires_alignment_telemetry"] is True
    assert run["config_path"].endswith("structure_context_m6_20k.yaml")
