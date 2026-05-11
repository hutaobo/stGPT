from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


def _load_summary_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "summarize_l3_43_v0_1.py"
    spec = importlib.util.spec_from_file_location("summarize_l3_43_v0_1", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_run(run_dir: Path) -> None:
    eval_dir = run_dir / "evaluation"
    export_dir = run_dir / "spatho_export"
    card_dir = run_dir / "checkpoint_card"
    eval_dir.mkdir(parents=True)
    export_dir.mkdir(parents=True)
    card_dir.mkdir(parents=True)
    (eval_dir / "evaluation_metrics.json").write_text(json.dumps({"status": "pass"}), encoding="utf-8")
    pd.DataFrame(
        [{"split": "overall", "category": "patch", "metric": "missing_image_count", "value": 0.0, "detail": "ok"}]
    ).to_csv(eval_dir / "failure_analysis.csv", index=False)
    pd.DataFrame([{"split": "overall", "same_label_recall": 0.2}]).to_csv(
        eval_dir / "label_retrieval_metrics.csv", index=False
    )
    pd.DataFrame([{"split": "overall", "value": 0.1}]).to_csv(eval_dir / "batch_mixing_metrics.csv", index=False)
    pd.DataFrame([{"label_column": "structure_id", "value": 0.1}]).to_csv(eval_dir / "embedding_qc.csv", index=False)
    (export_dir / "evidence_manifest.json").write_text(json.dumps({"artifacts": []}), encoding="utf-8")
    (card_dir / "stgpt_model_manifest.json").write_text(json.dumps({"checkpoint": "mock"}), encoding="utf-8")


def test_l3_43_v0_1_summary_outputs_decision_package(tmp_path: Path) -> None:
    module = _load_summary_module()
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    full_run = tmp_path / "full"
    baseline_run = tmp_path / "baseline"
    _write_run(full_run)
    _write_run(baseline_run)
    rows = [
        {
            "run_id": "smoke_5case_full_m6_lambda_0_01_500",
            "condition": "smoke",
            "status": "pass",
            "steps": 500,
            "checkpoint_role": "best_alignment",
            "lambda_align": 0.01,
            "image_source": "contour_store",
            "expected_prototypes": 64,
            "eval_gene_mse": 0.05,
            "eval_gene_correlation": 0.85,
            "eval_image_to_gene_top1": 0.002,
            "eval_gene_to_image_top1": 0.003,
            "eval_image_to_gene_top5": 0.01,
            "eval_gene_to_image_top5": 0.02,
            "eval_label_retrieval_top1": 0.38,
            "eval_label_retrieval_top5": 0.76,
            "eval_silhouette_mean": -0.26,
            "prototype_usage_export_global": 43,
            "prototype_mean_confidence": 0.05,
            "pointer_errors": 0,
        },
        {
            "run_id": "full_m6_contour_store_lambda_0_01_20k",
            "condition": "full",
            "status": "pass",
            "steps": 20000,
            "checkpoint_role": "best_alignment",
            "lambda_align": 0.01,
            "image_source": "contour_store",
            "expected_prototypes": 128,
            "eval_gene_mse": 0.000491266,
            "eval_gene_correlation": 0.998787,
            "eval_image_to_gene_top1": 0.9417355,
            "eval_gene_to_image_top1": 0.9563297,
            "eval_image_to_gene_top5": 0.9756,
            "eval_gene_to_image_top5": 0.9811,
            "eval_label_retrieval_top1": 0.0629601,
            "eval_label_retrieval_top5": 0.257748,
            "eval_silhouette_mean": -0.228,
            "prototype_usage_export_global": 127,
            "prototype_mean_confidence": 0.02365,
            "pointer_errors": 0,
        },
        {
            "run_id": "gene_spatial_contour_unit_20k",
            "condition": "baseline",
            "status": "pass",
            "steps": 20000,
            "checkpoint_role": "best_loss",
            "lambda_align": 0.0,
            "image_source": "contour_store",
            "expected_prototypes": 0,
            "eval_gene_mse": 0.000454858,
            "eval_gene_correlation": 0.998862,
            "eval_image_to_gene_top1": 0.0000034,
            "eval_gene_to_image_top1": 0.0000034,
            "eval_image_to_gene_top5": 0.000017,
            "eval_gene_to_image_top5": 0.000017,
            "eval_label_retrieval_top1": 0.128277,
            "eval_label_retrieval_top5": 0.390489,
            "eval_silhouette_mean": -0.4666,
            "prototype_usage_export_global": 0,
            "prototype_mean_confidence": None,
            "pointer_errors": 0,
        },
    ]
    (evidence_dir / "evidence_summary.json").write_text(json.dumps(rows), encoding="utf-8")
    (evidence_dir / "pointer_audit.json").write_text(
        json.dumps([{"run_id": row["run_id"], "pointer_errors": 0} for row in rows]),
        encoding="utf-8",
    )
    (evidence_dir / "run_status.json").write_text(json.dumps({"status": "pass"}), encoding="utf-8")

    result = module.summarize_l3_43_v0_1(
        evidence_dir=evidence_dir,
        full_run_dir=full_run,
        baseline_run_dir=baseline_run,
        output_dir=tmp_path / "out",
    )

    assert result["status"] == "pass"
    table = pd.read_csv(result["artifacts"]["metrics_table"])
    assert set(table["run_id"]) == {
        "smoke_5case_full_m6_lambda_0_01_500",
        "full_m6_contour_store_lambda_0_01_20k",
        "gene_spatial_contour_unit_20k",
    }
    full = table.loc[table["run_id"] == "full_m6_contour_store_lambda_0_01_20k"].iloc[0]
    baseline = table.loc[table["run_id"] == "gene_spatial_contour_unit_20k"].iloc[0]
    assert full["image_to_gene_top1"] > 0.94
    assert baseline["image_to_gene_top1"] < 0.001
    assert baseline["label_top1"] > full["label_top1"]
    paper = Path(result["artifacts"]["paper_summary_doc"]).read_text(encoding="utf-8")
    assert "not be framed as a complete foundation model" in paper
    recommendation = json.loads(Path(result["artifacts"]["next_experiment_recommendation"]).read_text(encoding="utf-8"))
    assert recommendation["recommended_order"][0]["experiment"] == "structure_context_m6"
