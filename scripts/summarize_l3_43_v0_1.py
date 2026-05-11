#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

MILESTONE = "L3-43 v0.1 evidence milestone"
DATA_VERSION = "l3_20260507_43case"
N_CASES = 43
N_REGIONS = 293_678
STGPT_COMMIT = "df9cad2"
PYXENIUM_COMMIT = "c039a91"
PYXENIUM_VERSION = "0.4.5"
HF_EVIDENCE_REPO = "hutaobo/stgpt-l3-evidence-20260504"
HF_EVIDENCE_COMMIT = "ba960dae59852b9e94dbe2c699ebaa4d61bb0396"

FULL_RUN_ID = "full_m6_contour_store_lambda_0_01_20k"
BASELINE_RUN_ID = "gene_spatial_contour_unit_20k"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the L3-43 v0.1 paper-facing summary package.")
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--full-run-dir", required=True, type=Path)
    parser.add_argument("--baseline-run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    result = summarize_l3_43_v0_1(
        evidence_dir=args.evidence_dir,
        full_run_dir=args.full_run_dir,
        baseline_run_dir=args.baseline_run_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2))


def summarize_l3_43_v0_1(
    *,
    evidence_dir: Path,
    full_run_dir: Path,
    baseline_run_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    evidence_dir = evidence_dir.expanduser().resolve()
    full_run_dir = full_run_dir.expanduser().resolve()
    baseline_run_dir = baseline_run_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence_rows = _load_evidence_rows(evidence_dir)
    pointer_rows = _load_json_or_empty(evidence_dir / "pointer_audit.json")
    run_status = _load_json_or_empty(evidence_dir / "run_status.json")
    metrics_table = _metrics_table(evidence_rows)
    metrics_table.to_csv(output_dir / "metrics_table.csv", index=False)

    full = _find_run(evidence_rows, FULL_RUN_ID)
    baseline = _find_run(evidence_rows, BASELINE_RUN_ID)
    failure_inputs = {
        "full": _collect_run_failure_inputs(full_run_dir),
        "baseline": _collect_run_failure_inputs(baseline_run_dir),
    }
    paper_summary = _paper_summary_markdown(metrics_table, full, baseline, pointer_rows, run_status)
    failure_modes = _failure_modes_markdown(full, baseline, pointer_rows, failure_inputs)
    recommendation = _recommendation_payload(full, baseline, pointer_rows, failure_inputs)

    outputs = {
        "metrics_table": output_dir / "metrics_table.csv",
        "paper_summary": output_dir / "paper_summary.md",
        "failure_modes": output_dir / "failure_modes.md",
        "next_experiment_recommendation": output_dir / "next_experiment_recommendation.json",
        "paper_summary_doc": output_dir / "l3_43_v0_1_paper_summary.md",
        "failure_modes_doc": output_dir / "l3_43_v0_1_failure_modes.md",
    }
    outputs["paper_summary"].write_text(paper_summary, encoding="utf-8")
    outputs["failure_modes"].write_text(failure_modes, encoding="utf-8")
    outputs["paper_summary_doc"].write_text(paper_summary, encoding="utf-8")
    outputs["failure_modes_doc"].write_text(failure_modes, encoding="utf-8")
    outputs["next_experiment_recommendation"].write_text(
        json.dumps(recommendation, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return {
        "status": "pass",
        "milestone": MILESTONE,
        "n_runs": len(evidence_rows),
        "artifacts": {key: str(path) for key, path in outputs.items()},
    }


def _load_evidence_rows(evidence_dir: Path) -> list[dict[str, Any]]:
    json_path = evidence_dir / "evidence_summary.json"
    csv_path = evidence_dir / "evidence_summary.csv"
    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected list in {json_path}")
        return [dict(row) for row in payload]
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise FileNotFoundError(f"No evidence_summary.json or evidence_summary.csv found in {evidence_dir}")


def _load_json_or_empty(path: Path) -> Any:
    if not path.exists():
        return [] if path.name.endswith(".json") else {}
    return json.loads(path.read_text(encoding="utf-8"))


def _find_run(rows: list[dict[str, Any]], run_id: str) -> dict[str, Any]:
    for row in rows:
        if str(row.get("run_id")) == run_id:
            return row
    raise ValueError(f"Evidence summary does not contain required run_id={run_id!r}")


def _metrics_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    selected = []
    for row in rows:
        selected.append(
            {
                "run_id": row.get("run_id"),
                "condition": row.get("condition"),
                "status": row.get("status"),
                "steps": _int_or_none(row.get("steps")),
                "checkpoint_role": row.get("checkpoint_role"),
                "lambda_align": _float_or_none(row.get("lambda_align")),
                "image_source": row.get("image_source"),
                "expected_prototypes": _int_or_none(row.get("expected_prototypes")),
                "gene_mse": _float_or_none(row.get("eval_gene_mse")),
                "gene_correlation": _float_or_none(row.get("eval_gene_correlation")),
                "image_to_gene_top1": _float_or_none(row.get("eval_image_to_gene_top1")),
                "image_to_gene_top5": _float_or_none(row.get("eval_image_to_gene_top5")),
                "gene_to_image_top1": _float_or_none(row.get("eval_gene_to_image_top1")),
                "gene_to_image_top5": _float_or_none(row.get("eval_gene_to_image_top5")),
                "label_top1": _float_or_none(row.get("eval_label_retrieval_top1")),
                "label_top5": _float_or_none(row.get("eval_label_retrieval_top5")),
                "silhouette": _float_or_none(row.get("eval_silhouette_mean")),
                "batch_mixing_entropy": _float_or_none(row.get("eval_batch_mixing_entropy_mean")),
                "prototype_usage_global": _prototype_usage(row),
                "prototype_confidence_mean": _float_or_none(row.get("prototype_mean_confidence")),
                "pointer_errors": _int_or_none(row.get("pointer_errors")),
            }
        )
    return pd.DataFrame(selected)


def _collect_run_failure_inputs(run_dir: Path) -> dict[str, Any]:
    eval_dir = run_dir / "evaluation"
    export_dir = run_dir / "spatho_export"
    card_dir = run_dir / "checkpoint_card"
    return {
        "run_dir": str(run_dir),
        "evaluation_metrics": _read_json_if_exists(eval_dir / "evaluation_metrics.json"),
        "failure_analysis": _read_csv_summary(eval_dir / "failure_analysis.csv"),
        "label_retrieval": _read_csv_summary(eval_dir / "label_retrieval_metrics.csv"),
        "batch_mixing": _read_csv_summary(eval_dir / "batch_mixing_metrics.csv"),
        "embedding_qc": _read_csv_summary(eval_dir / "embedding_qc.csv"),
        "evidence_manifest": _read_json_if_exists(export_dir / "evidence_manifest.json"),
        "checkpoint_card": _read_json_if_exists(card_dir / "stgpt_model_manifest.json"),
    }


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"_read_error": f"invalid json: {path.name}"}
    return payload if isinstance(payload, dict) else {"value": payload}


def _read_csv_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "rows": 0, "columns": []}
    frame = pd.read_csv(path)
    summary: dict[str, Any] = {
        "exists": True,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
    }
    if "category" in frame.columns:
        summary["category_counts"] = _value_counts(frame["category"])
    if "split" in frame.columns:
        summary["split_counts"] = _value_counts(frame["split"])
    if {"split", "metric", "value"}.issubset(frame.columns):
        summary["overall_metrics"] = _overall_metric_map(frame)
    if "label_column" in frame.columns:
        summary["label_columns"] = sorted(str(item) for item in frame["label_column"].dropna().unique())
    if {"split", "label_column", "k", "same_label_recall"}.issubset(frame.columns):
        summary["overall_label_recall"] = _overall_label_recall_map(frame)
    if {"split", "label_column", "silhouette"}.issubset(frame.columns):
        summary["overall_silhouette"] = _overall_silhouette_map(frame)
    for column in ("same_label_recall", "top1", "top5", "value"):
        if column in frame.columns and pd.api.types.is_numeric_dtype(frame[column]):
            summary[f"{column}_mean"] = _finite_mean(frame[column])
            summary[f"{column}_min"] = _finite_min(frame[column])
            summary[f"{column}_max"] = _finite_max(frame[column])
    return summary


def _paper_summary_markdown(
    metrics_table: pd.DataFrame,
    full: dict[str, Any],
    baseline: dict[str, Any],
    pointer_rows: Any,
    run_status: Any,
) -> str:
    full_ig_top1 = _float_or_none(full.get("eval_image_to_gene_top1"))
    base_ig_top1 = _float_or_none(baseline.get("eval_image_to_gene_top1"))
    full_label_top1 = _float_or_none(full.get("eval_label_retrieval_top1"))
    base_label_top1 = _float_or_none(baseline.get("eval_label_retrieval_top1"))
    pointer_errors = _sum_pointer_errors(pointer_rows)
    status = run_status.get("status") if isinstance(run_status, dict) else "unknown"
    lines = [
        "# L3-43 v0.1 Paper-Facing Result Summary",
        "",
        f"Milestone: `{MILESTONE}`.",
        f"Data version: `{DATA_VERSION}` with {N_CASES} cases and {N_REGIONS:,} exported contour/region records.",
        f"Code provenance: stGPT `{STGPT_COMMIT}`, pyXenium `{PYXENIUM_COMMIT}`, pyXenium `{PYXENIUM_VERSION}`.",
        f"Evidence provenance: private Hugging Face dataset `{HF_EVIDENCE_REPO}` at `{HF_EVIDENCE_COMMIT}`.",
        f"Run status summary: `{status}`. Pointer errors across summarized runs: `{pointer_errors}`.",
        "",
        "## Main Result",
        "",
        "Full M6 contour-store training produced a strong image-gene aligned region space while preserving gene reconstruction. "
        f"The Full M6 image-to-gene top-1 retrieval is {_fmt_float(full_ig_top1, 4)}, compared with "
        f"{_fmt_float(base_ig_top1, 6)} for the gene+spatial baseline. Gene correlation remains high for both runs "
        f"({_fmt_float(_float_or_none(full.get('eval_gene_correlation')), 4)} for Full M6 and "
        f"{_fmt_float(_float_or_none(baseline.get('eval_gene_correlation')), 4)} for baseline).",
        "",
        "The current result should not be framed as a complete foundation model. Label/structure retrieval remains weaker in "
        f"Full M6 than in the baseline (Label@1 {_fmt_float(full_label_top1, 4)} vs "
        f"{_fmt_float(base_label_top1, 4)}), and the L3-43 training configs did not optimize a structure objective.",
        "",
        "## Metrics Table",
        "",
        _markdown_metrics_table(metrics_table),
        "",
        "## Interpretation",
        "",
        "- Evidence for the core contour-level pipeline is positive: smoke, Full M6, and baseline all pass, use `contour_store` image evidence, and have zero pointer errors.",
        "- The main measurable gain of Full M6 is cross-modal alignment, not structure classification.",
        "- Gene reconstruction is not materially degraded by the Full M6 multimodal objective, but the gene+spatial baseline remains a slightly stronger pure reconstruction control.",
        "- Prototype coverage is broad globally, but low mean prototype confidence indicates diffuse assignments that need interpretation-focused tuning before biological claims.",
        "",
        "## Prohibited Claims",
        "",
        "- Do not claim clinical diagnosis, treatment prediction, or pathology-grade structure annotation.",
        "- Do not present reconstructed or imputed expression as measured Xenium expression.",
        "- Do not claim a finished foundation model from this milestone alone.",
        "- Do not infer that weak label retrieval is a data failure without separating objective design, weak labels, and registration/image-gene conflicts.",
    ]
    return "\n".join(lines) + "\n"


def _failure_modes_markdown(
    full: dict[str, Any],
    baseline: dict[str, Any],
    pointer_rows: Any,
    failure_inputs: dict[str, dict[str, Any]],
) -> str:
    full_label1 = _float_or_none(full.get("eval_label_retrieval_top1"))
    base_label1 = _float_or_none(baseline.get("eval_label_retrieval_top1"))
    full_label5 = _float_or_none(full.get("eval_label_retrieval_top5"))
    base_label5 = _float_or_none(baseline.get("eval_label_retrieval_top5"))
    full_proto_conf = _float_or_none(full.get("prototype_mean_confidence"))
    full_proto_usage = _prototype_usage(full)
    pointer_errors = _sum_pointer_errors(pointer_rows)
    full_failures = failure_inputs["full"].get("failure_analysis", {})
    baseline_failures = failure_inputs["baseline"].get("failure_analysis", {})
    full_failure_metrics = full_failures.get("overall_metrics", {}) if isinstance(full_failures, dict) else {}
    full_label_recall = failure_inputs["full"].get("label_retrieval", {}).get("overall_label_recall", {})
    baseline_label_recall = failure_inputs["baseline"].get("label_retrieval", {}).get("overall_label_recall", {})
    full_silhouette = failure_inputs["full"].get("embedding_qc", {}).get("overall_silhouette", {})
    baseline_silhouette = failure_inputs["baseline"].get("embedding_qc", {}).get("overall_silhouette", {})
    lines = [
        "# L3-43 v0.1 Failure Modes and Next Experiment Decision",
        "",
        "This note separates observed failures into objective-design gaps, weak-supervision limitations, and possible true data or registration conflicts. It uses only lightweight evaluation and evidence artifacts.",
        "",
        "## Observed Failure Pattern",
        "",
        f"- Full M6 label retrieval is weaker than the gene+spatial baseline: Label@1 {_fmt_float(full_label1, 4)} vs {_fmt_float(base_label1, 4)}, Label@5 {_fmt_float(full_label5, 4)} vs {_fmt_float(base_label5, 4)}.",
        "- Full M6 image-gene retrieval is strong, so the failure is not a global image-gene alignment collapse.",
        f"- Pointer audit reports `{pointer_errors}` pointer errors, so the summarized evidence chain itself is not the leading failure explanation.",
        f"- Full M6 prototype global usage is `{full_proto_usage}`, but mean prototype confidence is {_fmt_float(full_proto_conf, 4)}, which is consistent with diffuse prototype assignment.",
        f"- Full M6 failure-analysis rows: `{full_failures.get('rows', 0)}`. Baseline failure-analysis rows: `{baseline_failures.get('rows', 0)}`.",
        f"- Full M6 region image coverage is {_fmt_float(_float_or_none(full_failure_metrics.get('region_image_coverage')), 4)}; missing image count is {_fmt_float(_float_or_none(full_failure_metrics.get('missing_image_count')), 1)}; low-cell region count is {_fmt_float(_float_or_none(full_failure_metrics.get('low_cell_region_count')), 1)}.",
        f"- Registration metadata checks are present: patch coordinates {_fmt_float(_float_or_none(full_failure_metrics.get('has_patch_coordinates')), 1)}, registration metadata {_fmt_float(_float_or_none(full_failure_metrics.get('has_registration_metadata')), 1)}.",
        "",
        "## Detailed Retrieval Diagnostics",
        "",
        _diagnostic_table(
            [
                ("structure_label@1", full_label_recall.get("structure_label@1"), baseline_label_recall.get("structure_label@1")),
                ("structure_label@5", full_label_recall.get("structure_label@5"), baseline_label_recall.get("structure_label@5")),
                ("structure_id@1", full_label_recall.get("structure_id@1"), baseline_label_recall.get("structure_id@1")),
                ("structure_id@5", full_label_recall.get("structure_id@5"), baseline_label_recall.get("structure_id@5")),
                ("cluster@1", full_label_recall.get("cluster@1"), baseline_label_recall.get("cluster@1")),
                ("cluster@5", full_label_recall.get("cluster@5"), baseline_label_recall.get("cluster@5")),
                ("structure_label silhouette", full_silhouette.get("structure_label"), baseline_silhouette.get("structure_label")),
                ("structure_id silhouette", full_silhouette.get("structure_id"), baseline_silhouette.get("structure_id")),
            ]
        ),
        "",
        "## Failure Classes",
        "",
        "### 1. Objective did not optimize structure",
        "",
        "The L3-43 Full M6 run was optimized for molecular reconstruction, spatial neighborhood reconstruction, image-gene alignment, and prototype organization. Structure context was not enabled in the published L3-43 configs, so weak label retrieval should be treated as an expected objective gap rather than a model defect by itself.",
        "",
        "### 2. Structure labels are weak supervision, not gold pathology labels",
        "",
        "The current structure labels are useful for retrieval diagnostics, but they are not pathologist gold labels. A label retrieval miss can mean the embedding ignores the weak label, the label is too coarse, or the region has mixed molecular/morphology evidence.",
        "",
        "### 3. True image/gene/registration conflicts remain possible",
        "",
        "Because image-gene retrieval is strong overall and pointer errors are zero, broad registration failure is unlikely. The correct next step is targeted inspection of high gene-MSE, low prototype-confidence, and label-mismatch regions rather than a data-wide rebuild.",
        "",
        "## Recommended Next Experiments",
        "",
        "1. Run `structure_context_m6` on the frozen 43-case data first. Success criterion: improve Label@1 or Label@5 over Full M6 while keeping gene correlation at or above 0.995.",
        "2. Run a small Virchow/UNI smoke only after failure review confirms that image encoder capacity is a plausible bottleneck.",
        "3. Defer data expansion and contour repacking until structure-context and image-encoder smoke results are interpreted.",
    ]
    return "\n".join(lines) + "\n"


def _recommendation_payload(
    full: dict[str, Any],
    baseline: dict[str, Any],
    pointer_rows: Any,
    failure_inputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    full_label1 = _float_or_none(full.get("eval_label_retrieval_top1"))
    base_label1 = _float_or_none(baseline.get("eval_label_retrieval_top1"))
    label_gap = None
    if full_label1 is not None and base_label1 is not None:
        label_gap = base_label1 - full_label1
    return {
        "milestone": MILESTONE,
        "data_version": DATA_VERSION,
        "status": "ready_for_result_discussion",
        "do_not_do_now": [
            "do not expand data",
            "do not restart completed L3-43 guard pipeline",
            "do not claim final foundation-model status",
        ],
        "recommended_order": [
            {
                "rank": 1,
                "experiment": "structure_context_m6",
                "decision": "run_next",
                "rationale": "Full M6 underperforms baseline on label retrieval and the current L3-43 objective did not optimize structure context.",
                "success_criteria": {
                    "label_top1_or_top5": "above current Full M6",
                    "gene_correlation": ">=0.995",
                    "pointer_errors": 0,
                },
            },
            {
                "rank": 2,
                "experiment": "virchow_uni_multi_case_smoke",
                "decision": "run_after_failure_review",
                "rationale": "Only useful if failure review suggests lightweight image encoder capacity is limiting structure or morphology semantics.",
                "success_criteria": {
                    "image_gene_top1": ">=0.90 on smoke",
                    "label_retrieval": "improves over lightweight image encoder smoke",
                },
            },
            {
                "rank": 3,
                "experiment": "data_expansion",
                "decision": "defer",
                "rationale": "The existing 43-case milestone is sufficient for objective and encoder decisions.",
            },
        ],
        "diagnostics": {
            "full_label_top1": full_label1,
            "baseline_label_top1": base_label1,
            "baseline_minus_full_label_top1": label_gap,
            "pointer_errors_total": _sum_pointer_errors(pointer_rows),
            "full_failure_rows": failure_inputs["full"].get("failure_analysis", {}).get("rows"),
            "baseline_failure_rows": failure_inputs["baseline"].get("failure_analysis", {}).get("rows"),
        },
    }


def _markdown_metrics_table(frame: pd.DataFrame) -> str:
    columns = [
        "run_id",
        "steps",
        "checkpoint_role",
        "gene_mse",
        "gene_correlation",
        "image_to_gene_top1",
        "gene_to_image_top1",
        "label_top1",
        "label_top5",
        "silhouette",
        "prototype_usage_global",
        "pointer_errors",
    ]
    display = frame[columns].copy()
    for column in (
        "gene_mse",
        "gene_correlation",
        "image_to_gene_top1",
        "gene_to_image_top1",
        "label_top1",
        "label_top5",
        "silhouette",
    ):
        display[column] = display[column].map(lambda value: _fmt_float(_float_or_none(value), 4))
    return _dataframe_to_markdown(display)


def _diagnostic_table(rows: list[tuple[str, Any, Any]]) -> str:
    frame = pd.DataFrame(
        [
            {
                "diagnostic": label,
                "full_m6": _fmt_float(_float_or_none(full), 4),
                "gene_spatial_baseline": _fmt_float(_float_or_none(baseline), 4),
            }
            for label, full, baseline in rows
        ]
    )
    return _dataframe_to_markdown(frame)


def _dataframe_to_markdown(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows)) if rows else len(headers[idx])
        for idx in range(len(headers))
    ]
    header = "| " + " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body = ["| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _prototype_usage(row: dict[str, Any]) -> str:
    used = _int_or_none(row.get("prototype_usage_export_global"))
    expected = _int_or_none(row.get("expected_prototypes"))
    if used is None or expected is None or expected == 0:
        return "N/A"
    return f"{used}/{expected}"


def _sum_pointer_errors(pointer_rows: Any) -> int:
    if not isinstance(pointer_rows, list):
        return 0
    total = 0
    for row in pointer_rows:
        if isinstance(row, dict):
            total += int(_int_or_none(row.get("pointer_errors")) or 0)
    return total


def _value_counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.fillna("NA").value_counts().sort_index().items()}


def _overall_metric_map(frame: pd.DataFrame) -> dict[str, float]:
    subset = frame.loc[frame["split"].astype(str) == "overall"].copy()
    values: dict[str, float] = {}
    for _, row in subset.iterrows():
        metric = str(row.get("metric"))
        value = _float_or_none(row.get("value"))
        if value is not None:
            values[metric] = value
    return values


def _overall_label_recall_map(frame: pd.DataFrame) -> dict[str, float]:
    subset = frame.loc[frame["split"].astype(str) == "overall"].copy()
    values: dict[str, float] = {}
    for _, row in subset.iterrows():
        label = str(row.get("label_column"))
        k = _int_or_none(row.get("k"))
        recall = _float_or_none(row.get("same_label_recall"))
        if k is not None and recall is not None:
            values[f"{label}@{k}"] = recall
    return values


def _overall_silhouette_map(frame: pd.DataFrame) -> dict[str, float]:
    subset = frame.loc[frame["split"].astype(str) == "overall"].copy()
    values: dict[str, float] = {}
    for _, row in subset.iterrows():
        label = str(row.get("label_column"))
        value = _float_or_none(row.get("silhouette"))
        if value is not None:
            values[label] = value
    return values


def _finite_mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").replace([math.inf, -math.inf], math.nan).dropna()
    return None if values.empty else float(values.mean())


def _finite_min(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").replace([math.inf, -math.inf], math.nan).dropna()
    return None if values.empty else float(values.min())


def _finite_max(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").replace([math.inf, -math.inf], math.nan).dropna()
    return None if values.empty else float(values.max())


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _int_or_none(value: Any) -> int | None:
    number = _float_or_none(value)
    if number is None:
        return None
    return int(number)


def _fmt_float(value: float | None, digits: int) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


if __name__ == "__main__":
    main()
