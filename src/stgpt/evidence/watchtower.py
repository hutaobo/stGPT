from __future__ import annotations

import json
import math
import os
import re
import time
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import yaml  # type: ignore[import-untyped]

from .summary import EvidenceSuiteSpec, load_evidence_suite


def generate_watchtower_report(
    suite: EvidenceSuiteSpec | str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Summarize live E4 long-run telemetry without launching training.

    The watchtower is artifact-first: it reads suite specs, configs, metrics, and
    checkpoint directories, then emits lightweight CSV/JSON/Markdown/HTML reports.
    """
    suite_path = Path(suite).expanduser() if isinstance(suite, (str, Path)) else None
    spec = load_evidence_suite(suite_path) if suite_path is not None else suite
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    rows = [_summarize_watchtower_run(run, suite_path=suite_path) for run in spec.runs]
    frame = pd.DataFrame(rows)
    status_payload = {
        "suite_name": spec.suite_name,
        "generated_at_unix": time.time(),
        "n_runs": len(rows),
        "n_missing": int(sum(row["run_state"] == "missing" for row in rows)),
        "n_running": int(sum(row["run_state"] == "running" for row in rows)),
        "n_complete": int(sum(row["run_state"] == "complete" for row in rows)),
        "runs": rows,
    }

    summary_csv = out / "watchtower_summary.csv"
    summary_json = out / "watchtower_summary.json"
    summary_md = out / "watchtower_summary.md"
    summary_html = out / "watchtower_report.html"
    status_json = out / "watchtower_status.json"

    frame.to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(_json_safe(rows), indent=2), encoding="utf-8")
    status_json.write_text(json.dumps(_json_safe(status_payload), indent=2), encoding="utf-8")
    summary_md.write_text(_watchtower_markdown(spec.suite_name, rows), encoding="utf-8")
    summary_html.write_text(_watchtower_html(spec.suite_name, rows), encoding="utf-8")

    return {
        "suite_name": spec.suite_name,
        "n_runs": len(rows),
        "n_missing": status_payload["n_missing"],
        "n_running": status_payload["n_running"],
        "n_complete": status_payload["n_complete"],
        "artifacts": {
            "watchtower_summary_csv": str(summary_csv),
            "watchtower_summary_json": str(summary_json),
            "watchtower_summary_md": str(summary_md),
            "watchtower_report_html": str(summary_html),
            "watchtower_status": str(status_json),
        },
    }


def _summarize_watchtower_run(run: Any, *, suite_path: Path | None) -> dict[str, Any]:
    run_dir = _resolve_suite_path(run.run_dir, suite_path)
    config_path = _resolve_suite_path(run.config_path, suite_path)
    train_dir = run_dir / "train"
    metrics_path = train_dir / "metrics.json"
    checkpoint_dir = train_dir / "checkpoints"
    metrics = _read_metrics(metrics_path)
    config = _read_config(config_path)
    max_steps = _safe_int(config.get("training", {}).get("max_steps")) if isinstance(config.get("training"), dict) else None
    latest = metrics[-1] if metrics else {}
    latest_step = _safe_int(latest.get("step")) or (len(metrics) if metrics else None)
    alignment_series = _alignment_series(metrics)
    val_loss_series = _metric_series(metrics, "val_gene_loss")
    best_alignment_step, best_alignment_score = _best_series_value(alignment_series)
    burst_step, burst_delta = _largest_positive_delta(alignment_series)
    alignment_jitter = _largest_absolute_delta(alignment_series)
    val_loss_jitter = _largest_absolute_delta(val_loss_series)
    checkpoint_count, latest_checkpoint = _checkpoint_summary(checkpoint_dir)
    run_state = _run_state(run_dir, metrics_path, latest_step, max_steps, checkpoint_dir)
    progress = float(latest_step / max_steps) if latest_step is not None and max_steps else None

    return {
        "run_id": run.run_id,
        "tissue": run.tissue,
        "condition": run.condition,
        "suite_stage": run.suite_stage,
        "checkpoint_role": run.checkpoint_role,
        "lambda_align": run.lambda_align,
        "run_state": run_state,
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "metrics_present": bool(metrics),
        "latest_step": latest_step,
        "max_steps": max_steps,
        "progress": progress,
        "latest_lr": _safe_float(latest.get("lr")),
        "latest_loss": _safe_float(latest.get("loss")),
        "latest_val_gene_loss": _safe_float(latest.get("val_gene_loss")),
        "latest_i2g_top5": _safe_float(latest.get("val_image_to_gene_top5")),
        "latest_g2i_top5": _safe_float(latest.get("val_gene_to_image_top5")),
        "latest_alignment_score": _alignment_value(latest),
        "best_alignment_step": best_alignment_step,
        "best_alignment_score": best_alignment_score,
        "alignment_burst_step": burst_step,
        "alignment_burst_delta": burst_delta,
        "alignment_jitter_max": alignment_jitter,
        "val_gene_loss_jitter_max": val_loss_jitter,
        "prototype_usage_final": _safe_float(latest.get("prototype_usage_count")),
        "prototype_dead_codes_final": _safe_float(latest.get("prototype_dead_codes")),
        "prototype_entropy_final": _safe_float(latest.get("prototype_entropy_normalized")),
        "sinkhorn_row_residual_final": _safe_float(latest.get("sinkhorn_row_residual")),
        "sinkhorn_col_residual_final": _safe_float(latest.get("sinkhorn_col_residual")),
        "sinkhorn_nonfinite_count_final": _safe_float(latest.get("sinkhorn_nonfinite_count")),
        "checkpoint_count": checkpoint_count,
        "latest_checkpoint": latest_checkpoint,
    }


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def _read_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _alignment_value(row: dict[str, Any]) -> float | None:
    direct = _safe_float(row.get("val_alignment_score"))
    if direct is not None:
        return direct
    i2g = _safe_float(row.get("val_image_to_gene_top5"))
    g2i = _safe_float(row.get("val_gene_to_image_top5"))
    if i2g is None or g2i is None:
        return None
    return float((i2g + g2i) / 2.0)


def _alignment_series(metrics: list[dict[str, Any]]) -> list[tuple[int, float]]:
    series: list[tuple[int, float]] = []
    for index, row in enumerate(metrics, start=1):
        value = _alignment_value(row)
        if value is None:
            continue
        series.append((_safe_int(row.get("step")) or index, value))
    return series


def _metric_series(metrics: list[dict[str, Any]], key: str) -> list[tuple[int, float]]:
    series: list[tuple[int, float]] = []
    for index, row in enumerate(metrics, start=1):
        value = _safe_float(row.get(key))
        if value is not None:
            series.append((_safe_int(row.get("step")) or index, value))
    return series


def _best_series_value(series: list[tuple[int, float]]) -> tuple[int | None, float | None]:
    if not series:
        return None, None
    step, value = max(series, key=lambda item: (item[1], -item[0]))
    return int(step), float(value)


def _largest_positive_delta(series: list[tuple[int, float]]) -> tuple[int | None, float | None]:
    best_step: int | None = None
    best_delta: float | None = None
    for (_, previous), (step, current) in zip(series, series[1:], strict=False):
        delta = float(current - previous)
        if delta > 0 and (best_delta is None or delta > best_delta):
            best_step = int(step)
            best_delta = delta
    return best_step, best_delta


def _largest_absolute_delta(series: list[tuple[int, float]]) -> float | None:
    if len(series) < 2:
        return None
    return float(max(abs(current - previous) for (_, previous), (_, current) in zip(series, series[1:], strict=False)))


def _checkpoint_summary(checkpoint_dir: Path) -> tuple[int, str | None]:
    if not checkpoint_dir.exists():
        return 0, None
    checkpoints = sorted(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        return 0, None
    step_checkpoints = [(path, _checkpoint_step(path)) for path in checkpoints]
    numbered = [(path, step) for path, step in step_checkpoints if step is not None]
    latest = max(numbered, key=lambda item: item[1])[0] if numbered else max(checkpoints, key=lambda item: item.stat().st_mtime)
    return len(checkpoints), str(latest)


def _checkpoint_step(path: Path) -> int | None:
    match = re.search(r"step[_-](\d+)", path.stem)
    return int(match.group(1)) if match else None


def _run_state(
    run_dir: Path,
    metrics_path: Path,
    latest_step: int | None,
    max_steps: int | None,
    checkpoint_dir: Path,
) -> str:
    if not run_dir.exists() or not metrics_path.exists():
        return "missing"
    if max_steps is not None and latest_step is not None and latest_step >= max_steps:
        return "complete"
    if (checkpoint_dir / "last.pt").exists():
        return "complete"
    return "running"


def _resolve_suite_path(value: str, suite_path: Path | None) -> Path:
    path = Path(os.path.expandvars(value)).expanduser()
    if path.is_absolute():
        return path
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    if suite_path is not None:
        suite_candidate = (suite_path.parent / path).resolve()
        if suite_candidate.exists():
            return suite_candidate
    return cwd_candidate


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format(value: Any, *, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}g}" if math.isfinite(value) else "N/A"
    return str(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _watchtower_markdown(suite_name: str, rows: list[dict[str, Any]]) -> str:
    lines = [
        f"# Contour-Native Watchtower: {suite_name}",
        "",
        "Artifact-first monitor for E4 long runs. It reads existing telemetry and does not launch training.",
        "",
        "| Run | State | Step | Progress | Val gene loss | I->G@5 | G->I@5 | Alignment | Burst step | Burst delta | Dead codes |",
        "| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |",
    ]
    for row in rows:
        lines.append(
            "| {run_id} | {state} | {step} | {progress} | {val_loss} | {i2g} | {g2i} | {align} | {burst_step} | {burst_delta} | {dead} |".format(
                run_id=row["run_id"],
                state=row["run_state"],
                step=_format(row.get("latest_step"), digits=0),
                progress=_format(row.get("progress"), digits=3),
                val_loss=_format(row.get("latest_val_gene_loss")),
                i2g=_format(row.get("latest_i2g_top5")),
                g2i=_format(row.get("latest_g2i_top5")),
                align=_format(row.get("latest_alignment_score")),
                burst_step=_format(row.get("alignment_burst_step"), digits=0),
                burst_delta=_format(row.get("alignment_burst_delta")),
                dead=_format(row.get("prototype_dead_codes_final"), digits=0),
            )
        )
    return "\n".join(lines) + "\n"


def _watchtower_html(suite_name: str, rows: list[dict[str, Any]]) -> str:
    headers = [
        "Run",
        "State",
        "Step",
        "Progress",
        "Val gene loss",
        "I->G@5",
        "G->I@5",
        "Alignment",
        "Burst",
        "Dead codes",
        "Latest checkpoint",
    ]
    body_rows = []
    for row in rows:
        state = str(row.get("run_state"))
        body_rows.append(
            "<tr class='{state}'><td>{run}</td><td>{state}</td><td>{step}</td><td>{progress}</td>"
            "<td>{val}</td><td>{i2g}</td><td>{g2i}</td><td>{align}</td><td>{burst}</td><td>{dead}</td><td>{ckpt}</td></tr>".format(
                state=escape(state),
                run=escape(str(row.get("run_id"))),
                step=escape(_format(row.get("latest_step"), digits=0)),
                progress=escape(_format(row.get("progress"), digits=3)),
                val=escape(_format(row.get("latest_val_gene_loss"))),
                i2g=escape(_format(row.get("latest_i2g_top5"))),
                g2i=escape(_format(row.get("latest_g2i_top5"))),
                align=escape(_format(row.get("latest_alignment_score"))),
                burst=escape(f"{_format(row.get('alignment_burst_step'), digits=0)} / {_format(row.get('alignment_burst_delta'))}"),
                dead=escape(_format(row.get("prototype_dead_codes_final"), digits=0)),
                ckpt=escape(Path(str(row.get("latest_checkpoint"))).name if row.get("latest_checkpoint") else "N/A"),
            )
        )
    table = "\n".join(body_rows)
    header_html = "".join(f"<th>{escape(header)}</th>" for header in headers)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Contour-Native Watchtower: {escape(suite_name)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #172026; }}
    h1 {{ margin-bottom: 0; }}
    .subtitle {{ color: #50606c; margin-top: 6px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 20px; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #d7dee4; padding: 8px 10px; text-align: left; }}
    th {{ background: #eef3f7; }}
    tr.running td:first-child {{ border-left: 4px solid #3273dc; }}
    tr.complete td:first-child {{ border-left: 4px solid #2c8a4b; }}
    tr.missing td:first-child {{ border-left: 4px solid #c84c4c; }}
  </style>
</head>
<body>
  <h1>Contour-Native Watchtower</h1>
  <p class="subtitle">Suite: {escape(suite_name)}. Artifact-first E4 telemetry; no training is launched by this report.</p>
  <table>
    <thead><tr>{header_html}</tr></thead>
    <tbody>
      {table}
    </tbody>
  </table>
</body>
</html>
"""
