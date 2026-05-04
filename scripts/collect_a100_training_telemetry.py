"""Collect paper-facing telemetry for an A100 stGPT L3 training run.

This script is intentionally dependency-light: it uses only the Python standard
library plus command-line tools already expected on the remote training host.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run_cmd(cmd: list[str], timeout: int = 120) -> dict[str, object]:
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"cmd": cmd, "error": repr(exc)}


def file_sha256(path: Path, max_bytes: int | None = None) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        if max_bytes is None:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        else:
            digest.update(handle.read(max_bytes))
    return digest.hexdigest()


def iso_mtime(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def parse_queue_events(log_path: Path) -> list[dict[str, str]]:
    pattern = re.compile(
        r"^\[(?P<ts>[^\]]+)\] gpu=(?P<gpu>[^ ]+) "
        r"(?P<event>start|done|failed|no more configs)"
        r"(?: (?P<case>.*?))?(?: code=(?P<code>\d+))?$"
    )
    events: list[dict[str, str]] = []
    if not log_path.exists():
        return events
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line.strip())
        if match:
            row = {key: (value or "") for key, value in match.groupdict().items()}
            events.append(row)
    return events


def build_case_runs(events: list[dict[str, str]]) -> list[dict[str, str]]:
    starts_by_case: dict[str, list[dict[str, str]]] = {}
    case_runs: list[dict[str, str]] = []
    for event in events:
        case = event.get("case", "").strip()
        if not case or case == "configs":
            continue
        if event["event"] == "start":
            starts_by_case.setdefault(case, []).append(event)
        elif event["event"] in {"done", "failed"}:
            start = starts_by_case.get(case, []).pop() if starts_by_case.get(case) else {}
            case_runs.append(
                {
                    "case": case,
                    "gpu": event.get("gpu", ""),
                    "start_ts": start.get("ts", ""),
                    "end_ts": event.get("ts", ""),
                    "status": event.get("event", ""),
                    "code": event.get("code", ""),
                }
            )
    latest: dict[str, dict[str, str]] = {}
    for row in case_runs:
        latest[row["case"]] = row
    return sorted(latest.values(), key=lambda row: (row.get("end_ts", ""), row.get("case", "")))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_metric_and_checkpoint_rows(runs_dir: Path) -> tuple[list[dict], list[dict]]:
    metric_rows: list[dict] = []
    checkpoint_rows: list[dict] = []
    for case_dir in sorted(runs_dir.iterdir() if runs_dir.exists() else []):
        if not case_dir.is_dir():
            continue
        case = case_dir.name
        train_dir = case_dir / "train"
        metrics_path = train_dir / "metrics.json"
        metrics: list[dict] = []
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except Exception:
                metrics = []
        summary = {
            "case": case,
            "metrics_path": str(metrics_path),
            "metrics_exists": metrics_path.exists(),
            "metrics_mtime_utc": iso_mtime(metrics_path),
            "n_metric_rows": len(metrics),
            "last_pt_exists": (train_dir / "checkpoints" / "last.pt").exists(),
            "best_pt_exists": (train_dir / "checkpoints" / "best.pt").exists(),
        }
        if metrics:
            last = metrics[-1]
            for key, value in last.items():
                if isinstance(value, int | float):
                    summary[f"last_{key}"] = value
            for key in [
                "loss",
                "gene_loss",
                "neighbor_loss",
                "image_gene_loss",
                "prototype_loss",
                "prototype_entropy_normalized",
                "prototype_dead_codes",
                "val_loss",
                "val_gene_loss",
                "val_neighbor_loss",
                "val_image_gene_loss",
            ]:
                values = [row.get(key) for row in metrics if isinstance(row.get(key), int | float)]
                if values:
                    summary[f"min_{key}"] = min(values)
                    summary[f"max_{key}"] = max(values)
        metric_rows.append(summary)
        for checkpoint in sorted((train_dir / "checkpoints").glob("*.pt")):
            checkpoint_rows.append(
                {
                    "case": case,
                    "checkpoint": checkpoint.name,
                    "path": str(checkpoint),
                    "size_bytes": checkpoint.stat().st_size,
                    "mtime_utc": iso_mtime(checkpoint),
                    "sha256_head_16mb": file_sha256(checkpoint, max_bytes=16 * 1024 * 1024),
                }
            )
    return metric_rows, checkpoint_rows


def scan_logs(log_dir: Path) -> tuple[list[dict], list[dict]]:
    error_pattern = re.compile(r"Traceback|RuntimeError|CUDA out of memory|No module named|Exception|ERROR", re.I)
    log_rows: list[dict] = []
    error_rows: list[dict] = []
    for log in sorted(log_dir.glob("train_*.log")):
        text = log.read_text(encoding="utf-8", errors="replace")
        log_rows.append(
            {
                "log": str(log),
                "size_bytes": log.stat().st_size,
                "mtime_utc": iso_mtime(log),
                "line_count": text.count("\n") + (1 if text else 0),
                "sha256": file_sha256(log),
            }
        )
        for line_number, line in enumerate(text.splitlines(), start=1):
            if error_pattern.search(line):
                error_rows.append({"log": str(log), "line": line_number, "text": line[:1000]})
    return log_rows, error_rows


def main() -> None:
    project = Path(os.environ.get("STGPT_A100_PROJECT", "/data/taobo.hu/projects/stgpt_l3_20260504"))
    out = Path(os.environ.get("STGPT_TELEMETRY_DIR", project / "evidence" / "training_telemetry"))
    out.mkdir(parents=True, exist_ok=True)
    runs = project / "runs" / "l3_cases"
    logs = project / "logs"
    configs = project / "configs" / "l3_cases"
    slides = project / "data" / "xenium_slides"

    gpu_queries = {
        "gpu_static.csv": [
            "nvidia-smi",
            "--query-gpu=timestamp,index,name,uuid,pci.bus_id,driver_version,serial,memory.total",
            "--format=csv",
        ],
        "gpu_runtime_final.csv": [
            "nvidia-smi",
            "--query-gpu=timestamp,index,uuid,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,clocks.sm,clocks.mem",
            "--format=csv",
        ],
        "gpu_compute_apps_final.csv": [
            "nvidia-smi",
            "--query-compute-apps=timestamp,gpu_uuid,pid,process_name,used_memory",
            "--format=csv",
        ],
    }
    for filename, cmd in gpu_queries.items():
        result = run_cmd(cmd)
        text = str(result.get("stdout", ""))
        if result.get("stderr"):
            text += "\n# STDERR\n" + str(result["stderr"])
        (out / filename).write_text(text, encoding="utf-8")

    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": str(project),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "uploaded_complete_count": len(list(slides.glob("*/.upload_complete"))),
        "config_count": len(list(configs.glob("*.yaml"))),
        "done_last_count": len(list(runs.glob("*/train/checkpoints/last.pt"))),
        "done_metrics_count": len(list(runs.glob("*/train/metrics.json"))),
        "failed_count": len(list(runs.glob("*/.failed"))),
        "source_roots": {"stGPT": str(project / "repos" / "stGPT"), "pyXenium": str(project / "repos" / "pyXenium")},
        "local_git_commits_at_upload": {
            "stGPT": "5eece349ca921d69012a1ee19c220607173a97a9",
            "pyXenium": "830d25ae927d09b1d7124a3e397fa2a5bbce8102",
            "note": "Both repositories had uncommitted changes at upload; source tarballs were uploaded to the project packages directory.",
        },
    }
    (out / "environment_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    for cmd, filename in [
        ([sys.executable, "-m", "pip", "freeze"], "pip_freeze.txt"),
        ([sys.executable, "-m", "pip", "list", "--format=json"], "pip_list.json"),
        (
            [
                sys.executable,
                "-c",
                "import json, torch, zarr, anndata, stgpt, pyXenium; print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available(), 'zarr': zarr.__version__, 'anndata': anndata.__version__, 'stgpt_file': stgpt.__file__, 'pyXenium_file': pyXenium.__file__}, indent=2))",
            ],
            "python_package_probe.json",
        ),
    ]:
        result = run_cmd(cmd)
        text = str(result.get("stdout", ""))
        if result.get("stderr"):
            text += "\n# STDERR\n" + str(result["stderr"])
        (out / filename).write_text(text, encoding="utf-8")

    events = parse_queue_events(logs / "training_queue.log")
    write_csv(out / "training_queue_events.csv", events, ["ts", "gpu", "event", "case", "code"])
    case_runs = build_case_runs(events)
    write_csv(out / "case_training_runs.csv", case_runs, ["case", "gpu", "start_ts", "end_ts", "status", "code"])

    metric_rows, checkpoint_rows = collect_metric_and_checkpoint_rows(runs)
    write_csv(out / "case_metric_summary.csv", metric_rows)
    (out / "case_metric_summary.json").write_text(json.dumps(metric_rows, indent=2) + "\n", encoding="utf-8")
    write_csv(
        out / "checkpoint_manifest.csv",
        checkpoint_rows,
        ["case", "checkpoint", "path", "size_bytes", "mtime_utc", "sha256_head_16mb"],
    )

    log_rows, error_rows = scan_logs(logs)
    write_csv(out / "training_log_manifest.csv", log_rows, ["log", "size_bytes", "mtime_utc", "line_count", "sha256"])
    write_csv(out / "training_error_scan.csv", error_rows, ["log", "line", "text"])

    for source_name in [
        "training_manifest_l3.csv",
        "training_manifest_l3.json",
        "dataset_registry.csv",
        "dataset_registry.json",
        "readiness_summary.json",
        "l3_upgrade_summary.json",
    ]:
        source = slides / source_name
        if source.exists():
            shutil.copy2(source, out / source_name)

    config_rows = [
        {"case": config.stem, "config": str(config), "mtime_utc": iso_mtime(config), "sha256": file_sha256(config)}
        for config in sorted(configs.glob("*.yaml"))
    ]
    write_csv(out / "training_config_manifest.csv", config_rows, ["case", "config", "mtime_utc", "sha256"])

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "uploaded_complete_count": provenance["uploaded_complete_count"],
        "config_count": provenance["config_count"],
        "done_last_count": len(list(runs.glob("*/train/checkpoints/last.pt"))),
        "done_metrics_count": len(list(runs.glob("*/train/metrics.json"))),
        "failed_count": len(list(runs.glob("*/.failed"))),
        "case_metric_rows": len(metric_rows),
        "case_run_rows": len(case_runs),
        "checkpoint_rows": len(checkpoint_rows),
        "training_log_rows": len(log_rows),
        "error_rows": len(error_rows),
        "output_dir": str(out),
        "figure_ready_tables": [
            "case_training_runs.csv",
            "case_metric_summary.csv",
            "checkpoint_manifest.csv",
            "training_queue_events.csv",
            "gpu_static.csv",
            "gpu_runtime_final.csv",
            "training_error_scan.csv",
        ],
        "recommended_figures": [
            "Per-case training duration by GPU and tissue family",
            "Final/min validation loss, gene loss, neighbor loss, and image-gene loss across 39 cases",
            "Loss trajectories per case from metrics.json",
            "Checkpoint completion matrix across all L3 cases",
            "GPU utilization/memory/power timeline for future longer runs",
        ],
        "limitation": "This collector was installed after the short pilot completed, so it captures final GPU state and log-derived timing. Future runs should start a GPU sampler before training for dense time-series utilization figures.",
    }
    (out / "training_telemetry_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
