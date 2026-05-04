from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

LOCAL_REGISTRY_PATH_COLUMNS = (
    "output_dir",
    "reused_output_dir",
    "slide_manifest",
    "contour_source_manifest",
    "qc_report",
    "metadata_10x",
)

SLIDE_LOCAL_PATH_KEYS = (
    "output_dir",
    "slide_store",
)

SLIDE_ARTIFACT_KEYS = (
    "cell_to_contour",
    "structure_assignments",
    "contour_patches_manifest",
)


def inspect_registry(
    registry: str | Path,
    *,
    root: str | Path | None = None,
    output: str | Path | None = None,
    sample_images: int = 50,
) -> dict[str, Any]:
    """Inspect a XeniumSlide registry after data-root migration.

    The inspection is deliberately conservative: source data paths such as
    ``xenium_root`` may live outside the normalized output root, while generated
    artifacts such as ``slide_manifest`` and ``output_dir`` are expected to point
    at the current ``outputs/xenium_slides`` tree.
    """

    registry_path = Path(registry).expanduser()
    if not registry_path.exists():
        raise FileNotFoundError(f"registry does not exist: {registry_path}")
    expected_root = Path(root).expanduser() if root is not None else registry_path.parent
    frame = _read_registry(registry_path)
    case_reports = []
    for row_idx, row in frame.reset_index(drop=True).iterrows():
        report = _inspect_registry_row(
            row,
            row_idx=row_idx,
            registry_dir=registry_path.parent,
            expected_root=expected_root,
            sample_images=sample_images,
        )
        case_reports.append(report)

    selected = [item for item in case_reports if item["selected_for_build"]]
    summary = {
        "registry": str(registry_path),
        "expected_root": str(expected_root),
        "records": int(len(case_reports)),
        "selected_records": int(len(selected)),
        "cases_with_errors": int(sum(bool(item["errors"]) for item in selected)),
        "cases_with_warnings": int(sum(bool(item["warnings"]) for item in selected)),
        "missing_local_paths": int(sum(len(item["missing_local_paths"]) for item in selected)),
        "stale_local_paths": int(sum(len(item["stale_local_paths"]) for item in selected)),
        "slide_manifest_errors": int(sum(len(item["slide_manifest"].get("errors", [])) for item in selected)),
        "cases_with_contour_patches": int(sum(item["slide_manifest"].get("contour_patch_count", 0) > 0 for item in selected)),
        "cases_with_legacy_qc": int(sum(item["legacy_qc_codex_exists"] for item in selected)),
    }
    result = {"summary": summary, "cases": case_reports}
    if output is not None:
        output_path = Path(output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _read_registry(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            for key in ("records", "datasets", "cases"):
                value = payload.get(key)
                if isinstance(value, list):
                    return pd.DataFrame(value)
            if all(isinstance(value, dict) for value in payload.values()):
                return pd.DataFrame(list(payload.values()))
        raise ValueError(f"Unsupported registry JSON structure: {path}")
    raise ValueError("registry must be .csv, .json, or .parquet")


def _inspect_registry_row(
    row: pd.Series,
    *,
    row_idx: int,
    registry_dir: Path,
    expected_root: Path,
    sample_images: int,
) -> dict[str, Any]:
    case_name = _string_or_none(row.get("case_name")) or _string_or_none(row.get("case_slug")) or f"row_{row_idx}"
    selected = _truthy(row.get("selected_for_build", True)) and not _truthy(row.get("is_excluded_non_dataset", False))
    missing_paths: list[dict[str, str]] = []
    stale_paths: list[dict[str, str]] = []
    warnings: list[str] = []
    errors: list[str] = []

    resolved_paths: dict[str, str | None] = {}
    for column in LOCAL_REGISTRY_PATH_COLUMNS:
        value = _string_or_none(row.get(column))
        if value is None:
            resolved_paths[column] = None
            continue
        path = _resolve_path(value, registry_dir)
        resolved_paths[column] = str(path)
        if not path.exists():
            missing_paths.append({"column": column, "path": str(path)})
        if column != "reused_output_dir" and not _is_under(path, expected_root):
            stale_paths.append({"column": column, "path": str(path)})

    output_dir = _path_or_none(resolved_paths.get("output_dir"))
    legacy_qc = output_dir / "stgpt_qc_codex" if output_dir is not None else None
    slide_manifest_path = _path_or_none(resolved_paths.get("slide_manifest"))
    slide_report = (
        _inspect_slide_manifest(slide_manifest_path, expected_root=expected_root, sample_images=sample_images)
        if slide_manifest_path is not None and slide_manifest_path.exists()
        else {"errors": ["slide_manifest missing"], "warnings": [], "contour_patch_count": 0}
    )

    if missing_paths:
        errors.append(f"{len(missing_paths)} generated local path(s) are missing.")
    if stale_paths:
        errors.append(f"{len(stale_paths)} generated local path(s) are outside expected root.")
    errors.extend(slide_report.get("errors", []))
    warnings.extend(slide_report.get("warnings", []))

    return {
        "row_index": row_idx,
        "case_name": str(case_name),
        "selected_for_build": bool(selected),
        "output_dir": str(output_dir) if output_dir is not None else None,
        "legacy_qc_codex_exists": bool(legacy_qc is not None and legacy_qc.exists()),
        "resolved_paths": resolved_paths,
        "missing_local_paths": missing_paths,
        "stale_local_paths": stale_paths,
        "slide_manifest": slide_report,
        "warnings": warnings,
        "errors": errors,
    }


def _inspect_slide_manifest(path: Path, *, expected_root: Path, sample_images: int) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    payload = json.loads(path.read_text(encoding="utf-8"))
    local_paths: dict[str, str] = {}
    for key in SLIDE_LOCAL_PATH_KEYS:
        value = _string_or_none(payload.get(key))
        if value is None:
            continue
        local_paths[key] = value
        item = Path(value)
        if not item.exists():
            errors.append(f"slide_manifest.{key} missing: {value}")
        if not _is_under(item, expected_root):
            errors.append(f"slide_manifest.{key} outside expected root: {value}")

    artifacts = payload.get("artifacts", {})
    if isinstance(artifacts, dict):
        for key in SLIDE_ARTIFACT_KEYS:
            value = _string_or_none(artifacts.get(key))
            if value is None:
                continue
            local_paths[f"artifacts.{key}"] = value
            item = Path(value)
            if not item.exists():
                errors.append(f"slide_manifest.artifacts.{key} missing: {value}")
            if not _is_under(item, expected_root):
                errors.append(f"slide_manifest.artifacts.{key} outside expected root: {value}")

    patch_manifest = _path_or_none(local_paths.get("artifacts.contour_patches_manifest"))
    patch_report = _inspect_patch_manifest(patch_manifest, sample_images=sample_images) if patch_manifest is not None and patch_manifest.exists() else {}
    return {
        "path": str(path),
        "local_paths": local_paths,
        "contour_patch_count": int(patch_report.get("patch_count", 0)),
        "sampled_image_count": int(patch_report.get("sampled_image_count", 0)),
        "missing_sampled_images": patch_report.get("missing_sampled_images", []),
        "warnings": warnings + patch_report.get("warnings", []),
        "errors": errors + patch_report.get("errors", []),
    }


def _inspect_patch_manifest(path: Path, *, sample_images: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get("patches", payload.get("records", [])) if isinstance(payload, dict) else []
    missing_images = []
    sampled = rows[: max(0, int(sample_images))]
    for row_idx, row in enumerate(sampled):
        if not isinstance(row, dict):
            continue
        patch = row.get("patch") if isinstance(row.get("patch"), dict) else {}
        image_path = _string_or_none(row.get("image_path")) or _string_or_none(patch.get("path")) or _string_or_none(patch.get("image_path"))
        if image_path and not Path(image_path).exists():
            missing_images.append({"row_index": row_idx, "path": image_path})
    return {
        "patch_count": int(len(rows)),
        "sampled_image_count": int(len(sampled)),
        "missing_sampled_images": missing_images,
        "warnings": [],
        "errors": [f"{len(missing_images)} sampled contour patch image(s) are missing."] if missing_images else [],
    }


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base_dir / path


def _path_or_none(value: str | None) -> Path | None:
    if not value:
        return None
    return Path(value).expanduser()


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def _truthy(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass"}


def _string_or_none(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text if text and text.lower() not in {"nan", "none"} else None
