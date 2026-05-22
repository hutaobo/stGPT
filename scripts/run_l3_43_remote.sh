#!/usr/bin/env bash
set -Eeuo pipefail

ACTION="${1:-pack}"

PROJECT_ROOT="${STGPT_PROJECT_ROOT:-/data/taobo.hu/projects/stgpt_l3_20260504}"
REPO_DIR="${STGPT_REPO_DIR:-${PROJECT_ROOT}/repos/stGPT_e5_codex}"
SLIDE_ROOT="${STGPT_XENIUM_SLIDES:-${PROJECT_ROOT}/data/xenium_slides}"
OUTPUT_ROOT="${STGPT_OUTPUT_ROOT:-${PROJECT_ROOT}/runs}"
L3_VERSION="${STGPT_L3_VERSION:-l3_20260507_43case}"
MANIFEST="${STGPT_L3_MANIFEST:-${SLIDE_ROOT}/training_manifest_l3.csv}"
LOG_ROOT="${PROJECT_ROOT}/logs/${L3_VERSION}"
PACK_LOG_ROOT="${LOG_ROOT}/pack"
FREEZE_DIR="${SLIDE_ROOT}/_frozen/${L3_VERSION}"

export STGPT_XENIUM_SLIDES="${SLIDE_ROOT}"
export STGPT_OUTPUT_ROOT="${OUTPUT_ROOT}"
export STGPT_REPO_DIR="${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

if [[ -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/.venv/bin/activate"
fi

mkdir -p "${LOG_ROOT}" "${PACK_LOG_ROOT}" "${OUTPUT_ROOT}"

trap 'status=$?; echo "$(date --iso-8601=seconds) ERROR action=${ACTION} line=${LINENO} status=${status}" >&2' ERR

log_stage() {
  echo "$(date --iso-8601=seconds) STAGE action=${ACTION} ${*}" >&2
}

run_stgpt() {
  if command -v stgpt >/dev/null 2>&1; then
    stgpt "$@"
  else
    python -c 'from stgpt.cli import app; app()' "$@"
  fi
}

freeze_data_version() {
  mkdir -p "${FREEZE_DIR}"
  for name in \
    training_manifest_l3.csv \
    l3_upgrade_summary.json \
    l3_patch_extraction_20260507_results.json \
    dataset_inventory.json \
    dataset_registry.csv \
    dataset_registry.parquet \
    build_summary.json \
    metadata_resolution_report.json \
    failed_cases.csv \
    failed_cases.json; do
    if [[ -f "${SLIDE_ROOT}/${name}" ]]; then
      cp -f "${SLIDE_ROOT}/${name}" "${FREEZE_DIR}/${name}"
    fi
  done
  python - <<'PY' "${FREEZE_DIR}" "${MANIFEST}" "${L3_VERSION}"
import csv
import hashlib
import json
import sys
from pathlib import Path

freeze_dir = Path(sys.argv[1])
manifest = Path(sys.argv[2])
version = sys.argv[3]
rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8"))) if manifest.exists() else []
patches = sum(int(row.get("contour_patches") or 0) for row in rows)
files = {}
for path in sorted(freeze_dir.iterdir()):
    if path.is_file() and path.name != "data_version.json" and not path.name.startswith("l3_43_pack_"):
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
payload = {
    "data_version": version,
    "manifest": str(manifest),
    "n_cases": len(rows),
    "n_contour_patches": patches,
    "sha256": files,
}
(freeze_dir / "data_version.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
}

rewrite_patch_manifests() {
  python - <<'PY' "${SLIDE_ROOT}"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
keys = ("image_path", "object_image_path", "context_image_path", "mask_path")
changed = 0
for manifest in sorted(root.glob("*/contour_patches_manifest.json")):
    case_dir = manifest.parent
    case = case_dir.name
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"warn: could not read {manifest}: {exc}", file=sys.stderr)
        continue
    if not isinstance(payload, list):
        continue
    manifest_changed = False
    for item in payload:
        if not isinstance(item, dict):
            continue
        for key in keys:
            raw = item.get(key)
            if not raw:
                continue
            current = Path(str(raw))
            if current.exists():
                continue
            normalized = str(raw).replace("\\", "/")
            suffix = None
            marker = f"/{case}/"
            if marker in normalized:
                suffix = normalized.split(marker, 1)[1]
            elif normalized.startswith(f"{case}/"):
                suffix = normalized[len(case) + 1 :]
            if suffix is None:
                continue
            candidate = case_dir / suffix
            item[key] = str(candidate)
            manifest_changed = True
    if manifest_changed:
        manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        changed += 1
print(json.dumps({"rewritten_patch_manifests": changed}, indent=2))
PY
}

pack_all_cases() {
  freeze_data_version
  rewrite_patch_manifests
  local report="${FREEZE_DIR}/l3_43_pack_report.csv"
  echo "case_leaf,status,patch_rows,store,manifest,log" > "${report}"
  python - <<'PY' "${MANIFEST}" > "${PACK_LOG_ROOT}/case_leaves.txt"
import csv
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
for row in csv.DictReader(manifest.open(newline="", encoding="utf-8")):
    case = (row.get("case_leaf") or "").strip()
    if case:
        print(case)
PY
  while IFS= read -r case_leaf; do
    [[ -n "${case_leaf}" ]] || continue
    case_dir="${SLIDE_ROOT}/${case_leaf}"
    patch_manifest="${case_dir}/contour_patches_manifest.json"
    store="${case_dir}/contour_image_store.zarr"
    contour_manifest="${case_dir}/contour_image_manifest.parquet"
    log="${PACK_LOG_ROOT}/${case_leaf}.json"
    if [[ ! -f "${patch_manifest}" ]]; then
      echo "${case_leaf},missing_patch_manifest,0,${store},${contour_manifest},${log}" >> "${report}"
      continue
    fi
    if [[ -d "${store}" && -f "${contour_manifest}" && "${STGPT_L3_REPACK:-0}" != "1" ]]; then
      rows=$(python - <<'PY' "${contour_manifest}"
import pandas as pd
import sys
print(len(pd.read_parquet(sys.argv[1])))
PY
)
      echo "${case_leaf},exists,${rows},${store},${contour_manifest},${log}" >> "${report}"
      continue
    fi
    run_stgpt pack-contour-patches \
      --patch-manifest "${patch_manifest}" \
      --store "${store}" \
      --manifest "${contour_manifest}" \
      --slide-id "${case_leaf}" \
      --image-size "${STGPT_L3_PACK_IMAGE_SIZE:-64}" \
      --max-neighbors "${STGPT_L3_MAX_NEIGHBORS:-16}" \
      --chunk-size "${STGPT_L3_CHUNK_SIZE:-1024}" | tee "${log}"
    rows=$(python - <<'PY' "${contour_manifest}"
import pandas as pd
import sys
print(len(pd.read_parquet(sys.argv[1])))
PY
)
    echo "${case_leaf},packed,${rows},${store},${contour_manifest},${log}" >> "${report}"
  done < "${PACK_LOG_ROOT}/case_leaves.txt"
  audit_pack
}

audit_pack() {
  mkdir -p "${FREEZE_DIR}"
  python - <<'PY' "${MANIFEST}" "${SLIDE_ROOT}" "${FREEZE_DIR}"
import csv
import json
import sys
from pathlib import Path

import pandas as pd

manifest = Path(sys.argv[1])
root = Path(sys.argv[2])
out = Path(sys.argv[3])
rows = []
for row in csv.DictReader(manifest.open(newline="", encoding="utf-8")):
    case = (row.get("case_leaf") or "").strip()
    if not case:
        continue
    case_dir = root / case
    store = case_dir / "contour_image_store.zarr"
    contour_manifest = case_dir / "contour_image_manifest.parquet"
    failure_json = case_dir / "contour_patch_failures.json"
    n_rows = None
    if contour_manifest.exists():
        try:
            n_rows = int(len(pd.read_parquet(contour_manifest)))
        except Exception:
            n_rows = None
    rows.append(
        {
            "case_leaf": case,
            "store_exists": store.exists(),
            "manifest_exists": contour_manifest.exists(),
            "manifest_rows": n_rows,
            "expected_patches": int(row.get("contour_patches") or 0),
            "qc_warning": failure_json.exists(),
            "failure_json": str(failure_json) if failure_json.exists() else "",
        }
    )
frame = pd.DataFrame(rows)
summary = {
    "n_cases": int(len(frame)),
    "stores": int(frame["store_exists"].sum()) if not frame.empty else 0,
    "manifests": int(frame["manifest_exists"].sum()) if not frame.empty else 0,
    "missing_cases": frame.loc[~(frame["store_exists"] & frame["manifest_exists"]), "case_leaf"].tolist()
    if not frame.empty
    else [],
    "qc_warning_cases": frame.loc[frame["qc_warning"], "case_leaf"].tolist() if not frame.empty else [],
    "manifest_rows_total": int(frame["manifest_rows"].fillna(0).sum()) if not frame.empty else 0,
}
frame.to_csv(out / "l3_43_pack_audit.csv", index=False)
(out / "l3_43_pack_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
}

checkpoint_for_role() {
  local run_dir="$1"
  local role="$2"
  local train_dir="${run_dir}/train/checkpoints"
  if [[ "${role}" == "best_alignment" && -f "${train_dir}/best_alignment.pt" ]]; then
    echo "${train_dir}/best_alignment.pt"
  elif [[ -f "${train_dir}/best.pt" ]]; then
    echo "${train_dir}/best.pt"
  else
    echo "${train_dir}/last.pt"
  fi
}

run_pipeline() {
  local config="$1"
  local run_dir="$2"
  local role="$3"
  local gpu="${4:-0}"
  export CUDA_VISIBLE_DEVICES="${gpu}"
  mkdir -p "${run_dir}"
  log_stage "run=$(basename "${run_dir}") step=train gpu=${gpu}"
  run_stgpt train --config "${REPO_DIR}/${config}" | tee "${run_dir}/train_stdout.json"
  local splits="${run_dir}/train/splits.csv"
  if [[ "${STGPT_L3_RUN_QC:-0}" == "1" ]]; then
    log_stage "run=$(basename "${run_dir}") step=validate-data gpu=${gpu}"
    run_stgpt validate-data --config "${REPO_DIR}/${config}" --output "${run_dir}/qc" | tee "${run_dir}/validate_data.json"
  elif [[ -f "${splits}" ]]; then
    mkdir -p "${run_dir}/qc"
    cp -f "${splits}" "${run_dir}/qc/splits.csv"
    printf '{"status":"skipped","reason":"training wrote train/splits.csv","splits":"%s"}\n' "${splits}" > "${run_dir}/validate_data.json"
  else
    log_stage "run=$(basename "${run_dir}") step=validate-data-fallback gpu=${gpu}"
    run_stgpt validate-data --config "${REPO_DIR}/${config}" --output "${run_dir}/qc" | tee "${run_dir}/validate_data.json"
    splits="${run_dir}/qc/splits.csv"
  fi
  local ckpt
  ckpt="$(checkpoint_for_role "${run_dir}" "${role}")"
  mkdir -p "${run_dir}/contract"
  log_stage "run=$(basename "${run_dir}") step=check-contract checkpoint=${ckpt}"
  run_stgpt check-contract --checkpoint "${ckpt}" --config "${REPO_DIR}/${config}" --run-dir "${run_dir}" --output "${run_dir}/contract/check_contract.json" | tee "${run_dir}/contract/check_contract_stdout.json"
  log_stage "run=$(basename "${run_dir}") step=evaluate checkpoint=${ckpt}"
  run_stgpt evaluate --checkpoint "${ckpt}" --config "${REPO_DIR}/${config}" --splits "${splits}" --output "${run_dir}/evaluation" --batch-size 64 --device cuda | tee "${run_dir}/evaluation_stdout.json"
  log_stage "run=$(basename "${run_dir}") step=export-spatho checkpoint=${ckpt}"
  run_stgpt export-spatho --checkpoint "${ckpt}" --config "${REPO_DIR}/${config}" --output "${run_dir}/spatho_export" --batch-size 64 --device cuda | tee "${run_dir}/spatho_export_stdout.json"
  log_stage "run=$(basename "${run_dir}") step=package-model checkpoint=${ckpt}"
  run_stgpt package-model --checkpoint "${ckpt}" --eval "${run_dir}/evaluation/evaluation_metrics.json" --output "${run_dir}/checkpoint_card" --model-name "$(basename "${run_dir}")" | tee "${run_dir}/package_model_stdout.json"
  log_stage "run=$(basename "${run_dir}") step=complete"
}

start_background_pipeline() {
  local label="$1"
  local config="$2"
  local run_dir="$3"
  local role="$4"
  local gpu="$5"
  mkdir -p "${LOG_ROOT}/jobs"
  if pgrep -f "run_l3_43_remote.sh pipeline ${config} ${run_dir}" >/dev/null 2>&1; then
    echo "${label}: already running for ${run_dir}" >&2
    return 0
  fi
  if [[ -d "${run_dir}" ]] && [[ -n "$(find "${run_dir}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "${label}: refusing duplicate launch; non-empty run directory exists: ${run_dir}" >&2
    return 1
  fi
  nohup bash -lc "cd '${REPO_DIR}' && bash '${REPO_DIR}/scripts/run_l3_43_remote.sh' pipeline '${config}' '${run_dir}' '${role}' '${gpu}'" \
    > "${LOG_ROOT}/jobs/${label}.log" 2>&1 &
  echo "$!" > "${LOG_ROOT}/jobs/${label}.pid"
  echo "${label}: pid=$(cat "${LOG_ROOT}/jobs/${label}.pid") gpu=${gpu} log=${LOG_ROOT}/jobs/${label}.log"
}

sweep_runs_from_suite() {
  python - <<'PY' "${REPO_DIR}/configs/evidence/l3_43.yaml"
import os
import sys

import yaml

suite = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
for run in suite.get("runs", []):
    config = run.get("config_path")
    run_dir = os.path.expandvars(str(run.get("run_dir", "")))
    role = run.get("checkpoint_role", "best_loss")
    if config and run_dir and "$" not in run_dir:
        print(f"{config}|{run_dir}|{role}")
PY
}

# Distribute a list of runs across a GPU pool (default 2-7, leaving 0-1 free), one
# background pipeline per GPU, refilling a GPU when its run finishes. Runs come from
# explicit config paths ("$@") or, when none are given, from the l3_43 evidence suite.
# RAM is the binding constraint for concurrency, not the GPU: training reuses the
# train-split (validate-data's ~371GB RSS is skipped unless STGPT_L3_RUN_QC=1). Start
# conservative and raise STGPT_L3_MAX_CONCURRENT while watching RSS.
start_sweep() {
  mkdir -p "${LOG_ROOT}/jobs"
  local gpus
  read -r -a gpus <<< "${STGPT_L3_GPUS:-2 3 4 5 6 7}"
  local max_concurrent="${STGPT_L3_MAX_CONCURRENT:-${#gpus[@]}}"
  local runs=()
  if [[ "$#" -gt 0 ]]; then
    local config stem
    for config in "$@"; do
      stem="$(basename "${config}")"; stem="${stem%.yaml}"
      runs+=("${config}|${OUTPUT_ROOT}/pilot_runs/${L3_VERSION}/${stem}|${STGPT_L3_ROLE:-best_alignment}")
    done
  else
    local line
    while IFS= read -r line; do
      [[ -n "${line}" ]] && runs+=("${line}")
    done < <(sweep_runs_from_suite)
  fi
  local total="${#runs[@]}"
  if (( total == 0 )); then
    echo "start-sweep: no runs to launch" >&2
    return 1
  fi
  log_stage "start-sweep gpus=[${gpus[*]}] max_concurrent=${max_concurrent} runs=${total}"
  declare -A gpu_pid=()
  local idx=0
  while (( idx < total )) || (( ${#gpu_pid[@]} > 0 )); do
    if (( ${#gpu_pid[@]} > 0 )); then
      local g
      for g in "${!gpu_pid[@]}"; do
        if ! kill -0 "${gpu_pid[$g]}" 2>/dev/null; then
          unset 'gpu_pid['"$g"']'
        fi
      done
    fi
    local g
    for g in "${gpus[@]}"; do
      (( idx < total )) || break
      (( ${#gpu_pid[@]} < max_concurrent )) || break
      if [[ -z "${gpu_pid[$g]:-}" ]]; then
        local entry config run_dir role label pid
        entry="${runs[$idx]}"
        config="${entry%%|*}"; entry="${entry#*|}"
        run_dir="${entry%%|*}"; role="${entry##*|}"
        label="$(basename "${run_dir}")_gpu${g}"
        if start_background_pipeline "${label}" "${config}" "${run_dir}" "${role}" "${g}"; then
          pid="$(cat "${LOG_ROOT}/jobs/${label}.pid" 2>/dev/null || true)"
          if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            gpu_pid[$g]="${pid}"
          fi
        fi
        idx=$((idx + 1))
      fi
    done
    sleep 15
  done
  log_stage "start-sweep complete runs=${total}"
}

case "${ACTION}" in
  freeze)
    freeze_data_version
    ;;
  rewrite-paths)
    rewrite_patch_manifests
    ;;
  pack)
    pack_all_cases
    ;;
  audit-pack)
    audit_pack
    ;;
  smoke)
    run_pipeline \
      "configs/pilots/l3_43/smoke_5case_full_m6_lambda_0_01_500.yaml" \
      "${OUTPUT_ROOT}/pilot_runs/l3_20260507_43case/smoke_5case_full_m6_lambda_0_01_500" \
      "best_alignment" \
      "${STGPT_L3_GPU:-0}"
    run_stgpt evidence-summary --suite "${REPO_DIR}/configs/evidence/l3_43.yaml" --output "${OUTPUT_ROOT}/evidence/l3_20260507_43case" || true
    ;;
  start-smoke)
    start_background_pipeline \
      "smoke_5case_full_m6_lambda_0_01_500" \
      "configs/pilots/l3_43/smoke_5case_full_m6_lambda_0_01_500.yaml" \
      "${OUTPUT_ROOT}/pilot_runs/l3_20260507_43case/smoke_5case_full_m6_lambda_0_01_500" \
      "best_alignment" \
      "${STGPT_L3_SMOKE_GPU:-3}"
    ;;
  start-foundation)
    start_background_pipeline \
      "full_m6_contour_store_lambda_0_01_20k" \
      "configs/pilots/l3_43/full_m6_contour_store_lambda_0_01_20k.yaml" \
      "${OUTPUT_ROOT}/pilot_runs/l3_20260507_43case/full_m6_contour_store_lambda_0_01_20k" \
      "best_alignment" \
      "${STGPT_L3_FULL_GPU:-3}"
    start_background_pipeline \
      "gene_spatial_contour_unit_20k" \
      "configs/pilots/l3_43/gene_spatial_contour_unit_20k.yaml" \
      "${OUTPUT_ROOT}/pilot_runs/l3_20260507_43case/gene_spatial_contour_unit_20k" \
      "best_loss" \
      "${STGPT_L3_BASELINE_GPU:-4}"
    ;;
  start-sweep)
    shift || true
    start_sweep "$@"
    ;;
  pipeline)
    run_pipeline "$2" "$3" "$4" "$5"
    ;;
  evidence)
    run_stgpt evidence-summary --suite "${REPO_DIR}/configs/evidence/l3_43.yaml" --output "${OUTPUT_ROOT}/evidence/l3_20260507_43case"
    ;;
  *)
    echo "Usage: $0 {freeze|rewrite-paths|pack|audit-pack|smoke|start-smoke|start-foundation|start-sweep|pipeline|evidence}" >&2
    exit 2
    ;;
esac
