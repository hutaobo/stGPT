#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="${STGPT_PROJECT_ROOT:-/data/taobo.hu/projects/stgpt_l3_20260504}"
REPO="${STGPT_REPO_DIR:-${PROJECT}/repos/stGPT_e5_codex}"
ROOT="${STGPT_L3_RUN_ROOT:-${PROJECT}/runs/pilot_runs/l3_20260507_43case}"
LOGDIR="${STGPT_L3_LOG_DIR:-${PROJECT}/logs/l3_20260507_43case/jobs}"
PY="${PROJECT}/.venv/bin/python"

FULL="full_m6_contour_store_lambda_0_01_20k"
BASE="gene_spatial_contour_unit_20k"
FULL_CONFIG="configs/pilots/l3_43/full_m6_contour_store_lambda_0_01_20k.yaml"
BASE_CONFIG="configs/pilots/l3_43/gene_spatial_contour_unit_20k.yaml"

VALIDATE_FIRST="${REPO}/scripts/run_l3_43_pipeline_validate_first.sh"
REUSE_QC="${REPO}/scripts/run_l3_43_pipeline_reuse_qc.sh"
TAIL="${REPO}/scripts/run_l3_43_pipeline_tail.sh"
GUARD_LOG="${LOGDIR}/l3_43_foundation_guard.log"
GPU_POOL=(${STGPT_L3_GPU_POOL:-4 5 6 7})

export STGPT_XENIUM_SLIDES="${STGPT_XENIUM_SLIDES:-${PROJECT}/data/xenium_slides}"
export STGPT_OUTPUT_ROOT="${STGPT_OUTPUT_ROOT:-${PROJECT}/runs}"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"

if [[ -f "${PROJECT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${PROJECT}/.venv/bin/activate"
fi

mkdir -p "${LOGDIR}"
chmod +x "${VALIDATE_FIRST}" "${REUSE_QC}" "${TAIL}" 2>/dev/null || true

log() {
  echo "$(date --iso-8601=seconds) $*" | tee -a "${GUARD_LOG}"
}

run_dir() {
  echo "${ROOT}/$1"
}

job_log() {
  echo "${LOGDIR}/$1.log"
}

active_for_run() {
  local name="$1"
  ps -u taobo.hu -o pid=,cmd= \
    | grep -E "run_l3_43_pipeline_(validate_first|reuse_qc|tail).*${name}|stgpt (train|evaluate|export-spatho|package-model).*${name}" \
    | grep -v grep >/dev/null 2>&1
}

gpu_used_by_stgpt() {
  local gpu="$1"
  ps -u taobo.hu -o cmd= \
    | grep -E "run_l3_43_pipeline_(validate_first|reuse_qc|tail).*[[:space:]]${gpu}([[:space:]]|$)" \
    | grep -v grep >/dev/null 2>&1
}

choose_gpu() {
  local preferred="${1:-}"
  local gpu
  if [[ -n "${preferred}" ]] && ! gpu_used_by_stgpt "${preferred}"; then
    echo "${preferred}"
    return 0
  fi
  for gpu in "${GPU_POOL[@]}"; do
    if ! gpu_used_by_stgpt "${gpu}"; then
      echo "${gpu}"
      return 0
    fi
  done
  echo "${preferred:-${GPU_POOL[0]}}"
}

validate_passed() {
  local dir="$1"
  [[ -s "${dir}/validate_data.json" ]] || return 1
  "${PY}" - "${dir}/validate_data.json" <<'PY' >/dev/null 2>&1
import json, sys
with open(sys.argv[1]) as handle:
    payload = json.load(handle)
raise SystemExit(0 if payload.get("status") == "pass" else 1)
PY
}

checkpoint_for_role() {
  local dir="$1"
  local role="$2"
  if [[ "${role}" == "best_alignment" && -s "${dir}/train/checkpoints/best_alignment.pt" ]]; then
    echo "${dir}/train/checkpoints/best_alignment.pt"
  elif [[ "${role}" == "best_loss" && -s "${dir}/train/checkpoints/best.pt" ]]; then
    echo "${dir}/train/checkpoints/best.pt"
  elif [[ -s "${dir}/train/checkpoints/best.pt" ]]; then
    echo "${dir}/train/checkpoints/best.pt"
  elif [[ -s "${dir}/train/checkpoints/last.pt" ]]; then
    echo "${dir}/train/checkpoints/last.pt"
  else
    return 1
  fi
}

complete_run() {
  local dir="$1"
  [[ -s "${dir}/validate_data.json" ]] || return 1
  [[ -s "${dir}/train/checkpoints/last.pt" ]] || return 1
  [[ -s "${dir}/contract/check_contract_stdout.json" ]] || return 1
  [[ -s "${dir}/evaluation/evaluation_metrics.json" ]] || return 1
  [[ -s "${dir}/spatho_export/evidence_manifest.json" ]] || return 1
  [[ -s "${dir}/checkpoint_card/stgpt_model_manifest.json" ]] || return 1
  validate_passed "${dir}" || return 1
}

archive_run() {
  local name="$1"
  local reason="$2"
  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  local dir
  dir="$(run_dir "${name}")"
  if [[ -e "${dir}" ]]; then
    mv "${dir}" "${dir}_${reason}_${stamp}"
    log "ARCHIVE run=${name} reason=${reason} path=${dir}_${reason}_${stamp}"
  fi
  if [[ -f "$(job_log "${name}")" ]]; then
    mv "$(job_log "${name}")" "${LOGDIR}/${name}_${reason}_${stamp}.log"
    log "ARCHIVE_LOG run=${name} reason=${reason} path=${LOGDIR}/${name}_${reason}_${stamp}.log"
  fi
}

launch_validate_first() {
  local name="$1"
  local config="$2"
  local role="$3"
  local preferred_gpu="$4"
  local gpu
  gpu="$(choose_gpu "${preferred_gpu}")"
  log "LAUNCH_VALIDATE_FIRST run=${name} gpu=${gpu}"
  nohup "${VALIDATE_FIRST}" "${config}" "$(run_dir "${name}")" "${role}" "${gpu}" \
    > "$(job_log "${name}")" 2>&1 &
}

launch_reuse_qc() {
  local name="$1"
  local config="$2"
  local role="$3"
  local preferred_gpu="$4"
  local source_run="$5"
  local gpu
  gpu="$(choose_gpu "${preferred_gpu}")"
  log "LAUNCH_REUSE_QC run=${name} gpu=${gpu} source=${source_run}"
  nohup env STGPT_TRAIN_NUM_WORKERS_OVERRIDE="${STGPT_BASELINE_NUM_WORKERS:-2}" \
    "${REUSE_QC}" "${config}" "$(run_dir "${name}")" "${role}" "${gpu}" "${source_run}" \
    > "$(job_log "${name}")" 2>&1 &
}

launch_tail() {
  local name="$1"
  local config="$2"
  local role="$3"
  local preferred_gpu="$4"
  local gpu
  gpu="$(choose_gpu "${preferred_gpu}")"
  log "LAUNCH_TAIL run=${name} gpu=${gpu}"
  nohup "${TAIL}" "${config}" "$(run_dir "${name}")" "${role}" "${gpu}" \
    >> "$(job_log "${name}")" 2>&1 &
}

status_line() {
  local name="$1"
  local dir
  dir="$(run_dir "${name}")"
  local checkpoints=0
  if [[ -d "${dir}/train/checkpoints" ]]; then
    checkpoints="$(find "${dir}/train/checkpoints" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | wc -l)"
  fi
  local validate_bytes="missing"
  local train_bytes="missing"
  [[ -f "${dir}/validate_data.json" ]] && validate_bytes="$(stat -c%s "${dir}/validate_data.json")"
  [[ -f "${dir}/train_stdout.json" ]] && train_bytes="$(stat -c%s "${dir}/train_stdout.json")"
  log "STATUS run=${name} active=$(active_for_run "${name}" && echo yes || echo no) complete=$(complete_run "${dir}" && echo yes || echo no) validate_bytes=${validate_bytes} train_bytes=${train_bytes} checkpoints=${checkpoints}"
}

ensure_run() {
  local name="$1"
  local config="$2"
  local role="$3"
  local preferred_gpu="$4"
  local mode="$5"
  local source_run="${6:-}"
  local dir
  dir="$(run_dir "${name}")"

  if complete_run "${dir}"; then
    return 0
  fi
  if active_for_run "${name}"; then
    return 0
  fi
  if validate_passed "${dir}" && [[ -s "${dir}/train/checkpoints/last.pt" ]] && checkpoint_for_role "${dir}" "${role}" >/dev/null 2>&1; then
    launch_tail "${name}" "${config}" "${role}" "${preferred_gpu}"
    return 0
  fi

  archive_run "${name}" "guard_relaunch"
  if [[ "${mode}" == "reuse_qc" && -n "${source_run}" && -s "${source_run}/validate_data.json" ]]; then
    launch_reuse_qc "${name}" "${config}" "${role}" "${preferred_gpu}" "${source_run}"
  else
    launch_validate_first "${name}" "${config}" "${role}" "${preferred_gpu}"
  fi
}

run_evidence_if_ready() {
  local full_dir
  local base_dir
  full_dir="$(run_dir "${FULL}")"
  base_dir="$(run_dir "${BASE}")"
  if ! complete_run "${full_dir}" || ! complete_run "${base_dir}"; then
    return 1
  fi
  if [[ -s "${PROJECT}/runs/evidence/l3_20260507_43case/evidence_summary.json" ]]; then
    log "EVIDENCE_ALREADY_PRESENT output=${PROJECT}/runs/evidence/l3_20260507_43case"
    return 0
  fi
  log "EVIDENCE_SUMMARY_START output=${PROJECT}/runs/evidence/l3_20260507_43case"
  (
    cd "${REPO}"
    stgpt evidence-summary --suite configs/evidence/l3_43.yaml --output "${PROJECT}/runs/evidence/l3_20260507_43case"
  ) >> "${GUARD_LOG}" 2>&1
  log "EVIDENCE_SUMMARY_DONE output=${PROJECT}/runs/evidence/l3_20260507_43case"
}

log "GUARD_START root=${ROOT} gpu_pool=${GPU_POOL[*]}"
while true; do
  log "GPU $(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | tr '\n' ';')"
  ps -u taobo.hu -o pid,stat,etime,%cpu,%mem,cmd \
    | grep -E "run_l3_43_pipeline_(validate_first|reuse_qc|tail)|stgpt (train|evaluate|export-spatho|package-model)" \
    | grep -v grep | tee -a "${GUARD_LOG}" || true

  status_line "${FULL}"
  status_line "${BASE}"

  ensure_run "${FULL}" "${FULL_CONFIG}" "best_alignment" "4" "validate_first"
  ensure_run "${BASE}" "${BASE_CONFIG}" "best_loss" "6" "reuse_qc" "$(run_dir "${FULL}")"

  if run_evidence_if_ready; then
    log "FINAL_RESULT_READY"
    exit 0
  fi

  sleep "${STGPT_GUARD_INTERVAL_SECONDS:-300}"
done
