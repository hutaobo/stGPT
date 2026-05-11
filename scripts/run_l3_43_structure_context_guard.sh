#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="${STGPT_PROJECT_ROOT:-/data/taobo.hu/projects/stgpt_l3_20260504}"
REPO="${STGPT_REPO_DIR:-${PROJECT}/repos/stGPT_e5_codex}"
ROOT="${STGPT_L3_RUN_ROOT:-${PROJECT}/runs/pilot_runs/l3_20260507_43case}"
LOGDIR="${STGPT_L3_LOG_DIR:-${PROJECT}/logs/l3_20260507_43case/jobs}"
PY="${PROJECT}/.venv/bin/python"

RUN="structure_context_m6_20k"
CONFIG="configs/pilots/l3_43/structure_context_m6_20k.yaml"
ROLE="best_loss"
VALIDATE_FIRST="${REPO}/scripts/run_l3_43_pipeline_validate_first.sh"
TAIL="${REPO}/scripts/run_l3_43_pipeline_tail.sh"
GUARD_LOG="${LOGDIR}/l3_43_structure_context_guard.log"
GPU_POOL=(${STGPT_L3_GPU_POOL:-4 5 6 7})

export STGPT_XENIUM_SLIDES="${STGPT_XENIUM_SLIDES:-${PROJECT}/data/xenium_slides}"
export STGPT_OUTPUT_ROOT="${STGPT_OUTPUT_ROOT:-${PROJECT}/runs}"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"

if [[ -f "${PROJECT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${PROJECT}/.venv/bin/activate"
fi

mkdir -p "${LOGDIR}"
chmod +x "${VALIDATE_FIRST}" "${TAIL}" 2>/dev/null || true

log() {
  echo "$(date --iso-8601=seconds) $*" | tee -a "${GUARD_LOG}"
}

run_dir() {
  echo "${ROOT}/${RUN}"
}

job_log() {
  echo "${LOGDIR}/${RUN}.log"
}

active_for_run() {
  ps -u taobo.hu -o pid=,cmd= \
    | grep -E "run_l3_43_pipeline_(validate_first|tail).*${RUN}|stgpt (train|evaluate|export-spatho|package-model).*${RUN}" \
    | grep -v grep >/dev/null 2>&1
}

gpu_used_by_stgpt() {
  local gpu="$1"
  ps -u taobo.hu -o cmd= \
    | grep -E "run_l3_43_pipeline_(validate_first|tail).*[[:space:]]${gpu}([[:space:]]|$)" \
    | grep -v grep >/dev/null 2>&1
}

choose_gpu() {
  local gpu
  for gpu in "${GPU_POOL[@]}"; do
    if ! gpu_used_by_stgpt "${gpu}"; then
      echo "${gpu}"
      return 0
    fi
  done
  echo "${GPU_POOL[0]}"
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

checkpoint_for_role() {
  local dir="$1"
  if [[ -s "${dir}/train/checkpoints/best.pt" ]]; then
    echo "${dir}/train/checkpoints/best.pt"
  elif [[ -s "${dir}/train/checkpoints/last.pt" ]]; then
    echo "${dir}/train/checkpoints/last.pt"
  else
    return 1
  fi
}

archive_run() {
  local reason="$1"
  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  local dir
  dir="$(run_dir)"
  if [[ -e "${dir}" ]]; then
    mv "${dir}" "${dir}_${reason}_${stamp}"
    log "ARCHIVE run=${RUN} reason=${reason} path=${dir}_${reason}_${stamp}"
  fi
  if [[ -f "$(job_log)" ]]; then
    mv "$(job_log)" "${LOGDIR}/${RUN}_${reason}_${stamp}.log"
    log "ARCHIVE_LOG run=${RUN} reason=${reason} path=${LOGDIR}/${RUN}_${reason}_${stamp}.log"
  fi
}

launch_validate_first() {
  local gpu
  gpu="$(choose_gpu)"
  log "LAUNCH_VALIDATE_FIRST run=${RUN} gpu=${gpu}"
  nohup "${VALIDATE_FIRST}" "${CONFIG}" "$(run_dir)" "${ROLE}" "${gpu}" > "$(job_log)" 2>&1 &
}

launch_tail() {
  local gpu
  gpu="$(choose_gpu)"
  log "LAUNCH_TAIL run=${RUN} gpu=${gpu}"
  nohup "${TAIL}" "${CONFIG}" "$(run_dir)" "${ROLE}" "${gpu}" >> "$(job_log)" 2>&1 &
}

run_evidence_if_ready() {
  local dir
  dir="$(run_dir)"
  if ! complete_run "${dir}"; then
    return 1
  fi
  log "EVIDENCE_SUMMARY_START output=${PROJECT}/runs/evidence/l3_20260507_43case"
  (
    cd "${REPO}"
    stgpt evidence-summary --suite configs/evidence/l3_43.yaml --output "${PROJECT}/runs/evidence/l3_20260507_43case"
  ) >> "${GUARD_LOG}" 2>&1
  log "EVIDENCE_SUMMARY_DONE output=${PROJECT}/runs/evidence/l3_20260507_43case"
}

status_line() {
  local dir
  dir="$(run_dir)"
  local checkpoints=0
  if [[ -d "${dir}/train/checkpoints" ]]; then
    checkpoints="$(find "${dir}/train/checkpoints" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | wc -l)"
  fi
  log "STATUS run=${RUN} active=$(active_for_run && echo yes || echo no) complete=$(complete_run "${dir}" && echo yes || echo no) checkpoints=${checkpoints}"
}

log "GUARD_START run=${RUN} root=${ROOT} gpu_pool=${GPU_POOL[*]}"
while true; do
  log "GPU $(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | tr '\n' ';')"
  ps -u taobo.hu -o pid,stat,etime,%cpu,%mem,cmd \
    | grep -E "run_l3_43_pipeline_(validate_first|tail)|stgpt (train|evaluate|export-spatho|package-model)" \
    | grep -v grep | tee -a "${GUARD_LOG}" || true

  status_line
  if complete_run "$(run_dir)"; then
    if run_evidence_if_ready; then
      log "FINAL_RESULT_READY"
      exit 0
    fi
  elif active_for_run; then
    log "RUN_ACTIVE run=${RUN}"
  elif validate_passed "$(run_dir)" && checkpoint_for_role "$(run_dir)" >/dev/null 2>&1; then
    launch_tail
  else
    archive_run "guard_relaunch"
    launch_validate_first
  fi

  sleep "${STGPT_GUARD_INTERVAL_SECONDS:-300}"
done
