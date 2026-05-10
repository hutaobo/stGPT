#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${STGPT_PROJECT_ROOT:-/data/taobo.hu/projects/stgpt_l3_20260504}"
REPO_DIR="${STGPT_REPO_DIR:-${PROJECT_ROOT}/repos/stGPT_e5_codex}"
SLIDE_ROOT="${STGPT_XENIUM_SLIDES:-${PROJECT_ROOT}/data/xenium_slides}"
OUTPUT_ROOT="${STGPT_OUTPUT_ROOT:-${PROJECT_ROOT}/runs}"
L3_VERSION="${STGPT_L3_VERSION:-l3_20260507_43case}"

export STGPT_XENIUM_SLIDES="${SLIDE_ROOT}"
export STGPT_OUTPUT_ROOT="${OUTPUT_ROOT}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

if [[ -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/.venv/bin/activate"
fi

run_stgpt() {
  if command -v stgpt >/dev/null 2>&1; then
    stgpt "$@"
  else
    python -c 'from stgpt.cli import app; app()' "$@"
  fi
}

log_stage() {
  echo "$(date --iso-8601=seconds) STAGE validate_first run=$(basename "${RUN_DIR}") ${*}" >&2
}

checkpoint_for_role() {
  local role="$1"
  local train_dir="${RUN_DIR}/train/checkpoints"
  if [[ "${role}" == "best_alignment" && -f "${train_dir}/best_alignment.pt" ]]; then
    echo "${train_dir}/best_alignment.pt"
  elif [[ "${role}" == "best_loss" && -f "${train_dir}/best.pt" ]]; then
    echo "${train_dir}/best.pt"
  elif [[ -f "${train_dir}/best.pt" ]]; then
    echo "${train_dir}/best.pt"
  else
    echo "${train_dir}/last.pt"
  fi
}

if [[ "$#" -ne 4 ]]; then
  echo "Usage: $0 <config> <run_dir> <checkpoint_role> <gpu>" >&2
  exit 2
fi

CONFIG="$1"
RUN_DIR="$2"
CHECKPOINT_ROLE="$3"
GPU="$4"
CONFIG_PATH="${REPO_DIR}/${CONFIG}"

trap 'status=$?; echo "$(date --iso-8601=seconds) ERROR validate_first run=$(basename "${RUN_DIR}") line=${LINENO} status=${status}" >&2' ERR

export CUDA_VISIBLE_DEVICES="${GPU}"
mkdir -p "${RUN_DIR}"

log_stage "step=validate-data gpu=${GPU}"
run_stgpt validate-data --config "${CONFIG_PATH}" --output "${RUN_DIR}/qc" | tee "${RUN_DIR}/validate_data.json"
SPLITS="${RUN_DIR}/qc/splits.csv"

log_stage "step=train gpu=${GPU}"
run_stgpt train --config "${CONFIG_PATH}" | tee "${RUN_DIR}/train_stdout.json"

CKPT="$(checkpoint_for_role "${CHECKPOINT_ROLE}")"
mkdir -p "${RUN_DIR}/contract"

log_stage "step=check-contract checkpoint=${CKPT}"
run_stgpt check-contract --checkpoint "${CKPT}" --config "${CONFIG_PATH}" --run-dir "${RUN_DIR}" --output "${RUN_DIR}/contract/check_contract.json" | tee "${RUN_DIR}/contract/check_contract_stdout.json"

log_stage "step=evaluate checkpoint=${CKPT}"
run_stgpt evaluate --checkpoint "${CKPT}" --config "${CONFIG_PATH}" --splits "${SPLITS}" --output "${RUN_DIR}/evaluation" --batch-size 64 --device cuda | tee "${RUN_DIR}/evaluation_stdout.json"

log_stage "step=export-spatho checkpoint=${CKPT}"
run_stgpt export-spatho --checkpoint "${CKPT}" --config "${CONFIG_PATH}" --output "${RUN_DIR}/spatho_export" --batch-size 64 --device cuda | tee "${RUN_DIR}/spatho_export_stdout.json"

log_stage "step=package-model checkpoint=${CKPT}"
run_stgpt package-model --checkpoint "${CKPT}" --eval "${RUN_DIR}/evaluation/evaluation_metrics.json" --output "${RUN_DIR}/checkpoint_card" --model-name "$(basename "${RUN_DIR}")" | tee "${RUN_DIR}/package_model_stdout.json"

log_stage "step=complete"
