#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${STGPT_PROJECT_ROOT:-/data/taobo.hu/projects/stgpt_l3_20260504}"
REPO_DIR="${STGPT_REPO_DIR:-${PROJECT_ROOT}/repos/stGPT_e5_codex}"
SLIDE_ROOT="${STGPT_XENIUM_SLIDES:-${PROJECT_ROOT}/data/xenium_slides}"
OUTPUT_ROOT="${STGPT_OUTPUT_ROOT:-${PROJECT_ROOT}/runs}"

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
  echo "$(date --iso-8601=seconds) STAGE reuse_qc run=$(basename "${RUN_DIR}") ${*}" >&2
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

if [[ "$#" -ne 5 ]]; then
  echo "Usage: $0 <config> <run_dir> <checkpoint_role> <gpu> <source_validated_run_dir>" >&2
  exit 2
fi

CONFIG="$1"
RUN_DIR="$2"
CHECKPOINT_ROLE="$3"
GPU="$4"
SOURCE_RUN_DIR="$5"
CONFIG_PATH="${REPO_DIR}/${CONFIG}"
EFFECTIVE_CONFIG="${CONFIG_PATH}"

trap 'status=$?; echo "$(date --iso-8601=seconds) ERROR reuse_qc run=$(basename "${RUN_DIR}") line=${LINENO} status=${status}" >&2' ERR

export CUDA_VISIBLE_DEVICES="${GPU}"
mkdir -p "${RUN_DIR}"

if [[ ! -s "${SOURCE_RUN_DIR}/validate_data.json" || ! -s "${SOURCE_RUN_DIR}/qc/splits.csv" ]]; then
  echo "Source run is missing validated QC artifacts: ${SOURCE_RUN_DIR}" >&2
  exit 3
fi

if [[ -n "${STGPT_TRAIN_NUM_WORKERS_OVERRIDE:-}" ]]; then
  EFFECTIVE_CONFIG="${RUN_DIR}/effective_config.yaml"
  python - "${CONFIG_PATH}" "${EFFECTIVE_CONFIG}" "${STGPT_TRAIN_NUM_WORKERS_OVERRIDE}" <<'PY'
import sys
from pathlib import Path

import yaml

source = Path(sys.argv[1])
target = Path(sys.argv[2])
num_workers = int(sys.argv[3])
payload = yaml.safe_load(source.read_text())
payload.setdefault("training", {})["num_workers"] = num_workers
target.write_text(yaml.safe_dump(payload, sort_keys=False))
PY
fi

log_stage "step=reuse-validated-qc source=${SOURCE_RUN_DIR}"
rm -rf "${RUN_DIR}/qc"
mkdir -p "${RUN_DIR}/qc"
cp -a "${SOURCE_RUN_DIR}/qc/." "${RUN_DIR}/qc/"
python - "${SOURCE_RUN_DIR}" "${RUN_DIR}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
payload = json.loads((source / "validate_data.json").read_text())
if payload.get("status") != "pass":
    raise SystemExit(f"Source validate_data.json is not pass: {payload.get('status')}")
payload["case_manifest"] = str(target / "qc" / "case_manifest.json")
payload["qc_report_json"] = str(target / "qc" / "qc_report.json")
payload["qc_report_md"] = str(target / "qc" / "qc_report.md")
payload["splits"] = str(target / "qc" / "splits.csv")
(target / "validate_data.json").write_text(json.dumps(payload, indent=2) + "\n")
(target / "validate_data_reuse_source.json").write_text(
    json.dumps(
        {
            "source_run_dir": str(source),
            "source_validate_data": str(source / "validate_data.json"),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "reason": "Reused validated 43-case QC/splits after a concurrent validate-data OOM kill.",
        },
        indent=2,
    )
    + "\n"
)
PY
SPLITS="${RUN_DIR}/qc/splits.csv"

log_stage "step=train gpu=${GPU} config=${EFFECTIVE_CONFIG}"
run_stgpt train --config "${EFFECTIVE_CONFIG}" | tee "${RUN_DIR}/train_stdout.json"

CKPT="$(checkpoint_for_role "${CHECKPOINT_ROLE}")"
mkdir -p "${RUN_DIR}/contract"

log_stage "step=check-contract checkpoint=${CKPT}"
run_stgpt check-contract --checkpoint "${CKPT}" --config "${EFFECTIVE_CONFIG}" --run-dir "${RUN_DIR}" --output "${RUN_DIR}/contract/check_contract.json" | tee "${RUN_DIR}/contract/check_contract_stdout.json"

log_stage "step=evaluate checkpoint=${CKPT}"
run_stgpt evaluate --checkpoint "${CKPT}" --config "${EFFECTIVE_CONFIG}" --splits "${SPLITS}" --output "${RUN_DIR}/evaluation" --batch-size 64 --device cuda | tee "${RUN_DIR}/evaluation_stdout.json"

log_stage "step=export-spatho checkpoint=${CKPT}"
run_stgpt export-spatho --checkpoint "${CKPT}" --config "${EFFECTIVE_CONFIG}" --output "${RUN_DIR}/spatho_export" --batch-size 64 --device cuda | tee "${RUN_DIR}/spatho_export_stdout.json"

log_stage "step=package-model checkpoint=${CKPT}"
run_stgpt package-model --checkpoint "${CKPT}" --eval "${RUN_DIR}/evaluation/evaluation_metrics.json" --output "${RUN_DIR}/checkpoint_card" --model-name "$(basename "${RUN_DIR}")" | tee "${RUN_DIR}/package_model_stdout.json"

log_stage "step=complete"
