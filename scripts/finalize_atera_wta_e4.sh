#!/usr/bin/env bash
set -euo pipefail

STGPT="${STGPT:-stgpt}"
SUITE="${SUITE:-configs/evidence/atera_wta_v1_long_run.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/evidence/atera_wta_v1_long_run_final}"
BUNDLE="${BUNDLE:-outputs/evidence/atera_wta_v1_final_evidence.tar.gz}"
RUN_ROOT="${RUN_ROOT:-outputs/pilot_runs/atera_wta_v1_long_run}"
CONFIG_ROOT="${CONFIG_ROOT:-configs/pilots/atera_wta_v1_long_run}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
DEVICE="${DEVICE:-cuda}"
EXPECTED_STEP="${EXPECTED_STEP:-10000}"
SKIP_ABLATION="${SKIP_ABLATION:-0}"

RUN_NAMES=(
  "breast_full_m6_contour_store_10k"
  "cervical_full_m6_contour_store_10k"
)
RUN_CONFIGS=(
  "${CONFIG_ROOT}/breast_full_m6_contour_store_10k.yaml"
  "${CONFIG_ROOT}/cervical_full_m6_contour_store_10k.yaml"
)
RUN_DIRS=(
  "${RUN_ROOT}/breast_full_m6_contour_store_10k"
  "${RUN_ROOT}/cervical_full_m6_contour_store_10k"
)

max_step() {
  local run_dir="$1"
  local metrics_path="${run_dir}/train/metrics.json"
  if [[ ! -f "$metrics_path" ]]; then
    echo 0
    return
  fi
  python3 - "$metrics_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = json.loads(path.read_text()) if path.exists() else []
steps = [int(row.get("step", idx + 1)) for idx, row in enumerate(rows) if isinstance(row, dict)]
print(max(steps) if steps else 0)
PY
}

mkdir -p "$OUTPUT_ROOT" "$(dirname "$BUNDLE")"

for idx in "${!RUN_NAMES[@]}"; do
  step="$(max_step "${RUN_DIRS[$idx]}")"
  if [[ "$step" -lt "$EXPECTED_STEP" ]]; then
    echo "WARN: ${RUN_NAMES[$idx]} has max step ${step}, below expected ${EXPECTED_STEP}; writing available artifacts." >&2
  fi
done

"$STGPT" evidence-summary --suite "$SUITE" --output "$OUTPUT_ROOT"

for idx in "${!RUN_NAMES[@]}"; do
  name="${RUN_NAMES[$idx]}"
  config="${RUN_CONFIGS[$idx]}"
  run_dir="${RUN_DIRS[$idx]}"
  checkpoint="${run_dir}/train/checkpoints/best_alignment.pt"
  run_output="${OUTPUT_ROOT}/${name}"
  mkdir -p "$run_output"

  if [[ -f "$checkpoint" ]]; then
    "$STGPT" check-contract --checkpoint "$checkpoint" --config "$config" --run-dir "$run_dir" --output "${run_output}/contract_check.json"
  else
    echo "WARN: ${name} is missing best_alignment.pt" >&2
  fi

  if [[ -f "${run_dir}/spatho_export/contour_evidence_chains.jsonl" ]]; then
    failure_output="${run_output}/failure_gallery"
    "$STGPT" failure-gallery --run-dir "$run_dir" --output "$failure_output" --max-items 100
    targets="${failure_output}/ablation_targets.json"
    if [[ "$SKIP_ABLATION" != "1" && -f "$checkpoint" && -f "$targets" ]]; then
      "$STGPT" ablate --checkpoint "$checkpoint" --config "$config" --targets "$targets" --output "${run_output}/ablation" --batch-size "$EVAL_BATCH_SIZE" --device "$DEVICE"
    fi
  else
    echo "WARN: ${name} is missing Spatho evidence chains" >&2
  fi
done

"$STGPT" latent-manifold --suite "$SUITE" --output "${OUTPUT_ROOT}/latent_manifold" --reducer auto --max-html-points 5000

rm -f "$BUNDLE"
tar -czf "$BUNDLE" -C "$(dirname "$OUTPUT_ROOT")" "$(basename "$OUTPUT_ROOT")"
echo "Final evidence bundle: $BUNDLE"
