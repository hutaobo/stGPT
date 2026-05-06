#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/data/taobo.hu/projects/stgpt_l3_20260504}"
CODE="${CODE:-$PROJECT/repos/stGPT_e4_codex}"
LOG_DIR="$PROJECT/logs/e4"
mkdir -p "$LOG_DIR"
export PYTHONPATH="$CODE/src"

launch() {
  local gpu="$1"
  local run_id="$2"
  local cfg="$CODE/configs/pilots/atera_wta_v1_long_run/${run_id}.yaml"
  local run_dir="$PROJECT/runs/atera_wta_v1_long_run/$run_id"
  local train_dir="$run_dir/train"
  mkdir -p "$train_dir"
  date -Is > "$train_dir/train_10000.started.txt"
  echo "[$(date -Is)] launch gpu=$gpu run=$run_id cfg=$cfg" | tee -a "$LOG_DIR/e4_deep_launch.log"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$PYTHONPATH" "$PROJECT/.venv/bin/stgpt" train --config "$cfg" \
    > "$train_dir/train_10000.stdout.log" 2> "$train_dir/train_10000.stderr.log" &
  local pid=$!
  echo "$pid" > "$train_dir/pid.txt"
  echo "[$(date -Is)] pid=$pid gpu=$gpu run=$run_id" | tee -a "$LOG_DIR/e4_deep_launch.log"
}

launch 3 breast_gene_spatial_10k_contour_unit
launch 4 cervical_gene_spatial_10k_contour_unit
launch 5 breast_full_m6_contour_store_10k
launch 6 cervical_full_m6_contour_store_10k
