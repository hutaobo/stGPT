#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/data/taobo.hu/projects/stgpt_l3_20260504}"
OUT="${OUT:-$PROJECT/evidence/training_telemetry/live_sampler}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-10}"

mkdir -p "$OUT"

GPU_CSV="$OUT/gpu_timeseries.csv"
PROC_CSV="$OUT/gpu_process_timeseries.csv"
QUEUE_CSV="$OUT/training_queue_timeseries.csv"

if [ ! -f "$GPU_CSV" ]; then
  echo "sample_utc,timestamp,index,uuid,memory.used [MiB],memory.free [MiB],utilization.gpu [%],utilization.memory [%],temperature.gpu,power.draw [W],clocks.sm [MHz],clocks.mem [MHz]" > "$GPU_CSV"
fi
if [ ! -f "$PROC_CSV" ]; then
  echo "sample_utc,timestamp,gpu_uuid,pid,process_name,used_memory [MiB]" > "$PROC_CSV"
fi
if [ ! -f "$QUEUE_CSV" ]; then
  echo "sample_utc,uploaded_complete,done_last,done_metrics,failed,running,queue_pid,queue_running,stgpt_train_processes,latest_queue_log_mtime" > "$QUEUE_CSV"
fi

while true; do
  sample_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  nvidia-smi \
    --query-gpu=timestamp,index,uuid,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,clocks.sm,clocks.mem \
    --format=csv,noheader,nounits \
    | sed "s/^/$sample_utc,/" >> "$GPU_CSV" || true

  nvidia-smi \
    --query-compute-apps=timestamp,gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits \
    | sed "s/^/$sample_utc,/" >> "$PROC_CSV" || true

  uploaded="$(find "$PROJECT/data/xenium_slides" -maxdepth 2 -name .upload_complete 2>/dev/null | wc -l | tr -d ' ')"
  done_last="$(find "$PROJECT/runs/l3_cases" -path '*/train/checkpoints/last.pt' -print 2>/dev/null | wc -l | tr -d ' ')"
  done_metrics="$(find "$PROJECT/runs/l3_cases" -path '*/train/metrics.json' -print 2>/dev/null | wc -l | tr -d ' ')"
  failed="$(find "$PROJECT/runs/l3_cases" -name .failed -maxdepth 3 -print 2>/dev/null | wc -l | tr -d ' ')"
  running="$(find "$PROJECT/runs/l3_cases" -name .running -maxdepth 3 -print 2>/dev/null | wc -l | tr -d ' ')"
  queue_pid="$(cat "$PROJECT/logs/training_queue.pid" 2>/dev/null || true)"
  queue_running=0
  if [ -n "$queue_pid" ] && kill -0 "$queue_pid" 2>/dev/null; then
    queue_running=1
  fi
  train_processes="$(pgrep -af "$PROJECT/.venv/bin/stgpt train --config $PROJECT/configs/l3_cases" 2>/dev/null | wc -l | tr -d ' ')"
  latest_queue_log_mtime="$(stat -c %Y "$PROJECT/logs/training_queue.log" 2>/dev/null || echo '')"
  echo "$sample_utc,$uploaded,$done_last,$done_metrics,$failed,$running,$queue_pid,$queue_running,$train_processes,$latest_queue_log_mtime" >> "$QUEUE_CSV"

  sleep "$INTERVAL_SECONDS"
done
