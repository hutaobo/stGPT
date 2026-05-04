param(
    [string]$Remote = "sscb-a100.scilifelab.se",
    [string]$Project = "/data/taobo.hu/projects/stgpt_l3_20260504",
    [string]$LocalStGPT = "D:\GitHub\stGPT",
    [string]$LocalPyXenium = "D:\GitHub\pyXenium",
    [string]$SlideRoot = "D:\GitHub\stGPT\outputs\xenium_slides",
    [string]$WorkRoot = "D:\GitHub\stGPT\outputs\a100_upload_20260504",
    [string]$GpuList = "4,5,6,7",
    [int]$MaxSteps = 1000,
    [int]$BatchSize = 32
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Write-Log {
    param([string]$Message)
    $stamp = (Get-Date).ToString("s")
    $line = "[$stamp] $Message"
    Write-Host $line
    Add-Content -Path $script:LogPath -Value $line
}

function Invoke-Checked {
    param(
        [string]$Label,
        [scriptblock]$Command
    )
    Write-Log "START $Label"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
    Write-Log "DONE  $Label"
}

$work = New-Item -ItemType Directory -Force -Path $WorkRoot
$packages = New-Item -ItemType Directory -Force -Path (Join-Path $work.FullName "packages")
$casePackages = New-Item -ItemType Directory -Force -Path (Join-Path $packages.FullName "cases")
$logs = New-Item -ItemType Directory -Force -Path (Join-Path $work.FullName "logs")
$script:LogPath = Join-Path $logs.FullName "upload_and_train.log"
trap {
    $message = ($_ | Out-String).Trim()
    $stamp = (Get-Date).ToString("s")
    Add-Content -Path $script:LogPath -Value "[$stamp] ERROR $message"
    Write-Error $message
    exit 1
}
Write-Log "A100 upload/training automation started."
Write-Log "Remote=$Remote Project=$Project GPUs=$GpuList"

$caseList = Join-Path $work.FullName "l3_cases.txt"
@"
import csv
from pathlib import Path
root = Path(r"$SlideRoot")
with (root / "training_manifest_l3.csv").open(newline="", encoding="utf-8-sig") as handle:
    rows = list(csv.DictReader(handle))
for row in rows:
    print(row["case_leaf"])
"@ | python - | Set-Content -Path $caseList -Encoding ascii
$cases = Get-Content -Path $caseList | Where-Object { $_.Trim() }
if ($cases.Count -lt 1) {
    throw "No cases found in $SlideRoot\training_manifest_l3.csv"
}
Write-Log "Training cases: $($cases.Count)"

Invoke-Checked "prepare remote project directories" {
    ssh -o BatchMode=yes $Remote "set -e; mkdir -p '$Project/repos/stGPT' '$Project/repos/pyXenium' '$Project/data/xenium_slides' '$Project/configs/l3_cases' '$Project/logs' '$Project/packages/cases' '$Project/runs/l3_cases'"
}

$stgptPackage = Join-Path $packages.FullName "stGPT_src.tgz"
$pyxeniumPackage = Join-Path $packages.FullName "pyXenium_src.tgz"
Invoke-Checked "package stGPT source" {
    tar --force-local -C $LocalStGPT -czf $stgptPackage pyproject.toml README.md LICENSE NOTICE MANIFEST.in .readthedocs.yaml src tests configs docs scripts
}
Invoke-Checked "package pyXenium source" {
    tar --force-local -C $LocalPyXenium -czf $pyxeniumPackage pyproject.toml README.md LICENSE .readthedocs.yaml src tests scripts examples
}
Invoke-Checked "upload source packages" {
    scp $stgptPackage "${Remote}:$Project/packages/stGPT_src.tgz"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    scp $pyxeniumPackage "${Remote}:$Project/packages/pyXenium_src.tgz"
}
Invoke-Checked "extract source packages remotely" {
    ssh -o BatchMode=yes $Remote "set -e; rm -rf '$Project/repos/stGPT' '$Project/repos/pyXenium'; mkdir -p '$Project/repos/stGPT' '$Project/repos/pyXenium'; tar -C '$Project/repos/stGPT' -xzf '$Project/packages/stGPT_src.tgz'; tar -C '$Project/repos/pyXenium' -xzf '$Project/packages/pyXenium_src.tgz'"
}

$globalPackage = Join-Path $packages.FullName "xenium_slide_globals.tgz"
Invoke-Checked "package global XeniumSlide manifests" {
    tar --force-local -C $SlideRoot -czf $globalPackage training_manifest_l3.csv training_manifest_l3.json dataset_registry.csv dataset_registry.json l3_upgrade_summary.json readiness_summary.json blocked_cases.csv blocked_cases.json source_asset_resolution_report.csv source_asset_resolution_report.json backfill_results.csv backfill_results.json
}
Invoke-Checked "upload global XeniumSlide manifests" {
    scp $globalPackage "${Remote}:$Project/packages/xenium_slide_globals.tgz"
}
Invoke-Checked "extract global XeniumSlide manifests" {
    ssh -o BatchMode=yes $Remote "set -e; tar -C '$Project/data/xenium_slides' -xzf '$Project/packages/xenium_slide_globals.tgz'"
}

$uploaded = 0
foreach ($case in $cases) {
    $case = $case.Trim()
    if (-not $case) { continue }
    $remoteMarker = "$Project/data/xenium_slides/$case/.upload_complete"
    $already = ssh -o BatchMode=yes $Remote "test -f '$remoteMarker' && echo yes || echo no"
    if ($already.Trim() -eq "yes") {
        $uploaded += 1
        Write-Log "SKIP case already uploaded: $case ($uploaded/$($cases.Count))"
        continue
    }
    $casePackage = Join-Path $casePackages.FullName "$case.tar"
    if (Test-Path $casePackage) {
        Remove-Item -LiteralPath $casePackage -Force
    }
    Invoke-Checked "package case $case" {
        tar --force-local -C $SlideRoot -cf $casePackage $case
    }
    Invoke-Checked "upload case $case" {
        scp $casePackage "${Remote}:$Project/packages/cases/$case.tar"
    }
    Invoke-Checked "extract case $case" {
        ssh -o BatchMode=yes $Remote "set -e; rm -rf '$Project/data/xenium_slides/$case'; tar -C '$Project/data/xenium_slides' -xf '$Project/packages/cases/$case.tar'; touch '$remoteMarker'; test -d '$Project/data/xenium_slides/$case/xenium_slide.zarr'; test -s '$Project/data/xenium_slides/$case/contour_patches_manifest.json'"
    }
    Remove-Item -LiteralPath $casePackage -Force
    $uploaded += 1
    Write-Log "UPLOADED case $case ($uploaded/$($cases.Count))"
}

$remoteSetupPath = Join-Path $work.FullName "remote_setup_and_train.sh"
$remoteSetupContent = @"
#!/usr/bin/env bash
set -euo pipefail
PROJECT="$Project"
GPU_LIST="$GpuList"
MAX_STEPS="$MaxSteps"
BATCH_SIZE="$BatchSize"
LOG_DIR="`$PROJECT/logs"
SLIDE_ROOT="`$PROJECT/data/xenium_slides"
mkdir -p "`$LOG_DIR" "`$PROJECT/configs/l3_cases" "`$PROJECT/runs/l3_cases"

echo "[`$(date -Is)] rewrite uploaded Windows paths"
python3 - <<'PY'
import csv, json
from pathlib import Path
PROJECT = Path("$Project")
slide_root = PROJECT / "data" / "xenium_slides"
old_roots = [
    r"D:\GitHub\stGPT\outputs\xenium_slides",
    "D:/GitHub/stGPT/outputs/xenium_slides",
]
new_root = str(slide_root)

def replace_value(value):
    if isinstance(value, str):
        out = value
        for old in old_roots:
            out = out.replace(old, new_root)
        return out
    if isinstance(value, list):
        return [replace_value(item) for item in value]
    if isinstance(value, dict):
        return {key: replace_value(item) for key, item in value.items()}
    return value

json_files = list(slide_root.glob("*.json")) + [
    path for case in slide_root.iterdir() if case.is_dir()
    for path in [
        case / "slide_manifest.json",
        case / "qc_report.json",
        case / "metadata_10x.json",
        case / "contour_source_manifest.json",
        case / "contour_patches_manifest.json",
        case / "contour_patch_failures.json",
    ]
    if path.exists()
]
for path in json_files:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        continue
    path.write_text(json.dumps(replace_value(payload), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

for path in slide_root.glob("*.csv"):
    text = path.read_text(encoding="utf-8")
    for old in old_roots:
        text = text.replace(old, new_root)
    path.write_text(text, encoding="utf-8")
PY

echo "[`$(date -Is)] create/update Python environment"
if [ ! -d "`$PROJECT/.venv" ]; then
  python3 -m venv "`$PROJECT/.venv"
fi
. "`$PROJECT/.venv/bin/activate"
if ! python -m pip --version >/dev/null 2>&1; then
  python -m ensurepip --upgrade
fi
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e "`$PROJECT/repos/pyXenium"
python -m pip install -e "`$PROJECT/repos/stGPT"

echo "[`$(date -Is)] generate per-case stGPT configs"
python - <<'PY'
import csv
from pathlib import Path
PROJECT = Path("$Project")
slide_root = PROJECT / "data" / "xenium_slides"
config_dir = PROJECT / "configs" / "l3_cases"
config_dir.mkdir(parents=True, exist_ok=True)
with (slide_root / "training_manifest_l3.csv").open(newline="", encoding="utf-8-sig") as handle:
    rows = list(csv.DictReader(handle))
for row in rows:
    case = row["case_leaf"]
    case_dir = slide_root / case
    text = f"""case_name: {case}
data:
  mode: xenium_slide
  output_dir: {PROJECT}/runs/l3_cases/{case}/case
  slide_store: {case_dir}/xenium_slide.zarr
  patch_manifest: {case_dir}/contour_patches_manifest.json
  structure_assignments_csv: {case_dir}/structure_assignments.csv
  slide_id: {case}
  batch_id: 10x_public_l3
  stain: H&E
  min_cells_per_region: 1
  include_structure_context: false
model:
  d_model: 128
  n_heads: 4
  n_layers: 4
  dim_feedforward: 256
  max_genes: 512
  n_expression_bins: 32
  image_size: 128
  max_cells_per_region: 32
  n_prototypes: 128
  prototype_temperature: 0.1
  use_expression_values: true
  use_image_context: true
  use_spatial_context: true
  use_structure_context: false
  use_cell_context: true
training:
  batch_size: $BatchSize
  learning_rate: 0.0003
  weight_decay: 0.01
  max_steps: $MaxSteps
  warmup_steps: 50
  lr_schedule: cosine
  save_every_n_steps: 200
  mask_probability: 0.15
  neighborhood_k: 8
  image_gene_loss_weight: 0.05
  neighborhood_loss_weight: 0.25
  structure_loss_weight: 0.0
  prototype_loss_weight: 0.1
  prototype_queue_size: 4096
  prototype_queue_start_steps: 20
  prototype_sinkhorn_iterations: 3
  output_dir: {PROJECT}/runs/l3_cases/{case}/train
  device: cuda
  num_workers: 4
  seed: 7
split:
  strategy: spatial_block
  train_fraction: 0.7
  val_fraction: 0.15
  test_fraction: 0.15
  seed: 11
"""
    (config_dir / f"{case}.yaml").write_text(text, encoding="utf-8")
print(f"wrote {len(rows)} configs to {config_dir}")
PY

cat > "`$PROJECT/run_l3_training_queue.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
PROJECT="`${PROJECT:-/data/taobo.hu/projects/stgpt_l3_20260504}"
GPU_LIST="`${GPU_LIST:-4,5,6,7}"
CONFIG_DIR="`$PROJECT/configs/l3_cases"
RUN_DIR="`$PROJECT/runs/l3_cases"
LOG_DIR="`$PROJECT/logs"
QUEUE="`$PROJECT/logs/l3_training_queue.txt"
LOCK="`$PROJECT/logs/l3_training_queue.lock"
mkdir -p "`$LOG_DIR" "`$RUN_DIR"
find "`$CONFIG_DIR" -name '*.yaml' | sort > "`$QUEUE"
. "`$PROJECT/.venv/bin/activate"
IFS=',' read -ra GPUS <<< "`$GPU_LIST"
claim_next() {
  local gpu="`$1"
  python - "`$QUEUE" "`$LOCK" "`$RUN_DIR" "`$gpu" <<'PY'
import fcntl, sys
from pathlib import Path
queue = Path(sys.argv[1])
lock = Path(sys.argv[2])
run_dir = Path(sys.argv[3])
gpu = sys.argv[4]
with lock.open("w") as handle:
    fcntl.flock(handle, fcntl.LOCK_EX)
    configs = [line.strip() for line in queue.read_text().splitlines() if line.strip()]
    for cfg in configs:
        case = Path(cfg).stem
        status_dir = run_dir / case
        done = status_dir / "train" / "checkpoints" / "last.pt"
        metrics = status_dir / "train" / "metrics.json"
        running = status_dir / ".running"
        failed = status_dir / ".failed"
        if done.exists() or metrics.exists():
            continue
        if running.exists():
            continue
        status_dir.mkdir(parents=True, exist_ok=True)
        running.write_text(gpu)
        failed.unlink(missing_ok=True)
        print(cfg)
        break
PY
}
worker() {
  local gpu="`$1"
  while true; do
    cfg="`$(claim_next "`$gpu" | tail -n 1)"
    if [ -z "`$cfg" ]; then
      echo "[`$(date -Is)] gpu=`$gpu no more configs" | tee -a "`$LOG_DIR/training_queue.log"
      break
    fi
    case="`$(basename "`$cfg" .yaml)"
    log="`$LOG_DIR/train_`${case}_gpu`${gpu}.log"
    echo "[`$(date -Is)] gpu=`$gpu start `$case" | tee -a "`$LOG_DIR/training_queue.log"
    set +e
    CUDA_VISIBLE_DEVICES="`$gpu" stgpt train --config "`$cfg" > "`$log" 2>&1
    code=`$?
    set -e
    rm -f "`$RUN_DIR/`$case/.running"
    if [ "`$code" -eq 0 ]; then
      echo "[`$(date -Is)] gpu=`$gpu done `$case" | tee -a "`$LOG_DIR/training_queue.log"
    else
      echo "`$code" > "`$RUN_DIR/`$case/.failed"
      echo "[`$(date -Is)] gpu=`$gpu failed `$case code=`$code" | tee -a "`$LOG_DIR/training_queue.log"
    fi
  done
}
for gpu in "`${GPUS[@]}"; do
  worker "`$gpu" &
done
wait
SH
chmod +x "`$PROJECT/run_l3_training_queue.sh"

if [ ! -f "`$PROJECT/logs/training_queue.pid" ] || ! kill -0 "`$(cat "`$PROJECT/logs/training_queue.pid")" 2>/dev/null; then
  echo "[`$(date -Is)] launching training queue on GPUs `$GPU_LIST"
  nohup env PROJECT="`$PROJECT" GPU_LIST="`$GPU_LIST" bash "`$PROJECT/run_l3_training_queue.sh" > "`$PROJECT/logs/training_queue.nohup.log" 2>&1 &
  echo `$! > "`$PROJECT/logs/training_queue.pid"
else
  echo "[`$(date -Is)] training queue already running pid=`$(cat "`$PROJECT/logs/training_queue.pid")"
fi

echo "[`$(date -Is)] remote setup/training launch complete"
"@
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText($remoteSetupPath, $remoteSetupContent, $utf8NoBom)
Invoke-Checked "upload remote setup/training script" {
    scp $remoteSetupPath "${Remote}:$Project/packages/remote_setup_and_train.sh"
}
Invoke-Checked "run remote setup/training script" {
    ssh -o BatchMode=yes $Remote "set -e; bash '$Project/packages/remote_setup_and_train.sh' > '$Project/logs/remote_setup_and_train.log' 2>&1 & echo `$! > '$Project/logs/remote_setup_and_train.pid'; echo remote_setup_pid=`$(cat '$Project/logs/remote_setup_and_train.pid')"
}

Write-Log "A100 upload/training automation finished local phase. Remote setup/training is running in background."
