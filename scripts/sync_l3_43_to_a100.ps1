param(
    [string]$HostName = "sscb-a100.scilifelab.se",
    [string]$RemoteProject = "/data/taobo.hu/projects/stgpt_l3_20260504",
    [string]$RemoteRepo = "/data/taobo.hu/projects/stgpt_l3_20260504/repos/stGPT_e5_codex",
    [string]$RemoteSlides = "/data/taobo.hu/projects/stgpt_l3_20260504/data/xenium_slides",
    [string]$LocalSlides = "D:/GitHub/stGPT/outputs/xenium_slides",
    [switch]$ForceCases,
    [switch]$StartPack
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$localSlidesPath = Resolve-Path $LocalSlides
$cases = @(
    "Xenium_Prime_Cervical_Cancer_FFPE_outs",
    "Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_outs",
    "Xenium_V1_FFPE_Human_Breast_IDC_Big_1_outs",
    "Xenium_V1_FFPE_Human_Breast_IDC_Big_2_outs"
)
$rootFiles = @(
    "training_manifest_l3.csv",
    "l3_upgrade_summary.json",
    "l3_patch_extraction_20260507_results.json",
    "dataset_inventory.json",
    "dataset_registry.csv",
    "dataset_registry.parquet",
    "build_summary.json",
    "metadata_resolution_report.json",
    "failed_cases.csv",
    "failed_cases.json"
)

Write-Host "Syncing stGPT code/config/scripts to $HostName..."
ssh $HostName "mkdir -p '$RemoteRepo' '$RemoteSlides' '$RemoteProject/logs/l3_20260507_43case'"
scp -r `
    (Join-Path $repoRoot "src") `
    (Join-Path $repoRoot "configs") `
    (Join-Path $repoRoot "scripts") `
    (Join-Path $repoRoot "pyproject.toml") `
    "${HostName}:$RemoteRepo/"
ssh $HostName "chmod +x '$RemoteRepo/scripts/run_l3_43_remote.sh'"

Write-Host "Syncing frozen L3 root artifacts..."
foreach ($name in $rootFiles) {
    $path = Join-Path $localSlidesPath $name
    if (Test-Path $path) {
        scp $path "${HostName}:$RemoteSlides/"
    }
}

$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) "stgpt_l3_43_sync"
New-Item -ItemType Directory -Force -Path $tempRoot | Out-Null

foreach ($case in $cases) {
    $remoteCase = "$RemoteSlides/$case"
    $exists = ssh $HostName "test -d '$remoteCase' && echo exists || true"
    $existsText = ($exists | Out-String).Trim()
    if ($existsText -eq "exists" -and -not $ForceCases) {
        Write-Host "Case already present, skipping: $case"
        continue
    }
    $localCase = Join-Path $localSlidesPath $case
    if (-not (Test-Path $localCase)) {
        throw "Missing local case directory: $localCase"
    }
    $archive = Join-Path $tempRoot "$case.tar.gz"
    if (Test-Path $archive) {
        Remove-Item -LiteralPath $archive -Force
    }
    Write-Host "Packaging $case..."
    tar --force-local -C $localSlidesPath -czf $archive $case
    Write-Host "Uploading $case..."
    scp $archive "${HostName}:$RemoteProject/"
    ssh $HostName "mkdir -p '$RemoteSlides' && tar -C '$RemoteSlides' -xzf '$RemoteProject/$case.tar.gz' && rm -f '$RemoteProject/$case.tar.gz'"
}

if ($StartPack) {
    Write-Host "Starting remote L3-43 pack job..."
    ssh $HostName "cd '$RemoteRepo' && nohup bash scripts/run_l3_43_remote.sh pack > '$RemoteProject/logs/l3_20260507_43case/pack.log' 2>&1 & echo `$! > '$RemoteProject/logs/l3_20260507_43case/pack.pid' && cat '$RemoteProject/logs/l3_20260507_43case/pack.pid'"
} else {
    Write-Host "Remote sync complete. Start packing with:"
    Write-Host "ssh $HostName `"cd '$RemoteRepo' && bash scripts/run_l3_43_remote.sh pack`""
}
