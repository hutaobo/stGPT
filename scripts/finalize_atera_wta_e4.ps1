param(
    [string]$Stgpt = ".\.venv\Scripts\stgpt.exe",
    [string]$Suite = "configs\evidence\atera_wta_v1_long_run.yaml",
    [string]$OutputRoot = "outputs\evidence\atera_wta_v1_long_run_final",
    [string]$Bundle = "outputs\evidence\atera_wta_v1_final_evidence.tar.gz",
    [int]$EvalBatchSize = 32,
    [string]$Device = "cuda",
    [int]$ExpectedStep = 10000,
    [switch]$SkipAblation
)

$ErrorActionPreference = "Stop"

$fullRuns = @(
    @{Name="breast_full_m6_contour_store_10k"; Config="configs\pilots\atera_wta_v1_long_run\breast_full_m6_contour_store_10k.yaml"; RunDir="outputs\pilot_runs\atera_wta_v1_long_run\breast_full_m6_contour_store_10k"},
    @{Name="cervical_full_m6_contour_store_10k"; Config="configs\pilots\atera_wta_v1_long_run\cervical_full_m6_contour_store_10k.yaml"; RunDir="outputs\pilot_runs\atera_wta_v1_long_run\cervical_full_m6_contour_store_10k"}
)

function Get-MaxStep($runDir) {
    $metricsPath = Join-Path $runDir "train\metrics.json"
    if (-not (Test-Path $metricsPath)) { return 0 }
    $metrics = Get-Content $metricsPath -Raw | ConvertFrom-Json
    if ($null -eq $metrics) { return 0 }
    $steps = @($metrics | ForEach-Object { $_.step })
    if ($steps.Count -eq 0) { return 0 }
    return ($steps | Measure-Object -Maximum).Maximum
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

foreach ($run in $fullRuns) {
    $maxStep = Get-MaxStep $run.RunDir
    if ($maxStep -lt $ExpectedStep) {
        Write-Warning "$($run.Name) has max step $maxStep, below expected $ExpectedStep. Finalization will still write available artifacts."
    }
}

& $Stgpt evidence-summary --suite $Suite --output $OutputRoot

foreach ($run in $fullRuns) {
    $checkpoint = Join-Path $run.RunDir "train\checkpoints\best_alignment.pt"
    $runOutput = Join-Path $OutputRoot $run.Name
    New-Item -ItemType Directory -Force -Path $runOutput | Out-Null
    if (Test-Path $checkpoint) {
        & $Stgpt check-contract --checkpoint $checkpoint --config $run.Config --run-dir $run.RunDir --output (Join-Path $runOutput "contract_check.json")
    } else {
        Write-Warning "$($run.Name) is missing best_alignment.pt"
    }
    if (Test-Path (Join-Path $run.RunDir "spatho_export\contour_evidence_chains.jsonl")) {
        $failureOutput = Join-Path $runOutput "failure_gallery"
        & $Stgpt failure-gallery --run-dir $run.RunDir --output $failureOutput --max-items 100
        $targets = Join-Path $failureOutput "ablation_targets.json"
        if ((-not $SkipAblation) -and (Test-Path $checkpoint) -and (Test-Path $targets)) {
            & $Stgpt ablate --checkpoint $checkpoint --config $run.Config --targets $targets --output (Join-Path $runOutput "ablation") --batch-size $EvalBatchSize --device $Device
        }
    } else {
        Write-Warning "$($run.Name) is missing Spatho evidence chains"
    }
}

& $Stgpt latent-manifold --suite $Suite --output (Join-Path $OutputRoot "latent_manifold") --reducer auto --max-html-points 5000

$bundlePath = Resolve-Path -Path (Split-Path $Bundle -Parent) -ErrorAction SilentlyContinue
if ($null -eq $bundlePath) {
    New-Item -ItemType Directory -Force -Path (Split-Path $Bundle -Parent) | Out-Null
}
if (Test-Path $Bundle) {
    Remove-Item -LiteralPath $Bundle -Force
}

if (Get-Command tar -ErrorAction SilentlyContinue) {
    $parent = Split-Path $OutputRoot -Parent
    $leaf = Split-Path $OutputRoot -Leaf
    tar -czf $Bundle -C $parent $leaf
} else {
    $zip = [System.IO.Path]::ChangeExtension($Bundle, ".zip")
    if (Test-Path $zip) {
        Remove-Item -LiteralPath $zip -Force
    }
    Compress-Archive -Path (Join-Path $OutputRoot "*") -DestinationPath $zip
    Write-Warning "tar was not available; wrote $zip instead of $Bundle"
}
