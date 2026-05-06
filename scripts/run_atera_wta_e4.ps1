param(
    [ValidateSet("deep", "grid", "all")]
    [string]$Phase = "all",
    [int[]]$Gpus = @(0, 1, 2, 3),
    [string]$Stgpt = ".\.venv\Scripts\stgpt.exe",
    [int]$EvalBatchSize = 32,
    [switch]$Finalize
)

$ErrorActionPreference = "Stop"

$runs = @(
    @{Stage="deep"; Role="best_loss"; Config="configs\pilots\atera_wta_v1_long_run\breast_gene_spatial_10k_contour_unit.yaml"; RunDir="outputs\pilot_runs\atera_wta_v1_long_run\breast_gene_spatial_10k_contour_unit"},
    @{Stage="deep"; Role="best_loss"; Config="configs\pilots\atera_wta_v1_long_run\cervical_gene_spatial_10k_contour_unit.yaml"; RunDir="outputs\pilot_runs\atera_wta_v1_long_run\cervical_gene_spatial_10k_contour_unit"},
    @{Stage="deep"; Role="best_alignment"; Config="configs\pilots\atera_wta_v1_long_run\breast_full_m6_contour_store_10k.yaml"; RunDir="outputs\pilot_runs\atera_wta_v1_long_run\breast_full_m6_contour_store_10k"},
    @{Stage="deep"; Role="best_alignment"; Config="configs\pilots\atera_wta_v1_long_run\cervical_full_m6_contour_store_10k.yaml"; RunDir="outputs\pilot_runs\atera_wta_v1_long_run\cervical_full_m6_contour_store_10k"},
    @{Stage="grid"; Role="best_alignment"; Config="configs\pilots\atera_wta_v1_long_run\breast_full_m6_lambda_0_01.yaml"; RunDir="outputs\pilot_runs\atera_wta_v1_long_run\breast_full_m6_lambda_0_01"},
    @{Stage="grid"; Role="best_alignment"; Config="configs\pilots\atera_wta_v1_long_run\breast_full_m6_lambda_0_05.yaml"; RunDir="outputs\pilot_runs\atera_wta_v1_long_run\breast_full_m6_lambda_0_05"},
    @{Stage="grid"; Role="best_alignment"; Config="configs\pilots\atera_wta_v1_long_run\breast_full_m6_lambda_0_1.yaml"; RunDir="outputs\pilot_runs\atera_wta_v1_long_run\breast_full_m6_lambda_0_1"},
    @{Stage="grid"; Role="best_alignment"; Config="configs\pilots\atera_wta_v1_long_run\breast_full_m6_lambda_0_5.yaml"; RunDir="outputs\pilot_runs\atera_wta_v1_long_run\breast_full_m6_lambda_0_5"},
    @{Stage="grid"; Role="best_alignment"; Config="configs\pilots\atera_wta_v1_long_run\breast_full_m6_lambda_1_0.yaml"; RunDir="outputs\pilot_runs\atera_wta_v1_long_run\breast_full_m6_lambda_1_0"},
    @{Stage="grid"; Role="best_alignment"; Config="configs\pilots\atera_wta_v1_long_run\breast_full_m6_lambda_2_0.yaml"; RunDir="outputs\pilot_runs\atera_wta_v1_long_run\breast_full_m6_lambda_2_0"},
    @{Stage="grid"; Role="best_alignment"; Config="configs\pilots\atera_wta_v1_long_run\cervical_full_m6_lambda_0_01.yaml"; RunDir="outputs\pilot_runs\atera_wta_v1_long_run\cervical_full_m6_lambda_0_01"},
    @{Stage="grid"; Role="best_alignment"; Config="configs\pilots\atera_wta_v1_long_run\cervical_full_m6_lambda_0_05.yaml"; RunDir="outputs\pilot_runs\atera_wta_v1_long_run\cervical_full_m6_lambda_0_05"},
    @{Stage="grid"; Role="best_alignment"; Config="configs\pilots\atera_wta_v1_long_run\cervical_full_m6_lambda_0_1.yaml"; RunDir="outputs\pilot_runs\atera_wta_v1_long_run\cervical_full_m6_lambda_0_1"},
    @{Stage="grid"; Role="best_alignment"; Config="configs\pilots\atera_wta_v1_long_run\cervical_full_m6_lambda_0_5.yaml"; RunDir="outputs\pilot_runs\atera_wta_v1_long_run\cervical_full_m6_lambda_0_5"},
    @{Stage="grid"; Role="best_alignment"; Config="configs\pilots\atera_wta_v1_long_run\cervical_full_m6_lambda_1_0.yaml"; RunDir="outputs\pilot_runs\atera_wta_v1_long_run\cervical_full_m6_lambda_1_0"},
    @{Stage="grid"; Role="best_alignment"; Config="configs\pilots\atera_wta_v1_long_run\cervical_full_m6_lambda_2_0.yaml"; RunDir="outputs\pilot_runs\atera_wta_v1_long_run\cervical_full_m6_lambda_2_0"}
) | Where-Object { $Phase -eq "all" -or $_.Stage -eq $Phase }

function Get-ResumeCheckpoint($runDir) {
    $last = Join-Path $runDir "train\checkpoints\last.pt"
    if (Test-Path $last) { return $null }
    $checkpointDir = Join-Path $runDir "train\checkpoints"
    if (-not (Test-Path $checkpointDir)) { return $null }
    $step = Get-ChildItem $checkpointDir -Filter "step_*.pt" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $step) { return $null }
    return $step.FullName
}

function Get-RoleCheckpoint($run) {
    $name = if ($run.Role -eq "best_loss") { "best.pt" } else { "best_alignment.pt" }
    return Join-Path $run.RunDir "train\checkpoints\$name"
}

function Invoke-TrainingBatch($batch) {
    $jobs = @()
    for ($i = 0; $i -lt $batch.Count; $i++) {
        $run = $batch[$i]
        $gpu = $Gpus[$i % $Gpus.Count]
        $resume = Get-ResumeCheckpoint $run.RunDir
        $jobs += Start-Job -Name $run.RunDir -ArgumentList $run,$gpu,$resume,$Stgpt -ScriptBlock {
            param($run, $gpu, $resume, $stgpt)
            $ErrorActionPreference = "Stop"
            $env:CUDA_VISIBLE_DEVICES = "$gpu"
            if ($resume) {
                & $stgpt train --config $run.Config --resume $resume
            } else {
                & $stgpt train --config $run.Config
            }
        }
    }
    $jobs | Wait-Job | Receive-Job
    foreach ($job in $jobs) {
        if ($job.State -ne "Completed") {
            throw "Training job failed: $($job.Name)"
        }
    }
}

for ($offset = 0; $offset -lt $runs.Count; $offset += $Gpus.Count) {
    $batch = @($runs[$offset..([Math]::Min($offset + $Gpus.Count - 1, $runs.Count - 1))])
    Invoke-TrainingBatch $batch
    foreach ($run in $batch) {
        $checkpoint = Get-RoleCheckpoint $run
        $splits = Join-Path $run.RunDir "qc\splits.csv"
        & $Stgpt evaluate --checkpoint $checkpoint --config $run.Config --splits $splits --output (Join-Path $run.RunDir "evaluation") --batch-size $EvalBatchSize --device cuda
        & $Stgpt export-spatho --checkpoint $checkpoint --config $run.Config --output (Join-Path $run.RunDir "spatho_export") --batch-size $EvalBatchSize --device cuda
    }
    & $Stgpt evidence-summary --suite configs\evidence\atera_wta_v1_long_run.yaml --output outputs\evidence\atera_wta_v1_long_run
}

if ($Finalize) {
    & powershell -ExecutionPolicy Bypass -File scripts\finalize_atera_wta_e4.ps1 -Stgpt $Stgpt -EvalBatchSize $EvalBatchSize
}
