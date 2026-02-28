# ============================================================
# PenSCRIPT -- Phase 2 Máy B: Ablation (R18 + R19)
#   R18: No-canonicalization ablation (CẦN SỬA CODE TRƯỚC)
#   R19: Fine-tune only -- no EWC (ewc-lambda=0)
# Usage:  .\run_ablation_machineB.ps1
#         .\run_ablation_machineB.ps1 -SkipR18   # skip nếu chưa sửa code
# ============================================================

param(
    [switch]$SkipR18
)

$ErrorActionPreference = "Stop"

# -- Activate virtual environment --
$VenvActivate = Join-Path $PSScriptRoot "venv\Scripts\Activate.ps1"
if (Test-Path $VenvActivate) {
    Write-Host " Activating venv: $VenvActivate" -ForegroundColor Magenta
    & $VenvActivate
} else {
    Write-Host " ERROR: venv not found at $VenvActivate" -ForegroundColor Red
    exit 1
}

# -- Common arguments --
$SimScenarios    = "data/scenarios/chain/chain-msfexp_vul-sample-6_envs-seed_0.json"
$PenGymScenarios = @(
    "data/scenarios/generated/compiled/tiny_T1_000.yml",
    "data/scenarios/generated/compiled/tiny_T2_000.yml",
    "data/scenarios/generated/compiled/tiny_T3_000.yml",
    "data/scenarios/generated/compiled/tiny_T4_000.yml",
    "data/scenarios/generated/compiled/small-linear_T1_000.yml",
    "data/scenarios/generated/compiled/small-linear_T2_000.yml",
    "data/scenarios/generated/compiled/small-linear_T3_000.yml",
    "data/scenarios/generated/compiled/small-linear_T4_000.yml"
)
$EpisodeConfig   = "data/config/curriculum_episodes.json"
$TrainingMode    = "intra_topology"
$TransferStrat   = "conservative"
$Seed            = 42

# -- Define runs --
$Runs = @(
    @{
        Id        = "R18"
        Name      = "No-canonicalization ablation"
        OutputDir = "outputs/ablation_canon/no_canon"
        ExtraArgs = @("--no-canonicalization")
        FisherBeta = "0.3"
        EwcLambda  = "2000"
        NeedsCodeChange = $true
    },
    @{
        Id        = "R19"
        Name      = "Fine-tune only (no EWC)"
        OutputDir = "outputs/ablation_crl/finetune_only"
        ExtraArgs = @()
        FisherBeta = "0.3"
        EwcLambda  = "0"
        NeedsCodeChange = $false
    }
)

# -- Pre-check R18 --
if (-not $SkipR18) {
    $helpOutput = python run_strategy_c.py --help 2>&1 | Out-String
    if ($helpOutput -notmatch "no-canonicalization") {
        Write-Host ""
        Write-Host " WARNING: --no-canonicalization flag NOT found in run_strategy_c.py" -ForegroundColor Red
        Write-Host " R18 requires code changes first (see experiment_roadmap.md)." -ForegroundColor Red
        Write-Host " Options:" -ForegroundColor Yellow
        Write-Host "   1. Implement the code change, then re-run this script" -ForegroundColor Yellow
        Write-Host "   2. Run with -SkipR18 to skip R18 and only run R19:" -ForegroundColor Yellow
        Write-Host "      .\run_ablation_machineB.ps1 -SkipR18" -ForegroundColor Yellow
        Write-Host ""
        exit 1
    }
}

# -- Tracking --
$StartTimeAll = Get-Date
$TotalRuns    = if ($SkipR18) { 1 } else { $Runs.Count }
$CurrentIdx   = 0

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " Machine B -- Phase 2 Ablation" -ForegroundColor Cyan
Write-Host " Seed: $Seed  |  Mode: $TrainingMode" -ForegroundColor Cyan
if ($SkipR18) {
    Write-Host " Skipping R18 (no-canonicalization)" -ForegroundColor Yellow
}
Write-Host " Started at: $($StartTimeAll.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

foreach ($run in $Runs) {
    # Skip R18 if requested
    if ($SkipR18 -and $run.Id -eq "R18") {
        continue
    }

    $CurrentIdx++
    $StartTime = Get-Date

    Write-Host ""
    Write-Host "--------------------------------------------" -ForegroundColor Yellow
    Write-Host " [$CurrentIdx/$TotalRuns] $($run.Id): $($run.Name)" -ForegroundColor Yellow
    Write-Host " Output: $($run.OutputDir)" -ForegroundColor Yellow
    Write-Host " EWC lambda=$($run.EwcLambda), Fisher beta=$($run.FisherBeta)" -ForegroundColor Yellow
    Write-Host " Start:  $($StartTime.ToString('HH:mm:ss'))" -ForegroundColor Yellow
    Write-Host "--------------------------------------------" -ForegroundColor Yellow

    # Build command
    $cmdArgs = @(
        "run_strategy_c.py",
        "--sim-scenarios", $SimScenarios,
        "--pengym-scenarios") + $PenGymScenarios + @(
        "--episode-config", $EpisodeConfig,
        "--training-mode", $TrainingMode,
        "--transfer-strategy", $TransferStrat,
        "--fisher-beta", $run.FisherBeta,
        "--ewc-lambda", $run.EwcLambda,
        "--train-scratch",
        "--seed", $Seed,
        "--output-dir", $run.OutputDir
    ) + $run.ExtraArgs

    python @cmdArgs

    $ExitCode = $LASTEXITCODE
    $Elapsed  = (Get-Date) - $StartTime

    if ($ExitCode -ne 0) {
        Write-Host ""
        Write-Host " FAILED: $($run.Id) exited with code $ExitCode" -ForegroundColor Red
        Write-Host " Elapsed: $($Elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Red
        Write-Host " Stopping batch." -ForegroundColor Red
        exit $ExitCode
    }

    Write-Host ""
    Write-Host " DONE: $($run.Id)  Elapsed: $($Elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Green
}

$TotalElapsed = (Get-Date) - $StartTimeAll
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host " Machine B Phase 2 completed!" -ForegroundColor Cyan
Write-Host " Total elapsed: $($TotalElapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host " Results:" -ForegroundColor Green
if (-not $SkipR18) {
    Write-Host "   R18 no-canon      -> outputs/ablation_canon/no_canon" -ForegroundColor Green
}
Write-Host "   R19 finetune-only -> outputs/ablation_crl/finetune_only" -ForegroundColor Green
