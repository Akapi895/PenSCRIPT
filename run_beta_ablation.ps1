# ============================================================
# PenSCRIPT — Phase 2 Máy A: Fisher β Ablation (R13–R17)
# Chạy intra_topology với 5 giá trị β, seed=42
# β=0.3 đã có từ R5 (outputs/multiseed/seed_42) → skip
# Usage:  .\run_beta_ablation.ps1
# ============================================================

$ErrorActionPreference = "Stop"

# ── Activate virtual environment ──
$VenvActivate = Join-Path $PSScriptRoot "venv\Scripts\Activate.ps1"
if (Test-Path $VenvActivate) {
    Write-Host " Activating venv: $VenvActivate" -ForegroundColor Magenta
    & $VenvActivate
} else {
    Write-Host " ERROR: venv not found at $VenvActivate" -ForegroundColor Red
    exit 1
}

# ── Common arguments ──
$SimScenarios   = "data/scenarios/chain/chain-msfexp_vul-sample-6_envs-seed_0.json"
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
$EpisodeConfig  = "data/config/curriculum_episodes.json"
$TrainingMode   = "intra_topology"
$TransferStrat  = "conservative"
$Seed           = 42

# ── β values to test (skip 0.3 — already in outputs/multiseed/seed_42) ──
$BetaValues = @(0.0, 0.1, 0.5, 0.7, 1.0)

# ── Tracking ──
$TotalRuns   = $BetaValues.Count
$CurrentIdx  = 0
$StartTimeAll = Get-Date

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " Fisher beta ablation: $($BetaValues -join ', ')" -ForegroundColor Cyan
Write-Host " Seed: $Seed  |  Mode: $TrainingMode" -ForegroundColor Cyan
Write-Host " Note: beta=0.3 reused from outputs/multiseed/seed_42" -ForegroundColor Cyan
Write-Host " Started at: $($StartTimeAll.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

foreach ($beta in $BetaValues) {
    $CurrentIdx++
    $BetaStr   = $beta.ToString("0.0")
    $OutputDir = "outputs/ablation_beta/beta_$BetaStr"
    $StartTime = Get-Date

    Write-Host ""
    Write-Host "--------------------------------------------" -ForegroundColor Yellow
    Write-Host " [$CurrentIdx/$TotalRuns] beta=$BetaStr" -ForegroundColor Yellow
    Write-Host " Output: $OutputDir" -ForegroundColor Yellow
    Write-Host " Start:  $($StartTime.ToString('HH:mm:ss'))" -ForegroundColor Yellow
    Write-Host "--------------------------------------------" -ForegroundColor Yellow

    # Build and run the command
    $cmdArgs = @(
        "run_strategy_c.py",
        "--sim-scenarios", $SimScenarios,
        "--pengym-scenarios") + $PenGymScenarios + @(
        "--episode-config", $EpisodeConfig,
        "--training-mode", $TrainingMode,
        "--transfer-strategy", $TransferStrat,
        "--fisher-beta", $BetaStr,
        "--train-scratch",
        "--seed", $Seed,
        "--output-dir", $OutputDir
    )

    python @cmdArgs

    $ExitCode = $LASTEXITCODE
    $Elapsed  = (Get-Date) - $StartTime

    if ($ExitCode -ne 0) {
        Write-Host ""
        Write-Host " FAILED: beta=$BetaStr exited with code $ExitCode" -ForegroundColor Red
        Write-Host " Elapsed: $($Elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Red
        Write-Host " Stopping batch." -ForegroundColor Red
        exit $ExitCode
    }

    Write-Host ""
    Write-Host " DONE: beta=$BetaStr  Elapsed: $($Elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Green
}

$TotalElapsed = (Get-Date) - $StartTimeAll
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host " All beta ablation runs completed!" -ForegroundColor Cyan
Write-Host " Total elapsed: $($TotalElapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host " Results summary:" -ForegroundColor Green
Write-Host "   beta=0.0 -> outputs/ablation_beta/beta_0.0" -ForegroundColor Green
Write-Host "   beta=0.1 -> outputs/ablation_beta/beta_0.1" -ForegroundColor Green
Write-Host "   beta=0.3 -> outputs/multiseed/seed_42 (reuse)" -ForegroundColor Green
Write-Host "   beta=0.5 -> outputs/ablation_beta/beta_0.5" -ForegroundColor Green
Write-Host "   beta=0.7 -> outputs/ablation_beta/beta_0.7" -ForegroundColor Green
Write-Host "   beta=1.0 -> outputs/ablation_beta/beta_1.0" -ForegroundColor Green
