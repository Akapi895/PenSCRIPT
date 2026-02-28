# ============================================================
# PenSCRIPT — Auto-run cross_topology seeds: 2 → 3 → 42
# Usage:  .\run_cross_seeds.ps1
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
$SimScenarios = "data/scenarios/chain/chain-msfexp_vul-sample-6_envs-seed_0.json"
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
$EpisodeConfig = "data/config/curriculum_episodes.json"
$TrainingMode = "cross_topology"
$TransferStrat = "conservative"
$FisherBeta = "0.3"

# ── Seeds to run ──
$Seeds = @(3, 42)

# ── Tracking ──
$TotalSeeds = $Seeds.Count
$CurrentIdx = 0
$StartTimeAll = Get-Date

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " Cross-topology batch: seeds $($Seeds -join ', ')" -ForegroundColor Cyan
Write-Host " Started at: $($StartTimeAll.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

foreach ($seed in $Seeds) {
    $CurrentIdx++
    $OutputDir = "outputs/multiseed_cross/seed_$seed"
    $StartTime = Get-Date

    Write-Host ""
    Write-Host "--------------------------------------------" -ForegroundColor Yellow
    Write-Host " [$CurrentIdx/$TotalSeeds] Seed=$seed" -ForegroundColor Yellow
    Write-Host " Output: $OutputDir" -ForegroundColor Yellow
    Write-Host " Start:  $($StartTime.ToString('HH:mm:ss'))" -ForegroundColor Yellow
    Write-Host "--------------------------------------------" -ForegroundColor Yellow

    # Remove incomplete output for seed 2 (interrupted run)
    if ((Test-Path $OutputDir) -and $seed -eq 2) {
        Write-Host " Removing incomplete previous run for seed $seed..." -ForegroundColor Red
        Remove-Item -Recurse -Force $OutputDir
    }

    # Build and run the command
    $args = @(
        "run_strategy_c.py",
        "--sim-scenarios", $SimScenarios,
        "--pengym-scenarios") + $PenGymScenarios + @(
        "--episode-config", $EpisodeConfig,
        "--training-mode", $TrainingMode,
        "--transfer-strategy", $TransferStrat,
        "--fisher-beta", $FisherBeta,
        "--seed", $seed,
        "--output-dir", $OutputDir
    )

    python @args

    $ExitCode = $LASTEXITCODE
    $Elapsed = (Get-Date) - $StartTime

    if ($ExitCode -ne 0) {
        Write-Host ""
        Write-Host " FAILED: Seed=$seed exited with code $ExitCode" -ForegroundColor Red
        Write-Host " Elapsed: $($Elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Red
        Write-Host " Stopping batch." -ForegroundColor Red
        exit $ExitCode
    }

    Write-Host ""
    Write-Host " DONE: Seed=$seed  Elapsed: $($Elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Green
}

$TotalElapsed = (Get-Date) - $StartTimeAll
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host " All seeds completed!" -ForegroundColor Cyan
Write-Host " Total elapsed: $($TotalElapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
