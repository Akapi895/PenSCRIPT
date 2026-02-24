<#
.SYNOPSIS
    Run full Strategy C curriculum training: T1 → T2 → T3 → T4 (easy → hard).
    All 8 base scenarios × 4 tiers × 3 variants = 96 tasks, trained sequentially.
    Per-task episode count from curriculum_episodes.json.

.DESCRIPTION
    Pipeline: Phase 0 (validate) → Phase 1 (sim training) → Phase 2 (domain transfer)
    → Phase 3 (PenGym fine-tune, curriculum ordered) → Phase 4 (multi-episode eval K=20)
    + θ_pengym_scratch baseline (same curriculum, same order)

    Output structure:
      outputs/strategy_c/curriculum_full/
        ├── logs/strategy_c.log          # Full console + loguru log
        ├── tensorboard/                 # TensorBoard events
        │   ├── phase1_sim/
        │   ├── phase3_pengym/
        │   └── scratch_pengym/
        ├── models/                      # Saved model checkpoints
        │   ├── phase1_unified/
        │   ├── phase3_dual/
        │   └── pengym_scratch/
        ├── strategy_c_results.json      # Full pipeline results
        └── strategy_c_eval_report.json  # Phase 4 detailed eval report

.NOTES
    Estimated time:
    - tiny/small tiers: ~2h
    - medium tiers: ~20-28h
    - Total: ~24-30h (GPU dependent)
    Grand total episodes: ~332,400
#>

$ErrorActionPreference = "Stop"

# ────────────────────────────────────────────────────────────
# Build scenario list: ORDERED by tier (T1→T2→T3→T4), then by base name
# This ensures curriculum goes from easy → hard
# ────────────────────────────────────────────────────────────
$bases = @(
    "tiny", "tiny-hard", "tiny-small", "small-linear",
    "small-honeypot", "medium-single-site", "medium", "medium-multi-site"
)
$tiers = @("T1", "T2", "T3", "T4")
$variants = @("001", "002", "003")  # training split; _000=calibration, _004+=held-out

$scenarioList = @()
foreach ($tier in $tiers) {
    foreach ($base in $bases) {
        foreach ($v in $variants) {
            $f = "data/scenarios/generated/compiled/${base}_${tier}_${v}.yml"
            if (Test-Path $f) {
                $scenarioList += $f
            }
            else {
                Write-Warning "Missing: $f"
            }
        }
    }
}

Write-Host "============================================================"
Write-Host "  Strategy C — Full Curriculum Training (T1 → T4)"
Write-Host "============================================================"
Write-Host "  Scenarios:     $($scenarioList.Count) tasks"
Write-Host "  Ordering:      T1 (easy) → T2 → T3 → T4 (expert)"
Write-Host "  Base scenarios: $($bases -join ', ')"
Write-Host "  Episode config: data/config/curriculum_episodes.json"
Write-Host "============================================================"

# ────────────────────────────────────────────────────────────
# Run
# ────────────────────────────────────────────────────────────
python run_strategy_c.py `
    --sim-scenarios data/scenarios/chain/chain-msfexp_vul-sample-6_envs-seed_0.json `
    --pengym-scenarios $scenarioList `
    --episode-config data/config/curriculum_episodes.json `
    --train-scratch `
    --step-limit 150 `
    --episodes 1000 `
    --eval-freq 5 `
    --ewc-lambda 2000 `
    --seed 42 `
    --output-dir outputs/strategy_c/curriculum_full

Write-Host "`n============================================================"
Write-Host "  Done. Results: outputs/strategy_c/curriculum_full/"
Write-Host "============================================================"
