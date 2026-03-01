#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# PenSCRIPT -- Phase 2 May A: Fisher beta Ablation (tail)
# Current config runs beta=0.7.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
elif [[ -f "venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "venv/bin/activate"
else
  echo "ERROR: venv not found (.venv/bin/activate or venv/bin/activate)"
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: Python executable not found: ${PYTHON_BIN}"
  echo "Set PYTHON_BIN or activate a virtual environment."
  exit 1
fi

SIM_SCENARIOS="data/scenarios/chain/chain-msfexp_vul-sample-6_envs-seed_0.json"
PENGYM_SCENARIOS=(
  "data/scenarios/generated/compiled/tiny_T1_000.yml"
  "data/scenarios/generated/compiled/tiny_T2_000.yml"
  "data/scenarios/generated/compiled/tiny_T3_000.yml"
  "data/scenarios/generated/compiled/tiny_T4_000.yml"
  "data/scenarios/generated/compiled/small-linear_T1_000.yml"
  "data/scenarios/generated/compiled/small-linear_T2_000.yml"
  "data/scenarios/generated/compiled/small-linear_T3_000.yml"
  "data/scenarios/generated/compiled/small-linear_T4_000.yml"
)
EPISODE_CONFIG="data/config/curriculum_episodes.json"
TRAINING_MODE="intra_topology"
TRANSFER_STRAT="conservative"
SEED=42

# Current tail setup from run_beta_ablation_tail.ps1
BETA_VALUES=(1.0)
TOTAL_RUNS="${#BETA_VALUES[@]}"
CURRENT_IDX=0
START_ALL="$(date +%s)"

echo "============================================"
echo "Fisher beta ablation (tail): ${BETA_VALUES[*]}"
echo "Seed: ${SEED} | Mode: ${TRAINING_MODE}"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

for beta in "${BETA_VALUES[@]}"; do
  CURRENT_IDX=$((CURRENT_IDX + 1))
  output_dir="outputs/ablation_beta/beta_${beta}"
  start_run="$(date +%s)"

  echo
  echo "--------------------------------------------"
  echo "[${CURRENT_IDX}/${TOTAL_RUNS}] beta=${beta}"
  echo "Output: ${output_dir}"
  echo "Start:  $(date '+%H:%M:%S')"
  echo "--------------------------------------------"

  "${PYTHON_BIN}" run_strategy_c.py \
    --sim-scenarios "${SIM_SCENARIOS}" \
    --pengym-scenarios "${PENGYM_SCENARIOS[@]}" \
    --episode-config "${EPISODE_CONFIG}" \
    --training-mode "${TRAINING_MODE}" \
    --transfer-strategy "${TRANSFER_STRAT}" \
    --fisher-beta "${beta}" \
    --train-scratch \
    --seed "${SEED}" \
    --output-dir "${output_dir}"

  elapsed_run=$(( $(date +%s) - start_run ))
  printf '\nDONE: beta=%s  Elapsed: %02d:%02d:%02d\n' \
    "${beta}" $((elapsed_run/3600)) $(((elapsed_run%3600)/60)) $((elapsed_run%60))
done

elapsed_all=$(( $(date +%s) - START_ALL ))
echo
echo "============================================"
echo "Tail beta ablation completed!"
printf 'Total elapsed: %02d:%02d:%02d\n' \
  $((elapsed_all/3600)) $(((elapsed_all%3600)/60)) $((elapsed_all%60))
echo "============================================"
echo
echo "Results:"
echo "  beta=1.0 -> outputs/ablation_beta/beta_1.0"
