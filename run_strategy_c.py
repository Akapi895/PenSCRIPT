#!/usr/bin/env python3
"""
run_strategy_c.py — CLI entry point for Strategy C dual training pipeline.

Strategy C trains a SCRIPT CRL agent on simulation first, then transfers
the policy to PenGym via controlled domain transfer and fine-tunes with
EWC constraints.

Usage
-----
Using a benchmark preset (recommended):
    python run_strategy_c.py --preset standard
    python run_strategy_c.py --preset full --train-scratch

Manual scenario specification:
    python run_strategy_c.py \\
        --sim-scenarios data/scenarios/chain/chain_1.json \\
        --pengym-scenarios data/scenarios/tiny.yml data/scenarios/small-linear.yml

List available presets and scenarios:
    python run_strategy_c.py --list-presets

Design reference: docs/strategy_C_shared_state_dual_training.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import TeeLogger
from src.pipeline.benchmark_presets import get_preset, list_presets, list_scenarios

# ── Paths ────────────────────────────────────────────────────────────
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
STRATEGY_C_DIR = OUTPUTS_DIR / "strategy_c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strategy C — Dual Training Pipeline (Sim → Transfer → PenGym)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Benchmark presets ─────────────────────────────────────────────
    parser.add_argument(
        "--preset", type=str, default=None,
        choices=["quick", "standard", "full", "medium"],
        help="Use a benchmark preset (overrides --sim/--pengym-scenarios, "
             "--episodes, --step-limit, --eval-freq). "
             "Choices: quick | standard | full | medium.",
    )
    parser.add_argument(
        "--list-presets", action="store_true",
        help="List available presets and scenarios, then exit.",
    )

    # ── Scenarios ────────────────────────────────────────────────────
    parser.add_argument(
        "--sim-scenarios", nargs="+", default=None,
        help="Simulation scenario JSON files for Phase 1 training. "
             "(Not needed when using --preset.)",
    )
    parser.add_argument(
        "--pengym-scenarios", nargs="+", default=None,
        help="PenGym scenario YAML files for Phase 3 fine-tuning. "
             "(Not needed when using --preset.)",
    )

    # ── Transfer ─────────────────────────────────────────────────────
    parser.add_argument(
        "--transfer-strategy", type=str, default="conservative",
        choices=["aggressive", "conservative", "cautious"],
        help="Domain transfer strategy (default: conservative).",
    )
    parser.add_argument(
        "--fisher-beta", type=float, default=0.3,
        help="Fisher discount factor β for conservative/cautious transfer.",
    )
    parser.add_argument(
        "--lr-factor", type=float, default=0.1,
        help="Learning rate multiplier after domain transfer.",
    )
    parser.add_argument(
        "--warmup-episodes", type=int, default=10,
        help="Random-rollout episodes for normaliser warmup on PenGym.",
    )

    # ── Training ─────────────────────────────────────────────────────
    parser.add_argument(
        "--episodes", type=int, default=500,
        help="Training episodes per task.",
    )
    parser.add_argument(
        "--step-limit", type=int, default=100,
        help="Max steps per episode.",
    )
    parser.add_argument(
        "--eval-freq", type=int, default=5,
        help="Evaluate every N episodes during training.",
    )
    parser.add_argument(
        "--ewc-lambda", type=float, default=2000,
        help="EWC regularisation strength.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed.",
    )

    # ── Optional phases ──────────────────────────────────────────────
    parser.add_argument(
        "--skip-phase0", action="store_true",
        help="Skip Phase 0 validation checks.",
    )
    parser.add_argument(
        "--train-scratch", action="store_true",
        help="Also train θ_pengym_scratch for full 4-agent comparison.",
    )

    # ── Output ───────────────────────────────────────────────────────
    parser.add_argument(
        "--output-dir", type=str, default=str(STRATEGY_C_DIR),
        help="Output directory for logs, models, and results.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Handle --list-presets ────────────────────────────────────────
    if args.list_presets:
        print(list_presets())
        print()
        print(list_scenarios())
        sys.exit(0)

    # ── Resolve preset or manual scenarios ───────────────────────────
    preset_info = None
    if args.preset:
        preset_info = get_preset(args.preset)
        sim_scenarios    = preset_info["sim_scenarios"]
        pengym_scenarios = preset_info["pengym_scenarios"]
        # Preset provides sensible defaults; CLI flags can still override
        episodes   = args.episodes   if args.episodes   != 500 else preset_info["train_eps"]
        step_limit = args.step_limit if args.step_limit != 100 else preset_info["step_limit"]
        eval_freq  = args.eval_freq  if args.eval_freq  != 5   else preset_info["eval_freq"]
    else:
        # Manual mode — require both scenario lists
        if not args.sim_scenarios or not args.pengym_scenarios:
            print("ERROR: Provide --preset OR both --sim-scenarios and "
                  "--pengym-scenarios.  Use --list-presets to see options.")
            sys.exit(1)
        sim_scenarios    = args.sim_scenarios
        pengym_scenarios = args.pengym_scenarios
        episodes   = args.episodes
        step_limit = args.step_limit
        eval_freq  = args.eval_freq

    # Set up logging
    log_dir = Path(args.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    tee = TeeLogger(str(log_dir / "strategy_c.log"))

    print("=" * 70)
    print("  Strategy C — Dual Training Pipeline")
    print("=" * 70)
    if preset_info:
        print(f"  Preset:           {preset_info['name']} — {preset_info['description']}")
    print(f"  Sim scenarios:    {sim_scenarios}")
    print(f"  PenGym scenarios: {pengym_scenarios}")
    print(f"  Transfer:         {args.transfer_strategy} (β={args.fisher_beta}, LR×{args.lr_factor})")
    print(f"  Training:         {episodes} eps, {step_limit} steps, EWC λ={args.ewc_lambda}")
    print(f"  Seed:             {args.seed}")
    print(f"  Output:           {args.output_dir}")
    if preset_info:
        print(f"  Optimal rewards:  {preset_info['optimal_rewards']}")
    print("=" * 70)

    from src.training.dual_trainer import DualTrainer

    # Build DualTrainer
    trainer = DualTrainer(
        sim_scenarios=sim_scenarios,
        pengym_scenarios=pengym_scenarios,
        ppo_kwargs={
            "train_eps": episodes,
            "step_limit": step_limit,
            "eval_step_limit": step_limit,
        },
        script_kwargs={
            "ewc_lambda": args.ewc_lambda,
            "fisher_discount_beta": args.fisher_beta,
            "transfer_lr_factor": args.lr_factor,
            "norm_warmup_episodes": args.warmup_episodes,
            "transfer_strategy": args.transfer_strategy,
        },
        seed=args.seed,
        output_dir=args.output_dir,
    )

    t0 = time.time()

    # Run full pipeline
    results = trainer.run_full_pipeline(
        skip_phase0=args.skip_phase0,
        eval_freq=eval_freq,
    )

    # Optionally train scratch baseline
    if args.train_scratch:
        print("\n" + "=" * 60)
        print("  Training θ_pengym_scratch baseline...")
        print("=" * 60)
        scratch_results = trainer.train_pengym_scratch(eval_freq=eval_freq)
        results["scratch"] = scratch_results

        # Re-run Phase 4 with the scratch agent available
        print("\n  Re-running Phase 4 with all 4 agents...")
        results["phase4"] = trainer.phase4_evaluation()

    elapsed = time.time() - t0

    # Print summary
    print("\n" + "=" * 70)
    print("  Strategy C — Pipeline Complete")
    print("=" * 70)
    print(f"  Total time: {elapsed:.1f}s")

    if "phase4" in results:
        phase4 = results["phase4"]
        for agent_name, data in phase4.get("agents", {}).items():
            sr = data.get("success_rate", "N/A")
            if isinstance(sr, float):
                sr = f"{sr:.1%}"
            print(f"    {agent_name}: SR={sr}")
        transfer = phase4.get("transfer_metrics", {})
        if transfer:
            ft = transfer.get("forward_transfer")
            bt = transfer.get("backward_transfer")
            print(f"  Forward transfer:  {ft:+.2%}" if ft is not None else "")
            print(f"  Backward transfer: {bt:+.2%}" if bt is not None else "")

    # Save final results
    results_path = Path(args.output_dir) / "strategy_c_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved → {results_path}")

    tee.close()


if __name__ == "__main__":
    main()
