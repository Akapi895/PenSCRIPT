#!/usr/bin/env python3
"""
Strategy A: Sim-to-Real Transfer — Entry Point

Evaluate pre-trained SCRIPT agent on PenGym NASim environment.

Usage:
  # Zero-shot evaluation on tiny scenario
  python run_strategy_a.py --scenario tiny.yml --episodes 20

  # Evaluate with specific model checkpoint
  python run_strategy_a.py --scenario tiny.yml --model-dir outputs_baseline_sim/models_baseline_sim/chain/chain-msfexp_vul-sample-6_envs-seed_0

  # Verbose evaluation with more episodes
  python run_strategy_a.py --scenario small-linear.yml --episodes 50 --max-steps 200 --verbose

  # Stochastic evaluation (non-deterministic action selection)
  python run_strategy_a.py --scenario tiny.yml --no-deterministic

  # Full evaluation on medium scenario
  python run_strategy_a.py --scenario medium.yml --episodes 30 --max-steps 300
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path

# ============================================================
# Path Setup
# ============================================================
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Also ensure PenGym is importable
PENGYM_ROOT = PROJECT_ROOT.parent / "PenGym"
if PENGYM_ROOT.exists():
    sys.path.insert(0, str(PENGYM_ROOT))

# ============================================================
# Imports (after path setup)
# ============================================================
from src import OUTPUT_DIR, LOGS_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description="Strategy A: Sim-to-Real Transfer Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_strategy_a.py --scenario tiny.yml
  python run_strategy_a.py --scenario tiny.yml --episodes 50 --verbose
  python run_strategy_a.py --scenario small-linear.yml --max-steps 200
        """
    )

    # Required / key arguments
    parser.add_argument(
        "--scenario",
        type=str,
        default="tiny.yml",
        help="PenGym NASim scenario file name (in PenGym/database/scenarios/). Default: tiny.yml"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path to pre-trained model directory (containing PPO-actor.pt, etc.). "
             "Default: outputs/models_baseline_sim/chain/chain-msfexp_vul-sample-6_envs-seed_0"
    )

    # Evaluation parameters
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of evaluation episodes (default: 20)")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Max steps per episode (default: 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--no-deterministic", action="store_true",
                        help="Use stochastic action selection instead of deterministic")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Print per-episode details (default: True)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-episode output")

    # Output
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results. Default: outputs/logs/strategy_a/")

    # Baseline comparison
    parser.add_argument("--baseline", type=str, default=None,
                        help="Path to simulation baseline experiment_summary.json")

    # Observation mode
    parser.add_argument("--partially-obs", action="store_true",
                        help="Use partially observable mode (default: fully observable)")

    return parser.parse_args()


def resolve_scenario_path(scenario_name: str) -> Path:
    """Find the PenGym scenario file."""
    # Try PenGym database first
    candidates = [
        PENGYM_ROOT / "database" / "scenarios" / scenario_name,
        PROJECT_ROOT / "data" / "scenarios" / scenario_name,
        Path(scenario_name),  # absolute path
    ]
    for p in candidates:
        if p.exists():
            return p

    # Try without extension
    for ext in ['.yml', '.yaml']:
        for base in [PENGYM_ROOT / "database" / "scenarios",
                     PROJECT_ROOT / "data" / "scenarios"]:
            p = base / f"{scenario_name}{ext}"
            if p.exists():
                return p

    print(f"Error: Scenario '{scenario_name}' not found. Searched:")
    for c in candidates:
        print(f"  - {c}")
    sys.exit(1)


def resolve_model_dir(model_dir_arg: str) -> Path:
    """Find the pre-trained model directory."""
    if model_dir_arg:
        p = Path(model_dir_arg)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.exists():
            return p
        print(f"Error: Model directory not found: {p}")
        sys.exit(1)

    # Default: look for baseline model
    defaults = [
        PROJECT_ROOT / "outputs" / "models_baseline_sim" / "chain" / "chain-msfexp_vul-sample-6_envs-seed_0",
        PROJECT_ROOT / "outputs" / "models" / "chain" / "chain-msfexp_vul-sample-6_envs-seed_0",
    ]
    for d in defaults:
        if d.exists() and (d / "PPO-actor.pt").exists():
            return d

    print("Error: No pre-trained model found. Please specify --model-dir")
    print("Expected files: PPO-actor.pt, PPO-critic.pt in model directory")
    for d in defaults:
        print(f"  Searched: {d}")
    sys.exit(1)


def resolve_baseline_path(baseline_arg: str) -> Path:
    """Find baseline experiment summary."""
    if baseline_arg:
        p = Path(baseline_arg)
        return p if p.exists() else None

    defaults = [
        PROJECT_ROOT / "outputs" / "logs_baseline_sim" / "chain" / "baseline_standard_6targets_seed42" / "experiment_summary.json",
    ]
    for p in defaults:
        if p.exists():
            return p
    return None


def main():
    args = parse_args()

    # ============================================================
    # Resolve paths
    # ============================================================
    scenario_path = resolve_scenario_path(args.scenario)
    model_dir = resolve_model_dir(args.model_dir)
    baseline_path = resolve_baseline_path(args.baseline)
    verbose = args.verbose and not args.quiet
    deterministic = not args.no_deterministic
    fully_obs = not args.partially_obs

    output_dir = Path(args.output_dir) if args.output_dir else \
        LOGS_DIR / "strategy_a" / "zero_shot_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Print configuration
    # ============================================================
    scenario_name = scenario_path.stem
    print("\n" + "=" * 70)
    print("   STRATEGY A: Sim-to-Real Transfer Evaluation")
    print("=" * 70)
    print(f"  Scenario:       {scenario_path}")
    print(f"  Model dir:      {model_dir}")
    print(f"  Baseline:       {baseline_path or 'Not provided'}")
    print(f"  Episodes:       {args.episodes}")
    print(f"  Max steps:      {args.max_steps}")
    print(f"  Seed:           {args.seed}")
    print(f"  Deterministic:  {deterministic}")
    print(f"  Fully obs:      {fully_obs}")
    print(f"  Output dir:     {output_dir}")
    print("=" * 70)

    # ============================================================
    # Run evaluation
    # ============================================================
    from src.evaluation.sim_to_real_eval import SimToRealEvaluator

    evaluator = SimToRealEvaluator(
        model_dir=model_dir,
        pengym_scenario_path=scenario_path,
        baseline_summary_path=baseline_path,
        seed=args.seed,
        fully_obs=fully_obs,
        deterministic=deterministic,
    )

    evaluator.setup()

    start_time = time.time()
    results = evaluator.evaluate(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        verbose=verbose,
    )
    total_time = time.time() - start_time

    results['total_eval_time_seconds'] = round(total_time, 2)

    # ============================================================
    # Save results
    # ============================================================
    output_file = output_dir / f"pengym_{scenario_name}_results.json"
    evaluator.save_results(output_file, results)

    # Save gap analysis as separate file for easy reference
    if results.get('gap_analysis'):
        gap_file = output_dir / f"gap_analysis_{scenario_name}.json"
        with open(gap_file, 'w', encoding='utf-8') as f:
            json.dump(results['gap_analysis'], f, indent=2, ensure_ascii=False)
        print(f"Gap analysis saved to: {gap_file}")

    print(f"\nTotal evaluation time: {total_time:.1f}s")
    print("Done!")


if __name__ == "__main__":
    main()
