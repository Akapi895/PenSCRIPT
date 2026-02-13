#!/usr/bin/env python3
"""
run_pengym_eval.py — Evaluate a PenGym-trained PPO model.

Loads a model from ``run_pengym_train.py`` (or compatible checkpoint)
and evaluates it on a specified NASim scenario through the
``SingleHostPenGymWrapper``.

Usage
-----
Basic evaluation:
    python run_pengym_eval.py \\
        --scenario data/scenarios/tiny.yml \\
        --model-dir outputs/models_pengym/tiny_pengym \\
        --episodes 20

Cross-scenario evaluation (train on tiny, eval on small-linear):
    python run_pengym_eval.py \\
        --scenario data/scenarios/small-linear.yml \\
        --model-dir outputs/models_pengym/tiny_pengym \\
        --episodes 20

Design reference: docs/pengym_integration_architecture.md §5.4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.policy.config import PPO_Config
from src.envs.wrappers.reward_normalizer import IdentityNormalizer
from src.envs.wrappers.target_selector import ReachabilityAwareSelector
from src.training.pengym_trainer import PenGymTrainer


OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PenGym PPO model")
    p.add_argument("--scenario", type=str, default="data/scenarios/tiny.yml")
    p.add_argument("--model-dir", type=str, required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--verbose", action="store_true", default=True)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    verbose = args.verbose and not args.quiet

    scenario_path = str(Path(args.scenario))
    model_dir = Path(args.model_dir)
    scenario_stem = Path(args.scenario).stem

    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"PenGym Evaluation — {scenario_stem}")
    print(f"  Model:     {model_dir}")
    print(f"  Scenario:  {scenario_path}")
    print(f"  Episodes:  {args.episodes}")
    print(f"  Max steps: {args.max_steps}")
    print(f"{'='*60}\n")

    config = PPO_Config(
        eval_step_limit=args.max_steps,
        step_limit=args.max_steps,
    )

    trainer = PenGymTrainer(
        initial_scenario=scenario_path,
        config=config,
        seed=args.seed,
        reward_normalizer=IdentityNormalizer(),  # raw rewards for eval
        target_selector=ReachabilityAwareSelector(),
        tb_dir=None,
    )

    # Load model
    trainer.load(str(model_dir))

    # Print wrapper config
    print(trainer.wrapper.describe())
    print()

    # Evaluate
    t0 = time.time()
    total_reward, sr = trainer.evaluate(
        num_episodes=args.episodes,
        step_limit=args.max_steps,
        verbose=verbose,
    )
    elapsed = time.time() - t0

    avg_reward = total_reward / args.episodes if args.episodes > 0 else 0

    print(f"\n{'='*40}")
    print(f"Results ({args.episodes} episodes):")
    print(f"  Success Rate:   {sr*100:.1f}%")
    print(f"  Total Reward:   {total_reward:.1f}")
    print(f"  Avg Reward:     {avg_reward:.1f}")
    print(f"  Eval Time:      {elapsed:.2f}s")
    print(f"{'='*40}")

    # Save results
    output_dir = Path(args.output_dir) if args.output_dir else model_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "scenario": scenario_path,
        "model_dir": str(model_dir),
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "success_rate": float(sr),
        "total_reward": float(total_reward),
        "avg_reward": float(avg_reward),
        "eval_time_s": round(elapsed, 2),
    }

    out_path = output_dir / f"eval_{scenario_stem}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    trainer.close()
    print("Done!")


if __name__ == "__main__":
    main()
