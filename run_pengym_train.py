#!/usr/bin/env python3
"""
run_pengym_train.py — Train a PPO agent on PenGym NASim scenarios.

Usage
-----
Single scenario:
    python run_pengym_train.py --scenario data/scenarios/tiny.yml --episodes 300

Curriculum:
    python run_pengym_train.py --mode curriculum \\
        --scenario-dir data/scenarios/chain \\
        --episodes 2000

Evaluate a trained model:
    python run_pengym_train.py --mode eval \\
        --scenario data/scenarios/tiny.yml \\
        --model-dir outputs/models_pengym/tiny

Design reference: docs/pengym_integration_architecture.md §5.3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


class TeeLogger:
    """Duplicate stdout/stderr to a log file while still printing."""

    def __init__(self, log_path: str):
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_path, "w", encoding="utf-8")
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def write(self, data: str):
        self._stdout.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self._file.close()

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.policy.config import PPO_Config
from src.envs.wrappers.reward_normalizer import (
    ClipNormalizer,
    IdentityNormalizer,
    LinearNormalizer,
)
from src.envs.wrappers.target_selector import (
    PrioritySensitiveSelector,
    RoundRobinSelector,
    ValuePrioritySelector,
)
from src.training.pengym_trainer import PenGymTrainer


# ── Paths ────────────────────────────────────────────────────────────
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models_pengym"
LOGS_DIR = OUTPUTS_DIR / "logs" / "pengym"
TB_DIR = OUTPUTS_DIR / "tensorboard" / "pengym"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PenGym PPO training with SingleHostPenGymWrapper"
    )
    p.add_argument(
        "--mode",
        choices=["train", "eval", "curriculum"],
        default="train",
    )
    p.add_argument("--scenario", type=str, default="data/scenarios/tiny.yml")
    p.add_argument("--scenario-dir", type=str, default="data/scenarios/chain")
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--eval-step-limit", type=int, default=20)
    p.add_argument("--eval-freq", type=int, default=5)
    p.add_argument("--log-freq", type=int, default=50)
    p.add_argument("--save-freq", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-dir", type=str, default=None)
    p.add_argument("--output-suffix", type=str, default="pengym")
    p.add_argument(
        "--reward",
        choices=["linear", "clip", "identity"],
        default="linear",
    )
    p.add_argument(
        "--target-selector",
        choices=["priority", "roundrobin", "value"],
        default="priority",
    )
    p.add_argument("--no-state-norm", action="store_true")
    p.add_argument("--no-tb", action="store_true")
    p.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Save ALL terminal output (stdout+stderr) to this file. "
             "Auto-generated if set to 'auto'.",
    )
    return p.parse_args()


def build_normalizer(name: str):
    return {
        "linear": LinearNormalizer(),
        "clip": ClipNormalizer(scale=100, clip=10),
        "identity": IdentityNormalizer(),
    }[name]


def build_selector(name: str):
    return {
        "priority": PrioritySensitiveSelector(),
        "roundrobin": RoundRobinSelector(),
        "value": ValuePrioritySelector(),
    }[name]


def main():
    args = parse_args()

    scenario_path = str(Path(args.scenario))
    suffix = args.output_suffix
    scenario_stem = Path(args.scenario).stem

    model_dir = (
        args.model_dir
        if args.model_dir
        else str(MODELS_DIR / f"{scenario_stem}_{suffix}")
    )
    tb_dir = None if args.no_tb else str(TB_DIR / f"{scenario_stem}_{suffix}")

    # ── Setup log file ────────────────────────────────────────────
    tee = None
    if args.log_file:
        log_path = args.log_file
        if log_path == "auto":
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            log_path = str(
                LOGS_DIR / f"{scenario_stem}_{args.mode}_{ts}.log"
            )
        tee = TeeLogger(log_path)
        print(f"[LOG] All output is being saved to: {log_path}")

    config = PPO_Config(
        train_eps=args.episodes,
        step_limit=args.max_steps,
        eval_step_limit=args.eval_step_limit,
        use_state_norm=not args.no_state_norm,
    )

    reward_norm = build_normalizer(args.reward)
    selector = build_selector(args.target_selector)

    # ============================================================
    #  Mode: train (single scenario)
    # ============================================================
    if args.mode == "train":
        print(f"\n{'='*60}")
        print(f"PenGym Training — {scenario_stem}")
        print(f"  Episodes:     {args.episodes}")
        print(f"  Max steps:    {args.max_steps}")
        print(f"  Reward:       {args.reward}")
        print(f"  Selector:     {args.target_selector}")
        print(f"  State norm:   {not args.no_state_norm}")
        print(f"  Model dir:    {model_dir}")
        print(f"{'='*60}\n")

        trainer = PenGymTrainer(
            initial_scenario=scenario_path,
            config=config,
            seed=args.seed,
            reward_normalizer=reward_norm,
            target_selector=selector,
            tb_dir=tb_dir,
        )

        print(trainer.wrapper.describe())
        print()

        results = trainer.train(
            num_episodes=args.episodes,
            eval_freq=args.eval_freq,
            log_freq=args.log_freq,
            model_dir=model_dir,
            save_freq=args.save_freq,
        )

        # Final evaluation
        print(f"\n{'='*40}")
        print("Final Evaluation (10 episodes):")
        print(f"{'='*40}")
        final_r, final_sr = trainer.evaluate(
            num_episodes=10,
            step_limit=args.eval_step_limit,
        )
        print(f"\nFinal: reward={final_r:.1f}, SR={final_sr*100:.1f}%")

        # Save summary
        summary = {
            "mode": "train",
            "scenario": scenario_path,
            "episodes": args.episodes,
            "state_dim": trainer.state_dim,
            "action_dim": trainer.action_dim,
            "seed": args.seed,
            "reward_normalizer": args.reward,
            "target_selector": args.target_selector,
            "convergence": {
                "first_hit_eps": trainer.first_hit_eps,
                "convergence_eps": trainer.convergence_eps,
            },
            "results": {
                "train_time_s": results["train_time_s"],
                "total_steps": results["total_steps"],
                "best_return": float(trainer.best_return),
                "final_sr": final_sr,
            },
        }
        summary_path = Path(model_dir) / "experiment_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Summary → {summary_path}")

        trainer.close()

    # ============================================================
    #  Mode: curriculum
    # ============================================================
    elif args.mode == "curriculum":
        from src.pipeline.curriculum_controller import (
            CurriculumController,
            CurriculumConfig,
        )

        print(f"\n{'='*60}")
        print(f"PenGym Curriculum Training")
        print(f"  Scenario dir: {args.scenario_dir}")
        print(f"  Max episodes: {args.episodes}")
        print(f"{'='*60}\n")

        controller = CurriculumController(
            config=CurriculumConfig(),
            scenario_dir=args.scenario_dir,
            log_dir=str(LOGS_DIR / f"curriculum_{suffix}"),
        )

        # Use first scenario to init trainer
        first_scenario = controller.get_next_scenario()
        trainer = PenGymTrainer(
            initial_scenario=first_scenario,
            config=config,
            seed=args.seed,
            reward_normalizer=reward_norm,
            target_selector=selector,
            tb_dir=tb_dir,
        )

        # Put back the first scenario (controller already advanced)
        # We'll let train_curriculum handle the loop from here
        controller.record_episode(success=False, reward=0, steps=0)

        results = trainer.train_curriculum(
            controller=controller,
            eval_freq=args.eval_freq,
            log_freq=args.log_freq,
            model_dir=model_dir,
        )

        print(f"\nCurriculum complete: {results['total_episodes']} episodes")
        print(f"Final status: {results['final_status']}")

        trainer.close()

    # ============================================================
    #  Mode: eval
    # ============================================================
    elif args.mode == "eval":
        load_dir = args.model_dir or model_dir
        if not Path(load_dir).exists():
            print(f"Error: Model dir not found: {load_dir}")
            sys.exit(1)

        print(f"\n{'='*60}")
        print(f"PenGym Evaluation — {scenario_stem}")
        print(f"  Model: {load_dir}")
        print(f"{'='*60}\n")

        trainer = PenGymTrainer(
            initial_scenario=scenario_path,
            config=config,
            seed=args.seed,
            reward_normalizer=IdentityNormalizer(),  # raw rewards for eval
            target_selector=selector,
            tb_dir=None,
        )
        trainer.load(load_dir)

        total_r, sr = trainer.evaluate(
            num_episodes=10,
            step_limit=args.eval_step_limit,
        )
        print(f"\nResult: reward={total_r:.1f}, SR={sr*100:.1f}%")

        trainer.close()

    print("\nDone!")

    if tee:
        tee.close()


if __name__ == "__main__":
    main()
