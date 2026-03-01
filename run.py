#!/usr/bin/env python3
"""
run.py — CLI entry point for PenSCRIPT dual training pipeline.

PenSCRIPT trains a SCRIPT CRL agent on simulation first, then transfers
the policy to PenGym via controlled domain transfer and fine-tunes with
EWC constraints.

Usage
-----
Full pipeline (Phase 0→1→2→3→4):
    python run.py \\
        --sim-scenarios data/scenarios/chain/chain_1.json data/scenarios/chain/chain_2.json \\
        --pengym-scenarios data/scenarios/tiny.yml data/scenarios/small-linear.yml

With a specific transfer strategy:
    python run.py \\
        --sim-scenarios data/scenarios/chain/chain_1.json \\
        --pengym-scenarios data/scenarios/tiny.yml \\
        --transfer-strategy cautious

Include θ_pengym_scratch baseline (for full 4-agent comparison):
    python run.py \\
        --sim-scenarios data/scenarios/chain/chain_1.json \\
        --pengym-scenarios data/scenarios/tiny.yml \\
        --train-scratch

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

from src.utils.logging import TeeLogger, ENV_NOISE_PATTERNS

# ── Paths ────────────────────────────────────────────────────────────
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PENSCRIPT_DIR = OUTPUTS_DIR / "penscript"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PenSCRIPT — Dual Training Pipeline (Sim → Transfer → PenGym)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Scenarios ────────────────────────────────────────────────────
    parser.add_argument(
        "--sim-scenarios", nargs="+", required=True,
        help="Simulation scenario JSON files for Phase 1 training.",
    )
    parser.add_argument(
        "--pengym-scenarios", nargs="+", required=True,
        help="PenGym scenario YAML files for Phase 3 fine-tuning.",
    )
    parser.add_argument(
        "--heldout-scenarios", nargs="*", default=None,
        help="Heldout PenGym scenarios for generalization eval (Phase 4).",
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
    parser.add_argument(
        "--scratch-only", action="store_true",
        help="Only train θ_pengym_scratch and evaluate it (calibration mode). Skips Phases 0-3.",
    )
    parser.add_argument(
        "--episode-config", type=str, default=None,
        help="JSON file with per-scenario episode configuration. "
             "Supports multiplier mode (base_episodes × tier_multiplier) "
             "or rules mode (regex patterns → episodes). "
             "See data/config/curriculum_episodes.json for format.",
    )
    parser.add_argument(
        "--training-mode", type=str, default="intra_topology",
        choices=["intra_topology", "cross_topology"],
        help="Phase 3 training mode: 'intra_topology' (per-topology CRL "
             "streams, recommended) or 'cross_topology' (legacy "
             "tier-grouped CRL). Default: intra_topology.",
    )

    # ── Ablation ────────────────────────────────────────────────────
    parser.add_argument(
        "--no-canonicalization", action="store_true",
        help="Disable cross-domain canonicalization maps (ablation study).",
    )

    # ── Resume ───────────────────────────────────────────────────────
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Path to a previous (interrupted) output dir to resume from. "
             "Skips Phase 0/1 (loads model), skips completed streams, "
             "retrains only incomplete streams.",
    )

    # ── Output ───────────────────────────────────────────────────────
    parser.add_argument(
        "--output-dir", type=str, default=str(PENSCRIPT_DIR),
        help="Output directory for logs, models, and results.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set up logging
    log_dir = Path(args.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    tee = TeeLogger(str(log_dir / "penscript.log"),
                     console_suppress=ENV_NOISE_PATTERNS)

    print("=" * 70)
    print("  PenSCRIPT — Dual Training Pipeline")
    print("=" * 70)
    print(f"  Sim scenarios:    {args.sim_scenarios}")
    print(f"  PenGym scenarios: {args.pengym_scenarios}")
    print(f"  Heldout scenarios: {args.heldout_scenarios or '(none)'}")
    print(f"  Transfer:         {args.transfer_strategy} (β={args.fisher_beta}, LR×{args.lr_factor})")
    print(f"  Training:         {args.episodes} eps, {args.step_limit} steps, EWC λ={args.ewc_lambda}")
    print(f"  Seed:             {args.seed}")
    print(f"  Mode:             {'scratch-only (calibration)' if args.scratch_only else 'full pipeline'}")
    print(f"  Training mode:    {args.training_mode}")
    print(f"  Resume from:      {args.resume_from or '(none)'}")
    print(f"  Episode config:   {args.episode_config or '(uniform ' + str(args.episodes) + ' eps)'}")
    print(f"  Output:           {args.output_dir}")
    print("=" * 70)

    from src.training.dual_trainer import DualTrainer

    # Load episode config if provided
    episode_config = None
    if args.episode_config:
        with open(args.episode_config, "r") as f:
            episode_config = json.load(f)
        # Strip description keys (comments)
        episode_config.pop("_description", None)
        print(f"  Loaded episode config: {list(episode_config.keys())}")

    # Resolve training mode: CLI flag takes priority, then episode_config
    training_mode = args.training_mode
    if episode_config and "training_mode" in episode_config:
        # CLI default can be overridden by config file
        if args.training_mode == "intra_topology":  # unchanged from default
            training_mode = episode_config["training_mode"]

    # Build DualTrainer
    trainer = DualTrainer(
        sim_scenarios=args.sim_scenarios,
        pengym_scenarios=args.pengym_scenarios,
        heldout_scenarios=args.heldout_scenarios,
        ppo_kwargs={
            "train_eps": args.episodes,
            "step_limit": args.step_limit,
            "eval_step_limit": args.step_limit,
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
        episode_config=episode_config,
        training_mode=training_mode,
        use_canonicalization=not args.no_canonicalization,
    )

    t0 = time.time()

    if args.scratch_only:
        # Calibration mode: only train scratch + evaluate it
        results = {"mode": "scratch_only"}
        scratch_results = trainer.train_pengym_scratch(eval_freq=args.eval_freq)
        results["scratch"] = scratch_results
        results["phase4"] = trainer.phase4_evaluation()
    else:
        # Full pipeline
        results = trainer.run_full_pipeline(
            skip_phase0=args.skip_phase0,
            eval_freq=args.eval_freq,
            resume_from=args.resume_from,
        )

        # Optionally train scratch baseline
        if args.train_scratch:
            print("\n" + "=" * 60)
            print("  Training θ_pengym_scratch baseline...")
            print("=" * 60)
            scratch_results = trainer.train_pengym_scratch(eval_freq=args.eval_freq)
            results["scratch"] = scratch_results

            # Re-run Phase 4 with the scratch agent available
            print("\n  Re-running Phase 4 with all 4 agents...")
            results["phase4"] = trainer.phase4_evaluation()

    elapsed = time.time() - t0

    # Print summary
    print("\n" + "=" * 70)
    print("  PenSCRIPT — Pipeline Complete")
    print("=" * 70)
    print(f"  Total time: {elapsed:.1f}s")

    if "phase4" in results:
        phase4 = results["phase4"]
        for agent_name, data in phase4.get("agents", {}).items():
            sr = data.get("success_rate", "N/A")
            nr = data.get("normalized_reward")
            eta = data.get("step_efficiency")
            if isinstance(sr, float):
                sr_str = f"{sr:.1%}"
            else:
                sr_str = str(sr)
            parts = [f"SR={sr_str}"]
            if nr is not None:
                parts.append(f"NR={nr:.3f}")
            if eta is not None:
                parts.append(f"η={eta:.3f}")
            print(f"    {agent_name}: {', '.join(parts)}")
        transfer = phase4.get("transfer_metrics", {})
        if transfer:
            for key in ["FT_SR", "FT_NR", "FT_eta", "BT_SR", "BT_NR", "BT_eta"]:
                val = transfer.get(key)
                if val is not None:
                    print(f"  {key}: {val:+.4f}")
            for key in ["BT_KL", "BT_fisher_dist"]:
                val = transfer.get(key)
                if val is not None:
                    print(f"  {key}: {val:.6f}")

    # Save final results
    results_path = Path(args.output_dir) / "penscript_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved → {results_path}")

    tee.close()


if __name__ == "__main__":
    main()
