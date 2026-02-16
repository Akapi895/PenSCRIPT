#!/usr/bin/env python3
"""
Comprehensive SCRIPT CRL Experiment Runner
==========================================
Runs the full evaluation pipeline:
  1. Multi-seed 3-task SCRIPT CRL
  2. Finetune baseline (no CL pillars)
  3. 5-task scaling
  4. Ablation studies
  5. Generalization test (train 3 → eval unseen)

Usage:
    python run_experiments.py                    # Run ALL experiments
    python run_experiments.py --phase 1          # Run only phase 1 (multi-seed)
    python run_experiments.py --phase 2          # Run only phase 2 (finetune baseline)
    python run_experiments.py --phase 3          # Run only phase 3 (5-task)
    python run_experiments.py --phase 4          # Run only phase 4 (ablation)
    python run_experiments.py --phase 5          # Run only phase 5 (generalization)
    python run_experiments.py --phase 1 2        # Run phases 1 and 2
"""

import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# ── Paths ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
VENV_PYTHON = str(ROOT / "venv" / "Scripts" / "python.exe")
if not Path(VENV_PYTHON).exists():
    # Fallback to system python
    VENV_PYTHON = sys.executable
RESULTS_DIR = ROOT / "outputs" / "experiments"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Env ──────────────────────────────────────────────────────────────
ENV = os.environ.copy()
ENV["HF_HUB_OFFLINE"] = "1"


def run_cmd(cmd: list, label: str, timeout: int = 7200) -> dict:
    """Run a subprocess command, stream output, and capture key metrics."""
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {label}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*70}\n", flush=True)

    t0 = time.time()
    log_lines = []
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=str(ROOT), env=ENV,
        )

        # Stream output line by line
        for line in iter(proc.stdout.readline, ''):
            line = line.rstrip('\n')
            log_lines.append(line)
            # Print key lines to console for progress visibility
            if any(kw in line for kw in [
                'Final SR', 'After task', 'TRAINING', 'COMPLETE',
                'Training complete', 'SR:', 'SCRIPT CRL',
                '[PenGymScriptTrainer]', 'ERROR', 'Error',
            ]):
                print(f"  | {line}", flush=True)

        proc.stdout.close()
        proc.wait(timeout=timeout)
        elapsed = time.time() - t0
        stdout = '\n'.join(log_lines)

        # Parse final SR from output
        final_sr = None
        sr_per_task = {}
        for line in log_lines:
            if 'Final SR (all tasks):' in line:
                try:
                    final_sr = float(line.split(':')[-1].strip().replace('%', '')) / 100
                except:
                    pass
            if 'After task' in line and 'SR=' in line:
                try:
                    parts = line.strip().split()
                    task_id = int(parts[parts.index('task') + 1].rstrip(':'))
                    sr_val = float(line.split('SR=')[1].split('%')[0]) / 100
                    sr_per_task[task_id] = sr_val
                except:
                    pass

        info = {
            "label": label,
            "returncode": proc.returncode,
            "elapsed_s": round(elapsed, 1),
            "final_sr": final_sr,
            "sr_per_task": sr_per_task,
            "success": proc.returncode == 0 and final_sr is not None,
        }

        if proc.returncode != 0:
            print(f"  [FAILED] Exit code: {proc.returncode}")
            for l in log_lines[-20:]:
                print(f"    {l}")
        else:
            print(f"  [OK] SR={final_sr}, Time={elapsed:.0f}s")
            for tid, sr in sorted(sr_per_task.items()):
                print(f"    Task {tid}: SR={sr*100:.1f}%")

        return info

    except subprocess.TimeoutExpired:
        proc.kill()
        elapsed = time.time() - t0
        print(f"  [TIMEOUT] after {elapsed:.0f}s")
        return {
            "label": label, "returncode": -1, "elapsed_s": round(elapsed, 1),
            "final_sr": None, "sr_per_task": {}, "success": False,
            "error": "timeout",
        }
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [ERROR] {e}")
        return {
            "label": label, "returncode": -1, "elapsed_s": round(elapsed, 1),
            "final_sr": None, "sr_per_task": {}, "success": False,
            "error": str(e),
        }


def save_results(phase: str, results: list):
    """Save phase results to JSON."""
    out = RESULTS_DIR / f"phase_{phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Multi-seed 3-task SCRIPT CRL
# ═══════════════════════════════════════════════════════════════════════

def phase1_multiseed():
    """Run 3-task SCRIPT CRL with seeds 42, 123, 456."""
    print("\n" + "█"*70)
    print("  PHASE 1: Multi-seed 3-task SCRIPT CRL")
    print("█"*70)

    seeds = [42, 123, 456]
    scenarios = ["tiny", "small-linear", "medium"]
    episodes = 1000
    max_steps = 1000

    results = []
    for seed in seeds:
        info = run_cmd(
            [VENV_PYTHON, "run_benchmark.py", "script-train",
             "--scenarios"] + scenarios + [
             "--episodes", str(episodes),
             "--max-steps", str(max_steps),
             "--seed", str(seed)],
            label=f"SCRIPT-3task-seed{seed}",
            timeout=3600,
        )
        info["seed"] = seed
        info["scenarios"] = scenarios
        info["episodes"] = episodes
        info["max_steps"] = max_steps
        results.append(info)

    # Summary
    print("\n" + "-"*50)
    print("  PHASE 1 SUMMARY: Multi-seed Results")
    print("-"*50)
    srs = [r["final_sr"] for r in results if r["final_sr"] is not None]
    for r in results:
        print(f"  Seed {r['seed']}: SR={r.get('final_sr', 'N/A')}, Time={r['elapsed_s']:.0f}s")
    if srs:
        import statistics
        mean_sr = statistics.mean(srs)
        std_sr = statistics.stdev(srs) if len(srs) > 1 else 0
        print(f"\n  Mean SR: {mean_sr*100:.1f}% ± {std_sr*100:.1f}%")

    save_results("1_multiseed", results)
    return results


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Finetune Baseline (no CL pillars)
# ═══════════════════════════════════════════════════════════════════════

def phase2_finetune_baseline():
    """Run finetune baseline (sequential training, no KD/EWC/teacher)."""
    print("\n" + "█"*70)
    print("  PHASE 2: Finetune Baseline (no CL mechanisms)")
    print("█"*70)

    scenarios = ["tiny", "small-linear", "medium"]
    episodes = 1000
    max_steps = 1000
    seed = 42

    # Finetune = script with ALL pillars disabled
    # ewc_lambda=0, guide_kl=0 → no EWC, no teacher guide
    # We use script-no__retention config which has: no retrospection, no KD retention
    # Actually, the cleanest way is ewc_lambda=0 + guide_kl=0
    # But let's also test with config file for ablation compatibility

    info = run_cmd(
        [VENV_PYTHON, "run_benchmark.py", "script-train",
         "--scenarios"] + scenarios + [
         "--episodes", str(episodes),
         "--max-steps", str(max_steps),
         "--seed", str(seed),
         "--ewc-lambda", "0",
         "--guide-kl", "0"],
        label="Finetune-baseline-3task",
        timeout=3600,
    )
    info["seed"] = seed
    info["scenarios"] = scenarios
    info["config"] = "finetune (ewc=0, guide=0)"

    results = [info]

    print("\n" + "-"*50)
    print("  PHASE 2 SUMMARY: Finetune Baseline")
    print("-"*50)
    print(f"  SR={info.get('final_sr', 'N/A')}, Time={info['elapsed_s']:.0f}s")

    save_results("2_finetune", results)
    return results


# ═══════════════════════════════════════════════════════════════════════
# Phase 3: Scale to 5 tasks
# ═══════════════════════════════════════════════════════════════════════

def phase3_five_tasks():
    """Run full 5-task SCRIPT CRL curriculum."""
    print("\n" + "█"*70)
    print("  PHASE 3: 5-task SCRIPT CRL (full curriculum)")
    print("█"*70)

    scenarios = ["tiny", "small-linear", "medium", "medium-single-site", "medium-multi-site"]
    episodes = 1000
    max_steps = 1000
    seed = 42

    info = run_cmd(
        [VENV_PYTHON, "run_benchmark.py", "script-train",
         "--scenarios"] + scenarios + [
         "--episodes", str(episodes),
         "--max-steps", str(max_steps),
         "--seed", str(seed)],
        label="SCRIPT-5task",
        timeout=7200,
    )
    info["seed"] = seed
    info["scenarios"] = scenarios

    results = [info]

    print("\n" + "-"*50)
    print("  PHASE 3 SUMMARY: 5-task Scaling")
    print("-"*50)
    print(f"  SR={info.get('final_sr', 'N/A')}, Time={info['elapsed_s']:.0f}s")
    for tid, sr in sorted(info.get("sr_per_task", {}).items()):
        print(f"    After task {tid}: SR={sr*100:.1f}%")

    save_results("3_five_tasks", results)
    return results


# ═══════════════════════════════════════════════════════════════════════
# Phase 4: Ablation Studies
# ═══════════════════════════════════════════════════════════════════════

def phase4_ablation():
    """Run ablation: disable each SCRIPT pillar individually."""
    print("\n" + "█"*70)
    print("  PHASE 4: Ablation Studies")
    print("█"*70)

    scenarios = ["tiny", "small-linear", "medium"]
    episodes = 1000
    max_steps = 1000
    seed = 42

    ablations = [
        ("Full SCRIPT", None, {}),
        ("No Teacher Guide", "script-no_guide.yaml", {}),
        ("No KL Imitation", "script-no_imitation.yaml", {}),
        ("No Retrospection", "script-no_res.yaml", {}),
        ("No EWC", "script-no_wic.yaml", {}),
        ("No Teacher Reset", "script-no_reset.yaml", {}),
    ]

    results = []
    for name, config_file, extra_kwargs in ablations:
        cmd = [VENV_PYTHON, "run_benchmark.py", "script-train",
               "--scenarios"] + scenarios + [
               "--episodes", str(episodes),
               "--max-steps", str(max_steps),
               "--seed", str(seed)]

        if config_file:
            cmd += ["--config", config_file]

        info = run_cmd(cmd, label=f"Ablation: {name}", timeout=3600)
        info["ablation"] = name
        info["config_file"] = config_file
        info["seed"] = seed
        results.append(info)

    # Summary table
    print("\n" + "-"*60)
    print("  PHASE 4 SUMMARY: Ablation Results")
    print("-"*60)
    print(f"  {'Ablation':<25s}  {'SR':>8s}  {'Time':>8s}")
    print(f"  {'─'*25}  {'─'*8}  {'─'*8}")
    for r in results:
        sr_str = f"{r['final_sr']*100:.1f}%" if r['final_sr'] is not None else "FAIL"
        print(f"  {r['ablation']:<25s}  {sr_str:>8s}  {r['elapsed_s']:>7.0f}s")

    save_results("4_ablation", results)
    return results


# ═══════════════════════════════════════════════════════════════════════
# Phase 5: Generalization Test
# ═══════════════════════════════════════════════════════════════════════

def phase5_generalization():
    """Train on 3 tasks, evaluate on unseen scenarios."""
    print("\n" + "█"*70)
    print("  PHASE 5: Generalization (train 3 → eval unseen)")
    print("█"*70)

    # We already have the 3-task model from phase 1 (seed 42).
    # Now evaluate it on medium-single-site (unseen).
    # We need to load the saved model and evaluate.

    print("  Training on 3 tasks (reuse from Phase 1 seed=42)...")
    # First make sure the model exists by re-training if needed
    train_scenarios = ["tiny", "small-linear", "medium"]
    eval_scenarios = ["medium-single-site", "medium-multi-site"]

    info_train = run_cmd(
        [VENV_PYTHON, "run_benchmark.py", "script-train",
         "--scenarios"] + train_scenarios + [
         "--episodes", "1000",
         "--max-steps", "1000",
         "--seed", "42"],
        label="Generalization-train-3task",
        timeout=3600,
    )

    # For generalization eval, we need a custom eval script
    # Create inline eval
    eval_script = str(ROOT / "_eval_generalization.py")
    with open(eval_script, 'w') as f:
        f.write('''#!/usr/bin/env python3
"""Evaluate saved SCRIPT model on unseen scenarios."""
import sys, json
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.training.pengym_script_trainer import PenGymScriptTrainer

SCENARIO_DIR = ROOT / "data" / "scenarios"
MODEL_DIR = ROOT / "outputs" / "models_pengym" / "script_crl"

# All scenarios for evaluation
eval_scenarios = sys.argv[1:]
if not eval_scenarios:
    eval_scenarios = ["medium-single-site", "medium-multi-site"]

scenario_paths = [str(SCENARIO_DIR / f"{sc}.yml") for sc in eval_scenarios]

# Build trainer just for the eval scenarios (to create adapters)
trainer = PenGymScriptTrainer(
    scenario_list=scenario_paths,
    seed=42,
)

# Load trained model
trainer.load(str(MODEL_DIR))

# Evaluate
results = trainer.evaluate(step_limit=1000, verbose=True)
sr = results["success_rate"]
print(f"\\nGeneralization SR: {sr*100:.1f}%")
for i, pr in enumerate(results["per_task"]):
    status = "SUCCESS" if pr["success"] else "FAILED"
    print(f"  {eval_scenarios[i]}: {status}, reward={pr['reward']:.1f}, steps={pr['steps']}")

# Output for parsing
print(f"Final SR (all tasks): {sr*100:.1f}%")
for i, pr in enumerate(results["per_task"]):
    sr_val = 100.0 if pr["success"] else 0.0
    print(f"    After task {i}: SR={sr_val:.1f}%")
''')

    info_eval = run_cmd(
        [VENV_PYTHON, eval_script] + eval_scenarios,
        label="Generalization-eval-unseen",
        timeout=600,
    )

    results = [
        {**info_train, "phase": "train"},
        {**info_eval, "phase": "eval_unseen", "eval_scenarios": eval_scenarios},
    ]

    print("\n" + "-"*50)
    print("  PHASE 5 SUMMARY: Generalization")
    print("-"*50)
    print(f"  Train (3 tasks): SR={info_train.get('final_sr', 'N/A')}")
    print(f"  Eval (unseen):   SR={info_eval.get('final_sr', 'N/A')}")

    save_results("5_generalization", results)

    # Clean up temp script
    try:
        os.remove(eval_script)
    except:
        pass

    return results


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SCRIPT CRL Experiment Runner")
    parser.add_argument("--phase", nargs="*", type=int, default=None,
                        help="Which phases to run (1-5). Default: all")
    args = parser.parse_args()

    phases_to_run = args.phase or [1, 2, 3, 4, 5]

    print("="*70)
    print("  SCRIPT CRL — Full Evaluation Pipeline")
    print(f"  Phases: {phases_to_run}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    all_results = {}
    t0 = time.time()

    if 1 in phases_to_run:
        all_results["phase1_multiseed"] = phase1_multiseed()

    if 2 in phases_to_run:
        all_results["phase2_finetune"] = phase2_finetune_baseline()

    if 3 in phases_to_run:
        all_results["phase3_five_tasks"] = phase3_five_tasks()

    if 4 in phases_to_run:
        all_results["phase4_ablation"] = phase4_ablation()

    if 5 in phases_to_run:
        all_results["phase5_generalization"] = phase5_generalization()

    total_time = time.time() - t0

    # ── Final Summary ────────────────────────────────────────────────
    summary_path = RESULTS_DIR / f"full_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    all_results["total_time_s"] = round(total_time, 1)
    all_results["timestamp"] = datetime.now().isoformat()

    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "="*70)
    print("  EXPERIMENT PIPELINE COMPLETE")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Summary: {summary_path}")
    print("="*70)


if __name__ == "__main__":
    main()
