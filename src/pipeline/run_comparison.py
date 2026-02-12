"""
Flat vs Curriculum Training Comparison — Phase 3 Validation.

Runs two parallel training experiments:
  1) FLAT: Agent trains on randomly sampled scenarios (all tiers mixed)
  2) CURRICULUM: Agent trains T1→T2→T3→T4 with phase transitions

Both use the same SimpleDQNAgent and the same generated PenGym scenarios.
Compares convergence speed, final success rate, and training efficiency.

Usage:
  cd d:\\NCKH\\fusion\\pentest
  .\\venv\\Scripts\\python.exe src\\pipeline\\run_comparison.py
"""

import sys
import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from collections import deque

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nasim.scenarios import load_scenario
from src.pipeline.curriculum_controller import (
    CurriculumController, CurriculumConfig, PhaseConfig, FlatController
)
from src.pipeline.simple_dqn_agent import SimpleDQNAgent


def make_env(scenario_path: str):
    """Create a NASim env from scenario YAML."""
    import nasim
    scenario = load_scenario(scenario_path)
    env = nasim.NASimEnv(scenario, fully_obs=True,
                         flat_actions=True, flat_obs=True)
    return env


def train_one_episode(agent: SimpleDQNAgent, env, max_steps: int = 200):
    """Run one training episode.

    Returns:
        (success, total_reward, steps, avg_loss)
    """
    obs, info = env.reset()
    total_reward = 0.0
    losses = []

    for step in range(max_steps):
        action = agent.select_action(obs)
        next_obs, reward, done, truncated, info = env.step(action)

        agent.store_transition(obs, action, reward, next_obs,
                               float(done or truncated))

        loss = agent.update()
        if loss is not None:
            losses.append(loss)

        total_reward += reward
        obs = next_obs

        if done or truncated:
            break

    success = done and not truncated and total_reward > 0
    avg_loss = np.mean(losses) if losses else 0.0
    return success, total_reward, step + 1, avg_loss


def run_experiment(mode: str,
                   compiled_dir: str,
                   total_episodes: int = 600,
                   seed: int = 42,
                   max_steps: int = 200,
                   log_interval: int = 50) -> dict:
    """Run a single experiment (flat or curriculum).

    Args:
        mode: 'flat' or 'curriculum'
        compiled_dir: Path to compiled scenario YAMLs
        total_episodes: Max episodes for flat mode / sum of phase maxes for curriculum
        seed: Random seed
        max_steps: Max steps per episode
        log_interval: Episodes between progress logs

    Returns:
        Results dict with episode-level statistics
    """
    print(f"\n{'='*60}")
    print(f"Experiment: {mode.upper()} training")
    print(f"{'='*60}")

    # Determine obs/action dims from a sample scenario
    sample_scenarios = sorted(Path(compiled_dir).glob("*.yml"))
    if not sample_scenarios:
        raise RuntimeError(f"No scenarios found in {compiled_dir}")

    sample_env = make_env(str(sample_scenarios[0]))
    obs, _ = sample_env.reset()
    obs_dim = obs.shape[0]
    action_dim = sample_env.action_space.n
    sample_env.close()
    print(f"  Obs dim: {obs_dim}, Action dim: {action_dim}")

    # Create agent
    agent = SimpleDQNAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        lr=5e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=3000,
        buffer_size=10000,
        batch_size=64,
        target_update=100,
        hidden_dim=128,
        seed=seed,
    )

    # Create controller
    if mode == 'curriculum':
        config = CurriculumConfig(
            phases=[
                PhaseConfig(tier=1, sr_threshold=0.60, min_episodes=30,
                            max_episodes=200, sr_window=15, warmup_episodes=5),
                PhaseConfig(tier=2, sr_threshold=0.50, min_episodes=40,
                            max_episodes=250, sr_window=15, warmup_episodes=5),
                PhaseConfig(tier=3, sr_threshold=0.40, min_episodes=50,
                            max_episodes=300, sr_window=20, warmup_episodes=5),
                PhaseConfig(tier=4, sr_threshold=0.30, min_episodes=50,
                            max_episodes=300, sr_window=20, warmup_episodes=5),
            ],
            seed=seed,
            scenarios_per_tier=5,
        )
        controller = CurriculumController(config, compiled_dir)
    else:
        controller = FlatController(compiled_dir, max_episodes=total_episodes,
                                    seed=seed)

    # Training loop
    results = {
        'mode': mode,
        'seed': seed,
        'obs_dim': obs_dim,
        'action_dim': action_dim,
        'episodes': [],
        'per_tier_sr': {},
    }

    sr_window = deque(maxlen=50)
    reward_window = deque(maxlen=50)
    start_time = time.time()
    current_env = None
    current_scenario = None

    ep = 0
    while not controller.is_complete() and ep < total_episodes:
        ep += 1

        # Get scenario
        scenario_path = controller.get_next_scenario()

        # Load env (reuse if same scenario)
        if scenario_path != current_scenario:
            if current_env is not None:
                current_env.close()
            try:
                current_env = make_env(scenario_path)
                current_scenario = scenario_path

                # Adapt agent if action space differs
                new_action_dim = current_env.action_space.n
                new_obs, _ = current_env.reset()
                new_obs_dim = new_obs.shape[0]

                if new_action_dim != agent.action_dim or new_obs_dim != agent.obs_dim:
                    # Reinitialize networks for new dimensions
                    # (Keep replay buffer)
                    agent_old_buffer = agent.buffer
                    agent = SimpleDQNAgent(
                        obs_dim=new_obs_dim,
                        action_dim=new_action_dim,
                        lr=5e-4, gamma=0.99,
                        epsilon_start=max(agent.epsilon, 0.3),
                        epsilon_end=0.05,
                        epsilon_decay=3000,
                        buffer_size=10000,
                        batch_size=64,
                        target_update=100,
                        hidden_dim=128,
                        seed=seed + ep,
                    )
                    # Don't carry over buffer if dims changed
            except Exception as e:
                print(f"  ERROR loading {Path(scenario_path).name}: {e}")
                controller.record_episode(False, -10, 0)
                continue

        # Train one episode
        success, reward, steps, avg_loss = train_one_episode(
            agent, current_env, max_steps)

        # Record
        status = controller.record_episode(success, reward, steps)
        sr_window.append(1 if success else 0)
        reward_window.append(reward)

        ep_data = {
            'episode': ep,
            'success': success,
            'reward': round(reward, 2),
            'steps': steps,
            'epsilon': round(agent.epsilon, 4),
            'loss': round(avg_loss, 6),
            'tier': status.get('tier', 0),
            'sr_50': round(np.mean(list(sr_window)), 3),
        }
        results['episodes'].append(ep_data)

        # Log progress
        if ep % log_interval == 0 or (status.get('transition') is not None):
            elapsed = time.time() - start_time
            sr = np.mean(list(sr_window)) if sr_window else 0
            avg_r = np.mean(list(reward_window)) if reward_window else 0
            tier_str = f"T{status.get('tier', '?')}" if mode == 'curriculum' else 'ALL'
            print(f"  [{mode:10s}] Ep {ep:4d} | {tier_str} | "
                  f"SR={sr:.2f} | R={avg_r:.1f} | "
                  f"eps={agent.epsilon:.3f} | {elapsed:.0f}s")

            if status.get('transition'):
                t = status['transition']
                print(f"    >>> PHASE TRANSITION: T{t.get('from_tier','?')} → "
                      f"T{t.get('to_tier','?')} "
                      f"(SR={t.get('final_sr',0):.2f}, "
                      f"eps_in_phase={t.get('episodes_in_phase',0)}, "
                      f"reason={t.get('reason','?')})")

    if current_env is not None:
        current_env.close()

    # Final statistics
    elapsed = time.time() - start_time
    successes = [e['success'] for e in results['episodes']]
    rewards = [e['reward'] for e in results['episodes']]

    results['summary'] = {
        'total_episodes': ep,
        'overall_sr': np.mean(successes) if successes else 0,
        'final_sr_50': np.mean(list(sr_window)) if sr_window else 0,
        'avg_reward': np.mean(rewards) if rewards else 0,
        'total_time_s': round(elapsed, 1),
        'transitions': controller.phase_history if hasattr(controller, 'phase_history') else [],
    }

    # Per-tier breakdown
    for tier in range(1, 5):
        tier_eps = [e for e in results['episodes'] if e['tier'] == tier]
        if tier_eps:
            tier_sr = np.mean([e['success'] for e in tier_eps])
            results['per_tier_sr'][f'T{tier}'] = {
                'episodes': len(tier_eps),
                'sr': round(tier_sr, 3),
                'avg_reward': round(np.mean([e['reward'] for e in tier_eps]), 2),
            }

    print(f"\n  {mode.upper()} RESULTS:")
    print(f"    Total episodes: {ep}")
    print(f"    Overall SR: {results['summary']['overall_sr']:.3f}")
    print(f"    Final SR(50): {results['summary']['final_sr_50']:.3f}")
    print(f"    Avg reward: {results['summary']['avg_reward']:.2f}")
    print(f"    Time: {elapsed:.1f}s")
    if results['per_tier_sr']:
        for tk, tv in results['per_tier_sr'].items():
            print(f"    {tk}: SR={tv['sr']:.3f} ({tv['episodes']} eps)")

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description='Flat vs Curriculum training comparison')
    parser.add_argument('--episodes', type=int, default=600,
                        help='Max total episodes (default: 600)')
    parser.add_argument('--max-steps', type=int, default=200,
                        help='Max steps per episode (default: 200)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output dir for results JSON')
    parser.add_argument('--template', type=str, default='tiny',
                        help='Template to use for scenarios (default: tiny)')
    return parser.parse_args()


def main():
    args = parse_args()

    compiled_dir = str(
        PROJECT_ROOT / 'data' / 'scenarios' / 'generated' / 'compiled')
    output_dir = args.output_dir or str(
        PROJECT_ROOT / 'outputs' / 'curriculum_comparison')

    print("=" * 60)
    print("Phase 3: Flat vs Curriculum Training Comparison")
    print("=" * 60)
    print(f"  Compiled scenarios: {compiled_dir}")
    print(f"  Max episodes: {args.episodes}")
    print(f"  Max steps/ep: {args.max_steps}")
    print(f"  Seed: {args.seed}")

    # Run FLAT experiment
    flat_results = run_experiment(
        mode='flat',
        compiled_dir=compiled_dir,
        total_episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
    )

    # Run CURRICULUM experiment
    curriculum_results = run_experiment(
        mode='curriculum',
        compiled_dir=compiled_dir,
        total_episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
    )

    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    flat_sr = flat_results['summary']['final_sr_50']
    curr_sr = curriculum_results['summary']['final_sr_50']
    flat_r = flat_results['summary']['avg_reward']
    curr_r = curriculum_results['summary']['avg_reward']

    print(f"  {'Metric':<30s} {'FLAT':>10s} {'CURRICULUM':>12s}")
    print(f"  {'-'*54}")
    print(f"  {'Final SR (50-window)':<30s} {flat_sr:>10.3f} {curr_sr:>12.3f}")
    print(f"  {'Overall SR':<30s} "
          f"{flat_results['summary']['overall_sr']:>10.3f} "
          f"{curriculum_results['summary']['overall_sr']:>12.3f}")
    print(f"  {'Avg Reward':<30s} {flat_r:>10.2f} {curr_r:>12.2f}")
    print(f"  {'Total Episodes':<30s} "
          f"{flat_results['summary']['total_episodes']:>10d} "
          f"{curriculum_results['summary']['total_episodes']:>12d}")
    print(f"  {'Training Time (s)':<30s} "
          f"{flat_results['summary']['total_time_s']:>10.1f} "
          f"{curriculum_results['summary']['total_time_s']:>12.1f}")

    # Phase transitions
    transitions = curriculum_results['summary'].get('transitions', [])
    if transitions:
        print(f"\n  Curriculum Phase Transitions:")
        for t in transitions:
            print(f"    T{t.get('from_tier','?')} → T{t.get('to_tier','?')} "
                  f"after {t.get('episodes_in_phase', 0)} episodes "
                  f"(SR={t.get('final_sr', 0):.2f}, "
                  f"reason={t.get('reason','?')})")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'flat_results.json'), 'w') as f:
        json.dump(flat_results, f, indent=2, default=str)
    with open(os.path.join(output_dir, 'curriculum_results.json'), 'w') as f:
        json.dump(curriculum_results, f, indent=2, default=str)

    comparison = {
        'flat': flat_results['summary'],
        'curriculum': curriculum_results['summary'],
        'curriculum_better': curr_sr > flat_sr,
        'sr_improvement': round(curr_sr - flat_sr, 4),
    }
    with open(os.path.join(output_dir, 'comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=2, default=str)

    print(f"\n  Results saved to: {output_dir}")
    print(f"  Curriculum {'outperforms' if curr_sr > flat_sr else 'underperforms'} "
          f"flat by {abs(curr_sr - flat_sr):.3f} SR")
    print("=" * 60)


if __name__ == '__main__':
    main()
