"""
Sim-to-Real Evaluator — Run SCRIPT policy on PenGym and collect transfer metrics.

This module implements the core evaluation loop for Strategy A:
  1. Load pre-trained SCRIPT model (PPO + state norm)
  2. Create PenGym NASim environment
  3. For each episode:
     a. Reset env → get flat obs
     b. Choose target host (sensitive hosts = goal)
     c. Convert PenGym obs → SCRIPT 1538-dim state (via StateAdapter)
     d. Normalize state (using saved running mean/std from training)
     e. Policy selects action (deterministic or stochastic)
     f. Map action → PenGym action (via ActionMapper)
     g. Execute action, collect reward, update obs
  4. Aggregate metrics and compare with simulation baseline
"""

import os
import sys
import json
import time
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from loguru import logger as logging


class SimToRealEvaluator:
    """Evaluate pre-trained SCRIPT agent on PenGym environment."""

    def __init__(
        self,
        model_dir: Path,
        pengym_scenario_path: Path,
        baseline_summary_path: Optional[Path] = None,
        seed: int = 42,
        fully_obs: bool = True,
        deterministic: bool = True,
    ):
        """
        Args:
            model_dir: Directory containing PPO-actor.pt, PPO-critic.pt,
                       PPO-norm_mean.pt, PPO-norm_std.pt
            pengym_scenario_path: Path to PenGym NASim .yml scenario file
            baseline_summary_path: Path to simulation baseline experiment_summary.json
            seed: Random seed
            fully_obs: Whether to use fully observable mode
            deterministic: Whether to use deterministic action selection
        """
        self.model_dir = Path(model_dir)
        self.scenario_path = Path(pengym_scenario_path)
        self.seed = seed
        self.fully_obs = fully_obs
        self.deterministic = deterministic
        self.baseline = None

        # Load baseline results if provided
        if baseline_summary_path and Path(baseline_summary_path).exists():
            with open(baseline_summary_path, 'r') as f:
                self.baseline = json.load(f)
            print(f"[Eval] Loaded baseline: SR={self.baseline.get('experiment', {}).get('success_rate', '?')}")

        # Set seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # These will be initialized in setup()
        self.env = None
        self.agent = None
        self.state_adapter = None
        self.action_mapper = None

    def setup(self):
        """Initialize all components. Must be called before evaluate()."""
        print("\n" + "=" * 60)
        print("Strategy A: Sim-to-Real Transfer Evaluation Setup")
        print("=" * 60)

        # --- 1. Create PenGym environment (NASim simulation mode) ---
        print(f"\n[1/4] Creating PenGym environment from {self.scenario_path}...")
        self._setup_environment()

        # --- 2. Create State Adapter ---
        print(f"\n[2/4] Initializing State Adapter...")
        self._setup_state_adapter()

        # --- 3. Create Action Mapper ---
        print(f"\n[3/4] Building Action Mapper...")
        self._setup_action_mapper()

        # --- 4. Load pre-trained agent ---
        print(f"\n[4/4] Loading pre-trained model from {self.model_dir}...")
        self._setup_agent()

        print("\n" + "=" * 60)
        print("Setup complete. Ready for evaluation.")
        print("=" * 60 + "\n")

    def _setup_environment(self):
        """Create PenGym NASim environment."""
        from src.envs import load as load_pengym_env, utilities

        # Force simulation mode (NASim only, no real nmap/metasploit)
        utilities.ENABLE_PENGYM = False
        utilities.ENABLE_NASIM = True

        self.env = load_pengym_env(
            str(self.scenario_path),
            fully_obs=self.fully_obs,
            flat_actions=True,
            flat_obs=True,
        )
        self.env.action_space.seed(self.seed)

        print(f"  Action space: {self.env.action_space.n} actions")
        obs, _ = self.env.reset()
        print(f"  Observation shape: {obs.shape}")
        self.scenario = utilities.scenario

    def _setup_state_adapter(self):
        """Create PenGymStateAdapter."""
        from src.envs.adapters.state_adapter import PenGymStateAdapter
        self.state_adapter = PenGymStateAdapter(self.scenario)
        print(f"  {self.state_adapter.describe()}")

    def _setup_action_mapper(self):
        """Create ActionMapper."""
        from src.envs.adapters.action_mapper import ActionMapper
        from src.agent.actions import Action
        self.action_mapper = ActionMapper(Action, self.env)
        print(f"\n  Mapping stats: {self.action_mapper.get_mapping_stats()}")

    def _setup_agent(self):
        """Load pre-trained SCRIPT agent."""
        from src.agent.agent import Agent
        from src.agent.policy.config import PPO_Config

        config = PPO_Config()

        self.agent = Agent(
            policy_name="PPO",
            config=config,
            seed=self.seed,
            use_wandb=False,
            logger=None,
        )
        self.agent.load(self.model_dir)
        print(f"  Model loaded successfully (state_norm={self.agent.use_state_norm})")

    def evaluate(
        self,
        num_episodes: int = 20,
        max_steps: int = 100,
        verbose: bool = True,
    ) -> Dict:
        """Run evaluation episodes and collect metrics.

        The key challenge: SCRIPT is a single-target agent. PenGym is multi-host.
        Strategy: For each episode, iterate over sensitive hosts as targets,
        similar to how SCRIPT trains on target_list sequentially.

        Args:
            num_episodes: Number of evaluation episodes
            max_steps: Maximum steps per episode
            verbose: Print per-episode details

        Returns:
            Dict with comprehensive evaluation results
        """
        assert self.env is not None, "Call setup() first"

        print(f"\n{'='*60}")
        print(f"Evaluating: {num_episodes} episodes, max {max_steps} steps")
        print(f"Scenario: {self.scenario_path.name}")
        print(f"Model: {self.model_dir}")
        print(f"{'='*60}\n")

        # Get sensitive (goal) hosts
        sensitive_hosts = self.state_adapter.get_sensitive_hosts()
        if not sensitive_hosts:
            # Fallback: target all hosts except (0,x) which is the internet subnet
            sensitive_hosts = [addr for addr in self.state_adapter.host_num_map
                               if addr[0] > 0]
        print(f"Target (sensitive) hosts: {sensitive_hosts}")

        results = {
            'config': {
                'scenario': self.scenario_path.name,
                'model_dir': str(self.model_dir),
                'num_episodes': num_episodes,
                'max_steps': max_steps,
                'seed': self.seed,
                'fully_obs': self.fully_obs,
                'deterministic': self.deterministic,
                'sensitive_hosts': [list(h) for h in sensitive_hosts],
            },
            'episodes': [],
            'aggregate': {},
            'action_distribution': defaultdict(int),
            'failure_modes': defaultdict(int),
            'mapping_stats': {},
        }

        total_successes = 0
        total_rewards_all = []
        total_steps_all = []
        total_times = []

        for ep in range(num_episodes):
            ep_start = time.time()
            obs, info = self.env.reset()
            flat_obs = obs if obs.ndim == 1 else obs.flatten()

            ep_reward = 0.0
            ep_steps = 0
            ep_actions = []
            ep_mapped = 0
            ep_unmapped = 0
            ep_done = False
            ep_truncated = False

            while ep_steps < max_steps and not ep_done and not ep_truncated:
                # Choose which host to target
                # Strategy: cycle through reachable hosts that are not yet compromised
                target_host = self._select_target_host(flat_obs, sensitive_hosts)
                if target_host is None:
                    # No reachable uncompromised host → stuck
                    results['failure_modes']['no_reachable_target'] += 1
                    break

                # Convert PenGym obs → SCRIPT state for the target host
                script_state = self.state_adapter.convert(flat_obs, target_host)

                # Normalize using saved training statistics
                if self.agent.use_state_norm:
                    script_state = self.agent.state_norm(script_state, update=False)

                # Policy decision
                with torch.no_grad():
                    if self.deterministic:
                        script_action = self.agent.Policy.evaluate(
                            script_state, determinate=True)
                    else:
                        action_info = self.agent.Policy.select_action(
                            script_state, explore=False,
                            is_loaded_agent=True, num_episode=0)
                        script_action = action_info[0]

                # Map SCRIPT action → PenGym action
                pengym_action = self.action_mapper.map_action(
                    script_action, target_host)

                script_action_name = "UNKNOWN"
                if script_action < len(self.action_mapper.script_actions.legal_actions):
                    script_action_name = self.action_mapper.script_actions.legal_actions[script_action].name

                if pengym_action == -1:
                    # Unmappable → use random valid action as fallback
                    pengym_action = self.action_mapper.get_random_valid_action(target_host)
                    ep_unmapped += 1
                    results['failure_modes']['unmappable_action'] += 1
                    mapped_flag = False
                else:
                    ep_mapped += 1
                    mapped_flag = True

                # Execute on PenGym
                pengym_action_int = int(pengym_action)  # Ensure native int for NASim
                pengym_action_obj = self.env.action_space.get_action(pengym_action_int)
                obs, reward, ep_done, ep_truncated, info = self.env.step(pengym_action_int)
                flat_obs = obs if obs.ndim == 1 else obs.flatten()

                ep_reward += reward
                ep_steps += 1

                # Log action
                action_record = {
                    'step': ep_steps,
                    'target_host': list(target_host),
                    'script_action_idx': script_action,
                    'script_action_name': script_action_name,
                    'pengym_action_idx': pengym_action,
                    'pengym_action_name': str(pengym_action_obj),
                    'mapped': mapped_flag,
                    'reward': float(reward),
                    'done': bool(ep_done),
                }
                ep_actions.append(action_record)
                results['action_distribution'][script_action_name] += 1

                if verbose and ep_steps <= 10:
                    status = "✓" if mapped_flag else "✗"
                    print(f"  [Ep{ep+1} Step{ep_steps}] {status} "
                          f"host={target_host} "
                          f"SCRIPT:{script_action_name} → PenGym:{pengym_action_obj} "
                          f"r={reward:.1f} done={ep_done}")

            ep_time = time.time() - ep_start

            if ep_done:
                total_successes += 1

            ep_result = {
                'episode': ep + 1,
                'success': bool(ep_done),
                'truncated': bool(ep_truncated),
                'total_reward': float(ep_reward),
                'steps': ep_steps,
                'time_seconds': round(ep_time, 2),
                'mapped_actions': ep_mapped,
                'unmapped_actions': ep_unmapped,
                'valid_action_rate': ep_mapped / max(ep_mapped + ep_unmapped, 1),
                'actions': ep_actions,
            }
            results['episodes'].append(ep_result)
            total_rewards_all.append(ep_reward)
            total_steps_all.append(ep_steps)
            total_times.append(ep_time)

            if verbose:
                status = "SUCCESS" if ep_done else "FAILED"
                print(f"  Episode {ep+1}/{num_episodes}: {status} | "
                      f"reward={ep_reward:.1f} | steps={ep_steps} | "
                      f"mapped={ep_mapped}/{ep_mapped+ep_unmapped} | "
                      f"time={ep_time:.1f}s")
                print()

        # --- Aggregate Results ---
        success_rate = total_successes / num_episodes if num_episodes > 0 else 0
        avg_reward = np.mean(total_rewards_all) if total_rewards_all else 0
        avg_steps = np.mean(total_steps_all) if total_steps_all else 0
        avg_time = np.mean(total_times) if total_times else 0

        results['aggregate'] = {
            'success_rate': round(success_rate, 4),
            'avg_reward': round(float(avg_reward), 2),
            'avg_steps': round(float(avg_steps), 2),
            'avg_time_seconds': round(float(avg_time), 2),
            'total_successes': total_successes,
            'total_episodes': num_episodes,
        }

        # Mapping stats
        results['mapping_stats'] = self.action_mapper.get_mapping_stats()
        results['action_distribution'] = dict(results['action_distribution'])
        results['failure_modes'] = dict(results['failure_modes'])

        # --- Gap Analysis ---
        results['gap_analysis'] = self._compute_gap_analysis(results)

        # Print summary
        self._print_summary(results)

        return results

    def _select_target_host(
        self,
        flat_obs: np.ndarray,
        sensitive_hosts: List[Tuple[int, int]]
    ) -> Optional[Tuple[int, int]]:
        """Select the best target host to attack.

        Priority:
        1. Reachable sensitive host that is not yet compromised
        2. Reachable non-sensitive host (for pivoting)
        3. None if no reachable host exists
        """
        # Check sensitive hosts first
        for addr in sensitive_hosts:
            seg = self.state_adapter._get_host_segment(flat_obs, addr)
            if seg is not None:
                compromised = seg[self.state_adapter._compromised_offset]
                reachable = seg[self.state_adapter._reachable_offset]
                discovered = seg[self.state_adapter._discovered_offset]
                if (reachable > 0.5 or discovered > 0.5) and compromised < 0.5:
                    return addr

        # Check any reachable host
        for addr in self.state_adapter.host_num_map:
            seg = self.state_adapter._get_host_segment(flat_obs, addr)
            if seg is not None:
                compromised = seg[self.state_adapter._compromised_offset]
                reachable = seg[self.state_adapter._reachable_offset]
                discovered = seg[self.state_adapter._discovered_offset]
                if (reachable > 0.5 or discovered > 0.5) and compromised < 0.5:
                    return addr

        return None

    def _compute_gap_analysis(self, results: Dict) -> Dict:
        """Compute sim-to-real gap metrics."""
        gap = {}

        pengym_sr = results['aggregate']['success_rate']
        pengym_reward = results['aggregate']['avg_reward']

        # Transfer ratio
        if self.baseline:
            sim_sr = self.baseline.get('experiment', {}).get('success_rate',
                     self.baseline.get('training', {}).get('final_train_success_rate', [1.0])[-1])
            sim_reward = self.baseline.get('training', {}).get('best_return', 6640)
            gap['sim_success_rate'] = float(sim_sr) if isinstance(sim_sr, (int, float)) else 1.0
            gap['sim_best_return'] = float(sim_reward) if isinstance(sim_reward, (int, float)) else 0
            gap['transfer_ratio'] = pengym_sr / max(gap['sim_success_rate'], 1e-6)
            gap['reward_ratio'] = pengym_reward / max(gap['sim_best_return'], 1e-6)
        else:
            gap['sim_success_rate'] = 1.0  # Assumed from experiment_summary
            gap['sim_best_return'] = 6640
            gap['transfer_ratio'] = pengym_sr / 1.0
            gap['reward_ratio'] = pengym_reward / 6640

        # Action validity rate
        total_mapped = results['mapping_stats'].get('total_mapped_calls', 0)
        total_unmapped = results['mapping_stats'].get('total_unmapped_calls', 0)
        total_calls = total_mapped + total_unmapped
        gap['valid_action_rate'] = total_mapped / max(total_calls, 1)

        # Mapping coverage
        gap['mapping_coverage'] = results['mapping_stats'].get('coverage_pct', 0) / 100

        # Recommendation
        if pengym_sr >= 0.3:
            gap['recommendation'] = "PROCEED to Strategy C — transfer is viable"
        elif pengym_sr >= 0.05:
            gap['recommendation'] = "Strategy C possible with state standardization improvements"
        else:
            gap['recommendation'] = "Analyze root cause before Strategy C — gap too large"

        gap['pengym_success_rate'] = pengym_sr
        gap['pengym_avg_reward'] = pengym_reward

        return gap

    def _print_summary(self, results: Dict):
        """Print evaluation summary."""
        agg = results['aggregate']
        gap = results.get('gap_analysis', {})
        mapping = results.get('mapping_stats', {})

        print("\n" + "=" * 70)
        print("STRATEGY A: SIM-TO-REAL EVALUATION RESULTS")
        print("=" * 70)

        print(f"\n{'Metric':<35} {'PenGym':>12} {'Sim Baseline':>12}")
        print("-" * 60)
        print(f"{'Success Rate':<35} {agg['success_rate']:>12.1%} {gap.get('sim_success_rate', '?'):>12}")
        print(f"{'Average Reward':<35} {agg['avg_reward']:>12.1f} {gap.get('sim_best_return', '?'):>12}")
        print(f"{'Average Steps':<35} {agg['avg_steps']:>12.1f} {'~175':>12}")
        print(f"{'Average Time/Episode (s)':<35} {agg['avg_time_seconds']:>12.1f} {'<0.01':>12}")

        print(f"\n--- Transfer Metrics ---")
        print(f"{'Transfer Ratio (SR)':<35} {gap.get('transfer_ratio', 0):>12.4f}")
        print(f"{'Valid Action Rate':<35} {gap.get('valid_action_rate', 0):>12.1%}")
        print(f"{'Mapping Coverage':<35} {gap.get('mapping_coverage', 0):>12.1%}")

        print(f"\n--- Action Distribution ---")
        for action_name, count in sorted(results.get('action_distribution', {}).items(),
                                          key=lambda x: -x[1])[:10]:
            print(f"  {action_name:<40} {count:>6}")

        print(f"\n--- Failure Modes ---")
        for mode, count in sorted(results.get('failure_modes', {}).items(),
                                   key=lambda x: -x[1]):
            print(f"  {mode:<40} {count:>6}")

        print(f"\n--- Recommendation ---")
        print(f"  {gap.get('recommendation', 'N/A')}")

        print("\n" + "=" * 70)

    def save_results(self, output_path: Path, results: Dict):
        """Save evaluation results to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert non-serializable types
        def make_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, tuple):
                return list(obj)
            return obj

        serializable = json.loads(
            json.dumps(results, default=make_serializable, indent=2)
        )

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

        print(f"\n[Results saved to {output_path}]")
