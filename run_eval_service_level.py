"""
Service-Level Sim-to-Real Evaluator.

Evaluates a SCRIPT agent trained with service-level actions on PenGym.
Because both the agent and PenGym use service-level abstractions,
the action mapping is nearly 1:1 (vs. 3.4% with CVE-level).

Usage:
  cd d:\\NCKH\\fusion\\pentest
  .\\venv\\Scripts\\python.exe run_eval_service_level.py --scenario tiny.yml --model-dir outputs/models_service_level/chain_1 --episodes 20
"""

import sys
import os
import json
import time
import argparse
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from src import SCENARIOS_DIR, MODELS_DIR, LOGS_DIR, get_scenario_path
from src.agent.policy.config import PPO_Config
from src.agent.actions import Action
from src.agent.host import StateEncoder
from src.agent.actions.service_action_space import ServiceActionSpace


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate service-level SCRIPT agent on PenGym',
    )
    parser.add_argument('--scenario', type=str, default='tiny.yml',
                        help='PenGym scenario YAML file')
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Directory with trained service-level model')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic', action='store_true', default=True)
    parser.add_argument('--no-deterministic', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--fully-obs', action='store_true', default=True)
    parser.add_argument('--partially-obs', action='store_true')
    return parser.parse_args()


class ServiceLevelEvaluator:
    """Evaluate service-level SCRIPT agent on PenGym NASim."""

    def __init__(self, model_dir: Path, scenario_path: Path,
                 seed: int = 42, fully_obs: bool = True,
                 deterministic: bool = True):
        self.model_dir = Path(model_dir)
        self.scenario_path = Path(scenario_path)
        self.seed = seed
        self.fully_obs = fully_obs
        self.deterministic = deterministic

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.env = None
        self.agent = None
        self.state_adapter = None
        self.action_mapper = None
        self.sas = None

    def setup(self):
        """Initialize all components."""
        print("\n" + "=" * 60)
        print("Service-Level Sim-to-Real Evaluation Setup")
        print("=" * 60)

        # 1. Service Action Space
        print("\n[1/5] Creating Service Action Space...")
        self.sas = ServiceActionSpace(action_class=Action)
        print(self.sas.describe())

        # 2. PenGym environment
        print(f"\n[2/5] Creating PenGym environment: {self.scenario_path}...")
        from src.envs import load as load_pengym_env, utilities
        utilities.ENABLE_PENGYM = False
        utilities.ENABLE_NASIM = True

        self.env = load_pengym_env(
            str(self.scenario_path),
            fully_obs=self.fully_obs,
            flat_actions=True,
            flat_obs=True,
        )
        self.env.action_space.seed(self.seed)
        self.scenario = utilities.scenario

        obs, _ = self.env.reset()
        print(f"  Action space: {self.env.action_space.n} flat actions")
        print(f"  Observation shape: {obs.shape}")

        # 3. State Adapter
        print(f"\n[3/5] Initializing State Adapter...")
        from src.envs.adapters.state_adapter import PenGymStateAdapter
        self.state_adapter = PenGymStateAdapter(self.scenario)
        print(f"  {self.state_adapter.describe()}")

        # 4. Service Action Mapper (NOT CVE-level mapper)
        print(f"\n[4/5] Building Service Action Mapper...")
        from src.envs.adapters.service_action_mapper import ServiceActionMapper
        self.action_mapper = ServiceActionMapper(self.sas, self.env)
        print(f"  Mapping stats: {self.action_mapper.get_mapping_stats()}")

        # 5. Load agent
        print(f"\n[5/5] Loading service-level model from {self.model_dir}...")
        self._load_agent()

        print("\n" + "=" * 60)
        print("Setup complete. Ready for evaluation.")
        print("=" * 60 + "\n")

    def _load_agent(self):
        """Load agent with service-level action dim."""
        from src.agent.policy.PPO import PPO_agent
        from src.agent.policy.common import Normalization

        config = PPO_Config()
        self.policy = PPO_agent(
            cfg=config,
            logger=None,
            use_wandb=False,
            state_dim=StateEncoder.state_space,
            action_dim=self.sas.action_dim,
        )

        # Load weights
        actor_path = self.model_dir / 'PPO-actor.pt'
        critic_path = self.model_dir / 'PPO-critic.pt'
        if actor_path.exists():
            self.policy.actor.load_state_dict(
                torch.load(actor_path, map_location=self.policy.device,
                           weights_only=True))
            print(f"  Actor loaded: {actor_path}")
        else:
            print(f"  WARNING: No actor found at {actor_path}")

        if critic_path.exists():
            self.policy.critic.load_state_dict(
                torch.load(critic_path, map_location=self.policy.device,
                           weights_only=True))

        # Normalization
        self.use_state_norm = True
        self.state_norm = Normalization(shape=StateEncoder.state_space)
        mean_path = self.model_dir / 'PPO-norm_mean.pt'
        std_path = self.model_dir / 'PPO-norm_std.pt'
        if mean_path.exists() and std_path.exists():
            self.state_norm.running_ms.mean = torch.load(
                mean_path, map_location='cpu', weights_only=False)
            self.state_norm.running_ms.std = torch.load(
                std_path, map_location='cpu', weights_only=False)
            print(f"  State normalization loaded")
        else:
            self.use_state_norm = False
            print(f"  No normalization found, proceeding without")

    def evaluate(self, num_episodes=20, max_steps=100, verbose=True):
        """Run evaluation episodes."""
        assert self.env is not None, "Call setup() first"

        sensitive_hosts = self.state_adapter.get_sensitive_hosts()
        if not sensitive_hosts:
            sensitive_hosts = [addr for addr in self.state_adapter.host_num_map
                               if addr[0] > 0]
        print(f"Target (sensitive) hosts: {sensitive_hosts}")

        results = {
            'config': {
                'scenario': self.scenario_path.name,
                'model_dir': str(self.model_dir),
                'num_episodes': num_episodes,
                'max_steps': max_steps,
                'action_space': 'service_level',
                'action_dim': self.sas.action_dim,
            },
            'episodes': [],
            'aggregate': {},
        }

        all_returns = []
        all_successes = []
        all_steps = []
        action_counts = {}

        for ep in range(num_episodes):
            obs, _ = self.env.reset()
            ep_return = 0
            ep_steps = 0
            ep_actions = []
            done = False
            truncated = False

            while not done and not truncated and ep_steps < max_steps:
                # Select target host
                target_host = self._select_target(obs, sensitive_hosts)
                if target_host is None:
                    break

                # Convert PenGym obs → SCRIPT state
                state = self.state_adapter.convert(obs, target_host)
                if self.use_state_norm:
                    state = self.state_norm(state, update=False)

                # Policy selects SERVICE-LEVEL action
                with torch.no_grad():
                    if self.deterministic:
                        service_action_idx = self.policy.evaluate(state)
                    else:
                        action_info = self.policy.select_action(
                            observation=state, explore=False,
                            is_loaded_agent=True, num_episode=0)
                        service_action_idx = action_info[0]

                service_action_name = self.sas.action_names[service_action_idx]
                action_counts[service_action_name] = action_counts.get(service_action_name, 0) + 1

                # Map to PenGym action
                pengym_action = self.action_mapper.map_action(
                    service_action_idx, target_host)

                if pengym_action == -1:
                    pengym_action = self.action_mapper.get_random_valid_action(target_host)
                    ep_actions.append(f"{service_action_name}→fallback")
                else:
                    ep_actions.append(f"{service_action_name}→ok")

                obs, reward, done, truncated, info = self.env.step(int(pengym_action))
                ep_return += reward
                ep_steps += 1

            all_returns.append(ep_return)
            all_successes.append(1 if done and not truncated else 0)
            all_steps.append(ep_steps)

            if verbose and (ep < 5 or ep == num_episodes - 1):
                status = "SUCCESS" if done and not truncated else "FAILED"
                unique_actions = list(set(a.split('→')[0] for a in ep_actions))
                print(f"  Ep {ep+1:3d}: {status}, reward={ep_return:.1f}, "
                      f"steps={ep_steps}, actions={unique_actions[:5]}")

            results['episodes'].append({
                'episode': ep + 1,
                'return': ep_return,
                'success': all_successes[-1],
                'steps': ep_steps,
            })

        # Aggregate
        success_rate = np.mean(all_successes)
        avg_return = np.mean(all_returns)
        avg_steps = np.mean(all_steps)
        mapper_stats = self.action_mapper.get_mapping_stats()

        results['aggregate'] = {
            'success_rate': success_rate,
            'avg_return': avg_return,
            'avg_steps': avg_steps,
            'std_return': np.std(all_returns),
            'mapping_coverage': mapper_stats['coverage_pct'],
            'valid_action_rate': mapper_stats['valid_call_rate'],
            'action_distribution': action_counts,
        }

        print(f"\n{'='*50}")
        print(f"Results ({num_episodes} episodes):")
        print(f"  Success Rate:     {success_rate*100:.1f}%")
        print(f"  Avg Return:       {avg_return:.1f} ± {np.std(all_returns):.1f}")
        print(f"  Avg Steps:        {avg_steps:.1f}")
        print(f"  Mapping Coverage: {mapper_stats['coverage_pct']:.1f}%")
        print(f"  Valid Action Rate:{mapper_stats['valid_call_rate']:.1f}%")
        print(f"  Action dist:      {action_counts}")
        print(f"{'='*50}")

        return results

    def _select_target(self, obs, sensitive_hosts):
        """Select target host using the state adapter."""
        reachable_hosts = self.state_adapter.get_reachable_hosts(obs)

        # Prefer uncompromised sensitive hosts
        for host in sensitive_hosts:
            host_data = self.state_adapter.get_host_data(obs, host)
            if host_data and host_data.get('reachable') and not host_data.get('compromised'):
                return host

        # Any reachable uncompromised host
        for host in reachable_hosts:
            host_data = self.state_adapter.get_host_data(obs, host)
            if host_data and not host_data.get('compromised'):
                return host

        # Any sensitive host (even if not yet verified reachable)
        if sensitive_hosts:
            return sensitive_hosts[0]

        # Any non-internet host
        for host in self.state_adapter.host_num_map:
            if host[0] > 0:
                return host

        return None


def main():
    args = parse_args()

    scenario_path = Path(PROJECT_ROOT) / 'data' / 'scenarios' / args.scenario
    if not scenario_path.exists():
        # Try PenGym scenarios
        pengym_scenario = PROJECT_ROOT.parent / 'PenGym' / 'database' / 'scenarios' / args.scenario
        if pengym_scenario.exists():
            scenario_path = pengym_scenario
        else:
            print(f"Error: Scenario not found: {scenario_path}")
            sys.exit(1)

    fully_obs = args.fully_obs and not args.partially_obs
    deterministic = args.deterministic and not args.no_deterministic

    evaluator = ServiceLevelEvaluator(
        model_dir=Path(args.model_dir),
        scenario_path=scenario_path,
        seed=args.seed,
        fully_obs=fully_obs,
        deterministic=deterministic,
    )
    evaluator.setup()

    results = evaluator.evaluate(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        verbose=args.verbose or args.episodes <= 10,
    )

    # Save results
    output_dir = Path(args.output_dir) if args.output_dir else (
        LOGS_DIR.parent / 'logs_service_level' / 'strategy_a_eval')
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_name = args.scenario.replace('.yml', '').replace('.yaml', '')
    results_path = output_dir / f'eval_{scenario_name}.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
