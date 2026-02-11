#!/usr/bin/env python3
"""
Training script with Service-Level Action Abstraction.

Instead of training the PPO agent with 2064 CVE-level actions,
this script uses a 16-dim service-level action space:

  [port_scan, service_scan, os_scan, web_scan,
   exploit_ssh, exploit_ftp, exploit_http, exploit_smb, exploit_smtp,
   exploit_rdp, exploit_sql, exploit_java_rmi, exploit_misc,
   privesc_tomcat, privesc_schtask, privesc_daclsvc]

The RL policy learns WHICH TYPE of exploit to use (strategic).
The CVESelector picks WHICH SPECIFIC CVE to execute (tactical).

Benefits:
  - Action dim: 16 (fixed) vs 2064 (grows with CVE database)
  - 100% compatible with PenGym service-level actions
  - Ready for Strategy C dual-environment training

Usage:
  cd d:\\NCKH\\fusion\\pentest
  .\\venv\\Scripts\\python.exe run_train_service_level.py --scenario chain_1.json --episodes 1000
  .\\venv\\Scripts\\python.exe run_train_service_level.py --scenario chain_1.json --mode eval --model-dir outputs/models_service_level/chain_1
"""

import sys
import os
import json
import time
import random
import argparse
import datetime
import numpy as np
import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

# ============================================================
# Path setup
# ============================================================
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from src import SCENARIOS_DIR, MODELS_DIR, LOGS_DIR, get_scenario_path
from src.agent.policy.config import PPO_Config
from src.agent.host import HOST, StateEncoder
from src.agent.actions import Action
from src.agent.actions.service_action_space import ServiceActionSpace


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train SCRIPT agent with service-level action space',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    parser.add_argument('--scenario', type=str, required=True,
                        help='Scenario file name (e.g., chain_1.json)')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--eval-step-limit', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model-dir', type=str, default=None,
                        help='Model dir for eval mode or resume training')
    parser.add_argument('--output-suffix', type=str, default='service_level',
                        help='Suffix for output directories')
    parser.add_argument('--cve-strategy', choices=['rank', 'random', 'round_robin', 'match'],
                        default='match', help='CVE selection strategy (default: match)')
    parser.add_argument('--verbose', action='store_true')

    # CL parameters
    parser.add_argument('--cl-method', type=str, default=None,
                        choices=['script', 'finetune', 'ft'])
    parser.add_argument('--config-file', type=str, default=None)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ServiceLevelAgent:
    """Agent wrapper that uses service-level actions with CVE resolution.

    The PPO policy outputs a service-level action (dim=16).
    The CVESelector resolves it to a specific CVE exploit for execution.
    """

    def __init__(self, config: PPO_Config, seed: int,
                 service_action_space: ServiceActionSpace,
                 logger: SummaryWriter = None,
                 use_wandb: bool = False,
                 cve_strategy: str = 'rank'):
        from src.agent.policy.PPO import PPO_agent
        from src.agent.policy.common import Normalization, RewardScaling

        self.sas = service_action_space
        self.config = config
        self.cve_strategy = cve_strategy

        # Create PPO agent with service-level action dim
        self.Policy = PPO_agent(
            cfg=config,
            logger=logger,
            use_wandb=use_wandb,
            state_dim=StateEncoder.state_space,   # 1538 (same)
            action_dim=self.sas.action_dim,        # 16 (new!)
        )

        # Normalization / scaling
        self.use_state_norm = config.use_state_norm
        self.use_reward_scaling = config.use_reward_scaling
        self.use_lr_decay = config.use_lr_decay

        if self.use_state_norm:
            self.state_norm = Normalization(shape=StateEncoder.state_space)
        if self.use_reward_scaling:
            self.reward_scaling = RewardScaling(shape=1, gamma=config.gamma)

        # Tracking
        self.num_episodes = 0
        self.total_training_step = 0
        self.best_return = -float('inf')
        self.best_episode = -1
        self.first_hit_step = -1
        self.first_hit_eps = -1
        self.convergence_eps = -1
        self.hit_convergence_gap_eps = -1
        self.eval_rewards = 0
        self.eval_success_rate = 0

        # Convergence tracking
        self.convergence_judge_done_num = 20
        self.convergence_judge_done_list = [False] * self.convergence_judge_done_num

    def run_train_episode(self, target_list, explore=False, update_norm=True):
        """Run one training episode over all targets.

        Same structure as Agent.run_train_episode but uses service-level actions.
        """
        eps_steps = 0
        episode_return = 0
        success_num = 0
        self.num_episodes += 1
        task_num_episodes = self.num_episodes

        if self.use_reward_scaling:
            self.reward_scaling.reset()

        for target in target_list:
            done = 0
            target_step = 0
            o = target.reset()

            if self.use_state_norm:
                o = self.state_norm(o, update=update_norm)

            while not done:
                if target_step >= self.config.step_limit:
                    break

                # ---- Policy selects SERVICE-LEVEL action (0..15) ----
                action_info = self.Policy.select_action(
                    observation=o, explore=explore,
                    is_loaded_agent=False,
                    num_episode=task_num_episodes,
                )
                service_action_idx = action_info[0]  # 0..15

                # ---- CVE Selector resolves to specific exploit ----
                cve_action_idx = self.sas.select_cve(
                    service_action_idx,
                    host_info=target.info,
                    strategy=self.cve_strategy,
                    env_data=target.env_data,
                )

                # ---- Execute the resolved CVE action ----
                next_o, r, done, result = target.perform_action(cve_action_idx)
                self.total_training_step += 1
                eps_steps += 1
                target_step += 1
                episode_return += r

                if done:
                    success_num += 1
                    dw = True
                    if self.first_hit_step < 0:
                        self.first_hit_step = self.total_training_step
                    if self.first_hit_eps < 0:
                        self.first_hit_eps = task_num_episodes
                else:
                    dw = False

                # Convergence tracking
                self.convergence_judge_done_list[
                    (task_num_episodes - 1) % self.convergence_judge_done_num
                ] = dw
                if self.hit_convergence_gap_eps < 0 and all(self.convergence_judge_done_list):
                    self.hit_convergence_gap_eps = task_num_episodes - self.first_hit_eps
                    self.convergence_eps = task_num_episodes

                if self.use_state_norm:
                    next_o = self.state_norm(next_o, update=update_norm)
                if self.use_reward_scaling:
                    r = self.reward_scaling(r)[0]

                # Store transition (with service-level action)
                self.Policy.store_transtion(
                    observation=o,
                    action=action_info,  # service-level action (0..15)
                    reward=r,
                    next_observation=next_o,
                    done=dw,
                )

                # Update policy
                if not explore:
                    self.Policy.update_policy(
                        num_episode=task_num_episodes,
                        train_steps=self.total_training_step,
                    )
                    if self.use_lr_decay:
                        rate = max(
                            1 - task_num_episodes / self.config.train_eps,
                            self.config.min_decay_lr,
                        )
                        self.Policy.lr_decay(rate=rate)
                o = next_o

        success_rate = float(format(success_num / len(target_list), '.3f'))
        if episode_return >= self.best_return:
            self.best_return = episode_return
            self.best_episode = self.num_episodes
        return episode_return, eps_steps, success_rate

    def evaluate(self, target_list, step_limit=10, verbose=True):
        """Evaluate current policy on targets."""
        success_num = 0
        total_rewards = 0
        results = []

        for target in target_list:
            o = target.reset()
            if self.use_state_norm:
                o = self.state_norm(o, update=False)

            done = 0
            steps = 0
            task_return = 0

            while not done and steps < step_limit:
                with torch.no_grad():
                    service_action_idx = self.Policy.evaluate(o)

                cve_action_idx = self.sas.select_cve(
                    service_action_idx,
                    host_info=target.info,
                    strategy=self.cve_strategy,
                    env_data=target.env_data,
                )

                next_o, r, done, result = target.perform_action(cve_action_idx)
                steps += 1
                task_return += r

                if self.use_state_norm:
                    next_o = self.state_norm(next_o, update=False)
                o = next_o

            if done:
                success_num += 1
            total_rewards += task_return

            if verbose:
                status = "SUCCESS" if done else "FAILED"
                print(f"  Target {target.ip}: {status}, reward={task_return}, steps={steps}")

        success_rate = success_num / len(target_list) if target_list else 0
        self.eval_rewards = total_rewards
        self.eval_success_rate = success_rate
        return total_rewards, success_rate

    def save(self, model_dir: Path):
        """Save model, optimizer, and normalization parameters."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.Policy.actor.state_dict(),
                   model_dir / 'PPO-actor.pt')
        torch.save(self.Policy.critic.state_dict(),
                   model_dir / 'PPO-critic.pt')

        if self.use_state_norm:
            torch.save(self.state_norm.running_ms.mean,
                       model_dir / 'PPO-norm_mean.pt')
            torch.save(self.state_norm.running_ms.std,
                       model_dir / 'PPO-norm_std.pt')

        # Save service action space config
        sas_info = self.sas.summary_dict()
        sas_info['cve_strategy'] = self.cve_strategy
        with open(model_dir / 'service_action_space.json', 'w') as f:
            json.dump(sas_info, f, indent=2)

        print(f"Model saved to {model_dir}")

    def load(self, model_dir: Path):
        """Load pre-trained model."""
        model_dir = Path(model_dir)

        actor_path = model_dir / 'PPO-actor.pt'
        critic_path = model_dir / 'PPO-critic.pt'

        if actor_path.exists():
            self.Policy.actor.load_state_dict(
                torch.load(actor_path, map_location=self.Policy.device,
                           weights_only=True))
        if critic_path.exists():
            self.Policy.critic.load_state_dict(
                torch.load(critic_path, map_location=self.Policy.device,
                           weights_only=True))

        # Load normalization
        mean_path = model_dir / 'PPO-norm_mean.pt'
        std_path = model_dir / 'PPO-norm_std.pt'
        if self.use_state_norm and mean_path.exists() and std_path.exists():
            self.state_norm.running_ms.mean = torch.load(
                mean_path, map_location='cpu', weights_only=False)
            self.state_norm.running_ms.std = torch.load(
                std_path, map_location='cpu', weights_only=False)

        print(f"Model loaded from {model_dir}")


def main():
    args = parse_args()
    set_seed(args.seed)

    # Resolve paths
    scenario_path = get_scenario_path(args.scenario)
    if not scenario_path.exists():
        print(f"Error: Scenario not found: {scenario_path}")
        sys.exit(1)

    suffix = args.output_suffix
    log_dir = LOGS_DIR.parent / f'logs_{suffix}' / args.scenario.replace('.json', '').replace('.yml', '')
    model_dir_out = MODELS_DIR.parent / f'models_{suffix}' / args.scenario.replace('.json', '').replace('.yml', '')
    tb_dir = LOGS_DIR.parent / f'tensorboard_{suffix}' / args.scenario.replace('.json', '').replace('.yml', '')

    # ============================================================
    # Create Service-Level Action Space
    # ============================================================
    sas = ServiceActionSpace(action_class=Action)
    print(sas.describe())

    # ============================================================
    # Load scenario and create HOST targets
    # ============================================================
    with open(scenario_path, 'r', encoding='utf-8') as f:
        environment_data = json.load(f)

    target_list = []
    for host_data in environment_data:
        ip = host_data['ip']
        vul = host_data['vulnerability'][0]
        if vul not in Action.Vul_cve_set:
            print(f"[WARN] host vul {vul} not in Vul_cve_set, skipping")
            continue
        t = HOST(ip, env_data=host_data, env_file=scenario_path)
        target_list.append(t)

    print(f"\n[ENV] {len(target_list)} targets from {scenario_path.name}")
    print(f"[ENV] State dim: {StateEncoder.state_space}, "
          f"Action dim: {sas.action_dim} (service-level)")
    print(f"[ENV] (Original CVE-level action dim: {Action.action_space})")

    # ============================================================
    # Create agent
    # ============================================================
    config = PPO_Config(
        train_eps=args.episodes,
        step_limit=args.max_steps,
        eval_step_limit=args.eval_step_limit,
    )

    if args.mode == 'train':
        log_dir.mkdir(parents=True, exist_ok=True)
        model_dir_out.mkdir(parents=True, exist_ok=True)
        tb_dir.mkdir(parents=True, exist_ok=True)

        tb_logger = SummaryWriter(log_dir=str(tb_dir))

        agent = ServiceLevelAgent(
            config=config,
            seed=args.seed,
            service_action_space=sas,
            logger=tb_logger,
            cve_strategy=args.cve_strategy,
        )

        # Load existing model if specified (for resuming)
        if args.model_dir:
            agent.load(args.model_dir)

        print(f"\n{'='*60}")
        print(f"Service-Level Training: {args.episodes} episodes")
        print(f"Action space: {sas.action_dim} service-level actions")
        print(f"CVE strategy: {args.cve_strategy}")
        print(f"{'='*60}\n")

        train_start = time.time()

        train_rewards = []
        train_steps_list = []
        train_sr = []
        eval_rewards_list = []
        eval_sr_list = []
        eval_freq = 5

        for ep in range(1, args.episodes + 1):
            ep_return, ep_steps, success_rate = agent.run_train_episode(target_list)
            train_rewards.append(ep_return)
            train_steps_list.append(ep_steps)
            train_sr.append(success_rate)

            tb_logger.add_scalar('Train/Reward', ep_return, ep)
            tb_logger.add_scalar('Train/Steps', ep_steps, ep)
            tb_logger.add_scalar('Train/SuccessRate', success_rate, ep)

            if ep % eval_freq == 0:
                eval_r, eval_s = agent.evaluate(target_list,
                                                step_limit=args.eval_step_limit,
                                                verbose=False)
                eval_rewards_list.append(eval_r)
                eval_sr_list.append(eval_s)
                tb_logger.add_scalar('Eval/Reward', eval_r, ep)
                tb_logger.add_scalar('Eval/SuccessRate', eval_s, ep)

            if ep % 50 == 0 or ep == 1:
                avg_r = np.mean(train_rewards[-50:])
                avg_sr = np.mean(train_sr[-50:])
                eval_str = f", eval_sr={eval_sr_list[-1]*100:.1f}%" if eval_sr_list else ""
                print(f"  [ep {ep:4d}/{args.episodes}] avg_r={avg_r:.1f}, "
                      f"avg_sr={avg_sr*100:.1f}%{eval_str}")

        train_end = time.time()
        train_time = train_end - train_start

        # Save model
        agent.save(model_dir_out)

        # Final evaluation
        print(f"\n{'='*40}")
        print("Final Evaluation:")
        print(f"{'='*40}")
        final_r, final_sr = agent.evaluate(target_list, step_limit=args.eval_step_limit)
        print(f"\nFinal: reward={final_r}, success_rate={final_sr*100:.1f}%")

        # Save experiment summary
        summary = {
            'experiment': {
                'type': 'service_level_training',
                'scenario': str(scenario_path),
                'action_space': 'service_level',
                'action_dim': sas.action_dim,
                'original_cve_action_dim': Action.action_space,
                'state_dim': StateEncoder.state_space,
                'num_targets': len(target_list),
                'episodes': args.episodes,
                'max_steps': args.max_steps,
                'seed': args.seed,
                'cve_strategy': args.cve_strategy,
            },
            'convergence': {
                'first_hit_episode': agent.first_hit_eps,
                'first_hit_step': agent.first_hit_step,
                'converged_episode': agent.convergence_eps,
            },
            'training': {
                'total_time_seconds': round(train_time, 2),
                'total_steps': agent.total_training_step,
                'best_return': agent.best_return,
                'best_episode': agent.best_episode,
                'final_success_rate': final_sr,
                'final_eval_reward': final_r,
            },
            'service_action_space': sas.summary_dict(),
            'paths': {
                'model_dir': str(model_dir_out),
                'log_dir': str(log_dir),
                'tensorboard_dir': str(tb_dir),
            },
        }

        summary_path = log_dir / 'experiment_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSummary saved to {summary_path}")

        tb_logger.close()

    elif args.mode == 'eval':
        model_dir = Path(args.model_dir) if args.model_dir else model_dir_out
        if not model_dir.exists():
            print(f"Error: Model not found: {model_dir}")
            sys.exit(1)

        agent = ServiceLevelAgent(
            config=config,
            seed=args.seed,
            service_action_space=sas,
            cve_strategy=args.cve_strategy,
        )
        agent.load(model_dir)

        print(f"\n{'='*60}")
        print(f"Service-Level Evaluation")
        print(f"Model: {model_dir}")
        print(f"{'='*60}\n")

        total_r, sr = agent.evaluate(target_list, step_limit=args.eval_step_limit)
        print(f"\nResult: reward={total_r}, success_rate={sr*100:.1f}%")

    print("\nDone!")


if __name__ == '__main__':
    main()
