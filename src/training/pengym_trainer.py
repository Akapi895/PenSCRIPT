"""
PenGymTrainer — PPO training loop over PenGym via SingleHostPenGymWrapper.

Mirrors ``ServiceLevelAgent``'s interface but operates on the *real* PenGym
environment (NASim simulation mode) through the wrapper, rather than on
SCRIPT-simulated ``HOST`` targets.

Two training modes
------------------
1. **Single-scenario** (``train()``):
   Train on one NASim YAML. The wrapper auto-cycles targets.

2. **Curriculum** (``train_curriculum()``):
   Use a ``CurriculumController`` that supplies scenarios of increasing
   difficulty.  The trainer switches scenarios via ``wrapper.load_scenario()``
   when the controller advances phases.

Design reference: docs/pengym_integration_architecture.md §5.2
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.agent.policy.config import PPO_Config
from src.agent.policy.PPO import PPO_agent
from src.agent.policy.common import Normalization, RewardScaling
from src.envs.wrappers.single_host_wrapper import SingleHostPenGymWrapper
from src.envs.wrappers.reward_normalizer import (
    LinearNormalizer,
    IdentityNormalizer,
    RewardNormalizer,
)
from src.envs.wrappers.target_selector import (
    PrioritySensitiveSelector,
    TargetSelector,
)


class PenGymTrainer:
    """Train a PPO agent on PenGym through ``SingleHostPenGymWrapper``.

    Attributes
    ----------
    wrapper : SingleHostPenGymWrapper
        The bridge to PenGym (created from *initial_scenario*).
    Policy : PPO_agent
        Actor-critic policy.
    state_dim : int
        Always 1538.
    action_dim : int
        Always 16.
    """

    # -----------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------

    def __init__(
        self,
        initial_scenario: str,
        config: Optional[PPO_Config] = None,
        seed: int = 42,
        reward_normalizer: Optional[RewardNormalizer] = None,
        target_selector: Optional[TargetSelector] = None,
        tb_dir: Optional[str] = None,
        use_wandb: bool = False,
    ):
        """
        Args:
            initial_scenario: Path to a NASim scenario YAML file.
            config: PPO hyper-parameters. ``None`` → sensible defaults.
            seed: Random seed.
            reward_normalizer: Reward scaling strategy (default Linear).
            target_selector: Host picking strategy (default PrioritySensitive).
            tb_dir: TensorBoard log directory. ``None`` → disabled.
            use_wandb: Enable W&B logging.
        """
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # --- Config ---
        self.config = config if config is not None else PPO_Config()

        # --- Wrapper ---
        self.wrapper = SingleHostPenGymWrapper(
            scenario_path=initial_scenario,
            fully_obs=True,
            seed=seed,
            reward_normalizer=reward_normalizer or LinearNormalizer(),
            target_selector=target_selector or PrioritySensitiveSelector(),
            auto_select_target=True,
        )

        self.state_dim = self.wrapper.state_dim    # 1538
        self.action_dim = self.wrapper.action_dim  # 16

        # --- TensorBoard ---
        self._tb: Optional[SummaryWriter] = None
        if tb_dir:
            Path(tb_dir).mkdir(parents=True, exist_ok=True)
            self._tb = SummaryWriter(log_dir=tb_dir)

        # --- PPO ---
        self.Policy = PPO_agent(
            cfg=self.config,
            logger=self._tb,
            use_wandb=use_wandb,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
        )

        # --- Normalization / Scaling (mirrors ServiceLevelAgent) ---
        self.use_state_norm = self.config.use_state_norm
        self.use_reward_scaling = self.config.use_reward_scaling
        self.use_lr_decay = self.config.use_lr_decay

        if self.use_state_norm:
            self.state_norm = Normalization(shape=self.state_dim)
        if self.use_reward_scaling:
            self.reward_scaling = RewardScaling(shape=1, gamma=self.config.gamma)

        # --- Tracking ---
        self.num_episodes = 0
        self.total_training_steps = 0
        self.best_return = -float("inf")
        self.best_episode = -1
        self.first_hit_step = -1
        self.first_hit_eps = -1
        self.convergence_eps = -1

        self._convergence_window = 20
        self._convergence_flags = deque(
            [False] * self._convergence_window, maxlen=self._convergence_window
        )

    # -----------------------------------------------------------------
    # Single-scenario training
    # -----------------------------------------------------------------

    def train(
        self,
        num_episodes: int,
        eval_freq: int = 5,
        log_freq: int = 50,
        model_dir: Optional[str] = None,
        save_freq: int = 200,
    ) -> Dict:
        """Train for *num_episodes* on the current scenario.

        Args:
            num_episodes: Total training episodes.
            eval_freq: Evaluate every N episodes.
            log_freq: Print progress every N episodes.
            model_dir: Where to save checkpoints. ``None`` → no saving.
            save_freq: Save a checkpoint every N episodes.

        Returns:
            dict with ``train_rewards``, ``train_sr``, ``eval_sr``,
            ``train_time_s``, ``total_steps``.
        """
        if model_dir:
            Path(model_dir).mkdir(parents=True, exist_ok=True)

        train_rewards: List[float] = []
        train_sr: List[float] = []
        eval_sr: List[float] = []
        t0 = time.time()

        for ep in range(1, num_episodes + 1):
            ep_return, ep_steps, sr = self._run_episode(explore=False)
            train_rewards.append(ep_return)
            train_sr.append(sr)

            if self._tb:
                self._tb.add_scalar("Train/Reward", ep_return, ep)
                self._tb.add_scalar("Train/Steps", ep_steps, ep)
                self._tb.add_scalar("Train/SuccessRate", sr, ep)

            # --- Evaluation ---
            if ep % eval_freq == 0:
                _, e_sr = self.evaluate(verbose=False)
                eval_sr.append(e_sr)
                if self._tb:
                    self._tb.add_scalar("Eval/SuccessRate", e_sr, ep)

            # --- Logging ---
            if ep % log_freq == 0 or ep == 1:
                avg_r = np.mean(train_rewards[-log_freq:])
                avg_sr = np.mean(train_sr[-log_freq:])
                ev_str = (
                    f", eval_sr={eval_sr[-1]*100:.1f}%" if eval_sr else ""
                )
                print(
                    f"  [ep {ep:4d}/{num_episodes}] avg_r={avg_r:.1f}, "
                    f"avg_sr={avg_sr*100:.1f}%{ev_str}"
                )

            # --- Checkpoint ---
            if model_dir and ep % save_freq == 0:
                self.save(model_dir)

        elapsed = time.time() - t0

        # Final save
        if model_dir:
            self.save(model_dir)

        return {
            "train_rewards": train_rewards,
            "train_sr": train_sr,
            "eval_sr": eval_sr,
            "train_time_s": round(elapsed, 2),
            "total_steps": self.total_training_steps,
        }

    # -----------------------------------------------------------------
    # Curriculum training
    # -----------------------------------------------------------------

    def train_curriculum(
        self,
        controller,
        eval_freq: int = 5,
        log_freq: int = 50,
        model_dir: Optional[str] = None,
    ) -> Dict:
        """Train under a ``CurriculumController``.

        The controller determines which scenario to use and when to
        advance to the next tier.

        Args:
            controller: ``CurriculumController`` or ``FlatController``.
            eval_freq: Evaluate every N episodes within each phase.
            log_freq: Print progress every N episodes.
            model_dir: Base directory for model checkpoints.

        Returns:
            dict with per-phase stats and overall metrics.
        """
        if model_dir:
            Path(model_dir).mkdir(parents=True, exist_ok=True)

        phase_stats = []
        global_ep = 0
        t0 = time.time()

        while not controller.is_complete():
            scenario_path = controller.get_next_scenario()
            self.wrapper.load_scenario(scenario_path)

            ep_return, ep_steps, sr = self._run_episode(explore=False)
            global_ep += 1

            # Record in controller
            success = sr >= 1.0  # all targets compromised
            controller.record_episode(
                success=success,
                reward=ep_return,
                steps=ep_steps,
            )

            # TensorBoard
            tier = controller.current_tier
            if self._tb:
                self._tb.add_scalar("Curriculum/Reward", ep_return, global_ep)
                self._tb.add_scalar("Curriculum/SuccessRate", sr, global_ep)
                self._tb.add_scalar("Curriculum/Tier", tier, global_ep)

            # Logging
            if global_ep % log_freq == 0:
                status = controller.get_status()
                ctrl_sr = controller.get_success_rate()
                print(
                    f"  [global_ep {global_ep}] tier={tier}, "
                    f"ctrl_sr={ctrl_sr*100:.1f}%, reward={ep_return:.1f}"
                )

            # Phase-change checkpoint
            if model_dir and hasattr(controller, "phase_history"):
                n_transitions = len(controller.phase_history)
                if n_transitions > len(phase_stats):
                    self.save(
                        str(Path(model_dir) / f"tier{tier}_ep{global_ep}")
                    )
                    phase_stats.append(
                        controller.phase_history[-1]
                        if controller.phase_history
                        else {}
                    )

        elapsed = time.time() - t0

        if model_dir:
            self.save(model_dir)

        return {
            "total_episodes": global_ep,
            "total_steps": self.total_training_steps,
            "train_time_s": round(elapsed, 2),
            "phase_stats": phase_stats,
            "final_status": controller.get_status(),
        }

    # -----------------------------------------------------------------
    # Episode execution
    # -----------------------------------------------------------------

    def _run_episode(self, explore: bool = False) -> Tuple[float, int, float]:
        """Run a single episode on the current wrapper scenario.

        Returns:
            (episode_return, episode_steps, success_rate) where
            success_rate ∈ {0.0, 1.0} (binary: all sensitive hosts done?).
        """
        self.num_episodes += 1
        if self.use_reward_scaling:
            self.reward_scaling.reset()

        o = self.wrapper.reset()
        if self.use_state_norm:
            o = self.state_norm(o, update=True)

        ep_return = 0.0
        ep_steps = 0
        done = False

        while not done and ep_steps < self.config.step_limit:
            # --- Select action ---
            action_info = self.Policy.select_action(
                observation=o,
                explore=explore,
                is_loaded_agent=False,
                num_episode=self.num_episodes,
            )
            service_action_idx = action_info[0]

            # --- Step wrapper ---
            next_o, r, done, info = self.wrapper.step(service_action_idx)
            self.total_training_steps += 1
            ep_steps += 1
            ep_return += r

            dw = done  # done weight for PPO

            # Tracking
            if done:
                if self.first_hit_step < 0:
                    self.first_hit_step = self.total_training_steps
                if self.first_hit_eps < 0:
                    self.first_hit_eps = self.num_episodes

            self._convergence_flags.append(done)
            if self.convergence_eps < 0 and all(self._convergence_flags):
                self.convergence_eps = self.num_episodes

            # Normalisation
            if self.use_state_norm:
                next_o = self.state_norm(next_o, update=True)
            if self.use_reward_scaling:
                r = self.reward_scaling(r)[0]

            # Store + update
            self.Policy.store_transtion(
                observation=o,
                action=action_info,
                reward=r,
                next_observation=next_o,
                done=dw,
            )

            if not explore:
                self.Policy.update_policy(
                    num_episode=self.num_episodes,
                    train_steps=self.total_training_steps,
                )
                if self.use_lr_decay:
                    rate = max(
                        1 - self.num_episodes / self.config.train_eps,
                        self.config.min_decay_lr,
                    )
                    self.Policy.lr_decay(rate=rate)

            o = next_o

        # Track best
        if ep_return >= self.best_return:
            self.best_return = ep_return
            self.best_episode = self.num_episodes

        sr = 1.0 if done else 0.0
        return ep_return, ep_steps, sr

    # -----------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------

    def evaluate(
        self,
        num_episodes: int = 1,
        step_limit: Optional[int] = None,
        verbose: bool = True,
    ) -> Tuple[float, float]:
        """Evaluate current policy on the loaded scenario.

        Args:
            num_episodes: How many evaluation episodes.
            step_limit: Max steps per episode (default ``config.eval_step_limit``).
            verbose: Print per-episode results.

        Returns:
            (total_reward, success_rate) across *num_episodes*.
        """
        if step_limit is None:
            step_limit = self.config.eval_step_limit

        total_reward = 0.0
        successes = 0

        for ep_i in range(num_episodes):
            o = self.wrapper.reset()
            if self.use_state_norm:
                o = self.state_norm(o, update=False)

            done = False
            steps = 0
            ep_return = 0.0

            while not done and steps < step_limit:
                with torch.no_grad():
                    action_idx = self.Policy.evaluate(o)

                next_o, r, done, info = self.wrapper.step(action_idx)
                steps += 1
                ep_return += r

                if self.use_state_norm:
                    next_o = self.state_norm(next_o, update=False)
                o = next_o

            if done:
                successes += 1
            total_reward += ep_return

            if verbose:
                status = "SUCCESS" if done else "FAILED"
                target = info.get("target_host", "?")
                print(
                    f"  Eval ep {ep_i+1}: {status}, reward={ep_return:.1f}, "
                    f"steps={steps}, last_target={target}"
                )

        sr = successes / num_episodes if num_episodes > 0 else 0.0
        return total_reward, sr

    # -----------------------------------------------------------------
    # Save / Load
    # -----------------------------------------------------------------

    def save(self, model_dir: str) -> None:
        """Save model, optimizer, and normalisation state."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            self.Policy.actor.state_dict(), model_dir / "PPO-actor.pt"
        )
        torch.save(
            self.Policy.critic.state_dict(), model_dir / "PPO-critic.pt"
        )

        if self.use_state_norm:
            torch.save(
                self.state_norm.running_ms.mean,
                model_dir / "PPO-norm_mean.pt",
            )
            torch.save(
                self.state_norm.running_ms.std,
                model_dir / "PPO-norm_std.pt",
            )

        # Metadata
        meta = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "num_episodes": self.num_episodes,
            "total_steps": self.total_training_steps,
            "best_return": float(self.best_return),
            "best_episode": self.best_episode,
            "first_hit_eps": self.first_hit_eps,
            "convergence_eps": self.convergence_eps,
            "scenario": self.wrapper.scenario_path,
        }
        with open(model_dir / "trainer_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[PenGymTrainer] Model saved → {model_dir}")

    def load(self, model_dir: str) -> None:
        """Load pre-trained model."""
        model_dir = Path(model_dir)

        actor_path = model_dir / "PPO-actor.pt"
        critic_path = model_dir / "PPO-critic.pt"

        if actor_path.exists():
            self.Policy.actor.load_state_dict(
                torch.load(
                    actor_path,
                    map_location=self.Policy.device,
                    weights_only=True,
                )
            )
        if critic_path.exists():
            self.Policy.critic.load_state_dict(
                torch.load(
                    critic_path,
                    map_location=self.Policy.device,
                    weights_only=True,
                )
            )

        if self.use_state_norm:
            mean_p = model_dir / "PPO-norm_mean.pt"
            std_p = model_dir / "PPO-norm_std.pt"
            if mean_p.exists() and std_p.exists():
                self.state_norm.running_ms.mean = torch.load(
                    mean_p, map_location="cpu", weights_only=False
                )
                self.state_norm.running_ms.std = torch.load(
                    std_p, map_location="cpu", weights_only=False
                )

        print(f"[PenGymTrainer] Model loaded ← {model_dir}")

    # -----------------------------------------------------------------
    # Misc
    # -----------------------------------------------------------------

    def close(self) -> None:
        """Clean up resources."""
        if self._tb:
            self._tb.close()

    def __repr__(self) -> str:
        return (
            f"PenGymTrainer(scenario={Path(self.wrapper.scenario_path).name}, "
            f"episodes={self.num_episodes}, steps={self.total_training_steps})"
        )
