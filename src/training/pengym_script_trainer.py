"""
PenGymScriptTrainer — Full SCRIPT CRL training over PenGym environments.

Bridges the gap between ``Agent_CL`` (SCRIPT's continual-learning orchestrator)
and PenGym by wrapping each scenario as a ``PenGymHostAdapter`` — a thin
object that exposes the HOST interface (``reset()``, ``perform_action()``,
``.ip``, ``.env_data``) expected by ``ScriptAgent``, ``KnowledgeExplorer``,
``KnowledgeKeeper``, and ``Agent_CL.train_continually()``.

Architecture::

    PenGymScriptTrainer
        └── Agent_CL(method="script")
                └── ScriptAgent
                        ├── KnowledgeExplorer (Student)
                        │       └── ExplorePolicy (PPO + KL imitation)
                        └── KnowledgeKeeper (Teacher)
                                ├── PPO actor (persistent)
                                ├── old_net (retrospection)
                                └── OnlineEWC

Five SCRIPT pillars active:
  1. Teacher Guidance — Keeper guides Explorer in early episodes
  2. KL Imitation Loss — Explorer's PPO loss includes KL(student ‖ teacher)
  3. Knowledge Distillation — Explorer samples distilled into Keeper
  4. Retrospection — Keeper minimises KL(new ‖ old) to prevent forgetting
  5. EWC — Fisher-information regularisation on Keeper weights

Usage::

    from src.training.pengym_script_trainer import PenGymScriptTrainer

    trainer = PenGymScriptTrainer(
        scenario_list=[
            "data/scenarios/tiny.yml",
            "data/scenarios/small-linear.yml",
        ],
    )
    result = trainer.train()
    print(f"Final SR: {result['SR_previous_tasks'][-1]*100:.1f}%")
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import numpy as np
import torch
from loguru import logger as logging
from torch.utils.tensorboard import SummaryWriter

from src.agent.policy.config import PPO_Config, Script_Config
from src.agent.agent_continual import Agent_CL
from src.envs.adapters.pengym_host_adapter import PenGymHostAdapter


# ─── Default training parameters per scenario ────────────────────────────
_DEFAULT_SCENARIO_CFG = {
    "tiny":                {"train_eps": 500,  "step_limit": 100},
    "tiny-hard":           {"train_eps": 800,  "step_limit": 150},
    "tiny-small":          {"train_eps": 500,  "step_limit": 100},
    "small-linear":        {"train_eps": 1000, "step_limit": 200},
    "small-honeypot":      {"train_eps": 1000, "step_limit": 200},
    "medium":              {"train_eps": 1500, "step_limit": 300},
    "medium-single-site":  {"train_eps": 1500, "step_limit": 300},
    "medium-multi-site":   {"train_eps": 2000, "step_limit": 400},
}


class PenGymScriptTrainer:
    """Full SCRIPT CRL training over PenGym scenarios.

    Parameters
    ----------
    scenario_list : list[str]
        Ordered list of scenario YAML paths.  Each scenario becomes one
        CRL *task*.  The order defines the curriculum.
    ppo_kwargs : dict, optional
        Overrides for ``PPO_Config`` (e.g. ``train_eps``, ``step_limit``).
    script_kwargs : dict, optional
        Overrides for ``Script_Config`` (e.g. ``ewc_lambda``, ``guide_kl_scale``).
    seed : int
        Random seed.
    tb_dir : str, optional
        TensorBoard log directory.
    model_dir : str, optional
        Where to save model checkpoints.
    config_file : str, optional
        YAML config file (relative to ``data/config/``).
        If provided, ``ppo_kwargs`` and ``script_kwargs`` are ignored.
    """

    STATE_DIM = 1538
    ACTION_DIM = 16

    def __init__(
        self,
        scenario_list: List[str],
        ppo_kwargs: Optional[Dict] = None,
        script_kwargs: Optional[Dict] = None,
        seed: int = 42,
        tb_dir: Optional[str] = None,
        model_dir: Optional[str] = None,
        config_file: Optional[str] = None,
    ):
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.scenario_list = [str(p) for p in scenario_list]
        self.model_dir = model_dir

        # ── TensorBoard ──────────────────────────────────────────────
        self._tb: Optional[SummaryWriter] = None
        if tb_dir:
            Path(tb_dir).mkdir(parents=True, exist_ok=True)
            self._tb = SummaryWriter(log_dir=tb_dir)

        # ── Build task list (PenGymHostAdapter per scenario) ─────────
        self.task_list: List[PenGymHostAdapter] = []
        for sc_path in self.scenario_list:
            adapter = PenGymHostAdapter.from_scenario(
                scenario_path=sc_path,
                seed=seed,
            )
            self.task_list.append(adapter)
            logging.info(f"[PenGymScriptTrainer] Task added: {adapter.ip} <- {sc_path}")

        if not self.task_list:
            raise ValueError("scenario_list is empty — need at least 1 scenario")

        # ── Resolve PPO config ───────────────────────────────────────
        # Use the first scenario's defaults as baseline, then override
        first_sc_name = Path(self.scenario_list[0]).stem
        sc_defaults = _DEFAULT_SCENARIO_CFG.get(first_sc_name, {})

        ppo_args = {
            "state_dim": self.STATE_DIM,
            "action_dim": self.ACTION_DIM,
            "train_eps": sc_defaults.get("train_eps", 500),
            "step_limit": sc_defaults.get("step_limit", 100),
            "eval_step_limit": sc_defaults.get("step_limit", 100),
            "use_state_norm": True,
            "use_reward_scaling": False,
        }
        if ppo_kwargs:
            ppo_args.update(ppo_kwargs)

        self.ppo_config = PPO_Config(**ppo_args)

        # ── Resolve Script config ────────────────────────────────────
        script_args = {}
        if script_kwargs:
            script_args.update(script_kwargs)
        self.script_config = Script_Config(**script_args)

        # ── Build Agent_CL ───────────────────────────────────────────
        time_flag = datetime.now().strftime("%Y%m%d_%H%M%S")

        if config_file:
            # Let Agent_CL parse the YAML config file itself
            self.agent_cl = Agent_CL(
                time_flag=time_flag,
                logger=self._tb,
                use_wandb=False,
                method="script",
                policy_name="PPO",
                seed=seed,
                config_file=config_file,
            )
            # Patch dims into the resolved config
            self.agent_cl.config.state_dim = self.STATE_DIM
            self.agent_cl.config.action_dim = self.ACTION_DIM
        else:
            self.agent_cl = Agent_CL(
                time_flag=time_flag,
                logger=self._tb,
                use_wandb=False,
                method="script",
                policy_name="PPO",
                seed=seed,
                config=self.ppo_config,
                cl_config=self.script_config,
            )

        logging.info(
            f"[PenGymScriptTrainer] Initialized: "
            f"{len(self.task_list)} tasks, "
            f"method={self.agent_cl.method}, "
            f"state_dim={self.STATE_DIM}, action_dim={self.ACTION_DIM}"
        )

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------

    def train(
        self,
        eval_freq: int = 5,
        save_agent: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run the full SCRIPT CRL training loop over all tasks.

        This delegates to ``Agent_CL.train_continually()`` which internally:
          1. For each task: get_new_task_learner() → Explorer (with teacher guidance)
          2. learn_new_task() → Explorer trains on PenGymHostAdapter
          3. policy_preservation() → Distill Explorer → Keeper (KD + EWC + retro)
          4. Evaluate Keeper on all seen tasks

        Args:
            eval_freq: Evaluate every N episodes during per-task training.
            save_agent: Save model after each task.
            verbose: Print detailed progress.

        Returns:
            dict with CRL training matrix (same format as Agent_CL).
        """
        t0 = time.time()

        logging.info(f"[PenGymScriptTrainer] Starting SCRIPT CRL training...")
        logging.info(f"  Tasks: {[t.ip for t in self.task_list]}")
        logging.info(f"  PPO train_eps={self.ppo_config.train_eps}, "
                      f"step_limit={self.ppo_config.step_limit}")
        logging.info(f"  SCRIPT: ewc_lambda={self.script_config.ewc_lambda}, "
                      f"guide_kl={self.script_config.guide_kl_scale}, "
                      f"transfer={self.script_config.transfer_strength}")

        result = self.agent_cl.train_continually(
            task_list=self.task_list,
            eval_freq=eval_freq,
            save_agent=save_agent,
            verbose=verbose,
        )

        elapsed = time.time() - t0

        # Log summary
        final_sr = result.get("SR_previous_tasks", [0])[-1] if result.get("SR_previous_tasks") else 0
        logging.info(
            f"[PenGymScriptTrainer] Training complete in {elapsed:.1f}s. "
            f"Final SR across all tasks: {final_sr*100:.1f}%"
        )

        # Save final model
        if self.model_dir and save_agent:
            self.save(self.model_dir)

        # Attach extra metadata
        result["train_time_s"] = round(elapsed, 2)
        result["scenario_list"] = self.scenario_list
        result["task_names"] = [t.ip for t in self.task_list]
        result["state_dim"] = self.STATE_DIM
        result["action_dim"] = self.ACTION_DIM

        return result

    # -----------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------

    def evaluate(
        self,
        step_limit: Optional[int] = None,
        verbose: bool = True,
        on_train: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate the current policy on all tasks.

        Args:
            step_limit: Max steps per task.
            verbose: Print per-task results.
            on_train: If True, evaluate with Explorer; else with Keeper.

        Returns:
            dict with attack_path, rewards, success_rate.
        """
        if step_limit is None:
            step_limit = getattr(self.ppo_config, 'eval_step_limit',
                                 self.ppo_config.step_limit)

        evaluator = self.agent_cl.cl_agent.get_task_evaluator(on_train=on_train)

        attack_path, total_rewards, sr = evaluator.Evaluate(
            target_list=self.task_list,
            step_limit=step_limit,
            verbose=verbose,
        )

        results = {
            "attack_path": attack_path,
            "total_rewards": total_rewards,
            "success_rate": sr,
            "per_task": [],
        }

        for i, ap in enumerate(attack_path):
            task_result = {
                "task": self.task_list[i].ip,
                "reward": ap.get("reward", 0),
                "success": ap.get("success", False),
                "steps": len(ap.get("path", [])),
            }
            results["per_task"].append(task_result)
            if verbose:
                status = "SUCCESS" if task_result["success"] else "FAILED"
                logging.info(
                    f"  Task {i} ({task_result['task']}): {status}, "
                    f"reward={task_result['reward']:.1f}, "
                    f"steps={task_result['steps']}"
                )

        return results

    # -----------------------------------------------------------------
    # Save / Load
    # -----------------------------------------------------------------

    def save(self, model_dir: str) -> None:
        """Save the Keeper model and metadata."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        self.agent_cl.save(path=model_dir)

        meta = {
            "state_dim": self.STATE_DIM,
            "action_dim": self.ACTION_DIM,
            "method": "script",
            "scenario_list": self.scenario_list,
            "task_names": [t.ip for t in self.task_list],
            "seed": self.seed,
            "ppo_config": {
                "train_eps": self.ppo_config.train_eps,
                "step_limit": self.ppo_config.step_limit,
                "hidden_sizes": self.ppo_config.hidden_sizes,
            },
            "script_config": {
                "ewc_lambda": self.script_config.ewc_lambda,
                "guide_kl_scale": self.script_config.guide_kl_scale,
                "transfer_strength": self.script_config.transfer_strength,
                "consolidation_iteration_num": self.script_config.consolidation_iteration_num,
                "sample_batch": self.script_config.sample_batch,
            },
        }
        with open(model_dir / "script_trainer_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logging.info(f"[PenGymScriptTrainer] Saved → {model_dir}")

    def load(self, model_dir: str) -> None:
        """Load a previously saved Keeper model."""
        model_dir = Path(model_dir)
        self.agent_cl.load(path=model_dir)
        logging.info(f"[PenGymScriptTrainer] Loaded ← {model_dir}")

    # -----------------------------------------------------------------
    # Misc
    # -----------------------------------------------------------------

    def close(self) -> None:
        """Clean up resources."""
        if self._tb:
            self._tb.close()

    def __repr__(self) -> str:
        return (
            f"PenGymScriptTrainer("
            f"tasks={[t.ip for t in self.task_list]}, "
            f"method={self.agent_cl.method})"
        )
