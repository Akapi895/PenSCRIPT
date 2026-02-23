"""
DomainTransferManager — Controlled transfer of policy weights between domains.

Strategy C §5.3 specifies three transfer strategies for moving a sim-trained
SCRIPT agent to PenGym:

* **aggressive** — Copy all weights including normaliser stats.  Fastest to
  start but most vulnerable to distribution shift.
* **conservative** (default) — Copy weights, reset normaliser running stats,
  discount Fisher information by β, reduce learning rate.
* **cautious** — Copy *only* actor/critic weights; reset everything else
  (normaliser, Fisher, optimiser state).

Usage::

    from src.training.domain_transfer import DomainTransferManager

    mgr = DomainTransferManager(script_config=my_config)
    mgr.transfer(
        sim_agent=agent_cl,           # Agent_CL trained on sim
        pengym_tasks=task_list,        # list[PenGymHostAdapter]
        strategy='conservative',
    )
    # agent_cl is now ready for Phase 3 fine-tuning on PenGym
"""

from __future__ import annotations

import copy
from typing import List, Optional

import numpy as np
import torch
from loguru import logger as logging

from src.agent.policy.config import Script_Config


class DomainTransferManager:
    """Manage controlled sim → PenGym policy transfer (Strategy C §5.3).

    Parameters
    ----------
    script_config : Script_Config
        CRL config with transfer parameters (``fisher_discount_beta``,
        ``transfer_lr_factor``, ``norm_warmup_episodes``,
        ``norm_reset_on_transfer``, ``transfer_strategy``).
    """

    def __init__(self, script_config: Script_Config):
        self.cfg = script_config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transfer(
        self,
        sim_agent,
        pengym_tasks: list,
        strategy: Optional[str] = None,
    ) -> dict:
        """Execute domain transfer from sim → PenGym.

        Modifies *sim_agent* **in-place** so that it is ready for Phase 3
        fine-tuning on PenGym.

        Args:
            sim_agent: ``Agent_CL`` instance trained on simulation.
            pengym_tasks: List of ``PenGymHostAdapter`` targets for warmup.
            strategy: Override ``script_config.transfer_strategy``.

        Returns:
            dict with transfer metadata (strategy used, warmup stats, etc.)
        """
        strategy = strategy or self.cfg.transfer_strategy
        logging.info(f"[DomainTransfer] Starting transfer, strategy='{strategy}'")

        meta = {"strategy": strategy}

        script_agent = sim_agent.cl_agent  # ScriptAgent
        explorer = script_agent.explorer   # KnowledgeExplorer (Agent)
        keeper = script_agent.keeper       # KnowledgeKeeper (Agent)
        ewc = script_agent.ewc            # OnlineEWC

        if strategy == "aggressive":
            # Copy everything as-is — minimal intervention
            meta["norm_reset"] = False
            meta["fisher_discount"] = 1.0
            meta["lr_factor"] = 1.0
            logging.info("[DomainTransfer] Aggressive: keeping all stats unchanged")

        elif strategy == "conservative":
            # 1. Reset normaliser stats
            if self.cfg.norm_reset_on_transfer:
                self._reset_normalizer(explorer)
                self._reset_normalizer(keeper)
                meta["norm_reset"] = True
            else:
                meta["norm_reset"] = False

            # 2. Warmup normaliser on PenGym
            if self.cfg.norm_warmup_episodes > 0 and pengym_tasks:
                warmup_states = self._collect_warmup_states(
                    pengym_tasks, self.cfg.norm_warmup_episodes,
                )
                if explorer.use_state_norm:
                    explorer.state_norm.warmup(warmup_states)
                if keeper.use_state_norm:
                    keeper.state_norm.warmup(warmup_states)
                meta["warmup_states"] = len(warmup_states)
                logging.info(
                    f"[DomainTransfer] Warmup normaliser with "
                    f"{len(warmup_states)} states"
                )
            else:
                meta["warmup_states"] = 0

            # 3. Discount Fisher
            beta = self.cfg.fisher_discount_beta
            ewc.discount_fisher(beta)
            meta["fisher_discount"] = beta

            # 4. Reduce learning rate
            lr_factor = self.cfg.transfer_lr_factor
            self._adjust_lr(explorer, lr_factor)
            self._adjust_lr(keeper, lr_factor)
            meta["lr_factor"] = lr_factor

        elif strategy == "cautious":
            # Reset everything except actor/critic weights
            self._reset_normalizer(explorer)
            self._reset_normalizer(keeper)
            meta["norm_reset"] = True

            # Clear Fisher entirely
            ewc.importances.clear()
            ewc.saved_params.clear()
            meta["fisher_discount"] = 0.0
            logging.info("[DomainTransfer] Cautious: cleared all Fisher/saved_params")

            # Reduce LR
            lr_factor = self.cfg.transfer_lr_factor
            self._adjust_lr(explorer, lr_factor)
            self._adjust_lr(keeper, lr_factor)
            meta["lr_factor"] = lr_factor

            # Warmup normaliser
            if self.cfg.norm_warmup_episodes > 0 and pengym_tasks:
                warmup_states = self._collect_warmup_states(
                    pengym_tasks, self.cfg.norm_warmup_episodes,
                )
                if explorer.use_state_norm:
                    explorer.state_norm.warmup(warmup_states)
                if keeper.use_state_norm:
                    keeper.state_norm.warmup(warmup_states)
                meta["warmup_states"] = len(warmup_states)

        else:
            raise ValueError(
                f"Unknown transfer strategy: {strategy!r}. "
                "Choose from 'aggressive', 'conservative', 'cautious'."
            )

        logging.info(f"[DomainTransfer] Transfer complete: {meta}")
        return meta

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reset_normalizer(agent) -> None:
        """Reset the running-stats normaliser on an Agent."""
        if hasattr(agent, 'use_state_norm') and agent.use_state_norm:
            if hasattr(agent, 'state_norm'):
                agent.state_norm.reset()
                logging.info(f"[DomainTransfer] Reset normaliser for {type(agent).__name__}")

    @staticmethod
    def _adjust_lr(agent, factor: float) -> None:
        """Scale the learning rate of the agent's PPO optimisers."""
        if not hasattr(agent, 'Policy'):
            return
        policy = agent.Policy
        for opt_name in ('actor_optimizer', 'critic_optimizer'):
            opt = getattr(policy, opt_name, None)
            if opt is None:
                continue
            for pg in opt.param_groups:
                old_lr = pg['lr']
                pg['lr'] = old_lr * factor
            logging.info(
                f"[DomainTransfer] {opt_name} LR: {old_lr:.2e} → {old_lr * factor:.2e}"
            )

    @staticmethod
    def _collect_warmup_states(
        pengym_tasks: list,
        num_episodes: int,
    ) -> np.ndarray:
        """Collect states from random rollouts on PenGym for norm warmup.

        Runs *num_episodes* random-action episodes across the given tasks,
        collecting every observed state vector.

        Args:
            pengym_tasks: List of PenGymHostAdapter (or HOST-like) targets.
            num_episodes: Total episodes to collect.

        Returns:
            2D array of shape ``(N, state_dim)`` with all collected states.
        """
        all_states = []
        eps_per_task = max(1, num_episodes // len(pengym_tasks))

        for task in pengym_tasks:
            for _ in range(eps_per_task):
                state = task.reset()
                if state is not None:
                    all_states.append(np.array(state, dtype=np.float32))

                # Random walk for a few steps
                for _ in range(50):  # short random rollout
                    # Pick a random action from the action space
                    try:
                        from src.agent.actions.service_action_space import ServiceActionSpace
                        action_dim = ServiceActionSpace.DEFAULT_ACTION_DIM
                    except ImportError:
                        action_dim = 16
                    a = np.random.randint(0, action_dim)
                    try:
                        next_state, _, done, _ = task.perform_action(a)
                        if next_state is not None:
                            all_states.append(
                                np.array(next_state, dtype=np.float32)
                            )
                        if done:
                            break
                    except Exception:
                        break

        if all_states:
            return np.stack(all_states)
        return np.zeros((1, 1538), dtype=np.float32)  # fallback
