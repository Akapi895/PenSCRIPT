"""
Strategy C Evaluator — 4-agent comparison matrix and transfer metrics.

Strategy C §6.1 specifies comparison of four agents:

1. θ_sim_baseline  — SCRIPT trained on sim with original 1538-dim encoding
2. θ_sim_unified   — SCRIPT retrained on sim with 1540-dim unified encoding
3. θ_dual          — Dual-trained (sim → transfer → PenGym fine-tune)
4. θ_pengym_scratch — SCRIPT trained from scratch on PenGym

Each agent is evaluated on both simulation and PenGym scenarios with
multi-episode evaluation (K episodes per task).

Metrics computed:
- Per-task success rate (continuous, K episodes), normalized reward, step efficiency
- Forward Transfer: FT_SR, FT_NR, FT_eta (dual vs scratch on PenGym)
- Backward Transfer: BT_SR, BT_NR, BT_eta (dual vs unified on sim)
- Policy-level BT: D_KL, Fisher-weighted distance (injected externally)

Usage::

    from src.evaluation.strategy_c_eval import StrategyCEvaluator

    evaluator = StrategyCEvaluator(
        pengym_tasks={"theta_dual": tasks_d, "theta_scratch": tasks_s},
        sim_tasks=sim_tasks,
        eval_episodes=20,
        optimal_rewards={"tiny": 195},
        optimal_steps={"tiny": 6},
    )
    evaluator.register_agent("theta_dual", agent_cl_dual)
    evaluator.register_agent("theta_pengym_scratch", agent_cl_scratch)
    report = evaluator.evaluate_all()
    evaluator.print_report(report)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import re

import numpy as np
import torch
from loguru import logger as logging


def _resolve_base_scenario(task_name: str) -> str:
    """Extract base scenario name from an overlay task name.

    Examples:
        "tiny"               → "tiny"
        "tiny_T1_001"        → "tiny"
        "medium-multi-site_T3_005" → "medium-multi-site"
        "small-linear_T4_002"     → "small-linear"
    """
    m = re.match(r'^(.+?)_T\d+_\d+$', task_name)
    return m.group(1) if m else task_name


class StrategyCEvaluator:
    """Evaluate and compare multiple SCRIPT agents on given tasks.

    Parameters
    ----------
    pengym_tasks : dict[str, list] or list
        If dict: maps agent name → fresh adapter list (preferred, isolated).
        If list: shared adapter list for all agents (legacy).
    sim_tasks : list, optional
        List of HOST targets for simulation-side evaluation.
    step_limit : int
        Max steps per evaluation episode.
    eval_episodes : int
        Number of episodes per task for multi-episode SR estimation.
    optimal_rewards : dict, optional
        Map scenario name → optimal reward for NR computation.
    optimal_steps : dict, optional
        Map scenario name → optimal step count for step efficiency.
    """

    def __init__(
        self,
        pengym_tasks: Union[Dict[str, list], list],
        sim_tasks: Optional[list] = None,
        step_limit: int = 100,
        eval_episodes: int = 20,
        optimal_rewards: Optional[Dict[str, float]] = None,
        optimal_steps: Optional[Dict[str, int]] = None,
    ):
        self._pengym_tasks_dict: Optional[Dict[str, list]] = None
        self._pengym_tasks_shared: Optional[list] = None
        if isinstance(pengym_tasks, dict):
            self._pengym_tasks_dict = pengym_tasks
        else:
            self._pengym_tasks_shared = pengym_tasks

        self.sim_tasks = sim_tasks or []
        self.step_limit = step_limit
        self.eval_episodes = eval_episodes
        self.optimal_rewards = optimal_rewards or {}
        self.optimal_steps = optimal_steps or {}
        self._agents: Dict[str, Any] = {}

    def _get_pengym_tasks(self, agent_name: str) -> list:
        """Return PenGym tasks for a specific agent."""
        if self._pengym_tasks_dict is not None:
            return self._pengym_tasks_dict.get(agent_name, [])
        return self._pengym_tasks_shared or []

    def register_agent(self, name: str, agent_cl) -> None:
        """Register an ``Agent_CL`` instance under a descriptive name."""
        self._agents[name] = agent_cl
        logging.info(f"[StrategyCEval] Registered agent: {name}")

    # ------------------------------------------------------------------
    # Multi-episode Evaluation
    # ------------------------------------------------------------------

    def evaluate_agent(
        self,
        name: str,
        on_tasks: list,
        domain: str = "pengym",
    ) -> Dict[str, Any]:
        """Evaluate a single agent on K episodes per task.

        Returns dict with success_rate, normalized_reward, step_efficiency,
        per_task details, and eval_episodes count.
        """
        agent_cl = self._agents.get(name)
        if agent_cl is None:
            return {"error": f"Agent '{name}' not registered"}

        try:
            evaluator = agent_cl.cl_agent.get_task_evaluator(on_train=False)
        except Exception as e:
            logging.error(f"[StrategyCEval] Cannot get evaluator for {name}: {e}")
            return {"agent": name, "domain": domain, "error": str(e)}

        all_episodes = []
        K = self.eval_episodes

        for episode in range(K):
            for task in on_tasks:
                try:
                    o = task.reset()
                except Exception as e:
                    logging.warning(f"[StrategyCEval] reset failed for task: {e}")
                    continue

                if evaluator.use_state_norm:
                    o = evaluator.state_norm(o, update=False)

                done = 0
                steps = 0
                ep_reward = 0.0

                while not done and steps < self.step_limit:
                    with torch.no_grad():
                        a = evaluator.Policy.evaluate(o)
                    next_o, r, done, _ = task.perform_action(a)
                    if evaluator.use_state_norm:
                        next_o = evaluator.state_norm(next_o, update=False)
                    o = next_o
                    ep_reward += r
                    steps += 1

                task_name = getattr(task, 'ip', f'task_{id(task)}')
                all_episodes.append({
                    "task": task_name,
                    "success": bool(done),
                    "reward": float(ep_reward),
                    "steps": steps,
                })

        # Aggregate per-task
        per_task_data: Dict[str, Dict] = {}
        for ep in all_episodes:
            t = ep["task"]
            if t not in per_task_data:
                per_task_data[t] = {"successes": 0, "rewards": [], "steps_on_success": []}
            per_task_data[t]["successes"] += int(ep["success"])
            per_task_data[t]["rewards"].append(ep["reward"])
            if ep["success"]:
                per_task_data[t]["steps_on_success"].append(ep["steps"])

        task_results = []
        all_sr = []
        all_nr = []
        all_eta = []

        for t, data in per_task_data.items():
            sr = data["successes"] / K
            mean_reward = float(np.mean(data["rewards"]))
            std_reward = float(np.std(data["rewards"]))

            # Resolve base scenario for optimal value lookup
            base_t = _resolve_base_scenario(t)

            # Normalized Reward
            opt_r = self.optimal_rewards.get(t) or self.optimal_rewards.get(base_t)
            nr = mean_reward / opt_r if opt_r else None

            # Step Efficiency
            opt_s = self.optimal_steps.get(t) or self.optimal_steps.get(base_t)
            succ_steps = data["steps_on_success"]
            if opt_s and succ_steps:
                eta = float(np.mean([opt_s / s for s in succ_steps]))
            else:
                eta = None

            task_results.append({
                "task": t,
                "sr": sr,
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "normalized_reward": nr,
                "step_efficiency": eta,
                "mean_success_steps": float(np.mean(succ_steps)) if succ_steps else None,
                "eval_episodes": K,
            })
            all_sr.append(sr)
            if nr is not None:
                all_nr.append(nr)
            if eta is not None:
                all_eta.append(eta)

        overall_sr = float(np.mean(all_sr)) if all_sr else 0.0
        overall_nr = float(np.mean(all_nr)) if all_nr else None
        overall_eta = float(np.mean(all_eta)) if all_eta else None

        # Standard error for SR significance testing
        se = float(np.sqrt(overall_sr * (1 - overall_sr) / max(len(all_episodes), 1)))

        return {
            "agent": name,
            "domain": domain,
            "success_rate": overall_sr,
            "normalized_reward": overall_nr,
            "step_efficiency": overall_eta,
            "standard_error": se,
            "total_rewards": float(sum(ep["reward"] for ep in all_episodes)),
            "per_task": task_results,
            "num_tasks": len(on_tasks),
            "eval_episodes": K,
        }

    def evaluate_all(self) -> Dict[str, Any]:
        """Evaluate all registered agents on all available task sets.

        Returns comprehensive results dict with per-agent, per-domain
        results and computed transfer metrics.
        """
        results: Dict[str, Any] = {"agents": {}, "metrics": {}}

        for name in self._agents:
            results["agents"][name] = {}

            # Evaluate on PenGym (per-agent isolated tasks)
            pengym_tasks = self._get_pengym_tasks(name)
            if pengym_tasks:
                results["agents"][name]["pengym"] = self.evaluate_agent(
                    name, pengym_tasks, domain="pengym"
                )

            # Evaluate on sim (shared tasks — sim HOST has no class-level state issue)
            if self.sim_tasks:
                results["agents"][name]["sim"] = self.evaluate_agent(
                    name, self.sim_tasks, domain="sim"
                )

        # Compute transfer metrics
        results["metrics"] = self._compute_transfer_metrics(results)

        return results

    # ------------------------------------------------------------------
    # Transfer Metrics
    # ------------------------------------------------------------------

    def _compute_transfer_metrics(self, results: Dict) -> Dict[str, Any]:
        """Compute Strategy C transfer metrics from evaluation results.

        Forward Transfer (PenGym): FT_SR, FT_NR, FT_eta
        Backward Transfer (sim):   BT_SR, BT_NR, BT_eta
        Policy-level BT:           BT_KL, BT_fisher_dist (injected externally)
        """
        metrics = {}

        def _get(agent_name: str, domain: str, key: str):
            return results.get("agents", {}).get(agent_name, {}).get(domain, {}).get(key)

        # --- Forward Transfer (dual vs scratch on PenGym) ---
        dual_sr = _get("theta_dual", "pengym", "success_rate")
        scratch_sr = _get("theta_pengym_scratch", "pengym", "success_rate")
        if dual_sr is not None and scratch_sr is not None:
            metrics["FT_SR"] = dual_sr - scratch_sr
            metrics["transfer_ratio"] = dual_sr / max(scratch_sr, 1e-8)

        dual_nr = _get("theta_dual", "pengym", "normalized_reward")
        scratch_nr = _get("theta_pengym_scratch", "pengym", "normalized_reward")
        if dual_nr is not None and scratch_nr is not None:
            metrics["FT_NR"] = dual_nr - scratch_nr

        dual_eta = _get("theta_dual", "pengym", "step_efficiency")
        scratch_eta = _get("theta_pengym_scratch", "pengym", "step_efficiency")
        if dual_eta is not None and scratch_eta is not None:
            metrics["FT_eta"] = dual_eta - scratch_eta

        # Statistical significance for FT_SR
        dual_se = _get("theta_dual", "pengym", "standard_error") or 0
        scratch_se = _get("theta_pengym_scratch", "pengym", "standard_error") or 0
        pooled_se = float(np.sqrt(dual_se**2 + scratch_se**2))
        if pooled_se > 0 and "FT_SR" in metrics:
            metrics["FT_SR_significant"] = abs(metrics["FT_SR"]) > 2 * pooled_se

        # --- Backward Transfer (dual vs unified on sim) ---
        for key, label in [("success_rate", "BT_SR"),
                           ("normalized_reward", "BT_NR"),
                           ("step_efficiency", "BT_eta")]:
            dual_val = _get("theta_dual", "sim", key)
            uni_val = _get("theta_sim_unified", "sim", key)
            if dual_val is not None and uni_val is not None:
                metrics[label] = dual_val - uni_val

        # --- Policy-level BT (injected by DualTrainer) ---
        if "policy_metrics" in results:
            pm = results["policy_metrics"]
            if "kl_divergence" in pm:
                metrics["BT_KL"] = pm["kl_divergence"]
            if "fisher_distance" in pm:
                metrics["BT_fisher_dist"] = pm["fisher_distance"]

        # Zero-shot transfer (sim agent directly on PenGym)
        unified_pengym_sr = _get("theta_sim_unified", "pengym", "success_rate")
        if unified_pengym_sr is not None:
            metrics["zero_shot_sr"] = unified_pengym_sr

        return metrics

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_report(self, results: Dict[str, Any]) -> str:
        """Print a formatted comparison table."""
        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("Strategy C — Agent Comparison Report")
        lines.append("=" * 80)

        header = f"{'Agent':<25} | {'Domain':<8} | {'SR':>6} | {'NR':>6} | {'η':>6} | {'Reward':>8} | {'Tasks':>5}"
        lines.append(header)
        lines.append("-" * 80)

        for agent_name, domains in results.get("agents", {}).items():
            for domain, data in domains.items():
                if "error" in data:
                    lines.append(f"{agent_name:<25} | {domain:<8} | {'ERR':>6}")
                    continue
                sr = data.get("success_rate", 0)
                nr = data.get("normalized_reward")
                eta = data.get("step_efficiency")
                reward = data.get("total_rewards", 0)
                n_tasks = data.get("num_tasks", 0)
                nr_str = f"{nr:>5.3f}" if nr is not None else "  N/A"
                eta_str = f"{eta:>5.3f}" if eta is not None else "  N/A"
                lines.append(
                    f"{agent_name:<25} | {domain:<8} | {sr:>5.1%} | {nr_str} | {eta_str} | {reward:>8.1f} | {n_tasks:>5}"
                )

        metrics = results.get("metrics", {})
        if metrics:
            lines.append("\n" + "-" * 50)
            lines.append("Transfer Metrics:")
            for k, v in metrics.items():
                if isinstance(v, bool):
                    lines.append(f"  {k}: {v}")
                elif isinstance(v, float):
                    lines.append(f"  {k}: {v:+.4f}")
                else:
                    lines.append(f"  {k}: {v}")

        report = "\n".join(lines)
        logging.info(report)
        return report

    def save_report(self, results: Dict[str, Any], path: str) -> None:
        """Save evaluation results to JSON file."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logging.info(f"[StrategyCEval] Report saved → {out}")

    # ------------------------------------------------------------------
    # Forgetting Matrix & Zero-Shot Transfer Vector (Improvement B)
    # ------------------------------------------------------------------

    def compute_forgetting_matrix(
        self,
        tier_checkpoint_results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute Forgetting matrix F and Zero-Shot Transfer vector Z.

        Parameters
        ----------
        tier_checkpoint_results : dict
            Maps tier_name → ``evaluate_agent()`` result dict.
            Each result must contain a ``per_task`` list with
            ``"task"`` and ``"normalized_reward"`` keys.

        Returns
        -------
        dict with keys ``F_matrix``, ``Z_vector``, ``nr_after``,
        ``task_names``, ``tier_names``, ``summary``.
        """
        tier_names = sorted(tier_checkpoint_results.keys())

        # Collect all unique task names in order of first appearance
        all_tasks: List[str] = []
        for tier in tier_names:
            for t in tier_checkpoint_results[tier].get("per_task", []):
                if t["task"] not in all_tasks:
                    all_tasks.append(t["task"])

        n = len(all_tasks)
        m = len(tier_names)

        # NR matrix: nr_after[task_idx][tier_idx]
        nr_after = np.full((n, m), np.nan)
        for j, tier in enumerate(tier_names):
            task_nr = {
                t["task"]: t.get("normalized_reward")
                for t in tier_checkpoint_results[tier].get("per_task", [])
            }
            for i, task_name in enumerate(all_tasks):
                val = task_nr.get(task_name)
                if val is not None:
                    nr_after[i][j] = val

        # F[i][j] = NR_i_after_own_tier - NR_i_after_tier_j  (j > own tier)
        F_matrix = np.full((n, m), np.nan)
        for i in range(n):
            own_tier_idx = None
            for j in range(m):
                if not np.isnan(nr_after[i][j]) and own_tier_idx is None:
                    own_tier_idx = j
            if own_tier_idx is None:
                continue
            for j in range(own_tier_idx + 1, m):
                if not np.isnan(nr_after[i][own_tier_idx]) and not np.isnan(nr_after[i][j]):
                    F_matrix[i][j] = nr_after[i][own_tier_idx] - nr_after[i][j]

        # Z[i] = NR_task_i_before_own_tier - baseline(≈0)
        Z_vector = np.full(n, np.nan)
        for i in range(n):
            own_tier_idx = None
            for j in range(m):
                if not np.isnan(nr_after[i][j]):
                    own_tier_idx = j
                    break
            if own_tier_idx is not None and own_tier_idx > 0:
                nr_before = nr_after[i][own_tier_idx - 1]
                if not np.isnan(nr_before):
                    Z_vector[i] = nr_before  # baseline NR ≈ 0 (random)

        # Summary statistics
        f_vals = F_matrix[~np.isnan(F_matrix)]
        z_vals = Z_vector[~np.isnan(Z_vector)]
        summary = {
            "mean_forgetting": float(np.mean(f_vals)) if len(f_vals) > 0 else None,
            "max_forgetting": float(np.max(f_vals)) if len(f_vals) > 0 else None,
            "mean_zero_shot_transfer": float(np.mean(z_vals)) if len(z_vals) > 0 else None,
            "tasks_with_positive_transfer": int(np.sum(z_vals > 0)) if len(z_vals) > 0 else 0,
        }

        return {
            "F_matrix": F_matrix.tolist(),
            "Z_vector": Z_vector.tolist(),
            "nr_after": nr_after.tolist(),
            "task_names": all_tasks,
            "tier_names": tier_names,
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # TTT + AUC Learning-Speed Transfer (Improvement C)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_learning_speed(
        dual_training_data: Dict[str, list],
        scratch_training_data: Dict[str, list],
    ) -> Dict[str, Any]:
        """Compute Time-To-Threshold and AUC learning-speed transfer.

        Parameters
        ----------
        dual_training_data : dict
            Maps task_name → list of per-episode rewards (θ_dual).
        scratch_training_data : dict
            Maps task_name → list of per-episode rewards (θ_scratch).

        Returns
        -------
        dict with ``per_task`` list and ``aggregate`` summary.
        """
        results: Dict[str, Any] = {"per_task": [], "aggregate": {}}

        for task_name in dual_training_data:
            dual_rewards = dual_training_data[task_name]
            scratch_rewards = scratch_training_data.get(task_name, [])

            dual_auc = float(np.mean(dual_rewards)) if dual_rewards else 0.0
            scratch_auc = float(np.mean(scratch_rewards)) if scratch_rewards else 0.0

            # TTT: first episode with reward > 0 (successful penetration)
            dual_ttt = next(
                (i for i, r in enumerate(dual_rewards) if r > 0),
                len(dual_rewards),
            )
            scratch_ttt = next(
                (i for i, r in enumerate(scratch_rewards) if r > 0),
                len(scratch_rewards),
            )

            results["per_task"].append({
                "task": task_name,
                "dual_ttt": dual_ttt,
                "scratch_ttt": scratch_ttt,
                "ttt_speedup": scratch_ttt / max(dual_ttt, 1),
                "dual_auc": dual_auc,
                "scratch_auc": scratch_auc,
                "auc_ratio": dual_auc / max(abs(scratch_auc), 1e-8),
            })

        if results["per_task"]:
            results["aggregate"] = {
                "mean_ttt_speedup": float(np.mean(
                    [t["ttt_speedup"] for t in results["per_task"]]
                )),
                "mean_auc_ratio": float(np.mean(
                    [t["auc_ratio"] for t in results["per_task"]]
                )),
            }

        return results