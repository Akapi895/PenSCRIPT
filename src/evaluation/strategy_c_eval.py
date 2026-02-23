"""
Strategy C Evaluator — 4-agent comparison matrix and transfer metrics.

Strategy C §6.1 specifies comparison of four agents:

1. θ_sim_baseline  — SCRIPT trained on sim with original 1538-dim encoding
2. θ_sim_unified   — SCRIPT retrained on sim with 1540-dim unified encoding
3. θ_dual          — Dual-trained (sim → transfer → PenGym fine-tune)
4. θ_pengym_scratch — SCRIPT trained from scratch on PenGym

Each agent is evaluated on both simulation and PenGym scenarios.

Metrics computed (§6.2):
- Per-task success rate and reward
- Forward Transfer (FT) = SR(θ_dual) - SR(θ_scratch) on PenGym
- Backward Transfer (BT) = SR(θ_dual on sim after PenGym) - SR(θ_uni on sim)
- Transfer Ratio = SR(θ_dual) / SR(θ_scratch)
- EWC Compliance = ||θ_dual - θ_uni||² weighted by Fisher

Usage::

    from src.evaluation.strategy_c_eval import StrategyCEvaluator

    evaluator = StrategyCEvaluator(pengym_tasks=tasks)
    evaluator.register_agent("theta_dual", agent_cl_dual)
    evaluator.register_agent("theta_scratch", agent_cl_scratch)
    report = evaluator.evaluate_all()
    evaluator.print_report(report)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger as logging


class StrategyCEvaluator:
    """Evaluate and compare multiple SCRIPT agents on given tasks.

    Parameters
    ----------
    pengym_tasks : list
        List of PenGymHostAdapter (or HOST-like) targets for evaluation.
    sim_tasks : list, optional
        List of HOST targets for simulation-side evaluation.
    step_limit : int
        Max steps per evaluation episode.
    """

    def __init__(
        self,
        pengym_tasks: list,
        sim_tasks: Optional[list] = None,
        step_limit: int = 100,
    ):
        self.pengym_tasks = pengym_tasks
        self.sim_tasks = sim_tasks or []
        self.step_limit = step_limit
        self._agents: Dict[str, Any] = {}

    def register_agent(self, name: str, agent_cl) -> None:
        """Register an ``Agent_CL`` instance under a descriptive name.

        Args:
            name: Label for the agent (e.g. ``"theta_dual"``).
            agent_cl: ``Agent_CL`` instance.
        """
        self._agents[name] = agent_cl
        logging.info(f"[StrategyCEval] Registered agent: {name}")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_agent(
        self,
        name: str,
        on_tasks: list,
        domain: str = "pengym",
    ) -> Dict[str, Any]:
        """Evaluate a single registered agent on the given tasks.

        Args:
            name: Agent name (must be registered).
            on_tasks: List of HOST-like targets.
            domain: Label for the evaluation domain.

        Returns:
            dict with success_rate, total_rewards, per_task details.
        """
        agent_cl = self._agents.get(name)
        if agent_cl is None:
            return {"error": f"Agent '{name}' not registered"}

        try:
            evaluator = agent_cl.cl_agent.get_task_evaluator(on_train=False)
            attack_path, total_rewards, sr = evaluator.Evaluate(
                target_list=on_tasks,
                step_limit=self.step_limit,
                verbose=False,
            )
            per_task = []
            for i, ap in enumerate(attack_path):
                per_task.append({
                    "task_idx": i,
                    "task": on_tasks[i].ip if i < len(on_tasks) and hasattr(on_tasks[i], 'ip') else f"task_{i}",
                    "reward": ap.get("reward", 0),
                    "success": ap.get("success", False),
                    "steps": len(ap.get("path", [])),
                })
            return {
                "agent": name,
                "domain": domain,
                "success_rate": sr,
                "total_rewards": total_rewards,
                "per_task": per_task,
                "num_tasks": len(on_tasks),
            }
        except Exception as e:
            logging.error(f"[StrategyCEval] Failed to evaluate {name}: {e}")
            return {"agent": name, "domain": domain, "error": str(e)}

    def evaluate_all(self) -> Dict[str, Any]:
        """Evaluate all registered agents on all available task sets.

        Returns:
            Comprehensive results dict with per-agent, per-domain results
            and computed transfer metrics.
        """
        results: Dict[str, Any] = {"agents": {}, "metrics": {}}

        for name in self._agents:
            results["agents"][name] = {}

            # Evaluate on PenGym
            if self.pengym_tasks:
                results["agents"][name]["pengym"] = self.evaluate_agent(
                    name, self.pengym_tasks, domain="pengym"
                )

            # Evaluate on sim
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

        Metrics:
        - Forward Transfer (FT): SR(dual) - SR(scratch) on PenGym
        - Backward Transfer (BT): SR(dual) - SR(unified) on sim
        - Transfer Ratio: SR(dual) / SR(scratch) on PenGym
        """
        metrics = {}

        def _get_sr(agent_name: str, domain: str) -> Optional[float]:
            agent_results = results.get("agents", {}).get(agent_name, {})
            domain_results = agent_results.get(domain, {})
            return domain_results.get("success_rate")

        # Forward Transfer
        dual_pengym = _get_sr("theta_dual", "pengym")
        scratch_pengym = _get_sr("theta_pengym_scratch", "pengym")
        if dual_pengym is not None and scratch_pengym is not None:
            metrics["forward_transfer"] = dual_pengym - scratch_pengym
            metrics["transfer_ratio"] = dual_pengym / max(scratch_pengym, 1e-8)

        # Backward Transfer
        dual_sim = _get_sr("theta_dual", "sim")
        unified_sim = _get_sr("theta_sim_unified", "sim")
        if dual_sim is not None and unified_sim is not None:
            metrics["backward_transfer"] = dual_sim - unified_sim

        # Zero-shot transfer (sim agent directly on PenGym)
        unified_pengym = _get_sr("theta_sim_unified", "pengym")
        if unified_pengym is not None:
            metrics["zero_shot_sr"] = unified_pengym

        return metrics

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_report(self, results: Dict[str, Any]) -> str:
        """Print a formatted comparison table.

        Returns:
            The report as a string.
        """
        lines = []
        lines.append("\n" + "=" * 72)
        lines.append("Strategy C — Agent Comparison Report")
        lines.append("=" * 72)

        # Header
        header = f"{'Agent':<25} | {'Domain':<8} | {'SR':>6} | {'Reward':>8} | {'Tasks':>5}"
        lines.append(header)
        lines.append("-" * 72)

        for agent_name, domains in results.get("agents", {}).items():
            for domain, data in domains.items():
                if "error" in data:
                    lines.append(f"{agent_name:<25} | {domain:<8} | {'ERROR':>6} | {data['error'][:20]:>8}")
                    continue
                sr = data.get("success_rate", 0)
                reward = data.get("total_rewards", 0)
                n_tasks = data.get("num_tasks", 0)
                lines.append(
                    f"{agent_name:<25} | {domain:<8} | {sr:>5.1%} | {reward:>8.1f} | {n_tasks:>5}"
                )

        # Transfer metrics
        metrics = results.get("metrics", {})
        if metrics:
            lines.append("\n" + "-" * 40)
            lines.append("Transfer Metrics:")
            for k, v in metrics.items():
                if isinstance(v, float):
                    lines.append(f"  {k}: {v:.4f}")
                else:
                    lines.append(f"  {k}: {v}")

        report = "\n".join(lines)
        logging.info(report)
        return report

    def save_report(self, results: Dict[str, Any], path: str) -> None:
        """Save evaluation results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logging.info(f"[StrategyCEval] Report saved → {path}")
