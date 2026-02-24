"""
DualTrainer — Phase 0→1→2→3→4 orchestrator for Strategy C dual training.

Strategy C's core contribution: train on simulation first, then transfer
the policy to PenGym via controlled domain transfer, and fine-tune with
CRL constraints.

Pipeline::

    Phase 0  →  Validation (SBERT consistency, PenGym stability)
    Phase 1  →  Train SCRIPT CRL on simulation with unified encoding → θ_uni
    Phase 2  →  DomainTransferManager: reset norm, discount Fisher, reduce LR
    Phase 3  →  Fine-tune on PenGym with EWC constraints from Phase 1 → θ_dual
    Phase 4  →  Evaluate all 4 agents (baseline, unified, dual, scratch)

Usage::

    from src.training.dual_trainer import DualTrainer

    trainer = DualTrainer(
        sim_scenarios=["data/scenarios/chain/chain_1.json", ...],
        pengym_scenarios=["data/scenarios/tiny.yml", ...],
    )
    results = trainer.run_full_pipeline()
"""

from __future__ import annotations

import copy
import json
import re as _re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from loguru import logger as logging
from torch.utils.tensorboard import SummaryWriter

from src.agent.policy.config import PPO_Config, Script_Config
from src.agent.agent_continual import Agent_CL
from src.agent.host import HOST, StateEncoder
from src.envs.core.unified_state_encoder import UnifiedStateEncoder
from src.envs.wrappers.reward_normalizer import UnifiedNormalizer
from src.training.domain_transfer import DomainTransferManager
from src.agent.actions.service_action_space import ServiceActionSpace


class DualTrainer:
    """Orchestrate the full Strategy C dual-training pipeline.

    Parameters
    ----------
    sim_scenarios : list[str]
        Paths to simulation scenario JSON files (Phase 1 tasks).
    pengym_scenarios : list[str]
        Paths to PenGym scenario YAML files (Phase 3 tasks).
    ppo_kwargs : dict, optional
        Overrides for ``PPO_Config``.
    script_kwargs : dict, optional
        Overrides for ``Script_Config``.
    seed : int
        Random seed.
    output_dir : str
        Root output directory for logs, models, and results.
    """

    def __init__(
        self,
        sim_scenarios: List[str],
        pengym_scenarios: List[str],
        heldout_scenarios: Optional[List[str]] = None,
        ppo_kwargs: Optional[Dict] = None,
        script_kwargs: Optional[Dict] = None,
        seed: int = 42,
        output_dir: str = "outputs/strategy_c",
        episode_config: Optional[Dict] = None,
    ):
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.sim_scenarios = [str(p) for p in sim_scenarios]
        self.pengym_scenarios = [str(p) for p in pengym_scenarios]
        self.heldout_scenarios = [str(p) for p in (heldout_scenarios or [])]
        self.episode_config = episode_config  # per-scenario episode rules
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Build configs
        ppo_args = {
            "train_eps": 500,
            "step_limit": 100,
            "eval_step_limit": 100,
            "use_state_norm": True,
            "action_dim": ServiceActionSpace.DEFAULT_ACTION_DIM,  # 16
            "state_dim": UnifiedStateEncoder.TOTAL_DIM,  # 1540
            **(ppo_kwargs or {}),
        }
        self.ppo_config = PPO_Config(**ppo_args)

        script_args = {**(script_kwargs or {})}
        self.script_config = Script_Config(**script_args)

        # Unified encoder
        self.unified_encoder = UnifiedStateEncoder()

        # TensorBoard
        self.tb_dir = self.output_dir / "tensorboard"
        self.tb_dir.mkdir(parents=True, exist_ok=True)

        # Storage for trained agents
        self._theta_sim_baseline = None
        self._theta_sim_unified = None
        self._theta_dual = None
        self._theta_pengym_scratch = None

        # Results
        self.results: Dict[str, Any] = {}

    # ==================================================================
    # Utilities
    # ==================================================================

    @staticmethod
    def _extract_tier(scenario_path: str) -> str:
        """Extract tier (T1/T2/T3/T4) from scenario filename."""
        m = _re.search(r'_T(\d+)_', Path(scenario_path).stem)
        return f"T{m.group(1)}" if m else "T0"

    # ==================================================================
    # Episode Schedule Resolution
    # ==================================================================

    def _resolve_episode_schedule(self, scenario_paths: List[str]) -> Optional[Dict[int, int]]:
        """Build a task_index → episode_count mapping from episode_config.

        The episode_config dict supports two modes:

        1) Multiplier mode (recommended):
           {
             "base_episodes": {"tiny": 500, "medium": 5000, ...},
             "tier_multiplier": {"T1": 1.0, "T2": 1.5, "T3": 2.0, "T4": 3.0},
             "default_episodes": 1000
           }
           Scenario filename is parsed as ``{base}_T{tier}_{variant}`` and
           episodes = ceil(base_episodes[base] * tier_multiplier["T{tier}"]).

        2) Rules mode (explicit patterns):
           {
             "rules": [
               {"pattern": "tiny_T[12]", "episodes": 500},
               {"pattern": "medium_T4", "episodes": 12000}
             ],
             "default_episodes": 1000
           }
           First regex match wins.

        Returns None if episode_config is not set → caller uses ppo_config.train_eps.
        """
        import re, math

        if not self.episode_config:
            return None

        schedule = {}
        default_eps = self.episode_config.get("default_episodes", self.ppo_config.train_eps)

        if "rules" in self.episode_config:
            # ── Rules mode ──
            rules = self.episode_config["rules"]
            for idx, sc_path in enumerate(scenario_paths):
                stem = Path(sc_path).stem  # e.g. "medium_T4_003"
                matched = False
                for rule in rules:
                    if re.search(rule["pattern"], stem, re.IGNORECASE):
                        schedule[idx] = int(rule["episodes"])
                        matched = True
                        break
                if not matched:
                    schedule[idx] = default_eps
        else:
            # ── Multiplier mode ──
            base_eps = self.episode_config.get("base_episodes", {})
            tier_mult = self.episode_config.get("tier_multiplier", {})

            for idx, sc_path in enumerate(scenario_paths):
                stem = Path(sc_path).stem  # e.g. "medium-multi-site_T3_002"

                # Parse: try to extract base name and tier
                tier_match = re.search(r'_T(\d+)_', stem)
                if tier_match:
                    tier_key = f"T{tier_match.group(1)}"
                    base_name = stem[:tier_match.start()]
                else:
                    tier_key = None
                    base_name = stem  # base scenario without tier

                b_eps = base_eps.get(base_name, default_eps)
                t_mult = tier_mult.get(tier_key, 1.0) if tier_key else 1.0
                schedule[idx] = max(500, int(math.ceil(b_eps * t_mult)))

        if schedule:
            logging.info(f"[DualTrainer] Episode schedule: "
                         f"min={min(schedule.values())}, max={max(schedule.values())}, "
                         f"total={sum(schedule.values())} across {len(schedule)} tasks")
        return schedule if schedule else None

    # ==================================================================
    # Full Pipeline
    # ==================================================================

    def run_full_pipeline(
        self,
        skip_phase0: bool = False,
        eval_freq: int = 5,
    ) -> Dict[str, Any]:
        """Run the complete Phase 0→1→2→3→4 pipeline.

        Args:
            skip_phase0: Skip validation checks (not recommended).
            eval_freq: Evaluation frequency during training phases.

        Returns:
            Comprehensive results dict.
        """
        t0 = time.time()
        logging.info("=" * 70)
        logging.info("Strategy C — Dual Training Pipeline")
        logging.info("=" * 70)

        # Phase 0: Validation
        if not skip_phase0:
            phase0_results = self.phase0_validation()
            self.results["phase0"] = phase0_results
        else:
            logging.warning("[Phase 0] Skipped — running without validation")
            self.results["phase0"] = {"skipped": True}

        # Phase 1: Simulation training
        phase1_results = self.phase1_sim_training(eval_freq=eval_freq)
        self.results["phase1"] = phase1_results

        # Phase 2: Domain transfer
        phase2_results = self.phase2_domain_transfer()
        self.results["phase2"] = phase2_results

        # Phase 3: PenGym fine-tuning
        phase3_results = self.phase3_pengym_finetuning(eval_freq=eval_freq)
        self.results["phase3"] = phase3_results

        # Phase 4: Evaluation
        phase4_results = self.phase4_evaluation()
        self.results["phase4"] = phase4_results

        total_time = time.time() - t0
        self.results["total_time_s"] = round(total_time, 2)

        # Save results
        results_path = self.output_dir / "dual_trainer_results.json"
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        logging.info(f"[DualTrainer] Results saved → {results_path}")

        return self.results

    # ==================================================================
    # Phase 0: Validation
    # ==================================================================

    def phase0_validation(self) -> Dict[str, Any]:
        """Phase 0 — Validate assumptions before training.

        Checks:
        - SBERT consistency (same text → same embedding across calls)
        - PenGym scenario loadability
        - State dimension correctness
        """
        logging.info("\n" + "=" * 60)
        logging.info("[Phase 0] Validation")
        logging.info("=" * 60)

        results = {}

        # 0.1 SBERT consistency check
        from src.agent.nlp.Encoder import encoder
        test_texts = ["linux", "ssh", "22,80,443", "http,ftp,ssh"]
        embeddings_a = [encoder.encode_SBERT(t).flatten() for t in test_texts]
        embeddings_b = [encoder.encode_SBERT(t).flatten() for t in test_texts]
        cosine_sims = []
        for a, b in zip(embeddings_a, embeddings_b):
            cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            cosine_sims.append(float(cos))
        results["sbert_consistency"] = {
            "cosine_similarities": cosine_sims,
            "all_above_0.99": all(c > 0.99 for c in cosine_sims),
        }
        logging.info(
            f"  SBERT consistency: {cosine_sims} — "
            f"{'PASS' if results['sbert_consistency']['all_above_0.99'] else 'FAIL'}"
        )

        # 0.2 Unified encoder dimension check
        results["unified_dim"] = UnifiedStateEncoder.TOTAL_DIM
        results["dim_check"] = (UnifiedStateEncoder.TOTAL_DIM == 1540)
        logging.info(f"  Unified dim: {UnifiedStateEncoder.TOTAL_DIM} — "
                      f"{'PASS' if results['dim_check'] else 'FAIL'}")

        # 0.3 Canonicalization check
        os_test = UnifiedStateEncoder.canonicalize_os("Ubuntu")
        svc_test = UnifiedStateEncoder.canonicalize_service("openssh")
        results["canonicalization"] = {
            "ubuntu_→_linux": os_test == "linux",
            "openssh_→_ssh": svc_test == "ssh",
        }
        logging.info(f"  Canonicalization: {results['canonicalization']}")

        # 0.4 Cross-domain SBERT similarity
        enc = self.unified_encoder
        vec_linux_raw = enc._encode_sbert("linux")
        vec_ubuntu_canon = enc._encode_sbert(
            UnifiedStateEncoder.canonicalize_os("ubuntu")
        )
        cos_os = float(np.dot(vec_linux_raw, vec_ubuntu_canon) / (
            np.linalg.norm(vec_linux_raw) * np.linalg.norm(vec_ubuntu_canon) + 1e-8
        ))
        results["cross_domain_os_cosine"] = cos_os
        logging.info(f"  Cross-domain OS cosine (linux vs ubuntu→linux): {cos_os:.4f}")

        # 0.5 PenGym scenario loadability
        pengym_loadable = []
        for sc in self.pengym_scenarios:
            try:
                from src.envs.adapters.pengym_host_adapter import PenGymHostAdapter
                adapter = PenGymHostAdapter.from_scenario(sc, seed=self.seed)
                pengym_loadable.append({"scenario": sc, "ok": True, "ip": adapter.ip})
            except Exception as e:
                pengym_loadable.append({"scenario": sc, "ok": False, "error": str(e)})
        results["pengym_loadable"] = pengym_loadable
        logging.info(f"  PenGym loadable: {sum(1 for p in pengym_loadable if p['ok'])}/{len(pengym_loadable)}")

        return results

    # ==================================================================
    # Phase 1: Simulation Training
    # ==================================================================

    def phase1_sim_training(self, eval_freq: int = 5) -> Dict[str, Any]:
        """Phase 1 — Train SCRIPT CRL on simulation with unified encoding → θ_uni.

        This trains a standard SCRIPT agent using HOST targets from JSON
        scenario files.
        """
        logging.info("\n" + "=" * 60)
        logging.info("[Phase 1] Simulation CRL Training → θ_uni")
        logging.info("=" * 60)

        from src.agent.actions import Action

        tb_phase1 = SummaryWriter(log_dir=str(self.tb_dir / "phase1_sim"))

        time_flag = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Build service-level action space for hierarchical action selection
        sas = ServiceActionSpace(action_class=Action)
        logging.info(f"  ServiceActionSpace: {sas.action_dim} groups, "
                     f"{sum(len(a.cve_indices) for a in sas.actions)} CVEs mapped")

        # Build sim task list from JSON scenarios
        sim_tasks = []
        sim_reward_norm = UnifiedNormalizer(source='simulation')
        for sc_path in self.sim_scenarios:
            sc_path = Path(sc_path)
            with open(sc_path, 'r', encoding='utf-8') as f:
                env_data_list = json.load(f)
            for host_data in env_data_list:
                ip = host_data["ip"]
                vul = host_data["vulnerability"][0]
                if vul not in Action.Vul_cve_set:
                    logging.warning(f"Skipping host {ip}: vuln {vul} not in Vul_cve_set")
                    continue
                t = HOST(ip, env_data=host_data, env_file=sc_path,
                         service_action_space=sas,
                         unified_encoder=self.unified_encoder,
                         reward_normalizer=sim_reward_norm)
                sim_tasks.append(t)
        logging.info(f"  Sim tasks: {len(sim_tasks)} hosts from {len(self.sim_scenarios)} scenarios")

        if not sim_tasks:
            logging.error("No sim tasks loaded — cannot proceed with Phase 1")
            return {"error": "no_sim_tasks"}

        # Save for Phase 4 BT evaluation
        self._sim_tasks = sim_tasks

        # Build Agent_CL for simulation training
        sim_agent = Agent_CL(
            time_flag=time_flag,
            logger=tb_phase1,
            use_wandb=False,
            method="script",
            policy_name="PPO",
            seed=self.seed,
            config=copy.deepcopy(self.ppo_config),
            cl_config=copy.deepcopy(self.script_config),
        )

        t0 = time.time()
        cl_matrix = sim_agent.train_continually(
            task_list=sim_tasks,
            eval_freq=eval_freq,
            save_agent=False,
            verbose=True,
        )
        phase1_time = time.time() - t0

        # Store the trained agent for Phase 2
        self._theta_sim_unified = sim_agent

        # Save Phase 1 model
        phase1_model_dir = self.output_dir / "models" / "phase1_sim"
        phase1_model_dir.mkdir(parents=True, exist_ok=True)
        sim_agent.save(path=phase1_model_dir)

        tb_phase1.close()

        results = {
            "num_tasks": len(sim_tasks),
            "train_time_s": round(phase1_time, 2),
            "final_sr": sim_agent.eval_success_rate,
            "final_reward": sim_agent.eval_rewards,
            "model_dir": str(phase1_model_dir),
        }
        logging.info(f"[Phase 1] Complete: SR={results['final_sr']:.2%}, "
                      f"time={results['train_time_s']}s")
        return results

    # ==================================================================
    # Phase 2: Domain Transfer
    # ==================================================================

    def phase2_domain_transfer(self) -> Dict[str, Any]:
        """Phase 2 — Transfer θ_uni from sim → PenGym via DomainTransferManager."""
        logging.info("\n" + "=" * 60)
        logging.info("[Phase 2] Domain Transfer (Sim → PenGym)")
        logging.info("=" * 60)

        if self._theta_sim_unified is None:
            logging.error("[Phase 2] No sim agent from Phase 1 — cannot transfer")
            return {"error": "no_phase1_agent"}

        # Build PenGym task list
        from src.envs.adapters.pengym_host_adapter import PenGymHostAdapter

        pengym_tasks = []
        for sc_path in self.pengym_scenarios:
            adapter = PenGymHostAdapter.from_scenario(
                sc_path, seed=self.seed, use_unified_encoding=True,
            )
            pengym_tasks.append(adapter)

        if not pengym_tasks:
            logging.error("[Phase 2] No PenGym tasks — cannot transfer")
            return {"error": "no_pengym_tasks"}

        # Deep copy the sim agent for transfer (preserve original for eval)
        self._theta_dual = copy.deepcopy(self._theta_sim_unified)

        # Execute transfer
        mgr = DomainTransferManager(script_config=self.script_config)
        transfer_meta = mgr.transfer(
            sim_agent=self._theta_dual,
            pengym_tasks=pengym_tasks,
            strategy=self.script_config.transfer_strategy,
        )

        # Store PenGym tasks for Phase 3
        self._pengym_tasks = pengym_tasks

        results = {
            "strategy": transfer_meta["strategy"],
            "norm_reset": transfer_meta.get("norm_reset"),
            "fisher_discount": transfer_meta.get("fisher_discount"),
            "lr_factor": transfer_meta.get("lr_factor"),
            "warmup_states": transfer_meta.get("warmup_states", 0),
            "num_pengym_tasks": len(pengym_tasks),
        }
        logging.info(f"[Phase 2] Transfer complete: {results}")
        return results

    # ==================================================================
    # Phase 3: PenGym Fine-tuning
    # ==================================================================

    def phase3_pengym_finetuning(self, eval_freq: int = 5) -> Dict[str, Any]:
        """Phase 3 — Fine-tune the transferred agent on PenGym with EWC constraints.

        Training is split by tier groups (T1→T2→T3→T4) with a model
        checkpoint saved after each tier boundary.  Per-task episode
        rewards are collected for downstream learning-speed metrics.
        """
        logging.info("\n" + "=" * 60)
        logging.info("[Phase 3] PenGym Fine-tuning → θ_dual")
        logging.info("=" * 60)

        if self._theta_dual is None:
            logging.error("[Phase 3] No transferred agent from Phase 2")
            return {"error": "no_phase2_agent"}

        if not hasattr(self, '_pengym_tasks') or not self._pengym_tasks:
            logging.error("[Phase 3] No PenGym tasks")
            return {"error": "no_pengym_tasks"}

        tb_phase3 = SummaryWriter(log_dir=str(self.tb_dir / "phase3_pengym"))
        self._theta_dual.tf_logger = tb_phase3

        t0 = time.time()

        # Resolve per-task episode schedule
        episode_schedule = self._resolve_episode_schedule(self.pengym_scenarios)

        # ── Group tasks by tier for inter-tier checkpoints ──
        tier_groups: List[tuple] = []  # (tier_name, [global_task_indices])
        current_tier = None
        for idx, sc_path in enumerate(self.pengym_scenarios):
            tier = self._extract_tier(sc_path)
            if tier != current_tier:
                tier_groups.append((tier, []))
                current_tier = tier
            tier_groups[-1][1].append(idx)

        checkpoint_dir = self.output_dir / "models" / "tier_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        tier_checkpoints: Dict[str, str] = {}
        per_task_rewards: Dict[str, list] = {}  # task_name → episode rewards

        for tier_name, task_indices in tier_groups:
            tier_tasks = [self._pengym_tasks[i] for i in task_indices]
            # Build local episode schedule (0-indexed within this tier)
            tier_schedule = None
            if episode_schedule:
                tier_schedule = {
                    local: episode_schedule[task_indices[local]]
                    for local in range(len(task_indices))
                    if task_indices[local] in episode_schedule
                }

            cl_matrix = self._theta_dual.train_continually(
                task_list=tier_tasks,
                eval_freq=eval_freq,
                save_agent=False,
                verbose=True,
                episode_schedule=tier_schedule,
            )

            # Collect per-task episode rewards from CL_Train_matrix
            for local_idx, global_idx in enumerate(task_indices):
                task_obj = self._pengym_tasks[global_idx]
                task_name = getattr(task_obj, 'ip', f'task_{global_idx}')
                per_task_rewards[task_name] = (
                    cl_matrix.Rewards_current_task[local_idx:local_idx + 1]
                    if hasattr(cl_matrix, 'Rewards_current_task')
                    else []
                )

            # Checkpoint after each tier
            ckpt_path = checkpoint_dir / f"after_{tier_name}"
            self._theta_dual.save(path=ckpt_path)
            tier_checkpoints[tier_name] = str(ckpt_path)
            logging.info(f"[Phase 3] Checkpoint saved after {tier_name} → {ckpt_path}")

        phase3_time = time.time() - t0

        # Save Phase 3 final model
        phase3_model_dir = self.output_dir / "models" / "phase3_dual"
        phase3_model_dir.mkdir(parents=True, exist_ok=True)
        self._theta_dual.save(path=phase3_model_dir)

        tb_phase3.close()

        # Store for Phase 4 downstream metrics
        self._tier_checkpoints = tier_checkpoints
        self._dual_per_task_rewards = per_task_rewards

        results = {
            "num_tasks": len(self._pengym_tasks),
            "train_time_s": round(phase3_time, 2),
            "final_sr": self._theta_dual.eval_success_rate,
            "final_reward": self._theta_dual.eval_rewards,
            "model_dir": str(phase3_model_dir),
            "tier_checkpoints": tier_checkpoints,
        }
        logging.info(f"[Phase 3] Complete: SR={results['final_sr']:.2%}, "
                      f"time={results['train_time_s']}s")
        return results

    # ==================================================================
    # Phase 4: Evaluation (4-agent matrix)
    # ==================================================================

    def phase4_evaluation(self) -> Dict[str, Any]:
        """Phase 4 — Evaluate all available agents via StrategyCEvaluator.

        Possible agents:
        - θ_sim_unified : SCRIPT trained on sim (Phase 1)
        - θ_dual        : Dual-trained (Phase 1 → Phase 2 → Phase 3)
        - θ_pengym_scratch : SCRIPT trained from scratch on PenGym (if available)

        Each agent is evaluated on PenGym (fresh adapters per agent)
        and sim tasks. Transfer metrics (FT_SR, FT_NR, FT_eta,
        BT_SR, BT_NR, BT_eta, BT_KL, BT_fisher_dist) are computed
        automatically.
        """
        logging.info("\n" + "=" * 60)
        logging.info("[Phase 4] Multi-Agent Evaluation")
        logging.info("=" * 60)

        from src.evaluation.strategy_c_eval import StrategyCEvaluator

        # Collect agent names that will be evaluated
        agent_names = []
        if self._theta_sim_unified is not None:
            agent_names.append("theta_sim_unified")
        if self._theta_dual is not None:
            agent_names.append("theta_dual")
        if self._theta_pengym_scratch is not None:
            agent_names.append("theta_pengym_scratch")

        if not agent_names:
            logging.warning("[Phase 4] No agents to evaluate")
            return {"agents": {}}

        # Create fresh PenGym adapters per agent (isolated eval environments)
        per_agent_pengym_tasks = {
            name: self._create_eval_tasks_from(self.pengym_scenarios)
            for name in agent_names
        }

        step_limit = getattr(self.ppo_config, 'eval_step_limit',
                             self.ppo_config.step_limit)

        # Optimal rewards and steps per scenario (from YAML comments)
        optimal_rewards = {
            "tiny": 195, "tiny-hard": 192,
            "tiny-small": 189, "small-linear": 179,
            "small-honeypot": 186, "medium": 185,
            "medium-single-site": 191, "medium-multi-site": 187,
        }
        optimal_steps = {
            "tiny": 6, "tiny-hard": 5,
            "tiny-small": 7, "small-linear": 12,
            "small-honeypot": 8, "medium": 8,
            "medium-single-site": 4, "medium-multi-site": 7,
        }

        evaluator = StrategyCEvaluator(
            pengym_tasks=per_agent_pengym_tasks,
            sim_tasks=getattr(self, '_sim_tasks', None),
            step_limit=step_limit,
            eval_episodes=20,
            optimal_rewards=optimal_rewards,
            optimal_steps=optimal_steps,
        )

        if self._theta_sim_unified is not None:
            evaluator.register_agent("theta_sim_unified", self._theta_sim_unified)
        if self._theta_dual is not None:
            evaluator.register_agent("theta_dual", self._theta_dual)
        if self._theta_pengym_scratch is not None:
            evaluator.register_agent("theta_pengym_scratch", self._theta_pengym_scratch)

        # Run multi-episode evaluation
        report = evaluator.evaluate_all()

        # Compute policy-level BT metrics and inject before printing
        policy_metrics = self._compute_policy_bt_metrics()
        if policy_metrics:
            report["policy_metrics"] = policy_metrics
            # Re-compute transfer metrics with policy_metrics included
            report["metrics"] = evaluator._compute_transfer_metrics(report)

        evaluator.print_report(report)

        # Save detailed evaluation report
        eval_report_path = self.output_dir / "strategy_c_eval_report.json"
        evaluator.save_report(report, str(eval_report_path))

        # Flatten for backward-compatible results format
        results = {"agents": {}, "transfer_metrics": report.get("metrics", {})}
        for name, domains in report.get("agents", {}).items():
            pengym_data = domains.get("pengym", {})
            sim_data = domains.get("sim", {})
            results["agents"][name] = {
                "success_rate": pengym_data.get("success_rate", 0),
                "normalized_reward": pengym_data.get("normalized_reward"),
                "step_efficiency": pengym_data.get("step_efficiency"),
                "total_rewards": pengym_data.get("total_rewards", 0),
                "per_task": pengym_data.get("per_task", []),
                "sim_success_rate": sim_data.get("success_rate") if sim_data else None,
                "sim_step_efficiency": sim_data.get("step_efficiency") if sim_data else None,
            }
            sr = pengym_data.get("success_rate", "N/A")
            logging.info(f"    {name}: SR={sr:.2%}" if isinstance(sr, float) else f"    {name}: {sr}")

        metrics = results["transfer_metrics"]
        if metrics:
            for key in ["FT_SR", "FT_NR", "FT_eta", "BT_SR", "BT_NR", "BT_eta"]:
                val = metrics.get(key)
                if val is not None:
                    logging.info(f"  {key}: {val:+.4f}")
            for key in ["BT_KL", "BT_fisher_dist"]:
                val = metrics.get(key)
                if val is not None:
                    logging.info(f"  {key}: {val:.6f}")

        # ── Heldout evaluation (D) ──
        if self.heldout_scenarios:
            logging.info("[Phase 4] Evaluating on heldout scenarios...")
            per_agent_heldout = {
                name: self._create_eval_tasks_from(self.heldout_scenarios)
                for name in agent_names
            }
            heldout_evaluator = StrategyCEvaluator(
                pengym_tasks=per_agent_heldout,
                step_limit=step_limit,
                eval_episodes=20,
                optimal_rewards=optimal_rewards,
                optimal_steps=optimal_steps,
            )
            for name in agent_names:
                agent = (
                    self._theta_sim_unified if name == "theta_sim_unified"
                    else self._theta_dual if name == "theta_dual"
                    else self._theta_pengym_scratch
                )
                if agent is not None:
                    heldout_evaluator.register_agent(name, agent)
            heldout_report = heldout_evaluator.evaluate_all()
            results["heldout"] = heldout_report
            results["heldout_transfer_metrics"] = heldout_report.get("metrics", {})
            logging.info(f"[Phase 4] Heldout FT_SR: "
                         f"{results['heldout_transfer_metrics'].get('FT_SR', 'N/A')}")

        # ── Tier checkpoint eval + Forgetting / Zero-Shot (A+B) ──
        tier_checkpoints = getattr(self, '_tier_checkpoints', {})
        tier_eval_results: Dict[str, Dict[str, Any]] = {}
        if tier_checkpoints and self._theta_dual is not None:
            logging.info("[Phase 4] Evaluating tier checkpoints for F/Z matrices...")
            for tier_name, ckpt_path in tier_checkpoints.items():
                agent_copy = copy.deepcopy(self._theta_dual)
                agent_copy.load(path=ckpt_path)
                ckpt_agent_name = f"ckpt_{tier_name}"
                ckpt_tasks = self._create_eval_tasks_from(self.pengym_scenarios)
                evaluator.register_agent(ckpt_agent_name, agent_copy)
                tier_eval_results[tier_name] = evaluator.evaluate_agent(
                    ckpt_agent_name, ckpt_tasks, domain="pengym",
                )
            fz = evaluator.compute_forgetting_matrix(tier_eval_results)
            results["forgetting_matrix"] = fz
            logging.info(f"[Phase 4] F/Z summary: "
                         f"mean_F={fz['summary'].get('mean_forgetting', 'N/A')}, "
                         f"mean_Z={fz['summary'].get('mean_zero_shot_transfer', 'N/A')}")

        # ── Learning-speed transfer (C) ──
        dual_rewards = getattr(self, '_dual_per_task_rewards', {})
        scratch_rewards = getattr(self, '_scratch_per_task_rewards', {})
        if dual_rewards and scratch_rewards:
            speed = StrategyCEvaluator.compute_learning_speed(
                dual_rewards, scratch_rewards,
            )
            results["learning_speed"] = speed
            logging.info(f"[Phase 4] Learning speed: "
                         f"TTT speedup={speed['aggregate'].get('mean_ttt_speedup', 'N/A'):.2f}, "
                         f"AUC ratio={speed['aggregate'].get('mean_auc_ratio', 'N/A'):.2f}")

        # ── MetricStore: persist structured metrics (E+F+G) ──
        from src.evaluation.metric_store import MetricStore, FZComputer, CECurveGenerator

        store = MetricStore(seed=self.seed, output_dir=str(self.output_dir))

        # Populate from tier checkpoint evals
        if tier_eval_results:
            for tier_name, tier_result in tier_eval_results.items():
                store.add_checkpoint(f"after_{tier_name}", tier_result)

        # Final eval for theta_dual
        if self._theta_dual is not None:
            final_tasks = self._create_eval_tasks_from(self.pengym_scenarios)
            final_eval = evaluator.evaluate_agent(
                "theta_dual", final_tasks, domain="pengym",
            )
            store.add_checkpoint("final", final_eval)

        # Training curves
        dual_rewards = getattr(self, '_dual_per_task_rewards', {})
        for task_name, rewards in dual_rewards.items():
            ttt = next((i for i, r in enumerate(rewards) if r > 0), len(rewards))
            store.add_training_curve(task_name, rewards, ttt)

        # Transfer + forgetting
        store.set_transfer(results.get("transfer_metrics", {}))
        if "forgetting_matrix" in results:
            store.set_forgetting(results["forgetting_matrix"])

        store.save()
        logging.info(f"[Phase 4] MetricStore saved → {self.output_dir / 'metric_store.json'}")

        # Export F/Z CSV
        if "forgetting_matrix" in results:
            fz_csv_path = str(self.output_dir / "forgetting_matrix.csv")
            FZComputer.save_csv(results["forgetting_matrix"], fz_csv_path)
            logging.info(f"[Phase 4] F/Z CSV → {fz_csv_path}")

        # Export CE curves CSV
        curves = CECurveGenerator.extract_curves(store)
        ce_csv = CECurveGenerator.to_csv(curves, metric="nr")
        if ce_csv:
            ce_path = self.output_dir / "ce_curves_nr.csv"
            with open(ce_path, "w", newline="") as f:
                f.write(ce_csv)
            logging.info(f"[Phase 4] CE curves CSV → {ce_path}")

        return results

    # ==================================================================
    # Convenience: fresh PenGym adapter factory for eval isolation
    # ==================================================================

    def _create_eval_tasks_from(self, scenario_paths: List[str]) -> list:
        """Create fresh PenGym adapters from given scenario paths."""
        from src.envs.adapters.pengym_host_adapter import PenGymHostAdapter
        return [
            PenGymHostAdapter.from_scenario(
                p, seed=self.seed, use_unified_encoding=True,
            )
            for p in scenario_paths
        ]

    # ==================================================================
    # Policy-level backward transfer metrics
    # ==================================================================

    def _collect_sim_states(self, n: int = 200) -> "torch.Tensor":
        """Collect *n* observation vectors from sim tasks using θ_sim policy.

        States are gathered by rolling out the learned sim policy and
        recording observations.  If sim tasks or θ_sim are unavailable,
        returns an empty tensor.
        """
        import torch

        if self._theta_sim_unified is None:
            return torch.empty(0)

        sim_tasks = getattr(self, '_sim_tasks', None)
        if not sim_tasks:
            return torch.empty(0)

        states: list = []
        evaluator = self._theta_sim_unified.cl_agent.get_task_evaluator(
            on_train=False,
        )

        per_task = max(1, n // len(sim_tasks))
        for task in sim_tasks:
            obs = task.reset()
            for _ in range(per_task + 20):          # small buffer for early dones
                states.append(obs.copy() if hasattr(obs, 'copy') else obs)
                action = evaluator.Policy.evaluate(obs)
                obs, _reward, done, _info = task.step(action)
                if done:
                    obs = task.reset()
                if len(states) >= n:
                    break
            if len(states) >= n:
                break

        states = states[:n]
        return torch.FloatTensor(states)

    def _compute_policy_bt_metrics(self) -> Dict[str, float]:
        """Compute policy-level backward transfer metrics.

        BT_KL
            Average KL divergence  D_KL(π_sim ‖ π_dual) over sim states.
            Measures how much the action distribution shifted after
            PenGym fine-tuning.

        BT_fisher_dist
            Fisher-weighted L2 distance  Σ_k F_k (θ_dual_k − θ_sim_k)².
            Weights parameter drift by task-importance from EWC.
        """
        import torch

        if self._theta_sim_unified is None or self._theta_dual is None:
            return {}

        metrics: Dict[str, float] = {}

        # ---- BT_KL: D_KL(π_sim ‖ π_dual) on sim states ----
        try:
            states = self._collect_sim_states(n=200)
            if len(states) > 0:
                sim_actor = (
                    self._theta_sim_unified.cl_agent.keeper.Policy.actor
                )
                dual_actor = (
                    self._theta_dual.cl_agent.keeper.Policy.actor
                )
                sim_actor.eval()
                dual_actor.eval()

                device = next(sim_actor.parameters()).device
                states = states.to(device)

                with torch.no_grad():
                    p = sim_actor(states).clamp(min=1e-8)   # (N, A)
                    q = dual_actor(states).clamp(min=1e-8)  # (N, A)
                    kl = (p * (p.log() - q.log())).sum(dim=-1)  # (N,)
                    metrics["BT_KL"] = float(kl.mean().item())
        except Exception as e:
            logging.warning(f"[Phase 4] BT_KL computation failed: {e}")

        # ---- BT_fisher_dist: Σ_k F_k × (θ_dual_k − θ_sim_k)² ----
        try:
            ewc = self._theta_dual.cl_agent.ewc
            if hasattr(ewc, 'importances') and ewc.importances:
                # Use Fisher diagonal from the first sim task
                first_task_id = next(iter(ewc.importances))
                fisher = ewc.importances[first_task_id]
                saved = ewc.saved_params.get(first_task_id, {})

                dual_params = dict(
                    self._theta_dual.cl_agent.keeper.Policy.actor.named_parameters()
                )

                fisher_dist = 0.0
                for name, f_k in fisher.items():
                    if name in dual_params and name in saved:
                        diff = dual_params[name].data - saved[name]
                        fisher_dist += float((f_k * diff.pow(2)).sum().item())

                metrics["BT_fisher_dist"] = fisher_dist
        except Exception as e:
            logging.warning(f"[Phase 4] BT_fisher_dist computation failed: {e}")

        return metrics

    # ==================================================================
    # Convenience: train θ_pengym_scratch
    # ==================================================================

    def train_pengym_scratch(self, eval_freq: int = 5) -> Dict[str, Any]:
        """Train a SCRIPT agent from scratch on PenGym (for comparison).

        This creates θ_pengym_scratch to compare against θ_dual.
        """
        logging.info("\n" + "=" * 60)
        logging.info("[Scratch] Training SCRIPT from scratch on PenGym")
        logging.info("=" * 60)

        from src.envs.adapters.pengym_host_adapter import PenGymHostAdapter

        pengym_tasks = []
        for sc_path in self.pengym_scenarios:
            adapter = PenGymHostAdapter.from_scenario(
                sc_path, seed=self.seed, use_unified_encoding=True,
            )
            pengym_tasks.append(adapter)

        tb_scratch = SummaryWriter(
            log_dir=str(self.tb_dir / "scratch_pengym")
        )
        time_flag = datetime.now().strftime("%Y%m%d_%H%M%S")

        scratch_agent = Agent_CL(
            time_flag=time_flag,
            logger=tb_scratch,
            use_wandb=False,
            method="script",
            policy_name="PPO",
            seed=self.seed,
            config=copy.deepcopy(self.ppo_config),
            cl_config=copy.deepcopy(self.script_config),
        )

        t0 = time.time()
        episode_schedule = self._resolve_episode_schedule(self.pengym_scenarios)
        cl_matrix = scratch_agent.train_continually(
            task_list=pengym_tasks,
            eval_freq=eval_freq,
            save_agent=False,
            verbose=True,
            episode_schedule=episode_schedule,
        )
        train_time = time.time() - t0

        self._theta_pengym_scratch = scratch_agent

        # Collect per-task rewards for learning-speed metric (C)
        scratch_per_task_rewards: Dict[str, list] = {}
        for idx, task_obj in enumerate(pengym_tasks):
            task_name = getattr(task_obj, 'ip', f'task_{idx}')
            scratch_per_task_rewards[task_name] = (
                cl_matrix.Rewards_current_task[idx:idx + 1]
                if hasattr(cl_matrix, 'Rewards_current_task')
                else []
            )
        self._scratch_per_task_rewards = scratch_per_task_rewards

        scratch_model_dir = self.output_dir / "models" / "pengym_scratch"
        scratch_model_dir.mkdir(parents=True, exist_ok=True)
        scratch_agent.save(path=scratch_model_dir)

        tb_scratch.close()

        results = {
            "num_tasks": len(pengym_tasks),
            "train_time_s": round(train_time, 2),
            "final_sr": scratch_agent.eval_success_rate,
            "final_reward": scratch_agent.eval_rewards,
            "model_dir": str(scratch_model_dir),
        }
        logging.info(f"[Scratch] Complete: SR={results['final_sr']:.2%}")
        return results

    # ==================================================================
    # Save / Load
    # ==================================================================

    def save_results(self, path: Optional[str] = None) -> None:
        """Save current results to JSON."""
        path = Path(path) if path else self.output_dir / "dual_trainer_results.json"
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        logging.info(f"[DualTrainer] Results saved → {path}")
