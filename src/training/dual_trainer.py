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
        ppo_kwargs: Optional[Dict] = None,
        script_kwargs: Optional[Dict] = None,
        seed: int = 42,
        output_dir: str = "outputs/strategy_c",
    ):
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.sim_scenarios = [str(p) for p in sim_scenarios]
        self.pengym_scenarios = [str(p) for p in pengym_scenarios]
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
        """Phase 3 — Fine-tune the transferred agent on PenGym with EWC constraints."""
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

        # Reset task counter so the agent treats PenGym tasks as continuation
        # but with the transferred weights and discounted Fisher
        cl_matrix = self._theta_dual.train_continually(
            task_list=self._pengym_tasks,
            eval_freq=eval_freq,
            save_agent=False,
            verbose=True,
        )
        phase3_time = time.time() - t0

        # Save Phase 3 model
        phase3_model_dir = self.output_dir / "models" / "phase3_dual"
        phase3_model_dir.mkdir(parents=True, exist_ok=True)
        self._theta_dual.save(path=phase3_model_dir)

        tb_phase3.close()

        results = {
            "num_tasks": len(self._pengym_tasks),
            "train_time_s": round(phase3_time, 2),
            "final_sr": self._theta_dual.eval_success_rate,
            "final_reward": self._theta_dual.eval_rewards,
            "model_dir": str(phase3_model_dir),
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

        Each agent is evaluated on PenGym tasks. Transfer metrics
        (Forward Transfer, Backward Transfer, Transfer Ratio) are
        computed automatically by StrategyCEvaluator.
        """
        logging.info("\n" + "=" * 60)
        logging.info("[Phase 4] Multi-Agent Evaluation")
        logging.info("=" * 60)

        from src.evaluation.strategy_c_eval import StrategyCEvaluator

        pengym_tasks = getattr(self, '_pengym_tasks', None)
        if not pengym_tasks:
            logging.warning("[Phase 4] No PenGym tasks for evaluation")
            return {"agents": {}}

        step_limit = getattr(self.ppo_config, 'eval_step_limit',
                             self.ppo_config.step_limit)
        evaluator = StrategyCEvaluator(
            pengym_tasks=pengym_tasks,
            step_limit=step_limit,
        )

        if self._theta_sim_unified is not None:
            evaluator.register_agent("theta_sim_unified", self._theta_sim_unified)
        if self._theta_dual is not None:
            evaluator.register_agent("theta_dual", self._theta_dual)
        if self._theta_pengym_scratch is not None:
            evaluator.register_agent("theta_pengym_scratch", self._theta_pengym_scratch)

        report = evaluator.evaluate_all()
        evaluator.print_report(report)

        # Save detailed evaluation report
        eval_report_path = self.output_dir / "strategy_c_eval_report.json"
        evaluator.save_report(report, str(eval_report_path))

        # Flatten for backward-compatible results format
        results = {"agents": {}, "transfer_metrics": report.get("metrics", {})}
        for name, domains in report.get("agents", {}).items():
            pengym_data = domains.get("pengym", {})
            results["agents"][name] = {
                "success_rate": pengym_data.get("success_rate", 0),
                "total_rewards": pengym_data.get("total_rewards", 0),
                "per_task": pengym_data.get("per_task", []),
            }
            sr = pengym_data.get("success_rate", "N/A")
            logging.info(f"    {name}: SR={sr:.2%}" if isinstance(sr, float) else f"    {name}: {sr}")

        metrics = results["transfer_metrics"]
        if metrics:
            ft = metrics.get("forward_transfer")
            bt = metrics.get("backward_transfer")
            if ft is not None:
                logging.info(f"  Forward Transfer: {ft:+.2%}")
            if bt is not None:
                logging.info(f"  Backward Transfer: {bt:+.2%}")

        return results

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
        cl_matrix = scratch_agent.train_continually(
            task_list=pengym_tasks,
            eval_freq=eval_freq,
            save_agent=False,
            verbose=True,
        )
        train_time = time.time() - t0

        self._theta_pengym_scratch = scratch_agent

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
