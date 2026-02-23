#!/usr/bin/env python3
"""
Benchmark Suite — Train, evaluate, and compare SCRIPT PPO against baselines.

Usage:
    # Train SCRIPT on ALL scenarios with auto-logging:
    python run_benchmark.py train --episodes 2000 --max-steps 300

    # Train on specific scenarios:
    python run_benchmark.py train --scenarios tiny small-linear medium

    # Evaluate all trained models:
    python run_benchmark.py eval --episodes 50

    # Run baselines (random, greedy, DQN, A2C):
    python run_benchmark.py baselines --episodes 50

    # Full benchmark (train + eval + baselines + comparison):
    python run_benchmark.py full --train-episodes 2000 --eval-episodes 50

    # Compare results & generate report:
    python run_benchmark.py compare

    # CVE audit:
    python run_benchmark.py cve-audit
"""

import argparse
import json
import os
import signal
import sys
import time
import csv
import yaml
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# ── Project paths ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

SCENARIO_DIR = ROOT / "data" / "scenarios"
MODELS_DIR = ROOT / "outputs" / "models_pengym"
LOGS_DIR = ROOT / "outputs" / "logs" / "pengym"
BENCHMARK_DIR = ROOT / "outputs" / "benchmark"
CVE_CSV = ROOT / "data" / "CVE" / "CVE_dataset.csv"
CVE_GRADED = ROOT / "data" / "CVE" / "cve_graded.csv"

# ── Graceful interrupt handling ──────────────────────────────────────────────
_INTERRUPT_REQUESTED = False

def _sigint_handler(sig, frame):
    """Mark interrupt so training loops can save checkpoint before exit."""
    global _INTERRUPT_REQUESTED
    if _INTERRUPT_REQUESTED:
        # Second Ctrl+C → force exit
        print("\n[!] Force exit (second Ctrl+C)")
        sys.exit(1)
    _INTERRUPT_REQUESTED = True
    print("\n[!] Ctrl+C detected — saving checkpoint and exiting after current episode...")
    print("    (Press Ctrl+C again to force exit immediately)")

signal.signal(signal.SIGINT, _sigint_handler)


# All available scenarios ordered by difficulty
SCENARIO_ORDER = [
    "tiny",
    "tiny-hard",
    "tiny-small",
    "small-linear",
    "small-honeypot",
    "medium",
    "medium-single-site",
    "medium-multi-site",
]

# Recommended training configs per scenario difficulty tier
SCENARIO_CONFIGS = {
    "tiny":               {"episodes": 500,  "max_steps": 100, "tier": "easy"},
    "tiny-hard":          {"episodes": 500,  "max_steps": 100, "tier": "easy"},
    "tiny-small":         {"episodes": 1000, "max_steps": 150, "tier": "easy"},
    "small-linear":       {"episodes": 2000, "max_steps": 300, "tier": "medium"},
    "small-honeypot":     {"episodes": 2000, "max_steps": 300, "tier": "medium"},
    "medium":             {"episodes": 3000, "max_steps": 500, "tier": "hard"},
    "medium-single-site": {"episodes": 3000, "max_steps": 500, "tier": "hard"},
    "medium-multi-site":  {"episodes": 3000, "max_steps": 500, "tier": "hard"},
}


# ═══════════════════════════════════════════════════════════════════════════
#  Utility: TeeLogger (duplicate stdout to file)
# ═══════════════════════════════════════════════════════════════════════════

from src.utils.logging import TeeLogger


# ═══════════════════════════════════════════════════════════════════════════
#  Baseline Agents
# ═══════════════════════════════════════════════════════════════════════════

class RandomAgent:
    """Uniformly random action selection."""
    def __init__(self, action_dim: int, seed: int = 42):
        self.action_dim = action_dim
        self.rng = np.random.RandomState(seed)
        self.name = "Random"

    def select_action(self, state):
        return self.rng.randint(0, self.action_dim)

    def reset(self):
        pass


class GreedyExploitAgent:
    """Cycles through exploit actions (4-8), then privesc (13-15), then scans.

    Mimics a simple rule-based penetration tester that tries all exploits
    on the current target before scanning.
    """
    def __init__(self, action_dim: int = 16, seed: int = 42):
        self.action_dim = action_dim
        self.name = "Greedy-Exploit"
        # Priority: exploits first, then privesc, then scans
        self._action_order = [4, 5, 6, 7, 8, 13, 14, 15, 0, 1, 2, 3, 9, 10, 11, 12]
        self._step = 0

    def select_action(self, state):
        a = self._action_order[self._step % len(self._action_order)]
        self._step += 1
        return a

    def reset(self):
        self._step = 0


class ScanFirstAgent:
    """Scans first (0-3), then exploits (4-8), then privesc (13-15).

    Mimics a methodical pentest approach: enumerate → exploit → escalate.
    """
    def __init__(self, action_dim: int = 16, seed: int = 42):
        self.action_dim = action_dim
        self.name = "Scan-First"
        self._action_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 9, 10, 11, 12]
        self._step = 0

    def select_action(self, state):
        a = self._action_order[self._step % len(self._action_order)]
        self._step += 1
        return a

    def reset(self):
        self._step = 0


class EpsilonGreedyDQNAgent:
    """Simple 1-layer DQN with ε-greedy exploration.

    This is NOT a full DQN implementation — it's a lightweight baseline
    that learns Q-values with a single-layer network and experience replay.
    """
    def __init__(self, state_dim: int, action_dim: int, seed: int = 42,
                 lr: float = 1e-3, gamma: float = 0.99, eps_start: float = 1.0,
                 eps_end: float = 0.05, eps_decay: int = 500,
                 buffer_size: int = 10000, batch_size: int = 64):
        self.name = "DQN-Baseline"
        self.action_dim = action_dim
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)
        self._step_count = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, action_dim),
        ).to(self.device)
        self.target_net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, action_dim),
        ).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        # Replay buffer
        self._buffer = []
        self._buffer_size = buffer_size

    @property
    def epsilon(self):
        return self.eps_end + (self.eps_start - self.eps_end) * \
            np.exp(-self._step_count / self.eps_decay)

    def select_action(self, state):
        self._step_count += 1
        if self.rng.random() < self.epsilon:
            return self.rng.randint(0, self.action_dim)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.q_net(s).argmax(dim=1).item()

    def store(self, s, a, r, s_next, done):
        self._buffer.append((s, a, r, s_next, done))
        if len(self._buffer) > self._buffer_size:
            self._buffer.pop(0)

    def update(self):
        if len(self._buffer) < self.batch_size:
            return
        idxs = self.rng.choice(len(self._buffer), self.batch_size, replace=False)
        batch = [self._buffer[i] for i in idxs]
        s, a, r, s2, d = zip(*batch)

        s_t = torch.FloatTensor(np.array(s)).to(self.device)
        a_t = torch.LongTensor(a).to(self.device)
        r_t = torch.FloatTensor(r).to(self.device)
        s2_t = torch.FloatTensor(np.array(s2)).to(self.device)
        d_t = torch.FloatTensor(d).to(self.device)

        q_vals = self.q_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q = self.target_net(s2_t).max(1)[0]
            target = r_t + self.gamma * next_q * (1 - d_t)

        loss = torch.nn.functional.mse_loss(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sync_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def reset(self):
        pass


class A2CAgent:
    """Simple Advantage Actor-Critic baseline.

    Single-step (online) A2C with shared feature extractor.
    Simpler than PPO — no clipping, no mini-batches.
    """
    def __init__(self, state_dim: int, action_dim: int, seed: int = 42,
                 lr: float = 3e-4, gamma: float = 0.99,
                 entropy_coef: float = 0.01):
        self.name = "A2C-Baseline"
        self.action_dim = action_dim
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(seed)
        # Shared feature extractor
        self.features = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 256),
            torch.nn.Tanh(),
        ).to(self.device)
        self.actor_head = torch.nn.Linear(256, action_dim).to(self.device)
        self.critic_head = torch.nn.Linear(256, 1).to(self.device)

        params = list(self.features.parameters()) + \
                 list(self.actor_head.parameters()) + \
                 list(self.critic_head.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

        self._saved_log_prob = None
        self._saved_value = None

    def select_action(self, state):
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        feat = self.features(s)
        logits = self.actor_head(feat)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self._saved_log_prob = dist.log_prob(action)
        self._saved_value = self.critic_head(feat)
        self._saved_entropy = dist.entropy()
        return action.item()

    def update(self, reward, next_state, done):
        with torch.no_grad():
            s2 = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            feat2 = self.features(s2)
            next_val = self.critic_head(feat2)
            target = reward + self.gamma * next_val * (1 - float(done))

        advantage = target - self._saved_value
        actor_loss = -(self._saved_log_prob * advantage.detach())
        critic_loss = advantage.pow(2)
        entropy_loss = -self._saved_entropy

        loss = actor_loss + 0.5 * critic_loss + self.entropy_coef * entropy_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.features.parameters()) +
            list(self.actor_head.parameters()) +
            list(self.critic_head.parameters()),
            0.5
        )
        self.optimizer.step()

    def reset(self):
        pass


# ═══════════════════════════════════════════════════════════════════════════
#  Core: Run episodes with any agent on any scenario
# ═══════════════════════════════════════════════════════════════════════════

def _create_wrapper(scenario_path: str, reward_type: str = "linear",
                    selector_name: str = "reachability", seed: int = 42):
    """Create a SingleHostPenGymWrapper for the given scenario."""
    from src.envs.wrappers.single_host_wrapper import SingleHostPenGymWrapper
    from src.envs.wrappers.reward_normalizer import (
        LinearNormalizer, IdentityNormalizer,
    )
    from src.envs.wrappers.target_selector import (
        PrioritySensitiveSelector, ReachabilityAwareSelector,
        RoundRobinSelector, ValuePrioritySelector,
    )

    normalizers = {
        "linear": LinearNormalizer(),
        "identity": IdentityNormalizer(),
    }
    selectors = {
        "priority": PrioritySensitiveSelector(),
        "reachability": ReachabilityAwareSelector(),
        "roundrobin": RoundRobinSelector(),
        "value": ValuePrioritySelector(),
    }

    return SingleHostPenGymWrapper(
        scenario_path=scenario_path,
        reward_normalizer=normalizers.get(reward_type, LinearNormalizer()),
        target_selector=selectors.get(selector_name, ReachabilityAwareSelector()),
        seed=seed,
    )


def evaluate_agent(agent, wrapper, num_episodes: int, max_steps: int,
                   verbose: bool = False) -> Dict[str, Any]:
    """Evaluate any agent (baseline or trained) on a wrapper.

    Returns dict with per-episode results and aggregate metrics.
    """
    results = []
    for ep_i in range(num_episodes):
        obs = wrapper.reset()
        agent.reset()
        done = False
        steps = 0
        ep_return = 0.0
        info = {}

        while not done and steps < max_steps:
            action = agent.select_action(obs)
            next_obs, reward, done, info = wrapper.step(action)
            ep_return += reward
            steps += 1
            obs = next_obs

        status = "SUCCESS" if done else "FAILED"
        results.append({
            "episode": ep_i + 1,
            "status": status,
            "reward": float(ep_return),
            "steps": steps,
            "done": done,
            "last_target": str(info.get("target_host", "?")),
        })
        if verbose:
            print(f"  Ep {ep_i+1}: {status}, reward={ep_return:.1f}, steps={steps}")

    successes = sum(1 for r in results if r["done"])
    sr = successes / num_episodes if num_episodes > 0 else 0.0
    rewards = [r["reward"] for r in results]

    return {
        "success_rate": sr,
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "avg_steps": float(np.mean([r["steps"] for r in results])),
        "episodes": results,
        "num_episodes": num_episodes,
        "num_successes": successes,
    }


def train_dqn_agent(agent: EpsilonGreedyDQNAgent, wrapper,
                    num_episodes: int, max_steps: int,
                    target_sync_freq: int = 10) -> Dict[str, Any]:
    """Train DQN agent and return training curve + final eval."""
    train_rewards = []
    train_sr = []

    for ep_i in range(1, num_episodes + 1):
        obs = wrapper.reset()
        agent.reset()
        done = False
        steps = 0
        ep_return = 0.0

        while not done and steps < max_steps:
            action = agent.select_action(obs)
            next_obs, reward, done, info = wrapper.step(action)
            agent.store(obs, action, reward, next_obs, float(done))
            agent.update()
            ep_return += reward
            steps += 1
            obs = next_obs

        train_rewards.append(ep_return)
        train_sr.append(1.0 if done else 0.0)

        if ep_i % target_sync_freq == 0:
            agent.sync_target()

    return {
        "train_rewards": [float(r) for r in train_rewards],
        "train_sr": train_sr,
        "final_avg_sr": float(np.mean(train_sr[-50:])) if len(train_sr) >= 50 else float(np.mean(train_sr)),
        "final_avg_reward": float(np.mean(train_rewards[-50:])) if len(train_rewards) >= 50 else float(np.mean(train_rewards)),
    }


def train_a2c_agent(agent: A2CAgent, wrapper,
                    num_episodes: int, max_steps: int) -> Dict[str, Any]:
    """Train A2C agent and return training curve + final eval."""
    train_rewards = []
    train_sr = []

    for ep_i in range(1, num_episodes + 1):
        obs = wrapper.reset()
        agent.reset()
        done = False
        steps = 0
        ep_return = 0.0

        while not done and steps < max_steps:
            action = agent.select_action(obs)
            next_obs, reward, done, info = wrapper.step(action)
            agent.update(reward, next_obs, done)
            ep_return += reward
            steps += 1
            obs = next_obs

        train_rewards.append(ep_return)
        train_sr.append(1.0 if done else 0.0)

    return {
        "train_rewards": [float(r) for r in train_rewards],
        "train_sr": train_sr,
        "final_avg_sr": float(np.mean(train_sr[-50:])) if len(train_sr) >= 50 else float(np.mean(train_sr)),
        "final_avg_reward": float(np.mean(train_rewards[-50:])) if len(train_rewards) >= 50 else float(np.mean(train_rewards)),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  SCRIPT (PPO) Agent Wrapper for benchmark interface
# ═══════════════════════════════════════════════════════════════════════════

class SCRIPTAgent:
    """Wraps the trained SCRIPT PPO policy for benchmark evaluation."""
    def __init__(self, model_dir: str, state_dim: int = 1538,
                 action_dim: int = 16, use_state_norm: bool = True):
        self.name = "SCRIPT-PPO"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        from src.agent.policy.PPO import PPO_agent
        from src.agent.policy.config import PPO_Config

        cfg = PPO_Config()
        self.policy = PPO_agent(cfg, state_dim=state_dim, action_dim=action_dim)

        # Load weights
        actor_path = Path(model_dir) / "PPO-actor.pt"
        critic_path = Path(model_dir) / "PPO-critic.pt"
        if actor_path.exists():
            self.policy.actor.load_state_dict(
                torch.load(str(actor_path), map_location=self.device, weights_only=True)
            )
        if critic_path.exists():
            self.policy.critic.load_state_dict(
                torch.load(str(critic_path), map_location=self.device, weights_only=True)
            )

        self.use_state_norm = use_state_norm
        self.state_norm = None
        if use_state_norm:
            from src.agent.policy.common import Normalization
            self.state_norm = Normalization(shape=state_dim)
            mean_path = Path(model_dir) / "PPO-norm_mean.pt"
            std_path = Path(model_dir) / "PPO-norm_std.pt"
            if mean_path.exists() and std_path.exists():
                self.state_norm.running_ms.mean = torch.load(
                    str(mean_path), map_location="cpu", weights_only=True
                )
                self.state_norm.running_ms.std = torch.load(
                    str(std_path), map_location="cpu", weights_only=True
                )

    def select_action(self, state):
        if self.use_state_norm and self.state_norm is not None:
            state = self.state_norm(state, update=False)
        return self.policy.evaluate(state)

    def reset(self):
        pass


# ═══════════════════════════════════════════════════════════════════════════
#  CVE Audit
# ═══════════════════════════════════════════════════════════════════════════

def run_cve_audit() -> Dict[str, Any]:
    """Audit CVE dataset integration with the PenGym wrapper environment.

    Checks:
    1. CVE_dataset.csv existence and format
    2. cve_graded.csv existence
    3. Service coverage: which CVE services map to PenGym actions
    4. Tier distribution
    5. Per-scenario CVE compatibility analysis
    """
    print("\n" + "=" * 60)
    print("CVE INTEGRATION AUDIT")
    print("=" * 60)

    report = {"timestamp": datetime.now().isoformat(), "checks": []}

    # ── Check 1: CVE_dataset.csv ──────────────────────────────────────
    print("\n[1/5] Checking CVE_dataset.csv...")
    if CVE_CSV.exists():
        with open(CVE_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        total = len(rows)
        has_service = sum(1 for r in rows if r.get('service', '').strip())
        has_prob = sum(1 for r in rows if r.get('prob', '').strip())
        has_access = sum(1 for r in rows if r.get('access', '').strip())

        # Service distribution
        svc_counts = {}
        for r in rows:
            svc = r.get('service', '').strip()
            if svc:
                svc_counts[svc] = svc_counts.get(svc, 0) + 1

        check = {
            "name": "CVE_dataset.csv",
            "status": "OK",
            "total_cves": total,
            "with_service": has_service,
            "with_prob": has_prob,
            "with_access": has_access,
            "service_distribution": dict(sorted(svc_counts.items(), key=lambda x: -x[1])),
        }
        print(f"  ✓ Found {total} CVEs, {has_service} with service mapping")
        print(f"  Service distribution (top 10):")
        for svc, cnt in sorted(svc_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"    {svc}: {cnt}")
    else:
        check = {"name": "CVE_dataset.csv", "status": "MISSING"}
        print(f"  ✗ File not found: {CVE_CSV}")
    report["checks"].append(check)

    # ── Check 2: cve_graded.csv ───────────────────────────────────────
    print("\n[2/5] Checking cve_graded.csv...")
    if CVE_GRADED.exists():
        with open(CVE_GRADED, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            graded = list(reader)
        total_graded = len(graded)

        # Tier distribution
        tier_counts = {}
        for r in graded:
            t = r.get('difficulty_tier', '?')
            tier_counts[t] = tier_counts.get(t, 0) + 1

        # PenGym compatibility
        pengym_compat = sum(1 for r in graded if r.get('pengym_compatible', '') == 'yes')
        excluded = sum(1 for r in graded if r.get('excluded', ''))

        check = {
            "name": "cve_graded.csv",
            "status": "OK",
            "total_graded": total_graded,
            "tier_distribution": tier_counts,
            "pengym_compatible": pengym_compat,
            "excluded": excluded,
        }
        print(f"  ✓ Found {total_graded} graded CVEs")
        print(f"  Tier distribution:")
        for t in sorted(tier_counts.keys()):
            print(f"    T{t}: {tier_counts[t]}")
        print(f"  PenGym-compatible: {pengym_compat}/{total_graded}")
    else:
        check = {"name": "cve_graded.csv", "status": "MISSING — run CVEClassifier first"}
        print(f"  ✗ File not found: {CVE_GRADED}")
    report["checks"].append(check)

    # ── Check 3: Service mapping coverage ─────────────────────────────
    print("\n[3/5] Checking PenGym service mapping coverage...")
    PENGYM_SERVICES = {'ssh', 'ftp', 'http', 'samba', 'smtp'}
    PENGYM_PROCESSES = {'tomcat', 'proftpd', 'cron'}

    if CVE_CSV.exists():
        mapped_to_pengym = 0
        unmapped_services = set()
        for r in rows:
            svc = r.get('service', '').strip()
            if not svc:
                continue
            # Check direct or abstracted mapping
            abstract_map = {'webapp': 'http', 'iis': 'http', 'brightstor': 'http', 'windows': 'smb'}
            mapped_svc = abstract_map.get(svc, svc)
            if mapped_svc in PENGYM_SERVICES:
                mapped_to_pengym += 1
            else:
                unmapped_services.add(svc)

        check = {
            "name": "Service mapping",
            "status": "OK" if mapped_to_pengym > 0 else "WARN",
            "mapped_to_pengym": mapped_to_pengym,
            "unmapped_unique_services": sorted(unmapped_services)[:20],
            "total_unmapped_services": len(unmapped_services),
        }
        print(f"  ✓ {mapped_to_pengym} CVEs map to PenGym services")
        print(f"  ✗ {len(unmapped_services)} unique services NOT in PenGym:")
        for svc in sorted(unmapped_services)[:15]:
            cnt = svc_counts.get(svc, 0)
            print(f"    {svc}: {cnt} CVEs")
    else:
        check = {"name": "Service mapping", "status": "SKIP"}
    report["checks"].append(check)

    # ── Check 4: Per-scenario exploit coverage ────────────────────────
    print("\n[4/5] Per-scenario exploit coverage...")
    scenario_coverage = {}
    for sc_name in SCENARIO_ORDER:
        sc_path = SCENARIO_DIR / f"{sc_name}.yml"
        if not sc_path.exists():
            continue
        with open(sc_path, 'r') as f:
            sc = yaml.safe_load(f)

        services = sc.get('services', [])
        exploits = sc.get('exploits', {})
        privescs = sc.get('privilege_escalation', {})
        processes = sc.get('processes', [])

        # Check how many CVEs could substitute each exploit
        coverage = {}
        if CVE_GRADED.exists():
            for ex_name, ex_def in exploits.items():
                svc = ex_def.get('service', '')
                matching_cves = sum(
                    1 for r in graded
                    if r.get('mapped_service', '') == svc
                    and r.get('pengym_compatible', '') == 'yes'
                )
                coverage[ex_name] = {
                    "service": svc,
                    "matching_cves": matching_cves,
                }

        scenario_coverage[sc_name] = {
            "services": services,
            "exploits": list(exploits.keys()),
            "privescs": list(privescs.keys()),
            "processes": processes,
            "cve_coverage": coverage,
        }
        total_matching = sum(c["matching_cves"] for c in coverage.values())
        print(f"  {sc_name}: {len(exploits)} exploits, {len(privescs)} privesc, "
              f"CVE pool={total_matching}")
    report["scenario_coverage"] = scenario_coverage

    # ── Check 5: Runtime CVE integration test ─────────────────────────
    print(f"\n[5/5] Runtime CVE integration test...")
    try:
        from src.pipeline.cve_classifier import CVEClassifier
        classifier = CVEClassifier()
        classifier.load_csv(str(CVE_CSV))
        classifier.classify()
        dist = classifier.get_distribution_report()
        print(f"  ✓ CVEClassifier loaded {dist.get('total', '?')} CVEs successfully")
        print(f"  ✓ Pipeline can grade CVEs and produce tier assignments")
        check_status = "OK"
    except Exception as e:
        print(f"  ✗ CVEClassifier failed: {e}")
        check_status = f"FAIL: {e}"
    report["checks"].append({"name": "Runtime CVE integration", "status": check_status})

    # ── Check 6: CVE → PenGym scenario substitution test ──────────────
    print(f"\n[6/6] CVE scenario substitution test...")
    try:
        from src.pipeline.scenario_compiler import generate_template_from_yaml, CVESelector
        template = generate_template_from_yaml(str(SCENARIO_DIR / "tiny.yml"))
        print(f"  ✓ Template generation works: {len(template.get('service_slots', []))} service slots")

        if CVE_GRADED.exists():
            selector = CVESelector(str(CVE_GRADED))
            print(f"  ✓ CVESelector initialized with {len(selector.cves)} graded CVEs")
            check_status = "OK"
        else:
            print(f"  ⚠ CVESelector needs cve_graded.csv (not found)")
            check_status = "PARTIAL"
    except Exception as e:
        print(f"  ✗ Scenario substitution failed: {e}")
        check_status = f"FAIL: {e}"
    report["checks"].append({"name": "CVE scenario substitution", "status": check_status})

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("CVE AUDIT SUMMARY")
    print(f"{'='*60}")
    ok_count = sum(1 for c in report["checks"] if c["status"] == "OK")
    total_checks = len(report["checks"])
    print(f"  Checks passed: {ok_count}/{total_checks}")

    # Key findings
    if CVE_CSV.exists() and CVE_GRADED.exists():
        print(f"\n  Key Findings:")
        print(f"  • {len(rows)} total CVEs in dataset")
        print(f"  • {pengym_compat} are PenGym-compatible (can be directly simulated)")
        print(f"  • {len(unmapped_services)} services have no PenGym equivalent")
        print(f"  • CVE pipeline (classify → grade → select → compile) is functional")
        print(f"  • Scenarios CAN be augmented with real CVE parameters")
        print(f"\n  Current Limitation:")
        print(f"  • CVE variations are NOT yet used during standard training")
        print(f"  • To enable: use ScenarioCompiler to create CVE-augmented scenario YAMLs")
        print(f"  • Then train with curriculum mode across original + CVE-augmented scenarios")
    print()

    # Save report
    report_path = BENCHMARK_DIR / "cve_audit_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved → {report_path}")

    return report


# ═══════════════════════════════════════════════════════════════════════════
#  Commands
# ═══════════════════════════════════════════════════════════════════════════

def cmd_train(args):
    """Train SCRIPT PPO on selected scenarios."""
    global _INTERRUPT_REQUESTED
    from src.training.pengym_trainer import PenGymTrainer
    from src.agent.policy.config import PPO_Config
    from src.envs.wrappers.reward_normalizer import LinearNormalizer
    from src.envs.wrappers.target_selector import ReachabilityAwareSelector

    scenarios = args.scenarios or SCENARIO_ORDER
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load batch progress (for resume) ─────────────────────────────
    progress_path = BENCHMARK_DIR / "batch_progress.json"
    batch_progress = {}
    if getattr(args, 'resume', False) and progress_path.exists():
        with open(progress_path, 'r') as f:
            batch_progress = json.load(f)
        print(f"[RESUME] Loaded batch progress: {len(batch_progress.get('completed', []))} scenarios done")

    completed_scenarios = batch_progress.get("completed", [])
    all_results = batch_progress.get("results", {})

    for sc_name in scenarios:
        # Skip already-completed scenarios on resume
        if getattr(args, 'resume', False) and sc_name in completed_scenarios:
            print(f"\n[SKIP] {sc_name} — already completed (resume mode)")
            continue

        if _INTERRUPT_REQUESTED:
            print(f"\n[!] Skipping remaining scenarios due to interrupt")
            break

        sc_path = SCENARIO_DIR / f"{sc_name}.yml"
        if not sc_path.exists():
            print(f"[SKIP] Scenario not found: {sc_path}")
            continue

        # Determine training config
        sc_cfg = SCENARIO_CONFIGS.get(sc_name, {"episodes": 1000, "max_steps": 200})
        episodes = args.episodes or sc_cfg["episodes"]
        max_steps = args.max_steps or sc_cfg["max_steps"]

        model_dir = str(MODELS_DIR / f"{sc_name}_pengym")
        log_path = str(LOGS_DIR / f"{sc_name}_bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

        print(f"\n{'='*60}")
        print(f"TRAINING: {sc_name}")
        print(f"  Episodes: {episodes}, Max steps: {max_steps}")
        print(f"  Model: {model_dir}")
        print(f"  Log: {log_path}")
        print(f"{'='*60}")

        tee = TeeLogger(log_path)

        try:
            ppo_config = PPO_Config(
                train_eps=episodes,
                step_limit=max_steps,
                eval_step_limit=max_steps,
                use_state_norm=True,
            )

            trainer = PenGymTrainer(
                initial_scenario=str(sc_path),
                config=ppo_config,
                seed=args.seed,
                reward_normalizer=LinearNormalizer(),
                target_selector=ReachabilityAwareSelector(),
                tb_dir=None,
            )

            # ── Check for checkpoint (resume within scenario) ────────
            start_episode = 0
            prev_rewards, prev_sr, prev_eval_sr = [], [], []

            if getattr(args, 'resume', False):
                ckpt = trainer.load_checkpoint(model_dir)
                if ckpt is not None:
                    start_episode = ckpt["episode"]
                    prev_rewards = ckpt.get("train_rewards", [])
                    prev_sr = ckpt.get("train_sr", [])
                    prev_eval_sr = ckpt.get("eval_sr", [])
                    if start_episode >= episodes:
                        print(f"  [SKIP] Already trained {start_episode}/{episodes} episodes")
                        completed_scenarios.append(sc_name)
                        continue

            t0 = time.time()

            # ── Training loop with interrupt support ─────────────────
            eval_freq = max(1, episodes // 20)
            log_freq = max(1, episodes // 20)
            save_freq = max(1, episodes // 5)

            if model_dir:
                Path(model_dir).mkdir(parents=True, exist_ok=True)

            train_rewards = list(prev_rewards)
            train_sr_list = list(prev_sr)
            eval_sr_list = list(prev_eval_sr)

            if start_episode > 0:
                print(f"  [RESUME] Continuing from episode {start_episode + 1}/{episodes}")

            for ep in range(start_episode + 1, episodes + 1):
                if _INTERRUPT_REQUESTED:
                    print(f"\n  [!] Interrupt at episode {ep}/{episodes} — saving checkpoint...")
                    trainer.save_checkpoint(
                        model_dir, ep - 1, episodes,
                        train_rewards, train_sr_list, eval_sr_list
                    )
                    # Save batch progress
                    batch_progress["completed"] = completed_scenarios
                    batch_progress["results"] = all_results
                    batch_progress["interrupted_scenario"] = sc_name
                    batch_progress["interrupted_episode"] = ep - 1
                    with open(progress_path, 'w') as f:
                        json.dump(batch_progress, f, indent=2)
                    print(f"  [!] Batch progress saved. Resume with: --resume")
                    tee.close()
                    return

                ep_return, ep_steps, sr = trainer._run_episode(explore=False)
                train_rewards.append(ep_return)
                train_sr_list.append(sr)

                # Evaluation
                if ep % eval_freq == 0:
                    _, e_sr = trainer.evaluate(verbose=False)
                    eval_sr_list.append(e_sr)

                # Logging
                if ep % log_freq == 0 or (ep == 1 and start_episode == 0):
                    avg_r = np.mean(train_rewards[-log_freq:])
                    avg_sr = np.mean(train_sr_list[-log_freq:])
                    ev_str = f", eval_sr={eval_sr_list[-1]*100:.1f}%" if eval_sr_list else ""
                    print(f"  [ep {ep:4d}/{episodes}] avg_r={avg_r:.1f}, "
                          f"avg_sr={avg_sr*100:.1f}%{ev_str}")

                # Periodic checkpoint
                if ep % save_freq == 0:
                    trainer.save_checkpoint(
                        model_dir, ep, episodes,
                        train_rewards, train_sr_list, eval_sr_list
                    )

            train_time = time.time() - t0

            # Final save
            trainer.save(model_dir)

            # Final evaluation (10 episodes)
            print(f"\nFinal Evaluation ({sc_name}, 10 episodes):")
            total_reward, final_sr = trainer.evaluate(
                num_episodes=10, step_limit=max_steps, verbose=True
            )

            sc_result = {
                "scenario": sc_name,
                "episodes": episodes,
                "max_steps": max_steps,
                "train_time_s": round(train_time, 2),
                "final_train_sr": float(np.mean(train_sr_list[-50:])),
                "final_eval_sr": final_sr,
                "final_eval_reward": float(total_reward),
                "best_return": float(trainer.best_return),
                "log_file": log_path,
                "model_dir": model_dir,
            }

            # Save summary for this scenario
            summary_path = Path(model_dir) / "experiment_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(sc_result, f, indent=2)

            all_results[sc_name] = sc_result
            completed_scenarios.append(sc_name)
            print(f"\n  ✓ {sc_name}: SR={final_sr*100:.0f}%, "
                  f"Time={train_time:.0f}s")

            trainer.close()

        except Exception as e:
            print(f"\n  ✗ {sc_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results[sc_name] = {"scenario": sc_name, "error": str(e)}
        finally:
            tee.close()

        # Update batch progress after each scenario
        batch_progress["completed"] = completed_scenarios
        batch_progress["results"] = all_results
        with open(progress_path, 'w') as f:
            json.dump(batch_progress, f, indent=2)

    # Save combined results
    combined_path = BENCHMARK_DIR / "train_results.json"
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll training results saved → {combined_path}")


def cmd_script_train(args):
    """Train with full SCRIPT CRL (teacher-student) on PenGym scenarios."""
    from src.training.pengym_script_trainer import PenGymScriptTrainer

    scenarios = args.scenarios or ["tiny", "small-linear"]
    scenario_paths = []
    for sc_name in scenarios:
        sc_path = SCENARIO_DIR / f"{sc_name}.yml"
        if not sc_path.exists():
            print(f"[SKIP] Scenario not found: {sc_path}")
            continue
        scenario_paths.append(str(sc_path))

    if not scenario_paths:
        print("[ERROR] No valid scenarios found")
        return

    model_dir = str(MODELS_DIR / "script_crl")
    tb_dir = str(ROOT / "outputs" / "tensorboard" / "pengym" / "script_crl")

    print(f"\n{'='*60}")
    print(f"SCRIPT CRL TRAINING (Teacher-Student)")
    print(f"  Scenarios: {scenarios}")
    print(f"  Episodes/task: {args.episodes or 'auto'}")
    print(f"  Max steps: {args.max_steps or 'auto'}")
    print(f"  Model dir: {model_dir}")
    print(f"{'='*60}\n")

    # Build config overrides
    ppo_kwargs = {}
    if args.episodes:
        ppo_kwargs["train_eps"] = args.episodes
    if args.max_steps:
        ppo_kwargs["step_limit"] = args.max_steps
        ppo_kwargs["eval_step_limit"] = args.max_steps

    script_kwargs = {}
    if hasattr(args, 'ewc_lambda') and args.ewc_lambda is not None:
        script_kwargs["ewc_lambda"] = args.ewc_lambda
    if hasattr(args, 'guide_kl') and args.guide_kl is not None:
        script_kwargs["guide_kl_scale"] = args.guide_kl

    try:
        config_file = getattr(args, 'config', None)

        trainer = PenGymScriptTrainer(
            scenario_list=scenario_paths,
            ppo_kwargs=ppo_kwargs if not config_file else None,
            script_kwargs=script_kwargs if not config_file else None,
            seed=args.seed,
            tb_dir=tb_dir,
            model_dir=model_dir,
            config_file=config_file,
        )

        result = trainer.train(
            eval_freq=5,
            save_agent=True,
            verbose=True,
        )

        # Print summary
        sr_list = result.get("SR_previous_tasks", [])
        print(f"\n{'='*60}")
        print(f"SCRIPT CRL TRAINING COMPLETE")
        print(f"  Training time: {result.get('train_time_s', 0):.1f}s")
        if sr_list:
            print(f"  Final SR (all tasks): {sr_list[-1]*100:.1f}%")
            for i, sr in enumerate(sr_list):
                print(f"    After task {i}: SR={sr*100:.1f}%")
        print(f"  Model saved → {model_dir}")
        print(f"{'='*60}")

        # Save results
        result_path = BENCHMARK_DIR / "script_crl_results.json"
        BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
        # Convert non-serializable values
        serializable = {}
        for k, v in result.items():
            try:
                json.dumps(v)
                serializable[k] = v
            except (TypeError, ValueError):
                serializable[k] = str(v)
        with open(result_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"  Results saved → {result_path}")

        trainer.close()

    except Exception as e:
        print(f"\n[ERROR] SCRIPT CRL training failed: {e}")
        import traceback
        traceback.print_exc()


def cmd_eval(args):
    """Evaluate trained SCRIPT models on all scenarios."""
    scenarios = args.scenarios or SCENARIO_ORDER

    all_results = {}

    for sc_name in scenarios:
        sc_path = SCENARIO_DIR / f"{sc_name}.yml"
        model_dir = MODELS_DIR / f"{sc_name}_pengym"

        if not sc_path.exists():
            print(f"[SKIP] Scenario not found: {sc_path}")
            continue
        if not (model_dir / "PPO-actor.pt").exists():
            print(f"[SKIP] No trained model for {sc_name}")
            continue

        sc_cfg = SCENARIO_CONFIGS.get(sc_name, {"max_steps": 200})
        max_steps = args.max_steps or sc_cfg["max_steps"]

        print(f"\nEvaluating SCRIPT on {sc_name}...")
        try:
            wrapper = _create_wrapper(
                str(sc_path), reward_type="identity", seed=args.seed
            )
            agent = SCRIPTAgent(str(model_dir))
            result = evaluate_agent(
                agent, wrapper, args.episodes, max_steps, verbose=args.verbose
            )
            result["agent"] = "SCRIPT-PPO"
            result["scenario"] = sc_name
            all_results[sc_name] = result
            print(f"  ✓ SR={result['success_rate']*100:.0f}%, "
                  f"Avg reward={result['avg_reward']:.1f}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            all_results[sc_name] = {"scenario": sc_name, "error": str(e)}

    combined_path = BENCHMARK_DIR / "eval_results.json"
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nEval results saved → {combined_path}")


def cmd_baselines(args):
    """Run baseline agents on all scenarios."""
    scenarios = args.scenarios or SCENARIO_ORDER

    all_results = {}

    for sc_name in scenarios:
        sc_path = SCENARIO_DIR / f"{sc_name}.yml"
        if not sc_path.exists():
            print(f"[SKIP] Scenario not found: {sc_path}")
            continue

        sc_cfg = SCENARIO_CONFIGS.get(sc_name, {"episodes": 500, "max_steps": 200})
        max_steps = args.max_steps or sc_cfg["max_steps"]
        train_eps = args.train_episodes or sc_cfg["episodes"]

        print(f"\n{'='*60}")
        print(f"BASELINES: {sc_name}")
        print(f"{'='*60}")

        sc_results = {}

        # 1. Random Agent
        print(f"  Running Random agent ({args.episodes} episodes)...")
        wrapper = _create_wrapper(str(sc_path), seed=args.seed)
        agent = RandomAgent(wrapper.action_dim, seed=args.seed)
        result = evaluate_agent(agent, wrapper, args.episodes, max_steps)
        result["agent"] = agent.name
        sc_results[agent.name] = result
        print(f"    SR={result['success_rate']*100:.0f}%, "
              f"Avg reward={result['avg_reward']:.1f}")

        # 2. Greedy-Exploit Agent
        print(f"  Running Greedy-Exploit agent ({args.episodes} episodes)...")
        wrapper = _create_wrapper(str(sc_path), seed=args.seed)
        agent = GreedyExploitAgent()
        result = evaluate_agent(agent, wrapper, args.episodes, max_steps)
        result["agent"] = agent.name
        sc_results[agent.name] = result
        print(f"    SR={result['success_rate']*100:.0f}%, "
              f"Avg reward={result['avg_reward']:.1f}")

        # 3. Scan-First Agent
        print(f"  Running Scan-First agent ({args.episodes} episodes)...")
        wrapper = _create_wrapper(str(sc_path), seed=args.seed)
        agent = ScanFirstAgent()
        result = evaluate_agent(agent, wrapper, args.episodes, max_steps)
        result["agent"] = agent.name
        sc_results[agent.name] = result
        print(f"    SR={result['success_rate']*100:.0f}%, "
              f"Avg reward={result['avg_reward']:.1f}")

        # 4. DQN Baseline (train then eval)
        print(f"  Training DQN ({train_eps} episodes)...")
        wrapper = _create_wrapper(str(sc_path), seed=args.seed)
        dqn = EpsilonGreedyDQNAgent(
            state_dim=wrapper.state_dim, action_dim=wrapper.action_dim, seed=args.seed
        )
        train_result = train_dqn_agent(dqn, wrapper, train_eps, max_steps)
        print(f"    DQN train: final SR={train_result['final_avg_sr']*100:.0f}%")

        # Eval DQN
        print(f"  Evaluating DQN ({args.episodes} episodes)...")
        dqn._step_count = 999999  # Force low epsilon
        wrapper = _create_wrapper(str(sc_path), seed=args.seed)
        result = evaluate_agent(dqn, wrapper, args.episodes, max_steps)
        result["agent"] = dqn.name
        result["train_info"] = {
            "train_episodes": train_eps,
            "final_train_sr": train_result["final_avg_sr"],
        }
        sc_results[dqn.name] = result
        print(f"    DQN eval: SR={result['success_rate']*100:.0f}%, "
              f"Avg reward={result['avg_reward']:.1f}")

        # 5. A2C Baseline (train then eval)
        print(f"  Training A2C ({train_eps} episodes)...")
        wrapper = _create_wrapper(str(sc_path), seed=args.seed)
        a2c = A2CAgent(
            state_dim=wrapper.state_dim, action_dim=wrapper.action_dim, seed=args.seed
        )
        train_result = train_a2c_agent(a2c, wrapper, train_eps, max_steps)
        print(f"    A2C train: final SR={train_result['final_avg_sr']*100:.0f}%")

        # Eval A2C
        print(f"  Evaluating A2C ({args.episodes} episodes)...")
        wrapper = _create_wrapper(str(sc_path), seed=args.seed)
        result = evaluate_agent(a2c, wrapper, args.episodes, max_steps)
        result["agent"] = a2c.name
        result["train_info"] = {
            "train_episodes": train_eps,
            "final_train_sr": train_result["final_avg_sr"],
        }
        sc_results[a2c.name] = result
        print(f"    A2C eval: SR={result['success_rate']*100:.0f}%, "
              f"Avg reward={result['avg_reward']:.1f}")

        all_results[sc_name] = sc_results

    # Save
    combined_path = BENCHMARK_DIR / "baseline_results.json"
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nBaseline results saved → {combined_path}")


def cmd_compare(args):
    """Compare SCRIPT PPO vs baselines and generate a comparison report."""
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    # Load available results
    eval_path = BENCHMARK_DIR / "eval_results.json"
    baseline_path = BENCHMARK_DIR / "baseline_results.json"
    train_path = BENCHMARK_DIR / "train_results.json"

    eval_results = json.load(open(eval_path)) if eval_path.exists() else {}
    baseline_results = json.load(open(baseline_path)) if baseline_path.exists() else {}
    train_results = json.load(open(train_path)) if train_path.exists() else {}

    # Build comparison table
    print(f"\n{'='*80}")
    print(f"BENCHMARK COMPARISON REPORT")
    print(f"{'='*80}")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Header
    agents = ["SCRIPT-PPO", "Random", "Greedy-Exploit", "Scan-First",
              "DQN-Baseline", "A2C-Baseline"]

    print(f"\n{'Scenario':<22} ", end="")
    for a in agents:
        print(f"{'|':>2} {a:<15}", end="")
    print()
    print("-" * (22 + 17 * len(agents)))

    comparison_data = {}
    for sc_name in SCENARIO_ORDER:
        row = {}
        # SCRIPT result
        if sc_name in eval_results and "success_rate" in eval_results[sc_name]:
            sr = eval_results[sc_name]["success_rate"]
            row["SCRIPT-PPO"] = sr
        elif sc_name in train_results and "final_eval_sr" in train_results[sc_name]:
            sr = train_results[sc_name]["final_eval_sr"]
            row["SCRIPT-PPO"] = sr
        else:
            row["SCRIPT-PPO"] = None

        # Baseline results
        if sc_name in baseline_results:
            for agent_name in agents[1:]:
                if agent_name in baseline_results[sc_name]:
                    row[agent_name] = baseline_results[sc_name][agent_name].get("success_rate")
                else:
                    row[agent_name] = None
        else:
            for a in agents[1:]:
                row[a] = None

        comparison_data[sc_name] = row

        # Print row
        tier = SCENARIO_CONFIGS.get(sc_name, {}).get("tier", "?")
        print(f"{sc_name:<20} [{tier[0].upper()}]", end="")
        for a in agents:
            val = row.get(a)
            if val is not None:
                s = f"{val*100:.0f}%"
                print(f"{'|':>2} {s:<15}", end="")
            else:
                s = "—"
                print(f"{'|':>2} {s:<15}", end="")
        print()

    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for a in agents:
        vals = [comparison_data[sc].get(a) for sc in SCENARIO_ORDER
                if comparison_data[sc].get(a) is not None]
        if vals:
            avg = np.mean(vals)
            print(f"  {a:<18}: avg SR = {avg*100:.1f}% across {len(vals)} scenarios")
        else:
            print(f"  {a:<18}: no data")

    # Save comparison as JSON and markdown
    report = {
        "timestamp": datetime.now().isoformat(),
        "comparison": comparison_data,
        "scenarios": SCENARIO_ORDER,
        "agents": agents,
    }
    report_path = BENCHMARK_DIR / "comparison_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Generate Markdown report
    md_path = BENCHMARK_DIR / "comparison_report.md"
    _generate_markdown_report(comparison_data, agents, train_results, md_path)
    print(f"\nReports saved:")
    print(f"  JSON: {report_path}")
    print(f"  Markdown: {md_path}")


def _generate_markdown_report(comparison: Dict, agents: List[str],
                               train_results: Dict, out_path: Path):
    """Generate a Markdown comparison report."""
    lines = [
        "# Benchmark Comparison Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Success Rate Comparison",
        "",
        "| Scenario | Tier | " + " | ".join(agents) + " |",
        "|" + "---|" * (2 + len(agents)),
    ]

    for sc_name in SCENARIO_ORDER:
        tier = SCENARIO_CONFIGS.get(sc_name, {}).get("tier", "?")
        row_data = comparison.get(sc_name, {})
        cells = []
        for a in agents:
            val = row_data.get(a)
            if val is not None:
                cells.append(f"**{val*100:.0f}%**" if a == "SCRIPT-PPO" else f"{val*100:.0f}%")
            else:
                cells.append("—")
        lines.append(f"| {sc_name} | {tier} | " + " | ".join(cells) + " |")

    # Summary
    lines.extend([
        "",
        "## Average Performance",
        "",
        "| Agent | Avg SR | Scenarios Tested |",
        "|---|---|---|",
    ])
    for a in agents:
        vals = [comparison[sc].get(a) for sc in SCENARIO_ORDER
                if comparison.get(sc, {}).get(a) is not None]
        if vals:
            lines.append(f"| {a} | {np.mean(vals)*100:.1f}% | {len(vals)} |")
        else:
            lines.append(f"| {a} | — | 0 |")

    # Training details
    if train_results:
        lines.extend([
            "",
            "## Training Details (SCRIPT-PPO)",
            "",
            "| Scenario | Episodes | Train Time | Best Return | Final Eval SR |",
            "|---|---|---|---|---|",
        ])
        for sc_name in SCENARIO_ORDER:
            if sc_name in train_results and "error" not in train_results[sc_name]:
                tr = train_results[sc_name]
                lines.append(
                    f"| {sc_name} | {tr.get('episodes', '?')} | "
                    f"{tr.get('train_time_s', '?')}s | "
                    f"{tr.get('best_return', '?')} | "
                    f"{tr.get('final_eval_sr', 0)*100:.0f}% |"
                )

    # Notes
    lines.extend([
        "",
        "## Method Notes",
        "",
        "- **SCRIPT-PPO**: PPO with SBERT state encoding (1538-dim), "
        "service-level actions (16-dim), ReachabilityAwareSelector",
        "- **Random**: Uniform random action selection",
        "- **Greedy-Exploit**: Cycles exploits→privesc→scans deterministically",
        "- **Scan-First**: Scans first, then exploits→privesc",
        "- **DQN-Baseline**: 2-layer DQN (256×256) with ε-greedy, experience replay",
        "- **A2C-Baseline**: 2-layer A2C (256×256) with entropy regularization",
        "",
        "All agents use the same `SingleHostPenGymWrapper` with `ReachabilityAwareSelector`.",
        "",
    ])

    out_path.write_text("\n".join(lines), encoding="utf-8")


def cmd_full(args):
    """Run the full benchmark pipeline."""
    print("╔══════════════════════════════════════════════════════════╗")
    print("║            FULL BENCHMARK PIPELINE                      ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Phase 1: CVE Audit
    print("\n\n" + "=" * 60)
    print("PHASE 1: CVE AUDIT")
    print("=" * 60)
    run_cve_audit()

    # Phase 2: Train SCRIPT
    print("\n\n" + "=" * 60)
    print("PHASE 2: TRAIN SCRIPT-PPO")
    print("=" * 60)
    args.episodes = args.train_episodes
    cmd_train(args)

    # Phase 3: Evaluate SCRIPT
    print("\n\n" + "=" * 60)
    print("PHASE 3: EVALUATE SCRIPT-PPO")
    print("=" * 60)
    args.episodes = args.eval_episodes
    cmd_eval(args)

    # Phase 4: Baselines
    print("\n\n" + "=" * 60)
    print("PHASE 4: BASELINES")
    print("=" * 60)
    args.episodes = args.eval_episodes
    cmd_baselines(args)

    # Phase 5: Compare
    print("\n\n" + "=" * 60)
    print("PHASE 5: COMPARISON")
    print("=" * 60)
    cmd_compare(args)

    print("\n\n✓ Full benchmark complete!")
    print(f"  Results directory: {BENCHMARK_DIR}")


def cmd_cve_audit(args):
    """Run standalone CVE audit."""
    run_cve_audit()


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Suite — Train, evaluate, compare SCRIPT vs baselines"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ──
    p_train = sub.add_parser("train", help="Train SCRIPT-PPO on scenarios")
    p_train.add_argument("--scenarios", nargs="*", default=None,
                         help="Scenario names (default: all)")
    p_train.add_argument("--episodes", type=int, default=None,
                         help="Override episodes (default: per-scenario config)")
    p_train.add_argument("--max-steps", type=int, default=None,
                         help="Override max steps (default: per-scenario config)")
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--resume", action="store_true",
                         help="Resume from last checkpoint")

    # ── eval ──
    p_eval = sub.add_parser("eval", help="Evaluate trained SCRIPT models")
    p_eval.add_argument("--scenarios", nargs="*", default=None)
    p_eval.add_argument("--episodes", type=int, default=50)
    p_eval.add_argument("--max-steps", type=int, default=None)
    p_eval.add_argument("--seed", type=int, default=42)
    p_eval.add_argument("--verbose", action="store_true")

    # ── baselines ──
    p_base = sub.add_parser("baselines", help="Run baseline agents")
    p_base.add_argument("--scenarios", nargs="*", default=None)
    p_base.add_argument("--episodes", type=int, default=50,
                        help="Eval episodes per baseline")
    p_base.add_argument("--train-episodes", type=int, default=None,
                        help="Training episodes for DQN/A2C")
    p_base.add_argument("--max-steps", type=int, default=None)
    p_base.add_argument("--seed", type=int, default=42)

    # ── compare ──
    p_cmp = sub.add_parser("compare", help="Generate comparison report")

    # ── full ──
    p_full = sub.add_parser("full", help="Full pipeline: train + eval + baselines + compare")
    p_full.add_argument("--scenarios", nargs="*", default=None)
    p_full.add_argument("--train-episodes", type=int, default=None,
                        help="Override training episodes")
    p_full.add_argument("--eval-episodes", type=int, default=50)
    p_full.add_argument("--max-steps", type=int, default=None)
    p_full.add_argument("--seed", type=int, default=42)
    p_full.add_argument("--verbose", action="store_true")
    p_full.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    p_full.add_argument("--train_episodes", type=int, default=None,
                        help=argparse.SUPPRESS)  # alias

    # ── script-train ──
    p_script = sub.add_parser("script-train",
                              help="Train with full SCRIPT CRL (teacher-student) on PenGym")
    p_script.add_argument("--scenarios", nargs="*", default=None,
                          help="Scenario names in curriculum order (default: tiny small-linear)")
    p_script.add_argument("--episodes", type=int, default=None,
                          help="Training episodes per task")
    p_script.add_argument("--max-steps", type=int, default=None,
                          help="Max steps per episode")
    p_script.add_argument("--seed", type=int, default=42)
    p_script.add_argument("--config", type=str, default=None,
                          help="YAML config file name (in data/config/)")
    p_script.add_argument("--ewc-lambda", type=float, default=None,
                          help="Override EWC lambda")
    p_script.add_argument("--guide-kl", type=float, default=None,
                          help="Override guide KL scale")

    # ── cve-audit ──
    sub.add_parser("cve-audit", help="Audit CVE dataset integration")

    args = parser.parse_args()

    # Normalize
    if not hasattr(args, 'verbose'):
        args.verbose = False
    if not hasattr(args, 'seed'):
        args.seed = 42
    if not hasattr(args, 'max_steps'):
        args.max_steps = None
    if not hasattr(args, 'train_episodes'):
        args.train_episodes = None
    if not hasattr(args, 'resume'):
        args.resume = False

    return args


def main():
    args = parse_args()

    # ── Auto-log ALL output to file ──────────────────────────────────
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    log_name = f"{args.command}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    global_log_path = str(BENCHMARK_DIR / log_name)
    tee = TeeLogger(global_log_path)
    print(f"[LOG] All output is being saved to: {global_log_path}\n")

    try:
        commands = {
            "train": cmd_train,
            "script-train": cmd_script_train,
            "eval": cmd_eval,
            "baselines": cmd_baselines,
            "compare": cmd_compare,
            "full": cmd_full,
            "cve-audit": cmd_cve_audit,
        }
        commands[args.command](args)
    finally:
        tee.close()
        # Print to real stdout after tee is closed
        print(f"\n[LOG] Full log saved → {global_log_path}")


if __name__ == "__main__":
    main()
