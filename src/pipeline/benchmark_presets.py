"""
Benchmark presets for Strategy C experiments.

Centralises scenario metadata (optimal rewards/steps, training budgets)
and provides pre-defined bundles so experiments are reproducible and
correctly parameterised without manual guesswork.

Three preset tiers::

    quick    — tiny only                    (~2 min,  1 PenGym task)
    standard — tiny + tiny-hard + tiny-small (~15 min, 3 PenGym tasks)
    full     — all 5 core scenarios          (~60 min, 5 PenGym tasks)
    medium   — add medium-class scenarios    (~3 hr,  8 PenGym tasks)

Usage::

    from src.pipeline.benchmark_presets import get_preset, SCENARIO_META

    preset = get_preset("standard")
    scenarios   = preset["pengym_scenarios"]   # list of YAML paths
    step_limit  = preset["step_limit"]         # per-scenario max steps
    ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

# =====================================================================
# Scenario metadata — single source of truth
# =====================================================================

SCENARIO_META: Dict[str, Dict[str, Any]] = {
    # ── Tiny family (3 hosts) ─────────────────────────────────────
    "tiny": {
        "path": "data/scenarios/tiny.yml",
        "hosts": 3,
        "subnets": 3,
        "exploits": 1,
        "optimal_reward": 195.0,    # 200 - 5 action costs
        "optimal_steps": 6,
        "difficulty": 1,            # 1=easiest … 5=hardest
        "train_eps": 500,
        "step_limit": 100,
        "description": "Baseline — 1 exploit (ssh), 1 priv-esc, star topology",
    },
    "tiny-hard": {
        "path": "data/scenarios/tiny-hard.yml",
        "hosts": 3,
        "subnets": 3,
        "exploits": 3,
        "optimal_reward": 192.0,
        "optimal_steps": 5,
        "difficulty": 2,
        "train_eps": 800,
        "step_limit": 150,
        "description": "3 exploits, tight firewalls, must pick correct exploit per host",
    },
    "tiny-small": {
        "path": "data/scenarios/tiny-small.yml",
        "hosts": 5,
        "subnets": 4,
        "exploits": 3,
        "optimal_reward": 189.0,
        "optimal_steps": 7,
        "difficulty": 2,
        "train_eps": 800,
        "step_limit": 150,
        "description": "Bridge scenario — 4 subnets, 2 subnet scans required",
    },

    # ── Small family (8 hosts) ────────────────────────────────────
    "small-linear": {
        "path": "data/scenarios/small-linear.yml",
        "hosts": 8,
        "subnets": 6,
        "exploits": 3,
        "optimal_reward": 179.0,
        "optimal_steps": 10,
        "difficulty": 3,
        "train_eps": 1500,
        "step_limit": 200,
        "description": "Linear/ring topology, 2 internet entry points, 2 attack chains",
    },
    "small-honeypot": {
        "path": "data/scenarios/small-honeypot.yml",
        "hosts": 8,
        "subnets": 4,
        "exploits": 3,
        "optimal_reward": 186.0,
        "optimal_steps": 8,
        "difficulty": 3,
        "train_eps": 1500,
        "step_limit": 200,
        "description": "Honeypot trap (−100), agent must discriminate identical-looking hosts",
    },

    # ── Medium family (16 hosts) ──────────────────────────────────
    "medium": {
        "path": "data/scenarios/medium.yml",
        "hosts": 16,
        "subnets": 5,
        "exploits": 5,
        "optimal_reward": 185.0,
        "optimal_steps": 8,
        "difficulty": 4,
        "train_eps": 2000,
        "step_limit": 300,
        "description": "Standard medium — 5 exploits, 3 priv-esc, multi-hop pivoting",
    },
    "medium-single-site": {
        "path": "data/scenarios/medium-single-site.yml",
        "hosts": 16,
        "subnets": 1,
        "exploits": 5,
        "optimal_reward": 191.0,
        "optimal_steps": 4,
        "difficulty": 3,
        "train_eps": 1500,
        "step_limit": 300,
        "description": "Flat 16-host subnet — exploit selection in large network",
    },
    "medium-multi-site": {
        "path": "data/scenarios/medium-multi-site.yml",
        "hosts": 16,
        "subnets": 6,
        "exploits": 5,
        "optimal_reward": 187.0,
        "optimal_steps": 7,
        "difficulty": 5,
        "train_eps": 2500,
        "step_limit": 400,
        "description": "Enterprise WAN — 3 remote sites, complex firewall rules",
    },
}

# =====================================================================
# Default simulation scenarios (Phase 1)
# =====================================================================

DEFAULT_SIM_SCENARIOS = [
    "data/scenarios/chain/chain-msfexp_vul-sample-6_envs-seed_0.json",
]

# =====================================================================
# Preset bundles
# =====================================================================

_PRESETS: Dict[str, Dict[str, Any]] = {
    "quick": {
        "description": "Fast smoke test — tiny only (~2 min)",
        "scenarios": ["tiny"],
        "train_eps": 500,
        "step_limit": 100,
        "eval_freq": 10,
    },
    "standard": {
        "description": "Core benchmark — 3 tiny-family scenarios (~15 min)",
        "scenarios": ["tiny", "tiny-hard", "tiny-small"],
        "train_eps": 800,
        "step_limit": 150,
        "eval_freq": 5,
    },
    "full": {
        "description": "Full benchmark — all 5 ≤8-host scenarios (~60 min)",
        "scenarios": ["tiny", "tiny-hard", "tiny-small",
                       "small-linear", "small-honeypot"],
        "train_eps": 1500,
        "step_limit": 200,
        "eval_freq": 5,
    },
    "medium": {
        "description": "Extended — all 8 scenarios incl. 16-host medium (~3 hr)",
        "scenarios": ["tiny", "tiny-hard", "tiny-small",
                       "small-linear", "small-honeypot",
                       "medium", "medium-single-site", "medium-multi-site"],
        "train_eps": 2000,
        "step_limit": 300,
        "eval_freq": 10,
    },
}


# =====================================================================
# Public API
# =====================================================================

def get_preset(name: str) -> Dict[str, Any]:
    """Return a fully resolved preset dict.

    Returns
    -------
    dict with keys:
        - ``name``              : preset name
        - ``description``       : human-readable description
        - ``pengym_scenarios``  : list of YAML paths
        - ``sim_scenarios``     : list of JSON paths (Phase 1)
        - ``train_eps``         : default training episodes
        - ``step_limit``        : default step limit per episode
        - ``eval_freq``         : evaluation frequency
        - ``optimal_rewards``   : dict scenario_name → optimal reward
        - ``optimal_steps``     : dict scenario_name → optimal steps
        - ``per_scenario``      : dict scenario_name → full SCENARIO_META entry
        - ``scenario_names``    : ordered list of scenario names

    Raises
    ------
    ValueError
        If preset name is not recognised.
    """
    if name not in _PRESETS:
        valid = ", ".join(sorted(_PRESETS.keys()))
        raise ValueError(
            f"Unknown preset: {name!r}. Choose from: {valid}"
        )

    preset = _PRESETS[name]
    scenario_names = preset["scenarios"]

    pengym_paths = []
    optimal_rewards = {}
    optimal_steps = {}
    per_scenario = {}

    for sc_name in scenario_names:
        meta = SCENARIO_META[sc_name]
        pengym_paths.append(meta["path"])
        optimal_rewards[sc_name] = meta["optimal_reward"]
        optimal_steps[sc_name] = meta["optimal_steps"]
        per_scenario[sc_name] = meta

    return {
        "name": name,
        "description": preset["description"],
        "pengym_scenarios": pengym_paths,
        "sim_scenarios": list(DEFAULT_SIM_SCENARIOS),
        "train_eps": preset["train_eps"],
        "step_limit": preset["step_limit"],
        "eval_freq": preset["eval_freq"],
        "optimal_rewards": optimal_rewards,
        "optimal_steps": optimal_steps,
        "per_scenario": per_scenario,
        "scenario_names": list(scenario_names),
    }


def list_presets() -> str:
    """Return a formatted string listing all available presets."""
    lines = ["Available benchmark presets:", ""]
    for name, preset in _PRESETS.items():
        sc_names = preset["scenarios"]
        lines.append(
            f"  {name:<12} — {preset['description']}"
        )
        lines.append(
            f"  {'':12}   scenarios: {', '.join(sc_names)}"
        )
        lines.append(
            f"  {'':12}   train_eps={preset['train_eps']}, "
            f"step_limit={preset['step_limit']}"
        )
        lines.append("")
    return "\n".join(lines)


def list_scenarios() -> str:
    """Return a formatted table of all known scenarios."""
    lines = [
        f"{'Name':<22} {'Hosts':>5} {'Sub':>3} {'Exp':>3} "
        f"{'Opt.R':>6} {'Opt.S':>5} {'Diff':>4} {'Eps':>5} {'StepL':>5}",
        "-" * 72,
    ]
    for name, m in SCENARIO_META.items():
        lines.append(
            f"{name:<22} {m['hosts']:>5} {m['subnets']:>3} "
            f"{m['exploits']:>3} {m['optimal_reward']:>6.0f} "
            f"{m['optimal_steps']:>5} {'⭐' * m['difficulty']:>4} "
            f"{m['train_eps']:>5} {m['step_limit']:>5}"
        )
    return "\n".join(lines)
