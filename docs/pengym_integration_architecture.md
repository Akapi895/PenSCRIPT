# PenGym Integration — Technical Recovery & Architecture Stabilization

> **Version:** 1.0  
> **Date:** 2025-07  
> **Branch:** `adapter`  
> **Status:** DESIGN — Ready for Implementation  
> **Scope:** Problems 4 (SingleHostPenGymWrapper) + 1 (SCRIPT↔PenGym config integration)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Architecture Analysis](#2-current-architecture-analysis)
3. [Root Cause Analysis](#3-root-cause-analysis)
4. [Refactored Architecture Design](#4-refactored-architecture-design)
5. [Component Specifications](#5-component-specifications)
6. [SCRIPT Agent Adaptations](#6-script-agent-adaptations)
7. [Refactoring Roadmap](#7-refactoring-roadmap)
8. [Migration Plan](#8-migration-plan)
9. [Testing & Validation Plan](#9-testing--validation-plan)
10. [Before / After Comparison](#10-before--after-comparison)
11. [Definition of Done](#11-definition-of-done)

---

## 1. Executive Summary

### Problem Statement

Phases 1–4 of the CVE Difficulty & Expansion Pipeline are complete: 1,985 CVEs graded across 4 tiers, 80 NASim-valid scenarios compiled, a CurriculumController verified, and an extensible ServiceRegistry built. However, **none of this infrastructure can be used** because the bridge between SCRIPT's single-host RL agent and PenGym's multi-host network environment is incomplete, inconsistent, and partially broken.

### Specific Failures

| ID  | Failure                                                                      | Impact                                                                      |
| --- | ---------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| F1  | `PenGymStateAdapter.get_host_data()` does not exist                          | `run_eval_service_level.py` crashes with `AttributeError`                   |
| F2  | `ServiceActionMapper` not exported from `adapters/__init__.py`               | Import inconsistency, not discoverable                                      |
| F3  | `SimToRealEvaluator` uses CVE-level `ActionMapper` (3.4% coverage)           | Strategy A evaluation for service-level models is wrong                     |
| F4  | No `SingleHostPenGymWrapper` exists                                          | SCRIPT's per-host paradigm cannot interface with PenGym's network-level env |
| F5  | `run.py` line ~461: "PenGym training not yet implemented"                    | No PenGym training loop exists at all                                       |
| F6  | `CurriculumController` is not integrated into any training loop              | Phase 3 infrastructure is isolated                                          |
| F7  | Reward scales differ by ~100× (SCRIPT: ±1000, PenGym: scenario-defined ~±10) | Transferred policies have wrong value estimates                             |
| F8  | Two evaluation scripts with conflicting mapper usage                         | No single canonical evaluation path                                         |

### Goal

Design a **sustainable, extensible architecture** that:

1. Wraps PenGym's multi-host environment into SCRIPT's single-host-per-step interface
2. Standardizes state conversion and action mapping through a single, correct pipeline
3. Creates a PenGym-native training loop that integrates with CurriculumController
4. Preserves SCRIPT's core: PPO policy (1538×512×512×16), decision logic, multi-scenario capability

---

## 2. Current Architecture Analysis

### 2.1 Dependency Graph (Before)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ENTRY POINTS                                 │
├──────────────┬──────────────────┬───────────────────────────────────┤
│ run_strategy │ run_eval_service │ run_train_service_level.py        │
│ _a.py        │ _level.py        │ (SCRIPT sim only)                 │
└──────┬───────┴────────┬─────────┴────────┬──────────────────────────┘
       │                │                   │
       ▼                ▼                   ▼
┌──────────────┐ ┌──────────────┐ ┌────────────────────┐
│ SimToReal    │ │ ServiceLevel │ │ ServiceLevelAgent  │
│ Evaluator    │ │ Evaluator    │ │ (train_service)    │
│              │ │              │ │                    │
│ Uses:        │ │ Uses:        │ │ Uses:              │
│ ActionMapper │ │ ServiceAction│ │ HOST objects       │
│ (CVE-level!) │ │ Mapper       │ │ ServiceActionSpace │
│ 3.4% ✗      │ │ 100% ✓       │ │ CVESelector        │
└──────┬───────┘ └──────┬───────┘ └────────┬───────────┘
       │                │                  │
       ▼                ▼                  │ (no PenGym
┌──────────────┐ ┌──────────────┐          │  connection)
│ PenGymState  │ │ PenGymState  │          │
│ Adapter      │ │ Adapter      │          │
│              │ │              │          │
│ Has:         │ │ Calls:       │          │
│ convert()   ✓│ │ get_host_    │          │
│ convert_all()│ │ data() ✗     │          │
│ get_reacha..│ │ CRASHES!      │           │
│ get_sensit..│ │               │           │
│ _get_host_  │ │               │           │
│  segment() ✓│ │               │           │
└──────┬───────┘ └──────┬───────┘           │
       │                │                   │
       ▼                ▼                   ▼
┌──────────────────────────────────────────────────────────┐
│                     PenGymEnv (NASimEnv)                 │
│ reset() → (obs, info)                                    │
│ step(flat_action_int) → (obs, reward, done, trunc, info) │
│ Obs: (num_hosts+1) * host_vec_size  flat array           │
│ Action: 0..N-1 flat index                                │
└──────────────────────────────────────────────────────────┘
```

### 2.2 File Inventory

| File                                         | Role                                | Status                                   |
| -------------------------------------------- | ----------------------------------- | ---------------------------------------- |
| `src/envs/adapters/__init__.py`              | Package exports                     | **BUG:** Missing `ServiceActionMapper`   |
| `src/envs/adapters/state_adapter.py`         | PenGym obs → 1538-dim SCRIPT state  | **BUG:** Missing `get_host_data()`       |
| `src/envs/adapters/action_mapper.py`         | CVE-level action mapping (3.4%)     | Working, but **wrong** for service-level |
| `src/envs/adapters/service_action_mapper.py` | Service-level action mapping (100%) | Working, **not exported**                |
| `src/evaluation/sim_to_real_eval.py`         | Strategy A core evaluation loop     | Uses **wrong** mapper                    |
| `run_eval_service_level.py`                  | Service-level evaluation entry      | **CRASHES** (missing method)             |
| `run_train_service_level.py`                 | Service-level training              | **SCRIPT-sim only**, no PenGym           |
| `run_strategy_a.py`                          | Strategy A entry point              | Uses **wrong** evaluator path            |
| `src/envs/core/environment.py`               | `PenGymEnv` extends `NASimEnv`      | Working                                  |
| `src/agent/host.py`                          | `HOST` single-target simulation     | Working                                  |
| `src/agent/actions/service_action_space.py`  | 16-dim `ServiceActionSpace`         | Working                                  |
| `src/pipeline/curriculum_controller.py`      | Curriculum tier management          | Working, **not integrated**              |

### 2.3 Data Flow Contracts (Current)

**SCRIPT Simulation Path** (working):

```
Scenario JSON → HOST(ip, env_data) → HOST.reset() → state_vector (1538-dim)
→ PPO.select_action(state) → service_action_idx (0..15)
→ CVESelector.select(service_action_idx, host_info) → cve_action
→ HOST.step(cve_action.original_idx) → (next_state, reward, done, result)
```

**PenGym Evaluation Path** (broken):

```
Scenario YAML → PenGymEnv(scenario) → env.reset() → flat_obs (big 1D array)
→ PenGymStateAdapter.convert(flat_obs, host_addr) → state (1538-dim)  ✓
→ PPO.select_action(state) → service_action_idx (0..15)               ✓
→ ServiceActionMapper.map_action(idx, host_addr) → pengym_flat_idx     ✓
→ env.step(pengym_flat_idx) → (obs, reward, done, trunc, info)         ✓
BUT: _select_target() calls get_host_data() which DOES NOT EXIST       ✗
```

**PenGym Training Path** (nonexistent):

```
(nothing — "PenGym training not yet implemented")
```

---

## 3. Root Cause Analysis

### 3.1 Architectural Root Cause: Paradigm Mismatch

SCRIPT was designed as a **single-host-at-a-time** agent:

- `HOST` object wraps one target IP
- `HOST.reset()` → observation of _that host only_
- `HOST.step(action)` → executes action on _that host only_
- Training iterates over a `target_list: List[HOST]`

PenGym is a **multi-host network** environment:

- `PenGymEnv.reset()` → observation of _entire network_
- `PenGymEnv.step(action)` → action specifies _which host and which action_
- Single environment instance manages the full network state

The adapter layer was built **bottom-up** (state conversion first, action mapping second) without a **top-level wrapper** that reconciles the paradigm difference. Each evaluation script independently reinvented target selection, state extraction, and loop control — leading to inconsistency.

### 3.2 Interface Contract Violations

| Caller                                      | Expected Interface                               | Actual Interface                 | Root Cause                                               |
| ------------------------------------------- | ------------------------------------------------ | -------------------------------- | -------------------------------------------------------- |
| `ServiceLevelEvaluator._select_target()`    | `state_adapter.get_host_data(obs, host) → dict`  | Method does not exist            | Coded against imagined API; never tested                 |
| `SimToRealEvaluator._setup_action_mapper()` | `ServiceActionMapper` (for service-level models) | `ActionMapper` (CVE-level, 3.4%) | Written before service-level abstraction existed         |
| `adapters/__init__.py`                      | Export all adapter classes                       | Missing `ServiceActionMapper`    | Added service_action_mapper.py without updating **init** |

### 3.3 Training-Evaluation Asymmetry

The training loop (`run_train_service_level.py`) operates entirely within SCRIPT simulation:

```python
# Training: iterates HOST objects
for target in target_list:
    target.reset()
    state = target.state_vector.encoded_state  # 1538-dim from HOST
    action = policy.select_action(state)         # 0..15
    cve = cve_selector.select(action, target)    # service → specific CVE
    next_state, reward, done, result = target.step(cve.original_idx)
```

The evaluation loop attempts to use PenGym, but the interface gap is not bridged:

```python
# Evaluation: PenGym env, no HOST objects
obs, _ = env.reset()                                     # whole network
state = state_adapter.convert(obs, target_host)           # 1538-dim from PenGym
action = policy.select_action(state)                      # 0..15
pengym_action = action_mapper.map_action(action, target)  # service → PenGym flat idx
obs, reward, done, trunc, info = env.step(pengym_action)  # whole network
```

**The core issue**: training never touches PenGym, so the agent never learns PenGym's state distribution, reward dynamics, or transition mechanics. The `SingleHostPenGymWrapper` is the missing component that would allow SCRIPT's per-host training loop to drive PenGym.

### 3.4 Reward Scale Mismatch

| Source                  | Exploit Success                          | Scan Info | Action Cost      | Scale Range     |
| ----------------------- | ---------------------------------------- | --------- | ---------------- | --------------- |
| SCRIPT (`HOST.step()`)  | +1000                                    | +100      | -10              | [-10, +1000]    |
| PenGym (NASim scenario) | Scenario-defined value (e.g., +100, +50) | 0         | -cost (e.g., -1) | [-cost, +value] |

The PPO critic trained on SCRIPT rewards will produce value estimates ~10–100× larger than PenGym rewards. Without normalization, the advantage function is distorted.

---

## 4. Refactored Architecture Design

### 4.1 Architecture Overview (After)

```
┌────────────────────────────────────────────────────────────────────────┐
│                      UNIFIED ENTRY POINTS                              │
├──────────────────────────────┬─────────────────────────────────────────┤
│ run_pengym_train.py          │ run_pengym_eval.py                      │
│ (PenGym training w/          │ (PenGym evaluation - unified)           │
│  CurriculumController)       │                                         │
└──────────┬───────────────────┴────────────┬────────────────────────────┘
           │                                │
           ▼                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                   SingleHostPenGymWrapper                                │
│                                                                          │
│ Wraps PenGymEnv into SCRIPT's single-host-per-step interface             │
│                                                                          │
│ Contract:                                                                │
│   set_target(host_addr)                                                  │
│   reset() → state_1538                                                   │
│   step(service_action_idx) → (next_state_1538, reward, done, info)       │
│   get_available_targets() → List[host_addr]                              │
│   get_host_info(host_addr) → HostInfo(reachable, compromised, ...)       │
│                                                                          │
│ Internal Components:                                                     │
│   ┌──────────────┐  ┌──────────────────┐  ┌──────────────────────────┐   │
│   │ PenGymState  │  │ ServiceAction    │  │ RewardNormalizer         │   │
│   │ Adapter      │  │ Mapper           │  │ (configurable strategy)  │   │
│   │ (1538-dim)   │  │ (16→pengym)      │  │                          │   │
│   └──────────────┘  └──────────────────┘  └──────────────────────────┘   │
└──────────┬───────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         PenGymEnv (NASimEnv)                             │
│ Unchanged — Gymnasium-compliant multi-host environment                   │
└──────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Component Interactions

```
┌─────────────────────┐     ┌─────────────────────────────┐
│ CurriculumController│────▶│ SingleHostPenGymWrapper    │
│                     │     │                             │
│ get_next_scenario()─┼────▶│ load_scenario(path)        │
│ record_episode()  ◀─┼─────│ (reports success/reward)   │
└─────────────────────┘     │                             │
                            │ .set_target(host_addr)      │
                            │ .reset() → 1538-dim state   │
┌─────────────────────┐     │ .step(action_idx) →         │
│ PPO Policy          │◀────│   (state, reward, done,info)│
│ (1538→512→512→16)   │────▶│                             │
│ Unchanged           │     └─────────────┬───────────────┘
└─────────────────────┘                   │
                                          │ Uses internally:
                            ┌─────────────▼───────────────┐
                            │ PenGymStateAdapter          │
                            │ ServiceActionMapper         │
                            │ RewardNormalizer            │
                            │ TargetSelector              │
                            └─────────────────────────────┘
```

### 4.3 Key Design Decisions

| Decision                                                                       | Rationale                                                                                                        |
| ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| **SingleHostPenGymWrapper as the SOLE integration point**                      | Eliminates duplication of target selection, state conversion, action mapping across different eval/train scripts |
| **ServiceActionMapper only (deprecate CVE-level ActionMapper for evaluation)** | 100% coverage vs 3.4%. CVE-level mapper remains available for research but is not the default path               |
| **RewardNormalizer inside wrapper**                                            | Agent receives normalized rewards regardless of scenario source; configurable strategy (linear, z-score, clip)   |
| **TargetSelector as pluggable strategy**                                       | Decouples host selection logic from wrapper; allows priority-based, round-robin, or learned selection            |
| **Wrapper does NOT alter PenGymEnv**                                           | PenGymEnv remains untouched; wrapper is purely an adapter layer on top                                           |
| **CurriculumController integration via scenario_path injection**               | Controller selects scenario → wrapper loads it → trains → controller records result                              |

---

## 5. Component Specifications

### 5.1 SingleHostPenGymWrapper

**Location:** `src/envs/wrappers/single_host_wrapper.py`

```python
class SingleHostPenGymWrapper:
    """Wraps PenGymEnv to expose SCRIPT's single-host-per-step interface.

    This is the SOLE bridge between SCRIPT's RL agent and PenGym's
    multi-host network environment. It handles:
      1. State conversion (PenGym obs → 1538-dim SCRIPT vector)
      2. Action mapping (service-level 0..15 → PenGym flat index)
      3. Target host tracking
      4. Reward normalization
      5. Episode management
    """
```

#### 5.1.1 Constructor

```python
def __init__(
    self,
    scenario_path: str,
    fully_obs: bool = True,
    flat_actions: bool = True,
    flat_obs: bool = True,
    reward_normalizer: Optional[RewardNormalizer] = None,
    target_selector: Optional[TargetSelector] = None,
    seed: int = 42,
    auto_select_target: bool = True,
):
    """
    Args:
        scenario_path: Path to NASim scenario YAML file.
        fully_obs: PenGym observability mode.
        flat_actions: Use flat action space (must be True for mapping).
        flat_obs: Use flat observation (must be True for state adapter).
        reward_normalizer: Strategy for reward scaling. If None, uses
                          LinearNormalizer(src_range=(-1, 100), dst_range=(-10, 1000)).
        target_selector: Strategy for choosing next target host. If None,
                        uses PrioritySensitiveSelector().
        seed: Random seed.
        auto_select_target: If True, automatically selects next target when
                           current target is compromised.
    """
```

**Postconditions:**

- `self.env: PenGymEnv` — initialized
- `self.state_adapter: PenGymStateAdapter` — initialized from scenario
- `self.action_mapper: ServiceActionMapper` — initialized from SAS + env
- `self.sas: ServiceActionSpace` — 16-dim (no Action class needed for PenGym-only)
- `self._current_target: Optional[Tuple[int,int]]` — None until `set_target()` or auto-select

#### 5.1.2 Core Interface

```python
@property
def state_dim(self) -> int:
    """Always 1538."""
    return PenGymStateAdapter.STATE_DIM

@property
def action_dim(self) -> int:
    """Always 16 (service-level)."""
    return self.sas.action_dim

@property
def current_target(self) -> Optional[Tuple[int, int]]:
    """Currently targeted host address, or None."""
    return self._current_target

def load_scenario(self, scenario_path: str) -> None:
    """Load a new scenario, resetting all internal state.

    Called by CurriculumController when switching scenarios.
    Re-creates PenGymEnv, StateAdapter, ActionMapper.

    Args:
        scenario_path: Path to NASim scenario YAML.

    Postconditions:
        - self.env is a fresh PenGymEnv
        - self.state_adapter matches new scenario
        - self.action_mapper matches new env
        - self._current_target is None (must call reset() next)
    """

def set_target(self, host_addr: Tuple[int, int]) -> None:
    """Set the current target host.

    Args:
        host_addr: (subnet_id, host_id) tuple.

    Raises:
        ValueError: if host_addr not in scenario's host map.
    """

def reset(self) -> np.ndarray:
    """Reset environment and return initial state for current target.

    If no target is set and auto_select_target is True, selects the
    first available target using the TargetSelector.

    Returns:
        1538-dim float32 state vector.

    Raises:
        RuntimeError: if no target is set and auto_select is False.
    """

def step(self, service_action_idx: int) -> Tuple[np.ndarray, float, bool, dict]:
    """Execute a service-level action on the current target in PenGym.

    Args:
        service_action_idx: Index 0..15 in service action space.

    Returns:
        next_state: 1538-dim float32 vector (of current target).
        reward: Normalized reward.
        done: True if all sensitive hosts are compromised (episode done).
        info: Dict with keys:
            'raw_reward': float — PenGym's original reward
            'pengym_action': int — flat action index used
            'target_host': Tuple[int,int]
            'action_name': str — e.g., 'exploit_ssh'
            'mapped': bool — True if action was directly mapped (not fallback)
            'target_compromised': bool — True if current target is now compromised
            'network_done': bool — True if all goals achieved
            'reachable_hosts': List[Tuple[int,int]]
            'compromised_hosts': List[Tuple[int,int]]

    Side Effects:
        If auto_select_target is True and current target becomes compromised,
        automatically advances to next uncompromised target.
    """

def get_available_targets(self) -> List[Tuple[int, int]]:
    """Return list of reachable, uncompromised host addresses.

    Uses current PenGym observation (cached from last step/reset).
    """

def get_host_info(self, host_addr: Tuple[int, int]) -> dict:
    """Get structured info about a host from current observation.

    Returns:
        dict with keys:
            'address': Tuple[int,int]
            'reachable': bool
            'compromised': bool
            'discovered': bool
            'access_level': float (0=none, 1=user, 2=root)
            'os': str (decoded OS name or '')
            'services': List[str] (active service names)
            'value': float (scenario-defined host value)

    This method replaces the missing get_host_data() and is the
    canonical way to inspect host state.
    """
```

#### 5.1.3 Auxiliary Interface

```python
def get_all_host_states(self) -> Dict[Tuple[int,int], np.ndarray]:
    """Return 1538-dim state vectors for ALL hosts."""

def get_sensitive_hosts(self) -> List[Tuple[int,int]]:
    """Delegate to state_adapter."""

def get_episode_stats(self) -> dict:
    """Return stats for current episode (steps, total_reward, compromised_hosts)."""

def describe(self) -> str:
    """Return human-readable description of wrapper config."""
```

### 5.2 PenGymStateAdapter — Additions

**File:** `src/envs/adapters/state_adapter.py`

Add the following method to `PenGymStateAdapter`:

```python
def get_host_data(self, flat_obs: np.ndarray,
                  host_addr: Tuple[int, int]) -> Optional[dict]:
    """Extract structured host information from PenGym observation.

    This is the method that run_eval_service_level.py expects.
    Built on top of _get_host_segment().

    Args:
        flat_obs: Flat NASim observation.
        host_addr: (subnet_id, host_id) tuple.

    Returns:
        dict with keys:
            'address': Tuple[int,int]
            'reachable': bool
            'compromised': bool
            'discovered': bool
            'access_level': float
            'value': float
            'os': str
            'services': List[str]
            'ports': List[str]
            'processes': List[str]
        or None if host not in map.
    """
    seg = self._get_host_segment(flat_obs, host_addr)
    if seg is None:
        return None

    # Decode binary flags
    os_flags = seg[self._os_offset:self._os_offset + len(self.os_names)]
    service_flags = seg[self._service_offset:self._service_offset + len(self.service_names)]
    process_flags = seg[self._process_offset:self._process_offset + len(self.process_names)]

    active_services = [self.service_names[i] for i in np.where(service_flags > 0.5)[0]
                       if i < len(self.service_names)]
    active_processes = [self.process_names[i] for i in np.where(process_flags > 0.5)[0]
                        if i < len(self.process_names)]

    # Infer ports from services
    ports = []
    for svc in active_services:
        if svc in self.service_port_map:
            ports.append(self.service_port_map[svc])

    return {
        'address': host_addr,
        'reachable': bool(seg[self._reachable_offset] > 0.5),
        'compromised': bool(seg[self._compromised_offset] > 0.5),
        'discovered': bool(seg[self._discovered_offset] > 0.5),
        'access_level': float(seg[self._access_offset]),
        'value': float(seg[self._value_offset]),
        'os': self._decode_os(os_flags),
        'services': active_services,
        'ports': ports,
        'processes': active_processes,
    }
```

### 5.3 RewardNormalizer

**Location:** `src/envs/wrappers/reward_normalizer.py`

```python
from abc import ABC, abstractmethod

class RewardNormalizer(ABC):
    """Abstract base for reward normalization strategies."""

    @abstractmethod
    def normalize(self, raw_reward: float) -> float:
        """Transform raw PenGym reward to SCRIPT-compatible scale."""
        ...

    @abstractmethod
    def describe(self) -> str:
        """Human-readable description."""
        ...


class LinearNormalizer(RewardNormalizer):
    """Linear scaling from PenGym range to SCRIPT range.

    Default: PenGym [-1, 100] → SCRIPT [-10, 1000]
    """

    def __init__(self,
                 src_min: float = -1.0, src_max: float = 100.0,
                 dst_min: float = -10.0, dst_max: float = 1000.0):
        self.src_min, self.src_max = src_min, src_max
        self.dst_min, self.dst_max = dst_min, dst_max

    def normalize(self, raw_reward: float) -> float:
        # Clamp to source range
        clamped = max(self.src_min, min(self.src_max, raw_reward))
        # Linear map
        ratio = (clamped - self.src_min) / max(self.src_max - self.src_min, 1e-8)
        return self.dst_min + ratio * (self.dst_max - self.dst_min)

    def describe(self) -> str:
        return (f"LinearNormalizer([{self.src_min}, {self.src_max}] → "
                f"[{self.dst_min}, {self.dst_max}])")


class ClipNormalizer(RewardNormalizer):
    """Clip-and-scale: divide by a constant, clip to [-1, 1] range.

    This is commonly used in RL transfer learning.
    """

    def __init__(self, scale: float = 100.0, clip: float = 10.0):
        self.scale = scale
        self.clip = clip

    def normalize(self, raw_reward: float) -> float:
        return max(-self.clip, min(self.clip, raw_reward / self.scale))

    def describe(self) -> str:
        return f"ClipNormalizer(scale={self.scale}, clip={self.clip})"


class IdentityNormalizer(RewardNormalizer):
    """No-op normalizer. Passes through raw reward."""

    def normalize(self, raw_reward: float) -> float:
        return raw_reward

    def describe(self) -> str:
        return "IdentityNormalizer()"
```

### 5.4 TargetSelector

**Location:** `src/envs/wrappers/target_selector.py`

```python
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict
import numpy as np


class TargetSelector(ABC):
    """Strategy for choosing the next target host in a multi-host network."""

    @abstractmethod
    def select(self,
               available: List[Tuple[int,int]],
               sensitive: List[Tuple[int,int]],
               host_info_fn) -> Optional[Tuple[int,int]]:
        """Select next target.

        Args:
            available: Reachable, uncompromised hosts.
            sensitive: Goal (high-value) hosts.
            host_info_fn: Callable(host_addr) → dict with host info.

        Returns:
            Selected host address, or None if no valid target.
        """
        ...


class PrioritySensitiveSelector(TargetSelector):
    """Prioritize sensitive hosts, then any reachable host.

    This replicates the logic from SimToRealEvaluator._select_target_host()
    and ServiceLevelEvaluator._select_target(), unified into one place.
    """

    def select(self, available, sensitive, host_info_fn):
        # 1. Uncompromised sensitive hosts first
        for host in sensitive:
            info = host_info_fn(host)
            if info and info['reachable'] and not info['compromised']:
                return host

        # 2. Any reachable uncompromised host
        for host in available:
            info = host_info_fn(host)
            if info and not info['compromised']:
                return host

        # 3. Fallback: first sensitive host regardless
        if sensitive:
            return sensitive[0]

        # 4. Any non-internet host
        if available:
            return available[0]

        return None


class RoundRobinSelector(TargetSelector):
    """Cycle through hosts in order. Useful for ensuring all hosts are visited."""

    def __init__(self):
        self._idx = 0

    def select(self, available, sensitive, host_info_fn):
        if not available:
            return None
        target = available[self._idx % len(available)]
        self._idx += 1
        return target


class ValuePrioritySelector(TargetSelector):
    """Select the host with highest scenario-defined value."""

    def select(self, available, sensitive, host_info_fn):
        if not available:
            return None
        best = None
        best_val = -1
        for host in available:
            info = host_info_fn(host)
            if info and not info['compromised'] and info.get('value', 0) > best_val:
                best = host
                best_val = info['value']
        return best if best else (available[0] if available else None)
```

### 5.5 Updated `adapters/__init__.py`

```python
"""PenGym Adapters — State conversion and action mapping."""

from .state_adapter import PenGymStateAdapter
from .action_mapper import ActionMapper
from .service_action_mapper import ServiceActionMapper

__all__ = [
    'PenGymStateAdapter',
    'ActionMapper',           # CVE-level (research/legacy)
    'ServiceActionMapper',    # Service-level (default for all new code)
]
```

### 5.6 PenGym Training Loop

**Location:** `src/training/pengym_trainer.py`

```python
class PenGymTrainer:
    """Train SCRIPT's PPO agent directly on PenGym via SingleHostPenGymWrapper.

    Supports:
      - Single-scenario training
      - Curriculum training via CurriculumController
      - Same PPO update logic as run_train_service_level.py
    """

    def __init__(
        self,
        policy,                    # PPO_agent instance (1538→512→512→16)
        wrapper: SingleHostPenGymWrapper,
        curriculum: Optional[CurriculumController] = None,
        state_norm = None,         # Normalization instance
        config: dict = None,
    ):
        self.policy = policy
        self.wrapper = wrapper
        self.curriculum = curriculum
        self.state_norm = state_norm
        self.config = config or {}

    def train(
        self,
        total_episodes: int = 1000,
        max_steps_per_episode: int = 100,
        eval_interval: int = 50,
        save_interval: int = 100,
        save_dir: str = 'outputs/models_pengym/',
    ):
        """Main training loop.

        If curriculum is provided:
            scenario_path = curriculum.get_next_scenario()
            wrapper.load_scenario(scenario_path)
            ... train episode ...
            curriculum.record_episode(success, reward, steps)

        If no curriculum:
            Train on wrapper's current scenario.
        """
```

#### Training Loop Pseudocode

```python
for episode in range(total_episodes):
    # Curriculum scenario selection
    if self.curriculum:
        scenario_path = self.curriculum.get_next_scenario()
        if self.curriculum.current_scenario_changed:
            self.wrapper.load_scenario(scenario_path)

    # Episode
    state = self.wrapper.reset()  # 1538-dim
    if self.state_norm:
        state = self.state_norm(state, update=True)

    ep_reward = 0.0
    ep_steps = 0
    done = False

    while not done and ep_steps < max_steps_per_episode:
        # Policy
        action, log_prob, value = self.policy.select_action(state)

        # Environment step
        next_state, reward, done, info = self.wrapper.step(action)
        if self.state_norm:
            next_state = self.state_norm(next_state, update=True)

        # Store transition
        self.policy.buffer.store(state, action, reward, next_state,
                                done, log_prob, value)

        state = next_state
        ep_reward += reward
        ep_steps += 1

        # Auto-target advancement is handled internally by wrapper

    # PPO update
    if self.policy.buffer.is_full():
        self.policy.update()

    # Report to curriculum
    if self.curriculum:
        self.curriculum.record_episode(
            success=done,
            reward=ep_reward,
            steps=ep_steps
        )

    # Logging, saving, evaluation ...
```

### 5.7 Unified Evaluation Entry

**Location:** `run_pengym_eval.py` (replaces both `run_eval_service_level.py` and `run_strategy_a.py`)

```python
class PenGymEvaluator:
    """Unified PenGym evaluation using SingleHostPenGymWrapper.

    Replaces:
      - run_eval_service_level.py (ServiceLevelEvaluator)
      - run_strategy_a.py (via SimToRealEvaluator)

    Both paths now use the identical wrapper → identical state conversion,
    action mapping, target selection, reward handling.
    """

    def __init__(self, model_dir, scenario_path, **kwargs):
        self.wrapper = SingleHostPenGymWrapper(
            scenario_path=scenario_path,
            reward_normalizer=IdentityNormalizer(),  # raw rewards for eval
            **kwargs,
        )
        self._load_model(model_dir)

    def evaluate(self, num_episodes=20, max_steps=100):
        """Run evaluation episodes."""
        results = []
        for ep in range(num_episodes):
            state = self.wrapper.reset()
            if self.state_norm:
                state = self.state_norm(state, update=False)

            ep_reward = 0.0
            done = False
            steps = 0

            while not done and steps < max_steps:
                action = self.policy.evaluate(state, determinate=True)
                next_state, reward, done, info = self.wrapper.step(action)
                if self.state_norm:
                    next_state = self.state_norm(next_state, update=False)

                state = next_state
                ep_reward += reward  # Raw rewards for eval metrics
                steps += 1

            results.append({
                'episode': ep, 'success': done,
                'reward': ep_reward, 'steps': steps,
            })

        return self._aggregate(results)
```

---

## 6. SCRIPT Agent Adaptations

### 6.1 What MUST Be Preserved

| Component               | File                         | Contract                                                                                        | Rationale                                             |
| ----------------------- | ---------------------------- | ----------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| PPO Actor network       | `src/agent/policy/PPO.py`    | MLP(1538→512→512→16, softmax)                                                                   | Architecture defines learned representations          |
| PPO Critic network      | `src/agent/policy/PPO.py`    | MLP(1538→512→512→1)                                                                             | Value function must match state dim                   |
| State dimension         | `StateEncoder`               | 1538-dim = [access(2) \| os_sbert(384) \| port_sbert(384) \| service_sbert(384) \| web_fp(384)] | Changing dims requires full retraining                |
| Action dimension        | `ServiceActionSpace`         | 16 = 4 scan + 9 exploit + 3 privesc                                                             | Mapped 1:1 to PenGym                                  |
| State normalization     | `Normalization` class        | Running mean/std, per-dimension                                                                 | Critical for policy stability                         |
| PPO hyperparameters     | `PPO_Config`                 | batch=512, mini_batch=64, gamma=0.99, clip=0.2, entropy=0.02, gae_lambda=0.95                   | Tuned for SCRIPT; keep unless evidence demands change |
| Multi-scenario training | `run_train_service_level.py` | Iterates `target_list`, multiple episodes per HOST                                              | wrapper mimics this with `set_target()` cycling       |

### 6.2 What MUST Change

| Aspect                              | Current                                           | Target                                            | Why                                        |
| ----------------------------------- | ------------------------------------------------- | ------------------------------------------------- | ------------------------------------------ |
| Training environment                | HOST objects (SCRIPT sim) only                    | **Also** SingleHostPenGymWrapper (PenGym)         | Enable curriculum on 80 compiled scenarios |
| State source during PenGym training | N/A (no PenGym training)                          | `PenGymStateAdapter.convert()` → 1538-dim         | Same representation, different source      |
| Action execution during PenGym      | N/A                                               | `ServiceActionMapper.map_action()` → `env.step()` | Service-level, 100% coverage               |
| Reward during PenGym training       | N/A                                               | `RewardNormalizer.normalize(raw_reward)`          | Bridge reward scale gap                    |
| Target host selection               | Manual in eval scripts (duplicated logic)         | `TargetSelector` strategy inside wrapper          | Single canonical implementation            |
| Model checkpoint format             | PPO-actor.pt, PPO-critic.pt, PPO-norm_mean/std.pt | **Unchanged** but add wrapper config to metadata  | Allow eval to reconstruct correct pipeline |

### 6.3 What CAN Change (optional, future)

| Aspect                                          | Notes                                                                            |
| ----------------------------------------------- | -------------------------------------------------------------------------------- |
| State normalization re-running on PenGym states | Mean/std will shift; may want to warm-start from SCRIPT stats                    |
| PPO hyperparameter tuning for PenGym            | Entropy, learning rate may want adjustment for smaller reward scale              |
| Curriculum thresholds                           | Current T1=0.70, T2=0.60, T3=0.50, T4=0.40; may need tuning per PenGym scenarios |

---

## 7. Refactoring Roadmap

### Phase R1: Fix Critical Bugs (Day 1–2)

**Objective:** Make existing code runnable without crashes.

| Step | File                                 | Action                                                     | Risk                                          |
| ---- | ------------------------------------ | ---------------------------------------------------------- | --------------------------------------------- |
| R1.1 | `src/envs/adapters/state_adapter.py` | Add `get_host_data()` method (§5.2)                        | LOW — additive only                           |
| R1.2 | `src/envs/adapters/__init__.py`      | Export `ServiceActionMapper` (§5.5)                        | LOW — additive                                |
| R1.3 | `run_eval_service_level.py`          | Verify `_select_target()` works with new `get_host_data()` | LOW — existing code, new dependency satisfied |

**Validation:** `run_eval_service_level.py` runs without `AttributeError`.

### Phase R2: Create Wrapper Infrastructure (Day 2–4)

**Objective:** Build `SingleHostPenGymWrapper` and its dependencies.

| Step | File                                       | Action         | Risk                      |
| ---- | ------------------------------------------ | -------------- | ------------------------- |
| R2.1 | `src/envs/wrappers/__init__.py`            | Create package | NONE                      |
| R2.2 | `src/envs/wrappers/reward_normalizer.py`   | Implement §5.3 | LOW — standalone          |
| R2.3 | `src/envs/wrappers/target_selector.py`     | Implement §5.4 | LOW — standalone          |
| R2.4 | `src/envs/wrappers/single_host_wrapper.py` | Implement §5.1 | MEDIUM — core integration |
| R2.5 | Unit tests for wrapper                     | See §9         | LOW                       |

**Validation:** `SingleHostPenGymWrapper` can reset/step on `tiny.yml` scenario.

### Phase R3: PenGym Training Loop (Day 4–6)

**Objective:** Create training loop using wrapper + CurriculumController.

| Step | File                                    | Action                            | Risk   |
| ---- | --------------------------------------- | --------------------------------- | ------ |
| R3.1 | `src/training/__init__.py`              | Create package                    | NONE   |
| R3.2 | `src/training/pengym_trainer.py`        | Implement §5.6                    | MEDIUM |
| R3.3 | `run_pengym_train.py`                   | Entry point script                | LOW    |
| R3.4 | Integration with `CurriculumController` | Wire scenario selection → wrapper | MEDIUM |

**Validation:** Train 50 episodes on tiny.yml, verify reward improves.

### Phase R4: Unified Evaluation (Day 6–7)

**Objective:** Single canonical evaluation path.

| Step | File                        | Action                                                            | Risk |
| ---- | --------------------------- | ----------------------------------------------------------------- | ---- |
| R4.1 | `run_pengym_eval.py`        | Implement §5.7                                                    | LOW  |
| R4.2 | `run_strategy_a.py`         | Add deprecation warning, redirect to `run_pengym_eval.py`         | LOW  |
| R4.3 | `run_eval_service_level.py` | Add deprecation warning, redirect                                 | LOW  |
| R4.4 | Update `run.py`             | Remove "not yet implemented" comment, add PenGym train/eval paths | LOW  |

**Validation:** `run_pengym_eval.py` produces same or better results than `run_eval_service_level.py`.

### Phase R5: End-to-End Curriculum Training (Day 7–10)

**Objective:** Full pipeline from compiled scenarios through curriculum to trained model.

| Step | Action                                                          | Risk   |
| ---- | --------------------------------------------------------------- | ------ |
| R5.1 | Run curriculum training on T1 scenarios (20 scenarios, easiest) | MEDIUM |
| R5.2 | Evaluate T1-trained model on held-out T1 scenarios              | LOW    |
| R5.3 | Progress through T2→T3→T4 if T1 succeeds                        | MEDIUM |
| R5.4 | Compare curriculum vs flat training on PenGym                   | LOW    |

**Validation:** Curriculum-trained model achieves ≥50% SR on T1 PenGym scenarios.

---

## 8. Migration Plan

### 8.1 Backward Compatibility Guarantees

| Guarantee                                | How                                                             |
| ---------------------------------------- | --------------------------------------------------------------- |
| Existing 80 compiled scenarios untouched | New code only reads YAML files, never writes                    |
| SCRIPT simulation training still works   | `run_train_service_level.py` unchanged; new wrapper is additive |
| Existing model checkpoints loadable      | Same PPO architecture (1538×512×512×16), same file format       |
| CVE-level ActionMapper still available   | Remains in `adapters/`, just not the default path               |
| `run_strategy_a.py` still runnable       | Deprecation warning only; existing code path preserved          |
| `PenGymEnv` not modified                 | All changes are in adapter/wrapper layer                        |

### 8.2 New File Structure

```
src/
├── envs/
│   ├── adapters/
│   │   ├── __init__.py              ← MODIFIED (add ServiceActionMapper export)
│   │   ├── state_adapter.py         ← MODIFIED (add get_host_data())
│   │   ├── action_mapper.py         ← UNCHANGED
│   │   └── service_action_mapper.py ← UNCHANGED
│   ├── wrappers/                    ← NEW PACKAGE
│   │   ├── __init__.py
│   │   ├── single_host_wrapper.py   ← NEW (core component)
│   │   ├── reward_normalizer.py     ← NEW
│   │   └── target_selector.py       ← NEW
│   └── core/
│       └── environment.py           ← UNCHANGED
├── training/                        ← NEW PACKAGE
│   ├── __init__.py
│   └── pengym_trainer.py            ← NEW
├── pipeline/
│   └── curriculum_controller.py     ← UNCHANGED (consumed by PenGymTrainer)
└── agent/
    ├── host.py                      ← UNCHANGED
    ├── agent.py                     ← UNCHANGED
    └── actions/
        └── service_action_space.py  ← UNCHANGED

# Root-level scripts
run_pengym_train.py                  ← NEW (PenGym curriculum training)
run_pengym_eval.py                   ← NEW (unified PenGym evaluation)
run_train_service_level.py           ← UNCHANGED (SCRIPT sim training)
run_eval_service_level.py            ← DEPRECATED (redirect to run_pengym_eval.py)
run_strategy_a.py                    ← DEPRECATED (redirect to run_pengym_eval.py)
```

### 8.3 Migration Sequence

```
Step 1: R1 (bug fixes) → commit "fix: add get_host_data, export ServiceActionMapper"
Step 2: R2 (wrapper)   → commit "feat: SingleHostPenGymWrapper with reward/target strategies"
Step 3: R3 (training)  → commit "feat: PenGym training loop with CurriculumController"
Step 4: R4 (eval)      → commit "feat: unified PenGym evaluation, deprecate old scripts"
Step 5: R5 (e2e test)  → commit "test: curriculum training validation"
```

Each step is independently testable and committable. No step breaks existing functionality.

---

## 9. Testing & Validation Plan

### 9.1 Unit Tests

| Test                          | File                                | Validates                                             |
| ----------------------------- | ----------------------------------- | ----------------------------------------------------- |
| `test_get_host_data`          | `tests/test_state_adapter.py`       | `get_host_data()` returns correct dict for known obs  |
| `test_get_host_data_none`     | same                                | Returns None for invalid host address                 |
| `test_reward_normalizers`     | `tests/test_reward_normalizer.py`   | Linear, Clip, Identity produce correct outputs        |
| `test_reward_normalizer_edge` | same                                | Zero-range source, negative values, extreme inputs    |
| `test_target_selectors`       | `tests/test_target_selector.py`     | Each selector picks correct host given mock host_info |
| `test_selector_no_targets`    | same                                | Returns None when no valid targets                    |
| `test_wrapper_init`           | `tests/test_single_host_wrapper.py` | Constructor creates env, adapter, mapper correctly    |
| `test_wrapper_state_dim`      | same                                | `wrapper.state_dim == 1538`                           |
| `test_wrapper_action_dim`     | same                                | `wrapper.action_dim == 16`                            |
| `test_wrapper_load_scenario`  | same                                | Re-loading scenario resets internal state             |

### 9.2 Integration Tests

| Test                                | Validates                                                                                                          |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `test_wrapper_reset`                | `wrapper.reset()` returns 1538-dim vector, no crash                                                                |
| `test_wrapper_step`                 | `wrapper.step(0)` returns (state, reward, done, info) with correct types                                           |
| `test_wrapper_full_episode`         | Run 50 random-policy steps, verify no crash, done eventually reachable                                             |
| `test_wrapper_auto_target`          | Target auto-advances after compromise                                                                              |
| `test_wrapper_reward_normalization` | Linear normalizer transforms raw reward correctly                                                                  |
| `test_adapter_exports`              | `from src.envs.adapters import ServiceActionMapper` works                                                          |
| `test_pengym_trainer_one_episode`   | PenGymTrainer.train(total_episodes=1) completes without crash                                                      |
| `test_curriculum_integration`       | CurriculumController.get_next_scenario() → wrapper.load_scenario() → train 1 episode → curriculum.record_episode() |

### 9.3 Smoke Tests

| Test                | Command                                                                                                    | Expected                                                             |
| ------------------- | ---------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| Old eval still runs | `python run_eval_service_level.py --scenario tiny.yml --model-dir <dir>`                                   | Deprecation warning + results (or "no model found" if no checkpoint) |
| New eval runs       | `python run_pengym_eval.py --scenario tiny.yml --model-dir <dir>`                                          | Results output                                                       |
| New train runs      | `python run_pengym_train.py --scenario tiny.yml --episodes 10`                                             | Training log with reward progression                                 |
| Curriculum train    | `python run_pengym_train.py --curriculum --scenario-dir data/scenarios/generated/compiled/ --episodes 100` | Phase transitions logged                                             |

### 9.4 End-to-End Validation

| Metric                                                   | Threshold                        | How to Measure                                        |
| -------------------------------------------------------- | -------------------------------- | ----------------------------------------------------- |
| Wrapper reset/step no crash                              | 100%                             | Run 1000 random-policy steps on 5 different scenarios |
| State dimension invariant                                | `state.shape == (1538,)` always  | Assert in every test                                  |
| Action mapping coverage                                  | ≥95% (no fallback to random)     | Check `info['mapped']` across episodes                |
| Curriculum phase transition                              | T1→T2 occurs within 300 episodes | CurriculumController log                              |
| Training reward trend                                    | Positive slope over 200 episodes | Plot episode reward curve                             |
| Eval SR on tiny.yml (random policy via wrapper)          | ≥5% in 100 episodes              | Baseline for correctness (not performance)            |
| Model trained via PenGymTrainer loads in PenGymEvaluator | Yes                              | Save checkpoint, load in eval, run 1 episode          |

---

## 10. Before / After Comparison

### 10.1 Evaluation Pipeline

| Aspect           | Before                                                   | After                                      |
| ---------------- | -------------------------------------------------------- | ------------------------------------------ |
| Entry points     | 3 scripts with different logic                           | 1 unified `run_pengym_eval.py`             |
| Action mapper    | CVE-level (3.4%) or Service-level (100%), inconsistently | Service-level (100%) always                |
| Target selection | Duplicated in 3 places, each different                   | Single `TargetSelector` strategy           |
| State conversion | Correct but `get_host_data()` missing                    | `get_host_data()` added + wrapper-internal |
| Reward handling  | Raw PenGym rewards (scale mismatch ignored)              | Configurable `RewardNormalizer`            |
| Crash risk       | `AttributeError` on `get_host_data()`                    | Eliminated                                 |

### 10.2 Training Pipeline

| Aspect                 | Before                                   | After                                           |
| ---------------------- | ---------------------------------------- | ----------------------------------------------- |
| PenGym training        | "Not yet implemented"                    | `PenGymTrainer` with full episode loop          |
| Curriculum integration | `CurriculumController` exists but unused | Integrated: scenario selection → train → record |
| Scenario utilization   | 80 compiled scenarios unused             | Loaded by wrapper via curriculum controller     |
| Reward normalization   | N/A                                      | Configurable normalizer in wrapper              |
| Target management      | N/A                                      | Auto-select with pluggable strategy             |

### 10.3 Architecture

| Aspect                  | Before                                           | After                                                      |
| ----------------------- | ------------------------------------------------ | ---------------------------------------------------------- |
| Integration point count | 0 (no wrapper)                                   | 1 (`SingleHostPenGymWrapper`)                              |
| Adapter exports         | `PenGymStateAdapter`, `ActionMapper`             | + `ServiceActionMapper`                                    |
| Package structure       | Flat scripts + adapters                          | `wrappers/`, `training/` packages added                    |
| Interface contracts     | Implicit, violated                               | Explicit typing, documented contracts                      |
| New-CVE onboarding path | ServiceRegistry exists but training loop missing | Registry → scenario compile → curriculum → PenGym training |

### 10.4 Code Paths Eliminated

| Old Code Path                                                               | Replacement                                                 |
| --------------------------------------------------------------------------- | ----------------------------------------------------------- |
| `SimToRealEvaluator._setup_action_mapper()` creating CVE-level mapper       | `SingleHostPenGymWrapper` always uses `ServiceActionMapper` |
| `ServiceLevelEvaluator._select_target()` with broken `get_host_data()` call | `TargetSelector.select()` with working `get_host_data()`    |
| `run_strategy_a.py` → `SimToRealEvaluator` → CVE-level path                 | `run_pengym_eval.py` → `PenGymEvaluator` → wrapper          |
| Manual env/adapter/mapper setup in each script                              | All encapsulated in `SingleHostPenGymWrapper` constructor   |

---

## 11. Definition of Done

### 11.1 Hard Requirements (must all be TRUE)

- [ ] **R1-DONE:** `PenGymStateAdapter.get_host_data()` exists and returns correct dict
- [ ] **R1-DONE:** `from src.envs.adapters import ServiceActionMapper` works
- [ ] **R2-DONE:** `SingleHostPenGymWrapper` can `reset()` and `step()` on `tiny.yml`
- [ ] **R2-DONE:** `wrapper.state_dim == 1538` and `wrapper.action_dim == 16`
- [ ] **R2-DONE:** `RewardNormalizer` produces correct output for all three strategies
- [ ] **R2-DONE:** `TargetSelector` implementations select correct hosts
- [ ] **R3-DONE:** `PenGymTrainer` completes 50 episodes on `tiny.yml` without crash
- [ ] **R3-DONE:** `CurriculumController` integration: phase transitions occur
- [ ] **R4-DONE:** `run_pengym_eval.py` produces SR metrics on `tiny.yml`
- [ ] **R4-DONE:** Old scripts (`run_eval_service_level.py`, `run_strategy_a.py`) still runnable (deprecated)
- [ ] **R5-DONE:** Curriculum training through T1 scenarios shows non-zero SR

### 11.2 Soft Requirements (should be TRUE)

- [ ] All unit tests pass
- [ ] Training reward curve shows positive trend within 200 episodes
- [ ] Model checkpoint saved by `PenGymTrainer` loads correctly in `PenGymEvaluator`
- [ ] No new `TODO` or `FIXME` added without associated issue

### 11.3 Non-Requirements (explicitly deferred)

| Deferred Item                                           | Reason                                    | When                              |
| ------------------------------------------------------- | ----------------------------------------- | --------------------------------- |
| PPO hyperparameter tuning for PenGym                    | Requires training data first              | After R5 validates basic training |
| Multi-scenario joint training (mix SCRIPT sim + PenGym) | Complex; PenGym-only first                | After PenGym training is stable   |
| Partially observable mode support                       | SCRIPT always uses fully_obs=True         | When needed for specific research |
| Real PenGym with nmap/metasploit                        | Requires cyber range infrastructure       | Separate project phase            |
| Web fingerprint gap resolution                          | PenGym has no web_fp data; zeros accepted | When PenGym adds web_fp support   |

---

## Appendix A: Interface Quick Reference

### SingleHostPenGymWrapper

```
Constructor(scenario_path, fully_obs=True, reward_normalizer=None, target_selector=None, seed=42)
    .load_scenario(path) → None
    .set_target(host_addr) → None
    .reset() → np.ndarray[1538]
    .step(action_idx: int) → (np.ndarray[1538], float, bool, dict)
    .get_available_targets() → List[Tuple[int,int]]
    .get_host_info(host_addr) → dict
    .get_all_host_states() → Dict[Tuple[int,int], np.ndarray[1538]]
    .get_sensitive_hosts() → List[Tuple[int,int]]
    .state_dim → 1538
    .action_dim → 16
    .current_target → Optional[Tuple[int,int]]
```

### PenGymStateAdapter (updated)

```
Constructor(scenario, encoder=None, service_port_map=None)
    .convert(flat_obs, host_addr) → np.ndarray[1538]
    .convert_all_hosts(flat_obs) → Dict[Tuple[int,int], np.ndarray[1538]]
    .get_host_data(flat_obs, host_addr) → Optional[dict]          ← NEW
    .get_sensitive_hosts() → List[Tuple[int,int]]
    .get_reachable_hosts(flat_obs) → List[Tuple[int,int]]
    ._get_host_segment(flat_obs, host_addr) → Optional[np.ndarray]
```

### ServiceActionMapper

```
Constructor(sas: ServiceActionSpace, env: PenGymEnv)
    .map_action(service_action_idx, target_host) → int  (PenGym flat idx, or -1)
    .get_random_valid_action(target_host) → int
    .get_all_actions_for_host(target_host) → List[int]
    .get_mapping_stats() → dict
```

### CurriculumController

```
Constructor(config: CurriculumConfig, scenario_dir, log_dir=None)
    .get_next_scenario() → str              (path to YAML)
    .record_episode(success, reward, steps) → None
    .is_complete() → bool
    .current_tier → int
    .phase_history → List[dict]
    .get_summary() → dict
```

### RewardNormalizer

```
LinearNormalizer(src_min, src_max, dst_min, dst_max)
    .normalize(raw_reward) → float
ClipNormalizer(scale, clip)
    .normalize(raw_reward) → float
IdentityNormalizer()
    .normalize(raw_reward) → float
```

---

## Appendix B: PenGym Observation Layout Reference

For a scenario with `num_subnets=S`, `max_hosts_per_subnet=H`, `num_os=O`, `num_services=V`, `num_processes=P`:

```
Host vector size = S + H + 1(compromised) + 1(reachable) + 1(discovered)
                 + 1(value) + 1(disc_value) + 1(access) + O + V + P

Flat obs size = (total_hosts + 1) × host_vec_size

Per-host segment layout:
  [0..S-1]          : subnet one-hot
  [S..S+H-1]        : host-within-subnet one-hot
  [S+H]             : compromised (0/1)
  [S+H+1]           : reachable (0/1)
  [S+H+2]           : discovered (0/1)
  [S+H+3]           : value (float)
  [S+H+4]           : discovery_value (float)
  [S+H+5]           : access (0=none, 1=user, 2=root)
  [S+H+6..S+H+5+O]  : OS one-hot
  [S+H+6+O..+V]     : service binary flags
  [S+H+6+O+V..+P]   : process binary flags
```

### tiny.yml Example

```
Subnets: 3, Max hosts: 3
OS: ['linux'], Services: ['ssh'], Processes: ['apache']
Host vec size = 3 + 3 + 6 + 1 + 1 + 1 = 15
Flat obs = 4 × 15 = 60  (3 hosts + 1 auxiliary)
```

---

## Appendix C: SCRIPT State Vector Layout Reference

```
Total: 1538 dimensions

[0..1]           : access  (2-dim: [compromised?, reachable?])
[2..385]         : os_sbert  (384-dim SBERT of "linux" / "windows")
[386..769]       : port_sbert  (384-dim SBERT of "22,80,445")
[770..1153]      : service_sbert  (384-dim SBERT of "ssh,http,samba")
[1154..1537]     : web_fp_sbert  (384-dim SBERT of web fingerprint / process proxy)

SBERT model: all-MiniLM-L12-v2 (384-dim output)
Encoder: src/agent/nlp/Encoder.py (singleton)
```

---

## Appendix D: Service Action Space Reference

```
Idx  Name               Category  PenGym Name       CVE Coverage
───  ─────────────────  ────────  ────────────────  ──────────
0    port_scan          scan      subnet_scan       N/A
1    service_scan       scan      service_scan      N/A
2    os_scan            scan      os_scan           N/A
3    web_scan           scan      process_scan      N/A
4    exploit_ssh        exploit   e_ssh             ~200 CVEs
5    exploit_ftp        exploit   e_ftp             ~60 CVEs
6    exploit_http       exploit   e_http            ~800 CVEs
7    exploit_smb        exploit   e_samba           ~50 CVEs
8    exploit_smtp       exploit   e_smtp            ~30 CVEs
9    exploit_rdp        exploit   None              ~20 CVEs
10   exploit_sql        exploit   None              ~80 CVEs
11   exploit_java_rmi   exploit   None              ~15 CVEs
12   exploit_misc       exploit   None              catch-all
13   privesc_tomcat     privesc   pe_tomcat         N/A
14   privesc_schtask    privesc   pe_schtask        N/A
15   privesc_daclsvc    privesc   pe_daclsvc        N/A
```

Actions 9–12 have no PenGym equivalent (`pengym_name=None`). When these actions are selected during PenGym evaluation, `ServiceActionMapper.map_action()` returns -1 and the wrapper uses `get_random_valid_action()` as fallback. In practice, the policy rarely selects these because PenGym scenarios only contain ssh/ftp/http/samba/smtp services.

---

_End of document._
