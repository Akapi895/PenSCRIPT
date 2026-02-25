# Intra-Topology CRL — Implementation Blueprint

> **Status**: Draft — implementation-ready  
> **Created**: 2025-02-26  
> **Prerequisite docs**: `crl_framework_analysis.md`, `system_status.md`, `strategy_C_shared_state_dual_training.md`  
> **Target branch**: `strC_1`

---

## Table of Contents

1. [Design Goals & Architecture Principles](#1-design-goals--architecture-principles)
2. [New Training Pipeline Design](#2-new-training-pipeline-design)
3. [Specific Code Changes](#3-specific-code-changes)
4. [Proposed Code Structure After Refactor](#4-proposed-code-structure-after-refactor)
5. [Preserving Existing Components](#5-preserving-existing-components)
6. [Evaluation Strategy](#6-evaluation-strategy)
7. [Migration Plan & Risk Mitigation](#7-migration-plan--risk-mitigation)

---

## 1. Design Goals & Architecture Principles

### 1.1 Problem Statement

The current Phase 3 (`phase3_pengym_finetuning`) groups PenGym tasks **by tier
across all topologies**. Concretely, it builds `tier_groups` as:

```
tier_groups = [
    ("T1", [tiny_T1, small-linear_T1, medium_T1, ...]),   # all T1 variants
    ("T2", [tiny_T2, small-linear_T2, medium_T2, ...]),   # all T2 variants
    ...
]
```

Each tier group is fed into **one** `train_continually()` call, meaning EWC
consolidates parameters across topologies that have fundamentally different
state spaces, reward structures, and solvability profiles. When a hard
topology (e.g. `medium_T3`) fails, EWC locks the damaged parameters and
poisons subsequent tasks — the **cross-topology death spiral** observed in
exp2, where θ_dual achieved 25.8% SR overall (tiny 75–80%, small/medium 0%).

### 1.2 Solution: Intra-Topology CRL

Replace cross-topology CRL with **per-topology independent CRL streams**.
Each topology gets its own stream of T1→T2→T3→T4 tasks, trained with a fresh
`Agent_CL` instance forked from the Phase 2 transferred agent.

```
Stream "tiny":           tiny_T1 → tiny_T2 → tiny_T3 → tiny_T4
Stream "small-linear":   small-linear_T1 → small-linear_T2 → ... → T4
Stream "medium-single":  medium-single-site_T1 → ... → T4
...
```

### 1.3 Design Principles

| #   | Principle                                  | Rationale                                                                  |
| --- | ------------------------------------------ | -------------------------------------------------------------------------- |
| P1  | **One topology ↔ one CRL stream**          | Tasks within the same topology share network structure → EWC is meaningful |
| P2  | **Fresh Agent_CL per stream**              | Each stream forks from θ_transferred; no cross-stream EWC contamination    |
| P3  | **Preserve all 5 SCRIPT pillars**          | Teacher guidance, KL imitation, KD, retrospection, Online EWC              |
| P4  | **Cross-topology generalisation via eval** | Measured by heldout scenarios in Phase 4, not by training                  |
| P5  | **Best-stream selection for final agent**  | Pick the θ_dual with highest per-stream SR for final Phase 4 agent         |
| P6  | **Backward compatible CLI**                | `run_strategy_c.py` works unchanged; new behaviour activated by config     |

### 1.4 Key Architectural Decisions

1. **Fork-per-stream**: Each topology stream deep-copies `θ_transferred` from
   Phase 2. This is the simplest isolation strategy and guarantees zero
   cross-stream contamination. Memory cost is ~4× for 8 topologies at ~15 MB
   each = ~120 MB total — acceptable.

2. **Stream ordering**: Within each stream, tasks are ordered T1→T2→T3→T4
   (increasing difficulty). If multiple variants exist per tier (e.g.
   `tiny_T1_000`, `tiny_T1_001`, ..., `tiny_T1_009`), they are treated as
   **sub-tasks within that tier** — trained sequentially within the tier
   boundary.

3. **Stream merging**: After all streams complete, the best-performing stream's
   agent becomes `θ_dual` for Phase 4. Optionally, an ensemble or averaged
   model can be used (deferred to future work).

4. **Evaluation scope**: Phase 4 evaluates the selected θ_dual on
   **all** PenGym tasks (not just the stream it was trained on) — this
   measures cross-topology generalisation without training on cross-topology
   CRL.

---

## 2. New Training Pipeline Design

### 2.1 Updated Pipeline Overview

```
Phase 0  →  Validation (unchanged)
Phase 1  →  Sim CRL → θ_uni (unchanged)
Phase 2  →  Domain Transfer → θ_transferred (unchanged)
Phase 3  →  Intra-Topology CRL (CHANGED)
  3a. Parse scenario list → group by topology
  3b. For each topology:
      - Fork θ_transferred → θ_stream[topo]
      - Build T1→T4 task sequence within topology
      - train_continually(T1→T2→T3→T4)
      - Checkpoint after each tier
  3c. Select best θ_stream → θ_dual
Phase 4  →  Multi-agent eval (ENHANCED: per-stream + cross-topo)
```

### 2.2 Stream Splitting Algorithm

**Input**: `self.pengym_scenarios` — list of scenario paths, e.g.:

```
data/scenarios/generated/compiled/tiny_T1_000.yml
data/scenarios/generated/compiled/tiny_T2_003.yml
data/scenarios/generated/compiled/small-linear_T3_001.yml
data/scenarios/tiny.yml                          # base scenario (no tier)
...
```

**Grouping logic** (pseudocode):

```python
def _build_topology_streams(self, scenarios: List[str]) -> Dict[str, List[tuple]]:
    """Group scenarios into per-topology streams.

    Returns
    -------
    dict[str, list[tuple[int, str, str]]]
        topology_name → [(global_idx, tier_key, scenario_path), ...]
        Sorted by tier within each topology.
    """
    streams: Dict[str, list] = defaultdict(list)

    for idx, sc_path in enumerate(scenarios):
        stem = Path(sc_path).stem  # e.g. "tiny_T2_003"
        tier_match = re.search(r'_T(\d+)_', stem)
        if tier_match:
            tier_key = f"T{tier_match.group(1)}"
            base_name = stem[:tier_match.start()]  # "tiny"
        else:
            tier_key = "T0"  # base scenario
            base_name = stem

        streams[base_name].append((idx, tier_key, sc_path))

    # Sort each stream by tier: T0 < T1 < T2 < T3 < T4
    tier_order = {"T0": 0, "T1": 1, "T2": 2, "T3": 3, "T4": 4}
    for topo in streams:
        streams[topo].sort(key=lambda x: (tier_order.get(x[1], 99), x[2]))

    return dict(streams)
```

**Example output** for a 6-scenario run:

```python
{
    "tiny":         [(0, "T1", "tiny_T1_000.yml"), (1, "T2", "tiny_T2_000.yml")],
    "small-linear": [(2, "T1", "small-linear_T1_000.yml"), (3, "T2", "small-linear_T2_000.yml")],
    "medium-single-site": [(4, "T1", "medium-single-site_T1_000.yml"), (5, "T2", "medium-single-site_T2_000.yml")],
}
```

### 2.3 Per-Stream CRL Training

For each topology stream:

```python
for topo_name, stream_tasks in topology_streams.items():
    # 1. Fork the transferred agent (fresh EWC, preserved weights)
    stream_agent = copy.deepcopy(self._theta_dual)
    stream_tb = SummaryWriter(log_dir=str(tb_dir / f"stream_{topo_name}"))
    stream_agent.tf_logger = stream_tb

    # 2. Build PenGym adapters for this stream
    stream_pengym_tasks = [self._pengym_tasks[idx] for idx, _, _ in stream_tasks]

    # 3. Build episode & step_limit schedules (local indices: 0..N-1)
    stream_episode_schedule = {...}   # local_idx → episode_count
    stream_step_limit_schedule = {...}  # local_idx → step_limit

    # 4. Train CRL: T1→T2→T3→T4 within this topology
    cl_matrix = stream_agent.train_continually(
        task_list=stream_pengym_tasks,
        eval_freq=eval_freq,
        save_agent=False,
        verbose=True,
        episode_schedule=stream_episode_schedule,
        step_limit_schedule=stream_step_limit_schedule,
    )

    # 5. Evaluate stream performance
    stream_sr = stream_agent.eval_success_rate

    # 6. Save checkpoint
    stream_agent.save(path=checkpoint_dir / f"stream_{topo_name}")
    stream_results[topo_name] = {
        "agent": stream_agent,
        "sr": stream_sr,
        "cl_matrix": cl_matrix,
    }
```

### 2.4 Best-Stream Selection

After all streams complete:

```python
# Select best stream by highest eval SR
best_topo = max(stream_results, key=lambda t: stream_results[t]["sr"])
self._theta_dual = stream_results[best_topo]["agent"]

# Keep all stream agents for multi-stream eval in Phase 4
self._stream_agents = {
    topo: data["agent"] for topo, data in stream_results.items()
}
```

### 2.5 Multi-Stream Training: Interleaved vs Sequential

Two possible execution strategies:

| Strategy        | Description                                    | Pros                      | Cons                          |
| --------------- | ---------------------------------------------- | ------------------------- | ----------------------------- |
| **Sequential**  | Complete one stream fully before starting next | Simpler code, less memory | Total time = sum of streams   |
| **Interleaved** | Train one tier across all streams before next  | Better GPU utilisation    | More complex, unclear benefit |

**Recommendation**: Sequential (simpler, same total compute, clearer logs).

---

## 3. Specific Code Changes

### 3.1 File: `src/training/dual_trainer.py` (PRIMARY)

#### 3.1.1 New method: `_build_topology_streams()`

**Location**: After `_resolve_step_limit_schedule()` (line ~270).

```python
def _build_topology_streams(
    self, scenario_paths: List[str],
) -> Dict[str, List[tuple]]:
    """Group scenarios by base topology name into independent CRL streams.

    Parameters
    ----------
    scenario_paths : list[str]
        Scenario file paths (may be overlay or base).

    Returns
    -------
    dict[str, list[tuple[int, str, str]]]
        Mapping topology_name → [(global_index, tier_key, path), ...]
        Each list is sorted by tier (T0 < T1 < T2 < T3 < T4), then by
        variant index within the same tier.
    """
    import re
    from collections import defaultdict

    streams: Dict[str, list] = defaultdict(list)
    tier_order = {"T0": 0, "T1": 1, "T2": 2, "T3": 3, "T4": 4}

    for idx, sc_path in enumerate(scenario_paths):
        stem = Path(sc_path).stem
        tier_match = re.search(r'_T(\d+)_', stem)
        if tier_match:
            tier_key = f"T{tier_match.group(1)}"
            base_name = stem[:tier_match.start()]
        else:
            tier_key = "T0"
            base_name = stem
        streams[base_name].append((idx, tier_key, str(sc_path)))

    # Sort within each stream by tier then variant
    for topo in streams:
        streams[topo].sort(key=lambda x: (tier_order.get(x[1], 99), x[2]))

    return dict(streams)
```

#### 3.1.2 Rewrite: `phase3_pengym_finetuning()` (lines 530–635)

**Replace** the entire `tier_groups` logic with topology-streams logic:

```python
def phase3_pengym_finetuning(self, eval_freq: int = 5) -> Dict[str, Any]:
    """Phase 3 — Fine-tune via Intra-Topology CRL.

    Each topology gets an independent CRL stream forked from the
    Phase 2 transferred agent.  Tasks within a stream are ordered
    T1→T2→T3→T4 (intra-topology curriculum).  The best-performing
    stream's agent becomes θ_dual for Phase 4.
    """
    logging.info("\n" + "=" * 60)
    logging.info("[Phase 3] Intra-Topology CRL Fine-tuning → θ_dual")
    logging.info("=" * 60)

    if self._theta_dual is None:
        logging.error("[Phase 3] No transferred agent from Phase 2")
        return {"error": "no_phase2_agent"}

    if not hasattr(self, '_pengym_tasks') or not self._pengym_tasks:
        logging.error("[Phase 3] No PenGym tasks")
        return {"error": "no_pengym_tasks"}

    t0 = time.time()

    # Resolve per-task schedules (global indices)
    episode_schedule = self._resolve_episode_schedule(self.pengym_scenarios)
    step_limit_schedule = self._resolve_step_limit_schedule(self.pengym_scenarios)

    # ── Build per-topology streams ──
    topology_streams = self._build_topology_streams(self.pengym_scenarios)
    logging.info(f"[Phase 3] Topology streams: {list(topology_streams.keys())} "
                 f"({sum(len(v) for v in topology_streams.values())} total tasks)")

    checkpoint_dir = self.output_dir / "models" / "stream_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    stream_results: Dict[str, Dict[str, Any]] = {}
    all_per_task_rewards: Dict[str, list] = {}

    # ── Sequential stream training ──
    for topo_name, stream_entries in topology_streams.items():
        logging.info(f"\n{'─' * 50}")
        logging.info(f"[Phase 3] Stream '{topo_name}': "
                     f"{len(stream_entries)} tasks "
                     f"({', '.join(e[1] for e in stream_entries)})")
        logging.info(f"{'─' * 50}")

        # Fork transferred agent for this stream
        stream_agent = copy.deepcopy(self._theta_dual)
        stream_tb = SummaryWriter(
            log_dir=str(self.tb_dir / f"phase3_stream_{topo_name}")
        )
        stream_agent.tf_logger = stream_tb

        # Build local task list & schedules
        global_indices = [e[0] for e in stream_entries]
        stream_tasks = [self._pengym_tasks[gi] for gi in global_indices]

        local_ep_schedule = None
        if episode_schedule:
            local_ep_schedule = {
                local_i: episode_schedule[gi]
                for local_i, gi in enumerate(global_indices)
                if gi in episode_schedule
            }

        local_sl_schedule = None
        if step_limit_schedule:
            local_sl_schedule = {
                local_i: step_limit_schedule[gi]
                for local_i, gi in enumerate(global_indices)
                if gi in step_limit_schedule
            }

        # Train CRL within this topology stream
        cl_matrix = stream_agent.train_continually(
            task_list=stream_tasks,
            eval_freq=eval_freq,
            save_agent=False,
            verbose=True,
            episode_schedule=local_ep_schedule,
            step_limit_schedule=local_sl_schedule,
        )

        # Collect per-task rewards
        for local_i, gi in enumerate(global_indices):
            task_obj = self._pengym_tasks[gi]
            task_name = getattr(task_obj, 'ip', f'task_{gi}')
            all_per_task_rewards[task_name] = (
                cl_matrix.Rewards_current_task[local_i:local_i + 1]
                if hasattr(cl_matrix, 'Rewards_current_task')
                else []
            )

        # Save stream checkpoint
        stream_ckpt = checkpoint_dir / f"stream_{topo_name}"
        stream_agent.save(path=stream_ckpt)

        stream_sr = stream_agent.eval_success_rate
        stream_results[topo_name] = {
            "agent": stream_agent,
            "sr": stream_sr,
            "cl_matrix": cl_matrix,
            "num_tasks": len(stream_tasks),
            "checkpoint": str(stream_ckpt),
        }

        logging.info(f"[Phase 3] Stream '{topo_name}': SR={stream_sr:.2%}")
        stream_tb.close()

    # ── Select best stream → θ_dual ──
    best_topo = max(stream_results, key=lambda t: stream_results[t]["sr"])
    self._theta_dual = stream_results[best_topo]["agent"]
    self._stream_agents = {
        t: d["agent"] for t, d in stream_results.items()
    }
    self._stream_results = stream_results

    # Build tier_checkpoints equivalent from stream checkpoints
    self._tier_checkpoints = {
        f"stream_{t}": d["checkpoint"] for t, d in stream_results.items()
    }
    self._dual_per_task_rewards = all_per_task_rewards

    phase3_time = time.time() - t0

    # Save final θ_dual model
    phase3_model_dir = self.output_dir / "models" / "phase3_dual"
    phase3_model_dir.mkdir(parents=True, exist_ok=True)
    self._theta_dual.save(path=phase3_model_dir)

    results = {
        "mode": "intra_topology_crl",
        "num_streams": len(topology_streams),
        "streams": {
            t: {"sr": d["sr"], "num_tasks": d["num_tasks"],
                "checkpoint": d["checkpoint"]}
            for t, d in stream_results.items()
        },
        "best_stream": best_topo,
        "best_sr": stream_results[best_topo]["sr"],
        "num_tasks": sum(d["num_tasks"] for d in stream_results.values()),
        "train_time_s": round(phase3_time, 2),
        "final_sr": self._theta_dual.eval_success_rate,
        "final_reward": self._theta_dual.eval_rewards,
        "model_dir": str(phase3_model_dir),
    }
    logging.info(f"[Phase 3] Complete: best='{best_topo}' SR={results['best_sr']:.2%}, "
                 f"time={results['train_time_s']}s")
    return results
```

#### 3.1.3 New method: `_build_stream_tier_groups()` (optional)

If we want per-tier checkpoints **within** each stream (granularity):

```python
def _build_stream_tier_groups(
    self, stream_entries: List[tuple],
) -> List[tuple]:
    """Group stream entries by tier for intra-stream checkpointing.

    Parameters
    ----------
    stream_entries : list[tuple[int, str, str]]
        (global_idx, tier_key, path) — already sorted by tier.

    Returns
    -------
    list[tuple[str, list[int]]]
        (tier_name, [local_indices]) within this stream.
    """
    tier_groups: List[tuple] = []
    current_tier = None
    for local_i, (_, tier_key, _) in enumerate(stream_entries):
        if tier_key != current_tier:
            tier_groups.append((tier_key, []))
            current_tier = tier_key
        tier_groups[-1][1].append(local_i)
    return tier_groups
```

#### 3.1.4 Enhance `phase4_evaluation()` — Per-Stream Eval (lines 650+)

After the existing multi-agent eval, add per-stream evaluation:

```python
# ── Per-stream evaluation (NEW) ──
stream_agents = getattr(self, '_stream_agents', {})
if stream_agents:
    logging.info("[Phase 4] Evaluating per-stream agents...")
    per_stream_results: Dict[str, Dict[str, Any]] = {}

    for topo_name, stream_agent in stream_agents.items():
        agent_key = f"stream_{topo_name}"

        # Evaluate on OWN topology tasks only
        own_scenarios = [
            sc for sc in self.pengym_scenarios
            if Path(sc).stem.startswith(topo_name)
        ]
        if own_scenarios:
            own_tasks = self._create_eval_tasks_from(own_scenarios)
            evaluator.register_agent(agent_key, stream_agent)
            own_eval = evaluator.evaluate_agent(
                agent_key, own_tasks, domain="pengym",
            )
            per_stream_results[topo_name] = {
                "own_topology": own_eval,
            }

        # Evaluate on ALL PenGym tasks (cross-topology generalisation)
        all_tasks = self._create_eval_tasks_from(self.pengym_scenarios)
        cross_eval = evaluator.evaluate_agent(
            agent_key, all_tasks, domain="pengym",
        )
        per_stream_results.setdefault(topo_name, {})["cross_topology"] = cross_eval

    results["per_stream"] = per_stream_results
```

#### 3.1.5 Update `train_pengym_scratch()` to match topology-stream format

The scratch agent should also use topology-stream training for fair
comparison. Clone the same `_build_topology_streams()` logic so that
the scratch baseline trains with identical per-topology CRL streams.

> **Alternative**: If we decide the scratch agent should remain a single-stream
> baseline (showing what happens WITHOUT intra-topology structure), keep it
> unchanged and document this asymmetry.

**Recommendation**: Keep scratch unchanged (single-stream). The point of
scratch is to be the no-transfer baseline. If we also split scratch into
per-topology streams, the comparison loses meaning.

### 3.2 File: `data/config/curriculum_episodes.json` (MINOR)

Add a new config key to opt into intra-topology mode:

```json
{
  "training_mode": "intra_topology",
  "_training_mode_options": ["cross_topology", "intra_topology"],
  ...existing keys...
}
```

This is optional — the pipeline should default to `intra_topology` going
forward.

### 3.3 File: `run_strategy_c.py` (MINOR)

Add a CLI flag for training mode:

```python
parser.add_argument(
    "--training-mode", type=str, default="intra_topology",
    choices=["intra_topology", "cross_topology"],
    help="Phase 3 training mode: 'intra_topology' (per-topology CRL streams, "
         "recommended) or 'cross_topology' (legacy tier-grouped CRL).",
)
```

Pass to `DualTrainer` constructor. If `cross_topology`, fall back to
the original `tier_groups` logic (preserved as `_phase3_cross_topology()`
for backward compatibility).

### 3.4 File: `src/agent/agent_continual.py` (NO CHANGES NEEDED)

`train_continually()` already accepts `task_list`, `episode_schedule`,
`step_limit_schedule` — this interface supports per-topology streams
without modification. Each stream simply calls `train_continually()`
with its own local task list and schedules.

### 3.5 File: `src/agent/continual/Script.py` (NO CHANGES NEEDED)

ScriptAgent, KnowledgeExplorer, KnowledgeKeeper, OnlineEWC — all operate
at the task-list level. Since each stream forks a fresh `Agent_CL` via
`copy.deepcopy()`, the EWC state is independent per stream.

### 3.6 File: `src/evaluation/strategy_c_eval.py` (MINOR)

Add utility for per-stream evaluation report formatting:

```python
def format_per_stream_report(
    self, per_stream_results: Dict[str, Dict[str, Any]],
) -> str:
    """Format per-stream evaluation results as a table."""
    lines = ["Stream | Own SR | Cross SR | Own NR"]
    lines.append("-" * 45)
    for topo, data in per_stream_results.items():
        own_sr = data.get("own_topology", {}).get("success_rate", "N/A")
        cross_sr = data.get("cross_topology", {}).get("success_rate", "N/A")
        own_nr = data.get("own_topology", {}).get("normalized_reward", "N/A")
        own_sr_s = f"{own_sr:.1%}" if isinstance(own_sr, float) else str(own_sr)
        cross_sr_s = f"{cross_sr:.1%}" if isinstance(cross_sr, float) else str(cross_sr)
        own_nr_s = f"{own_nr:.3f}" if isinstance(own_nr, float) else str(own_nr)
        lines.append(f"{topo:20s} | {own_sr_s:>6s} | {cross_sr_s:>8s} | {own_nr_s:>6s}")
    return "\n".join(lines)
```

### 3.7 File: `src/training/domain_transfer.py` (NO CHANGES NEEDED)

Domain transfer operates on a single `Agent_CL` instance. The fork
happens after Phase 2 is complete (one `deepcopy` per stream), so the
transfer manager doesn't need to know about topology streams.

---

## 4. Proposed Code Structure After Refactor

### 4.1 New/Modified Files

```
src/training/
    dual_trainer.py               ← MODIFIED (primary change)
        + _build_topology_streams()        NEW method
        + phase3_pengym_finetuning()       REWRITTEN (intra-topology)
        + _phase3_cross_topology()         MOVED (legacy, optional)
        + phase4_evaluation()              ENHANCED (per-stream eval)
    domain_transfer.py            ← UNCHANGED

src/agent/
    agent_continual.py            ← UNCHANGED
    continual/Script.py           ← UNCHANGED

src/evaluation/
    strategy_c_eval.py            ← MINOR (per-stream report)
    metric_store.py               ← UNCHANGED

data/config/
    curriculum_episodes.json      ← MINOR (training_mode key)

run_strategy_c.py                 ← MINOR (--training-mode flag)
```

### 4.2 DualTrainer Internal State Changes

**New attributes after Phase 3**:

| Attribute           | Type                  | Description                                   |
| ------------------- | --------------------- | --------------------------------------------- |
| `_stream_agents`    | `Dict[str, Agent_CL]` | All per-topology trained agents               |
| `_stream_results`   | `Dict[str, dict]`     | Per-stream SR, checkpoint paths, cl_matrix    |
| `_theta_dual`       | `Agent_CL`            | Best stream's agent (selected by SR)          |
| `_tier_checkpoints` | `Dict[str, str]`      | Per-stream checkpoint paths (backward compat) |

**Removed attributes**:

| Attribute             | Replaced By                                        |
| --------------------- | -------------------------------------------------- |
| `tier_groups` (local) | `topology_streams` via `_build_topology_streams()` |

### 4.3 Data Flow Diagram

```
Phase 2 Output:  θ_transferred (single Agent_CL)
                        │
                  ┌─────┴─────┐
                  │ deepcopy() │  × N topologies
                  └─────┬─────┘
                        │
            ┌───────────┼───────────────────┐
            ▼           ▼                   ▼
    ┌──────────┐  ┌──────────┐       ┌──────────┐
    │ Stream   │  │ Stream   │  ...  │ Stream   │
    │ "tiny"   │  │ "small-  │       │ "medium" │
    │          │  │  linear" │       │          │
    │ T1→T2→  │  │ T1→T2→  │       │ T1→T2→  │
    │ T3→T4   │  │ T3→T4   │       │ T3→T4   │
    └────┬─────┘  └────┬─────┘       └────┬─────┘
         │             │                  │
         ▼             ▼                  ▼
    SR = 0.85     SR = 0.40          SR = 0.10
         │             │                  │
         └──────┬──────┘──────────────────┘
                │
          Best stream
          selection
                │
                ▼
           θ_dual = θ_stream["tiny"]
                │
                ▼
          Phase 4 Eval
          (all tasks + heldout)
```

---

## 5. Preserving Existing Components

### 5.1 Components That MUST NOT Change

| Component                        | File                                    | Why                                                   |
| -------------------------------- | --------------------------------------- | ----------------------------------------------------- |
| `UnifiedStateEncoder`            | `envs/core/unified_state_encoder.py`    | Shared 1540-dim encoding across all streams           |
| `ServiceActionSpace`             | `agent/actions/service_action_space.py` | 16-group action space, topology-agnostic              |
| `KnowledgeExplorer` / `Keeper`   | `agent/continual/Script.py`             | Forked per stream via deepcopy, not modified          |
| `OnlineEWC`                      | `agent/continual/Script.py`             | EWC state is per-stream (deepcopy gives independence) |
| `DomainTransferManager`          | `training/domain_transfer.py`           | Runs once in Phase 2 before fork                      |
| `PenGymHostAdapter`              | `envs/adapters/pengym_host_adapter.py`  | One adapter per task, unchanged                       |
| `PPO_Config` / `Script_Config`   | `agent/policy/config.py`                | Shared config, no per-stream overrides needed         |
| Phase 0, Phase 1, Phase 2        | `training/dual_trainer.py`              | Pre-fork phases, completely unchanged                 |
| `_resolve_episode_schedule()`    | `training/dual_trainer.py`              | Works on global indices, called before fork           |
| `_resolve_step_limit_schedule()` | `training/dual_trainer.py`              | Same — per-topology step limits already supported     |
| EWC λ, β, LR factor              | `agent/policy/config.py`                | Hyperparameters shared across streams                 |

### 5.2 5 SCRIPT Pillars — Preservation Guarantee

| Pillar                 | Where It Lives                    | Impact of Intra-Topo CRL                                    |
| ---------------------- | --------------------------------- | ----------------------------------------------------------- |
| **Teacher Guidance**   | `ScriptAgent.guide_kl_scale`      | ✅ Preserved — per-stream explorer uses keeper for KL guide |
| **KL Imitation**       | `ExplorePolicy.calcuate_ppo_loss` | ✅ Preserved — guide_policy is the stream's own keeper      |
| **Knowledge Distill.** | `ScriptAgent.policy_preservation` | ✅ Preserved — keeper updated after each task within stream |
| **Retrospection**      | `ScriptAgent.retrospect()`        | ✅ Preserved — retrospection buffer is per-stream           |
| **Online EWC**         | `OnlineEWC.update_importances()`  | ✅ Preserved AND improved — EWC only consolidates same-topo |

### 5.3 Backward Compatibility

The original `tier_groups` logic is preserved as a fallback:

```python
def _phase3_cross_topology(self, eval_freq: int = 5) -> Dict[str, Any]:
    """Legacy Phase 3 — cross-topology tier-grouped CRL.

    Preserved for backward compatibility and A/B comparison.
    """
    # ... (move current phase3 logic here, unchanged) ...
```

`phase3_pengym_finetuning()` dispatches based on `training_mode`:

```python
def phase3_pengym_finetuning(self, eval_freq: int = 5) -> Dict[str, Any]:
    mode = getattr(self, 'training_mode', 'intra_topology')
    if mode == 'cross_topology':
        return self._phase3_cross_topology(eval_freq=eval_freq)
    return self._phase3_intra_topology(eval_freq=eval_freq)
```

---

## 6. Evaluation Strategy

### 6.1 Intra-Topology Metrics (Per-Stream)

For each topology stream, compute within-stream CRL metrics:

| Metric       | Definition                                                 | Measures                         |
| ------------ | ---------------------------------------------------------- | -------------------------------- |
| **SR_own**   | Success rate on own topology's T1–T4 tasks                 | Within-topology learning         |
| **FT_own**   | SR on T*{k} immediately after training T*{k} with no prior | Forward transfer within topology |
| **BT_own**   | SR on T*{1..k-1} after training T*{k}                      | Backward transfer / forgetting   |
| **EWC_cost** | Fisher-weighted L2 drift from T1 end → T4 end              | EWC regularisation effectiveness |

These are computed by `train_continually()`'s built-in eval loop and
the existing `CL_Train_matrix`.

### 6.2 Cross-Topology Generalisation (Heldout)

Evaluate each stream's agent on **all other topologies**:

```
stream_tiny's agent:
  - Own: tiny_T1..T4        → SR_own = X
  - Cross: small-linear_T1..T4, medium_T1..T4, ... → SR_cross = Y
```

This is the **zero-shot cross-topology transfer** metric. If `SR_cross > 0`,
it shows that intra-topology CRL produces policies that generalise beyond
the training topology.

### 6.3 Aggregate Metrics for Paper

| Metric                | Computation                                           |
| --------------------- | ----------------------------------------------------- |
| **Mean SR_own**       | Average SR_own across all topology streams            |
| **Best SR_own**       | Max SR_own (= θ_dual's performance)                   |
| **Mean SR_cross**     | Average cross-topology SR                             |
| **FT_SR**             | θ_dual SR − θ_scratch SR (forward transfer from sim)  |
| **BT_SR**             | θ_dual sim_SR − θ_unified sim_SR (backward transfer)  |
| **Forgetting matrix** | Per-stream F\_{i,j} matrix (tasks within each stream) |
| **CE curves**         | Per-stream cumulative-error curves over T1→T4         |

### 6.4 Phase 4 Enhanced Report Structure

```json
{
  "agents": {
    "theta_dual": { "success_rate": 0.85, ... },
    "theta_pengym_scratch": { "success_rate": 0.12, ... }
  },
  "per_stream": {
    "tiny": {
      "own_topology": { "success_rate": 0.90, ... },
      "cross_topology": { "success_rate": 0.15, ... }
    },
    "small-linear": {
      "own_topology": { "success_rate": 0.40, ... },
      "cross_topology": { "success_rate": 0.05, ... }
    }
  },
  "best_stream": "tiny",
  "transfer_metrics": {
    "FT_SR": 0.73,
    "BT_SR": -0.02,
    "BT_KL": 0.03,
    "BT_fisher_dist": 0.15
  }
}
```

---

## 7. Migration Plan & Risk Mitigation

### 7.1 Implementation Steps (Ordered)

| Step | Action                                                               | Estimated Effort | Risk   |
| ---- | -------------------------------------------------------------------- | ---------------- | ------ |
| 1    | Add `_build_topology_streams()` to `DualTrainer`                     | 30 min           | Low    |
| 2    | Rename current `phase3_pengym_finetuning` → `_phase3_cross_topology` | 10 min           | Low    |
| 3    | Write new `_phase3_intra_topology()` method                          | 1–2 hr           | Medium |
| 4    | Add dispatch in `phase3_pengym_finetuning()`                         | 10 min           | Low    |
| 5    | Add `--training-mode` CLI flag to `run_strategy_c.py`                | 15 min           | Low    |
| 6    | Add per-stream eval section to `phase4_evaluation()`                 | 30 min           | Low    |
| 7    | Add `training_mode` key to `curriculum_episodes.json`                | 5 min            | Low    |
| 8    | Sanity test: tiny-only stream (1 topology, T1+T2)                    | 20 min (run)     | Low    |
| 9    | Full test: 3+ topologies × T1+T2                                     | 2–5 hr (run)     | Medium |
| 10   | Analyse results, compare with exp2 cross-topology baseline           | 1 hr             | Low    |

### 7.2 Risk Mitigation

| Risk                                     | Mitigation                                                     |
| ---------------------------------------- | -------------------------------------------------------------- |
| Memory pressure from N deepcopies        | Run top-3 topologies first; only ~15 MB per agent              |
| A stream learns nothing (SR=0)           | Per-stream early-stop: if SR=0 after T2, skip T3/T4            |
| Best-stream selection unstable           | Use K-episode eval (K=20) for robust SR estimation             |
| Scratch baseline unfair comparison       | Document that scratch = single-stream; FT computation adjusted |
| EWC locks after T1, prevents T2 learning | Step 1 within each stream — monitor per-tier SR to detect      |
| Forgetting matrix dims change            | F matrix is now per-stream (4×4 max) instead of global         |

### 7.3 Smoke Test Procedure

Before running full experiments:

```bash
# 1. Tiny stream only — 2 tasks, ~10 min
python run_strategy_c.py \
    --sim-scenarios data/scenarios/chain/chain_1.json \
    --pengym-scenarios data/scenarios/generated/compiled/tiny_T1_000.yml \
                       data/scenarios/generated/compiled/tiny_T2_000.yml \
    --episode-config data/config/curriculum_episodes.json \
    --training-mode intra_topology \
    --train-scratch \
    --output-dir outputs/strategy_c/smoke_intra

# Expected: 1 stream ("tiny"), 2 tasks, SR >> 0
```

```bash
# 2. Multi-stream — 3 topologies × T1+T2
python run_strategy_c.py \
    --sim-scenarios data/scenarios/chain/chain_1.json \
    --pengym-scenarios \
        data/scenarios/generated/compiled/tiny_T1_000.yml \
        data/scenarios/generated/compiled/tiny_T2_000.yml \
        data/scenarios/generated/compiled/small-linear_T1_000.yml \
        data/scenarios/generated/compiled/small-linear_T2_000.yml \
        data/scenarios/generated/compiled/medium-single-site_T1_000.yml \
        data/scenarios/generated/compiled/medium-single-site_T2_000.yml \
    --episode-config data/config/curriculum_episodes.json \
    --training-mode intra_topology \
    --train-scratch \
    --output-dir outputs/strategy_c/exp3_intra

# Expected: 3 independent streams, tiny stream SR high, others >0
```

### 7.4 Success Criteria

The intra-topology refactor is considered successful if:

1. **θ_dual SR ≥ 50%** on at least one topology (vs 25.8% cross-topology)
2. **No death spiral**: per-stream SR monotonically non-decreasing or
   decreasing < 10% after each tier
3. **EWC effective**: Fisher-weighted drift per stream < 50% of cross-topology
4. **FT_SR > 0**: θ_dual outperforms θ_scratch on the best stream
5. **Pipeline completes** in < 6 hours on CPU

---

## Appendix A: Current vs Proposed — Side-by-Side

```
CURRENT (cross-topology):                PROPOSED (intra-topology):
─────────────────────────                ──────────────────────────
θ_transferred                            θ_transferred
     │                                        │
     ▼                                   ┌────┼────┐
 tier_groups:                            │    │    │   deepcopy per topo
  T1: [tiny, small, med]   ──CRL──►     ▼    ▼    ▼
  T2: [tiny, small, med]   ──CRL──►    tiny  small  med
  T3: [tiny, small, med]   ──CRL──►    T1→4  T1→4   T1→4
  T4: [tiny, small, med]   ──CRL──►     │     │      │
     │                                   │     │      │
     ▼                               max SR ──►θ_dual
   θ_dual                                     │
     │                                         ▼
     ▼                                    Phase 4 eval
   Phase 4 eval                          (all tasks + heldout)

Problem: EWC contamination             Solution: isolated EWC per topo
across different topologies             within same topology only
```

## Appendix B: Glossary

| Term                   | Definition                                                                     |
| ---------------------- | ------------------------------------------------------------------------------ |
| **Topology**           | Base scenario (e.g., `tiny`, `small-linear`, `medium`)                         |
| **Tier / Overlay**     | Difficulty variant T1–T4 applied on top of base topology                       |
| **Stream**             | Independent CRL training sequence for one topology: T1→T2→T3→T4                |
| **Cross-topology CRL** | (Legacy) Training CRL across different topologies — causes death spiral        |
| **Intra-topology CRL** | (New) Training CRL within same topology, measuring cross-topo via eval only    |
| **Death spiral**       | EWC consolidating damaged params from failed tasks, poisoning subsequent tasks |
| **Fork**               | `copy.deepcopy(θ_transferred)` — creates independent agent for each stream     |
