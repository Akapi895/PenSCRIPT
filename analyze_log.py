#!/usr/bin/env python3
"""
Analyze SCRIPT CRL training log to diagnose failures.

Usage:
    python analyze_log.py outputs/logs/script-train_20260214_224945.log
"""

import re
import sys
import json
from collections import defaultdict, Counter
from pathlib import Path


def analyze_log(log_path: str):
    """Parse and analyze a SCRIPT CRL training log file."""

    print(f"Analyzing: {log_path}")
    print(f"File size: {Path(log_path).stat().st_size / 1024 / 1024:.1f} MB")
    print()

    # ── Counters ─────────────────────────────────────────────────────
    total_lines = 0
    current_task = -1
    current_phase = "init"  # init, training, compress, eval

    # Per-task metrics
    task_actions = defaultdict(Counter)       # task_id → {action_result: count}
    task_rewards = defaultdict(list)          # task_id → [episode_final_reward]
    task_steps = defaultdict(list)            # task_id → [episode_steps]
    task_episode_count = defaultdict(int)
    compress_losses = defaultdict(list)       # task_id → [(pct, loss, ewc, kd, res, trans)]
    sr_reports = []                           # [(task_id, sr_value, failed_list, total_steps)]
    task_scenarios = {}                       # task_id → scenario_info
    nan_tasks = set()
    connection_errors = defaultdict(int)
    permission_errors = defaultdict(int)

    # Episode tracking
    last_reward_str = None
    last_step_str = None

    # Action diversity per task
    task_unique_actions_success = defaultdict(set)

    # ANSI color code pattern (for stripping terminal colors)
    ansi_escape = re.compile(r'\x1b\[[0-9;]+m')
    
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            total_lines += 1
            # Strip ANSI color codes for reliable regex matching
            line = ansi_escape.sub('', line)

            # ── Task boundary ────────────────────────────────────────
            m = re.search(r'Training Task (\d+)', line)
            if m:
                t = int(m.group(1))
                if t != current_task:
                    current_task = t
                    current_phase = "training"

            # ── Compressing phase ────────────────────────────────────
            m = re.search(r'Compressing task (\d+)', line)
            if m:
                current_phase = "compress"
                compress_task = int(m.group(1))

            # ── StateAdapter info ────────────────────────────────────
            m = re.search(r'\[StateAdapter\] Initialized: (\d+) hosts, host_vec_size=(\d+), os=\[([^\]]*)\], services=\[([^\]]*)\], processes=\[([^\]]*)\]', line)
            if m and current_task >= 0 and current_task not in task_scenarios:
                task_scenarios[current_task] = {
                    'hosts': int(m.group(1)),
                    'host_vec_size': int(m.group(2)),
                    'services': m.group(4).replace("'", "").strip(),
                    'processes': m.group(5).replace("'", "").strip(),
                }

            # ── Action results ───────────────────────────────────────
            m = re.search(r"Action '(\w+)' (SUCCESS|FAILURE)", line)
            if m and current_task >= 0:
                action_name = m.group(1)
                result = m.group(2)
                task_actions[current_task][f"{action_name}_{result}"] += 1
                if result == "SUCCESS":
                    task_unique_actions_success[current_task].add(action_name)

            # ── Connection / permission errors ───────────────────────
            if 'connection_error=TRUE' in line and current_task >= 0:
                connection_errors[current_task] += 1
            if 'permission_error=TRUE' in line and current_task >= 0:
                permission_errors[current_task] += 1

            # ── Episode reward from progress bar ─────────────────────
            m = re.search(r"r='(-?\d+)/(\d+)'", line)
            if m and current_task >= 0:
                current_r = int(m.group(1))
                max_r = int(m.group(2))
                last_reward_str = (current_r, max_r)

            m = re.search(r"step='(\d+)'", line)
            if m and current_task >= 0:
                last_step_str = int(m.group(1))

            # ── Episode count from progress bar ──────────────────────
            m = re.search(r'Training Task (\d+).*?(\d+)/(\d+)', line)
            if m:
                t = int(m.group(1))
                ep = int(m.group(2))
                task_episode_count[t] = max(task_episode_count[t], ep)

            # ── Compress losses ──────────────────────────────────────
            m = re.search(
                r"Compressing task (\d+).*?(\d+)%.*?"
                r"loss='([^']+)'.*?loss_ewc='([^']+)'.*?"
                r"loss_kd='([^']+)'.*?loss_res='([^']+)'.*?"
                r"loss_trans='([^']+)'",
                line
            )
            if m:
                tid = int(m.group(1))
                pct = int(m.group(2))
                loss = m.group(3)
                ewc = m.group(4)
                kd = m.group(5)
                res = m.group(6)
                trans = m.group(7)
                compress_losses[tid].append((pct, loss, ewc, kd, res, trans))
                if 'nan' in loss.lower():
                    nan_tasks.add(tid)

            # ── SR reports ───────────────────────────────────────────
            # After ANSI strip: "After learning task 0, Previous Tasks SR: 1.0, failed_list: [],task_total_steps:14027"
            m = re.search(
                r'After learning task (\d+), Previous Tasks SR: ([0-9.]+), '
                r'failed_list:\s*\[([^\]]*)\],\s*task_total_steps:(\d+)',
                line
            )
            if m:
                sr_reports.append({
                    'task': int(m.group(1)),
                    'sr': float(m.group(2)),
                    'failed': m.group(3).strip(),
                    'total_steps': int(m.group(4)),
                })

            # ── Compress eval SR ─────────────────────────────────────
            # e_p_sr in progress bar
            m = re.search(r"e_p_sr='([^']+)'", line)
            if m and current_phase == "compress":
                pass  # captured in compress_losses context

    # ══════════════════════════════════════════════════════════════════
    # REPORT
    # ══════════════════════════════════════════════════════════════════

    print(f"Total lines: {total_lines:,}")
    print()

    # ── 1. Task Overview ─────────────────────────────────────────────
    print("=" * 80)
    print("1. TASK OVERVIEW")
    print("=" * 80)
    scenario_names = ["tiny", "small-linear", "medium", "medium-single-site", "medium-multi-site"]
    for tid in sorted(task_scenarios.keys()):
        sc = task_scenarios[tid]
        name = scenario_names[tid] if tid < len(scenario_names) else f"task_{tid}"
        total_actions = sum(task_actions[tid].values())
        success_actions = sum(v for k, v in task_actions[tid].items() if '_SUCCESS' in k)
        fail_actions = sum(v for k, v in task_actions[tid].items() if '_FAILURE' in k)
        sr_info = next((s for s in sr_reports if s['task'] == tid), None)
        sr_str = f"{sr_info['sr']*100:.0f}%" if sr_info else "N/A"

        print(f"\n  Task {tid}: {name}")
        print(f"    Hosts: {sc['hosts']}, Services: [{sc['services']}]")
        print(f"    Episodes completed: {task_episode_count.get(tid, '?')}")
        print(f"    Total actions: {total_actions:,} (success={success_actions:,}, fail={fail_actions:,})")
        print(f"    Success rate: {success_actions/max(total_actions,1)*100:.1f}%")
        print(f"    Connection errors: {connection_errors[tid]:,}")
        print(f"    Permission errors: {permission_errors[tid]:,}")
        print(f"    Unique successful action types: {sorted(task_unique_actions_success[tid])}")
        print(f"    Overall SR after task: {sr_str}")
        if sr_info:
            print(f"    Failed tasks: [{sr_info['failed']}]")
            print(f"    Total steps: {sr_info['total_steps']:,}")

    # ── 2. Action Distribution per Task ──────────────────────────────
    print()
    print("=" * 80)
    print("2. ACTION DISTRIBUTION (top 10 per task)")
    print("=" * 80)
    for tid in sorted(task_actions.keys()):
        acts = task_actions[tid]
        total = sum(acts.values())
        print(f"\n  Task {tid} ({total:,} total):")
        for k, v in sorted(acts.items(), key=lambda x: -x[1])[:10]:
            bar = "#" * int(v / total * 50)
            print(f"    {k:35s} {v:7,d} ({v/total*100:5.1f}%) {bar}")

    # ── 3. Compress Loss Progression ─────────────────────────────────
    print()
    print("=" * 80)
    print("3. KNOWLEDGE DISTILLATION (compress) LOSS PROGRESSION")
    print("=" * 80)
    for tid in sorted(compress_losses.keys()):
        data = compress_losses[tid]
        nan_marker = " *** NaN DETECTED ***" if tid in nan_tasks else ""
        print(f"\n  Task {tid} ({len(data)} checkpoints){nan_marker}:")
        print(f"    {'%':>5s}  {'total_loss':>12s}  {'ewc_loss':>12s}  {'kd_loss':>12s}  {'retro_loss':>12s}  {'trans_loss':>12s}")
        print(f"    {'─'*5}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}")
        # Sample at 10% intervals to avoid flooding output
        sampled = {}
        for pct, loss, ewc, kd, res, trans in data:
            bucket = (pct // 10) * 10
            if bucket not in sampled:
                sampled[bucket] = (pct, loss, ewc, kd, res, trans)
            # always keep last
            sampled['last'] = (pct, loss, ewc, kd, res, trans)
        for key in sorted(k for k in sampled if isinstance(k, int)):
            pct, loss, ewc, kd, res, trans = sampled[key]
            def fmt(v):
                try:
                    return f"{float(v):.6f}"
                except:
                    return v[:12]
            print(f"    {pct:4d}%  {fmt(loss):>12s}  {fmt(ewc):>12s}  {fmt(kd):>12s}  {fmt(res):>12s}  {fmt(trans):>12s}")
        if 'last' in sampled:
            pct, loss, ewc, kd, res, trans = sampled['last']
            def fmt(v):
                try:
                    return f"{float(v):.6f}"
                except:
                    return v[:12]
            print(f"    {pct:4d}%  {fmt(loss):>12s}  {fmt(ewc):>12s}  {fmt(kd):>12s}  {fmt(res):>12s}  {fmt(trans):>12s}  (last)")

    # ── 4. SR Progression ────────────────────────────────────────────
    print()
    print("=" * 80)
    print("4. SUCCESS RATE (SR) PROGRESSION AFTER EACH TASK")
    print("=" * 80)
    print(f"\n  {'Task':>6s}  {'Scenario':>20s}  {'SR':>8s}  {'Steps':>10s}  {'Failed Tasks':>20s}")
    print(f"  {'─'*6}  {'─'*20}  {'─'*8}  {'─'*10}  {'─'*20}")
    for sr in sr_reports:
        name = scenario_names[sr['task']] if sr['task'] < len(scenario_names) else f"task_{sr['task']}"
        print(f"  {sr['task']:6d}  {name:>20s}  {sr['sr']*100:7.1f}%  {sr['total_steps']:10,d}  [{sr['failed']}]")

    # ── 5. Diagnosis ─────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("5. DIAGNOSIS — ROOT CAUSE ANALYSIS")
    print("=" * 80)

    issues = []

    # Check NaN
    if nan_tasks:
        nan_task_details = []
        for t in sorted(nan_tasks):
            if t in task_scenarios:
                sc = task_scenarios[t]
                nan_task_details.append(f"Task {t} ({sc['hosts']} hosts)")
            else:
                nan_task_details.append(f"Task {t}")
        
        issues.append({
            'severity': 'CRITICAL',
            'title': 'NaN loss in Knowledge Distillation (compress)',
            'detail': (
                f"Tasks with NaN: {', '.join(nan_task_details)}. "
                "Once loss becomes NaN, ALL subsequent model weights are corrupted. "
                "KD, retrospection, EWC — all become ineffective. "
                "This corruption cascades to all subsequent tasks."
            ),
            'likely_cause': (
                "Explorer failed to learn the task adequately. "
                "Possible reasons: (1) insufficient episodes, (2) step_limit too low, "
                "(3) task too complex, (4) gradient explosion. "
                "Poor expert samples → KD loss diverges → NaN."
            ),
            'fix': (
                "1. Add gradient clipping in compress() to prevent NaN\n"
                "    2. Add NaN detection + skip logic in compress()\n"
                "    3. Increase episodes for complex scenarios\n"
                "    4. Increase step_limit to allow task completion\n"
                "    5. Verify single-task training works before CRL\n"
                "    6. Check learning rate and optimizer settings"
            ),
        })

    # Check action diversity
    for tid in sorted(task_actions.keys()):
        success_types = task_unique_actions_success[tid]
        if len(success_types) <= 2:
            issues.append({
                'severity': 'HIGH',
                'title': f'Task {tid}: Very low action diversity',
                'detail': (
                    f"Only {len(success_types)} successful action types: {sorted(success_types)}. "
                    "Agent is stuck repeating 1-2 actions without exploring the full action space."
                ),
                'likely_cause': (
                    "Policy converged to local minimum. Complex scenarios need more exploration. "
                    "The 16-action service-level space may not map well to all scenarios."
                ),
                'fix': "Increase entropy coefficient, add exploration bonus, or increase episodes.",
            })

    # Check connection errors ratio
    for tid in sorted(connection_errors.keys()):
        total_actions = sum(task_actions[tid].values())
        conn_ratio = connection_errors[tid] / max(total_actions, 1)
        if conn_ratio > 0.3:
            issues.append({
                'severity': 'HIGH',
                'title': f'Task {tid}: {conn_ratio*100:.0f}% actions are connection_error',
                'detail': (
                    f"{connection_errors[tid]:,} connection errors out of {total_actions:,} total. "
                    "Agent repeatedly attempts actions on unreachable hosts."
                ),
                'likely_cause': (
                    "The agent doesn't learn network topology (which hosts are reachable). "
                    "In larger networks, most hosts are initially unreachable. "
                    "Without subnet_scan first, all exploit/scan actions fail with connection_error."
                ),
                'fix': (
                    "Add step_limit increase, reward shaping for discovery, "
                    "or masking unreachable actions."
                ),
            })

    # Check episodes sufficient (only if SR is low or unavailable)
    for tid in sorted(task_scenarios.keys()):
        hosts = task_scenarios[tid]['hosts']
        episodes = task_episode_count.get(tid, 0)
        min_suggested = hosts * 100  # rough heuristic
        
        # Get SR for this task
        sr_info = next((s for s in sr_reports if s['task'] == tid), None)
        task_sr = sr_info['sr'] if sr_info else None
        
        # Only flag as issue if episodes are low AND (SR is low or unknown)
        if episodes < min_suggested and (task_sr is None or task_sr < 0.8):
            issues.append({
                'severity': 'MEDIUM',
                'title': f'Task {tid}: Insufficient episodes ({episodes} for {hosts} hosts)',
                'detail': (
                    f"Scenario has {hosts} hosts but only {episodes} episodes. "
                    f"Rough estimate: need ≥{min_suggested} episodes. "
                    f"Current SR: {task_sr*100:.0f}%" if task_sr else "SR data not available."
                ),
                'likely_cause': "Complex scenarios need more training time to converge.",
                'fix': f"Increase episodes to {min_suggested}+ for this scenario.",
            })

    for i, issue in enumerate(issues, 1):
        print(f"\n  [{issue['severity']}] #{i}: {issue['title']}")
        print(f"    Detail: {issue['detail']}")
        print(f"    Likely cause: {issue['likely_cause']}")
        print(f"    Fix: {issue['fix']}")

    # ── 6. Summary ───────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("6. SUMMARY")
    print("=" * 80)
    
    # Calculate overall statistics
    total_tasks = len(task_scenarios)
    tasks_with_sr = [s for s in sr_reports if s['sr'] > 0.0]
    avg_sr = sum(s['sr'] for s in sr_reports) / len(sr_reports) if sr_reports else 0.0
    
    print(f"\n  Total tasks trained: {total_tasks}")
    print(f"  Tasks with SR > 0: {len(tasks_with_sr)}/{len(sr_reports)}")
    print(f"  Average SR: {avg_sr*100:.1f}%")
    print(f"  Critical issues found: {len([i for i in issues if i['severity'] == 'CRITICAL'])}")
    print(f"  High priority issues: {len([i for i in issues if i['severity'] == 'HIGH'])}")
    
    if len(issues) == 0:
        print("\n  ✅ No major issues detected. Training appears healthy.")
    else:
        print(f"\n  ⚠️  {len(issues)} issue(s) detected. Review Section 5 for details.")
    
    # Performance bottleneck hints
    if nan_tasks:
        print("\n  🔴 NaN detected → All subsequent learning is corrupted")
        print("     → Check gradient clipping, learning rate, or add NaN guards")
    
    failing_tasks = [s for s in sr_reports if s['sr'] < 0.5]
    if failing_tasks and not nan_tasks:
        print("\n  📊 Low SR on some tasks (no NaN) → Possible causes:")
        print("     → Insufficient episodes for task complexity")
        print("     → Step limit too low for network size")
        print("     → Poor action space exploration")
        print("     → Reward shaping issues")
    
    if any(connection_errors[t] / max(sum(task_actions[t].values()), 1) > 0.3 for t in connection_errors):
        print("\n  🌐 High connection error rate → Agent not learning topology")
        print("     → Consider reward shaping for network discovery")
    
    print()

    return {
        'total_lines': total_lines,
        'sr_reports': sr_reports,
        'nan_tasks': list(nan_tasks),
        'issues': len(issues),
    }


if __name__ == "__main__":
    log_file = sys.argv[1] if len(sys.argv) > 1 else "outputs/benchmark/script-train_20260215_230235.log"
    result = analyze_log(log_file)
    print(f"\nAnalysis complete. Found {result['issues']} issues.")
