#!/usr/bin/env python3
"""Comprehensive analysis of Phase 1 + Phase 2 experimental results."""
import json
import numpy as np
import os

SEEDS = [0, 1, 2, 3, 42]
TASKS = [
    'tiny_T1_000', 'tiny_T2_000', 'tiny_T3_000', 'tiny_T4_000',
    'small-linear_T1_000', 'small-linear_T2_000', 'small-linear_T3_000', 'small-linear_T4_000'
]

def load_json(path):
    with open(path) as f:
        return json.load(f)

def get_per_task_sr(data, agent='theta_dual'):
    result = {}
    for pt in data['phase4']['agents'][agent]['per_task']:
        result[pt['task']] = pt['sr']
    return result

def print_sep(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

# ===== LOAD DATA =====
intra = {s: load_json(f'outputs/multiseed/seed_{s}/strategy_c_results.json') for s in SEEDS}
cross = {s: load_json(f'outputs/multiseed_cross/seed_{s}/strategy_c_results.json') for s in SEEDS}

print_sep("1. INTRA-TOPOLOGY (multiseed/) - theta_dual")

# Per-task SR
print("\n  Per-task SR across seeds:")
print(f"  {'Task':28s} {'Mean':>6s} {'Std':>6s}  Seeds")
for t in TASKS:
    srs = [get_per_task_sr(intra[s])[t] for s in SEEDS]
    print(f"  {t:28s} {np.mean(srs):6.3f} {np.std(srs):6.3f}  {srs}")

# Topology-level aggregation
print("\n  Topology-level SR:")
for topo in ['tiny', 'small-linear']:
    per_seed = []
    for s in SEEDS:
        ptsr = get_per_task_sr(intra[s])
        topo_srs = [v for k, v in ptsr.items() if k.startswith(topo)]
        per_seed.append(np.mean(topo_srs))
    print(f"  {topo:20s}: {np.mean(per_seed):.3f} +/- {np.std(per_seed):.3f}  {[f'{x:.3f}' for x in per_seed]}")

# Overall
overall_intra = [intra[s]['phase4']['agents']['theta_dual']['success_rate'] for s in SEEDS]
print(f"\n  Overall SR: {np.mean(overall_intra):.3f} +/- {np.std(overall_intra):.3f}")

# FT, BT
ft_intra = [intra[s]['phase4']['transfer_metrics']['FT_SR'] for s in SEEDS]
bt_intra = [intra[s]['phase4']['transfer_metrics']['BT_SR'] for s in SEEDS]
se_intra = [intra[s]['phase4']['agents']['theta_dual']['step_efficiency'] for s in SEEDS]
print(f"  FT_SR: {np.mean(ft_intra):.3f} +/- {np.std(ft_intra):.3f}")
print(f"  BT_SR: {np.mean(bt_intra):.3f} +/- {np.std(bt_intra):.3f}  {bt_intra}")
print(f"  Step eff: {np.mean(se_intra):.3f} +/- {np.std(se_intra):.3f}")

# Scratch
scratch_intra = [intra[s]['phase4']['agents']['theta_pengym_scratch']['success_rate'] for s in SEEDS]
print(f"  Scratch SR (all seeds): {scratch_intra}")

# Sim zero-shot
sim_intra = [intra[s]['phase4']['agents']['theta_sim_unified']['success_rate'] for s in SEEDS]
print(f"  Sim zero-shot SR: {np.mean(sim_intra):.3f} +/- {np.std(sim_intra):.3f}  {sim_intra}")

# Per-stream
print("\n  Per-stream own-topology SR:")
for topo in ['tiny', 'small-linear']:
    own = [intra[s]['phase4']['per_stream'][topo]['own_topology']['success_rate'] for s in SEEDS]
    print(f"  {topo:20s}: {np.mean(own):.3f} +/- {np.std(own):.3f}  {own}")

# Phase1
p1 = [intra[s]['phase1']['final_sr'] for s in SEEDS]
print(f"\n  Phase1 sim SR: {np.mean(p1):.3f} +/- {np.std(p1):.3f}")

print_sep("2. CROSS-TOPOLOGY (multiseed_cross/) - theta_dual")

print("\n  Per-task SR across seeds:")
print(f"  {'Task':28s} {'Mean':>6s} {'Std':>6s}  Seeds")
for t in TASKS:
    srs = [get_per_task_sr(cross[s])[t] for s in SEEDS]
    print(f"  {t:28s} {np.mean(srs):6.3f} {np.std(srs):6.3f}  {srs}")

# Topology-level
print("\n  Topology-level SR:")
for topo in ['tiny', 'small-linear']:
    per_seed = []
    for s in SEEDS:
        ptsr = get_per_task_sr(cross[s])
        topo_srs = [v for k, v in ptsr.items() if k.startswith(topo)]
        per_seed.append(np.mean(topo_srs))
    print(f"  {topo:20s}: {np.mean(per_seed):.3f} +/- {np.std(per_seed):.3f}  {[f'{x:.3f}' for x in per_seed]}")

overall_cross = [cross[s]['phase4']['agents']['theta_dual']['success_rate'] for s in SEEDS]
print(f"\n  Overall SR: {np.mean(overall_cross):.3f} +/- {np.std(overall_cross):.3f}")

bt_cross = [cross[s]['phase4']['transfer_metrics']['BT_SR'] for s in SEEDS]
se_cross = [cross[s]['phase4']['agents']['theta_dual']['step_efficiency'] for s in SEEDS]
print(f"  BT_SR: {np.mean(bt_cross):.3f} +/- {np.std(bt_cross):.3f}  {bt_cross}")
print(f"  Step eff: {np.mean(se_cross):.3f} +/- {np.std(se_cross):.3f}")

print_sep("3. INTRA vs CROSS STATISTICAL COMPARISON")

try:
    from scipy import stats as st
    t_stat, p_val = st.ttest_ind(overall_intra, overall_cross)
    better = "INTRA" if np.mean(overall_intra) > np.mean(overall_cross) else "CROSS"
    print(f"\n  Intra mean SR: {np.mean(overall_intra):.3f}")
    print(f"  Cross mean SR: {np.mean(overall_cross):.3f}")
    print(f"  t-test: t={t_stat:.3f}, p={p_val:.3f}")
    if p_val < 0.05:
        print(f"  Result: {better} significantly better (p<0.05)")
    else:
        print(f"  Result: NO significant difference (p={p_val:.3f})")

    # Per-topology comparison
    print("\n  Per-topology comparison:")
    for topo in ['tiny', 'small-linear']:
        intra_topo = []
        cross_topo = []
        for s in SEEDS:
            i_ptsr = get_per_task_sr(intra[s])
            c_ptsr = get_per_task_sr(cross[s])
            intra_topo.append(np.mean([v for k, v in i_ptsr.items() if k.startswith(topo)]))
            cross_topo.append(np.mean([v for k, v in c_ptsr.items() if k.startswith(topo)]))
        t2, p2 = st.ttest_ind(intra_topo, cross_topo)
        print(f"  {topo}: intra={np.mean(intra_topo):.3f}, cross={np.mean(cross_topo):.3f}, p={p2:.3f}")

    # FT_SR significance test: FT_SR > 0?
    print("\n  One-sample t-test: FT_SR > 0 (intra)?")
    t_ft, p_ft = st.ttest_1samp(ft_intra, 0)
    p_one = p_ft / 2  # one-tailed
    print(f"  FT_SR mean={np.mean(ft_intra):.3f}, t={t_ft:.3f}, p(one-tailed)={p_one:.4f}")
    if p_one < 0.05:
        print(f"  CONFIRMED: FT_SR > 0 with p={p_one:.4f} < 0.05")
    else:
        print(f"  NOT CONFIRMED: p={p_one:.4f}")
except ImportError:
    print("  scipy not available for statistical tests")

print_sep("4. ABLATION: Fisher Discount (beta)")

betas = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
print(f"\n  {'Beta':>5s} {'dual_SR':>8s} {'FT_SR':>8s} {'StepEff':>8s} {'BT_SR':>8s}")
for b in betas:
    if b == 0.3:
        d = intra[42]
    else:
        d = load_json(f'outputs/ablation_beta/beta_{b}/strategy_c_results.json')
    dual_sr = d['phase4']['agents']['theta_dual']['success_rate']
    se_val = d['phase4']['agents']['theta_dual']['step_efficiency']
    ft_val = d['phase4']['transfer_metrics']['FT_SR']
    bt_val = d['phase4']['transfer_metrics']['BT_SR']
    print(f"  {b:5.1f} {dual_sr:8.3f} {ft_val:8.3f} {se_val:8.3f} {bt_val:8.3f}")

print_sep("5. ABLATION: Canonicalization")

nc = load_json('outputs/ablation_canon/no_canon/strategy_c_results.json')
print(f"\n  With canon (seed_42):    SR={intra[42]['phase4']['agents']['theta_dual']['success_rate']:.3f}")
print(f"  Without canon (no_canon): SR={nc['phase4']['agents']['theta_dual']['success_rate']:.3f}")
print(f"  Difference: {intra[42]['phase4']['agents']['theta_dual']['success_rate'] - nc['phase4']['agents']['theta_dual']['success_rate']:.3f}")

# Per-task comparison
print("\n  Per-task comparison (canon vs no-canon):")
nc_ptsr = get_per_task_sr(nc)
c_ptsr = get_per_task_sr(intra[42])
for t in TASKS:
    print(f"  {t:28s}: canon={c_ptsr[t]:.2f}, no_canon={nc_ptsr[t]:.2f}")

print_sep("6. ABLATION: CRL (EWC) vs Finetune-only")

ft_only = load_json('outputs/ablation_crl/finetune_only/strategy_c_results.json')
print(f"\n  Full CRL (seed_42):   dual_SR={intra[42]['phase4']['agents']['theta_dual']['success_rate']:.3f}")
print(f"  Finetune-only:        dual_SR={ft_only['phase4']['agents']['theta_dual']['success_rate']:.3f}")
print(f"  Finetune-only FT_SR: {ft_only['phase4']['transfer_metrics']['FT_SR']:.3f}")

# Important: scratch in finetune_only
scratch_sr_ft = ft_only['phase4']['agents']['theta_pengym_scratch']['success_rate']
scratch_final = ft_only.get('scratch', {}).get('final_sr', 'N/A')
print(f"\n  Finetune-only scratch eval SR: {scratch_sr_ft:.3f}")
print(f"  Finetune-only scratch train final_sr: {scratch_final}")

print("\n  Finetune-only scratch per-task:")
for pt in ft_only['phase4']['agents']['theta_pengym_scratch']['per_task']:
    print(f"    {pt['task']:28s}: sr={pt['sr']:.2f}")

print("\n  Finetune-only dual per-task:")
for pt in ft_only['phase4']['agents']['theta_dual']['per_task']:
    print(f"    {pt['task']:28s}: sr={pt['sr']:.2f}")

print_sep("7. BENCHMARK BASELINES (tiny)")

bench = load_json('outputs/benchmark/baseline_results.json')
for name, data in bench.get('tiny', {}).items():
    print(f"  {name:20s}: SR={data['success_rate']:.1f}, avg_steps={data['avg_steps']:.1f}")

print_sep("8. CRITICAL FINDINGS SUMMARY")

print("""
  [CRITICAL] T3 BUG: tiny_T3_000 has access=user while T1/T2/T4 have access=root
    - This makes T3 require exploit(p=0.6) + privesc(p=0.6-0.8) = compound p~0.36-0.48
    - T3 SR = 0%% across ALL 10 seed-runs (5 intra + 5 cross)
    - Paper outline claimed bug was fixed but compiled scenario still has access=user

  [CRITICAL] Overall SR much lower than paper outline claims:
    - Paper outline claims SR=100%% on tiny and small-linear
    - Actual intra-topology mean: %.3f +/- %.3f
    - Actual cross-topology mean: %.3f +/- %.3f

  [CRITICAL] Intra vs Cross topology comparison:
    - Cross-topology is NOT worse than intra-topology (p=%.3f)
    - Death spiral claim from paper outline not supported by multi-seed data

  [IMPORTANT] Benchmark shows ALL agents (incl. random) achieve 100%% on tiny
    - This means tiny SR is not meaningful for demonstrating transfer

  [IMPORTANT] BT_SR predominantly -1.0:
    - Agent forgets sim knowledge almost completely after PenGym training

  [IMPORTANT] Finetune-only scratch achieves SR=0.875 (!):
    - This is an anomaly vs other experiments where scratch=0%%
    - Needs investigation: possibly different training configuration
""" % (
    np.mean(overall_intra), np.std(overall_intra),
    np.mean(overall_cross), np.std(overall_cross),
    p_val if 'p_val' in dir() else -1
))
