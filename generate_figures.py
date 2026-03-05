r"""
generate_figures.py  -  Publication-quality figures for PenSCRIPT paper.

Reads experiment outputs (JSON, CSV, TensorBoard event files) and produces
IEEE-formatted PDF figures ready for \includegraphics{}.

Usage:
    cd d:\NCKH\fusion\pentest
    python generate_figures.py

Output directory: outputs/figures/
"""

import json
import os
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

# ── Suppress TensorFlow/TensorBoard deprecation noise ──────────────────
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── Global style — IEEE column width (~3.5 in), double-column (~7.16 in)
matplotlib.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   7.5,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.03,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linewidth":    0.4,
    "axes.linewidth":    0.6,
    "lines.linewidth":   1.2,
    "lines.markersize":  4,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── Colour palette (colour-blind safe, IEEE-friendly) ──────────────────
COLORS = {
    "dual":     "#2166AC",   # steel blue
    "sim":      "#B2182B",   # brick red
    "scratch":  "#4DAF4A",   # green
    "tiny":     "#FF7F00",   # orange
    "small":    "#984EA3",   # purple
    "accent1":  "#E6AB02",   # gold
    "accent2":  "#66C2A5",   # teal
    "grey":     "#999999",
}

BETA_COLORS = sns.color_palette("viridis", 6)

BASE = Path("outputs")
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 42]
BETAS = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
TOPOLOGIES = ["tiny", "small-linear"]
TIERS = ["T1", "T2", "T3", "T4"]
TASKS = [f"{t}_{tier}_000" for t in TOPOLOGIES for tier in TIERS]


# ══════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_per_task_sr(data, agent="theta_dual"):
    """Return dict {task_name: sr} from phase4."""
    per_task = data["phase4"]["agents"][agent]["per_task"]
    return {t["task"]: t["sr"] for t in per_task}


def get_per_task_eta(data, agent="theta_dual"):
    """Return dict {task_name: eta} from phase4."""
    per_task = data["phase4"]["agents"][agent]["per_task"]
    return {t["task"]: t.get("step_efficiency") for t in per_task}


def get_overall_sr(data, agent="theta_dual"):
    return data["phase4"]["agents"][agent]["success_rate"]


def savefig(fig, name, formats=("pdf", "png")):
    """Save figure in multiple formats."""
    for fmt in formats:
        path = FIG_DIR / f"{name}.{fmt}"
        fig.savefig(path, format=fmt)
    plt.close(fig)
    print(f"  ✓ Saved: {name}")


# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 1: Learning Curves from TensorBoard
# ══════════════════════════════════════════════════════════════════════════

def fig1_learning_curves():
    """
    Training reward/success curves from TensorBoard for Phase 3.
    Overlay all seeds with mean ± std shaded.
    Separate subplot per topology stream.
    """
    print("\n[Fig 2] Learning Curves ...")
    try:
        from tbparse import SummaryReader
    except ImportError:
        print("  ✗ tbparse not installed — skipping learning curves.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8), sharey=True)

    for col, topo in enumerate(TOPOLOGIES):
        topo_label = "Tiny" if topo == "tiny" else "Small-linear"
        ax = axes[col]

        all_rewards = {}  # seed -> {step: reward}

        for seed in SEEDS:
            tb_dir = BASE / "multiseed" / f"seed_{seed}" / "tensorboard" / f"phase3_stream_{topo}"
            if not tb_dir.exists():
                continue

            try:
                reader = SummaryReader(str(tb_dir))
                df = reader.scalars
                # Try common tag names
                for tag_candidate in ["Train/EpisodeReward", "train/episode_reward",
                                      "Train/Episode Rewards", "episode_reward",
                                      "rollout/ep_rew_mean", "reward"]:
                    subset = df[df["tag"] == tag_candidate]
                    if len(subset) > 0:
                        break

                if len(subset) == 0:
                    # Fallback: use first scalar tag that looks like reward
                    reward_tags = [t for t in df["tag"].unique()
                                   if any(k in t.lower() for k in ["reward", "rew"])]
                    if reward_tags:
                        subset = df[df["tag"] == reward_tags[0]]
                    else:
                        # Use first available tag
                        if len(df) > 0:
                            first_tag = df["tag"].unique()[0]
                            subset = df[df["tag"] == first_tag]

                if len(subset) > 0:
                    all_rewards[seed] = subset[["step", "value"]].set_index("step")["value"]
            except Exception as e:
                print(f"  Warning: Could not read TB data for seed {seed}, {topo}: {e}")
                continue

        if not all_rewards:
            ax.text(0.5, 0.5, "No TensorBoard data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="grey")
            ax.set_title(f"({chr(97+col)}) {topo_label}", fontweight="bold")
            continue

        # Align all series to common step grid
        all_steps = sorted(set().union(*(r.index for r in all_rewards.values())))
        matrix = np.full((len(all_rewards), len(all_steps)), np.nan)
        for i, (seed, series) in enumerate(all_rewards.items()):
            for j, step in enumerate(all_steps):
                if step in series.index:
                    matrix[i, j] = series[step]

        # Forward-fill NaN values
        for i in range(matrix.shape[0]):
            last_val = np.nan
            for j in range(matrix.shape[1]):
                if np.isnan(matrix[i, j]):
                    matrix[i, j] = last_val
                else:
                    last_val = matrix[i, j]

        mean_curve = np.nanmean(matrix, axis=0)
        std_curve = np.nanstd(matrix, axis=0)

        # Smooth with rolling window
        window = max(1, len(all_steps) // 50)
        mean_smooth = pd.Series(mean_curve).rolling(window, min_periods=1).mean().values
        std_smooth = pd.Series(std_curve).rolling(window, min_periods=1).mean().values

        color = COLORS["tiny"] if topo == "tiny" else COLORS["small"]
        ax.plot(all_steps, mean_smooth, color=color, linewidth=1.4, label=f"Mean (n={len(all_rewards)})")
        ax.fill_between(all_steps,
                        mean_smooth - std_smooth,
                        mean_smooth + std_smooth,
                        alpha=0.2, color=color)

        # Draw tier boundaries (approximate: every 1/4 of episodes)
        n_steps = len(all_steps)
        for frac, tier_label in zip([0.25, 0.5, 0.75], ["T2", "T3", "T4"]):
            idx = int(frac * n_steps)
            if idx < n_steps:
                ax.axvline(all_steps[idx], color="grey", linestyle="--",
                           linewidth=0.6, alpha=0.5)
                ax.text(all_steps[idx], ax.get_ylim()[1] * 0.95, tier_label,
                        fontsize=7, color="grey", ha="center")

        ax.set_title(f"({chr(97+col)}) {topo_label}", fontweight="bold")
        ax.set_xlabel("Training Step")
        if col == 0:
            ax.set_ylabel("Episode Reward")
        ax.legend(loc="lower right", framealpha=0.8)

    fig.suptitle("Phase 3 Training Curves — Intra-Topology CRL", fontsize=10, fontweight="bold", y=1.02)
    fig.tight_layout()
    savefig(fig, "fig2_learning_curves")


# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 2: Per-Topology + Per-Tier SR with Error Bars (grouped bar)
# ══════════════════════════════════════════════════════════════════════════

def fig2_sr_by_topology_tier():
    """
    Grouped bar chart: SR of θ_dual vs θ_sim vs θ_scratch
    per topology, averaged across tiers, with error bars across seeds.
    """
    print("\n[Fig 3] SR by Topology and Agent ...")

    agents = ["theta_dual", "theta_sim_unified", "theta_pengym_scratch"]
    agent_labels = [r"$\theta_{\mathrm{dual}}$", r"$\theta_{\mathrm{sim}}$", r"$\theta_{\mathrm{scratch}}$"]
    agent_colors = [COLORS["dual"], COLORS["sim"], COLORS["scratch"]]

    # Collect per-topology SR for each agent across seeds
    data = {a: {t: [] for t in TOPOLOGIES} for a in agents}

    for seed in SEEDS:
        path = BASE / "multiseed" / f"seed_{seed}" / "strategy_c_results.json"
        if not path.exists():
            continue
        d = load_json(path)
        for agent in agents:
            pts = get_per_task_sr(d, agent)
            for topo in TOPOLOGIES:
                topo_tasks = [t for t in pts if t.startswith(topo)]
                if topo_tasks:
                    topo_sr = np.mean([pts[t] for t in topo_tasks])
                    data[agent][topo].append(topo_sr)

    # Build bar chart
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    x = np.arange(len(TOPOLOGIES))
    n_agents = len(agents)
    width = 0.22
    offsets = np.linspace(-(n_agents-1)*width/2, (n_agents-1)*width/2, n_agents)

    for i, (agent, label, color) in enumerate(zip(agents, agent_labels, agent_colors)):
        means = [np.mean(data[agent][t]) if data[agent][t] else 0 for t in TOPOLOGIES]
        stds  = [np.std(data[agent][t])  if len(data[agent][t]) > 1 else 0 for t in TOPOLOGIES]
        bars = ax.bar(x + offsets[i], means, width, yerr=stds,
                      label=label, color=color, edgecolor="white", linewidth=0.3,
                      capsize=3, error_kw={"linewidth": 0.8})
        # Value labels
        for bar, m in zip(bars, means):
            if m > 0.02:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                        f"{m:.2f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(["Tiny\n(3 hosts)", "Small-linear\n(8 hosts)"])
    ax.set_ylabel("Success Rate (SR)")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_title("Agent Performance by Topology", fontweight="bold")

    fig.tight_layout()
    savefig(fig, "fig3_sr_by_topology")


# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 3: β Ablation — SR vs Fisher Discount
# ══════════════════════════════════════════════════════════════════════════

def fig3_beta_ablation():
    """
    Line plot with markers: overall SR and per-topology SR vs β.
    Highlights the optimal region.
    """
    print("\n[Fig 4] Beta Ablation ...")

    beta_data = {"overall": [], "tiny": [], "small-linear": []}

    for beta in BETAS:
        if beta == 0.3:
            # beta_0.3 is a symlink to multiseed/seed_42
            path = BASE / "multiseed" / "seed_42" / "strategy_c_results.json"
        else:
            path = BASE / "ablation_beta" / f"beta_{beta}" / "strategy_c_results.json"

        if not path.exists():
            beta_data["overall"].append(np.nan)
            beta_data["tiny"].append(np.nan)
            beta_data["small-linear"].append(np.nan)
            continue

        d = load_json(path)
        sr_overall = get_overall_sr(d, "theta_dual")
        pts = get_per_task_sr(d, "theta_dual")

        tiny_sr = np.mean([pts[t] for t in pts if t.startswith("tiny")])
        sl_sr   = np.mean([pts[t] for t in pts if t.startswith("small-linear")])

        beta_data["overall"].append(sr_overall)
        beta_data["tiny"].append(tiny_sr)
        beta_data["small-linear"].append(sl_sr)

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    ax.plot(BETAS, beta_data["overall"], "o-", color=COLORS["dual"],
            linewidth=1.8, markersize=7, label="Overall SR", zorder=3)
    ax.plot(BETAS, beta_data["tiny"], "s--", color=COLORS["tiny"],
            linewidth=1.0, markersize=5, label="Tiny SR", alpha=0.8)
    ax.plot(BETAS, beta_data["small-linear"], "^--", color=COLORS["small"],
            linewidth=1.0, markersize=5, label="Small-linear SR", alpha=0.8)

    # Highlight β=0 (full elimination) with annotation
    ax.annotate(r"$\beta{=}0$ (full elim.)",
                xy=(0.0, beta_data["overall"][0]),
                xytext=(0.15, beta_data["overall"][0] - 0.12),
                fontsize=7, color=COLORS["sim"],
                arrowprops=dict(arrowstyle="->", color=COLORS["sim"], lw=0.8))

    # Shade the "any β>0" region
    ax.axvspan(0.05, 1.05, alpha=0.06, color=COLORS["dual"], label=r"$\beta > 0$ region")

    ax.set_xlabel(r"Fisher Discount $\beta$")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(BETAS)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=7)
    ax.set_title(r"Effect of Fisher Discount $\beta$ on SR", fontweight="bold")

    fig.tight_layout()
    savefig(fig, "fig4_beta_ablation")


# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 4: Per-Tier SR Heatmap (θ_dual, across seeds)
# ══════════════════════════════════════════════════════════════════════════

def fig4_tier_heatmap():
    """
    Heatmap: rows = tasks (topology × tier), columns = seeds.
    Cell color = SR (0 = red, 1 = blue). Shows per-task variability.
    """
    print("\n[Fig 5] Tier Heatmap ...")

    task_names = TASKS
    task_labels = [f"{'T' if t.startswith('tiny') else 'SL'}_{t.split('_')[1]}" for t in task_names]

    matrix = np.full((len(task_names), len(SEEDS)), np.nan)

    for j, seed in enumerate(SEEDS):
        path = BASE / "multiseed" / f"seed_{seed}" / "strategy_c_results.json"
        if not path.exists():
            continue
        d = load_json(path)
        pts = get_per_task_sr(d, "theta_dual")
        for i, task in enumerate(task_names):
            matrix[i, j] = pts.get(task, np.nan)

    # Add mean column
    mean_col = np.nanmean(matrix, axis=1, keepdims=True)
    matrix_ext = np.hstack([matrix, mean_col])
    col_labels = [f"S{s}" for s in SEEDS] + ["Mean"]

    fig, ax = plt.subplots(figsize=(3.5, 3.2))

    # Custom colormap: red → yellow → green
    cmap = sns.diverging_palette(10, 133, n=256, as_cmap=True)

    sns.heatmap(matrix_ext, ax=ax,
                annot=True, fmt=".2f", annot_kws={"size": 7},
                cmap=cmap, center=0.5, vmin=0, vmax=1,
                xticklabels=col_labels, yticklabels=task_labels,
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Success Rate", "shrink": 0.8})

    # Draw topology separator
    ax.axhline(4, color="black", linewidth=1.5)

    # Topology labels
    ax.text(-0.6, 2, "Tiny", ha="center", va="center", fontsize=8,
            fontweight="bold", rotation=90, transform=ax.transData)
    ax.text(-0.6, 6, "Small-\nlinear", ha="center", va="center", fontsize=7,
            fontweight="bold", rotation=90, transform=ax.transData)

    ax.set_title(r"$\theta_{\mathrm{dual}}$ SR by Task and Seed", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")

    fig.tight_layout()
    savefig(fig, "fig5_tier_heatmap")


# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 5: Forgetting Matrix Heatmap
# ══════════════════════════════════════════════════════════════════════════

def fig5_forgetting_heatmap():
    """
    Heatmap of average forgetting across seeds.
    Shows how much performance drops on task i after training on stream j.
    """
    print("\n[Fig 6] Forgetting Matrix ...")

    # Gather forgetting from all seeds
    all_F = []

    for seed in SEEDS:
        path = BASE / "multiseed" / f"seed_{seed}" / "strategy_c_results.json"
        if not path.exists():
            continue
        d = load_json(path)

        fm = d["phase4"].get("forgetting_matrix", {})
        f_matrix = fm.get("F_matrix", [])
        task_names = fm.get("task_names", TASKS)
        tier_names = fm.get("tier_names", ["stream_small-linear", "stream_tiny"])

        if f_matrix:
            arr = np.array(f_matrix, dtype=float)
            all_F.append(arr)

    if not all_F:
        print("  ✗ No forgetting data found.")
        return

    # Average across seeds (handle NaN)
    stacked = np.stack(all_F)
    mean_F = np.nanmean(stacked, axis=0)

    task_labels = [f"{'T' if t.startswith('tiny') else 'SL'}_{t.split('_')[1]}" for t in task_names]
    tier_labels = [tn.replace("stream_", "").replace("small-linear", "SL") for tn in tier_names]

    fig, ax = plt.subplots(figsize=(3.0, 3.2))

    # Diverging colormap: red for forgetting, blue for positive transfer
    cmap = sns.diverging_palette(10, 240, n=256, as_cmap=True)
    max_abs = np.nanmax(np.abs(mean_F[~np.isnan(mean_F)])) if np.any(~np.isnan(mean_F)) else 1.0

    mask = np.isnan(mean_F)
    sns.heatmap(mean_F, ax=ax,
                annot=True, fmt=".2f", annot_kws={"size": 7},
                cmap=cmap, center=0, vmin=-max_abs, vmax=max_abs,
                xticklabels=tier_labels, yticklabels=task_labels,
                mask=mask,
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "NR Change", "shrink": 0.8})

    # Separator
    ax.axhline(4, color="black", linewidth=1.5)

    ax.set_title("Forgetting Matrix\n(Mean over 5 Seeds)", fontweight="bold", fontsize=9)
    ax.set_xlabel("After Training Stream")
    ax.set_ylabel("")

    fig.tight_layout()
    savefig(fig, "fig6_forgetting_heatmap")


# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 6: Intra vs Cross Topology Comparison (Box Plot)
# ══════════════════════════════════════════════════════════════════════════

def fig6_intra_vs_cross():
    """
    Box/strip plot comparing θ_dual SR under intra-topology vs
    cross-topology CRL, with individual seed points.
    """
    print("\n[Fig 7] Intra vs Cross Topology ...")

    intra_sr = []
    cross_sr = []

    for seed in SEEDS:
        # Intra
        p1 = BASE / "multiseed" / f"seed_{seed}" / "strategy_c_results.json"
        if p1.exists():
            d = load_json(p1)
            intra_sr.append(get_overall_sr(d, "theta_dual"))

        # Cross
        p2 = BASE / "multiseed_cross" / f"seed_{seed}" / "strategy_c_results.json"
        if p2.exists():
            d = load_json(p2)
            cross_sr.append(get_overall_sr(d, "theta_dual"))

    fig, ax = plt.subplots(figsize=(2.8, 2.8))

    # Box + Strip
    positions = [0, 1]
    bp = ax.boxplot([intra_sr, cross_sr], positions=positions,
                    widths=0.5, patch_artist=True,
                    boxprops=dict(linewidth=0.8),
                    medianprops=dict(color="black", linewidth=1.2),
                    whiskerprops=dict(linewidth=0.8),
                    capprops=dict(linewidth=0.8),
                    flierprops=dict(markersize=3))

    bp["boxes"][0].set_facecolor(COLORS["dual"])
    bp["boxes"][0].set_alpha(0.4)
    bp["boxes"][1].set_facecolor(COLORS["sim"])
    bp["boxes"][1].set_alpha(0.4)

    # Scatter individual points
    jitter = 0.08
    for i, (vals, color) in enumerate([(intra_sr, COLORS["dual"]),
                                        (cross_sr, COLORS["sim"])]):
        x_pts = np.random.default_rng(42).normal(positions[i], jitter, len(vals))
        ax.scatter(x_pts, vals, color=color, s=30, zorder=5,
                   edgecolor="white", linewidth=0.5)

    # Means
    for i, vals in enumerate([intra_sr, cross_sr]):
        m = np.mean(vals)
        ax.plot(positions[i], m, "D", color="black", markersize=5, zorder=6)
        ax.text(positions[i] + 0.25, m, f"{m:.3f}", fontsize=7, va="center")

    # Statistical test
    if len(intra_sr) >= 2 and len(cross_sr) >= 2:
        from scipy.stats import ttest_ind
        t_stat, p_val = ttest_ind(intra_sr, cross_sr)
        sig = "n.s." if p_val > 0.05 else f"p={p_val:.3f}"
        ax.text(0.5, max(max(intra_sr), max(cross_sr)) + 0.08,
                sig, ha="center", fontsize=8, fontstyle="italic")
        # Bracket
        y_top = max(max(intra_sr), max(cross_sr)) + 0.05
        ax.plot([0, 0, 1, 1], [y_top-0.02, y_top, y_top, y_top-0.02],
                color="black", linewidth=0.8)

    ax.set_xticks(positions)
    ax.set_xticklabels(["Intra-topology\nCRL", "Cross-topology\nCRL"])
    ax.set_ylabel("Overall SR")
    ax.set_ylim(0, 1.2)
    ax.set_title("Intra vs Cross-Topology CRL", fontweight="bold")

    fig.tight_layout()
    savefig(fig, "fig7_intra_vs_cross")


# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 7: CVE Tier Difficulty — SR Decay per Tier
# ══════════════════════════════════════════════════════════════════════════

def fig7_tier_sr_decay():
    """
    Line plot: SR vs CVE tier (T1→T4), separate lines per topology.
    Mean ± std across seeds. Shows curriculum effect.
    """
    print("\n[Fig 8] Tier SR Decay (Curriculum) ...")

    tier_data = {topo: {tier: [] for tier in TIERS} for topo in TOPOLOGIES}

    for seed in SEEDS:
        path = BASE / "multiseed" / f"seed_{seed}" / "strategy_c_results.json"
        if not path.exists():
            continue
        d = load_json(path)
        pts = get_per_task_sr(d, "theta_dual")
        for topo in TOPOLOGIES:
            for tier in TIERS:
                task = f"{topo}_{tier}_000"
                if task in pts:
                    tier_data[topo][tier].append(pts[task])

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    topo_styles = [
        ("tiny",         COLORS["tiny"],  "o-",  "Tiny (3 hosts)"),
        ("small-linear", COLORS["small"], "s--", "Small-linear (8 hosts)"),
    ]

    x = np.arange(len(TIERS))
    for topo, color, style, label in topo_styles:
        means = [np.mean(tier_data[topo][t]) if tier_data[topo][t] else 0 for t in TIERS]
        stds  = [np.std(tier_data[topo][t])  if len(tier_data[topo][t]) > 1 else 0 for t in TIERS]
        marker = style[0]
        linestyle = "-" if "--" not in style else "--"
        ax.errorbar(x, means, yerr=stds, fmt=f"{marker}{linestyle}",
                    color=color, capsize=4, capthick=0.8,
                    linewidth=1.5, markersize=7, label=label)

    # Difficulty shading
    for i, (tier, diff) in enumerate(zip(TIERS, ["Easy", "Medium", "Hard", "Expert"])):
        alpha = 0.03 + i * 0.04
        ax.axvspan(i - 0.4, i + 0.4, alpha=alpha, color="red")
        ax.text(i, -0.08, diff, ha="center", fontsize=6.5, color="grey", fontstyle="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(TIERS)
    ax.set_xlabel("CVE Difficulty Tier")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(-0.15, 1.15)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_title("SR Decay Across CVE Difficulty Tiers", fontweight="bold")

    fig.tight_layout()
    savefig(fig, "fig8_tier_sr_decay")


# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 8: Backward Transfer (BT_SR) Distribution
# ══════════════════════════════════════════════════════════════════════════

def fig8_backward_transfer():
    """
    Bar + error: BT_SR per seed, with mean line and 0-reference.
    Shows catastrophic forgetting severity.
    """
    print("\n[Fig 9] Backward Transfer ...")

    bt_values = []
    for seed in SEEDS:
        path = BASE / "multiseed" / f"seed_{seed}" / "strategy_c_results.json"
        if not path.exists():
            continue
        d = load_json(path)
        bt = d["phase4"]["transfer_metrics"]["BT_SR"]
        bt_values.append(bt)

    if not bt_values:
        print("  ✗ No BT data.")
        return

    fig, ax = plt.subplots(figsize=(3.5, 2.4))

    x = np.arange(len(bt_values))
    seed_labels = [f"Seed {s}" for s in SEEDS[:len(bt_values)]]

    bars = ax.bar(x, bt_values, width=0.6,
                  color=[COLORS["sim"] if v < 0 else COLORS["accent2"] for v in bt_values],
                  edgecolor="white", linewidth=0.3)

    # Value labels
    for bar, v in zip(bars, bt_values):
        y_pos = v - 0.05 if v < 0 else v + 0.02
        ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                f"{v:.3f}", ha="center", va="top" if v < 0 else "bottom",
                fontsize=7, fontweight="bold")

    # Mean line
    mean_bt = np.mean(bt_values)
    ax.axhline(mean_bt, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(len(bt_values) - 0.5, mean_bt + 0.05,
            f"Mean = {mean_bt:.3f}", fontsize=7, ha="right")

    # Zero reference
    ax.axhline(0, color="grey", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(seed_labels, fontsize=7)
    ax.set_ylabel(r"$\mathrm{BT}_{\mathrm{SR}}$")
    ax.set_title("Backward Transfer on SCRIPT Tasks\n(Catastrophic Forgetting)", fontweight="bold", fontsize=9)
    ax.set_ylim(min(bt_values) - 0.15, 0.2)

    fig.tight_layout()
    savefig(fig, "fig9_backward_transfer")


# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 9: Ablation Summary — EWC vs Fine-tune vs Canonicalization
# ══════════════════════════════════════════════════════════════════════════

def fig9_ablation_summary():
    """
    Grouped horizontal bar chart comparing ablation conditions.
    Conditions: Full pipeline | No EWC (fine-tune only) | β=0 (full eliminate)
    """
    print("\n[Fig 10] Ablation Summary ...")

    conditions = []

    # Full pipeline (mean of multiseed)
    full_srs = []
    for seed in SEEDS:
        p = BASE / "multiseed" / f"seed_{seed}" / "strategy_c_results.json"
        if p.exists():
            d = load_json(p)
            full_srs.append(get_overall_sr(d, "theta_dual"))
    if full_srs:
        conditions.append(("Full Pipeline\n" + r"($\beta=0.3$, EWC)",
                          np.mean(full_srs), np.std(full_srs)))

    # Fine-tune only (no EWC)
    p_ft = BASE / "ablation_crl" / "finetune_only" / "strategy_c_results.json"
    if p_ft.exists():
        d = load_json(p_ft)
        conditions.append(("Fine-tune Only\n(No EWC)",
                          get_overall_sr(d, "theta_dual"), 0))

    # β = 0 (full elimination)
    p_b0 = BASE / "ablation_beta" / "beta_0.0" / "strategy_c_results.json"
    if p_b0.exists():
        d = load_json(p_b0)
        conditions.append((r"$\beta=0$" + "\n(Full Elim.)",
                          get_overall_sr(d, "theta_dual"), 0))

    # β = 1.0 (full Fisher)
    p_b1 = BASE / "ablation_beta" / "beta_1.0" / "strategy_c_results.json"
    if p_b1.exists():
        d = load_json(p_b1)
        conditions.append((r"$\beta=1.0$" + "\n(Full Fisher)",
                          get_overall_sr(d, "theta_dual"), 0))

    if not conditions:
        print("  ✗ No ablation data found.")
        return

    fig, ax = plt.subplots(figsize=(3.5, 2.4))

    labels, means, stds = zip(*conditions)
    y = np.arange(len(conditions))
    colors_list = [COLORS["dual"], COLORS["accent2"], COLORS["sim"], COLORS["accent1"]]

    bars = ax.barh(y, means, xerr=stds, height=0.55,
                   color=colors_list[:len(conditions)],
                   edgecolor="white", linewidth=0.3,
                   capsize=3, error_kw={"linewidth": 0.8})

    # Value labels
    for bar, m in zip(bars, means):
        ax.text(m + 0.02, bar.get_y() + bar.get_height()/2,
                f"{m:.3f}", ha="left", va="center", fontsize=7.5, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Overall Success Rate")
    ax.set_xlim(0, max(means) + 0.15)
    ax.set_title("Ablation Summary", fontweight="bold")
    ax.invert_yaxis()

    fig.tight_layout()
    savefig(fig, "fig10_ablation_summary")


# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 10: Radar/Spider Chart — Multi-dimensional Performance View
# ══════════════════════════════════════════════════════════════════════════

def fig10_radar_chart():
    """
    Radar chart comparing θ_dual vs θ_sim across multiple metrics:
    SR_tiny, SR_small, Step Efficiency, Forward Transfer, (1-|BT|).
    """
    print("\n[Fig 11] Radar Chart ...")

    # Collect mean metrics across seeds
    metrics_dual = {"SR Tiny": [], "SR Small-Lin": [], "Step Eff.": [],
                    "Fwd Transfer": [], "1−|BT|": []}
    metrics_sim  = {"SR Tiny": [], "SR Small-Lin": [], "Step Eff.": [],
                    "Fwd Transfer": [], "1−|BT|": []}

    for seed in SEEDS:
        p = BASE / "multiseed" / f"seed_{seed}" / "strategy_c_results.json"
        if not p.exists():
            continue
        d = load_json(p)

        for agent, mdict in [("theta_dual", metrics_dual),
                              ("theta_sim_unified", metrics_sim)]:
            pts = get_per_task_sr(d, agent)
            tiny_sr = np.mean([pts[t] for t in pts if t.startswith("tiny")])
            sl_sr   = np.mean([pts[t] for t in pts if t.startswith("small-linear")])
            overall_eta = d["phase4"]["agents"][agent].get("step_efficiency") or 0

            ft = d["phase4"]["transfer_metrics"]["FT_SR"]
            bt = d["phase4"]["transfer_metrics"]["BT_SR"]

            mdict["SR Tiny"].append(tiny_sr)
            mdict["SR Small-Lin"].append(sl_sr)
            mdict["Step Eff."].append(min(overall_eta, 1.0))
            mdict["Fwd Transfer"].append(max(ft, 0))
            mdict["1−|BT|"].append(max(1 - abs(bt), 0))

    if not metrics_dual["SR Tiny"]:
        print("  ✗ No data for radar.")
        return

    categories = list(metrics_dual.keys())
    N = len(categories)

    dual_vals = [np.mean(metrics_dual[k]) for k in categories]
    sim_vals  = [np.mean(metrics_sim[k])  for k in categories]

    # Close the polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    dual_vals += dual_vals[:1]
    sim_vals  += sim_vals[:1]
    angles    += angles[:1]

    fig, ax = plt.subplots(figsize=(3.2, 3.2), subplot_kw=dict(polar=True))

    ax.fill(angles, dual_vals, alpha=0.15, color=COLORS["dual"])
    ax.plot(angles, dual_vals, "o-", color=COLORS["dual"], linewidth=1.5,
            markersize=5, label=r"$\theta_{\mathrm{dual}}$")

    ax.fill(angles, sim_vals, alpha=0.10, color=COLORS["sim"])
    ax.plot(angles, sim_vals, "s--", color=COLORS["sim"], linewidth=1.2,
            markersize=4, label=r"$\theta_{\mathrm{sim}}$")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=7.5)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=6)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), framealpha=0.9)
    ax.set_title("Multi-Metric Performance Profile", fontweight="bold", pad=20, fontsize=9)

    fig.tight_layout()
    savefig(fig, "fig11_radar_chart")


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  PenSCRIPT — Generating Publication Figures")
    print("=" * 60)

    # Verify we're in the right directory
    if not (BASE / "multiseed").exists():
        print(f"\n✗ Cannot find {BASE / 'multiseed'}.")
        print("  Please run this script from the 'pentest' directory:")
        print("    cd d:\\NCKH\\fusion\\pentest")
        print("    python generate_figures.py")
        sys.exit(1)

    generators = [
        ("Fig 2", "Learning Curves (Phase 3 Training)",   fig1_learning_curves),
        ("Fig 3", "SR by Topology & Agent",               fig2_sr_by_topology_tier),
        ("Fig 4", "Fisher Discount β Ablation",           fig3_beta_ablation),
        ("Fig 5", "Per-Task SR Heatmap",                  fig4_tier_heatmap),
        ("Fig 6", "Forgetting Matrix",                    fig5_forgetting_heatmap),
        ("Fig 7", "Intra vs Cross-Topology CRL",          fig6_intra_vs_cross),
        ("Fig 8", "SR Decay Across CVE Tiers",            fig7_tier_sr_decay),
        ("Fig 9", "Backward Transfer Analysis",           fig8_backward_transfer),
        ("Fig 10", "Ablation Summary",                    fig9_ablation_summary),
        ("Fig 11", "Radar Performance Profile",           fig10_radar_chart),
    ]

    success = 0
    for fig_id, desc, func in generators:
        try:
            func()
            success += 1
        except Exception as e:
            print(f"\n  ✗ {fig_id} ({desc}) failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"  Done: {success}/{len(generators)} figures generated.")
    print(f"  Output directory: {FIG_DIR.resolve()}")
    print(f"  Files: {', '.join(f.name for f in sorted(FIG_DIR.glob('*.pdf')))}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
