"""MetricStore — Structured storage for continual learning metrics.

Provides three classes:

- **MetricStore**: Collect and persist evaluation metrics across
  checkpoints as a single JSON file.
- **FZComputer**: Export Forgetting / Zero-Shot Transfer matrices
  to CSV and produce human-readable summaries.
- **CECurveGenerator**: Extract Continual Evaluation curves from
  MetricStore data and export as CSV for downstream plotting.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Dict


# =====================================================================
# MetricStore (Improvement E)
# =====================================================================

class MetricStore:
    """Collect and persist evaluation metrics across checkpoints."""

    def __init__(self, seed: int, output_dir: str):
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.data: Dict[str, Any] = {
            "metadata": {"seed": seed},
            "checkpoints": {},
            "training_curves": {},
            "forgetting": {},
            "transfer": {},
        }

    # ── Population ───────────────────────────────────────────────────

    def add_checkpoint(self, name: str, eval_result: Dict[str, Any]) -> None:
        """Store per-task metrics from an ``evaluate_agent()`` result."""
        self.data["checkpoints"][name] = {
            t["task"]: {
                "sr": t.get("sr"),
                "nr": t.get("normalized_reward"),
                "eta": t.get("step_efficiency"),
            }
            for t in eval_result.get("per_task", [])
        }

    def add_training_curve(
        self, task_name: str, episode_rewards: list, ttt: int,
    ) -> None:
        """Store per-task training dynamics."""
        self.data["training_curves"][task_name] = {
            "episode_rewards": episode_rewards,
            "ttt": ttt,
        }

    def set_forgetting(self, fz_result: Dict[str, Any]) -> None:
        """Store Forgetting / Zero-Shot Transfer matrices."""
        self.data["forgetting"] = fz_result

    def set_transfer(self, metrics: Dict[str, Any]) -> None:
        """Store aggregate forward/backward transfer metrics."""
        self.data["transfer"] = metrics

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, filename: str = "metric_store.json") -> None:
        path = self.output_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.data, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "MetricStore":
        with open(path) as f:
            data = json.load(f)
        store = cls(
            seed=data["metadata"]["seed"],
            output_dir=str(Path(path).parent),
        )
        store.data = data
        return store


# =====================================================================
# FZComputer (Improvement F)
# =====================================================================

class FZComputer:
    """Export Forgetting / Transfer matrices to CSV with summary."""

    @staticmethod
    def to_csv(fz_result: Dict[str, Any]) -> str:
        """Convert F matrix + Z vector to a CSV string."""
        output = io.StringIO()
        writer = csv.writer(output)

        task_names = fz_result["task_names"]
        tier_names = fz_result["tier_names"]
        F = fz_result["F_matrix"]

        # ── F matrix ──
        writer.writerow(["task \\ after_tier"] + tier_names)
        for i, task in enumerate(task_names):
            row = [task] + [
                f"{F[i][j]:.4f}"
                if F[i][j] is not None
                and not (isinstance(F[i][j], float) and F[i][j] != F[i][j])
                else ""
                for j in range(len(tier_names))
            ]
            writer.writerow(row)

        # ── Z vector ──
        writer.writerow([])
        writer.writerow(["task", "zero_shot_transfer"])
        Z = fz_result["Z_vector"]
        for i, task in enumerate(task_names):
            z_val = Z[i]
            writer.writerow([
                task,
                f"{z_val:.4f}" if z_val is not None and z_val == z_val else "",
            ])

        return output.getvalue()

    @staticmethod
    def print_summary(fz_result: Dict[str, Any]) -> str:
        """Format F/Z summary as readable text."""
        s = fz_result.get("summary", {})
        lines = [
            "=== Forgetting / Transfer Summary ===",
            f"  Mean forgetting (F):         {s.get('mean_forgetting', 'N/A')}",
            f"  Max forgetting (F):          {s.get('max_forgetting', 'N/A')}",
            f"  Mean zero-shot transfer (Z): {s.get('mean_zero_shot_transfer', 'N/A')}",
            f"  Tasks with positive Z:       {s.get('tasks_with_positive_transfer', 'N/A')}",
        ]
        return "\n".join(lines)

    @staticmethod
    def save_csv(fz_result: Dict[str, Any], path: str) -> None:
        csv_content = FZComputer.to_csv(fz_result)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            f.write(csv_content)


# =====================================================================
# CECurveGenerator (Improvement G)
# =====================================================================

class CECurveGenerator:
    """Generate Continual Evaluation curves from MetricStore data."""

    @staticmethod
    def extract_curves(store: MetricStore) -> Dict[str, Dict[str, list]]:
        """Extract per-task metric curves across checkpoints.

        Returns
        -------
        dict mapping metric_name → {task_name: [(checkpoint_name, value), …]}
        """
        checkpoints = store.data.get("checkpoints", {})
        ckpt_names = sorted(checkpoints.keys())

        curves: Dict[str, Dict[str, list]] = {"sr": {}, "nr": {}, "eta": {}}
        all_tasks: set = set()
        for ckpt in ckpt_names:
            all_tasks.update(checkpoints[ckpt].keys())

        for task in sorted(all_tasks):
            for metric in curves:
                curves[metric][task] = [
                    (ckpt, checkpoints.get(ckpt, {}).get(task, {}).get(metric))
                    for ckpt in ckpt_names
                ]

        return curves

    @staticmethod
    def to_csv(curves: Dict[str, Dict[str, list]], metric: str = "nr") -> str:
        """Export CE curves for one metric as CSV (tasks × checkpoints)."""
        output = io.StringIO()
        writer = csv.writer(output)

        data = curves.get(metric, {})
        if not data:
            return ""

        first_task_values = next(iter(data.values()))
        ckpt_names = [c[0] for c in first_task_values]

        writer.writerow(["task"] + ckpt_names)
        for task, values in sorted(data.items()):
            row = [task] + [
                f"{v:.4f}" if v is not None else ""
                for _, v in values
            ]
            writer.writerow(row)

        return output.getvalue()
