# Cải Tiến Hệ Thống Đánh Giá — Đặc Tả Triển Khai

> **Ngày:** 2026-02-25  
> **Branch:** `strC_1`  
> **Phạm vi:** 7 cải tiến (A–G), sắp xếp theo ưu tiên P0→P1.

---

## Tổng quan

Hệ thống đánh giá hiện tại đã có bộ ba metric cốt lõi (SR K=20, NR, η), policy-level metrics ($D_{KL}$, $\Delta_F$), FT/BT đa chiều, và fresh adapter isolation. Các cải tiến sau đây bổ sung **continual learning diagnostics** chuẩn CORA mà không thay đổi training loop.

| ID  | Cải tiến                              | File cần sửa                           | LOC ước tính |
| --- | ------------------------------------- | -------------------------------------- | ------------ |
| A   | Tier boundary checkpoints             | `dual_trainer.py`                      | ~15          |
| B   | Ma trận $\mathcal{F}$ / $\mathcal{Z}$ | `strategy_c_eval.py`                   | ~60          |
| C   | TTT + AUC learning-speed transfer     | `strategy_c_eval.py`                   | ~35          |
| D   | Train/Heldout evaluation split        | `dual_trainer.py`, `run_strategy_c.py` | ~35          |
| E   | MetricStore (structured JSON)         | `src/evaluation/metric_store.py` (mới) | ~45          |
| F   | FZComputer (CSV export + summary)     | `src/evaluation/metric_store.py`       | ~50          |
| G   | CE curves (offline từ checkpoints)    | `src/evaluation/metric_store.py`       | ~40          |

**Tổng: ~280 dòng code mới, 0 dòng code xoá, 0 breaking change.**

---

## A. Tier Boundary Checkpoints (P0)

### Ý tưởng

Lưu model checkpoint mỗi khi Phase 3 chuyển từ tier này sang tier khác (T1→T2, T2→T3, T3→T4). Checkpoint là nền tảng để các metric B, F, G hoạt động — không có checkpoint thì không tính được forgetting matrix hay CE curves offline.

### Logic

`DualTrainer.phase3_pengym_finetuning()` hiện gọi `train_continually(task_list=self._pengym_tasks)`, train tuần tự qua tất cả tasks. Tasks được sắp xếp theo curriculum (T1 trước, T4 sau). Tier boundary = thời điểm task tiếp theo có tier khác task hiện tại.

Phát hiện boundary bằng cách parse tier từ tên scenario (`tiny_T1_001` → `T1`). Khi tier thay đổi, save model trước khi train task mới.

### Cách implement

**File:** `src/training/dual_trainer.py`

**Bước 1:** Thêm method phát hiện tier từ scenario path:

```python
@staticmethod
def _extract_tier(scenario_path: str) -> str:
    """Extract tier (T1/T2/T3/T4) from scenario filename."""
    import re
    m = re.search(r'_T(\d+)_', Path(scenario_path).stem)
    return f"T{m.group(1)}" if m else "T0"
```

**Bước 2:** Sửa `phase3_pengym_finetuning()` — thêm callback save checkpoint. Vì `train_continually()` train từng task tuần tự trong vòng for, ta cần insert checkpoint logic **trước** khi gọi `train_continually()` bằng cách wrap task list:

Cách tiếp cận: thay vì sửa `Agent_CL.train_continually()` (shared code, không nên sửa), loop qua từng nhóm tasks cùng tier và gọi `train_continually()` cho mỗi nhóm, save giữa các nhóm.

```python
def phase3_pengym_finetuning(self, eval_freq: int = 5) -> Dict[str, Any]:
    # ... (giữ nguyên setup code) ...

    # Nhóm tasks theo tier để checkpoint giữa các tier
    tier_groups = []  # list of (tier_name, [task_indices])
    current_tier = None
    for idx, sc_path in enumerate(self.pengym_scenarios):
        tier = self._extract_tier(sc_path)
        if tier != current_tier:
            tier_groups.append((tier, []))
            current_tier = tier
        tier_groups[-1][1].append(idx)

    checkpoint_dir = self.output_dir / "models" / "tier_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tier_checkpoints = {}

    for tier_name, task_indices in tier_groups:
        tier_tasks = [self._pengym_tasks[i] for i in task_indices]
        tier_schedule = {
            i: episode_schedule[task_indices[i]]
            for i in range(len(task_indices))
        } if episode_schedule else None

        self._theta_dual.train_continually(
            task_list=tier_tasks,
            eval_freq=eval_freq,
            save_agent=False,
            verbose=True,
            episode_schedule=tier_schedule,
        )

        # Checkpoint sau mỗi tier
        ckpt_path = checkpoint_dir / f"after_{tier_name}"
        self._theta_dual.save(path=ckpt_path)
        tier_checkpoints[tier_name] = str(ckpt_path)
        logging.info(f"[Phase 3] Checkpoint saved after {tier_name} → {ckpt_path}")

    # ... (giữ nguyên phần save final model) ...
    results["tier_checkpoints"] = tier_checkpoints
```

**Lưu ý quan trọng:** Khi tách `train_continually()` thành nhiều lần gọi (mỗi tier một lần), `current_task_id` trong `Agent_CL` sẽ reset về 0 mỗi lần gọi. Điều này **đúng hành vi mong muốn** vì EWC Fisher được tích luỹ qua `policy_preservation()` và không phụ thuộc task_id tuyệt đối. State normalizer running stats cũng persistent qua các lần gọi.

---

## B. Ma Trận Forgetting ($\mathcal{F}$) và Zero-Shot Transfer ($\mathcal{Z}$) (P0)

### Ý tưởng

Đo lường cụ thể:

- $\mathcal{F}_{i,j}$: task $i$ bị quên bao nhiêu sau khi học task $j$ (với $j > i$)
- $\mathcal{Z}_{j}$: task $j$ được hưởng lợi bao nhiêu từ kiến thức các task trước đó (zero-shot, trước khi train task $j$)

### Logic

Cả hai đều sử dụng NR (Normalized Reward) thay vì raw return, vì NR đã chuẩn hoá cross-scenario.

Với $N$ tasks trong curriculum:

$$\mathcal{F}_{i,j} = NR_i^{\text{after } i} - NR_i^{\text{after } j} \quad \forall \; j > i$$

$\mathcal{F}_{i,j} > 0$ = task $i$ bị quên sau khi học task $j$. Ma trận tam giác trên, kích thước $N \times N$.

$$\mathcal{Z}_j = NR_j^{\text{before } j} - NR_j^{\text{random}}$$

$\mathcal{Z}_j > 0$ = kiến thức từ tasks trước giúp task $j$ ngay lập tức. Random baseline = NR trung bình khi agent chưa train gì trên task đó (lấy từ `theta_pengym_scratch` episode 0, hoặc ước lượng NR ≈ 0).

### Cách implement

**File:** `src/evaluation/strategy_c_eval.py`

Thêm method vào class `StrategyCEvaluator`:

```python
def compute_forgetting_matrix(
    self,
    tier_checkpoint_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute Isolated Forgetting matrix from tier checkpoint eval results.

    Parameters
    ----------
    tier_checkpoint_results : dict
        Maps tier_name → evaluate_agent() result dict.
        Ví dụ: {"T1": eval_result_after_T1, "T2": eval_result_after_T2, ...}
        Mỗi eval_result chứa per_task list với "task" và "normalized_reward".

    Returns
    -------
    dict với keys: "F_matrix", "Z_vector", "task_names", "tier_names", "summary"
    """
    tier_names = sorted(tier_checkpoint_results.keys())
    # Tập hợp tất cả task names
    all_tasks = []
    for tier in tier_names:
        result = tier_checkpoint_results[tier]
        for t in result.get("per_task", []):
            if t["task"] not in all_tasks:
                all_tasks.append(t["task"])

    n = len(all_tasks)
    m = len(tier_names)

    # Build NR matrix: nr_after[task_idx][tier_idx] = NR of task i after training tier j
    nr_after = np.full((n, m), np.nan)
    for j, tier in enumerate(tier_names):
        result = tier_checkpoint_results[tier]
        task_nr_map = {t["task"]: t.get("normalized_reward") for t in result.get("per_task", [])}
        for i, task_name in enumerate(all_tasks):
            nr_val = task_nr_map.get(task_name)
            if nr_val is not None:
                nr_after[i][j] = nr_val

    # F[i][j] = NR_i_after_tier_i - NR_i_after_tier_j   (j > i's tier)
    F_matrix = np.full((n, m), np.nan)
    for i in range(n):
        # Tìm tier mà task i thuộc về
        task_tier_idx = None
        for j, tier in enumerate(tier_names):
            result = tier_checkpoint_results[tier]
            task_names_in_tier = [t["task"] for t in result.get("per_task", [])]
            # Task i "thuộc" tier j nếu đây là tier đầu tiên có data cho task i
            if not np.isnan(nr_after[i][j]) and task_tier_idx is None:
                task_tier_idx = j
        if task_tier_idx is None:
            continue
        for j in range(task_tier_idx + 1, m):
            if not np.isnan(nr_after[i][task_tier_idx]) and not np.isnan(nr_after[i][j]):
                F_matrix[i][j] = nr_after[i][task_tier_idx] - nr_after[i][j]

    # Z[j] = NR_task_j_before_training_j - NR_random_baseline
    # "before training j" ≈ NR from checkpoint of tier before j's tier
    Z_vector = np.full(n, np.nan)
    for i in range(n):
        task_tier_idx = None
        for j in range(m):
            if not np.isnan(nr_after[i][j]):
                task_tier_idx = j
                break
        if task_tier_idx is not None and task_tier_idx > 0:
            nr_before = nr_after[i][task_tier_idx - 1]
            if not np.isnan(nr_before):
                Z_vector[i] = nr_before - 0.0  # baseline NR ≈ 0 (random)

    # Summary
    f_values = F_matrix[~np.isnan(F_matrix)]
    z_values = Z_vector[~np.isnan(Z_vector)]
    summary = {
        "mean_forgetting": float(np.mean(f_values)) if len(f_values) > 0 else None,
        "max_forgetting": float(np.max(f_values)) if len(f_values) > 0 else None,
        "mean_zero_shot_transfer": float(np.mean(z_values)) if len(z_values) > 0 else None,
        "tasks_with_positive_transfer": int(np.sum(z_values > 0)) if len(z_values) > 0 else 0,
    }

    return {
        "F_matrix": F_matrix.tolist(),
        "Z_vector": Z_vector.tolist(),
        "nr_after": nr_after.tolist(),
        "task_names": all_tasks,
        "tier_names": tier_names,
        "summary": summary,
    }
```

**Tích hợp vào Phase 4:** Sau khi eval tất cả agents, DualTrainer load từng tier checkpoint (từ cải tiến A), chạy `evaluator.evaluate_agent()` trên all tasks cho mỗi checkpoint, rồi gọi `compute_forgetting_matrix()`.

```python
# Trong phase4_evaluation(), sau block eval hiện tại:
tier_checkpoints = getattr(self, '_tier_checkpoints', {})
if tier_checkpoints:
    tier_results = {}
    for tier_name, ckpt_path in tier_checkpoints.items():
        # Load checkpoint vào agent copy, eval trên all PenGym tasks
        agent_copy = copy.deepcopy(self._theta_dual)
        agent_copy.load(path=ckpt_path)
        evaluator.register_agent(f"ckpt_{tier_name}", agent_copy)
        tier_results[tier_name] = evaluator.evaluate_agent(
            f"ckpt_{tier_name}",
            self._create_eval_pengym_tasks(),
            domain="pengym",
        )
    fz = evaluator.compute_forgetting_matrix(tier_results)
    results["forgetting_matrix"] = fz
```

---

## C. TTT + AUC Learning-Speed Transfer (P0)

### Ý tưởng

Tách biệt 2 loại transfer mà FT gộp chung:

- **Zero-shot transfer**: agent tốt ngay từ episode đầu → đo bằng $\mathcal{Z}$
- **Learning acceleration**: agent hội tụ nhanh hơn → đo bằng TTT / AUC

### Logic

**Time-To-Threshold (TTT):** Số episode đầu tiên đạt SR ≥ threshold (mặc định 0.8).

$$TTT_{\text{task}} = \min \{e \;|\; \text{success}(e) = 1\}$$

Hiện tại `Agent_CL` đã track `first_hit_eps` — chính là TTT với threshold = first success. Giá trị này lưu trong player nhưng **không được export** ra ngoài khi kết thúc `learn_new_task()`.

**Area Under Curve (AUC):** Diện tích dưới đường cong reward training:

$$AUC_{\text{task}} = \frac{1}{N} \sum_{e=1}^{N} r_e$$

`Task_Train_matrix.Train_Episode_Rewards` lưu toàn bộ rewards per episode — AUC = mean.

### Cách implement

**File:** `src/evaluation/strategy_c_eval.py`

Thêm static method tính TTT + AUC từ training data đã có:

```python
@staticmethod
def compute_learning_speed(
    dual_training_data: Dict[str, list],
    scratch_training_data: Dict[str, list],
    ttt_threshold: float = 0.8,
) -> Dict[str, Any]:
    """Compute TTT and AUC learning-speed transfer metrics.

    Parameters
    ----------
    dual_training_data : dict
        Maps task_name → list of episode rewards (theta_dual).
    scratch_training_data : dict
        Maps task_name → list of episode rewards (theta_scratch).
    ttt_threshold : float
        Reward threshold cho TTT (fraction of optimal).

    Returns
    -------
    dict với per-task TTT/AUC comparison + aggregate speed transfer.
    """
    results = {"per_task": [], "aggregate": {}}

    for task_name in dual_training_data:
        dual_rewards = dual_training_data[task_name]
        scratch_rewards = scratch_training_data.get(task_name, [])

        dual_auc = float(np.mean(dual_rewards)) if dual_rewards else 0
        scratch_auc = float(np.mean(scratch_rewards)) if scratch_rewards else 0

        # TTT: first episode where cumulative success suggests convergence
        # Đơn giản: first episode with reward > 0 (thành công)
        dual_ttt = next((i for i, r in enumerate(dual_rewards) if r > 0), len(dual_rewards))
        scratch_ttt = next((i for i, r in enumerate(scratch_rewards) if r > 0), len(scratch_rewards))

        results["per_task"].append({
            "task": task_name,
            "dual_ttt": dual_ttt,
            "scratch_ttt": scratch_ttt,
            "ttt_speedup": scratch_ttt / max(dual_ttt, 1),
            "dual_auc": dual_auc,
            "scratch_auc": scratch_auc,
            "auc_ratio": dual_auc / max(abs(scratch_auc), 1e-8),
        })

    # Aggregate
    if results["per_task"]:
        results["aggregate"] = {
            "mean_ttt_speedup": float(np.mean([t["ttt_speedup"] for t in results["per_task"]])),
            "mean_auc_ratio": float(np.mean([t["auc_ratio"] for t in results["per_task"]])),
        }

    return results
```

**Nguồn data:** `Task_Train_matrix.Train_Episode_Rewards` từ `Agent_CL.learn_new_task()`. Hiện tại `DualTrainer` **không lưu** matrix này. Cần export nó:

Sửa `DualTrainer.phase3_pengym_finetuning()` để lưu training rewards:

```python
# Sau khi train_continually() kết thúc:
self._dual_training_rewards = {}
# Lấy từ CL_Train_matrix nếu Agent_CL expose, hoặc parse từ TensorBoard logs
```

Cách tốt hơn: `Agent_CL.train_continually()` đã trả về `CL_Train_matrix`. Thêm field `per_task_rewards` vào matrix trước khi return:

```python
# Trong Agent_CL.train_continually(), trước return:
CL_Train_matrix.per_task_rewards = {}
# → populate trong learn_new_task() (data đã có, chỉ cần forward ra)
```

Xem chi tiết tại mục implement cụ thể bên dưới.

---

## D. Train / Heldout Evaluation Split (P0)

### Ý tưởng

Dataset overlay đã có split convention:

- `_000`: calibration
- `_001`, `_002`, `_003`: training
- `_004`–`_009`: heldout

Phase 4 hiện chỉ eval trên training scenarios. Cần eval thêm trên heldout để chứng minh transfer là **structural** (học tấn công strategy), không phải **memorization** (nhớ scenario cụ thể).

### Logic

`DualTrainer.__init__()` nhận `pengym_scenarios` list. Thêm parameter `heldout_scenarios` riêng. Phase 4 eval trên cả 2 sets, báo cáo riêng:

- `pengym_train`: FT/BT trên scenarios đã train
- `pengym_heldout`: FT/BT trên scenarios chưa thấy

Nếu FT trên heldout ≈ FT trên train → transfer generalise tốt. Nếu FT heldout << FT train → agent memorize.

### Cách implement

**File 1:** `src/training/dual_trainer.py` — `__init__` thêm param

```python
def __init__(
    self,
    sim_scenarios: List[str],
    pengym_scenarios: List[str],
    heldout_scenarios: Optional[List[str]] = None,   # ← MỚI
    # ... (giữ nguyên params khác)
):
    # ... (giữ nguyên) ...
    self.heldout_scenarios = [str(p) for p in (heldout_scenarios or [])]
```

**File 1:** `phase4_evaluation()` — thêm block eval heldout

```python
# Sau block eval hiện tại, trước return:
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
    # Register same agents
    for name in agent_names:
        heldout_evaluator.register_agent(name, self._agents_map[name])
    heldout_report = heldout_evaluator.evaluate_all()
    results["heldout"] = heldout_report
    results["heldout_transfer_metrics"] = heldout_report.get("metrics", {})
```

Thêm helper method:

```python
def _create_eval_tasks_from(self, scenario_paths: List[str]) -> list:
    """Create fresh PenGym adapters from given scenario paths."""
    from src.envs.adapters.pengym_host_adapter import PenGymHostAdapter
    return [
        PenGymHostAdapter.from_scenario(p, seed=self.seed, use_unified_encoding=True)
        for p in scenario_paths
    ]
```

**File 2:** `run_strategy_c.py` — CLI flag

```python
parser.add_argument(
    "--heldout-scenarios", nargs="*", default=None,
    help="Heldout PenGym scenarios for generalization eval (Phase 4).",
)
```

Và trong `main()`, truyền vào DualTrainer:

```python
trainer = DualTrainer(
    # ... giữ nguyên ...
    heldout_scenarios=args.heldout_scenarios,
)
```

---

## E. MetricStore — Structured JSON Logging (P1)

### Ý tưởng

Format chuẩn lưu trữ metrics cho mọi downstream analysis (CE curves, F/Z matrices, AUC/TTT). Structure: `metrics[seed][checkpoint][task][metric]`.

### Logic

Một file JSON duy nhất chứa toàn bộ evaluation history:

```json
{
  "metadata": {"seed": 42, "date": "2026-02-25"},
  "checkpoints": {
    "after_T1": {
      "tiny_T1_001": {"sr": 0.95, "nr": 0.87, "eta": 0.72},
      "tiny_T2_001": {"sr": 0.40, "nr": 0.31, "eta": null}
    },
    "after_T2": { ... },
    "final": { ... }
  },
  "training_curves": {
    "tiny_T1_001": {"episode_rewards": [0.1, 0.3, ...], "ttt": 12},
    "tiny_T2_001": {"episode_rewards": [...], "ttt": 45}
  },
  "forgetting": {"F_matrix": [...], "Z_vector": [...]},
  "transfer": {"FT_SR": 0.15, "FT_NR": 0.08, ...}
}
```

### Cách implement

**File mới:** `src/evaluation/metric_store.py`

```python
"""MetricStore — Structured storage for continual learning metrics."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

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

    def add_checkpoint(self, name: str, eval_result: Dict[str, Any]):
        """Store per-task metrics from an evaluate_agent() result."""
        self.data["checkpoints"][name] = {
            t["task"]: {
                "sr": t.get("sr"),
                "nr": t.get("normalized_reward"),
                "eta": t.get("step_efficiency"),
            }
            for t in eval_result.get("per_task", [])
        }

    def add_training_curve(self, task_name: str, episode_rewards: list, ttt: int):
        """Store per-task training dynamics."""
        self.data["training_curves"][task_name] = {
            "episode_rewards": episode_rewards,
            "ttt": ttt,
        }

    def set_forgetting(self, fz_result: Dict[str, Any]):
        self.data["forgetting"] = fz_result

    def set_transfer(self, metrics: Dict[str, Any]):
        self.data["transfer"] = metrics

    def save(self, filename: str = "metric_store.json"):
        path = self.output_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.data, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "MetricStore":
        with open(path) as f:
            data = json.load(f)
        store = cls(seed=data["metadata"]["seed"], output_dir=str(Path(path).parent))
        store.data = data
        return store
```

**Tích hợp:** `DualTrainer.phase4_evaluation()` tạo MetricStore, populate từ eval results và tier checkpoints, save cuối Phase 4.

---

## F. FZComputer — CSV Export + Summary (P1)

### Ý tưởng

Chuyển đổi F/Z matrices thành table dễ đọc và CSV cho downstream plotting.

### Cách implement

Thêm vào `src/evaluation/metric_store.py`:

```python
import csv
import io

class FZComputer:
    """Compute and export Forgetting/Transfer matrices from MetricStore."""

    @staticmethod
    def to_csv(fz_result: Dict[str, Any]) -> str:
        """Convert F matrix to CSV string."""
        output = io.StringIO()
        writer = csv.writer(output)
        task_names = fz_result["task_names"]
        tier_names = fz_result["tier_names"]
        F = fz_result["F_matrix"]

        # Header
        writer.writerow(["task \\ after_tier"] + tier_names)
        for i, task in enumerate(task_names):
            row = [task] + [
                f"{F[i][j]:.4f}" if F[i][j] is not None and not (isinstance(F[i][j], float) and F[i][j] != F[i][j]) else ""
                for j in range(len(tier_names))
            ]
            writer.writerow(row)

        # Z vector
        writer.writerow([])
        writer.writerow(["task", "zero_shot_transfer"])
        Z = fz_result["Z_vector"]
        for i, task in enumerate(task_names):
            z_val = Z[i]
            writer.writerow([task, f"{z_val:.4f}" if z_val is not None and z_val == z_val else ""])

        return output.getvalue()

    @staticmethod
    def print_summary(fz_result: Dict[str, Any]) -> str:
        """Format F/Z summary as readable text."""
        s = fz_result.get("summary", {})
        lines = [
            "=== Forgetting / Transfer Summary ===",
            f"  Mean forgetting (F):        {s.get('mean_forgetting', 'N/A')}",
            f"  Max forgetting (F):         {s.get('max_forgetting', 'N/A')}",
            f"  Mean zero-shot transfer (Z): {s.get('mean_zero_shot_transfer', 'N/A')}",
            f"  Tasks with positive Z:      {s.get('tasks_with_positive_transfer', 'N/A')}",
        ]
        return "\n".join(lines)

    @staticmethod
    def save_csv(fz_result: Dict[str, Any], path: str):
        csv_content = FZComputer.to_csv(fz_result)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            f.write(csv_content)
```

---

## G. CE Curves — Offline từ Checkpoints (P1)

### Ý tưởng

Vẽ đường cong hiệu suất trên từng task qua thời gian training (Continual Evaluation curves). Sử dụng tier checkpoints (cải tiến A) + MetricStore (E) thay vì hook vào training loop.

### Logic

Từ MetricStore, extract `checkpoints[tier][task] → NR` cho mỗi cặp (tier, task). Plot NR_task(tier) cho tất cả tasks trên cùng biểu đồ. Đường giảm = forgetting, đường tăng trên task mới = positive transfer.

### Cách implement

Thêm vào `src/evaluation/metric_store.py`:

```python
class CECurveGenerator:
    """Generate Continual Evaluation curves from MetricStore data."""

    @staticmethod
    def extract_curves(store: MetricStore) -> Dict[str, Dict[str, list]]:
        """Extract per-task metric curves across checkpoints.

        Returns
        -------
        dict mapping metric_name → {task_name: [(checkpoint_name, value), ...]}
        """
        checkpoints = store.data.get("checkpoints", {})
        ckpt_names = sorted(checkpoints.keys())

        curves = {"sr": {}, "nr": {}, "eta": {}}
        all_tasks = set()
        for ckpt in ckpt_names:
            all_tasks.update(checkpoints[ckpt].keys())

        for task in sorted(all_tasks):
            for metric in curves:
                curves[metric][task] = []
                for ckpt in ckpt_names:
                    val = checkpoints.get(ckpt, {}).get(task, {}).get(metric)
                    curves[metric][task].append((ckpt, val))

        return curves

    @staticmethod
    def to_csv(curves: Dict[str, Dict[str, list]], metric: str = "nr") -> str:
        """Export CE curves for one metric as CSV (tasks × checkpoints)."""
        output = io.StringIO()
        writer = csv.writer(output)

        data = curves.get(metric, {})
        if not data:
            return ""

        # Get checkpoint names from first task
        first_task = next(iter(data.values()))
        ckpt_names = [c[0] for c in first_task]

        writer.writerow(["task"] + ckpt_names)
        for task, values in sorted(data.items()):
            row = [task] + [
                f"{v:.4f}" if v is not None else ""
                for _, v in values
            ]
            writer.writerow(row)

        return output.getvalue()
```

**Tích hợp vào Phase 4:**

```python
# Cuối phase4_evaluation():
from src.evaluation.metric_store import MetricStore, FZComputer, CECurveGenerator

store = MetricStore(seed=self.seed, output_dir=str(self.output_dir))
# Populate from tier checkpoint evals
for tier_name, result in tier_results.items():
    store.add_checkpoint(f"after_{tier_name}", result)
# Final eval
store.add_checkpoint("final", evaluator.evaluate_agent("theta_dual", ...))
store.set_transfer(results.get("transfer_metrics", {}))
if "forgetting_matrix" in results:
    store.set_forgetting(results["forgetting_matrix"])
store.save()

# Export CSVs
if "forgetting_matrix" in results:
    FZComputer.save_csv(results["forgetting_matrix"],
                        str(self.output_dir / "forgetting_matrix.csv"))

curves = CECurveGenerator.extract_curves(store)
ce_csv = CECurveGenerator.to_csv(curves, metric="nr")
with open(self.output_dir / "ce_curves_nr.csv", "w") as f:
    f.write(ce_csv)
```

---

## Checklist triển khai

| Bước | Cải tiến | File                             | Hành động                                                                                     |
| ---- | -------- | -------------------------------- | --------------------------------------------------------------------------------------------- |
| 1    | A        | `dual_trainer.py`                | Thêm `_extract_tier()`, sửa `phase3_pengym_finetuning()` loop theo tier groups                |
| 2    | D        | `dual_trainer.py`                | Thêm `heldout_scenarios` param, `_create_eval_tasks_from()`, block eval heldout trong Phase 4 |
| 3    | D        | `run_strategy_c.py`              | Thêm `--heldout-scenarios` CLI flag, truyền vào DualTrainer                                   |
| 4    | B        | `strategy_c_eval.py`             | Thêm `compute_forgetting_matrix()` method                                                     |
| 5    | C        | `strategy_c_eval.py`             | Thêm `compute_learning_speed()` static method                                                 |
| 6    | A+B      | `dual_trainer.py` Phase 4        | Load tier checkpoints → eval → gọi `compute_forgetting_matrix()`                              |
| 7    | E+F+G    | `src/evaluation/metric_store.py` | Tạo file mới: MetricStore, FZComputer, CECurveGenerator                                       |
| 8    | E+F+G    | `dual_trainer.py` Phase 4        | Tạo MetricStore, populate, save + export CSV                                                  |

**Thứ tự triển khai:** 1 → 2 → 3 → 4 → 5 → 7 → 6 → 8

Bước 1 (A) là nền tảng — không có tier checkpoints thì B, F, G không chạy. Bước 7 (E+F+G) là file mới, tạo trước khi tích hợp vào DualTrainer (bước 6, 8).

---

## Những gì KHÔNG implement (và lý do)

| Hạng mục                                   | Lý do bỏ                                                                                                                                  |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **EvalScheduler (hook vào training loop)** | Vi phạm single responsibility của `Agent_CL`. Tier checkpoints + offline sweep cho kết quả tương đương mà không sửa shared training code. |
| **Adaptive Reward Normalizer**             | Bounds static `[-3, 100]` đúng cho NASim per-step reward. `UnifiedNormalizer` constructor đã hỗ trợ override khi cần.                     |
| **Full K=20 eval trong training loop**     | Overhead 40-60% training time. Lightweight single-episode eval hiện có đã đủ cho monitoring. K=20 dành cho Phase 4 + offline sweep.       |
