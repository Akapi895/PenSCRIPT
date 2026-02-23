# Strategy C — Hướng dẫn chạy và xem kết quả

> **Strategy C: Shared-State Dual Training** — Huấn luyện SCRIPT CRL trên simulation trước, sau đó chuyển giao (transfer) sang PenGym và fine-tune với ràng buộc EWC.

---

## 1. Tổng quan Pipeline

```
Phase 0 → Validation     (kiểm tra SBERT, PenGym, canonicalization)
Phase 1 → Sim Training    (huấn luyện SCRIPT CRL trên sim → θ_uni)
Phase 2 → Domain Transfer (reset norm, discount Fisher, giảm LR)
Phase 3 → PenGym Finetune (fine-tune trên PenGym với EWC → θ_dual)
Phase 4 → Evaluation      (so sánh tất cả agents trên PenGym)
```

---

## 2. Cách chạy

### 2.1. Lệnh cơ bản (full pipeline)

```bash
python run_strategy_c.py \
    --sim-scenarios data/scenarios/chain/chain-msfexp_vul-sample-6_envs-seed_0.json \
    --pengym-scenarios data/scenarios/tiny.yml data/scenarios/small-linear.yml
```

### 2.2. Tuỳ chỉnh transfer strategy

Có 3 chiến lược chuyển giao:

| Strategy | Mô tả | Khi nào dùng |
|----------|--------|-------------|
| `aggressive` | Giữ nguyên tất cả weights + normalizer | Sim và PenGym rất giống nhau |
| `conservative` | Reset norm, discount Fisher (β), giảm LR | **Mặc định** — cân bằng giữa transfer và adapt |
| `cautious` | Chỉ giữ actor/critic weights, reset hết phần còn lại | Sim và PenGym khác nhau nhiều |

```bash
# Conservative (mặc định)
python run_strategy_c.py \
    --sim-scenarios data/scenarios/chain/chain-msfexp_vul-sample-6_envs-seed_0.json \
    --pengym-scenarios data/scenarios/tiny.yml \
    --transfer-strategy conservative

# Cautious — khi hai domain khác nhau nhiều
python run_strategy_c.py \
    --sim-scenarios data/scenarios/chain/chain-msfexp_vul-sample-6_envs-seed_0.json \
    --pengym-scenarios data/scenarios/tiny.yml \
    --transfer-strategy cautious

# Aggressive — khi muốn zero-shot transfer
python run_strategy_c.py \
    --sim-scenarios data/scenarios/chain/chain-msfexp_vul-sample-6_envs-seed_0.json \
    --pengym-scenarios data/scenarios/tiny.yml \
    --transfer-strategy aggressive
```

### 2.3. So sánh đầy đủ 4 agents (có thêm θ_scratch baseline)

Thêm `--train-scratch` để huấn luyện thêm một agent PenGym từ đầu, dùng làm baseline:

```bash
python run_strategy_c.py \
    --sim-scenarios data/scenarios/chain/chain-msfexp_vul-sample-6_envs-seed_0.json \
    --pengym-scenarios data/scenarios/tiny.yml data/scenarios/small-linear.yml \
    --train-scratch
```

### 2.4. Tất cả tham số CLI

```bash
python run_strategy_c.py --help
```

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--sim-scenarios` | *(bắt buộc)* | File JSON scenario simulation cho Phase 1 |
| `--pengym-scenarios` | *(bắt buộc)* | File YAML scenario PenGym cho Phase 3 |
| `--transfer-strategy` | `conservative` | Chiến lược transfer: `aggressive`/`conservative`/`cautious` |
| `--fisher-beta` | `0.3` | Hệ số discount Fisher (β ∈ [0.1, 0.5]) |
| `--lr-factor` | `0.1` | Nhân learning rate × factor sau khi transfer |
| `--warmup-episodes` | `10` | Số episode random rollout để warmup normalizer |
| `--episodes` | `500` | Số episode huấn luyện mỗi task |
| `--step-limit` | `100` | Số step tối đa mỗi episode |
| `--eval-freq` | `5` | Đánh giá mỗi N episode |
| `--ewc-lambda` | `2000` | Độ mạnh regularization EWC (λ) |
| `--seed` | `42` | Random seed |
| `--skip-phase0` | `false` | Bỏ qua Phase 0 validation |
| `--train-scratch` | `false` | Huấn luyện thêm θ_scratch baseline |
| `--output-dir` | `outputs/strategy_c` | Thư mục output |

### 2.5. Ví dụ chạy nhanh để test (tiny scenario)

```bash
python run_strategy_c.py \
    --sim-scenarios data/scenarios/chain/chain-msfexp_vul-sample-6_envs-seed_0.json \
    --pengym-scenarios data/scenarios/tiny.yml \
    --episodes 50 --step-limit 30 --eval-freq 10 \
    --skip-phase0
```

---

## 3. Cấu trúc output

Sau khi chạy xong, thư mục output sẽ có cấu trúc:

```
outputs/strategy_c/
├── strategy_c_results.json        ← KẾT QUẢ CHÍNH (JSON tổng hợp)
├── dual_trainer_results.json      ← Kết quả từ DualTrainer
├── logs/
│   └── strategy_c.log             ← Log toàn bộ quá trình
├── models/
│   ├── phase1_sim/                ← Model θ_uni (sau Phase 1)
│   │   ├── keeper_actor.pth
│   │   ├── keeper_critic.pth
│   │   └── ...
│   ├── phase3_dual/               ← Model θ_dual (sau Phase 3)
│   │   ├── keeper_actor.pth
│   │   ├── keeper_critic.pth
│   │   └── ...
│   └── pengym_scratch/            ← Model θ_scratch (nếu --train-scratch)
│       └── ...
└── tensorboard/
    ├── phase1_sim/                ← TensorBoard logs Phase 1
    ├── phase3_pengym/             ← TensorBoard logs Phase 3
    └── scratch_pengym/            ← TensorBoard logs scratch (nếu có)
```

---

## 4. Cách xem kết quả

### 4.1. File JSON chính — `strategy_c_results.json`

```bash
cat outputs/strategy_c/strategy_c_results.json | python -m json.tool
```

File này chứa toàn bộ kết quả, cấu trúc như sau:

```jsonc
{
  "phase0": {
    "sbert_consistency": { "all_above_0.99": true },    // SBERT ổn định?
    "dim_check": true,                                   // Unified dim = 1540?
    "canonicalization": { "ubuntu_→_linux": true },       // Canonicalization OK?
    "pengym_loadable": [{ "scenario": "...", "ok": true }]
  },

  "phase1": {
    "num_tasks": 6,
    "train_time_s": 123.4,
    "final_sr": 0.85,              // ← Success Rate trên sim
    "model_dir": "outputs/strategy_c/models/phase1_sim"
  },

  "phase2": {
    "strategy": "conservative",
    "fisher_discount": 0.3,        // Fisher × β
    "lr_factor": 0.1,              // LR giảm 10×
    "warmup_states": 500           // Số state đã warmup
  },

  "phase3": {
    "num_tasks": 2,
    "train_time_s": 67.8,
    "final_sr": 0.75,              // ← Success Rate sau fine-tune PenGym
    "model_dir": "outputs/strategy_c/models/phase3_dual"
  },

  "phase4": {
    "agents": {
      "theta_sim_unified": {
        "success_rate": 0.40,      // Sim agent test trực tiếp trên PenGym
        "total_rewards": 120.5
      },
      "theta_dual": {
        "success_rate": 0.75,      // Dual agent trên PenGym
        "total_rewards": 280.3
      },
      "theta_pengym_scratch": {    // Chỉ có nếu --train-scratch
        "success_rate": 0.55,
        "total_rewards": 180.0
      }
    },
    "transfer_metrics": {
      "sim_sr_on_pengym": 0.40,
      "dual_sr_on_pengym": 0.75,
      "transfer_gain": 0.35,       // ← θ_dual - θ_sim (trên PenGym)
      "transfer_ratio": 1.875      // ← θ_dual / θ_sim
    }
  },

  "total_time_s": 234.5
}
```

### 4.2. Các chỉ số quan trọng cần theo dõi

| Chỉ số | Ý nghĩa | Mong muốn |
|--------|---------|-----------|
| `phase1.final_sr` | SR trên sim sau CRL | Càng cao càng tốt (≥ 0.7) |
| `phase3.final_sr` | SR trên PenGym sau dual-train | Càng cao càng tốt |
| `phase4.transfer_gain` | SR(θ_dual) − SR(θ_sim) trên PenGym | > 0 (transfer có ích) |
| `phase4.transfer_ratio` | SR(θ_dual) / SR(θ_sim) trên PenGym | > 1.0 |
| Forward Transfer (FT) | SR(θ_dual) − SR(θ_scratch) trên PenGym | > 0 nghĩa là sim-pretrain giúp ích |
| Backward Transfer (BT) | SR(θ_dual trên sim) − SR(θ_uni trên sim) | ≈ 0 nghĩa là không quên sim |

### 4.3. Xem TensorBoard

```bash
tensorboard --logdir outputs/strategy_c/tensorboard
```

Mở browser tại `http://localhost:6006` để xem:
- **Phase 1 tab**: Reward và SR theo episode trên sim
- **Phase 3 tab**: Reward và SR theo episode trên PenGym
- **Scratch tab**: So sánh tốc độ học scratch vs dual

### 4.4. Xem log chi tiết

```bash
# Toàn bộ log
cat outputs/strategy_c/logs/strategy_c.log

# Chỉ xem kết quả Phase 4
grep "\[Phase 4\]" outputs/strategy_c/logs/strategy_c.log

# Xem transfer metadata
grep "\[DomainTransfer\]" outputs/strategy_c/logs/strategy_c.log
```

### 4.5. So sánh kết quả bằng Python

```python
import json

with open("outputs/strategy_c/strategy_c_results.json") as f:
    r = json.load(f)

# Tóm tắt nhanh
print(f"Phase 1 (Sim) SR:     {r['phase1']['final_sr']:.1%}")
print(f"Phase 3 (Dual) SR:    {r['phase3']['final_sr']:.1%}")
print(f"Transfer strategy:    {r['phase2']['strategy']}")
print(f"Transfer gain:        {r['phase4']['transfer_metrics']['transfer_gain']:+.1%}")
print(f"Total time:           {r['total_time_s']:.0f}s")

# Chi tiết từng agent
for name, data in r["phase4"]["agents"].items():
    sr = data.get("success_rate", "N/A")
    print(f"  {name}: SR={sr:.1%}" if isinstance(sr, float) else f"  {name}: {sr}")
```

---

## 5. Sử dụng từng module riêng lẻ (API)

### 5.1. UnifiedStateEncoder (mã hoá state 1540-dim)

```python
from src.envs.core.unified_state_encoder import UnifiedStateEncoder

enc = UnifiedStateEncoder()

# Encode từ simulation
vec_sim = enc.encode_from_sim(
    access="compromised", os="linux",
    ports=["22", "80"], services=["ssh", "http"],
    discovered=True,
)  # → (1540,)

# Encode từ PenGym
vec_pg = enc.encode_from_pengym(
    compromised=True, reachable=True, discovered=True,
    access_level=2, os="ubuntu",
    services=["openssh", "ftp"], ports=["22", "21"],
)  # → (1540,)

# Chuyển đổi state cũ 1538-dim → 1540-dim
legacy = ...  # numpy array (1538,)
unified = enc.pad_legacy_state(legacy)  # → (1540,)
```

### 5.2. DomainTransferManager (chuyển giao domain)

```python
from src.training.domain_transfer import DomainTransferManager
from src.agent.policy.config import Script_Config

cfg = Script_Config(
    fisher_discount_beta=0.3,
    transfer_strategy="conservative",
)
mgr = DomainTransferManager(script_config=cfg)

# Transfer sim agent → PenGym
meta = mgr.transfer(
    sim_agent=agent_cl,       # Agent_CL đã train trên sim
    pengym_tasks=task_list,    # list[PenGymHostAdapter]
    strategy="conservative",
)
print(meta)
# {'strategy': 'conservative', 'norm_reset': True,
#  'fisher_discount': 0.3, 'lr_factor': 0.1, 'warmup_states': 500}
```

### 5.3. UnifiedNormalizer (chuẩn hoá reward [-1, +1])

```python
from src.envs.wrappers.reward_normalizer import UnifiedNormalizer

# Cho simulation (max_reward=1000, min_reward=-10)
norm_sim = UnifiedNormalizer(source="simulation")
norm_sim.normalize(500)   # → 0.5
norm_sim.normalize(-5)    # → -0.5

# Cho PenGym (max_reward=100, min_reward=-3)
norm_pg = UnifiedNormalizer(source="pengym")
norm_pg.normalize(50)     # → 0.5
norm_pg.normalize(-1.5)   # → -0.5
```

### 5.4. PenGymStateAdapter — chế độ unified

```python
from src.envs.adapters.state_adapter import PenGymStateAdapter

adapter = PenGymStateAdapter(scenario=nasim_scenario)

# Legacy 1538-dim (backward compatible)
state_1538 = adapter.convert(flat_obs, host_addr=(1, 0))

# Unified 1540-dim (Strategy C)
state_1540 = adapter.convert_unified(flat_obs, host_addr=(1, 0))

# Tất cả hosts cùng lúc
all_unified = adapter.convert_all_hosts_unified(flat_obs)
```

---

## 6. Troubleshooting

| Vấn đề | Giải pháp |
|--------|-----------|
| `No sim tasks loaded` | Kiểm tra file JSON scenario có đúng format và vulnerability có trong `Vul_cve_set` |
| `PenGym scenario not loadable` | Kiểm tra file YAML scenario và PenGym đã cài đặt (`pip install pengym`) |
| Phase 4 SR = 0 cho tất cả agents | Tăng `--episodes` và `--step-limit` |
| Transfer gain < 0 | Thử `--transfer-strategy cautious` hoặc tăng `--fisher-beta` lên 0.5 |
| Out of memory | Giảm `--episodes` hoặc chạy trên GPU (`CUDA_VISIBLE_DEVICES=0`) |
| SBERT consistency FAIL | Kiểm tra model SBERT `all-MiniLM-L12-v2` đã download đúng |
