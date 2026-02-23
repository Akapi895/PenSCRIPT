# Strategy C — Đánh Giá Mức Độ Hoàn Thiện Hệ Thống

> **Cập nhật:** 2026-02-23 (dựa trên phân tích mã nguồn thực tế, branch `strC_1`)  
> **Phạm vi:** Đánh giá toàn bộ implementation của Strategy C: Shared-State Dual Training  
> **Phương pháp:** Phân tích trực tiếp từ source code — không dựa trên tài liệu thiết kế

---

## Mục Lục

1. [Tổng Quan Trạng Thái Hệ Thống](#1-tổng-quan-trạng-thái-hệ-thống)
2. [Kiến Trúc Pipeline](#2-kiến-trúc-pipeline)
3. [Thành Phần Đã Triển Khai Đầy Đủ](#3-thành-phần-đã-triển-khai-đầy-đủ)
4. [Thành Phần Triển Khai Một Phần](#4-thành-phần-triển-khai-một-phần)
5. [Thành Phần Chưa Tích Hợp Vào Pipeline](#5-thành-phần-chưa-tích-hợp-vào-pipeline)
6. [Ánh Xạ File → Chức Năng](#6-ánh-xạ-file--chức-năng)
7. [Khoảng Cách So Với Thiết Kế Ban Đầu](#7-khoảng-cách-so-với-thiết-kế-ban-đầu)
8. [Hướng Phát Triển Tiếp Theo](#8-hướng-phát-triển-tiếp-theo)

---

## 1. Tổng Quan Trạng Thái Hệ Thống

### Kết luận: **Pipeline Hoạt Động End-to-End, Còn 1 Khoảng Cách Chính**

Hệ thống đã triển khai **đầy đủ pipeline 5 giai đoạn** (Phase 0→1→2→3→4) với tất cả các thành phần cốt lõi: huấn luyện CRL trên sim, chuyển giao domain có kiểm soát, fine-tune trên PenGym, và đánh giá đa-agent. Kiến trúc hành động phân cấp (hierarchical action space) đã được tích hợp xuyên suốt.

**Khoảng cách duy nhất có ý nghĩa:** `UnifiedStateEncoder` (1540-dim) đã được triển khai code hoàn chỉnh nhưng **chưa được tích hợp vào luồng huấn luyện thực tế** — cả sim và PenGym đều sử dụng encoding 1538-dim legacy. Điều này có nghĩa chuyển giao sim→PenGym đang dựa vào sự tương đồng cấu trúc giữa hai bộ mã hóa riêng biệt, thay vì một bộ mã hóa thống nhất đảm bảo căn chỉnh ngữ nghĩa.

| Thành phần                           | Trạng thái                        | File chính                                  |
| ------------------------------------ | --------------------------------- | ------------------------------------------- |
| CLI Entry Point                      | ✅ Hoàn chỉnh                     | `run_strategy_c.py`                         |
| DualTrainer (Phase 0→4)              | ✅ Hoàn chỉnh                     | `src/training/dual_trainer.py`              |
| Hierarchical Action Space (16-dim)   | ✅ Hoàn chỉnh + tích hợp          | `src/agent/actions/service_action_space.py` |
| HOST + select_cve()                  | ✅ Hoàn chỉnh + tích hợp          | `src/agent/host.py`                         |
| DomainTransferManager (3 strategies) | ✅ Hoàn chỉnh + tích hợp          | `src/training/domain_transfer.py`           |
| OnlineEWC + discount_fisher()        | ✅ Hoàn chỉnh + tích hợp          | `src/agent/continual/Script.py`             |
| Script_Config (Strategy C params)    | ✅ Hoàn chỉnh                     | `src/agent/policy/config.py`                |
| StrategyCEvaluator (Phase 4)         | ✅ Hoàn chỉnh + tích hợp          | `src/evaluation/strategy_c_eval.py`         |
| SingleHostPenGymWrapper              | ✅ Hoàn chỉnh                     | `src/envs/wrappers/single_host_wrapper.py`  |
| PenGymHostAdapter (duck-typing HOST) | ✅ Hoàn chỉnh                     | `src/envs/adapters/pengym_host_adapter.py`  |
| PenGymStateAdapter (1538-dim)        | ✅ Hoàn chỉnh                     | `src/envs/adapters/state_adapter.py`        |
| SCRIPT CRL (5 trụ cột)               | ✅ Hoàn chỉnh                     | `src/agent/continual/Script.py`             |
| UnifiedStateEncoder (1540-dim)       | ⚠️ Code hoàn chỉnh, chưa tích hợp | `src/envs/core/unified_state_encoder.py`    |
| UnifiedNormalizer ([-1,+1])          | ⚠️ Code hoàn chỉnh, chưa tích hợp | `src/envs/wrappers/reward_normalizer.py`    |

---

## 2. Kiến Trúc Pipeline

### 2.1 Luồng Thực Thi (đã triển khai)

```
run_strategy_c.py
  └─ DualTrainer(sim_scenarios, pengym_scenarios, ppo_kwargs, script_kwargs)
      │
      ├─ Phase 0: Validation
      │   ├─ SBERT consistency (cosine > 0.99)
      │   ├─ UnifiedStateEncoder dim check (1540)
      │   ├─ Canonicalization check (ubuntu→linux, openssh→ssh)
      │   ├─ Cross-domain SBERT similarity
      │   └─ PenGym scenario loadability
      │
      ├─ Phase 1: Sim Training
      │   ├─ ServiceActionSpace(action_class=Action) → 16 groups
      │   ├─ HOST(ip, env_data, service_action_space=sas) per host
      │   ├─ PPO_Config(action_dim=16) → PPO Actor outputs 16-dim
      │   ├─ Agent_CL.train_continually(sim_tasks)
      │   └─ → θ_sim_unified
      │
      ├─ Phase 2: Domain Transfer
      │   ├─ deep copy θ_sim_unified → θ_dual
      │   ├─ DomainTransferManager.transfer(θ_dual, pengym_tasks, strategy)
      │   │   ├─ aggressive: giữ nguyên tất cả
      │   │   ├─ conservative: reset norm + warmup + discount Fisher + giảm LR
      │   │   └─ cautious: reset all trừ weights
      │   └─ → θ_dual (sẵn sàng cho Phase 3)
      │
      ├─ Phase 3: PenGym Fine-tuning
      │   ├─ PenGymHostAdapter.from_scenario() per scenario
      │   ├─ θ_dual.train_continually(pengym_tasks)
      │   └─ → θ_dual (fine-tuned)
      │
      └─ Phase 4: Evaluation
          ├─ StrategyCEvaluator.evaluate_all()
          ├─ Forward Transfer = SR(θ_dual) − SR(θ_scratch)
          ├─ Backward Transfer = SR(θ_dual on sim) − SR(θ_uni on sim)
          └─ Transfer Ratio = SR(θ_dual) / SR(θ_scratch)
```

### 2.2 Luồng Hành Động Phân Cấp (đã triển khai)

```
PPO Actor → service_action ∈ [0, 15]    (16-dim: 4 scan + 9 exploit + 3 privesc)
    │
HOST.perform_action(service_action)
    │
    └─ sas.select_cve(service_action, host_info, strategy='match', env_data)
        │
        ├─ Scan actions (0-3): → trả trực tiếp scan index
        ├─ Exploit/Privesc (4-15):
        │   ├─ Match strategy: tìm CVE khớp vulnerability thực của target
        │   ├─ Rank strategy: chọn CVE rank cao nhất (excellent > great > ...)
        │   ├─ Random strategy: chọn ngẫu nhiên trong group
        │   └─ Round-robin strategy: xoay vòng
        │
        └─ → cve_index trong Action.legal_actions
            │
            HOST.step(cve_index)  → thực thi CVE exploit cụ thể
```

### 2.3 Luồng Trạng Thái (hiện tại: 1538-dim)

```
SIM path:  HOST.state_vector = StateEncoder
           → access(2) + OS(384) + Port(384) + Service(384) + Aux(384) = 1538

PenGym path: SingleHostPenGymWrapper → PenGymStateAdapter.convert()
             → access(2) + OS(384) + Port(384) + Service(384) + Aux(384) = 1538

PPO Actor/Critic input: state_dim = StateEncoder.state_space = 1538
```

**Thiết kế mong muốn (chưa tích hợp):**

```
UnifiedStateEncoder.encode_from_sim() hoặc encode_from_pengym()
→ access(3) + discovery(1) + OS(384) + Port(384) + Service(384) + Aux(384) = 1540
  + canonicalization: ubuntu→linux, openssh→ssh, ...
```

---

## 3. Thành Phần Đã Triển Khai Đầy Đủ

### 3.1 DualTrainer — Orchestrator Pipeline

**File:** `src/training/dual_trainer.py` (592 dòng)

Bộ điều phối chính của Strategy C. Tất cả 5 phase đều được triển khai đầy đủ:

| Phase | Phương thức                  | Chức năng                                                          | Trạng thái |
| ----- | ---------------------------- | ------------------------------------------------------------------ | ---------- |
| 0     | `phase0_validation()`        | SBERT consistency, dim check, canonicalization, PenGym loadability | ✅         |
| 1     | `phase1_sim_training()`      | Huấn luyện CRL trên sim với action_dim=16                          | ✅         |
| 2     | `phase2_domain_transfer()`   | Chuyển giao sim→PenGym qua DomainTransferManager                   | ✅         |
| 3     | `phase3_pengym_finetuning()` | Fine-tune trên PenGym với EWC constraints                          | ✅         |
| 4     | `phase4_evaluation()`        | Đánh giá đa-agent qua StrategyCEvaluator                           | ✅         |

**Tích hợp đã được wiring:**

- `PPO_Config(action_dim=ServiceActionSpace.DEFAULT_ACTION_DIM)` → Actor output 16-dim
- `ServiceActionSpace(action_class=Action)` được build trong Phase 1 và truyền vào mỗi HOST
- `DomainTransferManager` được gọi trong Phase 2 với agent và PenGym tasks
- `StrategyCEvaluator` được gọi trong Phase 4 với Forward/Backward Transfer metrics
- Kết quả lưu vào `dual_trainer_results.json` + `strategy_c_eval_report.json`

### 3.2 ServiceActionSpace — Không Gian Hành Động 16-dim

**File:** `src/agent/actions/service_action_space.py` (512 dòng)

Trừu tượng hóa ~2060 CVE thành 16 nhóm hành động mức service:

| Index | Tên                                                 | Loại    | PenGym mapping                  |
| ----- | --------------------------------------------------- | ------- | ------------------------------- |
| 0     | port_scan                                           | scan    | subnet_scan                     |
| 1     | service_scan                                        | scan    | service_scan                    |
| 2     | os_scan                                             | scan    | os_scan                         |
| 3     | web_scan                                            | scan    | process_scan                    |
| 4-12  | exploit_ssh/ftp/http/smb/smtp/rdp/sql/java_rmi/misc | exploit | e_ssh/e_ftp/e_http/...          |
| 13-15 | privesc_tomcat/schtask/daclsvc                      | privesc | pe_tomcat/pe_schtask/pe_daclsvc |

**Tính năng chính:**

- `_build_cve_groups()`: Phân loại tự động CVE vào nhóm service bằng keyword + port fallback
- `select_cve()`: Tier-2 selector với 4 strategies — **đã tích hợp vào `HOST.perform_action()`**
- `to_pengym_action()` / `from_pengym_action()`: ánh xạ hai chiều với PenGym
- Class constant `DEFAULT_ACTION_DIM = 16` dùng xuyên suốt pipeline

### 3.3 HOST — Simulation Host + Hierarchical Action

**File:** `src/agent/host.py` (366 dòng)

Host mô phỏng — mỗi máy mục tiêu là một instance HOST. Đã tích hợp hierarchical action:

- `__init__(..., service_action_space=None)`: nhận `ServiceActionSpace` tùy chọn
- `perform_action(action_mask)`: khi có `self.sas`, tự động gọi `select_cve()` để dịch action service-level (0..15) → CVE index trước khi thực thi
- `step(cve_index)`: thực thi hành động CVE cụ thể trên simulation
- Backward compatible: khi `sas=None` (gọi từ code cũ), hành vi không đổi

### 3.4 DomainTransferManager — Chuyển Giao Domain

**File:** `src/training/domain_transfer.py` (250 dòng)

Quản lý chuyển giao policy sim → PenGym với 3 chiến lược:

| Strategy         | Norm Reset | Fisher Discount | LR Giảm | Warmup | Use case     |
| ---------------- | ---------- | --------------- | ------- | ------ | ------------ |
| aggressive       | ❌         | 1.0             | 1.0     | ❌     | Sim ≈ PenGym |
| **conservative** | ✅         | β=0.3           | ×0.1    | ✅     | **Mặc định** |
| cautious         | ✅         | 0.0 (clear)     | ×0.1    | ✅     | Sim ≠ PenGym |

**Cơ chế:**

- `_reset_normalizer(agent)`: Reset running stats của normalizer
- `_collect_warmup_states(tasks, episodes)`: Random rollout trên PenGym thu thập states để warmup
- `_adjust_lr(agent, factor)`: Scale LR của actor/critic optimizers
- Gọi `ewc.discount_fisher(beta)` để giảm trọng số Fisher từ sim

### 3.5 OnlineEWC + discount_fisher()

**File:** `src/agent/continual/Script.py` (dòng 748-840)

EWC (Elastic Weight Consolidation) cho continual learning:

- `compute_importances()`: Tính Fisher Information Matrix cho task hiện tại
- `before_backward()`: Thêm EWC penalty vào loss
- `discount_fisher(beta)`: **Strategy C addition** — nhân tất cả Fisher values với β ∈ (0, 1] khi chuyển domain, nới lỏng ràng buộc EWC để agent thích nghi với PenGym

### 3.6 ScriptAgent — CRL 5 Trụ Cột

**File:** `src/agent/continual/Script.py` (966 dòng)

Kiến trúc Teacher-Student đầy đủ:

| Trụ cột                | Component                                            | Vị trí                               |
| ---------------------- | ---------------------------------------------------- | ------------------------------------ |
| Teacher Guidance       | Keeper → Explorer via `set_guide_policy()`           | `ScriptAgent.get_new_task_learner()` |
| KL Imitation           | `imi_loss` trong `ExplorePolicy.calcuate_ppo_loss()` | `ExplorePolicy._update()`            |
| Knowledge Distillation | KD loss trong `KnowledgeKeeper.compress()`           | `Keeper.compress()`                  |
| Retrospection          | `calculate_retrospection()`                          | `Keeper.compress()`                  |
| EWC                    | `OnlineEWC.before_backward()`                        | Fisher penalty                       |

### 3.7 StrategyCEvaluator — Đánh Giá Đa-Agent

**File:** `src/evaluation/strategy_c_eval.py` (252 dòng)

Đánh giá và so sánh nhiều agent trên cả sim và PenGym:

- `register_agent(name, agent_cl)`: Đăng ký agent
- `evaluate_all()`: Đánh giá tất cả agent trên tất cả domain
- `_compute_transfer_metrics()`: Tính Forward Transfer, Backward Transfer, Transfer Ratio, Zero-shot SR
- `print_report()`, `save_report()`: Xuất báo cáo

**Đã tích hợp vào DualTrainer Phase 4** — output lưu tại `strategy_c_eval_report.json`.

### 3.8 Configuration — Strategy C Parameters

**File:** `src/agent/policy/config.py`

`Script_Config` có đầy đủ tham số cho Strategy C:

```python
fisher_discount_beta = 0.3       # β ∈ [0.1, 0.5]
transfer_lr_factor = 0.1         # LR × factor sau transfer
norm_warmup_episodes = 10        # Episodes warmup normalizer
norm_reset_on_transfer = True    # Reset running stats khi chuyển domain
transfer_strategy = 'conservative'  # aggressive / conservative / cautious
```

`PPO_Config` hỗ trợ `state_dim=None` và `action_dim=None`:

- Khi set: PPO Actor/Critic dùng giá trị explicit
- Khi None: fallback về `StateEncoder.state_space` / `Action.action_space`
- DualTrainer set `action_dim=16`; `state_dim` fallback về 1538

### 3.9 PenGym Integration Layer

**SingleHostPenGymWrapper** (`src/envs/wrappers/single_host_wrapper.py`, 543 dòng):

- Bọc PenGym multi-host thành giao diện single-host tương thích SCRIPT
- Auto-advance qua các target, failure rotation (ngưỡng 5), subnet discovery
- Output: 1538-dim state via `PenGymStateAdapter.convert()`
- Reward: `LinearNormalizer` (PenGym [-1,100] → SCRIPT [-10,1000])

**PenGymHostAdapter** (`src/envs/adapters/pengym_host_adapter.py`, 253 dòng):

- Duck-typing giao diện HOST cho CRL
- Factory: `from_scenario(scenario_path, name, seed)`
- Lazy wrapper creation — xử lý NASim class-level state corruption giữa scenarios
- Interface: `reset() → state`, `perform_action(a) → (next_state, reward, done, result)`

**PenGymStateAdapter** (`src/envs/adapters/state_adapter.py`, 420 dòng):

- Chuyển đổi NASim flat observation → SBERT 1538-dim state vector
- SBERT cache cho performance
- Port inference từ service name
- Có method `convert_unified()` → 1540-dim (tồn tại nhưng chưa được gọi)

### 3.10 CLI Entry Point

**File:** `run_strategy_c.py` (225 dòng)

Entry point hoàn chỉnh với tất cả parameters:

```bash
python run_strategy_c.py \
    --sim-scenarios data/scenarios/chain/chain_1.json \
    --pengym-scenarios data/scenarios/tiny.yml \
    --transfer-strategy conservative \
    --fisher-beta 0.3 --lr-factor 0.1 \
    --episodes 500 --step-limit 100 \
    --train-scratch --seed 42
```

Output: `outputs/strategy_c/strategy_c_results.json` + TensorBoard logs + model checkpoints.

---

## 4. Thành Phần Triển Khai Một Phần

### 4.1 UnifiedStateEncoder — Code Hoàn Chỉnh, Chưa Tích Hợp

**File:** `src/envs/core/unified_state_encoder.py` (432 dòng)

**Đã triển khai code:**

- `encode_from_sim()`: Mã hóa từ SCRIPT simulation → 1540-dim
- `encode_from_pengym()`: Mã hóa từ PenGym observation → 1540-dim
- `pad_legacy_state()`: Chuyển đổi 1538-dim → 1540-dim
- Canonicalization maps: `CANONICAL_OS_MAP`, `CANONICAL_SERVICE_MAP`
- SBERT caching + dimension safety

**Cấu trúc 1540-dim:**

```
access[0:3]         — 3-dim (none/user/root) thay vì 2-dim
discovery[3:4]      — 1-dim (đã khám phá host?)
os_embed[4:388]     — 384-dim SBERT embedding
port_embed[388:772] — 384-dim SBERT embedding
service_embed[772:1156]  — 384-dim SBERT embedding
aux_embed[1156:1540]     — 384-dim SBERT embedding
```

**Chưa tích hợp:**

- `encode_from_sim()` **không được gọi** ở bất kỳ đâu trong pipeline
- `HOST.state_vector` vẫn dùng `StateEncoder` (1538-dim)
- `SingleHostPenGymWrapper` gọi `state_adapter.convert()` (1538-dim), không gọi `convert_unified()` (1540-dim)
- `PPO_Config.state_dim` không được set → fallback `StateEncoder.state_space = 1538`
- `DualTrainer.__init__()` tạo `self.unified_encoder` nhưng chỉ dùng trong Phase 0 validation

**Tác động:** Chuyển giao sim→PenGym hoạt động nhưng dựa vào sự tương đồng cấu trúc giữa hai bộ mã hóa riêng biệt (`StateEncoder` cho sim, `PenGymStateAdapter` cho PenGym) thay vì một bộ mã hóa thống nhất đảm bảo căn chỉnh ngữ nghĩa chính xác.

### 4.2 UnifiedNormalizer — Code Hoàn Chỉnh, Chưa Tích Hợp

**File:** `src/envs/wrappers/reward_normalizer.py` (dòng 123-168)

- Chuẩn hóa reward về [-1, +1] cho cả hai domain
- Simulation: max=1000, min=-10 → [-1, +1]
- PenGym: max=100, min=-3 → [-1, +1]

**Chưa tích hợp:** `SingleHostPenGymWrapper` mặc định dùng `LinearNormalizer` (PenGym → SCRIPT scale).

**Tác động:** Fisher information từ sim và PenGym có magnitude khác nhau do thang reward khác nhau → EWC penalty có thể bị lệch.

### 4.3 PenGymScriptTrainer — Standalone, Không Nằm Trong Dual Pipeline

**File:** `src/training/pengym_script_trainer.py` (378 dòng)

Trainer độc lập cho PenGym (không qua DualTrainer):

- `STATE_DIM = 1538`, `ACTION_DIM = 16`
- Dùng cho `run_pengym_train.py` — huấn luyện SCRIPT CRL từ đầu trên PenGym
- **Không** được DualTrainer sử dụng (Phase 3 dùng `Agent_CL.train_continually()` trực tiếp)

---

## 5. Thành Phần Chưa Tích Hợp Vào Pipeline

Các module dưới đây đã có code hoàn chỉnh nhưng chưa ảnh hưởng đến luồng training thực tế:

| Module                                       | Lý do chưa tích hợp                                                           | Ưu tiên                                  |
| -------------------------------------------- | ----------------------------------------------------------------------------- | ---------------------------------------- |
| `UnifiedStateEncoder` (1540-dim)             | HOST dùng StateEncoder (1538-dim), wrapper dùng PenGymStateAdapter (1538-dim) | P0 — Cần cho căn chỉnh ngữ nghĩa         |
| `UnifiedNormalizer` ([-1,+1])                | Wrapper mặc định LinearNormalizer, sim không normalize reward                 | P1 — Ảnh hưởng EWC Fisher quality        |
| `convert_unified()` trong PenGymStateAdapter | Wrapper gọi `convert()` thay vì `convert_unified()`                           | Tự động khi tích hợp UnifiedStateEncoder |
| `pad_legacy_state()`                         | Không cần nếu chuyển hoàn toàn sang 1540-dim                                  | Phụ thuộc UnifiedStateEncoder            |

---

## 6. Ánh Xạ File → Chức Năng

### Core Pipeline (tích hợp đầy đủ)

| File                                | Dòng | Chức năng                                    | Phase |
| ----------------------------------- | ---- | -------------------------------------------- | ----- |
| `run_strategy_c.py`                 | 225  | CLI entry point, parse args, gọi DualTrainer | —     |
| `src/training/dual_trainer.py`      | 592  | Orchestrator Phase 0→4                       | 0-4   |
| `src/training/domain_transfer.py`   | 250  | DomainTransferManager (3 strategies)         | 2     |
| `src/evaluation/strategy_c_eval.py` | 252  | StrategyCEvaluator (transfer metrics)        | 4     |

### Agent & Policy (tích hợp đầy đủ)

| File                            | Dòng | Chức năng                                     | Phase |
| ------------------------------- | ---- | --------------------------------------------- | ----- |
| `src/agent/host.py`             | 366  | HOST + hierarchical action via select_cve()   | 1     |
| `src/agent/agent_continual.py`  | 499  | Agent_CL — CRL task orchestrator              | 1, 3  |
| `src/agent/continual/Script.py` | 966  | ScriptAgent, EWC, discount_fisher()           | 1-3   |
| `src/agent/policy/PPO.py`       | 349  | PPO Actor/Critic (state_dim, action_dim)      | 1, 3  |
| `src/agent/policy/config.py`    | 218  | PPO_Config, Script_Config + Strategy C params | Init  |

### Action Space (tích hợp đầy đủ)

| File                                        | Dòng | Chức năng                                 | Phase |
| ------------------------------------------- | ---- | ----------------------------------------- | ----- |
| `src/agent/actions/service_action_space.py` | 512  | ServiceActionSpace (16-dim, CVE selector) | 1     |
| `src/agent/actions/Action.py`               | 181  | Action space gốc (~2064 = 4 scan + N CVE) | 1     |

### PenGym Integration (tích hợp đầy đủ)

| File                                       | Dòng | Chức năng                              | Phase |
| ------------------------------------------ | ---- | -------------------------------------- | ----- |
| `src/envs/wrappers/single_host_wrapper.py` | 543  | Wrapper PenGym → single-host interface | 3     |
| `src/envs/adapters/pengym_host_adapter.py` | 253  | Duck-typing HOST for PenGym            | 2-4   |
| `src/envs/adapters/state_adapter.py`       | 420  | PenGym obs → SBERT 1538-dim state      | 3     |

### Chưa tích hợp vào pipeline

| File                                     | Dòng | Chức năng                                   | Trạng thái    |
| ---------------------------------------- | ---- | ------------------------------------------- | ------------- |
| `src/envs/core/unified_state_encoder.py` | 432  | Unified 1540-dim encoder + canonicalization | Phase 0 only  |
| `src/envs/wrappers/reward_normalizer.py` | 168  | UnifiedNormalizer ([-1,+1])                 | Defined only  |
| `src/training/pengym_script_trainer.py`  | 378  | Standalone PenGym trainer                   | Separate tool |

---

## 7. Khoảng Cách So Với Thiết Kế Ban Đầu

### So sánh với đặc tả (`docs/strategy_C_shared_state_dual_training.md`)

| #   | Yêu cầu thiết kế                     | Trạng thái thực tế                                 | Mức độ   |
| --- | ------------------------------------ | -------------------------------------------------- | -------- |
| 1   | Pipeline Phase 0→1→2→3→4             | ✅ Hoàn chỉnh, tất cả phases functional            | Đầy đủ   |
| 2   | Unified State Encoder (1540-dim)     | ⚠️ Code xong, chưa tích hợp vào training           | Một phần |
| 3   | Hierarchical Action Space (16-dim)   | ✅ Hoàn chỉnh + select_cve() tích hợp              | Đầy đủ   |
| 4   | DomainTransferManager (3 strategies) | ✅ Hoàn chỉnh + tích hợp Phase 2                   | Đầy đủ   |
| 5   | Fisher Discount (β)                  | ✅ `discount_fisher(beta)` trong OnlineEWC         | Đầy đủ   |
| 6   | Normalizer Reset + Warmup            | ✅ reset stats + collect warmup states             | Đầy đủ   |
| 7   | SBERT Canonicalization               | ⚠️ Code xong trong UnifiedStateEncoder, chưa dùng  | Một phần |
| 8   | Reward Normalization [-1,+1]         | ⚠️ UnifiedNormalizer code xong, chưa dùng          | Một phần |
| 9   | 4-Agent Evaluation Matrix            | ✅ StrategyCEvaluator + DualTrainer Phase 4        | Đầy đủ   |
| 10  | Forward/Backward Transfer            | ✅ StrategyCEvaluator.\_compute_transfer_metrics() | Đầy đủ   |
| 11  | SBERT Consistency Check (Phase 0)    | ✅ Cosine similarity test                          | Đầy đủ   |
| 12  | PenGym Stability Check (Phase 0)     | ✅ Scenario loadability test                       | Đầy đủ   |
| 13  | CRL 5 Trụ Cột                        | ✅ Không thay đổi, hoạt động trên cả sim và PenGym | Đầy đủ   |
| 14  | Black/Grey/White-box modes           | ❌ Không triển khai (đã loại bỏ khỏi scope)        | N/A      |

### Chi tiết khoảng cách còn lại

**Gap 1: UnifiedStateEncoder chưa tích hợp vào luồng training**

Tất cả components sử dụng 1538-dim thay vì 1540-dim:

- `HOST.state_vector` = `StateEncoder` (2 + 4×384 = 1538)
- `SingleHostPenGymWrapper.step()` → `state_adapter.convert()` → 1538
- `PPO_Config.state_dim` = None → fallback `StateEncoder.state_space` = 1538

Để tích hợp cần:

1. HOST sim: Thay `StateEncoder` bằng `UnifiedStateEncoder.encode_from_sim()` hoặc tạo wrapper
2. PenGym: Wrapper gọi `convert_unified()` thay vì `convert()`
3. PPO: Set `state_dim=1540` trong DualTrainer
4. DomainTransfer warmup fallback: Đã sửa sang dynamic (dùng `StateEncoder.state_space`)

**Gap 2: UnifiedNormalizer chưa dùng**

Cần inject `UnifiedNormalizer` vào:

1. `SingleHostPenGymWrapper` (thay `LinearNormalizer`)
2. Sim training HOST (chuẩn hóa reward trước khi truyền cho agent)

**Gap 3: Canonicalization chưa active**

`UnifiedStateEncoder` có `CANONICAL_OS_MAP` và `CANONICAL_SERVICE_MAP` nhưng chỉ được sử dụng bên trong `encode_from_sim()` / `encode_from_pengym()` — vốn chưa được gọi.

---

## 8. Hướng Phát Triển Tiếp Theo

### Ưu Tiên P0: Tích Hợp UnifiedStateEncoder

**Mục tiêu:** Cả sim và PenGym sử dụng cùng bộ mã hóa 1540-dim.

**Phạm vi thay đổi:**

1. `host.py` — HOST sử dụng UnifiedStateEncoder thay vì StateEncoder cho state vector
2. `single_host_wrapper.py` — Gọi `convert_unified()` thay vì `convert()`
3. `dual_trainer.py` — Set `state_dim=1540`
4. Verify: PPO_agent properly receives 1540 input dim

### Ưu Tiên P1: Tích Hợp UnifiedNormalizer

**Mục tiêu:** Reward scale đồng nhất [-1, +1] cho cả hai domain.

**Phạm vi thay đổi:**

1. `single_host_wrapper.py` — Default normalizer = `UnifiedNormalizer(source='pengym')`
2. `host.py` — Normalize reward trước khi trả về (hoặc trong training loop)

### Ưu Tiên P2: Chạy Thử Nghiệm End-to-End

```bash
# Test nhanh
python run_strategy_c.py \
    --sim-scenarios data/scenarios/chain/chain-msfexp_vul-sample-6_envs-seed_0.json \
    --pengym-scenarios data/scenarios/tiny.yml \
    --episodes 50 --step-limit 30 --eval-freq 10 \
    --skip-phase0 --train-scratch

# Full run
python run_strategy_c.py \
    --sim-scenarios data/scenarios/chain/chain-msfexp_vul-sample-6_envs-seed_0.json \
    --pengym-scenarios data/scenarios/tiny.yml data/scenarios/small-linear.yml \
    --train-scratch
```

### Ưu Tiên P3: Thu Thập và Phân Tích Kết Quả

Sau khi chạy, kiểm tra:

- `phase4.transfer_metrics.forward_transfer > 0` → sim pre-training giúp ích
- `phase4.transfer_metrics.backward_transfer ≈ 0` → không quên sim
- So sánh `conservative` vs `cautious` vs `aggressive`
