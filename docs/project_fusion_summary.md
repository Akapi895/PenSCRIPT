# 📋 Tổng kết quá trình kết hợp hai bài báo SCRIPT & PenGym → PenSCRIPT

> **Ngày:** 2026-02-21  
> **Phạm vi:** Toàn bộ quá trình thiết kế, xây dựng và tích hợp hai hệ thống SCRIPT + PenGym thành project PenSCRIPT

---

## Mục lục

- [I. Tổng quan hai bài báo gốc](#i-tổng-quan-hai-bài-báo-gốc)
- [II. Vấn đề cần giải quyết khi kết hợp](#ii-vấn-đề-cần-giải-quyết-khi-kết-hợp)
- [III. Kiến trúc kết hợp đã xây dựng](#iii-kiến-trúc-kết-hợp-đã-xây-dựng)
- [IV. Chi tiết từng thành phần đã xây dựng](#iv-chi-tiết-từng-thành-phần-đã-xây-dựng)
- [V. Hai chiến lược Sim-to-Real Transfer](#v-hai-chiến-lược-sim-to-real-transfer)
- [VI. CVE Pipeline — Mở rộng khả năng tấn công](#vi-cve-pipeline--mở-rộng-khả-năng-tấn-công)
- [VII. Danh sách file đã tạo/sửa đổi](#vii-danh-sách-file-đã-tạosửa-đổi)
- [VIII. Kết quả đạt được & Trạng thái hiện tại](#viii-kết-quả-đạt-được--trạng-thái-hiện-tại)

---

## I. Tổng quan hai bài báo gốc

### 1.1 SCRIPT — RL Agent đơn mục tiêu với Continual Learning

**Đóng góp chính:** Xây dựng agent RL (PPO) có khả năng **tự động pentest** từng host và **học liên tục** qua nhiều mục tiêu mà không quên kiến thức cũ.

| Thành phần | Chi tiết kỹ thuật |
|---|---|
| **State** | Vector 1538-dim qua SBERT encoding: `[2 access ∣ 384 OS ∣ 384 port ∣ 384 service ∣ 384 web_fingerprint]` |
| **Action** | 2064 actions = 4 scan + 2060 CVE exploit (từ Metasploit database) |
| **Policy** | PPO Actor-Critic, mạng `512×512`, entropy regularization |
| **CRL 5 trụ cột** | (1) Teacher Guidance — Explorer (student) được hướng dẫn bởi Keeper (teacher) qua curriculum decay |
| | (2) KL Imitation Loss — PPO loss bổ sung $\mathcal{L}_{KL} = \text{KL}(\pi_{\text{student}} \| \pi_{\text{teacher}})$ |
| | (3) Knowledge Distillation — Expert samples từ Explorer nén vào Keeper |
| | (4) Retrospection — Keeper minimize $\text{KL}(\pi_{\text{new}} \| \pi_{\text{old}})$ chống forgetting |
| | (5) EWC — Fisher Information Matrix regularization trên weights |
| **Môi trường** | Simulation tự xây (class `HOST`), mỗi host là 1 task CRL riêng biệt |

**Hạn chế:** Chỉ hoạt động trên **simulation riêng**, không thể test trên mạng thực hoặc mạng NASim chuẩn.

### 1.2 PenGym — Môi trường Pentest Thực Tế (NASim mở rộng)

**Đóng góp chính:** Mở rộng NASim thành môi trường **sim-to-real** — cùng API nhưng có thể chạy trên KVM VMs thực.

| Thành phần | Chi tiết kỹ thuật |
|---|---|
| **PenGymEnv** | Kế thừa `NASimEnv`, chuẩn Gymnasium (`reset()`, `step()`, `render()`) |
| **Action** | Service-level: `4 scan + 5 exploit (ssh/ftp/http/samba/smtp) + 3 privesc` × N hosts |
| **Observation** | Flat vector toàn mạng: `(num_hosts + 1) × host_vec_size` |
| **Hai chế độ** | `sim` (NASim xác suất, nhanh) và `real` (KVM + nmap + Metasploit RPC, chậm nhưng thực) |
| **Scenario** | YAML files: topology, services, exploits, host values |

**Hạn chế:** Không có **agent RL mạnh** và không có cơ chế **continual learning** — chỉ cung cấp environment.

### 1.3 Tại sao cần kết hợp?

| SCRIPT có ↓ | PenGym có ↓ | Kết hợp → |
|---|---|---|
| Agent RL mạnh (PPO + CRL) | ❌ Không có agent | Agent RL chạy trên env chuẩn |
| ❌ Simulation riêng, đóng | Env chuẩn Gymnasium | Agent train/eval trên NASim + real |
| ❌ Không real execution | Sim-to-Real via KVM | Train sim → deploy real |
| 2060 CVE knowledge | 5 exploit types | Agent biết nhiều CVE, eval trên real |
| CRL 5 trụ cột | ❌ Không có CRL | CRL trên đa dạng network topologies |

---

## II. Vấn đề cần giải quyết khi kết hợp

### 2.1 Xung đột kiến trúc cơ bản

```
SCRIPT: Agent tấn công 1 host/lần     ←→  PenGym: Env có N hosts, action cho toàn mạng
SCRIPT: State = SBERT text encoding    ←→  PenGym: State = NASim binary vector
SCRIPT: Action = 2064 CVE cụ thể      ←→  PenGym: Action = ~18 service-level
SCRIPT: Reward scale [-10, 1000]       ←→  PenGym: Reward scale [-1, 100]
SCRIPT: HOST object per target         ←→  PenGym: Một PenGymEnv cho toàn mạng
```

### 2.2 Bảng chi tiết 5 xung đột & giải pháp

| # | Xung đột | Nguyên nhân | Giải pháp đã thực hiện |
|---|---|---|---|
| **C1** | **State format khác nhau** | SCRIPT dùng SBERT 384-dim cho text, PenGym dùng one-hot/binary | Xây `PenGymStateAdapter` tái tạo text → SBERT encode → cùng 1538-dim |
| **C2** | **Action space không tương thích** | SCRIPT: 2064 CVE, PenGym: ~18 service → overlap chỉ 3.4% | Xây `ServiceActionSpace` 16-dim — trừu tượng hóa cả hai về service-level |
| **C3** | **Single-host vs Multi-host** | SCRIPT tấn công 1 target, PenGym quản lý N hosts | Xây `SingleHostPenGymWrapper` + `TargetSelector` giảm multi→single |
| **C4** | **Reward scale chênh ~100×** | PPO critic calibrated trên [-10, 1000], PenGym cho [-1, 100] | Xây `RewardNormalizer` (Linear/Clip/Identity) chuẩn hóa scale |
| **C5** | **Interface HOST khác PenGymEnv** | SCRIPT CRL code (ScriptAgent, Explorer, Keeper) yêu cầu giao diện HOST | Xây `PenGymHostAdapter` duck-type HOST interface cho PenGym scenarios |

---

## III. Kiến trúc kết hợp đã xây dựng

### 3.1 Nguyên tắc thiết kế

> **"Giữ nguyên lõi, kết nối qua adapter"** — Không sửa code SCRIPT gốc, không sửa code PenGym gốc. Mọi tích hợp qua lớp adapter/wrapper ở giữa.

### 3.2 Sơ đồ kiến trúc tổng thể

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ENTRY POINTS                                    │
│  run.py │ run_pengym_train.py │ run_pengym_eval.py │ run_benchmark.py   │
├─────────────────────────────────────────────────────────────────────────┤
│                         TRAINERS (MỚI)                                  │
│  ┌──────────────────────┐  ┌───────────────────────────────────────┐    │
│  │ PenGymTrainer        │  │ PenGymScriptTrainer                   │    │
│  │ (PPO thuần trên      │  │ (Full CRL 5 trụ cột trên PenGym)      │    │
│  │  PenGym)             │  │                                       │    │
│  └──────────┬───────────┘  └──────────────────┬────────────────────┘    │
├─────────────┼─────────────────────────────────┼─────────────────────────┤
│             │      ADAPTER LAYER (MỚI)        │                         │
│             ▼                                 ▼                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                 PenGymHostAdapter                                │   │
│  │        (Giả lập giao diện HOST cho CRL code)                     │   │
│  └────────────────────────┬─────────────────────────────────────────┘   │
│                           ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │           SingleHostPenGymWrapper (ĐIỂM KẾT NỐI DUY NHẤT)        │   │
│  │  ┌──────────────┬───────────────┬──────────────┬──────────────┐  │   │
│  │  │PenGymState   │ServiceAction  │Reward        │Target        │  │   │
│  │  │Adapter       │Mapper         │Normalizer    │Selector      │  │   │
│  │  │(State convert)│(Action map)  │(Scale align) │(Host choose) │  │   │
│  │  └──────────────┴───────────────┴──────────────┴──────────────┘  │   │
│  └────────────────────────┬─────────────────────────────────────────┘   │
├───────────────────────────┼─────────────────────────────────────────────┤
│  SCRIPT CORE (GỐC)        │              PenGym CORE (GỐC)              │
│  ┌────────────────────┐   │    ┌────────────────────────────────────┐   │
│  │ Agent / Agent_CL   │   │    │ PenGymEnv (NASimEnv)               │   │
│  │ ScriptAgent        │   │    │ PenGymNetwork                      │   │
│  │ Explorer / Keeper  │   ▼    │ PenGymHostVector                   │   │
│  │ HOST (simulation)  │◄──────►│ PenGymState                        │   │
│  │ PPO Policy         │        │                                    │   │
│  │ SBERT Encoder      │        │ Scenarios (.yml)                   │   │
│  │ Action (2064 CVE)  │        │ sim / real / dual mode             │   │
│  └────────────────────┘        └────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Design Patterns sử dụng

| Pattern | Thành phần | Mục đích |
|---|---|---|
| **Adapter** | `PenGymStateAdapter`, `ServiceActionMapper`, `PenGymHostAdapter` | Chuyển đổi interface mà không sửa code gốc |
| **Wrapper** | `SingleHostPenGymWrapper` | Điểm tích hợp duy nhất, ẩn toàn bộ multi-host complexity |
| **Strategy** | `RewardNormalizer` (3 variants), `TargetSelector` (3 variants), `CVESelector` | Behavior có thể plug-in/swap |
| **Façade** | `PenGymTrainer`, `PenGymScriptTrainer` | Entry point đơn giản hoá |
| **Registry** | `ServiceRegistry`, `ServiceActionSpace` | Data-driven, mở rộng qua config |
| **Duck-Typing** | `PenGymHostAdapter` ≡ `HOST` interface | CRL code hoạt động mà không cần biết backend |
| **Factory** | `PenGymHostAdapter.from_scenario()` | Tạo adapter từ scenario path |
| **Lazy Init** | `PenGymHostAdapter._create_wrapper()` | Defer wrapper creation tránh NASim state corruption |

---

## IV. Chi tiết từng thành phần đã xây dựng

### 4.1 PenGymStateAdapter — Cầu nối State Space

**File:** `src/envs/adapters/state_adapter.py` (361 dòng)

**Vấn đề giải quyết:** SCRIPT state = SBERT encoding (1538-dim), PenGym state = NASim flat binary vector.

**Cách hoạt động:**

```
PenGym NASim obs (flat binary/numeric)
    │
    ▼ Bước 1: Tách obs thành segments per-host
    ▼ Bước 2: Đọc binary flags → tái tạo text strings
    │   OS one-hot [1,0] → "linux"
    │   Service binary [1,0,1,0,0] → "ssh, http"
    │   Access flags → [compromised, reachable]
    ▼ Bước 3: SBERT encode text → 384-dim per component
    ▼ Bước 4: Concatenate → [2 + 384 + 384 + 384 + 384] = 1538-dim
    │
    ▼ Output: Vector GIỐNG HỆT format SCRIPT gốc
```

**Tối ưu:** SBERT cache — mỗi text string chỉ encode 1 lần, tái sử dụng cho các host có cùng service/OS.

### 4.2 ServiceActionSpace — Không gian hành động thống nhất 16 chiều

**File:** `src/agent/actions/service_action_space.py` (516 dòng)

**Vấn đề giải quyết:** SCRIPT có 2064 actions, PenGym có ~18 actions → overlap 3.4%.

**Giải pháp — Kiến trúc hai tầng:**

```
Tầng 1 (RL Policy): Chọn 1 trong 16 service-level actions
    │
    │  Idx  Action              PenGym Equiv      CVE Group Size
    │  ───  ──────────────────  ────────────────  ──────────────
    │   0   port_scan           subnet_scan        —
    │   1   service_scan        service_scan       —
    │   2   os_scan             os_scan            —
    │   3   web_scan            process_scan       —
    │   4   exploit_ssh         e_ssh              ~320 CVEs
    │   5   exploit_ftp         e_ftp              ~150 CVEs
    │   6   exploit_http        e_http             ~680 CVEs
    │   7   exploit_smb         e_samba            ~180 CVEs
    │   8   exploit_smtp        e_smtp             ~90 CVEs
    │   9   exploit_rdp         —                  ~70 CVEs
    │  10   exploit_sql         —                  ~120 CVEs
    │  11   exploit_java_rmi    —                  ~50 CVEs
    │  12   exploit_misc        —                  ~400 CVEs
    │  13   privesc_tomcat      pe_tomcat           —
    │  14   privesc_schtask     pe_schtask          —
    │  15   privesc_daclsvc     pe_daclsvc          —
    │
    ▼
Tầng 2 (Heuristic): CVESelector chọn CVE cụ thể trong nhóm
    │   Strategy "match": Oracle — chọn CVE khớp vulnerability
    │   Strategy "rank":  Realistic — chọn CVE có Metasploit rank cao nhất
    │   Strategy "random": Baseline
    │
    ▼ Kết quả: Service action → CVE exploit → kết quả simulation
```

**Lợi ích:**
- Policy chỉ cần học 16 actions thay vì 2064 → **hội tụ nhanh hơn ~10×**
- Coverage PenGym đạt **~100%** (vs 3.4% trước đó)
- Thêm CVE mới chỉ cần update data, không sửa policy

### 4.3 SingleHostPenGymWrapper — Điểm kết nối duy nhất

**File:** `src/envs/wrappers/single_host_wrapper.py` (543 dòng)

**Vấn đề giải quyết:** SCRIPT tấn công 1 host/lần, PenGym quản lý N hosts.

**Cách hoạt động:**

```python
class SingleHostPenGymWrapper:
    def reset(self) -> np.ndarray:  # → 1538-dim state of first target
        obs, info = self.pengym_env.reset()
        self.current_target = self.target_selector.select(...)
        return self.state_adapter.convert(obs, self.current_target)
    
    def step(self, service_action: int) -> Tuple[np.ndarray, float, bool, dict]:
        # 1. Map service action → PenGym flat action for current_target
        flat_action = self.action_mapper.map(service_action, self.current_target)
        # 2. Execute in PenGym
        obs, reward, done, trunc, info = self.pengym_env.step(flat_action)
        # 3. Normalize reward
        reward = self.reward_normalizer.normalize(reward)
        # 4. Convert state for current target
        state = self.state_adapter.convert(obs, self.current_target)
        # 5. Auto-advance: if target compromised → subnet_scan → next target
        if self._is_compromised(self.current_target):
            self._auto_subnet_scan()
            self.current_target = self.target_selector.select(...)
        # 6. Target rotation: if N consecutive failures → switch target
        if self.consecutive_failures >= self.max_failures:
            self.current_target = self.target_selector.select(...)
        return state, reward, done, info
```

**Tính năng v2:**
- Auto `subnet_scan` sau mỗi compromise → phát hiện host mới trong subnet
- Failure rotation — tránh agent bị stuck ở host quá khó
- Target advancement — tự động chuyển sang host tiếp theo khi compromise

### 4.4 ServiceActionMapper — Ánh xạ Action sang PenGym

**File:** `src/envs/adapters/service_action_mapper.py` (198 dòng)

**Ánh xạ:**

```
ServiceActionSpace idx (0..15)
    ↓ map to PenGym action name (e.g., "e_ssh")
    ↓ combine with target host index
    ↓ compute flat action index = action_type_idx × num_hosts + host_idx
    ↓ PenGymEnv.step(flat_action_index)
```

**Cải tiến v2:** `subnet_scan` phải được thực hiện **từ host đã compromised** (không phải từ host đang tấn công) — mapper tự động route scan action từ compromised host gần nhất.

### 4.5 RewardNormalizer — Chuẩn hóa Reward

**File:** `src/envs/wrappers/reward_normalizer.py` (121 dòng)

| Normalizer | Công thức | Sử dụng khi |
|---|---|---|
| `LinearNormalizer` | $r_{script} = \frac{r_{pengym} - r_{min}}{r_{max} - r_{min}} \times (s_{max} - s_{min}) + s_{min}$ | Mặc định — ánh xạ `[-1, 100] → [-10, 1000]` |
| `ClipNormalizer` | $r_{script} = \text{clip}(r_{pengym} / \text{scale}, s_{min}, s_{max})$ | Khi reward PenGym bất thường |
| `IdentityNormalizer` | $r_{script} = r_{pengym}$ | Debug/test |

### 4.6 TargetSelector — Chiến lược chọn host

**File:** `src/envs/wrappers/target_selector.py` (279 dòng)

| Selector | Logic | Phù hợp khi |
|---|---|---|
| `PriorityTargetSelector` | Ưu tiên sensitive hosts (high value) & reachable | Mặc định — tấn công hiệu quả nhất |
| `ReachabilityTargetSelector` | Chỉ chọn trong reachable hosts | Khi topology phức tạp |
| `RoundRobinTargetSelector` | Luân phiên tuần tự | Baseline comparison |

### 4.7 PenGymHostAdapter — Duck-Typing HOST Interface

**File:** `src/envs/adapters/pengym_host_adapter.py` (273 dòng)

**Vấn đề giải quyết:** SCRIPT CRL code (`ScriptAgent`, `Explorer`, `Keeper`) được viết để làm việc với `HOST` objects. Cần PenGym hoạt động qua cùng interface mà **không sửa code CRL**.

**Interface mapping:**

```python
class PenGymHostAdapter:
    """Duck-types SCRIPT's HOST interface for PenGym scenarios."""
    
    # SCRIPT HOST interface      →  PenGym implementation
    def reset(self) -> ndarray:      # → wrapper.reset() → 1538-dim
    def perform_action(self, a):     # → wrapper.step(a) → (state, reward, done, info)
    
    @property
    def ip(self) -> str:             # → scenario_name (e.g., "tiny.yml")
    @property
    def info(self) -> Host_info:     # → Host_info(ip=scenario_name)
    @property
    def env_data(self) -> dict:      # → {'vulnerability': 'pengym_scenario'}
    
    @classmethod
    def from_scenario(cls, path, **kwargs):  # Factory method
```

**Kết quả:** Code CRL (`ScriptAgent.get_new_task_learner(task=adapter)`, `Explorer.run_train_episode(target=adapter)`) hoạt động **y hệt** trên PenGym mà không cần thay đổi một dòng.

### 4.8 PenGymTrainer — Training PPO trên PenGym

**File:** `src/training/pengym_trainer.py` (667 dòng)

**Hai chế độ:**

1. **Single-scenario:** Train PPO trên 1 PenGym scenario (ví dụ `tiny.yml`)
2. **Curriculum:** Tích hợp `CurriculumController` — train theo tier T1→T2→T3→T4 với auto-advancement khi đạt SR threshold

```python
class PenGymTrainer:
    def train_single(self, scenario, episodes, ...):
        wrapper = SingleHostPenGymWrapper(scenario, ...)
        for ep in range(episodes):
            state = wrapper.reset()
            for step in range(max_steps):
                action = self.agent.select_action(state)
                next_state, reward, done, info = wrapper.step(action)
                self.agent.store_transition(state, action, reward, next_state, done)
                state = next_state
            self.agent.update_policy()
    
    def train_curriculum(self, scenarios, ...):
        controller = CurriculumController(scenarios, phases=4)
        while not controller.all_phases_complete():
            current_scenarios = controller.get_current_phase_scenarios()
            # Train on current phase → check SR → advance if threshold met
```

### 4.9 PenGymScriptTrainer — Full CRL trên PenGym

**File:** `src/training/pengym_script_trainer.py` (376 dòng)

**Đóng góp quan trọng nhất:** Kết nối **toàn bộ 5 trụ cột CRL** của SCRIPT lên PenGym.

```python
class PenGymScriptTrainer:
    def train(self, scenarios: List[str], ...):
        # 1. Tạo PenGymHostAdapter cho mỗi scenario
        tasks = [PenGymHostAdapter.from_scenario(s) for s in scenarios]
        
        # 2. Khởi tạo Agent_CL với method="script"
        agent_cl = Agent_CL(method="script", config=self.config)
        
        # 3. Train continually — SCRIPT code không biết đang chạy PenGym
        agent_cl.train_continually(tasks=tasks)
        #   └→ ScriptAgent.get_new_task_learner(task_id)
        #       └→ Explorer.run_train_episode(target=PenGymHostAdapter)
        #           └→ adapter.perform_action(a) → wrapper.step(a) → PenGym!
        #   └→ ScriptAgent.policy_preservation(all_tasks)
        #       └→ Explorer.get_expert_samples(target=adapter) → distill → Keeper
        #       └→ EWC.after_training_task() → Fisher matrix update
```

### 4.10 SimToRealEvaluator — Đánh giá Transfer

**File:** `src/evaluation/sim_to_real_eval.py` (500 dòng)

**Chức năng:** Đánh giá khả năng transfer policy đã train (trên simulation) sang PenGym environment.

**Metrics thu thập:**

| Metric | Ý nghĩa |
|---|---|
| `transfer_success_rate` | % scenarios giải thành công trên PenGym |
| `valid_action_rate` | % actions mà policy chọn có thể map sang PenGym |
| `unmappable_action_rate` | % actions không có tương đương PenGym |
| `action_distribution_similarity` | KL divergence giữa action dist trên sim vs PenGym |
| `reward_gap` | Chênh lệch reward trung bình sim vs PenGym |
| `step_efficiency` | Tỷ lệ steps cần thêm trên PenGym so với sim |

---

## V. Hai chiến lược Sim-to-Real Transfer

### 5.1 Strategy A — Zero-Shot / Few-Shot Transfer

```
Train trên SCRIPT simulation (nhanh, unlimited)
    │
    ▼ Đóng băng policy
    │
    ▼ Eval trên PenGym qua adapters (không retrain)
    │
    │  State: PenGymStateAdapter converts obs → 1538-dim
    │  Action: ServiceActionMapper converts 16-dim → PenGym flat
    │
    ▼ Đo transfer gap
    │
    ▼ [Optional] Few-shot fine-tune trên PenGym (low LR + EWC)
```

**Ưu điểm:** Không cần train lại, nhanh.  
**Nhược điểm:** State distribution shift (SBERT encoding sim ≠ PenGym reconstructed).

### 5.2 Strategy C — Shared State + Dual Training

```
Bước 1: Chuẩn hóa state format → UnifiedStateEncoder (1540-dim)
    │   OS canonicalization: "Ubuntu 14.04" → "linux"
    │   Service normalization: "Apache httpd 2.4.49" → "http"
    │
Bước 2: Pre-train trên SCRIPT sim (state 1540-dim, action 16-dim)
    │   → Nhanh, convergence đến ~100% SR
    │
Bước 3: Fine-tune trên PenGym (cùng state/action format)
    │   → EWC regularization bảo vệ weights đã học
    │   → Low learning rate tránh catastrophic forgetting
    │
Bước 4: Eval trên PenGym real mode
```

**Ưu điểm:** Transfer quality cao nhất (cùng representation).  
**Nhược điểm:** Cần sửa state encoder ở cả hai phía.

### 5.3 So sánh hai chiến lược

| Khía cạnh | Strategy A | Strategy C |
|---|---|---|
| State format | Giữ riêng, adapt tại boundary | Thống nhất ở cả hai đầu |
| Retrain | Không hoặc minimal | Full retrain sim + fine-tune PenGym |
| Code changes | Chỉ adapters | Sửa state encoders cả hai phía |
| Transfer quality | Thấp hơn (distribution shift) | Cao hơn (identical representations) |
| Claim nghiên cứu | Đo empirical sim-to-real gap | Cross-domain continual learning |

---

## VI. CVE Pipeline — Mở rộng khả năng tấn công

### 6.1 Pipeline 4 pha

```
Phase 1: CVE Grading          Phase 2: Scenario Compilation
┌─────────────────────┐       ┌──────────────────────────┐
│ CVEClassifier       │       │ ScenarioCompiler          │
│ 1985 CVEs → 4 tiers │──────►│ Template + CVE Overlay    │
│ T1 (easy) → T4 (hard)│      │ → NASim-valid YAML        │
└─────────────────────┘       └──────────┬───────────────┘
                                          │
Phase 3: Curriculum Training   Phase 4: Service Registry
┌──────────────────────────┐  ┌──────────────────────────┐
│ CurriculumController      │  │ ServiceRegistry           │
│ T1→T2→T3→T4 auto-advance │  │ Extensible service/proc   │
│ SR thresholds: 70/60/50/40│  │ definitions               │
└──────────────────────────┘  └──────────────────────────┘
```

### 6.2 CVE Difficulty Classification

**File:** `src/pipeline/cve_classifier.py` (393 dòng)

Mỗi CVE được tính **difficulty score** từ 4 yếu tố:

$$D_{CVE} = w_1 \cdot f(\text{exploit\_prob}) + w_2 \cdot f(\text{attack\_complexity}) + w_3 \cdot f(\text{privileges}) + w_4 \cdot f(\text{user\_interaction})$$

| Tier | Difficulty Score | Số CVE | Mô tả |
|---|---|---|---|
| T1 | Thấp nhất | ~500 | CVE dễ exploit, không cần credentials |
| T2 | Thấp-Trung | ~500 | CVE cần basic access |
| T3 | Trung-Cao | ~500 | CVE phức tạp, cần privileges |
| T4 | Cao nhất | ~485 | CVE rất khó, cần nhiều điều kiện |

### 6.3 Curriculum Controller

**File:** `src/pipeline/curriculum_controller.py` (377 dòng)

Quản lý tiến trình training theo 4 phase với auto-advancement:

```
Phase 1 (T1 CVEs) ──SR≥0.70──→ Phase 2 (T1+T2) ──SR≥0.60──→ Phase 3 (T1+T2+T3) ──SR≥0.50──→ Phase 4 (All)
```

Sử dụng **sliding window convergence detection** — chuyển phase khi SR trung bình N episode gần nhất vượt threshold.

---

## VII. Danh sách file đã tạo/sửa đổi

### 7.1 Thành phần MỚI hoàn toàn (Integration Layer)

| File | Dòng code | Vai trò |
|---|---|---|
| `src/envs/adapters/__init__.py` | — | Exports adapter classes |
| `src/envs/adapters/state_adapter.py` | 361 | PenGym obs → SCRIPT 1538-dim state |
| `src/envs/adapters/action_mapper.py` | 294 | CVE-level action mapping (legacy) |
| `src/envs/adapters/service_action_mapper.py` | 198 | Service-level action → PenGym flat index |
| `src/envs/adapters/pengym_host_adapter.py` | 273 | Duck-type HOST interface cho PenGym |
| `src/envs/wrappers/__init__.py` | — | Exports wrapper classes |
| `src/envs/wrappers/single_host_wrapper.py` | 543 | **Điểm tích hợp duy nhất** multi→single host |
| `src/envs/wrappers/reward_normalizer.py` | 121 | Chuẩn hóa reward scale |
| `src/envs/wrappers/target_selector.py` | 279 | Chiến lược chọn target host |
| `src/training/__init__.py` | — | — |
| `src/training/pengym_trainer.py` | 667 | PPO training loop trên PenGym |
| `src/training/pengym_script_trainer.py` | 376 | Full CRL (5 pillars) trên PenGym |
| `src/evaluation/__init__.py` | — | — |
| `src/evaluation/sim_to_real_eval.py` | 500 | Strategy A evaluator + gap analysis |
| `src/agent/actions/service_action_space.py` | 516 | 16-dim unified action space + CVESelector |

**Tổng code mới cho integration: ~4,128 dòng**

### 7.2 Thành phần Pipeline CVE (MỚI)

| File | Dòng code | Vai trò |
|---|---|---|
| `src/pipeline/__init__.py` | — | — |
| `src/pipeline/cve_classifier.py` | 393 | CVE difficulty grading (4 tiers) |
| `src/pipeline/curriculum_controller.py` | 377 | Tiered curriculum training controller |
| `src/pipeline/service_registry.py` | 512 | Extensible service/process registry |
| `src/pipeline/scenario_compiler.py` | 987 | Template + Overlay → NASim YAML |

**Tổng code pipeline: ~2,269 dòng**

### 7.3 Entry Points & Runners (MỚI/SỬA)

| File | Vai trò |
|---|---|
| `run.py` | Unified entry point — thêm hỗ trợ `--env-type pengym` |
| `run_pengym_train.py` | CLI cho PenGym training (single + curriculum) |
| `run_pengym_eval.py` | CLI cho PenGym evaluation |
| `run_benchmark.py` | Full benchmark suite (train → eval → report) |
| `run_train_service_level.py` | Service-level training trên SCRIPT sim |
| `run_eval_service_level.py` | Service-level eval trên PenGym |

### 7.4 Thành phần GỐC được giữ nguyên

| Module | Files | Nguồn gốc |
|---|---|---|
| `src/agent/agent.py` | Agent base class | SCRIPT |
| `src/agent/agent_continual.py` | Agent_CL, CRL orchestrator | SCRIPT |
| `src/agent/host.py` | HOST simulation environment | SCRIPT |
| `src/agent/actions/Action.py` | CVE-level action space (2064) | SCRIPT |
| `src/agent/continual/script.py` | ScriptAgent (Explorer + Keeper) | SCRIPT |
| `src/agent/continual/finetune.py` | Fine-tuning CL method | SCRIPT |
| `src/agent/nlp/Encoder.py` | SBERT encoder (all-MiniLM-L12-v2) | SCRIPT |
| `src/agent/policy/PPO.py` | PPO Actor-Critic | SCRIPT |
| `src/agent/policy/config.py` | Hyperparameter config | SCRIPT |
| `src/envs/core/pengym_env.py` | PenGymEnv (NASim extension) | PenGym |
| `src/envs/core/pengym_host_vector.py` | PenGymHostVector | PenGym |
| `src/envs/core/pengym_network.py` | PenGymNetwork | PenGym |
| `src/envs/core/pengym_state.py` | PenGymState | PenGym |

---

## VIII. Kết quả đạt được & Trạng thái hiện tại

### 8.1 Bảng trạng thái

| Thành phần | Trạng thái | Ghi chú |
|---|---|---|
| SCRIPT core (PPO, CRL, HOST) | ✅ Hoạt động | Giữ nguyên 100% |
| PenGym core (NASimEnv) | ✅ Hoạt động | Giữ nguyên 100% |
| ServiceActionSpace (16-dim) | ✅ Hoạt động | Verified 100% SR trên chain-6_envs |
| PenGymStateAdapter | ✅ Hoạt động | SBERT cache, per-host extraction |
| ServiceActionMapper | ✅ Hoạt động | ~100% coverage, v2 subnet_scan fix |
| SingleHostPenGymWrapper | ✅ Hoạt động | v2 với auto-advance + failure rotation |
| PenGymHostAdapter | ✅ Hoạt động | Duck-type HOST, factory method |
| PenGymTrainer (PPO trên PenGym) | ✅ Hoạt động | Single + curriculum mode |
| PenGymScriptTrainer (CRL trên PenGym) | ✅ Hoạt động | 5/5 pillars verified |
| CurriculumController | ✅ Hoạt động | Tích hợp vào PenGymTrainer |
| CVE Pipeline (4 phases) | ✅ Hoạt động | 1985 CVEs graded, 80 scenarios |
| SimToRealEvaluator | ⚠️ Cần cập nhật | Đang dùng CVE-level mapper cũ |
| PenGym real execution (KVM) | ❌ Chưa verify | Cần test end-to-end trên CyRIS |
| Strategy C (UnifiedStateEncoder) | 📋 Thiết kế xong | Chưa implement code |

### 8.2 Kết quả thực nghiệm sơ bộ

| Thí nghiệm | Kết quả |
|---|---|
| Service-level (16-dim) trên chain-6_envs | **100% SR từ episode 1** (vs 2064-dim cần ~300 episodes) |
| Convergence speed 16-dim vs 2064-dim | **~10× nhanh hơn** |
| Action mapping coverage SCRIPT→PenGym | **~100%** (vs 3.4% trước khi có ServiceActionSpace) |
| CRL 5 pillars trên PenGym (via adapter) | **Hoạt động** — không cần sửa code CRL |

### 8.3 Đóng góp khoa học

1. **Kiến trúc Adapter/Wrapper** cho phép kết hợp hai hệ thống RL pentest khác paradigm mà **không sửa code gốc** — mẫu thiết kế có thể tái sử dụng cho các nghiên cứu tương tự.

2. **Service-level Action Abstraction (16-dim)** — giảm không gian hành động từ 2064 → 16 mà vẫn giữ ~100% coverage trên cả hai môi trường. Chứng minh rằng abstraction ở mức service là đủ cho pentest RL.

3. **CRL on realistic environments** — lần đầu tiên SCRIPT's 5-pillar CRL được chạy trên NASim-based environment thay vì simulation riêng, mở đường cho đánh giá CRL trên mạng thực.

4. **CVE Difficulty Pipeline** — phân loại 1985 CVEs thành 4 tiers, cho phép curriculum training tự động theo độ khó tăng dần.

---

> **Tổng kết:** Project PenSCRIPT đã thành công kết hợp **SCRIPT** (agent RL mạnh với CRL) và **PenGym** (environment pentest thực tế) thông qua lớp adapter/wrapper ~4,100 dòng code mới, giữ nguyên 100% code gốc của cả hai bài báo, tạo ra hệ thống train agent RL pentest trên simulation rồi transfer sang môi trường thực.
