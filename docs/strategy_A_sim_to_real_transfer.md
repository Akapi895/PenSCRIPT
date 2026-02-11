# Chiến Lược A: Sim-to-Real Transfer

> Train CL on Simulation → Transfer Policy → Evaluate on PenGym

---

## 1. Tổng Quan

### 1.1 Mục Tiêu

Đánh giá khả năng **zero-shot transfer** và **few-shot adaptation** của SCRIPT agent (đã huấn luyện trên môi trường simulation) khi triển khai trực tiếp trên PenGym — một cyber range thực tế sử dụng Metasploit/nmap trên các VM ảo hóa qua KVM.

**Câu hỏi nghiên cứu chính:**

1. Policy đã học trên simulation có thể tạo ra hành vi hợp lý trên PenGym ở mức nào?
2. Performance gap (sim vs. real) lớn bao nhiêu, và nguyên nhân chủ yếu đến từ đâu?
3. Liệu fine-tuning nhẹ trên PenGym có đủ để thu hẹp khoảng cách này không?

### 1.2 Phạm Vi Áp Dụng

| Phạm vi             | Mô tả                                                                                                                      |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **Input**           | Model checkpoint đã train: `outputs_baseline_sim/models_baseline_sim/chain/chain-msfexp_vul-sample-6_envs-seed_0/PPO-*.pt` |
| **Môi trường đích** | PenGym với các NASim-compatible scenario (tiny, small-linear, medium)                                                      |
| **Đầu ra**          | Báo cáo đánh giá performance, gap analysis, và quyết định có nên tiến tới Strategy C                                       |
| **Không bao gồm**   | Thay đổi kiến trúc mạng neural, thay đổi state encoding, retrain từ đầu                                                    |

### 1.3 Giả Định Nền Tảng

1. **Model đã converged trên simulation:** Experiment summary xác nhận success_rate = 1.0 trên 6 targets, converge tại episode 39.
2. **State encoding tương thích cơ bản:** Cả hai môi trường đều có thể biểu diễn thông tin dưới dạng vector số, nhưng format cụ thể **khác nhau đáng kể** (xem Section 3).
3. **PenGym environment đã sẵn sàng:** CyRIS cyber range đã cấu hình, Metasploit RPC và nmap hoạt động.
4. **Không yêu cầu real-time performance:** Mỗi episode trên PenGym mất ~80-120 giây (so với milliseconds trên simulation).

---

## 2. Phân Tích Khác Biệt Giữa Hai Môi Trường

### 2.1 State Representation

Đây là **điểm khác biệt quan trọng nhất** và cũng là rào cản lớn nhất cho transfer:

| Thành phần          | Simulation (SCRIPT)                                   | PenGym (NASim-based)                                              |
| ------------------- | ----------------------------------------------------- | ----------------------------------------------------------------- |
| **Format**          | SBERT embedding vector (1538-dim)                     | NASim observation vector (variable-dim)                           |
| **Access state**    | 2-dim one-hot: `[1,0]`=compromised, `[0,1]`=reachable | Scalars: `compromised` (0/1), `reachable` (0/1), `access` (0/1/2) |
| **OS info**         | SBERT(384-dim) embedding của OS string                | One-hot vector theo `scenario.os`                                 |
| **Port info**       | SBERT(384-dim) embedding của joined port string       | Không trực tiếp (part of services)                                |
| **Services**        | SBERT(384-dim) embedding của joined service string    | Binary vector: `{ssh: 1.0, ftp: 0.0, ...}`                        |
| **Web fingerprint** | SBERT(384-dim) average embedding                      | N/A trong NASim standard                                          |
| **Processes**       | Không sử dụng                                         | Binary vector: `{tomcat: 1, cron: 0, ...}`                        |
| **Scope**           | Một host tại một thời điểm (single-target)            | Toàn bộ network (multi-host flat/matrix)                          |

### 2.2 Action Space

| Thuộc tính     | Simulation (SCRIPT)                            | PenGym                                          |
| -------------- | ---------------------------------------------- | ----------------------------------------------- |
| **Cấu trúc**   | `4 scans + N exploits` (N từ vulnerability DB) | `num_hosts × (scans + exploits + privesc)`      |
| **Scope**      | Hành động trên 1 target host                   | Hành động trên bất kỳ host nào trong network    |
| **Scan types** | PORT_SCAN, SERVICE_SCAN, OS_SCAN, WEB_SCAN     | ServiceScan, OSScan, SubnetScan, ProcessScan    |
| **Exploits**   | CVE-based (40+ loại từ MSF DB)                 | Scenario-specific (e_ssh, e_ftp, e_http, ...)   |
| **Constraint** | Implicit via reward penalty                    | `host_vector.perform_action()` permission check |

### 2.3 Reward Structure

| Hành động          | Simulation            | PenGym                                         |
| ------------------ | --------------------- | ---------------------------------------------- |
| Scan thành công    | +100 (OS/Service/Web) | `0 - action_cost` (cost = 1)                   |
| Exploit thành công | +1000                 | `host_value - action_cost` (value từ scenario) |
| Hành động sai      | -5 đến -10            | `0 - action_cost` hoặc `permission_error`      |
| Mục tiêu episode   | Compromise 1 target   | Compromise tất cả sensitive hosts (ROOT)       |

### 2.4 Tóm Tắt Gap Analysis

```
┌─────────────────────────────────────────────────────────────────┐
│                    SIM-TO-REAL GAP ANALYSIS                     │
├──────────────────┬──────────────────────────────────────────────┤
│ State Format     │ ████████████████████  CRITICAL               │
│                  │ SBERT 1538-dim ≠ NASim obs vector            │
│                  │ → Policy KHÔNG THỂ nhận input PenGym trực tiếp│
├──────────────────┼──────────────────────────────────────────────┤
│ Action Space     │ ██████████████       HIGH                    │
│                  │ Single-target ≠ Multi-host, exploit set khác │
│                  │ → Action mapping cần thiết kế cẩn thận       │
├──────────────────┼──────────────────────────────────────────────┤
│ Reward Scale     │ ████████             MEDIUM                  │
│                  │ Scale khác nhưng hướng tương tự              │
│                  │ → Cần rescale nếu fine-tune                  │
├──────────────────┼──────────────────────────────────────────────┤
│ Timing/Noise     │ ██████               LOW-MEDIUM              │
│                  │ Real execution adds latency + noise          │
│                  │ → Ảnh hưởng fine-tuning, không ảnh hưởng     │
│                  │   inference trực tiếp                        │
└──────────────────┴──────────────────────────────────────────────┘
```

---

## 3. Tiếp Cận Triển Khai

### 3.1 Tổng Quan Pipeline

```
Phase 1: State Adapter      Phase 2: Action Adapter      Phase 3: Evaluation
┌──────────────────┐       ┌──────────────────┐         ┌──────────────────┐
│ PenGym obs       │       │ SCRIPT action_id │         │ Run episodes     │
│      ↓           │       │      ↓           │         │ Collect metrics  │
│ Adapter Layer    │──────▶│ Action Mapper    │────────▶│ Compare baseline │
│      ↓           │       │      ↓           │         │ Gap analysis     │
│ SBERT-like 1538d │       │ PenGym action_id │         └──────────────────┘
└──────────────────┘       └──────────────────┘
                                                         Phase 4: Fine-tune
                                                         ┌──────────────────┐
                                                         │ Optional few-shot│
                                                         │ adaptation trên  │
                                                         │ PenGym env       │
                                                         └──────────────────┘
```

### 3.2 Phase 1: State Adapter Layer

**Mục tiêu:** Chuyển đổi PenGym observation → SCRIPT-compatible 1538-dim vector.

**Cách tiếp cận:** Xây dựng `PenGymStateAdapter` sử dụng cùng SBERT encoder mà SCRIPT đã dùng để encode state.

```python
# pentest/src/envs/adapters/state_adapter.py

import numpy as np
from src.agent.nlp.Encoder import encoder  # Cùng SBERT model

class PenGymStateAdapter:
    """
    Chuyển đổi PenGym NASim observation → SCRIPT StateEncoder format (1538-dim).

    PenGym obs (flat mode): [compromised, reachable, discovered, value,
                             discovery_value, access, *os_onehot, *services_binary,
                             *processes_binary] — per host

    SCRIPT state: [access(2) | os_sbert(384) | port_sbert(384) |
                   service_sbert(384) | web_fp_sbert(384)] = 1538 dim
    """

    STATE_DIM = 1538
    SBERT_DIM = 384  # encoder.SBERT_model_dim

    def __init__(self, scenario):
        self.scenario = scenario
        self.os_names = scenario.os        # ['linux', 'windows', ...]
        self.service_names = scenario.services  # ['ssh', 'ftp', 'http', ...]
        self.process_names = scenario.processes  # ['tomcat', 'cron', ...]

        # Pre-compute SBERT embeddings cho tốc độ
        self._cache = {}

    def _encode_cached(self, text: str) -> np.ndarray:
        if text not in self._cache:
            self._cache[text] = encoder.encode_SBERT(text).flatten()
        return self._cache[text]

    def convert(self, pengym_obs: np.ndarray, host_address: tuple,
                host_map: dict = None) -> np.ndarray:
        """
        Convert PenGym per-host observation to SCRIPT state vector.

        Lưu ý: Hàm này extract thông tin từ 1 host cụ thể trong obs vector,
        vì SCRIPT agent hoạt động ở chế độ single-target.
        """
        state = np.zeros(self.STATE_DIM, dtype=np.float32)

        # 1. Access vector (2-dim)
        # Map PenGym access levels to SCRIPT format
        compromised = self._extract_compromised(pengym_obs, host_address)
        reachable = self._extract_reachable(pengym_obs, host_address)
        if compromised:
            state[0] = 1.0  # [1, 0] = compromised
        elif reachable:
            state[1] = 1.0  # [0, 1] = reachable

        # 2. OS embedding (384-dim)
        os_info = self._extract_os_string(pengym_obs, host_address)
        if os_info:
            state[2:386] = self._encode_cached(os_info)

        # 3. Port embedding (384-dim)
        port_info = self._extract_port_string(pengym_obs, host_address, host_map)
        if port_info:
            state[386:770] = self._encode_cached(port_info)

        # 4. Service embedding (384-dim)
        service_info = self._extract_service_string(pengym_obs, host_address)
        if service_info:
            state[770:1154] = self._encode_cached(service_info)

        # 5. Web fingerprint embedding (384-dim)
        # PenGym không có web fingerprint → zeros (đây là accepted gap)

        return state

    def _extract_compromised(self, obs, host_addr):
        """Extract compromised flag from NASim observation cho specific host."""
        # Implementation depends on flat vs matrix obs format
        ...

    def _extract_os_string(self, obs, host_addr):
        """
        Reconstruct OS string từ NASim one-hot OS vector.
        e.g., os_vector = [1, 0] với os_names = ['linux', 'windows'] → 'linux'
        """
        ...

    def _extract_service_string(self, obs, host_addr):
        """
        Reconstruct service string từ NASim binary service vector.
        e.g., services = [1, 0, 1, 0, 0] → 'ssh,http'
        """
        ...

    def _extract_port_string(self, obs, host_addr, host_map=None):
        """
        Reconstruct port string từ host_map hoặc service-to-port mapping.
        PenGym cung cấp port info qua host_map['services'],
        hoặc suy ra từ CONFIG.yml service_port mapping.
        """
        ...
```

**Tham chiếu triển khai:**

- State encoder gốc: `pentest/src/agent/host.py` class `StateEncoder` (line 161+)
- SBERT encoder: `pentest/src/agent/nlp/Encoder.py`
- PenGym observation: `PenGym/pengym/envs/host_vector.py` → `perform_action()`

### 3.3 Phase 2: Action Mapper

**Mục tiêu:** Map action index từ SCRIPT policy output → PenGym action index.

```python
# pentest/src/envs/adapters/action_mapper.py

class ActionMapper:
    """
    Map SCRIPT action space → PenGym NASim action space.

    SCRIPT:  [PORT_SCAN, SERVICE_SCAN, OS_SCAN, WEB_SCAN, Exploit_0, ..., Exploit_N]
    PenGym:  flat_action_idx = host_idx * num_actions_per_host + action_type_idx

    Mapping cần xử lý:
    1. SCRIPT chọn action type → cần kết hợp với target host
    2. SCRIPT exploits (CVE-based) → PenGym exploits (service-based)
    3. PORT_SCAN → ServiceScan (closest equivalent)
    4. WEB_SCAN → không tồn tại trong PenGym → skip hoặc map sang ProcessScan
    """

    def __init__(self, script_actions, pengym_scenario):
        self.script_actions = script_actions  # List[Action_Class]
        self.pengym_scenario = pengym_scenario
        self._build_mapping()

    def _build_mapping(self):
        """
        Xây dựng bảng mapping giữa hai action spaces.

        Strategy: Map theo SEMANTIC SIMILARITY, không theo index.
        - PORT_SCAN → service_scan (cả hai đều discover services)
        - SERVICE_SCAN → service_scan
        - OS_SCAN → os_scan
        - WEB_SCAN → process_scan (closest in PenGym)
        - Exploit CVE-X → e_<service> nếu CVE-X exploit service đó
        """
        self.scan_mapping = {
            'Port Scan': 'service_scan',
            'Service Scan': 'service_scan',
            'OS Detect': 'os_scan',
            'Web Detect': 'process_scan',  # best-effort mapping
        }
        # Exploit mapping: CVE → target service → PenGym exploit
        self.exploit_mapping = {}
        # ...build from vulnerability database...

    def map_action(self, script_action_idx: int,
                   target_host_addr: tuple) -> int:
        """
        Convert (script_action, target_host) → PenGym flat action index.
        Returns -1 nếu không có mapping hợp lệ.
        """
        ...
```

### 3.4 Phase 3: Evaluation Pipeline

**Mục tiêu:** Chạy SCRIPT policy trên PenGym và thu thập metrics so sánh.

```python
# pentest/src/evaluation/sim_to_real_eval.py

class SimToRealEvaluator:
    """
    Evaluate pre-trained SCRIPT agent on PenGym environment.

    Metrics thu thập:
    1. Success rate: % episodes đạt goal
    2. Average reward per episode
    3. Average steps to goal (nếu thành công)
    4. Action distribution comparison (sim vs real)
    5. Failure mode analysis
    """

    def __init__(self, model_path, pengym_env, state_adapter, action_mapper):
        self.agent = self._load_agent(model_path)
        self.env = pengym_env
        self.adapter = state_adapter
        self.mapper = action_mapper

    def evaluate(self, num_episodes=50, max_steps=100):
        results = {
            'success_rate': 0,
            'avg_reward': 0,
            'avg_steps': 0,
            'action_distribution': {},
            'failure_modes': [],
            'per_episode': []
        }

        for ep in range(num_episodes):
            obs = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            episode_actions = []

            while not done and steps < max_steps:
                # 1. Convert PenGym obs → SCRIPT state
                state = self.adapter.convert(obs, current_target)

                # 2. Normalize (sử dụng running stats từ training)
                if self.agent.use_state_norm:
                    state = self.agent.state_norm(state, update=False)

                # 3. Get action from policy
                action_info = self.agent.Policy.select_action(state)
                script_action = action_info[0]

                # 4. Map to PenGym action
                pengym_action = self.mapper.map_action(
                    script_action, current_target)

                if pengym_action == -1:
                    # Unmappable action → log as failure mode
                    ...
                    continue

                # 5. Execute on PenGym
                obs, reward, done, truncated, info = self.env.step(pengym_action)
                total_reward += reward
                steps += 1
                episode_actions.append((script_action, pengym_action))

            results['per_episode'].append({
                'success': done,
                'reward': total_reward,
                'steps': steps,
                'actions': episode_actions
            })

        # Aggregate
        results['success_rate'] = sum(
            ep['success'] for ep in results['per_episode']) / num_episodes
        results['avg_reward'] = sum(
            ep['reward'] for ep in results['per_episode']) / num_episodes

        return results
```

### 3.5 Phase 4: Few-Shot Fine-Tuning (Optional)

Nếu zero-shot transfer cho kết quả thấp, thực hiện fine-tuning nhẹ:

```python
# Cấu hình fine-tuning trên PenGym
finetune_config = {
    'episodes': 50,         # Rất ít episodes (PenGym chậm)
    'actor_lr': 1e-5,       # LR thấp hơn 10x so với training gốc
    'critic_lr': 5e-6,      # Giữ critic conservative
    'freeze_layers': [0],   # Freeze layer đầu, chỉ tune layers sau
    'use_ewc': True,        # Giữ EWC để tránh catastrophic forgetting
    'ewc_lambda': 5000,     # Tăng lambda vì ít data
}
```

---

## 4. Metrics và Evaluation Framework

### 4.1 Metrics So Sánh Chính

| Metric                | Baseline (Sim)     | Target (PenGym) | Cách tính                             |
| --------------------- | ------------------ | --------------- | ------------------------------------- |
| **Success Rate**      | 1.0 (100%)         | ?               | `#success_episodes / #total_episodes` |
| **Average Return**    | 6640               | ?               | Mean total reward per episode         |
| **Steps to Goal**     | ~175 steps/episode | ?               | Mean steps in successful episodes     |
| **Convergence Speed** | Episode 39         | ?               | First episode đạt stable success      |
| **Inference Time**    | 0.62ms             | ?               | Time for action selection (excl. env) |

### 4.2 Gap Metrics (Mới)

```python
# Metrics đặc trưng cho sim-to-real gap assessment
gap_metrics = {
    # Transfer Efficiency
    'transfer_ratio': pengym_success_rate / sim_success_rate,

    # Action Validity Rate
    'valid_action_rate': valid_pengym_actions / total_actions_attempted,

    # Unmappable Action Rate
    'unmappable_rate': unmappable_actions / total_actions,

    # Behavioral Similarity (cosine sim of action distributions)
    'action_dist_similarity': cosine_sim(sim_action_dist, real_action_dist),

    # State Adapter Quality
    'state_reconstruction_error': mse(original_sbert_state, adapted_state),
}
```

### 4.3 Output Structure

```
pentest/outputs/
├── logs_baseline_sim/           # ← Baseline đã đổi tên
│   └── chain/
│       ├── baseline_cl_script_40tasks_seed42/
│       └── baseline_standard_6targets_seed42/
├── models_baseline_sim/         # ← Baseline đã đổi tên
│   └── chain/
│       └── chain-msfexp_vul-sample-6_envs-seed_0/
├── tensorboard_baseline_sim/    # ← Baseline đã đổi tên
│
├── logs/                        # ← Thư mục mới cho Strategy A
│   └── strategy_a/
│       ├── zero_shot_eval/
│       │   ├── pengym_tiny_results.json
│       │   ├── pengym_small_linear_results.json
│       │   └── gap_analysis_report.json
│       └── finetune_eval/       # Nếu Phase 4 được thực hiện
│           └── ...
├── models/                      # ← Models mới (nếu fine-tune)
│   └── strategy_a/
│       └── finetuned_on_pengym/
└── tensorboard/
    └── strategy_a/
```

---

## 5. Kế Hoạch Triển Khai Từng Bước

### Step 1: Chuẩn Bị Môi Trường

```bash
# 1.1 Đảm bảo PenGym hoạt động
cd PenGym
python run.py  # Verify PenGym demo works

# 1.2 Verify baseline model loads correctly
cd pentest
python -c "
import torch
model = torch.load('outputs_baseline_sim/models_baseline_sim/chain/chain-msfexp_vul-sample-6_envs-seed_0/PPO-actor.pt')
print('Actor loaded:', type(model))
print('Keys:', model.keys() if isinstance(model, dict) else 'state_dict')
"
```

### Step 2: Implement State Adapter

1. Tạo `pentest/src/envs/adapters/__init__.py`
2. Implement `PenGymStateAdapter` class
3. **Unit test:** So sánh adapter output với SCRIPT StateEncoder output cho cùng một host info
4. **Validation:** Verify output dimension = 1538, value ranges hợp lý

### Step 3: Implement Action Mapper

1. Phân tích PenGym scenario file để biết action space
2. Tạo explicit mapping table
3. **Unit test:** Verify mỗi SCRIPT action có ít nhất 1 PenGym mapping
4. Log unmappable actions

### Step 4: Zero-Shot Evaluation

1. Load pre-trained SCRIPT model
2. Chạy evaluation loop trên PenGym scenario (bắt đầu với `tiny.yml`)
3. Thu thập tất cả metrics
4. **Lưu kết quả** vào `outputs/logs/strategy_a/zero_shot_eval/`

### Step 5: Gap Analysis Report

1. So sánh metrics với baseline
2. Phân loại failure modes
3. Đánh giá feasibility của Strategy C dựa trên kết quả
4. **Document:** Viết report trong `docs/strategy_a_results.md`

### Step 6: Fine-Tuning (Conditional)

Chỉ thực hiện nếu zero-shot success_rate < 0.3:

1. Fine-tune với EWC protection
2. Limit episodes (PenGym chậm)
3. So sánh before/after fine-tuning

---

## 6. Rủi Ro Đã Xác Định

| Rủi ro                                                      | Mức độ     | Mitigation                                            |
| ----------------------------------------------------------- | ---------- | ----------------------------------------------------- |
| State adapter cho output distribution khác xa training      | Cao        | Validate bằng state_norm statistics từ training       |
| Action mapping coverage thấp (nhiều exploit không map được) | Cao        | Thiết kế fallback: random valid action khi unmappable |
| PenGym scenario quá khác sim scenario → evaluation vô nghĩa | Trung bình | Chọn scenario PenGym gần nhất với training scenario   |
| PenGym execution time quá lâu cho meaningful evaluation     | Thấp       | Giới hạn episodes, parallelism nếu có thể             |
| State normalization statistics mismatch                     | Trung bình | Option: reset normalizer và warm-up trên PenGym data  |

---

## 7. Tiêu Chí Thành Công

### Kết quả tối thiểu chấp nhận được:

- [ ] State adapter chạy đúng, output 1538-dim vector cho mọi PenGym observation
- [ ] Action mapper cover ≥70% action space của SCRIPT
- [ ] Zero-shot evaluation chạy hoàn chỉnh ≥ 20 episodes trên PenGym
- [ ] Gap analysis report hoàn chỉnh với ≥ 5 metrics so sánh

### Kết quả lý tưởng:

- [ ] Zero-shot success rate ≥ 0.2 (cho thấy knowledge transfer có ý nghĩa)
- [ ] Fine-tuned success rate ≥ 0.6 (cho thấy sim-trained policy adaptable)
- [ ] Behavioral similarity ≥ 0.5 (policy học được pattern đúng hướng)

### Quyết định tiếp theo:

| Kết quả Zero-Shot  | Diễn giải                                 | Action                                        |
| ------------------ | ----------------------------------------- | --------------------------------------------- |
| Success rate ≥ 0.3 | Transfer khả thi                          | Tiến thẳng Strategy C                         |
| 0.05 ≤ SR < 0.3    | Transfer có tiềm năng nhưng cần cải thiện | Strategy C với state standardization          |
| SR < 0.05          | Gap quá lớn, transfer gần như thất bại    | Phân tích sâu root cause trước khi Strategy C |

---

## 8. Tham Chiếu File Quan Trọng

| Mục đích            | File                                                                                                             |
| ------------------- | ---------------------------------------------------------------------------------------------------------------- |
| SCRIPT StateEncoder | `pentest/src/agent/host.py` — class `StateEncoder`                                                               |
| SBERT Encoder       | `pentest/src/agent/nlp/Encoder.py`                                                                               |
| PPO Policy          | `pentest/src/agent/policy/PPO.py` — class `PPO_agent`, `Actor`, `Critic`                                         |
| SCRIPT CL Agent     | `pentest/src/agent/continual/Script.py`                                                                          |
| Model checkpoint    | `pentest/outputs_baseline_sim/models_baseline_sim/chain/chain-msfexp_vul-sample-6_envs-seed_0/`                  |
| Baseline results    | `pentest/outputs_baseline_sim/logs_baseline_sim/chain/baseline_standard_6targets_seed42/experiment_summary.json` |
| PenGym Environment  | `PenGym/pengym/envs/environment.py` — class `PenGymEnv`                                                          |
| PenGym Host Vector  | `PenGym/pengym/envs/host_vector.py` — class `PenGymHostVector`                                                   |
| PenGym Config       | `PenGym/pengym/CONFIG.yml`                                                                                       |
| PenGym Scenarios    | `PenGym/database/scenarios/`                                                                                     |
| PenGym Utilities    | `PenGym/pengym/utilities.py` — Metasploit/nmap integration                                                       |
| Baseline experiment | `pentest/outputs_baseline_sim/logs_baseline_sim/chain/baseline_standard_6targets_seed42/experiment_summary.json` |
