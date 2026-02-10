# 🛡️ Fusion Pentest – RL-based Automated Penetration Testing

An RL (Reinforcement Learning) agent trained via PPO to autonomously plan and execute multi-step penetration testing attacks against network environments. Supports **Sim-to-Real** execution via NASim (simulation) and PenGym (real-world).

---

## 📁 Project Structure

```
pentest/
├── run.py                   # Main entry point (train / eval / demo)
├── setup_and_run.py         # One-click setup + run automation
├── requirements.txt         # Python dependencies
├── src/
│   ├── agent/               # RL agent (PPO, NLP encoder, actions, continual learning)
│   │   ├── agent.py         # Agent class (training & evaluation loops)
│   │   ├── host.py          # HOST environment wrapper per target
│   │   ├── actions/         # Action space definitions (CVE-based exploits)
│   │   ├── nlp/             # NLP Encoder (Sentence-BERT embeddings)
│   │   ├── policy/          # PPO policy, replay buffers, configs
│   │   ├── continual/       # Continual learning methods (Script, Finetune, EWC)
│   │   └── config.ini       # Agent configuration
│   └── envs/                # PenGym environment (Sim-to-Real)
│       ├── mode.py          # Execution mode switcher (sim/real/dual)
│       └── core/            # Host vectors, network logic
├── data/scenarios/          # Attack scenarios (JSON + YAML)
│   ├── chain/               # Multi-target chain scenarios (JSON)
│   ├── msfexp_vul/          # Single-CVE scenarios (JSON)
│   └── *.yml                # PenGym YAML scenarios
└── outputs/
    ├── logs/                # TensorBoard logs
    └── models/              # Saved model checkpoints
```

---

## 🚀 Quick Start

### 1. Setup (one-time)

```bash
cd pentest
python setup_and_run.py --setup-only
```

Hoặc manual:
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Train

```bash
# Train cơ bản (6 targets, 500 episodes)
python run.py --mode train --scenario chain/chain-msfexp_vul-sample-6_envs-seed_0.json --episodes 500

# Train nhanh để test (10 episodes)
python run.py --mode train --scenario chain/chain-msfexp_vul-sample-6_envs-seed_0.json --episodes 10

# Train với full 40 targets
python run.py --mode train --scenario chain/chain-msfexp_vul-sample-40_envs-seed_0.json --episodes 1000

# Train 1 CVE cụ thể
python run.py --mode train --scenario msfexp_vul/env-CVE-2021-44228.json --episodes 200
```

### 3. Evaluate

```bash
python run.py --mode eval \
    --scenario chain/chain-msfexp_vul-sample-6_envs-seed_0.json \
    --model-path outputs/models/chain/chain-msfexp_vul-sample-6_envs-seed_0
```

### 4. Demo

```bash
python run.py --mode demo --scenario chain/chain-msfexp_vul-sample-6_envs-seed_0.json
```

---

## ⚙️ Command-line Arguments

| Argument | Default | Mô tả |
|---|---|---|
| `--mode` | `train` | Chế độ: `train`, `eval`, `demo` |
| `--scenario` | `tiny.yml` | File scenario trong `data/scenarios/` |
| `--episodes` | `1000` | Số episodes training |
| `--max-steps` | `100` | Số bước tối đa mỗi episode |
| `--seed` | `42` | Random seed |
| `--device` | `cuda` | `cuda` hoặc `cpu` |
| `--env-type` | `simulation` | `simulation` (Script) hoặc `pengym` |
| `--execution-mode` | `sim` | `sim`, `real`, `dual` (cho PenGym) |
| `--model-path` | `None` | Đường dẫn model cho eval |
| `--log-dir` | auto | Thư mục TensorBoard logs |

---

## 📊 Xem Kết quả với TensorBoard

### Khởi động TensorBoard

```bash
# Xem logs của 1 lần train cụ thể
tensorboard --logdir outputs/logs --host localhost --port 6006

# Sau đó mở trình duyệt tại:
# http://localhost:6006
```

### Các metrics cần theo dõi

| Metric | Ý nghĩa | Mục tiêu |
|---|---|---|
| `Train_Episode_Rewards` | Tổng reward mỗi episode | **Tăng dần** → agent đang học |
| `Train_Episode_Steps` | Số bước mỗi episode | **Giảm dần** → agent tìm ra đường ngắn hơn |
| `Train_Success_Rate` | Tỷ lệ thành công (0.0 – 1.0) | **Tiến đến 1.0** |
| `Eval_Episode_Rewards` | Reward khi đánh giá | **Cao & ổn định** |
| `Eval_Success_Rate` | Tỷ lệ thành công khi eval | **≥ 0.9 là tốt** |
| `loss/actor_loss` | Actor (policy) loss | **Giảm & ổn định** |
| `loss/critic_loss` | Critic (value) loss | **Giảm & ổn định** |

### Cách đọc kết quả Training Output

```
'Training': 100%|█| 500/500 [05:23, rate_e='95.0%', rate_t='100.0%', re_e='600', re_t='600/600']
                                    ^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^  ^^^^^^^^^^  ^^^^^^^^^^^^^^
                                    │                │                │           │
                                    │                │                │           └─ Reward train / Best reward
                                    │                │                └─ Eval reward
                                    │                └─ Train success rate (100% = penetrated all targets)
                                    └─ Eval success rate (95% = 95% targets successfully penetrated)
```

**Lần train 10 episodes của bạn:**
- `rate_t='0.0%'` → Chưa penetrate được target nào ← **Bình thường**, cần nhiều episodes hơn
- `re_t='-5990/-5990'` → Reward âm ← Agent đang random actions
- 10 episodes quá ít để PPO hội tụ, thử **500–1000 episodes**

---

## 📈 Training Tips

### Episodes mà agent cần để học

| Scenario | Targets | Episodes khuyến nghị | Thời gian ước tính |
|---|---|---|---|
| `msfexp_vul/env-CVE-*.json` | 1 | 200–500 | 1–3 phút |
| `chain/...-6_envs-seed_0.json` | 6 | 500–1000 | 5–15 phút |
| `chain/...-40_envs-seed_0.json` | 40 | 1000–3000 | 30–60 phút |

### Dấu hiệu training thành công

1. ✅ **Success rate > 0** sau ~50–100 episodes (agent bắt đầu tìm ra exploit đúng)
2. ✅ **Success rate → 1.0** sau ~300–500 episodes
3. ✅ **Episode steps giảm** (agent tối ưu hóa attack path)
4. ✅ **Reward tăng & ổn định** (không dao động mạnh)

### Dấu hiệu có vấn đề

1. ❌ Success rate luôn 0.0 sau 500+ episodes → Kiểm tra scenario file
2. ❌ Reward không thay đổi → Learning rate quá nhỏ hoặc quá lớn
3. ❌ Reward dao động mạnh → Giảm `policy_clip`, tăng `batch_size`

---

## 🔧 Tuning Hyperparameters

Chỉnh hyperparameters qua code (file `src/agent/policy/config.py`):

```python
class PPO_Config:
    batch_size = 512          # Tăng nếu training không ổn định
    mini_batch_size = 64      # Thường = batch_size / 8
    gamma = 0.99              # Discount factor
    actor_lr = 1e-4           # Learning rate actor
    critic_lr = 5e-5          # Learning rate critic
    hidden_sizes = [512, 512] # Network architecture
    entropy_coef = 0.02       # Exploration coefficient
    ppo_update_time = 8       # PPO epochs per update
```

Hoặc qua `run.py` args:
```bash
python run.py --mode train --episodes 1000 --max-steps 50 --seed 0
```

---

## 📂 Available Scenarios

### JSON Scenarios (cho `--env-type simulation`)

| File | Targets | Mô tả |
|---|---|---|
| `chain/chain-msfexp_vul-sample-6_envs-seed_0.json` | 6 | Chain nhỏ, test nhanh |
| `chain/chain-msfexp_vul-sample-40_envs-seed_0.json` | 40 | Chain đầy đủ |
| `chain/all_scenario_msf-41-seed_4.json` | 41 | Full scenario |
| `msfexp_vul/env-CVE-*.json` | 1 each | 45 CVE riêng lẻ |

### YAML Scenarios (cho `--env-type pengym`)

| File | Mô tả |
|---|---|
| `tiny.yml` | Mạng nhỏ nhất (2-3 hosts) |
| `tiny-small.yml` / `tiny-hard.yml` | Biến thể tiny |
| `small-linear.yml` / `small-honeypot.yml` | Mạng nhỏ |
| `medium.yml` / `medium-single-site.yml` / `medium-multi-site.yml` | Mạng trung bình |

> ⚠️ YAML scenarios yêu cầu PenGym (`--env-type pengym`) và các tools: nmap, metasploit.

---

## 🔄 Sim-to-Real (PenGym)

```bash
# Simulation only (NASim)
python run.py --env-type pengym --execution-mode sim --scenario tiny.yml

# Real-world execution (requires nmap + metasploit)
python run.py --env-type pengym --execution-mode real --scenario tiny.yml

# Dual mode (both sim + real)
python run.py --env-type pengym --execution-mode dual --scenario tiny.yml
```

> ⚠️ Real/Dual mode yêu cầu: `nmap`, `metasploit-framework`, và mạng target đang chạy.

---

## 🧠 Continual Learning

Project hỗ trợ các phương pháp Continual Learning (trong `src/agent/continual/`):

- **Finetune** – Fine-tuning trực tiếp trên tasks mới
- **Script** – Knowledge consolidation + curriculum-guided imitation
- **Policy Distillation** – Transfer knowledge giữa các policies
- **EWC** – Elastic Weight Consolidation

> CL methods hiện chỉ sử dụng qua `Bot` class API trực tiếp.

---

## 📝 License

Research project – NCKH 2026.
