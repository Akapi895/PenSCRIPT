# 🛡️ PenSCRIPT — Sim-to-Real Dual Training for RL Pentesting

A Continual Reinforcement Learning agent (PPO + SCRIPT) that learns penetration testing in **simulation first**, then **transfers to realistic PenGym environments** via controlled domain transfer with EWC constraints.

---

## 📁 Project Structure

```
PenSCRIPT/
├── run.py                       # Main entry point — Dual training pipeline
├── run_benchmark.py             # Benchmark suite (baselines comparison)
├── test_integration.py          # Integration smoke test
├── requirements.txt
├── src/
│   ├── agent/                   # RL agent (PPO, SCRIPT CRL, NLP encoder)
│   │   ├── agent.py             # Base Agent (training & eval loops)
│   │   ├── agent_continual.py   # Continual Learning agent
│   │   ├── host.py              # HOST target wrapper + StateEncoder
│   │   ├── actions/             # Action space (CVE exploits, service-level)
│   │   ├── nlp/                 # NLP Encoder (Sentence-BERT)
│   │   ├── policy/              # PPO policy, configs
│   │   └── continual/           # CL methods (Script, Finetune, EWC)
│   ├── envs/                    # Environment layer
│   │   ├── core/                # UnifiedStateEncoder, host vectors, network
│   │   ├── adapters/            # PenGym ↔ SCRIPT adapters
│   │   └── wrappers/            # SingleHostWrapper, reward normalizer
│   ├── training/                # Training orchestration
│   │   ├── dual_trainer.py      # DualTrainer — Phase 0→4 orchestrator
│   │   ├── domain_transfer.py   # Sim→PenGym policy transfer
│   │   ├── pengym_trainer.py    # PenGym PPO training loop
│   │   └── pengym_script_trainer.py  # SCRIPT CRL over PenGym
│   ├── evaluation/              # Evaluation & metrics
│   │   ├── strategy_c_eval.py   # 4-agent comparative evaluation (Phase 4)
│   │   └── metric_store.py      # MetricStore, FZ transfer metrics
│   ├── pipeline/                # Scenario tools & curriculum
│   │   ├── scenario_compiler.py # Template → PenGym YAML compiler
│   │   ├── curriculum_controller.py
│   │   └── cve_classifier.py    # CVE difficulty grading
│   └── utils/
│       └── logging.py           # TeeLogger (console + file)
├── data/
│   ├── CVE/                     # CVE dataset + service registry
│   ├── config/                  # Training configs (YAML/JSON)
│   └── scenarios/               # Attack scenarios
│       ├── chain/               # Sim scenarios (JSON)
│       ├── msfexp_vul/          # Single-CVE scenarios (JSON)
│       ├── templates/           # Scenario templates
│       └── *.yml                # PenGym base scenarios (YAML)
└── outputs/                     # Training outputs (gitignored)
    ├── penscript/               # Pipeline results
    ├── models/                  # Saved checkpoints
    ├── logs/                    # Training logs
    └── tensorboard/             # TensorBoard events
```

---

## 🔬 Training Pipeline

```
Phase 0  →  Validation (SBERT, PenGym stability checks)
Phase 1  →  Train SCRIPT CRL on simulation (JSON) → θ_unified
Phase 2  →  Domain transfer: reset normalizer, discount Fisher, reduce LR
Phase 3  →  Fine-tune on PenGym (YAML) with EWC constraints → θ_dual
Phase 4  →  Evaluate 4 agents: θ_baseline, θ_unified, θ_dual, θ_scratch
```

---

## 🚀 Quick Start

### 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
pip install -r requirements.txt
```

### 2. Full Pipeline

```bash
# Train simulation → transfer → fine-tune on PenGym → evaluate
python run.py \
    --sim-scenarios data/scenarios/chain/chain-msfexp_vul-sample-6_envs-seed_0.json \
    --pengym-scenarios data/scenarios/tiny.yml data/scenarios/small-linear.yml \
    --train-scratch
```

### 3. With Transfer Strategy

```bash
python run.py \
    --sim-scenarios data/scenarios/chain/chain-msfexp_vul-sample-6_envs-seed_0.json \
    --pengym-scenarios data/scenarios/tiny.yml \
    --transfer-strategy cautious \
    --fisher-beta 0.3
```

### 4. Calibration Mode (scratch-only)

```bash
python run.py \
    --sim-scenarios data/scenarios/chain/chain-msfexp_vul-sample-6_envs-seed_0.json \
    --pengym-scenarios data/scenarios/tiny.yml \
    --scratch-only
```

### 5. Resume Interrupted Run

```bash
python run.py \
    --sim-scenarios data/scenarios/chain/chain-msfexp_vul-sample-6_envs-seed_0.json \
    --pengym-scenarios data/scenarios/tiny.yml \
    --resume-from outputs/penscript/previous_run
```

---

## ⚙️ Command-line Arguments

### Scenarios

| Argument | Default | Mô tả |
|---|---|---|
| `--sim-scenarios` | *(required)* | Simulation scenario JSON files (Phase 1) |
| `--pengym-scenarios` | *(required)* | PenGym scenario YAML files (Phase 3) |
| `--heldout-scenarios` | `None` | Heldout scenarios for generalization eval |

### Transfer

| Argument | Default | Mô tả |
|---|---|---|
| `--transfer-strategy` | `conservative` | `aggressive` / `conservative` / `cautious` |
| `--fisher-beta` | `0.3` | Fisher discount factor β |
| `--lr-factor` | `0.1` | Learning rate multiplier sau transfer |
| `--warmup-episodes` | `10` | Normalizer warmup episodes trên PenGym |

### Training

| Argument | Default | Mô tả |
|---|---|---|
| `--episodes` | `500` | Episodes per task |
| `--step-limit` | `100` | Max steps per episode |
| `--eval-freq` | `5` | Evaluate every N episodes |
| `--ewc-lambda` | `2000` | EWC regularisation strength |
| `--seed` | `42` | Random seed |
| `--episode-config` | `None` | JSON file for per-scenario episode config |
| `--training-mode` | `intra_topology` | `intra_topology` / `cross_topology` |

### Pipeline Control

| Argument | Mô tả |
|---|---|
| `--skip-phase0` | Skip Phase 0 validation |
| `--train-scratch` | Also train θ_pengym_scratch (full 4-agent comparison) |
| `--scratch-only` | Only scratch baseline (calibration mode) |
| `--no-canonicalization` | Disable canonicalization maps (ablation) |
| `--resume-from PATH` | Resume from interrupted run |
| `--output-dir` | Output directory (default: `outputs/penscript`) |

---

## 📊 Xem Kết quả

### TensorBoard

```bash
tensorboard --logdir outputs/penscript/tensorboard --host localhost --port 6006
```

### Phase 4 Output

Sau khi pipeline hoàn thành, kết quả so sánh 4 agents được lưu tại `outputs/penscript/penscript_results.json`:


| Agent | Mô tả |
|---|---|
| θ_sim_baseline | Trained trên sim, KHÔNG unified encoding |
| θ_sim_unified | Trained trên sim VỚI unified encoding |
| θ_dual | **PenSCRIPT** — transferred + fine-tuned trên PenGym |
| θ_pengym_scratch | Trained từ đầu trên PenGym (baseline) |

Key metrics: **SR** (Success Rate), **NR** (Normalized Reward), **η** (Step Efficiency), **FT/BT** (Forward/Backward Transfer).

---

## 📂 Available Scenarios

### Simulation (JSON) — Phase 1

| File | Targets | Mô tả |
|---|---|---|
| `chain/chain-msfexp_vul-sample-6_envs-seed_0.json` | 6 | Chain nhỏ, test nhanh |
| `chain/chain-msfexp_vul-sample-40_envs-seed_0.json` | 40 | Chain đầy đủ |
| `chain/all_scenario_msf-41-seed_4.json` | 41 | Full scenario |
| `msfexp_vul/env-CVE-*.json` | 1 each | 45 CVE riêng lẻ |

### PenGym (YAML) — Phase 3

| File | Hosts | Services | Mô tả |
|---|---|---|---|
| `tiny.yml` | 3 | 1 (SSH) | Nhỏ nhất, smoke test |
| `tiny-small.yml` / `tiny-hard.yml` | 3–4 | 1 | Biến thể tiny |
| `small-linear.yml` | 8 | 3 | Linear topology |
| `small-honeypot.yml` | 8 | 3 | Có honeypot |
| `medium.yml` | 16 | 5 | Multi-vuln, multi-service |
| `medium-single-site.yml` | 16 | 5 | Single subnet |
| `medium-multi-site.yml` | 16 | 5 | Multi-subnet |

---

## 🧠 Continual Learning Methods

| Method | Mô tả |
|---|---|
| **SCRIPT** | Knowledge consolidation + curriculum-guided imitation + EWC |
| **Finetune** | Fine-tuning trực tiếp trên tasks mới |
| **EWC** | Elastic Weight Consolidation (chống catastrophic forgetting) |

---

## 📝 License

Research project — NCKH 2026.
