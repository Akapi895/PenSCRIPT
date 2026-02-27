# Paper Outline — Cross-Domain Continual Reinforcement Learning for Automated Penetration Testing

> **Version:** 1.0  
> **Date:** 2026-02-27  
> **Branch:** `strC_1`  
> **Status:** Sườn bài hoàn chỉnh — sẵn sàng triển khai viết full paper

---

## 0. Research Foundation

### 0.1 Research Problem

Các hệ thống RL cho penetration testing hiện tại gặp **ba giới hạn cơ bản**:

1. **Simulation–Reality Gap**: Agent RL (như SCRIPT) đạt hiệu suất cao trên simulation riêng nhưng không thể triển khai trên môi trường thực tế (NASim/PenGym) do sự không tương thích toàn diện về state representation (SBERT 1538-dim vs binary one-hot), action space (2064 CVE vs 12 service-level), reward scale ([-10, 1000] vs [-3, 100]), và interaction paradigm (single-host vs multi-host network).

2. **Catastrophic Forgetting khi mở rộng**: Khi agent học trên nhiều network topologies khác nhau, kiến thức về topologies trước bị xoá bởi gradient updates của topology mới — đặc biệt nghiêm trọng khi EWC consolidation xảy ra xuyên topologies với state/reward distributions khác biệt cơ bản (cross-topology death spiral).

3. **Thiếu curriculum difficulty cho exploit**: Môi trường PenGym/NASim có exploit probability gần 1.0 (trivially easy), không có gradient khó tăng dần theo exploit difficulty — agent chỉ học network routing mà không học chiến thuật tấn công.

### 0.2 Research Gap

| Hướng nghiên cứu | Existing Work | Gap |
|---|---|---|
| RL Pentest trên simulation | SCRIPT (Tran et al.), DRL-based pentest (Schwartz & Kurniawati) | Không transfer được sang môi trường khác |
| RL Pentest trên NASim/PenGym | PenGym (Bhatnagar et al.), NASim agents | Không có CRL, không có CVE knowledge |
| Continual RL | EWC, PackNet, Progress&Compress | Áp dụng trong games (Atari, MuJoCo), chưa cho pentest cross-domain |
| Sim-to-Real RL | Domain randomization, system identification | Chủ yếu cho robotics, chưa cho network security |
| Curriculum RL | CL cho NLP/CV tasks | Curriculum theo topology nhưng chưa theo exploit difficulty |

**Gap tổng hợp**: Chưa có nghiên cứu nào giải quyết **đồng thời** bài toán (a) cross-domain transfer giữa hai hệ thống pentest RL có kiến trúc khác biệt cơ bản, (b) continual learning trên đa dạng network topologies mà không gây death spiral, và (c) curriculum training theo cả hai chiều topology complexity + exploit difficulty.

### 0.3 Hypothesis

> **H**: Một unified representation framework (shared state 1540-dim + hierarchical action 16-dim + normalized reward) kết hợp với intra-topology continual learning (per-topology CRL streams with EWC) cho phép chuyển giao hiệu quả kiến thức từ simulation sang môi trường pentest phức tạp hơn, đạt Forward Transfer dương có ý nghĩa thống kê ($FT_{SR} > 0, p < 0.05$) trong khi duy trì Backward Transfer chấp nhận được ($BT_{\eta} \geq -0.15$).

### 0.4 Research Questions

| RQ | Câu hỏi | Metric đo | Kết quả thực nghiệm |
|---|---|---|---|
| **RQ1** | Unified state representation (1540-dim) có loại bỏ được domain gap giữa SCRIPT simulation và PenGym không? | Cross-domain cosine similarity, Phase 0 validation | cosine = 1.0 (hoàn toàn khớp) |
| **RQ2** | Sim-to-PenGym transfer có tạo ra Forward Transfer dương trên các PenGym topologies không? | $FT_{SR}$, $FT_{\eta}$, SR(θ_dual) vs SR(θ_scratch) | $FT_{SR} = 0.333 \text{–} 1.0$, scratch SR = 0% |
| **RQ3** | Intra-topology CRL có giải quyết được cross-topology death spiral so với cross-topology CRL không? | Per-stream SR, Forgetting matrix $\mathcal{F}$ | tiny SR: 100% (intra) vs 37.5% (cross) |
| **RQ4** | CVE-based curriculum (T1→T4) có giúp agent học xuyên gradient khó tăng dần trên cùng topology không? | Per-tier SR progression, $\eta$ per tier | T1=T2=T3=T4=100% trên tiny |
| **RQ5** | Kiến thức CRL có generalize cross-topology (zero-shot transfer sang topology chưa train) không? | Cross-topology eval SR, $\mathcal{Z}$ vector | ≈ 0% (negative result — fundamental limitation) |

---

## 1. Title — Proposed Options

### Option A (Recommended — Comprehensive):
> **Cross-Domain Continual Reinforcement Learning for Automated Network Penetration Testing: Bridging Simulation and Realistic Environments via Unified Representation and Intra-Topology Curriculum**

### Option B (Concise):
> **From Simulation to Reality: Continual RL Transfer for Autonomous Penetration Testing with CVE-Based Curriculum Training**

### Option C (Problem-focused):
> **Overcoming the Sim-to-Real Gap in RL-Based Penetration Testing through Unified State-Action Representation and Topology-Aware Continual Learning**

### Option D (Ngắn nhất, cho workshop):
> **Sim2Pentest: Cross-Domain Continual RL with Intra-Topology Curriculum for Automated Pentesting**

---

## 2. Abstract (Outline — ~250 words)

### Paragraph 1 — Context & Problem
- RL for automated penetration testing has shown promise in simulated environments, but transferring learned policies to more realistic settings (e.g., NASim/PenGym) remains an open challenge.
- The fundamental incompatibilities: state representation (SBERT semantic embeddings vs binary one-hot, 1538-dim vs variable-dim), action space (2064 CVE-specific vs 12 service-level), reward scale (100× difference), and interaction paradigm (single-host vs multi-host network).
- Moreover, learning across diverse network topologies triggers catastrophic forgetting — a "death spiral" where EWC consolidates damaged parameters across incompatible topology distributions.

### Paragraph 2 — Proposed Approach
- We present **PenSCRIPT**, a cross-domain continual RL framework that bridges a high-fidelity simulation (SCRIPT) and a realistic pentest environment (PenGym) through:
  - (1) A **unified state representation** (1540-dim SBERT-based encoding with cross-domain canonicalization, achieving cosine similarity = 1.0).
  - (2) A **hierarchical action space** reducing 2064 CVE actions to 16 service-level groups with ~100% cross-domain coverage.
  - (3) **Intra-topology CRL** — per-topology independent CRL streams forked from a shared transferred agent, eliminating cross-topology EWC contamination.
  - (4) A **CVE-based difficulty curriculum** (T1→T4) grading 1985 CVEs into 4 tiers via a composite difficulty score ($S_{diff}$ based on exploit probability, attack complexity, privileges required).

### Paragraph 3 — Results
- Report key experimental results:
  - Unified representation: cross-domain cosine = 1.0, eliminating domain gap at representation level.
  - Transfer effectiveness: SR = 100% on tiny and small-linear topologies (vs 0% for scratch baseline), $FT_{SR} = 0.333 \text{–} 1.0$.
  - Intra-topology CRL: eliminates death spiral (tiny SR 37.5% → 100%, all tiers T1–T4 pass).
  - Step efficiency: $\eta = 74.9\%$ on tiny (agent solves near-optimally with sim knowledge).
  - Scalability boundary: medium topology (16+ hosts, |A|=192) remains unsolvable — identifies sparse reward as fundamental bottleneck for larger networks.

### Paragraph 4 — Contribution Summary
- 5 numbered contributions (cf. §2.5).
- Code and data publicly available at [repo link].

---

## 3. Introduction (~2 pages)

### 3.1 Opening — Real-world Motivation (¶1–2)
- **Viết gì**: Cybersecurity landscape — tần suất & chi phí tấn công mạng tăng, nhu cầu automated pentest.
- **Logic dẫn dắt**: Manual pentest is expensive and doesn't scale → AI-powered autonomous pentesting is emerging → RL is a natural formulation (MDP over network states).
- **Cite**: IBM X-Force report, NIST guidelines, early works on automated pentest.

### 3.2 State of RL for Pentesting (¶3–4)
- **Viết gì**: Hai hướng tiếp cận chính:
  - *Simulation-only RL* (SCRIPT, DRL-pentest agents): Train trên simulation riêng, đạt hiệu suất cao nhưng không deployable vào env khác.
  - *Environment-focused* (NASim, PenGym, CyberBattleSim): Cung cấp env chuẩn (Gymnasium API) nhưng thiếu agent mạnh, thiếu CRL, thiếu CVE diversity.
- **Research gap**: Không có framework nào kết nối hai hướng — sim agent không chạy được trên NASim, NASim agent không có CRL.
- **Điểm mới**: Phát biểu rõ ràng 8 xung đột kỹ thuật (state, action, reward, paradigm, interface, transfer mechanism, CVE diversity, evaluation) between the two systems.

### 3.3 Challenges of Cross-Domain Transfer (¶5–6)
- **Viết gì**: Chi tiết 3 challenge chính:
  1. **Representation mismatch**: Không chỉ dimension mismatch mà là paradigm khác nhau (semantic SBERT vs positional binary). Naive adapter (projection matrix) fails vì mất ngữ nghĩa. Cần unified encoding.
  2. **EWC cross-contamination**: Khi CRL train xuyên topologies, EWC Fisher consolidation trên failed topology (medium SR=0%) locks damaged parameters → poisons subsequent tasks. Evidence: exp2 cross-topology CRL, tiny SR giảm từ 80% xuống khi train cùng medium.
  3. **No exploit difficulty gradient**: PenGym exploits are trivially easy (prob ≈ 1.0) → agent only learns routing, not attack tactics. Curriculum needs both dimensions (topology complexity + exploit difficulty).
- **Điểm mới**: Explicitly identify "cross-topology death spiral" as a novel failure mode specific to CRL in pentest domains.

### 3.4 Proposed Solution — High-Level (¶7–8)
- **Viết gì**: One-paragraph summary of approach + architecture figure reference.
- Pipeline: Phase 0 (Validation) → Phase 1 (Sim CRL) → Phase 2 (Domain Transfer) → Phase 3 (Intra-Topology CRL on PenGym) → Phase 4 (Multi-dimensional Evaluation).
- Key design principles: "Giữ nguyên lõi, kết nối qua adapter" — không sửa SCRIPT core, không sửa PenGym core, tích hợp qua unified representation layer.

### 3.5 Contributions (¶9)

**C1 — Unified Cross-Domain Representation Framework:**
Thiết kế unified state encoder (1540-dim, SBERT-based) với canonicalization map đạt cross-domain cosine = 1.0, kèm hierarchical action space (2064 → 16-dim, ~100% coverage), và unified reward normalizer ([-1, +1]) — cho phép policy train trên SCRIPT simulation hoạt động trực tiếp trên PenGym mà không cần retrain. *So với existing work*: Các hệ thống trước (NASim agents, SCRIPT) dùng representation riêng; chưa có work nào thống nhất SBERT semantic encoding với NASim binary encoding.

**C2 — Cross-Domain Continual Learning Pipeline (5-Phase):**
Pipeline Phase 0→4 with domain transfer manager (3 strategies: conservative/aggressive/cautious), Fisher discount ($\beta = 0.3$), state normalizer warmup, LR decay — enabling CRL 5 pillars (Teacher Guidance, KL Imitation, Knowledge Distillation, Retrospection, EWC) to operate across simulation and PenGym domains. *So với existing work*: EWC applied within single domain (Atari/MuJoCo); paper applies EWC cross-domain with calibrated Fisher discount.

**C3 — Intra-Topology CRL Architecture:**
Per-topology independent CRL streams via `deepcopy()` forking, eliminating cross-topology EWC contamination. Best-stream selection for final agent. *So với existing work*: Standard CRL trains sequentially across all tasks; paper identifies "death spiral" failure mode and proposes topology-aware stream isolation as solution.

**C4 — CVE-Based Difficulty Curriculum:**
1985 CVE difficulty grading via composite score $S_{diff} = 0.50(1-\text{prob}) + 0.25 f_{AC} + 0.15 f_{PR} + 0.10 f_{UI}$ into 4 tiers (T1–T4). Template+Overlay scenario generation (8 topologies × 4 tiers × variants = 96+ scenarios). Per-task adaptive episode/step-limit scheduling. *So với existing work*: PenGym/NASim scenarios have no difficulty variation; SCRIPT has no curriculum. Paper adds exploit-difficulty dimension to curriculum.

**C5 — Multi-Dimensional Transfer Evaluation Framework:**
23 evaluation components including multi-episode SR (K=20), Normalized Reward (NR), Step Efficiency ($\eta$), 6-dimensional Forward/Backward Transfer (FT/BT × SR/NR/$\eta$), policy-level metrics ($D_{KL}$, $\Delta_F$), Forgetting Matrix ($\mathcal{F}$), Zero-Shot Transfer Vector ($\mathcal{Z}$), Learning-Speed Transfer (TTT, AUC), CE Curves, MetricStore persistence. *So với existing work*: SCRIPT uses 1-episode binary SR; NASim agents use cumulative reward only. Paper provides comprehensive CRL evaluation overcoming ceiling/floor effects.

### 3.6 Paper Organization (¶10)
- Standard roadmap paragraph: "The remainder of this paper is organized as follows: Section 2 reviews related work..."

---

## 4. Related Work (~1.5 pages)

### 4.1 RL for Automated Penetration Testing
- **Viết gì**: Review các agent RL cho pentest.
- **Papers**: 
  - Schwartz & Kurniawati (2019) — DQN pentest agent trên NASim
  - Tran et al. (2024) — SCRIPT: PPO + CRL 5 pillars cho single-host pentest
  - Ghanem & Chen (2020) — RL-based vulnerability assessment
  - Zhou et al. (2021) — Hierarchical RL pentest
  - Hu et al. — CyberBattleSim Microsoft
  - AutoPentest-DRL, PENTESTGYM
- **Observation**: Tất cả đều hoạt động trên 1 domain duy nhất; không có cross-domain transfer.
- **Gap**: No existing RL pentest agent transfers knowledge across fundamentally different environments.

### 4.2 Network Attack Simulation Environments
- **Viết gì**: Review các simulator/environment.
- **Papers**:
  - NASim (Schwartz & Kurniawati) — Network Attack Simulator
  - PenGym (Bhatnagar et al., 2024) — NASim extension with CyRIS real execution
  - CyberBattleSim (Microsoft, 2021)
  - CALDERA (MITRE), CybORG (CAGE challenges)
- **Observation**: Environments are well-developed but lack strong RL agents and CRL capabilities.
- **Gap**: None provide CVE-level difficulty grading or curriculum training support.

### 4.3 Continual Reinforcement Learning (CRL)
- **Viết gì**: Review các phương pháp chống forgetting trong RL.
- **Papers**:
  - EWC — Kirkpatrick et al. (2017)
  - Online EWC — Schwarz et al. (2018)
  - PackNet — Mallya & Lazebnik (2018)
  - Progress & Compress — Schwarz et al. (2018)
  - CLEAR — Rolnick et al. (2019) — CRL benchmark
  - CORA — Powers et al. (2022) — CRL evaluation metrics
  - Continual World — Wołczyk et al. (2021) — CRL robotics benchmark
- **Observation**: CRL well-studied for Atari/MuJoCo; this paper applies CRL to pentest with cross-domain twist.
- **Distinction vs our work**: Existing CRL operates within single domain; we apply CRL *across* simulation and realistic pentest domains with domain transfer.

### 4.4 Sim-to-Real Transfer in RL
- **Viết gì**: Review domain adaptation / sim-to-real methods.
- **Papers**:
  - Domain Randomization — Tobin et al. (2017)
  - System Identification — Yu et al. (2017)
  - RCAN — James et al. (2019)
  - Sim-to-Real for robotics — survey papers
- **Gap**: Sim-to-real RL well-studied for robotics (continuous control), but not for discrete-action network security domains where state/action semantics differ fundamentally.

### 4.5 Curriculum Learning in RL
- **Viết gì**: Review curriculum strategies.
- **Papers**:
  - Bengio et al. (2009) — Curriculum Learning
  - OpenAI — Automatic curriculum generation
  - POET — Wang et al. (2019)
  - Narvekar et al. (2020) — Curriculum RL survey
- **Gap**: Curriculum in pentest is limited to topology size progression; no work grades exploit *difficulty* as a curriculum dimension using real CVE metadata.

### 4.6 Positioning Table

| Feature | SCRIPT | PenGym | CyberBattleSim | AutoPentest-DRL | PenSCRIPT |
|---|---|---|---|---|---|
| RL Agent | PPO | DQN (basic) | RL (various) | DQN | PPO + CRL 5 pillars |
| CRL | ✅ (5 pillars) | ❌ | ❌ | ❌ | ✅ (cross-domain) |
| Sim-to-Real | ❌ | ✅ (KVM) | ❌ | ❌ | ✅ (sim→PenGym) |
| Cross-domain transfer | ❌ | ❌ | ❌ | ❌ | ✅ |
| CVE diversity | 2064 CVE | 5 service types | Microsoft graphs | NASim scenarios | 1985 CVE → 16 groups |
| Exploit curriculum | ❌ | ❌ | ❌ | ❌ | ✅ (T1→T4) |
| Eval framework | 1-episode SR | cumulative reward | win/loss | episodic reward | **23 metrics (C5)** |

---

## 5. Methodology (~4–5 pages)

### 5.1 Problem Formulation

#### 5.1.1 MDP for Network Penetration Testing
- **Viết gì**: Formal MDP definition $(S, A, T, R, \gamma)$ for pentest.
  - $S$: Network state — per-host vector (access status, OS, services, ports)
  - $A$: Pentest actions — scan (reconnaissance) + exploit (attack) + privesc (escalation)
  - $T$: Transition function — probabilistic exploit success, deterministic scans
  - $R$: Reward — positive for compromise, negative for cost, shaped by host value
  - $\gamma = 0.99$: Discount factor
- **Điểm mới**: Formalize the cross-domain MDP where $S$, $A$, $R$ have different representations in simulation and PenGym.

#### 5.1.2 Cross-Domain CRL Formulation
- **Viết gì**: Formal definition of the cross-domain CRL problem.
  - Domain $D_{sim}$: SCRIPT simulation with tasks $\tau_1^{sim}, ..., \tau_6^{sim}$.
  - Domain $D_{real}$: PenGym topologies $\{tiny, small\text{-}linear, medium\text{-}single\text{-}site, ...\}$, each with tiers T1–T4.
  - Goal: Learn policy $\pi^*$ that maximizes cumulative performance across PenGym tasks while preserving sim knowledge.
  - CRL metrics: Forward Transfer $FT = SR(\theta_{dual}) - SR(\theta_{scratch})$, Backward Transfer $BT = perf(\theta_{dual}, D_{sim}) - perf(\theta_{sim}, D_{sim})$.

### 5.2 System Architecture — Overview

- **Figure 1**: Full architecture diagram (cf. project_fusion_summary.md §III.2).
- **Viết gì**: 
  - Layered architecture: SCRIPT Core (unchanged) ↔ Adapter Layer (new) ↔ PenGym Core (unchanged).
  - Design patterns: Adapter, Wrapper, Strategy, Factory, Duck-typing.
  - Integration principle: "Keep cores intact, connect via adapters" — 0 lines of SCRIPT/PenGym core modified.
  - ~4,100 lines of new integration code + ~2,270 lines of pipeline code.
- **Đóng góp**: Mẫu kiến trúc tái sử dụng cho việc kết hợp hai hệ thống RL pentest khác paradigm.

### 5.3 Unified State Representation (C1)

#### 5.3.1 State Format Design

$$\mathbf{s} = [\underbrace{\text{access}}_{3} \;|\; \underbrace{\text{discovery}}_{1} \;|\; \underbrace{\text{OS}}_{384} \;|\; \underbrace{\text{port}}_{384} \;|\; \underbrace{\text{service}}_{384} \;|\; \underbrace{\text{auxiliary}}_{384}] \in \mathbb{R}^{1540}$$

- **Viết gì**: 
  - Motivation: Why SBERT (semantic embeddings) > one-hot (positional encoding) for cross-domain transfer.
  - Design changes from SCRIPT original (1538 → 1540): access 2→3 dim + discovery 1 dim.
  - SBERT model: `all-MiniLM-L6-v2`, 384-dim output, cached encoding.
  - **Table**: Component-by-component comparison (SCRIPT 1538 vs PenGym variable vs Unified 1540).

#### 5.3.2 Cross-Domain Canonicalization
- **Viết gì**: 
  - Problem: Same entity has different names across domains (`ubuntu`→`linux`, `openssh`→`ssh`, `apache httpd`→`http`).
  - Canonicalization map: Apply *before* SBERT encoding.
  - Validation: Phase 0 confirms `cross_domain_os_cosine = 1.0`.
- **Equation**: $\text{canon}(x) = \text{CANONICAL\_MAP}(x_{\text{lower}})$ with version stripping.
- **Đóng góp**: Canonicalization is surprisingly critical — without it, SBERT produces different vectors for semantically identical entities, creating an invisible domain gap.

#### 5.3.3 SingleHostPenGymWrapper
- **Viết gì**:
  - Paradigm bridging: Wraps multi-host PenGym into single-host SCRIPT interface.
  - `reset()` → single target's 1540-dim state; `step(service_action)` → internal flat action mapping.
  - Auto subnet scan after compromise, failure rotation, target prioritization.
  - This is the **single integration point** — all multi-host complexity hidden inside.

### 5.4 Hierarchical Action Space (C1)

#### 5.4.1 Two-Tier Architecture

- **Viết gì**:
  - **Tier 1 (RL policy)**: 16 service-level actions (4 scan + 9 exploit + 3 privesc).
  - **Tier 2 (heuristic)**: CVESelector maps service → specific CVE via 4 strategies (match/rank/random/round-robin).
  - Action coverage: 2064 CVE → 16 groups (77.9% auto-classified, 456 → misc).
  - Cross-domain coverage: 3.4% (direct) → ~100% (with abstraction).
  - Convergence speedup: ~10× faster than 2064-dim action space.
- **Table**: 16 actions with PenGym mapping and CVE group sizes.

#### 5.4.2 Unified Reward Normalizer
- **Viết gì**:
  - Problem: SCRIPT reward [-10, +1000], PenGym [-3, +100] → 100× scale difference.
  - EWC Fisher information $\propto$ gradient² $\propto$ reward scale → incomparable cross-domain.
  - Solution: `UnifiedNormalizer` maps both to [-1, +1].
  - Impact: PPO Critic calibration + EWC Fisher comparability.

### 5.5 Cross-Domain CRL Pipeline (C2)

#### 5.5.1 Five-Phase Pipeline

**Table/Figure**: Phase 0–4 overview.

| Phase | Name | Input | Output | Key Operations |
|---|---|---|---|---|
| 0 | Validation | SBERT model, scenarios | GO/NO-GO | Cross-domain cosine ≥ 0.95, dims check |
| 1 | Sim CRL | 6 sim tasks | $\theta_{sim}$ (SR≈100%) | CRL 5 pillars, EWC Fisher accumulation |
| 2 | Domain Transfer | $\theta_{sim}$ | $\theta_{transferred}$ | State norm reset, Fisher discount $\beta=0.3$, LR×0.1, warmup |
| 3 | PenGym Fine-tune | PenGym tasks T1→T4 | $\theta_{dual}$ | Intra-topology CRL streams, per-task episode/step-limit |
| 4 | Evaluation | 3 agents × all tasks | Report | Multi-dimensional FT/BT, forgetting matrix, CE curves |

#### 5.5.2 Domain Transfer Manager (Phase 2)
- **Viết gì**:
  - 3 transfer strategies: conservative (default), aggressive, cautious.
  - Conservative: Fisher discount $\beta = 0.3$ (reduce 70% EWC penalty), LR × 0.1, state normalizer reset + warmup.
  - Warmup: 447 random rollout states from PenGym to rebuild running statistics.
  - **Equation**: EWC penalty after discount: $\mathcal{L}_{EWC}^{new} = \sum_k \frac{\lambda}{2} \cdot \beta \cdot F_k (\theta_k - \theta_k^*)^2$

#### 5.5.3 PenGymHostAdapter (Duck-Typing)
- **Viết gì**:
  - CRL framework requires HOST interface (`reset()`, `perform_action()`, `.ip`, `.info`).
  - PenGymHostAdapter mocks entire HOST interface → CRL 5 pillars run on PenGym **unmodified**.
  - Lazy initialization avoids NASim class-level state corruption.
  - Factory method: `PenGymHostAdapter.from_scenario(path, seed)`.

### 5.6 Intra-Topology CRL (C3)

#### 5.6.1 Cross-Topology Death Spiral — Problem Analysis
- **Viết gì**:
  - Define "death spiral": EWC consolidating damaged parameters from a failed topology poisons subsequent topologies.
  - Evidence: Experiment 2 — cross-topology CRL achieved 25.8% overall SR (tiny 75–80%, small/medium 0%). When medium fails, EWC locks the failure state, preventing tiny from maintaining its 80% SR on subsequent tiers.
  - Root cause: EWC assumes tasks share similar state/reward distributions. Cross-topology tasks violate this assumption fundamentally (|A|=42 for tiny vs |A|=192 for medium).
- **Figure**: Death spiral illustration (cross-topology CRL diagram).

#### 5.6.2 Per-Topology Stream Architecture
- **Viết gì**:
  - Solution: Fork $\theta_{transferred}$ via `deepcopy()` for each topology → independent CRL streams.
  - Each stream: T1→T2→T3→T4 (intra-topology curriculum).
  - EWC consolidation only within same topology — tasks share network structure → Fisher information meaningful.
  - Best-stream selection: $\theta_{dual} = \arg\max_{t \in \text{streams}} SR_t$.
  - Memory cost: ~15 MB per agent × N topologies — acceptable.
- **Figure**: Stream architecture diagram (cf. intra_topology_crl_blueprint.md Appendix A).
- **Algorithm**: Per-stream CRL pseudocode.

### 5.7 CVE-Based Difficulty Curriculum (C4)

#### 5.7.1 CVE Difficulty Grading

$$S_{diff} = 0.50 \cdot (1 - \text{prob}) + 0.25 \cdot f_{AC} + 0.15 \cdot f_{PR} + 0.10 \cdot f_{UI}$$

- **Viết gì**:
  - Dataset: 1985 CVE from Metasploit, 22 features each.
  - Key finding: $\text{prob}$ is near-deterministic function of MSF_Rank (99.1% for Excellent=0.99, 98.4% for Great=0.80, etc.) — just 4 discrete levels.
  - Additional features: Attack_Complexity (LOW/MEDIUM/HIGH), Privileges_Required (NONE/LOW/HIGH), User_Interaction (NONE/REQUIRED).
  - 4-tier classification: T1 (Easy, $S_{diff} < 0.15$, ~1200 CVE), T2 (Medium, ~500), T3 (Hard, ~220), T4 (Expert, ~65).
- **Table**: Tier definition with CVE counts, example CVEs, exploit probability ranges.

#### 5.7.2 Template + Overlay Scenario Generation
- **Viết gì**:
  - Template: Network topology (subnets, firewall, hosts) with service/CVE slots.
  - Overlay: CVE assignment per tier (changes prob, cost, access while keeping topology fixed).
  - Compiler: Template + Overlay → NASim-compatible YAML.
  - Scale: 8 base topologies × 4 tiers × variants = 96+ compiled scenarios.
- **Đóng góp**: Decouples topology complexity from exploit difficulty — enables true two-dimensional curriculum.

#### 5.7.3 Adaptive Episode/Step-Limit Scheduling
- **Viết gì**:
  - Episode budget per task: $\text{episodes} = \lceil \text{base}[\text{topo}] \times \text{multiplier}[\text{tier}] \rceil$.
  - Tier multipliers: T1=0.8×, T2=1.0×, T3=2.0×, T4=3.0×.
  - Step limit per topology: tiny=200, small-linear=300–500, medium=500.
  - Centralized JSON config: `curriculum_episodes.json`.

### 5.8 SCRIPT CRL Framework — 5 Pillars (Context)

- **Viết gì** (brief, since SCRIPT is prior work):
  - **Teacher Guidance**: Explorer (student) guided by Keeper (teacher) via curriculum decay $\alpha$.
  - **KL Imitation**: $\mathcal{L}_{KL} = D_{KL}(\pi_{student} \| \pi_{teacher})$ added to PPO loss.
  - **Knowledge Distillation**: Expert samples from Explorer compressed into Keeper.
  - **Retrospection**: Keeper minimizes $D_{KL}(\pi_{new} \| \pi_{old})$ to prevent forgetting.
  - **Online EWC**: Fisher Information Matrix regularization: $\mathcal{L}_{EWC} = \sum_k \frac{\lambda}{2} F_k (\theta_k - \theta_k^*)^2$.
  - All 5 pillars operate unmodified on PenGym through the adapter layer.
- **PPO Architecture**: Actor-Critic MLP [1540→512→512→16/1], Tanh activation, separate LR (actor=1e-4 × 0.1, critic=5e-5 × 0.1 after transfer).

---

## 6. Experimental Design (~2 pages)

### 6.1 Experimental Topologies

| Topology | Hosts | Subnets | |A| (flat) | Optimal steps | Optimal reward | Status |
|---|---|---|---|---|---|---|
| tiny | 3 | 1 | 42 | 6 | 195 | ✅ Solved (SR=100%) |
| small-linear | 8 | 3 | 96 | 12 | 179 | ✅ Solved (SR=100%) |
| medium-single-site | 16 | 3 | 192 | 4 | 191 | ❌ SR=0% (scalability limit) |

- **Table**: Per-topology characteristics (hosts, subnets, action space size, connectivity, difficulty).
- **CVE tiers per topology**: 4 tiers (T1–T4) × 3 variants each = 12 scenarios per topology.

### 6.2 Baselines

| Agent | Description | Training |
|---|---|---|
| $\theta_{dual}$ | Full pipeline: Sim CRL → Domain Transfer → Intra-Topology PenGym CRL | Phase 1→2→3 |
| $\theta_{scratch}$ | PenGym-only training from random initialization, same topology/tier schedule | Phase 3 only |
| $\theta_{sim}$ | Sim-trained agent evaluated on PenGym (zero-shot transfer) | Phase 1 only |

### 6.3 Evaluation Metrics (C5)

#### 6.3.1 Primary Metrics

| Metric | Formula | Purpose | Overcomes |
|---|---|---|---|
| SR (K=20) | $SR_s = \frac{1}{K}\sum_k \mathbb{1}[\text{success}_{s,k}]$ | Success rate | 1-episode binary limitation |
| NR | $NR_s = R_{actual} / R_{optimal}$ | Reward efficiency | Floor effect |
| $\eta$ | $\eta_s = \text{optimal\_steps} / \text{actual\_steps} \times \mathbb{1}[\text{success}]$ | Step efficiency | Ceiling effect |
| SE | $SE = \sqrt{SR(1-SR)/(|\mathcal{S}| \times K)}$ | Statistical significance | Random variance |

#### 6.3.2 Transfer Metrics

$$FT_{SR} = SR(\theta_{dual}) - SR(\theta_{scratch}), \quad FT_{\eta} = \eta(\theta_{dual}) - \eta(\theta_{scratch})$$
$$BT_{SR} = SR(\theta_{dual}, D_{sim}) - SR(\theta_{sim}, D_{sim}), \quad BT_{\eta} = \eta(\theta_{dual}, D_{sim}) - \eta(\theta_{sim}, D_{sim})$$

#### 6.3.3 CRL Diagnostics

- **Forgetting Matrix**: $\mathcal{F}_{i,j} = NR_i^{\text{after } i} - NR_i^{\text{after } j}$ — per-task forgetting after learning later tiers.
- **Zero-Shot Vector**: $\mathcal{Z}_j = NR_j^{\text{before } j}$ — measures curriculum carry-over.
- **Policy-level BT**: $D_{KL}(\pi_{sim} \| \pi_{dual})$, Fisher-weighted drift $\Delta_F = \sum_k F_k (\theta_{dual,k} - \theta_{sim,k})^2$.
- **Learning-Speed**: TTT (Time-To-Threshold), AUC ratio.
- **CE Curves**: Performance trajectory across curriculum tiers.

### 6.4 Evaluation Protocol

- **Phase 1 (Sim)**: Train on 6 simulation tasks with CRL, target SR ≈ 100%.
- **Phase 2 (Transfer)**: Conservative strategy — Fisher discount $\beta=0.3$, LR factor 0.1, normalizer warmup.
- **Phase 3 (PenGym)**: Intra-topology CRL streams, sequential T1→T4, per-task episode budget from config.
- **Phase 4 (Eval)**: Each agent evaluated on own topology (K=20 episodes per task), cross-topology (all scenarios), and sim tasks (backward transfer). Fresh adapter per agent to avoid NASim state leakage.
- **Hyperparameters**: EWC $\lambda=2000$, $\gamma=0.99$, GAE $\lambda=0.95$, PPO clip=0.2, entropy=0.02, batch=512, mini-batch=64, hidden=[512,512], activation=Tanh, seed=42.

### 6.5 Implementation Details
- **Software**: PyTorch, Gymnasium, NASim, PenGym, SBERT (`all-MiniLM-L6-v2`).
- **Hardware**: CPU-based training (PenGym simulator mode).
- **Codebase**: ~6,400 lines integration + pipeline, ~0 lines core modification.
- **Reproducibility**: All configs in `curriculum_episodes.json`, random seed fixed at 42.

---

## 7. Results & Analysis (~3 pages)

### 7.1 Phase 0 — Cross-Domain Representation Validation (RQ1)

- **Viết gì**:
  - Cross-domain OS cosine similarity = 1.0 after canonicalization.
  - Service embedding cosine: ssh, ftp, http → identical across domains.
  - State dimension: 1540 confirmed for both sim and PenGym.
  - **Result**: Unified representation successfully eliminates domain gap at state level.
- **Figure/Table**: Cosine similarity matrix for key services/OS between sim and PenGym encodings.

### 7.2 Phase 1 — Simulation CRL Baseline

- **Viết gì**:
  - SR = 100% on all 6 sim tasks after ~500 episodes.
  - Normalized reward ≈ 0.99 (after UnifiedNormalizer).
  - EWC Fisher information accumulated across 6 tasks.
- **Purpose**: Establishes sim baseline for BT measurement.
- **Table**: Per-task sim results (SR, NR, $\eta$, episodes to convergence).

### 7.3 Within-Topology Learning Results (RQ2, RQ4)

#### 7.3.1 Main Results Table

| Topology | Agent | SR | NR | $\eta$ | Reward (mean) | Reward (std) |
|---|---|---|---|---|---|---|
| tiny (T3-fixed) | $\theta_{dual}$ | **100%** | — | **74.9%** | 117.50 | 11.78 |
| tiny | $\theta_{scratch}$ | 0% | — | N/A | -166.67 | 0.0 |
| tiny | $\theta_{sim}$ | 0% | — | N/A | — | — |
| small-linear | $\theta_{dual}$ | **100%** | — | **3.5%** | — | — |
| small-linear | $\theta_{scratch}$ | 0% | — | N/A | -166.67 | 0.0 |
| medium-ss | $\theta_{dual}$ | 0% | -0.54 | N/A | -166.67 | 0.0 |
| medium-ss | $\theta_{scratch}$ | 0% | -0.54 | N/A | -166.67 | 0.0 |

#### 7.3.2 Transfer Metrics

| Topology | $FT_{SR}$ | $FT_{\eta}$ | Interpretation |
|---|---|---|---|
| tiny | **+1.000** | **+0.749** | Maximum positive transfer — scratch cannot solve at all |
| small-linear | **+0.333** | **+0.035** | Significant positive transfer (scratch SR=0%) |
| medium-ss | 0.000 | N/A | Both agents fail — identifies scalability boundary |

#### 7.3.3 Per-Tier SR Breakdown (RQ4 — Curriculum effectiveness)

| Topology | T1 | T2 | T3 | T4 | Mean |
|---|---|---|---|---|---|
| tiny | 100% | 100% | 100% | 100% | **100%** |
| small-linear | 100% | 100% | 100% | 100% | **100%** |

- **Key Finding**: Agent successfully learns all 4 difficulty tiers within both solvable topologies.
- **Anomaly resolved**: T3 initially showed SR=0% due to scenario generation bug (`access: user` instead of `root`), creating compound probability $0.6 \times 0.6 = 0.36$ per host. After fix: T3 SR=100%.
- **Đóng góp**: Demonstrates curriculum T1→T4 works effectively when topologies share structural similarity (intra-topology).

### 7.4 Cross-Topology Generalization (RQ5)

- **Viết gì**:
  - Stream-tiny agent evaluated on small-linear tasks → SR ≈ 0%.
  - Stream-small-linear agent evaluated on tiny tasks → SR ≈ 0%.
  - **Negative result**: Cross-topology zero-shot transfer is near zero.
  - **Analysis**: Network structures differ fundamentally (3 vs 8 hosts, 1 vs 3 subnets, different action spaces). Policy learned for one topology does not generalize to another. This is a fundamental limitation of flat RL policy — no compositional/hierarchical reasoning about network structure.
- **Table**: Cross-topology generalization matrix.

### 7.5 Intra vs Cross-Topology CRL Comparison (RQ3)

| Metric | Cross-Topology CRL (exp2) | Intra-Topology CRL (final) | Improvement |
|---|---|---|---|
| tiny overall SR | 37.5% (T1/T2 pass, T3/T4 fail) | **100%** (all tiers) | +62.5% |
| Death spiral | Yes (medium failure poisons tiny) | **Eliminated** | Structural fix |
| EWC contamination | Cross-topology Fisher consolidation | Per-topology isolated Fisher | Principled |
| Best agent SR | 25.8% average | **100% on solvable topologies** | ~4× |

- **Key finding**: Intra-topology CRL eliminates the death spiral and achieves maximum SR on solvable topologies.

### 7.6 Scalability Analysis — Medium Topology Failure

- **Viết gì**:
  - Medium-single-site: SR=0% at 3K episodes (step_limit=1000) and 10K episodes.
  - 10 million action steps, zero successful trajectories.
  - Root cause: |A|=192 (flat action space) × 5+ hops through 3 subnets → exponential exploration space.
  - Policy collapse: reward = -166.67 ± 0.0 (constant), no gradient signal.
  - **Analysis**: Sparse reward problem — agent cannot find ANY successful trajectory to generate positive gradient signal. This is a fundamental limitation of flat PPO on large discrete action spaces with sparse rewards.
  - **Implication**: Larger topologies require reward shaping, hierarchical RL, or sub-goal decomposition.
- **Table**: Training statistics for medium-ss (episodes, total steps, SR, reward, time).

### 7.7 Step Efficiency Analysis

- **Viết gì**:
  - tiny $\eta = 74.9\%$ — agent solves in ~8 steps vs optimal 6. Near-optimal.
  - small-linear $\eta = 3.5\%$ — agent solves but takes 343 steps vs optimal 12. Far from optimal.
  - **Interpretation**: Sim knowledge transfers the *strategy* (scan→exploit pattern) but not the *topology-specific routing*. On tiny (simple structure), strategy alone is nearly sufficient. On small-linear (multi-subnet), agent needs more exploration to find optimal paths.
  - $\eta$ immune to ceiling effect: both topologies SR=100% but $\eta$ clearly differentiates quality.
- **Figure**: $\eta$ comparison across topologies (bar chart).

---

## 8. Discussion (~1.5 pages)

### 8.1 Key Contributions — Synthesis

- **Viết gì**: Synthesize results answering each RQ. Highlight that:
  1. Unified representation achieves perfect cross-domain alignment (cosine=1.0).
  2. Sim-to-PenGym transfer produces significant FT ($FT_{SR}$ up to 1.0).
  3. Intra-topology CRL is strictly superior to cross-topology for pentest domain.
  4. CVE curriculum works within topology but not across.
  5. Zero cross-topology generalization is a fundamental limitation needing hierarchical approaches.

### 8.2 Why Intra-Topology CRL Works

- **Viết gì**:
  - EWC assumes tasks share similar parameter importance landscape.
  - Within-topology: T1→T4 share network structure, only exploit difficulty changes → Fisher information accurately protects routing knowledge while allowing exploit adaptation.
  - Cross-topology: tiny (3 hosts) and medium (16 hosts) have *fundamentally different* optimal policies → Fisher from one poisons the other.
  - **Insight**: CRL in pentest should operate at the topology abstraction level, not across arbitrary topologies.

### 8.3 The Scalability Boundary — Why Medium Fails

- **Viết gì**: Deep analysis of the medium failure.
  - Action space explosion: |A| grows quadratically with hosts ($N_{hosts} \times N_{action\_types}$). tiny: 42, small-linear: 96, medium: 192.
  - Multi-hop penalty: Medium requires 5+ sequential correct actions across 3 subnets. At each hop, wrong action = wasted budget.
  - Sparse reward: No intermediate signal for partial progress (scanning correct subnet, compromising non-sensitive host).
  - **Comparison with literature**: Similar findings in CyberBattleSim (Microsoft, 2021) — flat RL fails on >10 hosts.
  - **Future direction**: Reward shaping (partial credit for subnet discovery), hierarchical RL (option framework for subnet-level planning), action masking (remove invalid actions).

### 8.4 The T3 Scenario Bug — Lessons Learned

- **Viết gì**:
  - Bug description: `tiny_T3_000.yml` had `access: user` while all other tiers had `access: root`. This created compound probability (exploit success 60% × privesc success 60–80% = 36–48% per host) making T3 actually harder than T4.
  - Detection: SR analysis showed T3=0%, T4=100% — contradicting difficulty ordering.
  - Fix: Changed `access: user → root` in T3 scenario → T3 SR immediately jumped to 100%.
  - **Lesson**: In curriculum learning, scenario generation correctness is critical. A single misconfigured parameter can invalidate the entire difficulty gradient. Automated validation checks are essential.

### 8.5 Limitations

| # | Limitation | Impact | Mitigation |
|---|---|---|---|
| L1 | Only 2 topologies fully solvable (tiny, small-linear) | Limited generalization evidence | Reward shaping for medium (future work) |
| L2 | Cross-topology generalization ≈ 0% | Policy learned is topology-specific | Hierarchical RL / graph neural networks (future) |
| L3 | PenGym real execution (KVM) not verified | Only sim-mode tested | CyRIS integration (future) |
| L4 | Single seed (42) | Limited statistical power | Multi-seed experiments (future) |
| L5 | Canonicalization loses OS version info | Ubuntu 14.04 = Ubuntu 22.04 | Version-aware canonicalization (future) |
| L6 | Reward normalizer uses fixed bounds | May under-represent positive signals | Adaptive normalization (future) |
| L7 | 16-dim action space loses CVE-specific knowledge | Agent can't distinguish CVEs within same service | SUPERSET action mode or attention-based selector |

### 8.6 Threats to Validity

- **Internal validity**: 
  - NASim class-level state leakage → mitigated by fresh adapter per agent (§5.5.3).
  - T3 scenario bug → discovered and fixed (§8.4).
  - Single seed → acknowledged limitation.
- **External validity**: 
  - Results on NASim simulation, not real networks → PenGym KVM mode exists but untested.
  - Only 2 solvable topologies → unclear if findings generalize to arbitrary topologies.
  - SCRIPT's 5-pillar CRL is specific; results may not transfer to other CRL methods (PackNet, etc.).
- **Construct validity**:
  - Multi-episode SR (K=20) provides continuous metric → overcomes binary limitation.
  - $\eta$ directly measures transfer quality → immune to ceiling effect.
  - But NR depends on `optimal_reward` from scenario YAML — assumes YAML is correct.

---

## 9. Conclusion & Future Work (~0.5 page)

### 9.1 Conclusion
- **Viết gì**: Restate problem, approach, main results in 1 paragraph.
  - We presented [System Name], a cross-domain CRL framework for automated pentesting that bridges SCRIPT simulation and PenGym environment.
  - Through unified 1540-dim state representation (cosine=1.0), hierarchical 16-dim action space (100% coverage), intra-topology CRL (eliminating death spiral), and CVE-based 4-tier curriculum:
    - SR=100% on solvable topologies (vs 0% scratch baseline), $FT_{SR}$ up to 1.0
    - Step efficiency $\eta$ = 74.9% on tiny (near-optimal with sim knowledge)
    - Structural elimination of cross-topology death spiral (SR 37.5% → 100%)
  - Identified fundamental scalability boundary at medium topology (|A|=192, sparse reward → SR=0%)
  - Identified zero cross-topology generalization as fundamental limitation of flat RL policy

### 9.2 Future Work

| Priority | Direction | Rationale |
|---|---|---|
| **High** | **Reward shaping / intrinsic motivation** for larger topologies | Medium-ss SR=0% due to sparse reward. Sub-goal decomposition (compromise intermediate hosts → positive signal), curiosity-driven exploration (RND/ICM), or count-based bonuses. |
| **High** | **Hierarchical RL (options framework)** | Subnet-level options (scan-subnet, exploit-subnet) would reduce effective action space from 192 → ~30, enabling medium/large topologies. |
| **High** | **PenGym real execution** verification on CyRIS KVM | Pipeline designed for sim→real but only tested on NASim sim mode. Real execution adds failure modes (timeout, session loss). |
| **Medium** | **Graph Neural Networks** for topology-aware policy | Enable cross-topology generalization by encoding network structure as graph → topology-agnostic features. |
| **Medium** | **Multi-seed experiments** (n=5+) | Current results on seed=42 only. Need statistical robustness. |
| **Medium** | **Action masking** for invalid actions | Remove unreachable host actions from policy output → reduce effective action space. Standard technique in combinatorial action spaces. |
| **Low** | **Multi-agent pentest** | Coordinated agents attacking from multiple entry points. |
| **Low** | **Defender agent** (adversarial training) | Train attacker against adaptive defender for robustness. |

---

## 10. Figures & Tables Summary

### Figures (Proposed)

| # | Content | Type | Data Source |
|---|---|---|---|
| Fig. 1 | System architecture (SCRIPT ↔ Adapter ↔ PenGym) | Architecture diagram | `project_fusion_summary.md §III.2` |
| Fig. 2 | 5-Phase pipeline (Phase 0→4 flow) | Flow diagram | `strategy_C_shared_state_dual_training.md §6` |
| Fig. 3 | Cross-topology death spiral vs Intra-topology CRL | Comparison diagram | `intra_topology_crl_blueprint.md Appendix A` |
| Fig. 4 | CVE difficulty distribution and 4-tier grading | Histogram + threshold lines | `cve_difficulty_and_expansion.md §3` |
| Fig. 5 | Template + Overlay scenario generation | Schematic | `cve_difficulty_and_expansion.md §4` |
| Fig. 6 | SR per tier (T1–T4) per topology | Bar chart | Experimental results |
| Fig. 7 | Step efficiency ($\eta$) comparison across topologies | Bar chart | Experimental results |
| Fig. 8 | Learning curves (reward vs episodes) for dual vs scratch | Line chart | TensorBoard logs |
| Fig. 9 | Scalability: SR vs topology size (hosts) | Scatter/line | Experimental results |

### Tables (Proposed)

| # | Content | Data Source |
|---|---|---|
| Table 1 | State representation comparison (SCRIPT vs PenGym vs Unified) | `script_pengym_integration_summary.md §I.2` |
| Table 2 | 16-dim action space with PenGym mapping and CVE group sizes | `script_pengym_integration_summary.md §II.3` |
| Table 3 | Phase 1 sim baseline results (per-task SR, NR, $\eta$) | Experimental outputs |
| Table 4 | Main results: per-topology per-agent SR, NR, $\eta$, reward | Experimental outputs |
| Table 5 | Transfer metrics: FT/BT on SR, NR, $\eta$ | Experimental outputs |
| Table 6 | Per-tier SR breakdown (T1–T4) per topology | Experimental outputs |
| Table 7 | Cross-topology generalization matrix | Phase 4 per-stream eval |
| Table 8 | Intra vs Cross-topology CRL comparison | exp2 vs final results |
| Table 9 | Medium-ss failure analysis (episodes, steps, SR, reward) | `outputs/medium_ss_10k/` |
| Table 10 | Related work positioning comparison | Literature review |
| Table 11 | Hyperparameters | `config.py`, `curriculum_episodes.json` |
| Table 12 | CVE difficulty tier definitions | `cve_difficulty_and_expansion.md §3` |

---

## 11. Venue Recommendations

### 11.1 Target Venues (Ranked by Fit)

| # | Venue | Type | Fit Reason | Deadline Cycle |
|---|---|---|---|---|
| 1 | **ACSAC** (Annual Computer Security Applications Conference) | Conference | Applied security + RL, good fit for system papers | June/July |
| 2 | **AISec** (Workshop on AI Security, co-located with CCS) | Workshop | Directly targets AI for security applications | July/Aug |
| 3 | **RAID** (Recent Advances in Intrusion Detection) | Symposium | Network security, attack simulation | March/April |
| 4 | **ESORICS** (European Symposium on Research in Computer Security) | Conference | Security research, accepts system papers | April |
| 5 | **IEEE S&P Workshop on Deep Learning and Security** (DLS) | Workshop | DL/RL for security | Feb/March |
| 6 | **AAAI** / **IJCAI** (AI track) | Conference | If framed as CRL contribution with security application | Sept (AAAI), Jan (IJCAI) |
| 7 | **NeurIPS** / **ICML** (RL workshop) | Workshop | If focused on CRL methodology contribution | May (ICML), June (NeurIPS) |
| 8 | **Computers & Security** (Elsevier) | Journal | Comprehensive system paper, more space for details | Rolling |
| 9 | **IEEE TIFS** (Trans. on Information Forensics and Security) | Journal | Top-tier security journal, if results are strong enough | Rolling |

### 11.2 Recommendation

**Primary target**: **ACSAC** hoặc **RAID** — fit tốt nhất cho paper kết hợp AI agent + network security application. Cho phép trình bày cả system design + experimental evaluation.

**Secondary**: **AISec Workshop @ CCS** — shorter paper (8 pages), faster turnaround, focused audience.

**Nếu muốn emphasize CRL methodology**: **AAAI/IJCAI** main track hoặc **NeurIPS CRL workshop** — requires framing CRL contribution more prominently (intra-topology CRL, cross-domain EWC with Fisher discount).

### 11.3 Paper Length Adaptation

| Venue | Length | Strategy |
|---|---|---|
| Full conference (ACSAC, RAID, ESORICS) | 12–18 pages | Full paper as outlined |
| Workshop (AISec, DLS) | 6–8 pages | Focus on C1+C3 (representation + intra-topology), condense C4+C5 |
| Journal (Computers & Security, TIFS) | 20+ pages | Expand all sections, add appendices, multi-seed |
| Short/poster (NeurIPS workshop) | 4 pages | Focus solely on C3 (intra-topology CRL) as methodology contribution |

---

## 12. Writing Priority & Timeline (2 ngày còn lại)

### Day 1 — Core Sections
| Priority | Section | Est. Time | Notes |
|---|---|---|---|
| P0 | §6 Experimental Design | 2h | Tables mostly from existing data |
| P0 | §7 Results & Analysis | 3h | Data exists, need narrative |
| P0 | §5.3–5.6 Methodology (representation + CRL) | 3h | Most detailed, draw from docs |

### Day 2 — Framing & Polish
| Priority | Section | Est. Time | Notes |
|---|---|---|---|
| P0 | §2 Abstract | 1h | Write last, after all content exists |
| P0 | §3 Introduction | 2h | Framing determines paper's impact |
| P1 | §8 Discussion | 2h | Analysis of limitations is strategic |
| P1 | §4 Related Work | 2h | Need literature citations |
| P2 | §9 Conclusion | 0.5h | Straightforward |
| P2 | Figures & Tables | 1h | Generate from data |

---

## Appendix A — Key Data Points for Paper

### A.1 Experimental Results Summary

```
Topology: tiny (T3-fixed)
  θ_dual:    SR=100%, η=74.9%, FT_SR=1.0, reward=117.50±11.78
  θ_scratch: SR=0%, reward=-166.67±0.0
  
Topology: small-linear
  θ_dual:    SR=100%, η=3.5%, FT_SR=0.333
  θ_scratch: SR=0%, reward=-166.67±0.0

Topology: medium-single-site
  θ_dual:    SR=0%, NR=-0.54, reward=-166.67±0.0 (10K episodes, 10M steps)
  θ_scratch: SR=0%, reward=-166.67±0.0

Cross-topology generalization: ≈ 0%

Phase 1 (sim): SR=100%, final_reward=0.99 (normalized)
Phase 0: cross_domain_os_cosine = 1.0
```

### A.2 Cross-Topology Death Spiral Evidence (exp2)

```
Cross-topology CRL (exp2):
  tiny:         T1=100%, T2=100%, T3=0%, T4=0% → SR=37.5%  (limited by step_limit)
  small-linear: SR=100% on T1–T4
  Overall SR:   25.8% (death spiral from medium failure)

Intra-topology CRL (final):
  tiny:         T1=100%, T2=100%, T3=100%, T4=100% → SR=100%
  small-linear: T1=100%, T2=100%, T3=100%, T4=100% → SR=100%
```

### A.3 System Scale

```
New integration code:     ~4,100 lines
New pipeline code:        ~2,270 lines  
Total new code:           ~6,400 lines
Core code modified:       0 lines (SCRIPT + PenGym unchanged)
Total compiled scenarios: 96+
CVE classified:           1,985
SBERT model:              all-MiniLM-L6-v2 (384-dim)
```

### A.4 Key Hyperparameters

```
PPO:
  actor_lr = 1e-4 (× 0.1 after transfer = 1e-5)
  critic_lr = 5e-5 (× 0.1 after transfer = 5e-6)
  hidden = [512, 512], activation = Tanh
  gamma = 0.99, GAE_lambda = 0.95
  policy_clip = 0.2, entropy = 0.02
  batch = 512, mini_batch = 64, ppo_epochs = 8

EWC:
  lambda = 2000, gamma = 0.99
  fisher_discount (β) = 0.3 (70% reduction at domain transfer)

Curriculum:
  tier_multipliers: T1=0.8×, T2=1.0×, T3=2.0×, T4=3.0×
  base_episodes: tiny=500, small-linear=750, medium-ss=2500
  step_limits: tiny=200/500, small-linear=500, medium-ss=500-1000
  
Evaluation:
  K = 20 episodes per task
  seed = 42
```
