# Paper Outline — Cross-Domain Continual Reinforcement Learning for Automated Penetration Testing

> **Version:** 2.0 (Honest Rewrite)  
> **Date:** 2026-03-01  
> **Branch:** `strC_1`  
> **Status:** Rewritten with multi-seed evidence. Framing based on methodology contribution + empirical findings (including negative results).
>
> **Framing Strategy:** Paper vị trí là _first empirical study_ bridging two fundamentally different pentest RL systems. Contribution chính là FRAMEWORK + METHODOLOGY + EMPIRICAL INSIGHTS (kể cả negative). Không overclaim transfer effectiveness — thay vào đó, cung cấp bài học cho cộng đồng về challenges, pitfalls, và design principles cho cross-domain CRL trong cybersecurity.

---

## 0. Research Foundation

### 0.1 Research Problem

Các hệ thống RL cho penetration testing phát triển **cô lập trong từng domain**, tạo ra hai hướng nghiên cứu không giao nhau:

1. **Domain Isolation**: Simulation-based RL agents (SCRIPT, DRL-pentest) đạt hiệu suất cao trên environment riêng nhưng không thể triển khai trên environment khác — do sự không tương thích cơ bản về state representation (SBERT semantic 1538-dim vs positional binary), action space (2064 CVE-specific vs 12 service-level), reward scale (100× difference), và interaction paradigm (single-host sequential vs multi-host network). Chưa có nghiên cứu nào xây dựng cầu nối giữa hai systems.

2. **Thiếu multi-dimensional curriculum**: Môi trường PenGym/NASim có exploit probability gần 1.0 (trivially easy) và không có cơ chế phân loại độ khó exploit — agent chỉ học network routing mà không có gradient khó tăng dần từ CVE metadata thực.

3. **Thiếu evaluation framework cho cross-domain CRL**: Existing work dùng 1-episode binary SR hoặc cumulative reward — không đủ để đánh giá forward/backward transfer, catastrophic forgetting, hay step efficiency trong ngữ cảnh cross-domain.

### 0.2 Research Gap

| Hướng nghiên cứu             | Existing Work                                                   | Gap                                                                |
| ---------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------ |
| RL Pentest trên simulation   | SCRIPT (Tran et al.), DRL-based pentest (Schwartz & Kurniawati) | Agent chỉ hoạt động trên simulation riêng, không transfer được     |
| RL Pentest trên NASim/PenGym | PenGym (Bhatnagar et al.), NASim agents                         | Không có CRL, không có CVE-level difficulty, eval giới hạn         |
| Continual RL                 | EWC, PackNet, Progress&Compress                                 | Áp dụng trong Atari/MuJoCo, chưa cho network security cross-domain |
| Domain Transfer RL           | Domain randomization, sim-to-real (robotics)                    | Cho continuous control; chưa cho discrete-action network security  |
| Curriculum RL                | CL cho task scheduling                                          | Curriculum theo topology size, chưa theo CVE exploit difficulty    |

**Gap tổng hợp**: Chưa có nghiên cứu nào (a) xây dựng unified representation framework để kết nối hai hệ thống pentest RL có kiến trúc khác biệt cơ bản, (b) khảo sát empirically liệu CRL cross-domain có hiệu quả trong cybersecurity, và (c) cung cấp CVE-based difficulty curriculum + comprehensive evaluation framework cho pentest CRL.

### 0.3 Hypothesis & Research Scope

> **H (Primary):** Một unified representation framework (shared state 1540-dim SBERT encoding + hierarchical action 16-dim + normalized reward) cho phép policy train trên một simulation hoạt động trên environment khác — đạt cross-domain alignment tại representation level (cosine similarity ≥ 0.95).

> **H (Secondary — Exploratory):** CRL with EWC and Fisher discount khi áp dụng cross-domain sẽ duy trì _một phần_ sim knowledge (BT > -0.5) và tạo learning advantage so với training from scratch.

> **Kết quả thực tế (n=5 seeds):**
>
> - **H-Primary: SUPPORTED** — cosine similarity = 1.0, representation alignment hoàn hảo ✅
> - **H-Secondary: PARTIALLY REFUTED** — dual agent SR = 0.562–0.637, nhưng:
>   - BT_SR = -0.767 (EWC không bảo vệ sim knowledge — catastrophic forgetting)
>   - Scratch baseline có bug (inherits EWC λ=2000 → SR=0%), true scratch (λ=0) SR ≈ 0.75–0.875
>   - Forward transfer chưa xác nhận positive (FT_SR estimated ≈ -0.19 khi so với true scratch)
>   - Fisher discount β>0 cải thiện SR so với β=0 (0.375 → 0.525), nhưng EWC tổng thể vẫn gây catastrophic BT
>
> **Paper scope:** Trình bày framework, pipeline, methodology contributions (C1-C5) là đóng góp chính. Report experimental results trung thực bao gồm cả negative findings — đây là bài học quan trọng cho cộng đồng về challenges của cross-domain CRL trong cybersecurity.

### 0.4 Research Questions

| RQ      | Câu hỏi                                                                                               | Metric                                             | Kết quả thực nghiệm (n=5×2 seeds)                                                       |
| ------- | ----------------------------------------------------------------------------------------------------- | -------------------------------------------------- | --------------------------------------------------------------------------------------- |
| **RQ1** | Unified representation (1540-dim SBERT) có loại bỏ domain gap tại representation level không?         | Cross-domain cosine similarity, Phase 0 validation | cosine = 1.0 ✅ — representation alignment hoàn hảo                                     |
| **RQ2** | Agent với transferred weights đạt SR bao nhiêu trên PenGym? Các yếu tố nào ảnh hưởng (β, canon, CRL)? | SR, NR, η per topology/tier                        | dual SR=0.562±0.096 (intra), 0.637±0.123 (cross). β>0 giúp (+0.15 SR). Canon Δ=0.05 nhỏ |
| **RQ3** | EWC Fisher discount (β) ảnh hưởng thế nào đến cân bằng plasticity–stability trong domain transfer?    | SR vs β, BT_SR                                     | β=0→0.375, β=0.3→0.525. EWC giữ plasticity (β>0) nhưng vẫn catastrophic BT (-0.767)     |
| **RQ4** | CVE-based 4-tier curriculum tạo difficulty gradient có ý nghĩa trên cùng topology không?              | Per-tier SR progression                            | Inconclusive: T2=T4=1.0, T1=0.8, T3=0.0 (scenario bug). small-linear high variance      |
| **RQ5** | Cross-topology zero-shot transfer có khả thi không?                                                   | Cross-topology eval SR                             | ≈ 0% — negative result confirmed. Policy không generalize qua topologies ✅             |

---

## 1. Title — Proposed Options

### Option A (Recommended — Framework + Empirical Study):

> **PenSCRIPT: A Cross-Domain Continual Reinforcement Learning Framework for Automated Penetration Testing**

### Option B (Emphasize Empirical Findings):

> **Bridging Simulation and Realistic Pentest Environments: An Empirical Study of Cross-Domain Continual RL Transfer**

### Option C (Concise, Workshop-oriented):

> **PenSCRIPT: Cross-Domain CRL for Pentesting via Unified Representation and CVE Curriculum**

### Option D (Highlight Challenges — cho negative-result-friendly venue):

> **Challenges of Cross-Domain Transfer in RL-Based Penetration Testing: Lessons from Bridging SCRIPT and PenGym**

> **Lưu ý:** Tránh dùng "Sim-to-Real" vì cả hai domain đều là simulation (SCRIPT sim → PenGym NASim mode). Dùng "Cross-Domain" hoặc "Cross-Framework" thay thế.

---

## 2. Abstract (Outline — ~250 words)

### Paragraph 1 — Context & Problem

- RL for automated penetration testing has demonstrated success in individual simulation environments, yet policies trained in one system cannot operate in another due to fundamental incompatibilities in state representation, action space, reward scale, and interaction paradigm.
- This domain isolation limits both the practical applicability and the cumulative knowledge development across pentest RL systems.
- Furthermore, there is no standardized difficulty curriculum based on real CVE metadata, and existing evaluation relies on single-episode binary metrics insufficient for assessing transfer and forgetting.

### Paragraph 2 — Proposed Approach

- We present **PenSCRIPT**, a cross-domain continual RL framework that bridges SCRIPT simulation and PenGym environment through:
  - (1) A **unified state representation** (1540-dim SBERT-based encoding with cross-domain canonicalization, achieving cosine similarity = 1.0).
  - (2) A **hierarchical action space** reducing 2064 CVE actions to 16 service-level groups (~100% cross-domain coverage).
  - (3) A **5-phase CRL pipeline** with domain transfer manager using Fisher discount (β) to calibrate EWC cross-domain.
  - (4) A **CVE-based difficulty curriculum** grading 1985 CVEs into 4 tiers via composite difficulty score.
  - (5) A **multi-dimensional evaluation framework** (23 metrics including FT/BT, forgetting matrix, step efficiency).
- The framework integrates via adapter pattern with 0 lines of core code modification (~6,400 lines of new integration code).

### Paragraph 3 — Key Results (Honest)

- Multi-seed experiments (n=5 per configuration, 17 runs total) reveal:
  - **Representation alignment succeeds**: cross-domain cosine = 1.0, confirming unified encoding eliminates the representation gap (RQ1).
  - **PenGym task learning**: Dual agent achieves SR = 0.562±0.096 (intra-topology) and 0.637±0.123 (cross-topology), solving both solvable topologies with varying success.
  - **Fisher discount matters**: β>0 improves SR by +0.15 over β=0, confirming the need to reduce sim Fisher penalty during domain transfer (RQ3). However, exact β value is not sensitive (β=0.3 ≈ β=0.7).
  - **EWC fails to prevent backward forgetting**: BT_SR = -0.767, indicating EWC's quadratic penalty is insufficient for preserving knowledge across fundamentally different domains.
  - **Cross-topology generalization ≈ 0%**: Policies are topology-specific; flat RL without structural reasoning cannot generalize across network architectures (RQ5).
  - **Experimental limitations**: Scratch baseline bug and T3 scenario configuration error affect two metrics; these are documented transparently.
- Results provide the first empirical evidence base for cross-domain CRL in cybersecurity, identifying both achievable goals (representation unification) and fundamental challenges (catastrophic backward transfer, topology-specific policies).

### Paragraph 4 — Contribution Summary

- 5 contributions: unified representation framework (C1), cross-domain CRL pipeline (C2), intra-topology CRL architecture (C3), CVE curriculum system (C4), multi-dimensional evaluation framework (C5).
- Negative findings on EWC backward transfer and cross-topology generalization serve as important baselines for future work.
- Code and data publicly available at [repo link].

---

## 3. Introduction (~2 pages)

### 3.1 Opening — Real-world Motivation (¶1–2)

- **Viết gì**: Cybersecurity landscape — tần suất & chi phí tấn công mạng tăng, nhu cầu automated pentest.
- **Logic dẫn dắt**: Manual pentest is expensive and doesn't scale → AI-powered autonomous pentesting is emerging → RL is a natural formulation (MDP over network states).
- **Cite**: IBM X-Force report, NIST guidelines, early works on automated pentest.

### 3.2 State of RL for Pentesting (¶3–4)

- **Viết gì**: Hai hướng tiếp cận chính:
  - _Simulation-only RL_ (SCRIPT, DRL-pentest agents): Train trên simulation riêng, đạt hiệu suất cao nhưng không deployable vào env khác.
  - _Environment-focused_ (NASim, PenGym, CyberBattleSim): Cung cấp env chuẩn (Gymnasium API) nhưng thiếu agent mạnh, thiếu CRL, thiếu CVE diversity.
- **Research gap**: Không có framework nào kết nối hai hướng — sim agent không chạy được trên NASim, NASim agent không có CRL.
- **Điểm mới**: Phát biểu rõ ràng 8 xung đột kỹ thuật (state, action, reward, paradigm, interface, transfer mechanism, CVE diversity, evaluation) between the two systems.

### 3.3 Challenges of Cross-Domain Transfer (¶5–6)

- **Viết gì**: Chi tiết 3 challenge chính (framed as OPEN CHALLENGES — paper tackles but does not fully solve all):
  1. **Representation mismatch**: Không chỉ dimension mismatch mà là paradigm khác nhau (semantic SBERT vs positional binary). Cần unified encoding đảm bảo same entities → identical representations across domains.
  2. **EWC calibration across domains**: Fisher Information computed on sim domain encodes sim-specific parameter importance. Naively applying sim Fisher penalty during cross-domain fine-tuning restricts plasticity — requires calibration (Fisher discount β). Even with calibration, preserving backward knowledge remains a fundamental challenge.
  3. **No exploit difficulty gradient**: PenGym exploits are trivially easy (prob ≈ 1.0) → agent only learns routing, not attack tactics. Curriculum needs exploit-difficulty dimension based on real CVE metadata.
- **Điểm mới**: Explicitly identify these as OPEN CHALLENGES in the intersection of CRL and cybersecurity. Paper provides first empirical investigation.

### 3.4 Proposed Solution — High-Level (¶7–8)

- **Viết gì**: One-paragraph summary of approach + architecture figure reference.
- Pipeline: Phase 0 (Validation) → Phase 1 (Sim CRL) → Phase 2 (Domain Transfer) → Phase 3 (Intra-Topology CRL on PenGym) → Phase 4 (Multi-dimensional Evaluation).
- Key design principles: "Giữ nguyên lõi, kết nối qua adapter" — không sửa SCRIPT core, không sửa PenGym core, tích hợp qua unified representation layer.

### 3.5 Contributions (¶9)

**C1 — Unified Cross-Domain Representation Framework:**
Thiết kế unified state encoder (1540-dim, SBERT-based) với canonicalization map đạt cross-domain cosine = 1.0, kèm hierarchical action space (2064 → 16-dim, ~100% coverage), và unified reward normalizer ([-1, +1]) — cho phép policy train trên SCRIPT simulation thực thi trực tiếp trên PenGym mà không cần architecture modification. Validation: Phase 0 xác nhận perfect alignment. _So với existing work_: Các hệ thống trước dùng representation riêng; chưa có work nào thống nhất SBERT semantic encoding với NASim binary encoding cho pentest RL.

**C2 — Cross-Domain CRL Pipeline (5-Phase):**
Pipeline Phase 0→4 with domain transfer manager using Fisher discount (β) để calibrate EWC cross-domain. Empirical finding: β>0 cải thiện SR +0.15 so với β=0 (0.375 → 0.525), xác nhận Fisher discount là cần thiết. Pipeline cho phép CRL 5 pillars hoạt động across domains thông qua adapter pattern (~6,400 lines integration, 0 lines core modification). _So với existing work_: EWC applied within single domain; paper is first to apply and empirically evaluate EWC cross-domain with Fisher discount trong cybersecurity.

**C3 — Intra-Topology CRL Architecture:**
Per-topology independent CRL streams via `deepcopy()` forking, cho phép mỗi topology có independent EWC Fisher accumulation. Empirical investigation: intra (0.562±0.096) vs cross (0.637±0.123), p=0.365 — no significant difference in 2-topology setup. However, prior evidence shows cross-topology CRL with failing topologies triggers EWC contamination. Architecture provides isolation pattern. _So với existing work_: Standard CRL trains sequentially across all tasks; paper proposes and evaluates topology-aware stream isolation.

**C4 — CVE-Based Difficulty Curriculum:**
1985 CVE difficulty grading via composite score $S_{diff} = 0.50(1-\text{prob}) + 0.25 f_{AC} + 0.15 f_{PR} + 0.10 f_{UI}$ into 4 tiers (T1–T4). Template+Overlay scenario generation (8 topologies × 4 tiers × variants = 96+ scenarios). _So với existing work_: PenGym/NASim scenarios have no difficulty variation; paper adds exploit-difficulty dimension to curriculum as reusable resource.

**C5 — Multi-Dimensional Evaluation Framework + Empirical Baseline:**
23 evaluation components including multi-episode SR (K=20), Normalized Reward, Step Efficiency (η), FT/BT Transfer metrics, Forgetting Matrix, Zero-Shot Transfer Vector. Most importantly: comprehensive empirical results (17 runs, 5 seeds × 2 configs + 7 ablations) establishing the FIRST baseline for cross-domain CRL in pentesting — including negative findings (catastrophic BT, topology-specific policies, EWC limitations) that inform future research directions. _So với existing work_: SCRIPT uses 1-episode binary SR; paper provides first comprehensive CRL evaluation for pentest domain.

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
- **Distinction vs our work**: Existing CRL operates within single domain; we apply CRL _across_ simulation and realistic pentest domains with domain transfer.

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
- **Gap**: Curriculum in pentest is limited to topology size progression; no work grades exploit _difficulty_ as a curriculum dimension using real CVE metadata.

### 4.6 Positioning Table

| Feature               | SCRIPT         | PenGym            | CyberBattleSim   | AutoPentest-DRL | PenSCRIPT            |
| --------------------- | -------------- | ----------------- | ---------------- | --------------- | -------------------- |
| RL Agent              | PPO            | DQN (basic)       | RL (various)     | DQN             | PPO + CRL 5 pillars  |
| CRL                   | ✅ (5 pillars) | ❌                | ❌               | ❌              | ✅ (cross-domain)    |
| Sim-to-Real           | ❌             | ✅ (KVM)          | ❌               | ❌              | ⚠️ (cross-framework) |
| Cross-domain transfer | ❌             | ❌                | ❌               | ❌              | ✅                   |
| CVE diversity         | 2064 CVE       | 5 service types   | Microsoft graphs | NASim scenarios | 1985 CVE → 16 groups |
| Exploit curriculum    | ❌             | ❌                | ❌               | ❌              | ✅ (T1→T4)           |
| Eval framework        | 1-episode SR   | cumulative reward | win/loss         | episodic reward | **23 metrics (C5)**  |

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
  - Canonicalization map: Apply _before_ SBERT encoding.
  - Validation: Phase 0 confirms `cross_domain_os_cosine = 1.0`.
- **Equation**: $\text{canon}(x) = \text{CANONICAL\_MAP}(x_{\text{lower}})$ with version stripping.
- **Đóng góp**: Canonicalization cải thiện alignment — ablation: canon ON SR=0.525, canon OFF SR=0.475 (Δ=0.05). Hiệu ứng nhỏ nhưng positive; cosine=1.0 đạt được WITH canonicalization, confirming nó đóng vai trò cần thiết cho perfect alignment.

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

| Phase | Name             | Input                  | Output                   | Key Operations                                            |
| ----- | ---------------- | ---------------------- | ------------------------ | --------------------------------------------------------- |
| 0     | Validation       | SBERT model, scenarios | GO/NO-GO                 | Cross-domain cosine ≥ 0.95, dims check                    |
| 1     | Sim CRL          | 6 sim tasks            | $\theta_{sim}$ (SR=100%) | CRL 5 pillars, EWC Fisher accumulation                    |
| 2     | Domain Transfer  | $\theta_{sim}$         | $\theta_{transferred}$   | State norm reset, Fisher discount $\beta$, LR×0.1, warmup |
| 3     | PenGym Fine-tune | PenGym tasks T1→T4     | $\theta_{dual}$          | CRL streams, per-task episode/step-limit                  |
| 4     | Evaluation       | 3 agents × all tasks   | Report                   | Multi-dimensional FT/BT, forgetting matrix                |

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

#### 5.6.1 Cross-Topology EWC Interaction — Motivation Analysis

- **Viết gì**:
  - Define the concern: When CRL trains sequentially across topologies with fundamentally different state/reward distributions, EWC Fisher consolidation from one topology may constrain parameter adaptation for another.
  - Preliminary evidence (exp2, single-seed, 3 topologies): cross-topology CRL achieved 25.8% overall SR — when medium fails (SR=0%), EWC consolidates failure parameters.
  - **Nuanced finding (multi-seed, 2 topologies):** In the 2-topology setup (tiny + small-linear, both partially solvable), cross-topology CRL (SR=0.637) actually performs comparably to intra-topology (SR=0.562), p=0.365. This suggests EWC contamination is primarily triggered when a FAILING topology is included in the CRL sequence — not by cross-topology training per se.
  - Intra-topology CRL via deepcopy remains a sound PRECAUTIONARY architecture — it eliminates any risk of cross-topology interference, even if the risk doesn't materialize in small-scale setups.
- **Figure**: Stream architecture diagram (cf. intra_topology_crl_blueprint.md Appendix A).

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

| Topology           | Hosts | Subnets | \|A\| (flat) | Optimal steps | Status                                                  |
| ------------------ | ----- | ------- | ------------ | ------------- | ------------------------------------------------------- |
| tiny               | 3     | 1       | 42           | 6             | ✅ Solved (SR=0.70 intra, 0.62 cross)                   |
| small-linear       | 8     | 3       | 96           | 12            | ⚠️ Partially (SR=0.43 intra, 0.65 cross, high variance) |
| medium-single-site | 16    | 3       | 192          | —             | ❌ SR=0% (scalability limit)                            |

- **Table**: Per-topology characteristics (hosts, subnets, action space, connectivity, difficulty).
- **CVE tiers per topology**: 4 tiers (T1–T4) × variants.
- **Note**: Benchmark baselines (random, greedy, DQN, A2C) ALL achieve SR=100% on tiny → tiny is trivially easy. small-linear is the primary topology for evaluating transfer value.

### 6.2 Baselines

| Agent              | Description                                                                  | Training     |
| ------------------ | ---------------------------------------------------------------------------- | ------------ |
| $\theta_{dual}$    | Full pipeline: Sim CRL → Domain Transfer → PenGym CRL                        | Phase 1→2→3  |
| $\theta_{scratch}$ | PenGym-only training from random initialization, same topology/tier schedule | Phase 3 only |
| $\theta_{sim}$     | Sim-trained agent evaluated on PenGym (zero-shot transfer)                   | Phase 1 only |

> **⚠️ Experimental Note (Transparency):** Scratch baseline implementation contained a bug — `train_pengym_scratch()` inherited `ewc_lambda=2000` from experiment config, causing EWC to penalize deviation from random initialization → SR=0% (artificial). One control run (finetune_only, λ=0) confirmed true scratch SR = 0.75–0.875. FT_SR cannot be reliably computed from current data. Paper reports dual agent absolute performance and documents this limitation.

### 6.3 Evaluation Metrics (C5)

#### 6.3.1 Primary Metrics

| Metric    | Formula                                                                                   | Purpose           | Overcomes                   |
| --------- | ----------------------------------------------------------------------------------------- | ----------------- | --------------------------- | ------------------------ | --------------- |
| SR (K=20) | $SR_s = \frac{1}{K}\sum_k \mathbb{1}[\text{success}_{s,k}]$                               | Success rate      | 1-episode binary limitation |
| NR        | $NR_s = R_{actual} / R_{optimal}$                                                         | Reward efficiency | Floor effect                |
| $\eta$    | $\eta_s = \text{optimal\_steps} / \text{actual\_steps} \times \mathbb{1}[\text{success}]$ | Step efficiency   | Ceiling effect              |
| SE        | $SE = \sqrt{SR(1-SR)/(                                                                    | \mathcal{S}       | \times K)}$                 | Statistical significance | Random variance |

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
  - SR = 100% on all 6 sim tasks (consistent across all seeds).
  - Normalized reward ≈ 0.99 (after UnifiedNormalizer).
  - EWC Fisher information accumulated across 6 tasks → transferred to Phase 2.
- **Purpose**: Establishes sim baseline. This confirms sim agent is well-trained before domain transfer.
- **Table**: Per-task sim results (SR, NR, $\eta$, episodes to convergence).
- **Note**: Sim performance is ceiling (SR=1.0), consistent with SCRIPT prior work.

### 7.3 Within-Topology Learning Results (RQ2, RQ4)

#### 7.3.1 Main Results Table — ⚠️ CẬP NHẬT VỚI MULTI-SEED DATA

> **CRITICAL WARNING:** Các kết quả dưới đây từ single seed (seed=42) ĐÃ ĐƯỢC THAY THẾ bởi multi-seed data.
> Scratch SR=0% là ARTIFACT của ewc_lambda bug. Cần fix và re-run.

**Multi-seed Intra-Topology (n=5):**

| Topology       | Agent                 | Mean SR         | Std       | Seeds                          |
| -------------- | --------------------- | --------------- | --------- | ------------------------------ |
| tiny (overall) | θ_dual                | 0.700           | 0.100     | [0.75, 0.50, 0.75, 0.75, 0.75] |
| tiny_T1        | θ_dual                | 0.800           | 0.400     | [1.0, 0.0, 1.0, 1.0, 1.0]      |
| tiny_T2        | θ_dual                | 1.000           | 0.000     | all 1.0                        |
| **tiny_T3**    | **θ_dual**            | **0.000**       | **0.000** | **all 0.0 (SCENARIO BUG)**     |
| tiny_T4        | θ_dual                | 1.000           | 0.000     | all 1.0                        |
| small-linear   | θ_dual                | 0.425           | 0.291     | [0.24, 1.0, 0.24, 0.35, 0.30]  |
| ALL            | θ_dual                | **0.562**       | **0.096** |                                |
| ALL            | θ_scratch             | **0.000**       | 0.000     | ⚠️ BUGGY (λ=2000 inherited)    |
| ALL            | θ_scratch (true, λ=0) | **0.750–0.875** | N/A       | from finetune_only run         |

**Multi-seed Cross-Topology (n=5):**

| Topology     | Agent  | Mean SR   | Std       |
| ------------ | ------ | --------- | --------- |
| tiny         | θ_dual | 0.622     | 0.207     |
| small-linear | θ_dual | 0.653     | 0.299     |
| ALL          | θ_dual | **0.637** | **0.123** |

#### 7.3.2 Transfer Metrics — ⚠️ CHƯA HỢP LỆ (scratch bug)

> FT_SR computed against buggy scratch (SR=0%). True scratch SR ≈ 0.75.
> Cần re-run sau khi fix `train_pengym_scratch()` để override `ewc_lambda=0`.

| Metric        | Value (vs buggy scratch) | Value (vs true scratch, estimated)    |
| ------------- | ------------------------ | ------------------------------------- |
| Intra FT_SR   | +0.562 ± 0.096           | **≈ -0.19** (0.562 - 0.75) ← NEGATIVE |
| Cross FT_SR   | +0.637 ± 0.123           | **≈ -0.11** (0.637 - 0.75) ← NEGATIVE |
| BT_SR (intra) | -0.767 ± 0.389           | unchanged                             |
| t-test FT>0   | p=0.0002                 | **VÔ HIỆU**                           |

#### 7.3.3 Per-Tier SR Breakdown (RQ4 — Curriculum effectiveness) — CẬP NHẬT

**Multi-seed Intra-Topology (n=5):**

| Task            | Mean SR   | Std       | Notes                             |
| --------------- | --------- | --------- | --------------------------------- |
| tiny_T1         | 0.800     | 0.400     | 1 seed fails (seed_1 = 0.0)       |
| tiny_T2         | 1.000     | 0.000     | ✅ Consistent                     |
| **tiny_T3**     | **0.000** | **0.000** | **❌ SCENARIO BUG (access=user)** |
| tiny_T4         | 1.000     | 0.000     | ✅ Consistent                     |
| small-linear_T1 | 0.500     | 0.277     | High variance                     |
| small-linear_T2 | 0.510     | 0.260     | High variance                     |
| small-linear_T3 | 0.330     | 0.336     | High variance                     |
| small-linear_T4 | 0.360     | 0.328     | High variance                     |

- **Key Finding**: Performance NOT uniform across tiers or seeds. small-linear results highly variable.
- **T3 BUG**: `compiled/tiny_T3_000.yml` has `access: user` (1st host) → compound probability 0.36–0.48 per host → unsolvable. Fixed in `compiled_tiny/` but not used in training scripts.

### 7.4 Cross-Topology Generalization (RQ5)

- **Viết gì**:
  - Stream-tiny agent evaluated on small-linear tasks → SR ≈ 0%.
  - Stream-small-linear agent evaluated on tiny tasks → SR ≈ 0%.
  - Sim zero-shot on PenGym: SR = 0.051 ± 0.067 (negligible).
  - **Confirmed negative result**: Cross-topology zero-shot transfer is near zero.
  - **Analysis**: Network structures differ fundamentally (3 vs 8 hosts, 1 vs 3 subnets, different action spaces). Flat RL policy encodes topology-specific routing — no compositional reasoning about network structure.
  - **Positive framing**: This negative result is an important empirical contribution, confirming that pentest RL requires topology-aware architectures (e.g., GNN-based) for cross-topology generalization.
- **Table**: Cross-topology generalization matrix.

### 7.5 Intra vs Cross-Topology CRL Comparison (RQ3)

**Multi-seed Statistical Comparison (n=5 each):**

| Metric          | Intra-Topology | Cross-Topology | t-test p       |
| --------------- | -------------- | -------------- | -------------- |
| Overall SR      | 0.562 ± 0.096  | 0.637 ± 0.123  | **0.365** (NS) |
| tiny SR         | 0.700 ± 0.100  | 0.622 ± 0.207  | 0.519 (NS)     |
| small-linear SR | 0.425 ± 0.291  | 0.653 ± 0.299  | 0.307 (NS)     |
| BT_SR           | -0.767 ± 0.389 | -0.367 ± 0.521 | —              |

**Interpretation:**

- No statistically significant difference between intra and cross-topology CRL in 2-topology setup (p=0.365).
- Cross-topology CRL is NOT harmful when both topologies are partially solvable.
- **Reconciliation with preliminary evidence**: EWC contamination (exp2: 3 topologies, medium SR=0%) appears triggered by including a FAILING topology — not by cross-topology training per se.
- **Revised finding**: Intra-topology CRL provides an isolation guarantee (sound architecture) but may not be necessary when all topologies are learnable. Cross-topology BT_SR = -0.367 (better than intra's -0.767) — an interesting signal worth further investigation.
- **Limitation**: Only 2 topologies tested, need wider evaluation.

### 7.6 Fisher Discount Ablation (RQ3)

- **Viết gì**: Ablation over β ∈ {0.0, 0.1, 0.3, 0.5, 0.7, 1.0} (single-seed, intra-topology):

| β   | Dual SR | Interpretation                         |
| --- | ------- | -------------------------------------- |
| 0.0 | 0.375   | Full sim Fisher → restricts plasticity |
| 0.1 | 0.456   | +0.081 vs β=0                          |
| 0.3 | 0.525   | +0.150 vs β=0 (default)                |
| 0.5 | 0.406   | Lower — non-monotonic                  |
| 0.7 | 0.525   | Same as β=0.3                          |
| 1.0 | 0.487   | Full discount (remove Fisher entirely) |

- **Key Finding**: Fisher discount is NECESSARY — β=0 worst at 0.375. β>0 consistently outperforms.
- **Nuance**: No clear optimal β; relationship non-monotonic. β ∈ [0.1, 0.7] all acceptable.
- **Insight**: Sim Fisher provides some useful signal but needs substantial reduction for cross-domain transfer.

### 7.7 Canonicalization Ablation

- Single-seed: Canon ON SR=0.525, Canon OFF SR=0.475 (Δ=0.05).
- Small positive effect. Canonicalization ensures cosine=1.0 (perfect alignment).
- **Limitation**: Single-seed, small Δ. SBERT already maps similar strings closely; canon most impactful for edge cases ("openssh"→"ssh").

### 7.8 CRL vs Finetune-Only Ablation

- Full CRL (EWC λ=2000): dual SR=0.525
- Finetune-only (λ=0): dual SR=0.362
- Δ=+0.163 → EWC helps within PenGym fine-tuning.
- **Critical caveat**: Finetune-only scratch (λ=0) achieves SR=0.875 (training) / 0.750 (eval) — the ONLY clean scratch measurement. All other scratch runs inherit λ=2000 → SR=0%.
- **Implication**: EWC helps dual agent (+0.163) but clean scratch (λ=0) learns BETTER (0.75-0.875 vs 0.525). Single-seed evidence, but suggests sim transfer may not provide net benefit — an important finding requiring further investigation.

### 7.9 Scalability Analysis — Medium Topology Failure

- Medium-single-site: SR=0% at 3K and 10K episodes. 10M action steps, zero success.
- Root cause: |A|=192 (flat action space) × 5+ multi-hop → exponential exploration space.
- Policy collapse: reward = -166.67 ± 0.0, no gradient signal.
- **Analysis**: Sparse reward problem — fundamental limitation of flat PPO on large discrete action spaces.
- **Implication**: Identifies clear scalability boundary. Larger topologies require reward shaping, hierarchical RL, or action masking.
- **Comparison with literature**: Similar to CyberBattleSim (Microsoft, 2021) findings — flat RL fails on >10 hosts.
- **Table**: Training statistics for medium-ss.

### 7.10 Step Efficiency Analysis

- tiny η = 74.9% — agent solves in ~8 steps vs optimal 6. Near-optimal.
- small-linear η = 3.5% — agent solves but takes 343 steps vs optimal 12. Very inefficient.
- **Interpretation**: On tiny (simple), scan→exploit pattern nearly sufficient. On small-linear (multi-subnet), agent lacks efficient multi-subnet routing knowledge — relies on extensive exploration.
- η differentiates quality even when SR is similar — confirms its utility as complementary metric.

---

## 8. Discussion (~2 pages)

### 8.1 Key Findings — Honest Synthesis

- **Viết gì**: Structured synthesis answering each RQ, distinguishing strong evidence from weak/negative:

**What Works (Strong Evidence):**

1. **Representation alignment** (RQ1): Unified 1540-dim SBERT encoding with canonicalization achieves cosine=1.0 across domains. This is the clearest positive contribution — it's deterministic, reproducible, and necessary for any cross-domain attempt.
2. **Fisher discount** (RQ3): β>0 improves SR by +0.15 over β=0. Sim Fisher information provides useful signal but needs 30-90% reduction for cross-domain use. Novel finding specific to cross-domain EWC.
3. **Framework and pipeline** (C2): 5-phase pipeline successfully enables CRL 5 pillars to operate on PenGym through adapter pattern, with 0 lines of core code modification. Reusable architecture contribution.

**What Partially Works (Mixed Evidence):** 4. **Intra-topology CRL** (RQ3/C3): Architecture is sound, but multi-seed data shows no significant advantage over cross-topology in 2-topology setup. Value is in isolation guarantee, not in measured performance difference. 5. **CVE curriculum** (RQ4/C4): 1985 CVEs graded, scenario generation works, but T3 bug prevents full evaluation. T1/T2/T4 results on tiny show some tier-dependent patterns.

**What Doesn't Work (Important Negative Findings):** 6. **Backward transfer** (RQ2): BT_SR = -0.767 — EWC catastrophically fails to preserve sim knowledge during cross-domain fine-tuning. Quadratic penalty insufficient when domain distributions differ fundamentally. 7. **Cross-topology generalization** (RQ5): ≈0% — policy is topology-specific. Flat RL without structural reasoning cannot generalize across network architectures. 8. **Forward transfer** (RQ2): Due to scratch bug, FT_SR cannot be computed reliably. One clean scratch measurement suggests true scratch may outperform dual agent, indicating sim knowledge may NOT provide net positive transfer — a critical open question.

### 8.2 Why EWC Fails Cross-Domain — Analysis

- **Viết gì**: Deep analysis of EWC's cross-domain failure:
  - EWC's quadratic penalty $\mathcal{L}_{EWC} = \sum_k \frac{\lambda}{2} F_k(\theta_k - \theta_k^*)^2$ assumes parameter importance (Fisher) from source domain is relevant to target domain.
  - In cross-domain CRL: sim Fisher encodes importance for 2064-CVE single-host tasks. PenGym requires multi-host network routing with 16 service-level actions → different parameters are important.
  - Fisher discount β partially addresses this (β>0 helps), but even with β, EWC constraints are "directionally wrong" — protecting parameters that matter for sim but may need to change for PenGym.
  - **BT_SR = -0.767 implication**: After PenGym training, agent has overwritten sim-relevant parameters despite EWC. Fisher discount reduces but doesn't prevent this.
  - **Comparison**: In within-domain CRL (Atari/MuJoCo), tasks share state/action semantics → Fisher IS relevant across tasks. Cross-domain violates this assumption.
  - **Insight for community**: EWC (and likely other regularization-based CRL methods) need fundamentally different approaches for cross-domain transfer — perhaps selective layer freezing, progressive networks, or domain-specific heads instead of uniform quadratic penalty.

### 8.3 The Scalability Boundary — Implications

- **Viết gì**:
  - Medium topology failure (SR=0%, |A|=192) identifies a clear scalability limit for flat PPO on sparse-reward pentest tasks.
  - Action space grows quadratically: $|A| = N_{hosts} \times N_{action\_types}$. tiny: 42, small-linear: 96, medium: 192.
  - Multi-hop penalty: 5+ sequential correct actions through 3 subnets. Random correct probability vanishingly small.
  - **Consistency with literature**: CyberBattleSim (Microsoft, 2021) reports similar failures for flat RL on >10 hosts.
  - **Future direction**: Reward shaping (partial credit for subnet discovery), hierarchical RL (options framework), action masking, GNN-based policy.
  - **Relevance beyond this paper**: This scalability boundary likely applies to ALL flat RL pentest agents, not just PenSCRIPT.

### 8.4 Experimental Limitations — Transparency

> **Phương châm**: Report limitations trung thực. Negative results + transparency = academic credibility.

| #   | Limitation                                  | Impact                                                         | Framing trong paper                                                                   |
| --- | ------------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| L1  | Scratch baseline bug (ewc_lambda=2000 leak) | FT_SR unreliable; one clean measurement suggests FT may be ≤ 0 | Report dual SR directly; note scratch bug in Limitations; compute FT vs clean scratch |
| L2  | T3 scenario bug (access=user)               | T3 SR=0% artifact, reduces overall SR ~12.5%                   | Report T3 separately; note bug; adjust overall SR interpretation                      |
| L3  | Only 2 solvable topologies                  | Limited generalization evidence                                | Acknowledge scope; frame as "empirical study on small-scale" → future: more topos     |
| L4  | PenGym real execution (KVM) untested        | Only NASim simulation mode                                     | Note as "cross-framework" not "sim-to-real"; KVM as future work                       |
| L5  | Tiny trivially easy (random SR=100%)        | Tiny results don't demonstrate transfer value                  | Focus analysis on small-linear; note tiny as "sanity check only"                      |
| L6  | small-linear high variance (std=0.291)      | Results unreliable single-seed; multi-seed CI wide             | Report with CI; acknowledge instability; frame as inherent difficulty variance        |
| L7  | BT_SR = -0.767 (catastrophic forgetting)    | EWC fails to preserve sim knowledge                            | Report honestly as negative finding; analyze WHY (§8.2); future: better CRL methods   |
| L8  | Single-seed ablations                       | β, canon, CRL ablations each n=1                               | Frame as "preliminary"; identify trends but don't overclaim significance              |

### 8.5 Threats to Validity

- **Internal validity**:
  - NASim class-level state leakage → mitigated by fresh adapter per agent.
  - T3 scenario bug → discovered, documented, and factored into analysis.
  - Scratch baseline bug → documented transparently; FT_SR reported with caveat.
  - Multi-seed (n=5): Provides statistical power for main results but ablations remain single-seed.
- **External validity**:
  - Results on NASim simulation mode, not real networks → PenGym KVM mode untested.
  - Only 2 solvable topologies assessed out of 3 → generalizability limited.
  - SCRIPT's 5-pillar CRL is specific framework; results may not generalize to other CRL approaches (PackNet, Progress&Compress).
- **Construct validity**:
  - Multi-episode SR (K=20) provides continuous metric → overcomes binary limitation.
  - η directly measures step quality → immune to ceiling effect. Useful complementary metric.
  - FT_SR requires valid scratch baseline → currently compromised. Paper acknowledges and adjusts.

---

## 9. Conclusion & Future Work (~0.5 page)

### 9.1 Conclusion

- **Viết gì**: Restate problem, approach, main results — honest and balanced.
  - We presented **PenSCRIPT**, a cross-domain CRL framework for automated pentesting that bridges SCRIPT simulation and PenGym environment through unified representation, a 5-phase pipeline, CVE-based curriculum, and multi-dimensional evaluation.
  - **What we achieved**:
    - Unified 1540-dim SBERT encoding achieves perfect cross-domain alignment (cosine=1.0), enabling a single policy architecture to operate across domains with 0 core code modification.
    - Dual agent achieves SR=0.562–0.637 on PenGym topologies (n=5 seeds per configuration), demonstrating the pipeline is functional.
    - Fisher discount (β>0) improves cross-domain CRL by +0.15 SR — a novel finding for EWC calibration across domains.
    - CVE curriculum (1985 CVEs, 4 tiers, 96+ scenarios) and evaluation framework (23 metrics) provide reusable resources for the community.
  - **What we discovered doesn't work (equally important contributions)**:
    - EWC catastrophically fails backward transfer (BT_SR = -0.767) — quadratic penalty insufficient for cross-domain knowledge preservation. New CRL approaches needed.
    - Cross-topology generalization ≈ 0% — flat RL policy is inherently topology-specific. Requires structural reasoning (GNN, hierarchical RL).
    - Flat PPO fails on topologies with |A|>100 and sparse rewards (medium SR=0%). Scalability requires reward shaping or hierarchical decomposition.
  - These findings establish the first empirical baseline for cross-domain CRL in network security, providing both a reusable framework and honest lessons about what works and what doesn't.

### 9.2 Future Work

| Priority   | Direction                                                  | Rationale                                                                                                       |
| ---------- | ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| **High**   | **Fix scratch baseline + re-evaluate FT_SR**               | Resolve ewc_lambda bug; establish clean forward transfer measurement. Critical for validating transfer value.   |
| **High**   | **Better CRL methods (PackNet, Progressive Nets)**         | EWC's quadratic penalty fails cross-domain. Methods with dedicated capacity per domain may preserve BT.         |
| **High**   | **Reward shaping / hierarchical RL for larger topologies** | Medium SR=0% due to sparse reward. Sub-goal decomposition, option framework, or curiosity-driven exploration.   |
| **High**   | **PenGym KVM execution verification**                      | Pipeline tested on NASim sim mode only. Real execution verification is necessary to validate practical utility. |
| **Medium** | **Graph Neural Networks for topology-aware policy**        | Zero cross-topology generalization → need structural graph encoding for topology-agnostic features.             |
| **Medium** | **Action masking for invalid actions**                     | Reduce effective action space by removing unreachable host actions.                                             |
| **Medium** | **Multi-seed ablations (n=5) for β, canon, CRL**           | Current ablations single-seed. Statistical power insufficient for definitive conclusions.                       |
| **Low**    | **Diverse topology set (5+ solvable topologies)**          | Only 2 solvable topologies provide limited evidence base. Need wider evaluation.                                |

---

## 10. Figures & Tables Summary

### Figures (Proposed)

| #      | Content                                                  | Type                        | Data Source                                   |
| ------ | -------------------------------------------------------- | --------------------------- | --------------------------------------------- |
| Fig. 1 | System architecture (SCRIPT ↔ Adapter ↔ PenGym)          | Architecture diagram        | `project_fusion_summary.md §III.2`            |
| Fig. 2 | 5-Phase pipeline (Phase 0→4 flow)                        | Flow diagram                | `strategy_C_shared_state_dual_training.md §6` |
| Fig. 3 | Cross-topology EWC interaction (conditional finding)     | Diagram                     | `intra_topology_crl_blueprint.md Appendix A`  |
| Fig. 4 | CVE difficulty distribution and 4-tier grading           | Histogram + threshold lines | `cve_difficulty_and_expansion.md §3`          |
| Fig. 5 | Template + Overlay scenario generation                   | Schematic                   | `cve_difficulty_and_expansion.md §4`          |
| Fig. 6 | SR per tier (T1–T4) per topology                         | Bar chart                   | Experimental results                          |
| Fig. 7 | Step efficiency ($\eta$) comparison across topologies    | Bar chart                   | Experimental results                          |
| Fig. 8 | Learning curves (reward vs episodes) for dual vs scratch | Line chart                  | TensorBoard logs                              |
| Fig. 9 | Scalability: SR vs topology size (hosts)                 | Scatter/line                | Experimental results                          |

### Tables (Proposed)

| #        | Content                                                        | Data Source                                  |
| -------- | -------------------------------------------------------------- | -------------------------------------------- |
| Table 1  | State representation comparison (SCRIPT vs PenGym vs Unified)  | `script_pengym_integration_summary.md §I.2`  |
| Table 2  | 16-dim action space with PenGym mapping and CVE group sizes    | `script_pengym_integration_summary.md §II.3` |
| Table 3  | Phase 1 sim baseline results (per-task SR, NR, $\eta$)         | Experimental outputs                         |
| Table 4  | Main results: per-topology per-agent multi-seed SR (n=5)       | Experimental outputs                         |
| Table 5  | Transfer metrics: BT_SR, estimated FT_SR with caveats          | Experimental outputs                         |
| Table 6  | Per-tier SR breakdown (T1–T4) per topology (incl. T3 bug note) | Experimental outputs                         |
| Table 7  | Fisher discount ablation (β=0 to 1.0)                          | Ablation results                             |
| Table 8  | Intra vs Cross-topology CRL comparison (n=5, t-test)           | Multi-seed results                           |
| Table 9  | Medium-ss failure analysis (episodes, steps, SR, reward)       | `outputs/medium_ss_10k/`                     |
| Table 10 | Related work positioning comparison                            | Literature review                            |
| Table 11 | Hyperparameters                                                | `config.py`, `curriculum_episodes.json`      |
| Table 12 | CVE difficulty tier definitions                                | `cve_difficulty_and_expansion.md §3`         |

---

## 11. Venue Recommendations

### 11.1 Target Venues (Ranked by Fit)

| #   | Venue                                                             | Type       | Fit Reason                                              | Deadline Cycle             |
| --- | ----------------------------------------------------------------- | ---------- | ------------------------------------------------------- | -------------------------- |
| 1   | **ACSAC** (Annual Computer Security Applications Conference)      | Conference | Applied security + RL, good fit for system papers       | June/July                  |
| 2   | **AISec** (Workshop on AI Security, co-located with CCS)          | Workshop   | Directly targets AI for security applications           | July/Aug                   |
| 3   | **RAID** (Recent Advances in Intrusion Detection)                 | Symposium  | Network security, attack simulation                     | March/April                |
| 4   | **ESORICS** (European Symposium on Research in Computer Security) | Conference | Security research, accepts system papers                | April                      |
| 5   | **IEEE S&P Workshop on Deep Learning and Security** (DLS)         | Workshop   | DL/RL for security                                      | Feb/March                  |
| 6   | **AAAI** / **IJCAI** (AI track)                                   | Conference | If framed as CRL contribution with security application | Sept (AAAI), Jan (IJCAI)   |
| 7   | **NeurIPS** / **ICML** (RL workshop)                              | Workshop   | If focused on CRL methodology contribution              | May (ICML), June (NeurIPS) |
| 8   | **Computers & Security** (Elsevier)                               | Journal    | Comprehensive system paper, more space for details      | Rolling                    |
| 9   | **IEEE TIFS** (Trans. on Information Forensics and Security)      | Journal    | Top-tier security journal, if results are strong enough | Rolling                    |

### 11.2 Recommendation

**Primary target**: **ACSAC** hoặc **AISec Workshop @ CCS** — fit tốt nhất cho paper kết hợp framework contribution + empirical study trong applied security. ACSAC (12-18 pages) cho full paper; AISec (6-8 pages) nếu condense.

**Paper strength with current data**: Framework + methodology contribution + honest negative results. Venues valuing "empirical study" và "lessons learned" sẽ đánh giá cao tính transparent. Tránh venues yêu cầu SOTA results (NeurIPS/ICML main track) vì SR và FT_SR không competitive.

**Secondary**: **RAID** hoặc **Computers & Security** (journal) — journal format cho phép trình bày chi tiết hơn limitations + discussion, phù hợp với empirical study framing.

**Secondary**: **AISec Workshop @ CCS** — shorter paper (8 pages), faster turnaround, focused audience.

**Nếu muốn emphasize CRL methodology**: **AAAI/IJCAI** main track hoặc **NeurIPS CRL workshop** — requires framing CRL contribution more prominently (intra-topology CRL, cross-domain EWC with Fisher discount).

### 11.3 Paper Length Adaptation

| Venue                                  | Length      | Strategy                                                                |
| -------------------------------------- | ----------- | ----------------------------------------------------------------------- |
| Full conference (ACSAC, RAID, ESORICS) | 12–18 pages | Full paper as outlined                                                  |
| Workshop (AISec, DLS)                  | 6–8 pages   | Focus on C1+C3 (representation + intra-topology), condense C4+C5        |
| Journal (Computers & Security, TIFS)   | 20+ pages   | Expand all sections, add appendices, multi-seed                         |
| Short/poster (NeurIPS workshop)        | 4 pages     | Focus solely on representation alignment (C1) + Fisher discount finding |

---

## 12. Writing Strategy

### Key Writing Principles

1. **Lead with methodology contributions (C1-C5)** — these are valid regardless of transfer results.
2. **Report results honestly** — dual agent SR, ablation trends, negative findings. Avoid computing FT_SR against buggy scratch.
3. **Frame negative results as contributions** — BT failure, zero cross-topology generalization, scalability boundary are all valuable empirical findings.
4. **Distinguish strong vs weak evidence** — cosine=1.0 is deterministic. β>0 trend is consistent. single-seed ablations are preliminary.
5. **Limitations section is a STRENGTH** — transparent reporting builds credibility. Describe bugs, their impact, and what can/cannot be concluded.

### Section Priority for Writing

| Priority | Section                | Rationale                                       |
| -------- | ---------------------- | ----------------------------------------------- |
| P0       | §5 Methodology         | Framework contribution — most space, most value |
| P0       | §7 Results             | Data exists; needs careful honest narrative     |
| P0       | §8 Discussion          | Strategic — turns limitations into insights     |
| P0       | §3 Introduction        | Framing sets reader expectations correctly      |
| P1       | §2 Abstract            | Write LAST — distill the honest version         |
| P1       | §4 Related Work        | Position against existing work                  |
| P1       | §6 Experimental Design | Tables from existing data                       |
| P2       | §9 Conclusion          | Summarize                                       |
| P2       | Figures & Tables       | Generate from data                              |

---

## Appendix A — Key Data Points for Paper (Final — 2026-03-01)

### A.1 Experimental Results Summary

```
=== MULTI-SEED MAIN RESULTS ===

INTRA-TOPOLOGY (n=5 seeds: 0,1,2,3,4):
  Overall SR: 0.562 ± 0.096
    tiny:         0.700 ± 0.100  (T2=T4=1.0, T1=0.8, T3=0.0 [scenario bug])
    small-linear: 0.425 ± 0.291  (high variance — seed 1 outlier at 1.0)
  BT_SR: -0.767 ± 0.389  (catastrophic sim forgetting)
  Step eff: 0.310 ± 0.028
  Phase 1 (sim): SR=1.000 ± 0.000

CROSS-TOPOLOGY (n=5 seeds: 0,1,2,3,4):
  Overall SR: 0.637 ± 0.123  (HIGHER than intra, p=0.365 NS)
    tiny:         0.622 ± 0.207
    small-linear: 0.653 ± 0.299
  BT_SR: -0.367 ± 0.521

STATISTICAL COMPARISON:
  Intra vs Cross: t=-0.961, p=0.365 → NO significant difference

=== ABLATION RESULTS (single-seed, seed=42) ===

Fisher Discount (β):
  β=0.0: SR=0.375 | β=0.1: 0.456 | β=0.3: 0.525
  β=0.5: SR=0.406 | β=0.7: 0.525 | β=1.0: 0.487
  → β>0 helps (+0.15 over β=0). Non-monotonic. β=0.3 ≈ β=0.7.

Canonicalization:
  Canon ON: SR=0.525 | Canon OFF: SR=0.475 | Δ=0.050

CRL (EWC) vs Finetune-only:
  Full CRL (λ=2000): dual SR=0.525 | Finetune-only (λ=0): dual SR=0.362
  → EWC helps dual agent (+0.163)
  ⚠️ Finetune-only scratch (λ=0): SR=0.875 (train) / 0.750 (eval)
  [This is the ONLY clean scratch measurement]

=== BENCHMARK BASELINES (tiny only) ===
  Random: SR=1.0 (38.9 steps) | Greedy: SR=1.0 (28.1)
  Scan-First: SR=1.0 (31.7)   | DQN: SR=1.0 (20.5) | A2C: SR=1.0 (28.1)
  → tiny trivially easy — ALL agents achieve 100%

=== TOTAL EXPERIMENT RUNS: 17 ===
  5 intra-topology multi-seed
  5 cross-topology multi-seed
  6 β-ablation (single-seed)
  1 canonicalization ablation (single-seed)
  + benchmark baselines
```

### A.2 Known Experimental Issues (Documented for Transparency)

```
ISSUE 1 — Scratch baseline ewc_lambda leak:
  Location: src/training/dual_trainer.py, train_pengym_scratch()
  Cause: cl_config=copy.deepcopy(self.script_config) → inherits ewc_lambda=2000
  Effect: EWC penalizes deviation from random init → scratch SR=0% (artificial)
  Evidence: finetune_only (λ=0) scratch SR = 0.75-0.875 (true scratch)
  Impact on paper: FT_SR cannot be reliably computed. Paper reports dual SR directly.
  Status: NOT FIXED (no time for re-runs). Documented as limitation.

ISSUE 2 — T3 scenario configuration:
  Location: data/scenarios/generated/compiled/tiny_T3_000.yml line 75
  Cause: First host has access: user (should be root for consistency)
  Effect: T3 SR=0% across ALL 10 multi-seed runs
  Impact on paper: Reduces overall SR ~12.5%. T3 results excluded from curriculum analysis.
  Status: NOT FIXED in training pipeline. Documented as limitation.
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
  seeds = [0, 1, 2, 3, 4] for main experiments
  seed = 42 for ablations (single-seed)
```

### A.5 Evidence Strength Summary

| Claim                                      | Evidence Level    | Basis                          |
| ------------------------------------------ | ----------------- | ------------------------------ |
| Cosine = 1.0 (representation alignment)    | **Strong**        | Deterministic, reproducible    |
| β>0 improves SR                            | **Moderate**      | 6-point ablation, single-seed  |
| Dual agent learns PenGym tasks             | **Strong**        | n=5 seeds × 2 configs          |
| EWC fails backward transfer                | **Strong**        | n=5, BT_SR=-0.767±0.389        |
| Cross-topology generalization ≈ 0%         | **Strong**        | Confirmed across all configs   |
| Canonicalization helps (Δ=0.05)            | **Weak**          | Single-seed, small effect      |
| CRL (EWC) helps within PenGym (+0.163)     | **Weak**          | Single-seed ablation           |
| Forward transfer positive                  | **Unverifiable**  | Scratch bug invalidates        |
| Death spiral in 2-topology setup           | **Not supported** | p=0.365, cross ≥ intra         |
| CVE curriculum creates difficulty gradient | **Inconclusive**  | T3 bug, limited tier variation |
