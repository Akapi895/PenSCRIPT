# PenSCRIPT — Experiment Roadmap

> **Ngày tạo:** 2026-02-27  
> **Mục tiêu:** Roadmap thực nghiệm có hệ thống để hoàn thiện bài báo, dựa trên phân tích trong `analyze.md`  
> **Nguyên tắc:** Mỗi task đều có mục tiêu rõ ràng, metric đo được, và liên kết trực tiếp đến weakness/gap được xác định trong review

---

## Tổng Quan Vấn Đề (từ analyze.md)

| ID  | Vấn đề                                                           | Mức nghiêm trọng | Loại        |
| --- | ---------------------------------------------------------------- | ---------------- | ----------- |
| W1  | Single seed (seed=42) — không có statistical confidence          | **Fatal**        | Thực nghiệm |
| W2  | Không có ablation study thực sự                                  | Cao              | Thực nghiệm |
| W3  | Baseline quá yếu (θ_scratch SR=0% không chứng minh được gì)      | Cao              | Thực nghiệm |
| W4  | small-linear η=3.5% mâu thuẫn với transfer claim                 | Trung bình       | Phân tích   |
| L1  | Không có real execution (chỉ NASim sim mode)                     | Trung bình       | Thực nghiệm |
| L2  | Chỉ 2 topology solvable (quá ít)                                 | Trung bình       | Thực nghiệm |
| L3  | Backward transfer chưa đo đầy đủ                                 | Trung bình       | Thực nghiệm |
| F1  | Framing "sim-to-real" sai — chỉ là sim-to-sim                    | Cao              | Framing     |
| F2  | Hypothesis đòi p<0.05 nhưng single seed không compute được       | Cao              | Framing     |
| F3  | Premise ban đầu (HIB, LLM domain rand) không khớp implementation | Cao              | Framing     |
| F4  | Fisher discount β=0.3 không có justification lý thuyết           | Thấp             | Phương pháp |

---

## P0 — Bắt Buộc (Blocking — Không có = không submit được)

### P0.1 — Multi-Seed Experiments (n=5)

**Giải quyết:** W1, F2  
**Vì sao:** Single seed = reviewer reject ngay. Cần distribution để tính mean, std, confidence interval. Hypothesis H yêu cầu p<0.05 → cần ≥5 samples.

**Việc cần làm:**

1. **Cập nhật code: Thêm multi-seed runner**
   - Thêm argument `--seeds 0,1,2,3,42` vào `run_strategy_c.py`
   - Tạo loop chạy full pipeline (Phase 0→4) cho mỗi seed
   - Output riêng per-seed: `outputs/multiseed/seed_{s}/`
   - Aggregation script: đọc tất cả `strategy_c_results.json`, tính mean ± std cho SR, NR, η, FT, BT

2. **Chạy thực nghiệm:**
   - Seeds: `[0, 1, 2, 3, 42]` (giữ seed=42 để backward-compatible)
   - Topologies: tiny (T1–T4), small-linear (T1–T4)
   - Agents: θ_dual + θ_scratch cho mỗi seed
   - **Estimated compute:** ~5 × thời gian 1 run hiện tại

3. **Metric cần báo cáo:**
   - SR: mean ± std across 5 seeds, per topology
   - η: mean ± std across 5 seeds
   - FT*SR, FT*η: mean ± std
   - p-value (paired t-test hoặc Wilcoxon): θ_dual vs θ_scratch trên SR
   - 95% confidence interval cho SR

4. **Tiêu chí hoàn thành:**
   - [ ] Code multi-seed runner hoạt động
   - [ ] 5 seeds × 2 topologies × 2 agents = 20 experiment runs hoàn thành
   - [ ] Aggregation table có mean ± std + p-value
   - [ ] SR(θ_dual) significantly > SR(θ_scratch) với p < 0.05

---

### P0.2 — Baseline Nâng Cấp

**Giải quyết:** W3  
**Vì sao:** θ_scratch SR=0% trên mọi topology là baseline quá yếu. Reviewer sẽ hỏi: "PPO from scratch cũng fail sao? Hay chỉ vì episode budget quá thấp?"

**Việc cần làm:**

1. **Baseline A — Extended θ_scratch (budget lớn hơn):**
   - Chạy PPO from scratch với 3× episode budget hiện tại (tiny: 1500 eps, small-linear: 2250 eps)
   - Nếu vẫn SR=0%: chứng minh transfer là necessary, không phải budget issue
   - Nếu SR>0%: cần report honest, so sánh η với θ_dual

2. **Baseline B — PPO from scratch + reward shaping:**
   - Thêm intermediate reward cho scan thành công (+0.1), subnet discovery (+0.2)
   - Giữ topology/episode budget giống θ_dual
   - Mục tiêu: isolate transfer benefit vs reward engineering benefit
   - _Lưu ý: reward shaping cần implement thêm trong wrapper_

3. **Baseline C — Fine-tune only (no CRL pillars):**
   - Transfer θ_sim → PenGym, fine-tune bằng vanilla PPO (tắt EWC, retrospection, KL imitation)
   - Sử dụng config: `script-no__retention.yaml` (đã có sẵn)
   - Mục tiêu: isolate CRL contribution vs simple fine-tuning

4. **Baseline D — Sử dụng existing baselines (Random, Greedy, DQN):**
   - Đã implement trong `run_benchmark.py`
   - Chạy trên cùng scenario set (tiny T1–T4, small-linear T1–T4)
   - Report SR, NR, η cho consistency

5. **Tiêu chí hoàn thành:**
   - [ ] ≥3 baselines có SR > 0% HOẶC documented evidence rằng task khó đến mức chỉ transfer mới giải được
   - [ ] Bảng so sánh: θ_dual vs θ_scratch vs θ_extended_scratch vs θ_finetune_only vs Random vs Greedy vs DQN
   - [ ] Mỗi baseline chạy với 5 seeds

---

### P0.3 — Intra vs Cross-Topology Controlled Comparison

**Giải quyết:** W2 (ablation), RQ3  
**Vì sao:** exp2 (cross-topology CRL) vs final (intra-topology CRL) là core contribution C3, nhưng data hiện tại từ hai experiment khác nhau, không phải controlled comparison.

**Việc cần làm:**

1. **Chạy lại cross-topology CRL** với cùng:
   - Hyperparameters giống intra-topology run
   - Episode budget giống nhau
   - 5 seeds
   - Training sequence: tiny→small-linear→medium (sequential)

2. **Chạy lại intra-topology CRL** với:
   - Cùng hyperparameters, cùng 5 seeds
   - Per-topology independent streams

3. **Metric so sánh:**
   - Per-topology SR: intra vs cross, per seed
   - Forgetting matrix F[i][j]: intra vs cross
   - Training time comparison

4. **Tiêu chí hoàn thành:**
   - [ ] Cross-topology CRL chạy xong 5 seeds
   - [ ] Intra-topology CRL chạy xong 5 seeds (có thể reuse từ P0.1)
   - [ ] Statistical comparison (paired t-test) cho tiny SR: intra vs cross
   - [ ] Forgetting matrix hiển thị death spiral rõ ràng trong cross-topology

---

## P1 — Quan Trọng (Tăng chất lượng đáng kể — nên làm trước khi submit)

### P1.1 — Canonicalization Ablation (C1 Validation)

**Giải quyết:** W2 (ablation cho C1)  
**Vì sao:** C1 (canonicalization) được claim là đóng góp, nhưng chưa có evidence rằng nó thực sự matters. Cần so sánh có/không canonicalization.

**Việc cần làm:**

1. **Thêm flag `--no-canonicalization`** vào unified state encoder
   - Khi bật: skip `CANONICAL_MAP`, gửi raw entity names trực tiếp vào SBERT
   - Cần sửa `unified_state_encoder.py`

2. **Chạy 3 conditions:**

   | Condition                 | Canonicalization | SBERT             | Kỳ vọng                             |
   | ------------------------- | ---------------- | ----------------- | ----------------------------------- |
   | A — No canon              | ❌               | off-the-shelf     | cosine < 1.0, SR có thể thấp hơn    |
   | B — With canon (hiện tại) | ✅               | off-the-shelf     | cosine = 1.0, SR = 100%             |
   | C — Fine-tuned SBERT      | ❌               | domain fine-tuned | cosine cao hơn A, SR tương đương B? |

3. **Metric:**
   - Cross-domain cosine similarity: per entity (OS, service, port)
   - SR trên tiny, small-linear sau full pipeline
   - Training convergence speed (episodes to SR>80%)

4. **Tiêu chí hoàn thành:**
   - [ ] 3 conditions chạy xong (ít nhất seed=42, tốt nhất 5 seeds)
   - [ ] Cosine table: Condition A vs B vs C
   - [ ] SR table: Condition A vs B vs C
   - [ ] Evidence rằng canonicalization matters (hoặc evidence rằng nó không matter — cũng là finding có giá trị)

---

### P1.2 — Fisher Discount (β) Ablation (C2 Validation)

**Giải quyết:** W2 (ablation cho C2), F4  
**Vì sao:** β=0.3 được chọn mà không có justification. Cần evidence cho giá trị tối ưu.

**Việc cần làm:**

1. **Chạy domain transfer với các β khác nhau:**

   | β   | Ý nghĩa                                           |
   | --- | ------------------------------------------------- |
   | 0.0 | Xóa hoàn toàn Fisher → no EWC constraint from sim |
   | 0.1 | Giữ rất ít sim knowledge                          |
   | 0.3 | Hiện tại (giữ 30% sim EWC penalty)                |
   | 0.5 | Balanced                                          |
   | 1.0 | Giữ nguyên sim Fisher (no discount)               |

2. **Đo trên tiny và small-linear:**
   - SR, η per β
   - Backward Transfer per β (BT_SR trên sim tasks)

3. **Tiêu chí hoàn thành:**
   - [ ] 5 giá trị β × 2 topologies chạy xong (seed=42 minimum)
   - [ ] Line chart: SR vs β, η vs β
   - [ ] Xác định β\* optimal (hoặc plateau range)
   - [ ] Interpret: β=0.3 nằm ở đâu so với optimal?

---

### P1.3 — Step Limit Sensitivity Test (η Analysis)

**Giải quyết:** W4  
**Vì sao:** small-linear η=3.5% (343 steps vs optimal 12) gây nghi ngờ transfer quality. Reviewer sẽ hỏi: "Agent đang random walk + may mắn?"

**Việc cần làm:**

1. **Chạy θ_dual và θ_scratch trên small-linear với step limits khác nhau:**

   | Step limit | Mục tiêu                                    |
   | ---------- | ------------------------------------------- |
   | 50         | Chỉ ~4× optimal → buộc agent phải efficient |
   | 100        | ~8× optimal                                 |
   | 200        | ~17× optimal                                |
   | 500        | Hiện tại                                    |

2. **Nếu θ_dual SR>0% ở limit=100 mà θ_scratch SR=0%:** Transfer benefit là genuine
3. **Nếu θ_dual cũng SR=0% ở limit=100:** Agent thực sự random walk, cần giải thích rõ

4. **Tiêu chí hoàn thành:**
   - [ ] SR vs step_limit table cho θ_dual và θ_scratch
   - [ ] Xác định step_limit threshold nơi θ_dual bắt đầu fail
   - [ ] Narrative giải thích η thấp dựa trên evidence

---

### P1.4 — Backward Transfer Evaluation (Đo đầy đủ)

**Giải quyết:** L3  
**Vì sao:** BT được define trong evaluation framework nhưng chưa có đủ data. Cần đo θ_dual trên sim tasks sau PenGym fine-tuning.

**Việc cần làm:**

1. **Sau khi có θ_dual (từ P0.1):**
   - Load θ_dual (mỗi seed)
   - Evaluate trên 6 sim tasks (SCRIPT scenarios gốc)
   - Tính BT_SR = SR(θ_dual, sim) - SR(θ_sim, sim)
   - Tính BT_η, BT_NR

2. **So sánh θ_sim_unified vs θ_dual trên sim:**
   - Nếu BT ≈ 0: PenGym fine-tuning không phá sim knowledge → tốt
   - Nếu BT < 0: Có forgetting → cần report + analyze

3. **Tiêu chí hoàn thành:**
   - [ ] BT table: per-sim-task, per-seed
   - [ ] Mean BT ± std
   - [ ] Confirm BT_η ≥ -0.15 (hypothesis requirement)

---

## P2 — Nên Làm (Tăng khả năng được accept ở venue tốt hơn)

### P2.1 — Thêm Topology (medium với reward shaping)

**Giải quyết:** L2  
**Vì sao:** Chỉ 2 topology solvable quá ít. Nếu medium solve được với reward shaping → 3 solvable topologies, tăng evidence đáng kể.

**Việc cần làm:**

1. **Implement reward shaping cho PenGym:**
   - Trong `reward_normalizer.py` hoặc `single_host_wrapper.py`
   - Intermediate rewards:
     - +0.05 cho scan phát hiện host mới
     - +0.1 cho scan phát hiện service mới
     - +0.2 cho exploit thành công trên non-sensitive host
     - +1.0 cho goal host compromise (giữ nguyên)
   - Tạo flag `--reward-shaping` để bật/tắt

2. **Chạy θ_dual trên medium-single-site với reward shaping:**
   - Episode budget: 5000–10000
   - Step limit: 1000
   - 5 seeds

3. **Tiêu chí hoàn thành:**
   - [ ] Reward shaping implemented + tested
   - [ ] medium SR > 0% với reward shaping (target: >50%)
   - [ ] So sánh: medium SR with/without reward shaping

---

### P2.2 — CRL Method Comparison (PackNet, Fine-tune)

**Giải quyết:** Reviewer concern: "Tại sao chỉ EWC? So sánh với CRL methods khác?"

**Việc cần làm:**

1. **Implement hoặc tìm library cho:**
   - **PackNet** (network pruning-based CRL)
   - **Progress & Compress** (knowledge distillation-based CRL)
   - **Vanilla fine-tune** (no CRL at all — đã có config `script-no__retention.yaml`)

2. **Chạy trên tiny intra-topology CRL (T1→T4):**
   - So sánh EWC vs PackNet vs P&C vs fine-tune
   - Metric: SR per tier, forgetting F[i][j]

3. **Tiêu chí hoàn thành:**
   - [ ] ≥2 CRL methods so sánh
   - [ ] Bảng so sánh: per-method SR, forgetting, memory overhead

---

### P2.3 — Real Execution Partial Verification (CyRIS KVM)

**Giải quyết:** L1, F1  
**Vì sao:** Paper claim "bridging sim and realistic environments" nhưng chỉ test NASim sim mode. Dù chỉ partial verification cũng tăng credibility rất nhiều.

**Việc cần làm:**

1. **Setup CyRIS/KVM** cho tiny topology (3 hosts)
2. **Chạy θ_dual trên real PenGym** (KVM mode) — chỉ eval, không train
3. **So sánh output:**
   - Real exec SR vs sim SR
   - State observation: real vs sim → cosine similarity
   - Action latency, failure modes

4. **Nếu không có CyRIS access:**
   - **Fallback:** Explicitly acknowledge trong Discussion section
   - Dùng "cross-framework transfer" thay "sim-to-real" trong title/abstract
   - Report gap analysis giữa NASim sim và expected KVM behavior

5. **Tiêu chí hoàn thành:**
   - [ ] CyRIS setup thành công HOẶC documented attempt + fallback framing
   - [ ] Nếu thành công: real vs sim comparison table

---

### P2.4 — SCRIPT CRL Pillar Ablation (Existing Configs)

**Giải quyết:** Tăng depth cho methodology section  
**Vì sao:** Đã có 6 YAML configs cho ablation (no_guide, no_imitation, no_res, no_reset, no_wic, no_retention) nhưng chưa chạy trên PenGym.

**Việc cần làm:**

1. **Tạo ablation runner script:**
   - Loop qua 6 configs
   - Chạy từng config trên tiny (T1–T4) với intra-topology CRL
   - Collect SR, η, forgetting per config

2. **Output:** Ablation table showing contribution of each CRL pillar

3. **Tiêu chí hoàn thành:**
   - [ ] 6 ablation configs chạy xong (seed=42 minimum)
   - [ ] Bảng: per-pillar contribution to SR

---

## P3 — Nice-to-Have (Chỉ làm nếu có thời gian)

### P3.1 — Off-the-shelf SBERT vs Fine-tuned SBERT

- Chạy pipeline với SBERT fine-tuned trên pentest corpus (như SCRIPT gốc)
- So sánh SR với off-the-shelf SBERT
- Mục tiêu: trả lời "domain adaptation của encoder có cần thiết không?"

### P3.2 — Cross-Topology Generalization with GNN

- Prototype Graph Neural Network policy thay flat MLP
- Encode topology structure → topology-agnostic features
- Test zero-shot cross-topology transfer
- _Lưu ý: Scope lớn, chỉ nên làm nếu có ≥2 tuần_

### P3.3 — Learning Curve Visualization

- Plot reward/SR vs episodes cho θ_dual vs θ_scratch vs các baselines
- TensorBoard logs đã có → chỉ cần extract và plot
- Tạo Figure 8 trong paper outline

### P3.4 — Action Distribution Analysis

- Phân tích action distribution của θ_dual vs θ_scratch
- Nếu θ_dual có action distribution khác biệt rõ (concentrated vs uniform) → evidence cho transfer
- Sử dụng policy-level metrics đã implement (D_KL, Δ_F)

---

## Code Changes Required (Summary)

### Cần implement mới:

| File                                     | Thay đổi                                                       | Priority |
| ---------------------------------------- | -------------------------------------------------------------- | -------- |
| `run_strategy_c.py`                      | Thêm `--seeds` argument, multi-seed loop, per-seed output dirs | P0       |
| `src/evaluation/aggregate_seeds.py`      | **Tạo mới** — đọc per-seed results, tính mean±std, p-value, CI | P0       |
| `src/envs/core/unified_state_encoder.py` | Thêm `--no-canonicalization` flag                              | P1       |
| `src/training/domain_transfer.py`        | Thêm configurable β từ CLI                                     | P1       |
| `src/envs/wrappers/reward_normalizer.py` | Thêm reward shaping mode                                       | P2       |
| `scripts/run_ablation_suite.py`          | **Tạo mới** — loop qua ablation configs, aggregate             | P2       |

### Cần sửa nhỏ:

| File                                | Thay đổi                                                           | Priority |
| ----------------------------------- | ------------------------------------------------------------------ | -------- |
| `src/evaluation/strategy_c_eval.py` | Thêm step_limit param cho sensitivity test                         | P1       |
| `run_benchmark.py`                  | Chạy existing baselines (Random, Greedy, DQN) trên T1–T4 scenarios | P0       |

---

## Timeline Đề Xuất

### Tuần 1: P0 (Critical)

| Ngày | Task                                                   | Output                        |
| ---- | ------------------------------------------------------ | ----------------------------- |
| 1–2  | Implement multi-seed runner + start experiments        | Code ready, seeds 0,1 running |
| 3    | Chạy tiếp seeds 2,3,42 + extended scratch baseline     | 5 seeds hoàn thành            |
| 4    | Chạy cross-topology CRL (5 seeds) + existing baselines | Controlled comparison data    |
| 5    | Aggregate results + p-value + báo cáo P0               | P0 complete                   |

### Tuần 2: P1 (Important)

| Ngày | Task                                                  | Output               |
| ---- | ----------------------------------------------------- | -------------------- |
| 1    | Implement canonicalization ablation + chạy            | Ablation C1 data     |
| 2    | Fisher discount (β) ablation                          | Ablation C2 data     |
| 3    | Step limit sensitivity test + backward transfer eval  | η analysis + BT data |
| 4-5  | Aggregate all P1 results + integrate vào paper tables | P1 complete          |

### Tuần 3: P2 (If Time Permits)

| Ngày | Task                                            | Output                   |
| ---- | ----------------------------------------------- | ------------------------ |
| 1–2  | Reward shaping implementation + medium topology | Medium results           |
| 3    | CRL method comparison (PackNet/fine-tune)       | CRL comparison table     |
| 4–5  | CyRIS setup attempt / Paper writing             | Real exec or framing fix |

---

## Bảng Mapping: Task → Paper Section

| Task                   | Paper Section Affected                         | RQ            |
| ---------------------- | ---------------------------------------------- | ------------- |
| P0.1 Multi-seed        | §7.3, §7.5, §8.5 (removes L4)                  | RQ2, RQ3      |
| P0.2 Baselines         | §6.2, §7.3 (Table 4)                           | RQ2           |
| P0.3 Intra vs Cross    | §7.5 (Table 8)                                 | RQ3           |
| P1.1 Canon ablation    | §7.1 (RQ1 ablation), §5.3                      | RQ1           |
| P1.2 β ablation        | §5.5.2, §7 new section                         | C2 validation |
| P1.3 Step limit        | §7.7, §8 discussion                            | RQ2 defense   |
| P1.4 Backward transfer | §7 new subsection                              | H validation  |
| P2.1 Medium + reward   | §7.6 (upgrades from failure → partial success) | RQ2           |
| P2.2 CRL comparison    | §4.3, §7 new section                           | RQ3           |
| P2.3 Real execution    | §8.5 (removes L1)                              | F1 fix        |
| P2.4 Pillar ablation   | §5.8, §7 new section                           | Depth         |

---

## Tiêu Chí Paper-Ready

Bài báo đủ mạnh để submit khi đạt được:

### Minimum Viable (AISec Workshop — 8 pages):

- [x] P0.1 — Multi-seed (5 seeds, tiny + small-linear)
- [x] P0.2 — ≥1 baseline bổ sung (extended scratch hoặc fine-tune only)
- [x] P0.3 — Controlled intra vs cross comparison

### Competitive (ACSAC/RAID — 12-18 pages):

- Tất cả P0 + thêm:
- [x] P1.1 — Canonicalization ablation
- [x] P1.2 — β ablation
- [x] P1.3 — Step limit sensitivity
- [x] P1.4 — Backward transfer evaluation

### Strong (Journal — Computers & Security):

- Tất cả P0 + P1 + thêm:
- [x] P2.1 — Medium topology với reward shaping
- [x] P2.2 — CRL method comparison
- [x] P2.3 — Real execution partial verification

---

## Lưu Ý Quan Trọng

1. **Framing phải điều chỉnh trước khi viết paper:**
   - KHÔNG dùng "sim-to-real" nếu không có KVM execution → dùng "cross-framework transfer"
   - KHÔNG nhắc đến HIB, contrastive learning, LLM domain randomization → đã bỏ
   - Redefine hypothesis H: bỏ p<0.05 requirement nếu không đủ seeds, hoặc chờ P0.1 hoàn thành

2. **Tất cả experiments phải reproducible:**
   - Mỗi run ghi lại: seed, config file, git commit hash, output directory
   - MetricStore JSON lưu metadata đầy đủ

3. **Ưu tiên depth cho RQ3 (death spiral):**
   - Đây là đóng góp mạnh nhất → cần evidence mạnh nhất
   - Cross vs intra comparison phải là controlled experiment, không phải cherry-pick từ 2 runs khác nhau

---

## Kế Hoạch Thực Thi Tối Ưu (2 Máy Song Song)

> **Ngày cập nhật:** 2026-02-27  
> **Mục tiêu:** Giảm tối đa số lần train bằng cách gom thí nghiệm, phân bổ 2 máy song song  
> **Ước tính tham khảo:** exp_B (2 topology, 12 tasks, seed=42) mất ~2.7h; Phase 1 sim ~7 min

### Nguyên Tắc Gom Thí Nghiệm

1. **P0.1 (multi-seed intra) absorb P0.3 (intra side)** — Mỗi seed chạy full pipeline (Phase 0→4) với `--training-mode intra_topology` + `--train-scratch` → thu được cả θ_dual, θ_scratch, FT, BT cho mỗi seed. Đây chính là dữ liệu cần cho P0.1 VÀ nửa "intra" của P0.3.

2. **P0.3 (cross side) = thêm 5 runs cross_topology** — Cùng seed set, cùng scenarios, chỉ đổi `--training-mode cross_topology`. So sánh trực tiếp intra vs cross per seed.

3. **P0.2 (baselines) = 1 run scratch-only** + baselines benchmark — Scratch-only đã được gom vào P0.1 (`--train-scratch`). Chỉ cần chạy thêm extended scratch (3× budget) và existing baselines (Random, Greedy, DQN) riêng.

4. **P1.2 (β ablation) = không cần sửa code** — `--fisher-beta` đã là CLI argument, chỉ cần chạy 5 lệnh với β khác nhau.

5. **P1.1 (canon ablation) = cần sửa 1 file duy nhất** (`unified_state_encoder.py`), sau đó chạy 1 lần.

6. **P1.3 (step limit) + P1.4 (backward transfer) = eval-only** — Dùng model đã train từ P0.1, chỉ evaluate lại với step_limit khác / trên sim tasks → không cần train thêm.

---

### Tổng Hợp Runs Cần Chạy

| Run ID      | Mục đích                      | Training Mode    | Seed       | Scenarios                         | Flags                               | ~Thời gian | Đáp ứng                            |
| ----------- | ----------------------------- | ---------------- | ---------- | --------------------------------- | ----------------------------------- | ---------- | ---------------------------------- |
| **R1–R5**   | Multi-seed intra + scratch    | `intra_topology` | 0,1,2,3,42 | tiny(T1–T4) + small-linear(T1–T4) | `--train-scratch`                   | ~3h/run    | P0.1 + P0.3(intra) + P0.2(scratch) |
| **R6–R10**  | Multi-seed cross-topology     | `cross_topology` | 0,1,2,3,42 | tiny(T1–T4) + small-linear(T1–T4) | (no scratch)                        | ~2.5h/run  | P0.3(cross)                        |
| **R11**     | Extended scratch (3× budget)  | scratch-only     | 42         | tiny + small-linear               | `--scratch-only`, 3× episodes       | ~4h        | P0.2                               |
| **R12**     | Baselines (Random/Greedy/DQN) | benchmark        | 42         | tiny + small-linear T1–T4         | `run_benchmark.py baselines`        | ~1h        | P0.2                               |
| **R13–R17** | β ablation (5 values)         | `intra_topology` | 42         | tiny(T1–T4) + small-linear(T1–T4) | `--fisher-beta 0.0/0.1/0.5/0.7/1.0` | ~3h/run    | P1.2                               |
| **R18**     | No-canonicalization           | `intra_topology` | 42         | tiny(T1–T4) + small-linear(T1–T4) | `--no-canonicalization` (\*)        | ~3h        | P1.1                               |
| **R19**     | Fine-tune only (no CRL)       | `intra_topology` | 42         | tiny(T1–T4) + small-linear(T1–T4) | config: `script-no__retention.yaml` | ~3h        | P0.2(C) + P1                       |

> (\*) R18 cần sửa code trước (thêm toggle vào `unified_state_encoder.py` — xem phần Code Changes bên dưới)

**Tổng: 19 runs.** Nhưng 2 máy song song → có thể hoàn thành trong ~4-5 ngày nếu chạy liên tục.

---

### Phase 1 — Chia Máy Tối Ưu (Ngày 1–3: P0)

#### Lưu ý chung

- Sim scenarios (Phase 1): `data/scenarios/chain/chain-msfexp_vul-sample-6_envs-seed_0.json`
- PenGym scenarios (Phase 3): dùng `compiled/` (10 variants/tier) thay `compiled_tiny/` (5 variants)
- Episode config: `data/config/curriculum_episodes.json`
- Mỗi run output vào thư mục riêng để không ghi đè

#### Máy A — Runs intra-topology (R1→R5)

Chạy lần lượt 5 seeds, mỗi seed là 1 full pipeline run:

```powershell
# ── Máy A: R1 (seed=0) ──
cd D:\NCKH\fusion\pentest

python run_strategy_c.py ^
  --sim-scenarios data/scenarios/chain/chain-msfexp_vul-sample-6_envs-seed_0.json ^
  --pengym-scenarios ^
    data/scenarios/generated/compiled/tiny_T1_000.yml ^
    data/scenarios/generated/compiled/tiny_T2_000.yml ^
    data/scenarios/generated/compiled/tiny_T3_000.yml ^
    data/scenarios/generated/compiled/tiny_T4_000.yml ^
    data/scenarios/generated/compiled/small-linear_T1_000.yml ^
    data/scenarios/generated/compiled/small-linear_T2_000.yml ^
    data/scenarios/generated/compiled/small-linear_T3_000.yml ^
    data/scenarios/generated/compiled/small-linear_T4_000.yml ^
  --episode-config data/config/curriculum_episodes.json ^
  --training-mode intra_topology ^
  --transfer-strategy conservative ^
  --fisher-beta 0.3 ^
  --train-scratch ^
  --seed 0 ^
  --output-dir outputs/multiseed/seed_0
```

```powershell
# ── Máy A: R2 (seed=1) ──
python run_strategy_c.py ^
  --sim-scenarios data/scenarios/chain/chain-msfexp_vul-sample-6_envs-seed_0.json ^
  --pengym-scenarios ^
    data/scenarios/generated/compiled/tiny_T1_000.yml ^
    data/scenarios/generated/compiled/tiny_T2_000.yml ^
    data/scenarios/generated/compiled/tiny_T3_000.yml ^
    data/scenarios/generated/compiled/tiny_T4_000.yml ^
    data/scenarios/generated/compiled/small-linear_T1_000.yml ^
    data/scenarios/generated/compiled/small-linear_T2_000.yml ^
    data/scenarios/generated/compiled/small-linear_T3_000.yml ^
    data/scenarios/generated/compiled/small-linear_T4_000.yml ^
  --episode-config data/config/curriculum_episodes.json ^
  --training-mode intra_topology ^
  --transfer-strategy conservative ^
  --fisher-beta 0.3 ^
  --train-scratch ^
  --seed 1 ^
  --output-dir outputs/multiseed/seed_1
```

```powershell
# ── Máy A: R3 (seed=2) ──
# Lệnh giống R1/R2, thay --seed 2 --output-dir outputs/multiseed/seed_2

python run_strategy_c.py ^
  --sim-scenarios data/scenarios/chain/chain-msfexp_vul-sample-6_envs-seed_0.json ^
  --pengym-scenarios ^
    data/scenarios/generated/compiled/tiny_T1_000.yml ^
    data/scenarios/generated/compiled/tiny_T2_000.yml ^
    data/scenarios/generated/compiled/tiny_T3_000.yml ^
    data/scenarios/generated/compiled/tiny_T4_000.yml ^
    data/scenarios/generated/compiled/small-linear_T1_000.yml ^
    data/scenarios/generated/compiled/small-linear_T2_000.yml ^
    data/scenarios/generated/compiled/small-linear_T3_000.yml ^
    data/scenarios/generated/compiled/small-linear_T4_000.yml ^
  --episode-config data/config/curriculum_episodes.json ^
  --training-mode intra_topology ^
  --transfer-strategy conservative ^
  --fisher-beta 0.3 ^
  --train-scratch ^
  --seed 2 ^
  --output-dir outputs/multiseed/seed_2
```

```powershell
# ── Máy A: R4 (seed=3), R5 (seed=42) ──
# Tương tự, thay --seed và --output-dir tương ứng:
#   R4: --seed 3  --output-dir outputs/multiseed/seed_3
#   R5: --seed 42 --output-dir outputs/multiseed/seed_42
```

**Máy A tổng thời gian ước tính:** 5 runs × ~3h = **~15h** (chạy liên tục ~2 ngày).

#### Máy B — Runs cross-topology (R6→R10)

Song song với Máy A, Máy B chạy cross-topology cho cùng 5 seeds:

```powershell
# ── Máy B: R6 (seed=0, cross-topology) ──
cd D:\NCKH\fusion\pentest

python run_strategy_c.py `
  --sim-scenarios data/scenarios/chain/chain-msfexp_vul-sample-6_envs-seed_0.json `
  --pengym-scenarios `
    data/scenarios/generated/compiled/tiny_T1_000.yml `
    data/scenarios/generated/compiled/tiny_T2_000.yml `
    data/scenarios/generated/compiled/tiny_T3_000.yml `
    data/scenarios/generated/compiled/tiny_T4_000.yml `
    data/scenarios/generated/compiled/small-linear_T1_000.yml `
    data/scenarios/generated/compiled/small-linear_T2_000.yml `
    data/scenarios/generated/compiled/small-linear_T3_000.yml `
    data/scenarios/generated/compiled/small-linear_T4_000.yml `
  --episode-config data/config/curriculum_episodes.json `
  --training-mode cross_topology `
  --transfer-strategy conservative `
  --fisher-beta 0.3 `
  --seed 0 `
  --output-dir outputs/multiseed_cross/seed_0
```

```powershell
# ── Máy B: R7–R10 ──
# Tương tự R6, thay:
#   R7:  --seed 1  --output-dir outputs/multiseed_cross/seed_1
#   R8:  --seed 2  --output-dir outputs/multiseed_cross/seed_2
#   R9:  --seed 3  --output-dir outputs/multiseed_cross/seed_3
#   R10: --seed 42 --output-dir outputs/multiseed_cross/seed_42
```

**Máy B tổng thời gian ước tính (cross):** 5 runs × ~2.5h = **~12.5h**.

#### Máy B — Sau khi xong R6–R10, chạy tiếp R11 + R12

```powershell
# ── Máy B: R11 — Extended scratch (3× budget) ──
# Tạo config mới với 3× episodes
python run_strategy_c.py ^
  --sim-scenarios data/scenarios/chain/chain-msfexp_vul-sample-6_envs-seed_0.json ^
  --pengym-scenarios ^
    data/scenarios/generated/compiled/tiny_T1_000.yml ^
    data/scenarios/generated/compiled/tiny_T2_000.yml ^
    data/scenarios/generated/compiled/tiny_T3_000.yml ^
    data/scenarios/generated/compiled/tiny_T4_000.yml ^
    data/scenarios/generated/compiled/small-linear_T1_000.yml ^
    data/scenarios/generated/compiled/small-linear_T2_000.yml ^
    data/scenarios/generated/compiled/small-linear_T3_000.yml ^
    data/scenarios/generated/compiled/small-linear_T4_000.yml ^
  --episode-config data/config/curriculum_episodes_3x.json ^
  --scratch-only ^
  --seed 42 ^
  --output-dir outputs/baselines/extended_scratch
```

```powershell
# ── Máy B: R12 — Existing baselines ──
python run_benchmark.py baselines ^
  --episodes 50 ^
  --scenarios tiny small-linear
```

> **Lưu ý R11:** Cần tạo file `data/config/curriculum_episodes_3x.json` — copy từ `curriculum_episodes.json` rồi nhân `base_episodes` × 3. Xem phần "Configs Cần Tạo" bên dưới.

---

### Phase 2 — Chia Máy Tối Ưu (Ngày 3–5: P1)

> Bắt đầu sau khi P0 hoàn thành trên cả 2 máy.  
> P1.3 (step limit) và P1.4 (backward transfer) **không cần train mới** — chỉ eval lại model từ P0.1.

#### Máy A — β ablation (R13–R17)

5 giá trị β, chỉ cần chạy trên seed=42:

```powershell
# ── Máy A: R13 (β=0.0) ──
python run_strategy_c.py ^
  --sim-scenarios data/scenarios/chain/chain-msfexp_vul-sample-6_envs-seed_0.json ^
  --pengym-scenarios ^
    data/scenarios/generated/compiled/tiny_T1_000.yml ^
    data/scenarios/generated/compiled/tiny_T2_000.yml ^
    data/scenarios/generated/compiled/tiny_T3_000.yml ^
    data/scenarios/generated/compiled/tiny_T4_000.yml ^
    data/scenarios/generated/compiled/small-linear_T1_000.yml ^
    data/scenarios/generated/compiled/small-linear_T2_000.yml ^
    data/scenarios/generated/compiled/small-linear_T3_000.yml ^
    data/scenarios/generated/compiled/small-linear_T4_000.yml ^
  --episode-config data/config/curriculum_episodes.json ^
  --training-mode intra_topology ^
  --transfer-strategy conservative ^
  --fisher-beta 0.0 ^
  --seed 42 ^
  --output-dir outputs/ablation_beta/beta_0.0
```

```powershell
# ── Máy A: R14–R17 ──
# Tương tự R13, chỉ thay --fisher-beta và --output-dir:
#   R14: --fisher-beta 0.1  --output-dir outputs/ablation_beta/beta_0.1
#   R15: --fisher-beta 0.5  --output-dir outputs/ablation_beta/beta_0.5
#   R16: --fisher-beta 0.7  --output-dir outputs/ablation_beta/beta_0.7
#   R17: --fisher-beta 1.0  --output-dir outputs/ablation_beta/beta_1.0
```

> **Tối ưu thêm:** β=0.3 đã có data từ R5 (seed=42, intra) → không cần chạy lại. Dùng trực tiếp `outputs/multiseed/seed_42/` làm β=0.3 data point → chỉ cần chạy 4 runs (R13,R14,R15,R16,R17 = 5 values nhưng bỏ 0.3).

**Máy A thời gian β ablation:** 4 runs × ~3h = **~12h**.

#### Máy B — Canonicalization ablation (R18) + Fine-tune only (R19)

```powershell
# ── Máy B: R18 — No canonicalization ──
# ⚠ CẦN SỬA CODE TRƯỚC (xem phần Code Changes bên dưới)
python run_strategy_c.py ^
  --sim-scenarios data/scenarios/chain/chain-msfexp_vul-sample-6_envs-seed_0.json ^
  --pengym-scenarios ^
    data/scenarios/generated/compiled/tiny_T1_000.yml ^
    data/scenarios/generated/compiled/tiny_T2_000.yml ^
    data/scenarios/generated/compiled/tiny_T3_000.yml ^
    data/scenarios/generated/compiled/tiny_T4_000.yml ^
    data/scenarios/generated/compiled/small-linear_T1_000.yml ^
    data/scenarios/generated/compiled/small-linear_T2_000.yml ^
    data/scenarios/generated/compiled/small-linear_T3_000.yml ^
    data/scenarios/generated/compiled/small-linear_T4_000.yml ^
  --episode-config data/config/curriculum_episodes.json ^
  --training-mode intra_topology ^
  --no-canonicalization ^
  --seed 42 ^
  --output-dir outputs/ablation_canon/no_canon
```

```powershell
# ── Máy B: R19 — Fine-tune only (no EWC, no retrospection) ──
python run_strategy_c.py ^
  --sim-scenarios data/scenarios/chain/chain-msfexp_vul-sample-6_envs-seed_0.json ^
  --pengym-scenarios ^
    data/scenarios/generated/compiled/tiny_T1_000.yml ^
    data/scenarios/generated/compiled/tiny_T2_000.yml ^
    data/scenarios/generated/compiled/tiny_T3_000.yml ^
    data/scenarios/generated/compiled/tiny_T4_000.yml ^
    data/scenarios/generated/compiled/small-linear_T1_000.yml ^
    data/scenarios/generated/compiled/small-linear_T2_000.yml ^
    data/scenarios/generated/compiled/small-linear_T3_000.yml ^
    data/scenarios/generated/compiled/small-linear_T4_000.yml ^
  --episode-config data/config/curriculum_episodes.json ^
  --training-mode intra_topology ^
  --ewc-lambda 0 ^
  --seed 42 ^
  --output-dir outputs/ablation_crl/finetune_only
```

**Máy B thời gian:** R18 (~3h) + R19 (~3h) = **~6h**.

#### Cả 2 Máy — P1.3 & P1.4 (Eval-only, sau P0)

P1.3 (step limit sensitivity) và P1.4 (backward transfer) chỉ cần **evaluate model đã train**, không cần train mới. Cần viết script eval nhỏ hoặc dùng lại Phase 4 evaluator. Chi tiết:

- **P1.3:** Load θ_dual từ `outputs/multiseed/seed_42/models/`, evaluate trên small-linear với step_limit = [50, 100, 200, 500]. Đây là eval-only loop (~5 phút/config).
- **P1.4:** Load θ_dual từ mỗi seed, evaluate trên 6 sim scenarios. Phase 4 evaluator đã có backward transfer logic — chỉ cần chạy eval mode.

> Hai tasks này có thể chạy trên bất kỳ máy nào rảnh trước, mỗi task chỉ mất ~30 phút.

---

### Phase 3 — P2 Song Song Với Viết Paper (Ngày 5+)

Sau khi P0+P1 hoàn tất, bắt đầu viết paper. Đồng thời, 1 hoặc 2 máy chạy thêm P2 nếu cần:

| Run     | Máy | Mục tiêu                                         | Thời gian ước tính |
| ------- | --- | ------------------------------------------------ | ------------------ |
| R20     | A   | P2.1 — Medium + reward shaping (cần sửa code)    | ~6–10h             |
| R21–R26 | B   | P2.4 — 6 CRL pillar ablations (existing configs) | ~18h total         |

---

### Sơ Đồ Gantt (2 Máy)

```
Ngày  │ Máy A                              │ Máy B
──────┼─────────────────────────────────────┼──────────────────────────────────────
  1   │ R1 (seed=0, intra, ~3h)            │ R6 (seed=0, cross, ~2.5h)
      │ R2 (seed=1, intra, ~3h)            │ R7 (seed=1, cross, ~2.5h)
──────┼─────────────────────────────────────┼──────────────────────────────────────
  2   │ R3 (seed=2, intra, ~3h)            │ R8 (seed=2, cross, ~2.5h)
      │ R4 (seed=3, intra, ~3h)            │ R9 (seed=3, cross, ~2.5h)
──────┼─────────────────────────────────────┼──────────────────────────────────────
  3   │ R5 (seed=42, intra, ~3h)           │ R10 (seed=42, cross, ~2.5h)
      │ [P0 aggregate + verify]            │ R11 (extended scratch, ~4h)
      │                                     │ R12 (baselines benchmark, ~1h)
──────┼─────────────────────────────────────┼──────────────────────────────────────
  4   │ R13 β=0.0 (~3h)                    │ R18 no-canon (~3h) [sau sửa code]
      │ R14 β=0.1 (~3h)                    │ R19 finetune-only (~3h)
──────┼─────────────────────────────────────┼──────────────────────────────────────
  5   │ R15 β=0.5 (~3h)                    │ P1.3 step-limit eval (~0.5h)
      │ R16 β=0.7 (~3h)                    │ P1.4 backward transfer eval (~0.5h)
      │ R17 β=1.0 (~3h)                    │ [P1 aggregate + tables]
──────┼─────────────────────────────────────┼──────────────────────────────────────
 6+   │ 📝 Viết paper                      │ P2 runs (nếu kịp)
      │ + P2.1 medium reward-shaping       │ P2.4 pillar ablations
```

---

### Configs Cần Tạo Trước Khi Chạy

#### 1. `data/config/curriculum_episodes_3x.json` (cho R11 extended scratch)

```json
{
  "base_episodes": {
    "tiny": 1500,
    "tiny-hard": 1500,
    "tiny-small": 1500,
    "small-linear": 2250,
    "small-honeypot": 3000,
    "medium-single-site": 4500,
    "medium": 15000,
    "medium-multi-site": 15000
  },
  "base_step_limit": {
    "tiny": 200,
    "tiny-hard": 200,
    "tiny-small": 200,
    "small-linear": 500,
    "small-honeypot": 500,
    "medium-single-site": 500,
    "medium": 1000,
    "medium-multi-site": 1000
  },
  "tier_multiplier": { "T1": 0.8, "T2": 1.0, "T3": 2.0, "T4": 3.0 },
  "default_episodes": 3000,
  "default_step_limit": 500,
  "training_mode": "intra_topology"
}
```

---

### Code Changes Cần Thiết (Trước Khi Chạy)

#### Thay đổi 1: Canonicalization toggle (cho R18 — P1.1)

**File:** `src/envs/core/unified_state_encoder.py`

Thêm `use_canonicalization` vào constructor và bọc conditional trong `encode()`:

```python
# Trong __init__:
def __init__(self, encoder=None, use_canonicalization: bool = True):
    self.use_canonicalization = use_canonicalization
    # ... existing code ...

# Trong encode() — thay 2 dòng canonicalize:
canonical_os = self.canonicalize_os(os_name) if self.use_canonicalization else os_name
canonical_services = (self.canonicalize_services(services)
                      if self.use_canonicalization else services)
```

**File:** `src/training/dual_trainer.py`

Thread `use_canonicalization` qua `DualTrainer.__init__()` → `UnifiedStateEncoder(use_canonicalization=...)`.

**File:** `run_strategy_c.py`

Thêm CLI argument:

```python
parser.add_argument(
    "--no-canonicalization", action="store_true",
    help="Disable cross-domain canonicalization (ablation study).",
)
```

Truyền vào DualTrainer:

```python
use_canonicalization=not args.no_canonicalization
```

#### Thay đổi 2: Aggregate script (cho post-processing P0/P1)

**File:** `src/evaluation/aggregate_seeds.py` — tạo mới

Script đọc tất cả `outputs/multiseed/seed_*/strategy_c_results.json`, tính:

- Mean ± std cho SR, NR, η per topology
- Mean ± std cho FT*SR, FT*η, BT*SR, BT*η
- Paired t-test (scipy.stats.ttest_rel): θ_dual SR vs θ_scratch SR
- 95% CI cho SR
- Export bảng tổng hợp ra CSV + JSON

> Script này không block training — có thể viết song song trong lúc chờ runs hoàn thành.

---

### Checklist Trước Khi Bắt Đầu

- [ ] Verify scenarios tồn tại: `ls data/scenarios/generated/compiled/tiny_T*_000.yml` và `small-linear_T*_000.yml`
- [ ] Tạo file `data/config/curriculum_episodes_3x.json`
- [ ] Sửa code canonicalization toggle (chỉ cần trước Ngày 4)
- [ ] Clone repo sang Máy B (hoặc sync qua git)
- [ ] Verify 2 máy có cùng Python env + dependencies
- [ ] Tạo sẵn thư mục output: `outputs/multiseed/`, `outputs/multiseed_cross/`, `outputs/ablation_beta/`, `outputs/ablation_canon/`, `outputs/ablation_crl/`, `outputs/baselines/`

```powershell
# Tạo sẵn thư mục
mkdir -p outputs/multiseed outputs/multiseed_cross outputs/ablation_beta outputs/ablation_canon outputs/ablation_crl outputs/baselines
```

### Tóm Tắt Hiệu Quả

| Metric                    | Trước tối ưu               | Sau tối ưu                                         |
| ------------------------- | -------------------------- | -------------------------------------------------- |
| Tổng runs cần train       | ~30+ (mỗi task riêng lẻ)   | **19 runs**                                        |
| Thời gian nếu 1 máy       | ~57h                       | ~57h                                               |
| Thời gian 2 máy song song | —                          | **~30h (~5 ngày, 6h/ngày)**                        |
| Code changes cần trước P0 | multi-seed runner phức tạp | **0 (dùng shell loop)**                            |
| Code changes cần trước P1 | 3 files                    | **1 file** (`unified_state_encoder.py` + CLI flag) |
| P0.1 + P0.3 gom lại       | 2 experiment sets riêng    | **1 set (R1–R5 đáp ứng cả hai)**                   |
| P1.2 (β ablation)         | Cần sửa code               | **0 sửa code** (CLI flag đã có)                    |
| P1.3 + P1.4               | Cần train mới              | **0 train** (eval-only trên model P0)              |
