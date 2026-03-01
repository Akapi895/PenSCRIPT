# Đánh giá toàn diện Phase 1 & Phase 2 — Đối chiếu analyze.md vs Dữ liệu thực nghiệm

> **Ngày:** 2026-03-01  
> **Dữ liệu:** 17 strategy_c_results.json (5 intra, 5 cross, 6 β-ablation, 1 canon, 1 CRL)  
> **Công cụ:** analyze_results.py (numpy + scipy.stats)

---

## I. Tổng quan Kết quả Thực nghiệm

### 1.1 Intra-Topology (multiseed/ — 5 seeds)

| Task            | Mean SR   | Std       | Seeds                         |
| --------------- | --------- | --------- | ----------------------------- |
| tiny_T1         | 0.800     | 0.400     | [1.0, 0.0, 1.0, 1.0, 1.0]     |
| tiny_T2         | 1.000     | 0.000     | [1.0, 1.0, 1.0, 1.0, 1.0]     |
| **tiny_T3**     | **0.000** | **0.000** | **[0.0, 0.0, 0.0, 0.0, 0.0]** |
| tiny_T4         | 1.000     | 0.000     | [1.0, 1.0, 1.0, 1.0, 1.0]     |
| small-linear_T1 | 0.500     | 0.277     | [0.2, 1.0, 0.3, 0.55, 0.45]   |
| small-linear_T2 | 0.510     | 0.260     | [0.35, 1.0, 0.35, 0.55, 0.3]  |
| small-linear_T3 | 0.330     | 0.336     | [0.15, 1.0, 0.15, 0.2, 0.15]  |
| small-linear_T4 | 0.360     | 0.328     | [0.25, 1.0, 0.15, 0.1, 0.3]   |

- **Overall SR: 0.562 ± 0.096**
- **FT_SR: 0.562 ± 0.096** (vs buggy scratch SR=0)
- **BT_SR: -0.767 ± 0.389** (catastrophic sim forgetting)
- **Phase 1 sim SR: 1.000 ± 0.000** (sim training converges perfectly)

### 1.2 Cross-Topology (multiseed_cross/ — 5 seeds)

- **Overall SR: 0.637 ± 0.123** (HIGHER than intra)
- **BT_SR: -0.367 ± 0.521** (less forgetting than intra)

### 1.3 Intra vs Cross Statistical Comparison

|              | Intra | Cross | p-value                     |
| ------------ | ----- | ----- | --------------------------- |
| Overall SR   | 0.562 | 0.637 | **0.365** (NOT significant) |
| tiny         | 0.700 | 0.622 | 0.519                       |
| small-linear | 0.425 | 0.653 | 0.307                       |

**One-sample FT_SR > 0:** t=11.707, p=0.0002 (significant, BUT computed against buggy scratch=0%)

### 1.4 Ablation Results

| Ablation      | SR        | Notes                      |
| ------------- | --------- | -------------------------- |
| β=0.0         | 0.375     | No Fisher discount         |
| β=0.1         | 0.456     |                            |
| **β=0.3**     | **0.525** | Default setting            |
| β=0.5         | 0.406     |                            |
| **β=0.7**     | **0.525** | Tied with β=0.3            |
| β=1.0         | 0.487     | Full Fisher (no discount)  |
| Canon ON      | 0.525     |                            |
| Canon OFF     | 0.475     | Δ=0.05 (minimal)           |
| Full CRL      | 0.525     |                            |
| Finetune-only | 0.362     | BUT scratch SR=0.875 here! |

### 1.5 Benchmark Baselines (tiny)

| Agent          | SR  | Avg Steps |
| -------------- | --- | --------- |
| Random         | 1.0 | 38.9      |
| Greedy-Exploit | 1.0 | 28.1      |
| Scan-First     | 1.0 | 31.7      |
| DQN-Baseline   | 1.0 | 20.5      |
| A2C-Baseline   | 1.0 | 28.1      |

---

## II. Đánh giá từng Vấn đề trong analyze.md

### W1 — Single seed (seed=42)

- **Trạng thái:** ✅ ĐÃ KHẮC PHỤC — 5 seeds cho cả intra và cross
- **Nhưng:** Kết quả multi-seed cho thấy performance yếu hơn NHIỀU so với paper outline claims
  - Paper outline: "SR=100% on tiny and small-linear"
  - **Thực tế: SR=0.562 ± 0.096 (intra), 0.637 ± 0.123 (cross)**
  - seed_42 (original) cho SR=0.525 — trung bình, không phải best case
- **Đánh giá:** Vấn đề seeds đã giải quyết, nhưng kết quả multi-seed phá vỡ narrative "SR=100%"

### W2 — Không có ablation study

- **Trạng thái:** ✅ ĐÃ KHẮC PHỤC — có 3 loại ablation
- **Nhưng:** Kết quả ablation yếu:
  - **β ablation:** β=0.3 và β=0.7 tied ở 0.525, không có clear trend. β=0.0 (0.375) < β=0.3 (0.525) gợi ý Fisher discount giúp ích, nhưng β=0.5 (0.406) thấp hơn β=1.0 (0.487) — pattern không đơn giản
  - **Canon ablation:** Δ=0.05 — canonicalization hầu như không có tác dụng trên SR. Contribution C1 bị yếu
  - **CRL ablation (finetune vs EWC):** Phát hiện scratch bug (xem N1)
- **Đánh giá:** Ablation đã có nhưng không strongly support bất kỳ claim nào

### W3 — Baseline quá yếu (scratch SR=0%)

- **Trạng thái:** ❌ NGHIÊM TRỌNG HƠN — Phát hiện bug
- **Chi tiết:**
  - `train_pengym_scratch()` trong dual_trainer.py kế thừa `ewc_lambda` từ experiment config
  - Khi λ=2000 (tất cả runs tiêu chuẩn): EWC penalize deviation from random init → agent KHÔNG THỂ HỌC
  - Khi λ=0 (chỉ finetune_only run): scratch train bình thường → **SR=0.875 (train), 0.75 (eval)**
  - **Scratch SR=0% trên tất cả runs tiêu chuẩn là ARTIFACT của bug, không phải inability to learn**
- **Hệ quả:**
  - FT_SR = dual_SR - scratch_SR = 0.562 - 0.0 = 0.562 → **SAI**
  - FT_SR thực sự ≈ 0.562 - 0.75 = **-0.188** (NEGATIVE TRANSFER!)
  - t-test FT_SR > 0 (p=0.0002) trở nên **vô nghĩa**
  - **Claim trung tâm "transfer learning creates positive FT" bị phá vỡ**
- **Đánh giá:** ĐÂY LÀ VẤN ĐỀ NGHIÊM TRỌNG NHẤT. Cần fix bug và re-run tất cả experiments

### W4 — small-linear η=3.5% mâu thuẫn

- **Trạng thái:** ⚠️ VẪN TỒN TẠI nhưng context mới
- **Thực tế multi-seed:** small-linear SR = 0.425 ± 0.291 (intra), 0.653 ± 0.299 (cross)
- SR không còn 100% qua seeds → η trở nên ít relevant hơn
- Vấn đề lớn hơn: variance CỰC CAO trên small-linear — một seed đạt 100%, seeds khác chỉ 20-35%

### L1 — Không có real execution (KVM)

- **Trạng thái:** ❌ CHƯA GIẢI QUYẾT
- Vẫn chỉ NASim simulation mode

### L2 — Chỉ 3 topologies (quá ít)

- **Trạng thái:** ❌ CHƯA GIẢI QUYẾT
- Vẫn chỉ tiny + small-linear solvable

### L3 — Backward transfer thiếu data

- **Trạng thái:** ✅ CÓ DATA nhưng kết quả tiêu cực
- BT_SR = -0.767 ± 0.389 (intra): agent quên gần hết sim knowledge
- BT predominantly -1.0: EWC KHÔNG bảo vệ sim knowledge
- Hypothesis yêu cầu BT_η ≥ -0.15 → **THẤT BẠI HOÀN TOÀN**

### F1 — Sim-to-real framing misleading

- **Trạng thái:** ⚠️ CẦN REFRAME
- Tất cả cần dùng "cross-framework transfer" thay "sim-to-real"

### F2 — p<0.05 requirement

- **Trạng thái:** ❌ BỊ VÔ HIỆU bởi scratch bug
- t-test hiện tại so sánh dual vs buggy scratch → vô nghĩa
- Cần re-compute sau khi fix scratch bug

### F4 — β=0.3 justification

- **Trạng thái:** ⚠️ YẾU
- β=0.3 và β=0.7 cùng SR=0.525
- Pattern: β=0 (0.375) < β≠0, nhưng không có sweet spot rõ ràng ở 0.3
- Có thể frame: "β > 0 helps (Fisher discount is beneficial), but not sensitive to exact value"

---

## III. Vấn đề MỚI phát hiện từ dữ liệu

### N1 — SCRATCH BUG (CRITICAL — ĐỘ NGHIÊM TRỌNG: 10/10)

**Bản chất:** `train_pengym_scratch()` kế thừa `cl_config` từ experiment, bao gồm `ewc_lambda=2000`.
Sau task 0, EWC compute Fisher matrix trên random weights. Từ task 1 trở đi, penalty = λ × Σ F_k(θ-θ\*)² rất lớn → gradient bị triệt tiêu → agent locked ở random policy.

**Bằng chứng:** finetune_only (λ=0) scratch SR=0.875; tất cả runs khác (λ=2000) scratch SR=0.0.

**Tác động:** TOÀN BỘ FT_SR claims vô hiệu. Cần:

1. Fix bug: `scratch_cl_config.ewc_lambda = 0`
2. Re-run ít nhất 5 seeds intra + 5 seeds cross với scratch đúng
3. Tính lại FT_SR với scratch baseline đúng
4. Nếu FT_SR < 0 → transfer không giúp ích, cần reconceptualize approach

### N2 — T3 SCENARIO BUG (CRITICAL — 8/10)

**Bản chất:** `compiled/tiny_T3_000.yml` host đầu tiên có `access: user` thay vì `access: root`.
Đã fix trong `compiled_tiny/` nhưng KHÔNG DÙNG trong training scripts.

**Tác động:** T3 SR=0% xuyên suốt 10 runs (5 intra + 5 cross) là artifact. Nếu T3 thực sự access=root thì:

- tiny SR sẽ cao hơn (thêm 1 task solvable)
- Overall SR sẽ tăng
- Nhưng CẦN RE-RUN để xác nhận

### N3 — Cross-topology KHÔNG tệ hơn Intra (7/10)

**Paper outline claim:** "Intra-topology CRL eliminates death spiral → SR 37.5% → 100%"
**Thực tế multi-seed:** Cross SR (0.637) ≥ Intra SR (0.562), p=0.365

**Phân tích:**

- Death spiral evidence từ exp2 (single seed, 3 topologies inc. medium) có thể vẫn đúng trong setup cũ
- Nhưng trong setup 2-topology (tiny + small-linear, không có medium), cross-topology KHÔNG tệ hơn
- Death spiral có thể chỉ xảy ra khi có topology fail (medium) trong sequence
- Cần reframe: "death spiral is triggered by including a failing topology, not by cross-topology training per se"

### N4 — Benchmark cho thấy tiny quá dễ (6/10)

Random agent đạt SR=100% trên tiny (38.9 steps).
→ tiny SR không meaningful cho demonstrating transfer value.
→ small-linear là topology duy nhất có potential, nhưng SR = 0.425 ± 0.291 (intra), unstable.

### N5 — BT_SR = -1.0: EWC thất bại hoàn toàn (6/10)

BT_SR predominantly -1.0 nghĩa là agent quên hoàn toàn sim knowledge sau PenGym training.
EWC λ=2000 + Fisher discount β=0.3 KHÔNG ngăn chặn forgetting.
→ Claim "CRL preserves knowledge during transfer" bị bác bỏ.

### N6 — Phương sai cực cao trên small-linear (5/10)

small-linear SR: seed 1 = 1.0, seed 0 = 0.238, seed 2 = 0.237, seed 3 = 0.350, seed 4 = 0.300.
Std = 0.291 → kết quả essentially random across seeds.
→ Agent không robust; seed 1 là outlier may mắn.

---

## IV. Tóm tắt Tình trạng — Hypothesis (§0.3) vs Reality

| Hypothesis Component  | Claim                    | Evidence                                             | Verdict                |
| --------------------- | ------------------------ | ---------------------------------------------------- | ---------------------- |
| FT_SR > 0             | Dual outperforms scratch | Scratch buggy (λ=2000 → SR=0%); true scratch SR=0.75 | **❌ CHƯA XÁC NHẬN**   |
| p < 0.05              | Statistical significance | t-test so với buggy scratch: p=0.0002                | **❌ VÔ HIỆU**         |
| BT_η ≥ -0.15          | Maintain sim knowledge   | BT_SR = -0.767                                       | **❌ THẤT BẠI**        |
| Death spiral resolved | Intra > Cross            | Cross (0.637) ≥ Intra (0.562), p=0.365               | **❌ KHÔNG SUPPORTED** |
| CVE curriculum works  | T1→T4 gradient           | T3=0% (bug), others mixed                            | **⚠️ INCONCLUSIVE**    |

---

## V. Khuyến nghị Hành động (theo thứ tự ưu tiên)

### P0 — PHẢI LÀM TRƯỚC KHI VIẾT BÀI

1. **Fix scratch bug:** Thêm `scratch_cl_config.ewc_lambda = 0` trong `train_pengym_scratch()`
2. **Fix T3 scenario:** Cập nhật `compiled/tiny_T3_000.yml` → `access: root`
3. **Re-run 5 seeds intra + 5 seeds cross** với cả 2 fixes trên
4. **Tính lại FT_SR** với scratch đúng
5. Nếu FT_SR < 0: reconceptualize — transfer KHÔNG giúp ích, paper cần framing khác hoàn toàn

### P1 — KHUYẾN NGHỊ MẠNH

6. Re-run β ablation với scratch fix để confirm β effect
7. Thêm baseline: PPO-from-scratch (λ=0, tương đương finetune_only scratch) làm proper control
8. Analyze: tại sao cross-topology KHÔNG tệ hơn intra trong 2-topology setup? Death spiral chỉ xảy ra khi có failing topology?

### P2 — NÊN LÀM

9. Extend benchmark baselines sang small-linear (hiện chỉ có tiny)
10. Test với step limit giảm (50, 100) để check nếu transfer giúp convergence nhanh hơn
11. Ablation: EWC λ values (500, 1000, 2000, 5000) cho dual agent

---

## VI. Paper Outline — Sections cần sửa

| Section             | Vấn đề                                                     | Mức độ sửa                    |
| ------------------- | ---------------------------------------------------------- | ----------------------------- |
| §0.3 Hypothesis     | BT_η ≥ -0.15 → sai; FT_SR claim chưa verified              | **Viết lại hoàn toàn**        |
| §0.4 RQ table       | Kết quả thực nghiệm cũ (SR=100%, FT_SR=1.0)                | **Cập nhật data**             |
| §2 Abstract         | Claims SR=100%, FT_SR up to 1.0                            | **Viết lại với data đúng**    |
| §3.3 Para 5         | "100% SR vs 37.5% for naive cross-topology"                | **Remove or reframe**         |
| §5.6.1 Death spiral | Evidence exp2 vẫn valid nhưng multi-seed contradicts       | **Thêm nuance**               |
| §6.2 Baselines      | θ_scratch definition — cần note bug fix                    | **Thêm proper control**       |
| §7.3 Main results   | SR=100%, FT_SR=1.0 — tất cả sai                            | **Viết lại hoàn toàn**        |
| §7.5 Intra vs Cross | "Intra eliminates death spiral"                            | **Reframe dựa trên data mới** |
| §8.4 T3 bug         | Claims "fixed" nhưng compiled/ chưa fix                    | **Cập nhật**                  |
| §8.5 Limitations    | L4 "single seed" → đã fix; thêm scratch bug + new findings | **Cập nhật**                  |
| Appendix A          | Data points cũ                                             | **Thay thế hoàn toàn**        |
