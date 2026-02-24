# Strategy C — Trạng Thái Hệ Thống

> Cập nhật: 2026-02-24 (rev.5). Rev.5: Calibration results, per-task episode schedule, curriculum T1→T4 support, optimal_rewards/steps fix.

---

## 1. Những gì đã đạt được

### 1.1 Vượt qua giới hạn của PenGym và SCRIPT

**PenGym gốc** là một môi trường penetration testing được mô phỏng với observation space rời rạc, thiếu ngữ nghĩa. Agent trong PenGym chỉ "thấy" vector nhị phân mô tả trạng thái mạng, không có khả năng hiểu nghĩa của OS hay service. **SCRIPT gốc** được huấn luyện hoàn toàn trên simulation riêng (không phải PenGym), với state encoding 1538-dim sử dụng SBERT để biểu diễn OS/service — nhưng hai hệ thống dùng state khác nhau, không thể chuyển giao chính sách trực tiếp.

Strategy C giải quyết vấn đề này qua các cải tiến sau:

**a) Unified State Encoder (1540-dim)**

Thay vì để SCRIPT-sim và PenGym dùng hai bộ mã hóa khác nhau, hệ thống thống nhất về một representation:

- `access[0:3]` — 3-dim (none / user / root) thay vì 2-dim binary
- `discovery[3:4]` — 1-dim mới, xác định host đã được quét chưa
- 4 × SBERT(384) cho OS, port, service, auxiliary
- Canonicalization map đảm bảo rằng `ubuntu` và `linux`, hay `openssh` và `ssh`, được mã hóa giống nhau ở cả hai domain

Kết quả: Phase 0 xác nhận `cross_domain_os_cosine = 1.0` — OS description từ sim và PenGym cho cùng embedding, loại bỏ domain gap tại tầng state representation.

**b) Unified Reward Normalizer**

SCRIPT-sim reward có thể lên đến +1000 (tổng reward của nhiều hosts), trong khi PenGym reward tối đa +100 per host. Nếu fine-tune trực tiếp, EWC Fisher từ sim sẽ bị "át" bởi gradient scale khác nhau. UnifiedNormalizer ánh xạ cả hai về `[-1, +1]` trước khi tính Fisher penalty, để EWC so sánh được trên cùng thang đo.

**c) Hierarchical Action Space (ServiceActionSpace)**

PenGym gốc có action space hàng nghìn chiều (mỗi \<exploit, host\> là một action riêng). SCRIPT có action space 16-dim dựa trên service abstraction. Strategy C giữ nguyên SCRIPT action interface 16-dim và thêm `select_cve()` bên trong HOST để ánh xạ service-level action → CVE cụ thể, đồng thời build `ServiceActionSpace` với 2064 CVE được nhóm thành 16 nhóm. Agent không cần biết chi tiết CVE — đây là abstraction quan trọng để policy học được scale.

**d) Domain Transfer với EWC**

Quy trình Phase 1 → 2 → 3:

- Phase 1: Train trên 6 sim tasks đến SR=100% (`final_reward=0.99` sau normalization)
- Phase 2 (`conservative`): Reset state normalizer, thu thập 447 warmup states từ PenGym để rebuild phân phối, áp fisher_discount=0.3 (giảm 70% memory từ sim để nhường chỗ cho PenGym), giảm learning rate xuống 10% (lr_factor=0.1)
- Phase 3: Fine-tune trên PenGym với EWC penalty — không học từ đầu, giữ lại cấu trúc chính sách từ sim

Kết quả so sánh trực tiếp trên `tiny`: `theta_dual` giải trong 10 steps vs `theta_sim_unified` cần 31 steps — transfer thực sự xảy ra, agent dual-trained hội tụ nhanh hơn nhiều trên môi trường PenGym.

**e) Pipeline đo lường transfer (Phase 4)**

Hệ thống đủ infrastructure để đo Forward Transfer, Backward Transfer, Transfer Ratio — thứ mà cả PenGym và SCRIPT gốc không có. Evaluator so sánh 3 agents cùng lúc trên cùng tasks, đảm bảo kết quả comparable.

---

## 2. Vấn đề còn tồn tại

### 2.1 Định nghĩa SR hiện tại sai về bản chất — và hệ quả ceiling/floor effect

**Hiện trạng code:** `Agent.Evaluate()` chạy **1 episode duy nhất** trên mỗi task, rồi tính `SR = len(success_list) / len(target_list)`. Với 2 tasks, SR chỉ nhận 3 giá trị: 0.0, 0.5, 1.0. Đây **không phải success _rate_** — đây là binary success ratio trên 1 lần thử. "Rate" đúng nghĩa đòi hỏi _nhiều lần đo_ trên cùng task để tính xác suất thành công.

**Hệ quả trực tiếp:**

- **Floor effect:** Khi scenario quá khó với budget hiện tại (vd: `small-linear`, 500 eps, step*limit=100), cả `theta_dual` lẫn `theta_scratch` đều SR=0 trên task đó → FT = 0, dù dual agent \_có thể* đã tiến bộ hơn scratch (ít bước hơn, scan nhiều host hơn trước khi timeout).
- **Ceiling effect:** Nếu tăng `train_eps` và `step_limit` đủ lớn để mọi agent giải được tất cả scenarios → SR = 1.0 cho cả hai → FT = 0, dù dual agent _có thể_ hội tụ nhanh hơn gấp 5 lần.
- **Cả hai trường hợp đều cho FT = 0 với ý nghĩa hoàn toàn khác nhau.** Metric hiện tại không phân biệt được.

**Ceiling effect không phải vấn đề cần "tránh" — nó là tín hiệu rằng metric sai, không phải method sai.** Nếu cả hai agent đều SR=100% nhưng `theta_dual` giải trong 10 steps còn `theta_scratch` cần 80 steps, thì transfer rõ ràng đang hoạt động — chỉ là SR binary không capture được điều đó.

### 2.2 Giải pháp: Multi-episode SR + Normalized Reward + Step Efficiency

**a) Multi-episode SR — sửa đúng bản chất**

Chạy K episodes per task (K=20). PPO policy output có stochasticity (entropy > 0) nên mỗi episode cho kết quả khác. SR per task trở thành giá trị liên tục [0, 1]:

$$SR_s = \frac{1}{K}\sum_{k=1}^{K} \mathbb{1}[\text{success}_{s,k}]$$

$$SR_{agent} = \frac{1}{|\mathcal{S}|}\sum_{s \in \mathcal{S}} SR_s$$

Với 4 scenarios × 20 episodes = 80 datapoints → FT resolution = 0.0125, phát hiện được cải tiến ~1%. Kèm standard error:

$$SE = \sqrt{\frac{SR_{agent}(1 - SR_{agent})}{|\mathcal{S}| \times K}}$$

FT có ý nghĩa thống kê khi $|FT| > 2 \times SE$.

**b) Normalized Reward (NR) — đo hiệu quả, không chỉ thành/bại**

$$NR_s = \frac{R_{actual,s}}{R_{optimal,s}}$$

$R_{optimal}$ lấy từ comment scenario YAML: `tiny=195`, `tiny-hard=192`, `tiny-small=189`, `small-linear=179`. NR ∈ (−∞, 1], cho phép phân biệt "gần giải được" (NR ≈ 0.8) vs "hoàn toàn lạc hướng" (NR < 0) ngay cả khi binary success = 0.

**c) Step Efficiency (SE) — đo tốc độ giải quyết**

$$\eta_s = \frac{\text{optimal\_steps}_s}{\text{actual\_steps}_s} \times \mathbb{1}[\text{success}]$$

Chỉ tính cho episodes thành công. $\eta = 1.0$ nghĩa là giải bằng đúng số bước tối ưu. Metric này _hoàn toàn immune_ với ceiling effect: ngay cả khi SR = 100%, step efficiency vẫn phân biệt được agents.

**d) Forward Transfer mới — tổ hợp 3 metrics:**

$$FT_{SR} = SR_{\theta_{dual}} - SR_{\theta_{scratch}}$$
$$FT_{NR} = NR_{\theta_{dual}} - NR_{\theta_{scratch}}$$
$$FT_{\eta} = \eta_{\theta_{dual}} - \eta_{\theta_{scratch}}$$

Báo cáo cả 3. FT*SR đo khả năng giải, FT_NR đo reward gap, FT*η đo tốc độ. Nếu FT*SR = 0 nhưng FT*η > 0 → transfer giúp giải nhanh hơn, SR đơn thuần không thấy.

### 2.3 Backward Transfer — ceiling effect trên sim và cách khắc phục

**Vấn đề gốc (thiếu data):** Phase 4 hiện chỉ truyền `pengym_tasks` vào evaluator, không truyền `sim_tasks`. `_compute_transfer_metrics()` đã có code tính BT nhưng không bao giờ nhận được `sim` domain data → `backward_transfer` không xuất hiện trong kết quả.

**Công thức CRL chuẩn:**

$$BT = \frac{1}{n-1}\sum_{i=1}^{n-1} \left[ R_{n,i} - R_{i,i} \right]$$

với $R_{j,i}$ = performance trên task $i$ _sau khi_ train xong task $j$. Áp dụng cho Strategy C: $R_{1,1}$ = SR(θ*sim_unified trên sim), $R*{2,1}$ = SR(θ*dual trên sim), $BT = R*{2,1} - R\_{1,1}$.

**Vấn đề sâu hơn — ceiling effect trên BT_SR:**

Ngay cả khi truyền `sim_tasks` vào Phase 4, BT_SR vẫn **không có ý nghĩa**. Lý do:

1. Phase 1 train θ_sim_unified đến SR ≈ 100% trên sim (final_reward = 0.99)
2. EWC có _nhiệm vụ chính_ là bảo toàn performance trên sim → θ_dual cũng đạt SR ≈ 100% trên sim
3. $BT_{SR} = 100\% - 100\% = 0$ **bất kể EWC hoạt động tốt hay kém**, miễn agent vẫn "giải được" task

Đây là ceiling effect tương tự §2.1 nhưng ở chiều ngược lại. Sim tasks quá dễ sau khi đã train đến convergence → binary SR không đủ nhạy để phát hiện degradation. Ngay cả multi-episode SR (K=20) cũng không giúp nếu cả 20 episodes đều thành công.

**Ví dụ cụ thể:** Giả sử θ*sim_unified giải `tiny` sim trong 6 bước, θ_dual giải trong 15 bước (kiến thức PenGym "nhiễu" decision boundary). Cả hai đều \_thành công* → SR = 100% → BT_SR = 0. Nhưng rõ ràng có forgetting — agent chậm hơn 2.5x.

**Giải pháp — 3 tầng, từ outcome-level đến policy-level:**

**Tầng 1: Multi-dimensional BT (outcome-level, tương tự FT)**

Áp dụng cùng logic đã đề xuất cho FT trong §2.2, nhưng trên sim domain:

$$BT_{\eta} = \eta(\theta_{dual}, sim) - \eta(\theta_{sim}, sim)$$
$$BT_{NR} = NR(\theta_{dual}, sim) - NR(\theta_{sim}, sim)$$

- $BT_{\eta}$ giải quyết triệt để ví dụ trên: $\eta_{sim} = 6/6 = 1.0$, $\eta_{dual} = 6/15 = 0.4$, $BT_{\eta} = -0.6$ → phát hiện rõ forgetting ngay cả khi SR = 100%
- $BT_{NR}$ phát hiện reward degradation: θ_dual có thể thành công nhưng tốn nhiều action cost hơn → NR giảm
- **Đây là tầng quan trọng nhất** — dễ implement, trực tiếp interpretable, và immune với ceiling effect khi task vẫn solvable

Kỳ vọng: $BT_{\eta} \approx 0$ nghĩa EWC hoạt động tốt. $BT_{\eta} < -0.15$ → EWC lambda cần tăng.

**Tầng 2: Policy-level divergence (fine-grained, cho deep analysis)**

Đo trực tiếp policy thay đổi bao nhiêu, hoàn toàn tránh được ceiling effect vì không phụ thuộc vào task success:

**a) Action Distribution Divergence trên sim states:**

Lấy N sim states (có sẵn từ Phase 1 replay buffer), chạy cả hai policy:

$$D_{KL}(\pi_{sim} \| \pi_{dual}) = \frac{1}{N}\sum_{i=1}^{N} \sum_{a} \pi_{sim}(a|s_i) \log\frac{\pi_{sim}(a|s_i)}{\pi_{dual}(a|s_i)}$$

$D_{KL} = 0$ → policy identically. $D_{KL} > 0$ → policy shift. Không cần chạy episode, chỉ forward pass → rất nhanh.

**b) Fisher-weighted Parameter Distance:**

Fisher information từ Phase 1 đã lưu trong `OnlineEWC.importances[task_id][param_name]` (diagonal approximation). Tham số tham chiếu lưu ở `saved_params[task_id]`. Tính:

$$\Delta_F = \sum_{k} F_k \cdot (\theta_{dual,k} - \theta_{sim,k})^2$$

trong đó $F_k$ = Fisher diagonal cho parameter $k$, sum trên toàn bộ parameters. Metric này **trực tiếp đo** lượng thay đổi trên chính các dimensions mà EWC đang bảo vệ:

- $\Delta_F \approx 0$ → EWC enforcement thành công, weights không di chuyển
- $\Delta_F$ lớn → learning rate hoặc PenGym gradient vượt qua EWC penalty

Code tham khảo:

```python
def compute_fisher_distance(ewc_method, theta_dual_model):
    """Compute Δ_F between dual and sim-unified parameters."""
    total = 0.0
    # importances[t] chứa Fisher diag cho task t cuối cùng
    latest_task = max(ewc_method.importances.keys())
    fisher = ewc_method.importances[latest_task]
    saved = ewc_method.saved_params[latest_task]

    for name, param in theta_dual_model.named_parameters():
        if name in fisher and name in saved:
            f_k = fisher[name].data
            delta = param.data - saved[name].data
            total += (f_k * delta.pow(2)).sum().item()
    return total
```

**Khi nào dùng tầng nào?**

| Tầng | Metric      | Khi nào dùng                                       | Chi phí                              |
| ---- | ----------- | -------------------------------------------------- | ------------------------------------ |
| 1    | BT_η, BT_NR | **Luôn luôn** — primary BT metrics                 | Thấp (cùng eval loop với FT)         |
| 2    | D_KL, Δ_F   | Khi BT_η ≈ 0 nhưng muốn xác nhận / khi debug EWC λ | Trung bình (forward pass all states) |

**Cần lưu sim_tasks** từ Phase 1 và truyền vào Phase 4 evaluator. Fisher information đã có sẵn trong `OnlineEWC.importances` — không cần thay đổi Phase 1 để access.

### 2.4 Eval adapter reuse — vấn đề NASim class-level state

**Bản chất kỹ thuật:** NASim lưu `HostVector` dimension indices dưới dạng **class attributes** (không phải instance). Khi scenario B load sau scenario A, class vars bị ghi đè → mọi environment instance của scenario A bị corrupt. `_ensure_wrapper()` xử lý bằng cách check `_active_scenario` (class var) và recreate wrapper khi scenario thay đổi.

**Vấn đề trong Phase 4:** Evaluator gọi `evaluate_agent()` lần lượt cho mỗi agent trên **cùng list `_pengym_tasks`**. Flow:

```
Agent A eval: tiny.reset() → _ensure_wrapper(tiny) → small-linear.reset() → _ensure_wrapper(small-linear)
Agent B eval: tiny.reset() → _ensure_wrapper(tiny) [recreate!] → small-linear.reset() → _ensure_wrapper(small-linear) [recreate!]
```

Mỗi lần recreate wrapper, environment state **hoàn toàn mới** (gọi lại `env.reset()` bên trong). Điều này đúng ra nên deterministic nếu cùng seed — nhưng:

1. `SingleHostPenGymWrapper.__init__()` tạo NASim env mới → random state phụ thuộc seed + creation order
2. Wrapper creation order khác nhau giữa agents vì Phase 3 đã eval xen kẽ trên `_pengym_tasks`
3. NASim env internal RNG state không được reset một cách controlled

**Hiện tại:** Chênh lệch reward (-33.33 vs -96.67) đã phân tích kỹ ở phiên trước là do **chất lượng policy thực sự khác nhau** (dual agent chọn action rẻ cost=1, scratch agent chọn ssh cost=3). Tuy nhiên, **không chứng minh được** eval environment state đồng nhất 100% giữa agents. Cần fix để loại bỏ nghi ngờ.

**Fix:** Tạo fresh adapter set cho mỗi agent trong Phase 4, cùng seed, đảm bảo mỗi agent đều eval trên env state identically initialized.

### 2.5 Reward signal bị méo do `min_reward` hardcode

`UnifiedNormalizer(source='pengym')` dùng `min_reward=-3.0` (max exploit cost trong `tiny`). Nhưng scenarios khác nhau có cost structures khác nhau. Khi agent thực hiện action cost=1 thất bại: `normalize(-1) = -1/3 ≈ -0.333`. Khi cost=3: `normalize(-3) = -1.0`. **Tỷ lệ phạt** giữa hai loại hành động là 1:3, giống hệt raw reward — normalizer không phá hỏng tương quan cost, chỉ co scale.

**Vấn đề thực chất:** Không phải méo signal, mà là **max_reward=100 quá lớn so với reward thực tế**. Reward dương lớn nhất trong PenGym khi exploit thành công là `+100` (per sensitive host). Nhưng reward dương trung gian (scan thành công, pivot host compromised) thường nhỏ hơn nhiều → bị co lại thành < 0.01 → gradient dương rất yếu so với gradient âm.

Ảnh hưởng Fisher: Fisher information tỷ lệ với gradient², nên parameters tham gia vào positive reward bị đánh giá thấp → EWC ít bảo vệ chúng → dễ bị overwrite khi chuyển domain. Đây là rủi ro trung bình — chỉ ảnh hưởng khi EWC lambda đủ lớn.

---

## 3. Trạng thái triển khai (đã implement)

> Tất cả thay đổi bên dưới đã được implement trực tiếp trên code. §3 này mô tả _những gì đã làm_, không phải kế hoạch.

### 3.1 `StrategyCEvaluator` — đã rewrite hoàn toàn

**File:** `src/evaluation/strategy_c_eval.py`

**Interface mới:**

```python
StrategyCEvaluator(
    pengym_tasks: Union[Dict[str, list], list],  # per-agent isolated hoặc shared
    sim_tasks: Optional[list] = None,
    step_limit: int = 100,
    eval_episodes: int = 20,
    optimal_rewards: Optional[Dict[str, float]] = None,
    optimal_steps: Optional[Dict[str, float]] = None,
)
```

**Thay đổi chính:**

- `evaluate_agent()` chạy K episodes per task (K=`eval_episodes`) với eval loop riêng — không gọi `Agent.Evaluate()`. Tính SR liên tục, NR, step_efficiency (η), standard error.
- `_get_pengym_tasks(agent_name)` trả về bộ adapters riêng nếu `pengym_tasks` là dict, hoặc shared list nếu là list.
- `_compute_transfer_metrics()` tính FT_SR/FT_NR/FT_eta + FT_SR_significant + BT_SR/BT_NR/BT_eta + BT_KL/BT_fisher_dist (injected via `policy_metrics`).
- `print_report()` hiển thị cột NR, η, SE bên cạnh SR.

### 3.2 `DualTrainer.phase4_evaluation()` — đã rewrite

**File:** `src/training/dual_trainer.py`

**Thay đổi đã thực hiện:**

1. **`self._sim_tasks = sim_tasks`** — lưu trong `phase1_sim_training()` để Phase 4 eval BT trên sim.
2. **`_create_eval_pengym_tasks()`** — factory method tạo bộ adapter mới (cùng seed) cho mỗi lần eval. Mỗi agent nhận fresh adapter set → loại bỏ NASim class-level state leakage.
3. **`phase4_evaluation()`** — tạo `per_agent_pengym_tasks` dict, khởi tạo evaluator với `sim_tasks`, `eval_episodes=20`, `optimal_rewards`, `optimal_steps`. Inject `policy_metrics` trước khi tính transfer metrics.

### 3.3 Policy-level BT metrics — đã implement

**File:** `src/training/dual_trainer.py`

**Methods mới:**

- **`_collect_sim_states(n=200)`**: Roll out θ_sim policy trên sim_tasks, thu thập N observation vectors.
- **`_compute_policy_bt_metrics()`**: Tính `BT_KL` (D_KL(π_sim ‖ π_dual) trên sim states, batch forward pass) và `BT_fisher_dist` (Σ_k F_k (θ_dual_k − θ_sim_k)² từ `OnlineEWC.importances`).

**Access path thực tế trong code:**

```python
# Actor (action probs): agent_cl.cl_agent.keeper.Policy.actor(state_tensor)
# Fisher diagonal:      agent_cl.cl_agent.ewc.importances[task_id][param_name]
# Saved params:         agent_cl.cl_agent.ewc.saved_params[task_id][param_name]
```

### 3.4 Calibration results (2026-02-24)

**32 calibration runs** (8 base scenarios × 4 episode counts, `--scratch-only`, step_limit=150, K=20 eval):

| Scenario           |     | A      |                          | Opt R/Steps             | 500 eps           | 1000 eps   | 2000 eps | 3000 eps |
| ------------------ | --- | ------ | ------------------------ | ----------------------- | ----------------- | ---------- | -------- | -------- |
| tiny               | 42  | 195/6  | SR=1.0 η=0.60 NR=−0.004  | ← same                  | ← same            | ← same     |
| tiny-hard          | 42  | 192/5  | SR=1.0 η=1.25 NR=+0.002  | ← same                  | ← same            | ← same     |
| tiny-small         | 64  | 189/7  | SR=1.0 η=0.64 NR=−0.013  | ← same                  | ← same            | ← same     |
| small-linear       | 96  | 179/12 | SR=1.0 η=0.52 NR=−0.063  | ← same                  | ← same            | ← same     |
| small-honeypot     | 96  | 186/8  | **SR=0.0**               | SR=1.0 η=0.53 NR=−0.025 | ← same            | ← same     |
| medium-single-site | 192 | 191/4  | SR=1.0 η=0.087 NR=−0.128 | η=0.08 NR=−0.100        | η=0.095 NR=−0.082 | ← same     |
| medium             | 192 | 185/8  | **SR=0.0**               | **SR=0.0**              | **SR=0.0**        | **SR=0.0** |
| medium-multi-site  | 192 | 187/7  | **SR=0.0**               | **SR=0.0**              | **SR=0.0**        | **SR=0.0** |

**Nhận xét:**

- **saturated ngay** (tiny, tiny-hard, tiny-small, small-linear): SR=1.0 deterministic tại mọi eps. 500 eps đã đủ.
- **small-honeypot**: Honeypot exploration challenge → min 1000 eps.
- **medium-single-site**: SR=1.0 nhưng η rất thấp (~0.09, agent mất 42–50 steps vs optimal 4). Quality cải thiện đến 2000 eps rồi saturate.
- **medium, medium-multi-site**: SR=0.0 tại mọi eps → |A|=192 + multi-subnet quá khó cho DQN exploration budget 3000 eps. **Cần curriculum training T1→T4** với overlay để giảm dần difficulty.

**Bug fix:** `optimal_rewards` và `optimal_steps` trong `dual_trainer.py` chỉ có 4/8 scenarios → NR/η null cho 4 scenarios còn lại. **Đã fix rev.5** — bổ sung đủ 8 scenarios.

### 3.5 Per-task episode schedule — đã implement

**File thay đổi:** `agent_continual.py`, `dual_trainer.py`, `run_strategy_c.py`

**Vấn đề:** `train_eps` là global (cùng giá trị cho mọi task trong `train_continually()`). Curriculum T1→T4 cần episode khác nhau: tiny_T1 cần 500 eps nhưng medium_T4 cần 15000.

**Giải pháp:**

1. **`agent_continual.py`**: `train_continually()` nhận `episode_schedule: Optional[Dict[int, int]]` — mapping task_index → episode_count. `learn_new_task()` nhận `max_episodes` override.
2. **`dual_trainer.py`**: `_resolve_episode_schedule()` hỗ trợ 2 mode:
   - **Multiplier mode**: `episodes = base_episodes[base_name] × tier_multiplier["T{n}"]`. Parse tên file scenario.
   - **Rules mode**: Regex pattern matching, first match wins.
3. **`run_strategy_c.py`**: `--episode-config path/to/config.json`

**Config file:** `data/config/curriculum_episodes.json`

### 3.6 Experiment protocol — quy trình chạy thí nghiệm

**Bước 1: Calibration** — ✅ Đã hoàn thành (§3.4)

**Bước 2: Curriculum T1→T4 training** — sử dụng compiled overlay scenarios:

```bash
# Ví dụ: chạy full pipeline cho tiny với curriculum T1→T4 (3 training variants mỗi tier)
python run_strategy_c.py \
    --sim-scenarios data/scenarios/chain/chain-msfexp_vul-sample-6_envs-seed_0.json \
    --pengym-scenarios \
        data/scenarios/generated/compiled/tiny_T1_001.yml \
        data/scenarios/generated/compiled/tiny_T1_002.yml \
        data/scenarios/generated/compiled/tiny_T1_003.yml \
        data/scenarios/generated/compiled/tiny_T2_001.yml \
        data/scenarios/generated/compiled/tiny_T2_002.yml \
        data/scenarios/generated/compiled/tiny_T2_003.yml \
        data/scenarios/generated/compiled/tiny_T3_001.yml \
        data/scenarios/generated/compiled/tiny_T3_002.yml \
        data/scenarios/generated/compiled/tiny_T3_003.yml \
        data/scenarios/generated/compiled/tiny_T4_001.yml \
        data/scenarios/generated/compiled/tiny_T4_002.yml \
        data/scenarios/generated/compiled/tiny_T4_003.yml \
    --episode-config data/config/curriculum_episodes.json \
    --train-scratch --step-limit 150

# Chạy full pipeline (24~30h)
cd d:\NCKH\fusion\pentest

# Build ordered scenario list: T1→T2→T3→T4
$bases = @("tiny","tiny-hard","tiny-small","small-linear","small-honeypot","medium-single-site","medium","medium-multi-site")
$sc = @(); foreach ($t in @("T1","T2","T3","T4")) { foreach ($b in $bases) { foreach ($v in @("001","002","003")) { $sc += "data/scenarios/generated/compiled/${b}_${t}_${v}.yml" } } }

python run_strategy_c.py `
    --sim-scenarios data/scenarios/chain/chain-msfexp_vul-sample-6_envs-seed_0.json `
    --pengym-scenarios $sc `
    --episode-config data/config/curriculum_episodes.json `
    --train-scratch `
    --step-limit 150 `
    --episodes 1000 `
    --eval-freq 5 `
    --ewc-lambda 2000 `
    --seed 42 `
    --output-dir outputs/strategy_c/curriculum_full
```

**Episode budget (từ curriculum_episodes.json, multiplier mode):**

| Base scenario      | T1   | T2   | T3    | T4    | Total/variant | Total (×3) |
| ------------------ | ---- | ---- | ----- | ----- | ------------- | ---------- |
| tiny               | 500  | 500  | 1000  | 1500  | 3,500         | 10,500     |
| tiny-hard          | 500  | 500  | 1000  | 1500  | 3,500         | 10,500     |
| tiny-small         | 500  | 500  | 1000  | 1500  | 3,500         | 10,500     |
| small-linear       | 600  | 750  | 1500  | 2250  | 5,100         | 15,300     |
| small-honeypot     | 1200 | 1500 | 3000  | 4500  | 10,200        | 30,600     |
| medium-single-site | 2000 | 2500 | 5000  | 7500  | 17,000        | 51,000     |
| medium             | 4000 | 5000 | 10000 | 15000 | 34,000        | 102,000    |
| medium-multi-site  | 4000 | 5000 | 10000 | 15000 | 34,000        | 102,000    |

**Lưu ý:** `--episodes` trên CLI vẫn cần set (dùng làm fallback), nhưng khi `--episode-config` có, mỗi task dùng episode từ config.

**Dataset split:**

- `_000`: calibration (đã dùng)
- `_001`, `_002`, `_003`: training
- `_004`–`_009`: held-out evaluation

**Bước 3: Đánh giá** — Tiêu chí pass/fail:

| Metric             | Ngưỡng pass                | Ý nghĩa                                   |
| ------------------ | -------------------------- | ----------------------------------------- |
| FT_SR > 0          | Dual SR cao hơn scratch    | Sim pre-training giúp giải nhiều task hơn |
| FT_NR > 0.05       | Dual reward tốt hơn 5%     | Sim pre-training giúp giải hiệu quả hơn   |
| FT_SR > 2×SE       | Significant FT             | Cải thiện không do random variance        |
| BT_η ≥ −0.15       | Step efficiency giảm ≤ 15% | EWC giữ tốc độ giải sim (primary BT)      |
| BT_NR ≥ −0.10      | NR sim giảm ≤ 10%          | EWC giữ reward quality trên sim           |
| BT_KL < 0.5        | Policy shift nhỏ           | Policy trên sim ít thay đổi (diagnostic)  |
| BT_Δ_F < threshold | Fisher-weighted drift nhỏ  | EWC enforcement thành công (diagnostic)   |

---

## Tổng kết ưu tiên

| #   | Vấn đề                              | Mức độ     | Trạng thái | File đã sửa                             |
| --- | ----------------------------------- | ---------- | ---------- | --------------------------------------- |
| 1   | SR binary 1-episode                 | Cao        | ✅ Done    | `strategy_c_eval.py`                    |
| 2   | Thiếu NR và step efficiency         | Cao        | ✅ Done    | `strategy_c_eval.py`                    |
| 3   | Fresh adapter per agent             | Cao        | ✅ Done    | `dual_trainer.py`, `strategy_c_eval.py` |
| 4   | BT ceiling effect + thiếu sim_tasks | Cao        | ✅ Done    | `dual_trainer.py`, `strategy_c_eval.py` |
| 5   | Quá ít scenarios (2)                | Trung bình | ✅ Done    | curriculum config, `run_strategy_c.py`  |
| 6   | optimal_rewards chỉ có 4/8          | Cao        | ✅ Done    | `dual_trainer.py`                       |
| 7   | Per-task episode schedule           | Cao        | ✅ Done    | `agent_continual.py`, `dual_trainer.py` |
| 8   | min_reward/max_reward hardcode      | Thấp       | Chưa       | `reward_normalizer.py`                  |
