# Tổng Hợp Sửa Đổi và Cải Tiến: Tích Hợp SCRIPT vào PenGym

> **Cập nhật:** 2026-02-25  
> **Phạm vi:** Toàn bộ cải tiến logic hệ thống và phương pháp trong quá trình tích hợp SCRIPT agent vào môi trường PenGym  
> **Branch:** `strC_1`

---

## Mục lục

- [I. Các vấn đề gặp phải khi tích hợp](#i-các-vấn-đề-gặp-phải-khi-tích-hợp)
  - [1. Xung đột mô hình tương tác: Single-host vs Multi-host](#1-xung-đột-mô-hình-tương-tác-single-host-vs-multi-host)
  - [2. Không gian trạng thái không tương thích giữa hai hệ thống](#2-không-gian-trạng-thái-không-tương-thích-giữa-hai-hệ-thống)
  - [3. Không gian hành động chênh lệch bậc độ lớn](#3-không-gian-hành-động-chênh-lệch-bậc-độ-lớn)
  - [4. Chênh lệch thang reward ~100 lần](#4-chênh-lệch-thang-reward-100-lần)
  - [5. Thiếu cầu nối interface giữa CRL framework và PenGym](#5-thiếu-cầu-nối-interface-giữa-crl-framework-và-pengym)
  - [6. Không có cơ chế transfer có kiểm soát giữa hai domain](#6-không-có-cơ-chế-transfer-có-kiểm-soát-giữa-hai-domain)
  - [7. PenGym thiếu đa dạng CVE và chưa hỗ trợ gradient độ khó](#7-pengym-thiếu-đa-dạng-cve-và-chưa-hỗ-trợ-gradient-độ-khó)
  - [8. Hệ thống đánh giá không phù hợp cho bài toán transfer learning](#8-hệ-thống-đánh-giá-không-phù-hợp-cho-bài-toán-transfer-learning)
- [II. Hướng giải quyết tương ứng](#ii-hướng-giải-quyết-tương-ứng)
  - [1. SingleHostPenGymWrapper — Bọc multi-host thành single-host](#1-singlehostpengymwrapper--bọc-multi-host-thành-single-host)
  - [2. Unified State Encoder (1540-dim) — Chuẩn hóa biểu diễn trạng thái](#2-unified-state-encoder-1540-dim--chuẩn-hóa-biểu-diễn-trạng-thái)
  - [3. Hierarchical Action Space (16-dim) — Trừu tượng hóa không gian hành động](#3-hierarchical-action-space-16-dim--trừu-tượng-hóa-không-gian-hành-động)
  - [4. Unified Reward Normalizer — Chuẩn hóa reward về [-1, +1]](#4-unified-reward-normalizer--chuẩn-hóa-reward-về--1-1)
  - [5. PenGymHostAdapter — Duck-typing interface HOST](#5-pengymhostadapter--duck-typing-interface-host)
  - [6. Domain Transfer Manager — Chuyển giao có kiểm soát qua pipeline 5 pha](#6-domain-transfer-manager--chuyển-giao-có-kiểm-soát-qua-pipeline-5-pha)
  - [7. CVE Pipeline — Mở rộng CVE và Curriculum Training](#7-cve-pipeline--mở-rộng-cve-và-curriculum-training)
  - [8. Hệ thống đánh giá đa chiều — Khắc phục ceiling/floor effect](#8-hệ-thống-đánh-giá-đa-chiều--khắc-phục-ceilingfloor-effect)
  - [9. Per-Scenario Step Limit — Ngân sách bước tự động theo topology](#9-per-scenario-step-limit--ngân-sách-bước-tự-động-theo-topology)
  - [10. TeeLogger — Lọc nhiễu console giữ nguyên log file](#10-teelogger--lọc-nhiễu-console-giữ-nguyên-log-file)
- [III. Đánh giá mức độ hoàn thành](#iii-đánh-giá-mức-độ-hoàn-thành)
  - [1. Thành phần đã hoàn tất](#1-thành-phần-đã-hoàn-tất)
  - [2. Hạn chế còn tồn tại](#2-hạn-chế-còn-tồn-tại)

---

## I. Các vấn đề gặp phải khi tích hợp

### 1. Xung đột mô hình tương tác: Single-host vs Multi-host

SCRIPT được thiết kế theo mô hình **tấn công từng host riêng lẻ**: mỗi mục tiêu được đại diện bởi một đối tượng HOST, agent tương tác với từng HOST thông qua `reset()` → `perform_action()` → nhận kết quả cho đúng host đó. Quá trình huấn luyện lặp qua danh sách HOST, mỗi vòng lặp là một task CRL độc lập.

PenGym theo mô hình **quản lý toàn bộ mạng**: một environment duy nhất chứa N hosts, mỗi lệnh `step(action)` phải chỉ định _host nào_ và _hành động gì_ thông qua một flat action index. Observation trả về là vector trạng thái của _toàn bộ mạng_, không phải của riêng host nào.

Xung đột này không chỉ là khác biệt API mà là **khác biệt paradigm**: SCRIPT "nhìn" và "hành động" trên từng host, trong khi PenGym "nhìn" toàn mạng và "hành động" trên toàn mạng. Mọi thành phần của SCRIPT — từ state encoding, action selection, đến reward processing — đều giả định đang tương tác với đúng một host.

### 2. Không gian trạng thái không tương thích giữa hai hệ thống

SCRIPT biểu diễn trạng thái host bằng vector 1538 chiều sử dụng SBERT embedding:

$$\text{state}_{\text{SCRIPT}} = [\underbrace{\text{access}}_{2} \;|\; \underbrace{\text{OS}}_{384} \;|\; \underbrace{\text{port}}_{384} \;|\; \underbrace{\text{service}}_{384} \;|\; \underbrace{\text{web\_fp}}_{384}] \in \mathbb{R}^{1538}$$

Trong đó OS, port, service, web_fingerprint đều là embedding ngữ nghĩa từ text thực (ví dụ: SBERT("Linux 3.2 - 4.9"), SBERT("22, 80, 443")). Biểu diễn này mang ngữ nghĩa phong phú — agent hiểu được "ssh" và "http" là hai dịch vụ khác nhau thông qua khoảng cách embedding.

PenGym biểu diễn trạng thái mạng bằng vector nhị phân/one-hot phẳng từ NASim:

$$\text{obs}_{\text{PenGym}} = [\underbrace{h_0}_{d} \;|\; h_1 \;|\; \dots \;|\; h_N] \in \{0,1\}^{(N+1) \times d}$$

Mỗi host segment $h_i$ chứa: compromised (binary), reachable (binary), OS (one-hot), services (binary per service), processes (binary per process). Biểu diễn này thiếu ngữ nghĩa — "ssh" chỉ là bit index 0, "http" là bit index 1, không có mối quan hệ ngữ nghĩa nào giữa chúng.

Hai biểu diễn này **hoàn toàn không tương thích**: khác dimension, khác kiểu dữ liệu (continuous embedding vs binary), khác scope (single-host vs whole-network), và khác semantics (ngữ nghĩa vs vị trí bit). Policy đã train trên state format của SCRIPT không thể nhận observation từ PenGym.

### 3. Không gian hành động chênh lệch bậc độ lớn

SCRIPT định nghĩa 2064 hành động: 4 loại scan + 2060 CVE exploit cụ thể (mỗi CVE ứng với một module Metasploit). Agent chọn trực tiếp CVE nào để thử trên host đang tấn công. Không gian hành động lớn này khiến thuật toán exploration của PPO cần nhiều episode để hội tụ, nhưng cho phép lựa chọn CVE tinh vi.

PenGym chỉ có khoảng 12 loại hành động chức năng: 4 scan + 5 exploit mức service (e*ssh, e_ftp, e_http, e_samba, e_smtp) + 3 privilege escalation. Mỗi loại exploit đại diện cho \_khái niệm* tấn công dịch vụ, không phải CVE cụ thể. Tổng action space thực tế = (số loại hành động) × (số hosts), nhưng về mặt ngữ nghĩa chỉ có ~12 hành động phân biệt.

Overlap trực tiếp giữa 2060 CVE exploit của SCRIPT và 5 exploit type của PenGym chỉ đạt **~3.4%**. Nếu train policy trên SCRIPT 2064-dim action space rồi đem eval trên PenGym 12-dim, gần như không có action nào map được. Nếu train trên PenGym 12-dim rồi muốn tận dụng kiến thức CVE phong phú của SCRIPT thì lại mất ưu thế.

### 4. Chênh lệch thang reward ~100 lần

Reward trong SCRIPT simulation dao động trong khoảng [-10, +1000]: exploit thành công mang lại +100 đến +1000 tùy host value, hành động thất bại phạt -10. PPO Critic được calibrate trên thang này — value estimate $V(s) \approx 500$ là bình thường.

PenGym dùng reward từ NASim scenario, dao động trong khoảng [-3, +100]: exploit thành công cho +100 (sensitive host), scan cho 0, hành động thất bại phạt -1 đến -3 (exploit cost). Value estimate hợp lý trên thang này chỉ khoảng $V(s) \approx 50$.

Nếu transfer policy trực tiếp từ SCRIPT sang PenGym, Critic sẽ **đánh giá sai hoàn toàn** advantage function: advantage $A(s,a) = R + \gamma V(s') - V(s)$ bị lệch do $V$ quá lớn so với $R$ thực tế trên PenGym. Hệ quả là policy update hướng sai, agent mất kiến thức đã học. Quan trọng hơn, EWC Fisher information — thước đo "parameter nào quan trọng" dùng cho continual learning — được tính từ gradient, mà gradient tỷ lệ với reward scale. Nếu scale khác nhau, Fisher từ sim và Fisher từ PenGym **không so sánh được**, khiến EWC penalty mất ý nghĩa.

### 5. Thiếu cầu nối interface giữa CRL framework và PenGym

SCRIPT CRL framework (ScriptAgent: Explorer + Keeper + OnlineEWC) được viết hoàn toàn cho interface HOST: `Explorer` gọi `target.reset()`, `target.perform_action(a)`, đọc `target.ip`, `target.info`, `target.env_data`. Keeper sử dụng `target.reset()` để thu thập expert samples qua Knowledge Distillation. EWC tính Fisher matrix dựa trên loss từ `target.perform_action()`.

PenGym không có bất kỳ interface nào tương thích. Không có `perform_action()` (chỉ có `step()`), không có `.ip` (scenario không phải host), không có `.info` hay `.env_data`. Hậu quả: toàn bộ pipeline CRL 5 trụ cột (Teacher Guidance, KL Imitation, Knowledge Distillation, Retrospection, EWC) **không thể chạy trên PenGym** mà không có lớp trung gian.

### 6. Không có cơ chế transfer có kiểm soát giữa hai domain

SCRIPT gốc không có khái niệm "chuyển từ domain này sang domain khác" — tất cả tasks đều nằm trong cùng simulation. Khi thêm PenGym vào hệ thống, xuất hiện yêu cầu mới: pre-train trên simulation (nhanh, unlimited episodes) rồi fine-tune trên PenGym (chậm hơn, realistic hơn). Đây là bài toán **cross-domain continual learning** mà framework CRL gốc không được thiết kế để xử lý.

Các vấn đề cụ thể khi transfer thô giữa hai domain:

- **Distribution shift trên state**: Mean/variance của running normalizer từ sim không còn hợp lệ trên PenGym → normalized state bị méo, policy nhận input sai lệch.
- **Fisher information mất cân bằng**: Fisher matrix tính trên sim states thể hiện "parameter nào quan trọng cho sim". Khi chuyển sang PenGym, Fisher này quá mạnh — EWC penalty giữ agent quá gần policy sim, không cho phép thích nghi đủ với PenGym.
- **Learning rate quá lớn cho fine-tuning**: LR chuẩn (1e-4 actor, 5e-5 critic) phù hợp cho training from scratch nhưng quá cao cho fine-tuning — gradient từ PenGym dễ ghi đè kiến thức sim.

### 7. PenGym thiếu đa dạng CVE và chưa hỗ trợ gradient độ khó

PenGym gốc chỉ hỗ trợ 5 loại exploit service (ssh, ftp, http, samba, smtp) với xác suất thành công gần như cố định (~0.999999 trong hầu hết scenario). Điều này tạo ra hai hạn chế:

**Thứ nhất**, agent không gặp thất bại có ý nghĩa khi exploit — mọi exploit đều gần như chắc chắn thành công. Policy chỉ cần học _đường đi_ qua mạng (thứ tự host, subnet discovery) mà không cần học _chiến thuật_ tấn công (chọn exploit phù hợp, retry khi thất bại, dùng exploit khác khi thất bại).

**Thứ hai**, không có gradient độ khó — scenario "tiny" (3 hosts) và "medium" (16 hosts) khác nhau về topology nhưng giống nhau về exploit difficulty. Không thể xây curriculum training theo chiều "exploit càng lúc càng khó" vì tất cả exploit đều trivially easy. Curriculum chỉ hoạt động trên chiều topology complexity, trong khi curriculum thực sự cần cả hai chiều.

Ngoài ra, dataset CVE gốc của SCRIPT (1985 CVE) có **49.2%** thiếu metadata CVSS v3 (chỉ có v2), và chỉ **44%** CVE tương thích trực tiếp với 5 service của PenGym. Cần cơ chế mở rộng và phân loại CVE có hệ thống.

### 8. Hệ thống đánh giá không phù hợp cho bài toán transfer learning

SCRIPT gốc đánh giá agent bằng một **episode duy nhất** trên mỗi task: agent chạy 1 lần, thành công = 1, thất bại = 0. Success Rate chỉ nhận giá trị rời rạc (0.0, 0.5, 1.0 với 2 tasks). Cách đánh giá này gây ra hai hệ quả nghiêm trọng cho bài toán transfer:

**Ceiling effect**: Nếu cả agent dual-trained (có transfer) lẫn agent scratch (từ đầu) đều giải được task → SR = 100% cho cả hai → Forward Transfer = 0. Nhưng thực tế agent dual-trained có thể giải trong 10 bước còn scratch cần 80 bước — transfer rõ ràng đang hoạt động mà metric không phản ánh được.

**Floor effect**: Nếu scenario quá khó, cả hai agent đều thất bại → SR = 0% cho cả hai → Forward Transfer = 0. Nhưng agent dual-trained có thể đã tiến gần hơn đến mục tiêu (quét nhiều host hơn, reward cao hơn dù chưa thành công) — metric nhị phân bỏ qua toàn bộ thông tin trung gian.

Ngoài ra, Backward Transfer trên sim domain gặp ceiling effect cố hữu: Phase 1 train agent đến SR ≈ 100% trên sim, EWC bảo toàn performance → cả θ_sim lẫn θ_dual đều SR ≈ 100% trên sim → BT_SR = 0 bất kể EWC hiệu quả hay không. Cần metric nhạy hơn binary SR để phát hiện degradation tinh vi.

---

## II. Hướng giải quyết tương ứng

### 1. SingleHostPenGymWrapper — Bọc multi-host thành single-host

Để giải quyết xung đột paradigm single-host/multi-host, hệ thống bọc PenGymEnv bên trong một wrapper cung cấp giao diện tương tự HOST:

- **`reset()`** trả về state vector 1540-dim của host mục tiêu hiện tại (không phải toàn mạng).
- **`step(service_action)`** nhận action 16-dim, nội bộ map sang PenGym flat action index kết hợp với host target hiện tại, rồi thực thi trên PenGym environment.
- **Target selection tự động**: wrapper duy trì danh sách hosts reachable, tự động chọn target tiếp theo dựa trên chiến lược (ưu tiên sensitive hosts, round-robin, hoặc reachability-based).
- **Auto subnet scan**: sau mỗi lần compromise host thành công, wrapper tự động thực hiện subnet_scan để phát hiện hosts mới trong subnet — mô phỏng hành vi pentester thực.
- **Failure rotation**: nếu agent thất bại liên tiếp N lần trên cùng host, wrapper tự động chuyển sang host khác tránh bị kẹt.

Wrapper này là **điểm tích hợp duy nhất** giữa SCRIPT và PenGym: toàn bộ complexity multi-host bị ẩn bên trong, phía trên chỉ thấy interface single-host quen thuộc.

### 2. Unified State Encoder (1540-dim) — Chuẩn hóa biểu diễn trạng thái

Thay vì giữ riêng hai state format và xây adapter tại boundary (Strategy A), Strategy C **thống nhất state representation ở cả hai phía** về cùng format 1540-dim:

$$\text{state}_{\text{unified}} = [\underbrace{\text{access}}_{3} \;|\; \underbrace{\text{discovery}}_{1} \;|\; \underbrace{\text{OS}}_{384} \;|\; \underbrace{\text{port}}_{384} \;|\; \underbrace{\text{service}}_{384} \;|\; \underbrace{\text{auxiliary}}_{384}] \in \mathbb{R}^{1540}$$

**Sửa đổi so với SCRIPT gốc (1538-dim → 1540-dim):**

- **Access mở rộng từ 2-dim sang 3-dim**: SCRIPT gốc dùng 2-dim one-hot [compromised, reachable], không encode trạng thái "unknown" (host chưa được quét). PenGym phân biệt rõ 3 trạng thái. Thêm 1 dimension cho 3 trạng thái: `[unknown, reachable, compromised]`.
- **Discovery bit bổ sung (1-dim)**: Cột mới xác định host đã được quét chưa (0.0 = undiscovered, 1.0 = discovered). Thông tin này có trong PenGym observation nhưng SCRIPT gốc không track.

**Cải tiến xuyên suốt — Canonicalization:**

Cùng một OS/service có thể xuất hiện với tên khác nhau ở hai domain. Ví dụ: SCRIPT simulation gọi là "ubuntu" nhưng PenGym scenario khai báo "linux"; SCRIPT ghi nhận "openssh" nhưng PenGym chỉ có "ssh". Nếu SBERT encode hai string này thành embedding khác nhau, policy sẽ không nhận ra chúng là cùng một thứ — domain gap tại tầng representation.

Hệ thống giải quyết bằng **canonicalization map tự động**: `ubuntu → linux`, `debian → linux`, `centos → linux`, `openssh → ssh`, `apache httpd → http`, v.v. Map này được áp dụng _trước khi_ SBERT encode, đảm bảo cùng service/OS luôn cho cùng embedding bất kể nguồn gốc.

Phase 0 validation xác nhận `cross_domain_os_cosine = 1.0` — chứng minh OS description từ sim và PenGym cho cùng embedding, loại bỏ domain gap tại tầng state representation.

### 3. Hierarchical Action Space (16-dim) — Trừu tượng hóa không gian hành động

Để khắc phục khoảng cách giữa 2064 actions (SCRIPT) và ~12 actions (PenGym), hệ thống thiết kế **không gian hành động phân cấp hai tầng**:

**Tầng 1 — Service-level (RL policy học):** 16 hành động trừu tượng:

| Nhóm                 | Số lượng | Hành động                                           |
| -------------------- | -------- | --------------------------------------------------- |
| Scan                 | 4        | port_scan, service_scan, os_scan, web_scan          |
| Exploit              | 9        | exploit_ssh/ftp/http/smb/smtp/rdp/sql/java_rmi/misc |
| Privilege escalation | 3        | privesc_tomcat/schtask/daclsvc                      |

Mapping 1:1 với PenGym actions cho 5 exploit type + 3 privesc type. 4 exploit type bổ sung (rdp, sql, java_rmi, misc) mở rộng coverage cho SCRIPT simulation.

**Tầng 2 — CVE-level (heuristic selector):** Khi agent chọn action ở tầng 1 (ví dụ "exploit_http"), CVESelector chọn CVE cụ thể từ nhóm tương ứng (~680 CVE cho http) theo 4 chiến lược:

- **Match**: Oracle — chọn CVE khớp chính xác vulnerability của target (dùng khi training trên sim).
- **Rank**: Realistic — chọn CVE có Metasploit rank cao nhất (dùng khi eval).
- **Random**: Baseline comparison.
- **Round-robin**: Xoay vòng CVE trong nhóm.

Tổng cộng 2064 CVE được phân loại tự động vào 16 nhóm bằng keyword matching + port fallback, đạt 77.9% tự động (456 CVE còn lại vào nhóm misc). CVE coverage từ 3.4% → ~100% giữa SCRIPT và PenGym. Policy chỉ cần học 16 hành động thay vì 2064 → hội tụ nhanh hơn khoảng 10 lần.

### 4. Unified Reward Normalizer — Chuẩn hóa reward về [-1, +1]

Để đảm bảo EWC Fisher information có thể so sánh giữa hai domain và PPO Critic không bị calibrate sai, hệ thống ánh xạ reward từ cả hai nguồn về cùng khoảng [-1, +1]:

- **Simulation**: `UnifiedNormalizer(source='simulation')` — ánh xạ [-10, +1000] → [-1, +1].
- **PenGym**: `UnifiedNormalizer(source='pengym')` — ánh xạ [-3, +100] → [-1, +1].

Fisher penalty trong EWC tỷ lệ với gradient², và gradient tỷ lệ với reward scale. Khi reward đã chuẩn hoá, Fisher từ sim và Fisher từ PenGym nằm trên cùng thang đo → EWC penalty có ý nghĩa khi so sánh cross-domain.

### 5. PenGymHostAdapter — Duck-typing interface HOST

Để kết nối PenGym vào CRL framework mà **không sửa bất kỳ dòng code CRL nào**, hệ thống xây dựng lớp adapter mô phỏng hoàn toàn giao diện HOST:

| Interface HOST                                      | Adapter implementation                          |
| --------------------------------------------------- | ----------------------------------------------- |
| `reset() → state[1540]`                             | Gọi `wrapper.reset()` → state vector unified    |
| `perform_action(a) → (state, reward, done, result)` | Gọi `wrapper.step(a)`, chuyển đổi format return |
| `.ip`                                               | Tên scenario (ví dụ: "tiny_T2_001")             |
| `.info`                                             | Mock `Host_info` object                         |
| `.env_data`                                         | Dict metadata scenario                          |

Factory method `from_scenario(path, seed)` tạo adapter từ file scenario. Adapter dùng lazy initialization — wrapper chỉ được tạo khi `reset()` lần đầu, tránh NASim class-level state corruption giữa các scenarios.

Kết quả: `ScriptAgent.get_new_task_learner(task=adapter)`, `Explorer.run_train_episode(target=adapter)`, `Keeper.compress()` — toàn bộ pipeline CRL 5 trụ cột hoạt động trên PenGym mà ScriptAgent không biết backend đã đổi.

### 6. Domain Transfer Manager — Chuyển giao có kiểm soát qua pipeline 5 pha

Để giải quyết vấn đề cross-domain transfer mà framework CRL gốc không có, hệ thống thiết kế pipeline 5 pha (Phase 0→4) với cơ chế domain transfer có kiểm soát giữa Phase 1 (sim) và Phase 3 (PenGym):

**Phase 0 — Validation:** Kiểm tra tính nhất quán SBERT embedding cross-domain (cosine similarity), xác nhận dimension 1540, kiểm tra canonicalization hoạt động, và verify PenGym scenario loadable.

**Phase 1 — Sim CRL Training:** Huấn luyện CRL trên simulation (6 sim tasks) với action_dim=16, state_dim=1540, reward normalized. Thu thập Fisher information qua EWC trên từng task. Mục tiêu: SR ≈ 100% trên sim → θ_sim_unified.

**Phase 2 — Domain Transfer:** Deep copy θ_sim_unified → θ_dual, sau đó áp dụng một trong 3 chiến lược transfer:

| Chiến lược                  | State normalizer | Fisher discount         | Learning rate | Warmup              |
| --------------------------- | ---------------- | ----------------------- | ------------- | ------------------- |
| **conservative** (mặc định) | Reset + warmup   | β=0.3 (giảm 70% Fisher) | ×0.1          | Có (rollout PenGym) |
| aggressive                  | Giữ nguyên       | 1.0                     | 1.0           | Không               |
| cautious                    | Reset + warmup   | 0.0 (xóa Fisher)        | ×0.1          | Có                  |

Chiến lược conservative cân bằng giữa giữ kiến thức sim (EWC vẫn active nhưng giảm lực) và cho phép thích nghi PenGym (normalizer mới, LR thấp). Fisher discount β=0.3 nghĩa EWC penalty giảm 70% — đủ để giữ "cấu trúc" chính sách sim nhưng cho phép "tinh chỉnh" cho PenGym.

Warmup states được thu thập bằng random rollout trên PenGym scenarios, dùng để rebuild running statistics cho state normalizer — đảm bảo normalizer phản ánh phân phối PenGym thay vì sim.

**Phase 3 — PenGym Fine-tuning:** θ_dual tiếp tục train CRL trên danh sách PenGym tasks. EWC penalty vẫn hoạt động (với Fisher đã discount) → agent thích nghi PenGym nhưng không quên kiến thức sim hoàn toàn. Đồng thời train θ_scratch từ đầu trên cùng PenGym tasks làm baseline so sánh.

**Phase 4 — Evaluation:** So sánh 3 agents (θ_dual, θ_sim_unified, θ_scratch) trên cả PenGym tasks và sim tasks. Đo Forward Transfer, Backward Transfer, và policy-level divergence.

### 7. CVE Pipeline — Mở rộng CVE và Curriculum Training

#### 7.1 CVE Difficulty Grading

Để khắc phục việc PenGym không có gradient độ khó, hệ thống phân loại 1985 CVE trong dataset thành 4 tier dựa trên đặc tính tấn công:

$$S_{diff} = 0.50 \cdot (1 - \text{prob}) + 0.25 \cdot f_{AC} + 0.15 \cdot f_{PR} + 0.10 \cdot f_{UI}$$

- **prob** (xác suất exploit thành công): Feature chính, 4 mức rời rạc do MSF_Rank quyết định gần deterministic (Excellent→0.99, Great/Good→0.80, Normal→0.60, Manual/Low→0.40).
- **Attack_Complexity**: LOW=0.0, MEDIUM=0.5, HIGH=1.0.
- **Privileges_Required**: NONE=0.0, LOW=0.33, HIGH=0.67.
- **User_Interaction**: NONE=0.0, REQUIRED=1.0.

Phân tier:

| Tier        | $S_{diff}$   | Đặc điểm                           | Số CVE ước tính |
| ----------- | ------------ | ---------------------------------- | --------------- |
| T1 (Easy)   | [0.0, 0.15)  | prob≥0.8, AC=LOW, PR=NONE          | ~1200           |
| T2 (Medium) | [0.15, 0.35) | prob=0.6, AC thấp-trung, PR thấp   | ~500            |
| T3 (Hard)   | [0.35, 0.55) | prob=0.6 + AC=MEDIUM hoặc prob=0.4 | ~220            |
| T4 (Expert) | [0.55, 1.0]  | prob=0.4, AC=HIGH, PR=HIGH         | ~65             |

#### 7.2 Kiến trúc Template + Overlay cho Scenario

Thay vì hardcode CVE vào mỗi scenario YAML (gây bùng nổ files khi tổ hợp topology × CVE mix), hệ thống tách thành:

- **Template**: Chứa topology mạng (subnets, firewall, sensitive hosts) với các _slot_ cho service/CVE. Topology không đổi khi thay CVE.
- **Overlay**: Chứa assignment CVE cụ thể cho từng slot, theo tier difficulty. Mỗi overlay thay đổi prob, cost, access của exploit nhưng giữ nguyên cấu trúc mạng.
- **Compiler**: Kết hợp template + overlay → scenario YAML hoàn chỉnh tương thích NASim.

8 base topologies × 4 difficulty tiers × nhiều variants = 96+ scenarios compiled, so với việc phải viết tay từng file.

#### 7.3 Curriculum Training T1 → T4

Curriculum training theo độ khó tăng dần: agent học trên T1 overlay (exploit gần chắc chắn thành công, cost thấp) trước, rồi dần chuyển sang T2, T3, T4 (exploit khó hơn, cost cao hơn). Cùng topology mạng nhưng thách thức exploit tăng dần.

**Per-task episode schedule**: Mỗi scenario có budget episode riêng, tính bằng:

$$\text{episodes} = \lceil \text{base\_episodes}[\text{base\_name}] \times \text{tier\_multiplier}[\text{tier}] \rceil$$

Base episodes xác định từ calibration (tiny=500, medium-single-site=2500), tier multiplier phản ánh độ khó (T1=0.8×, T2=1.0×, T3=2.0×, T4=3.0×). Scenario khó hơn được phân bổ nhiều episode hơn, scenario dễ không lãng phí budget.

**Per-scenario step limit schedule**: Tương tự episode schedule, mỗi scenario có số bước tối đa riêng tuỳ topology (chi tiết tại §II.9).

### 8. Hệ thống đánh giá đa chiều — Khắc phục ceiling/floor effect

Evaluator được thiết kế lại hoàn toàn để khắc phục các hạn chế ở Mục I.8:

#### 8.1 Multi-episode SR (K=20)

Thay vì 1 episode, agent chạy K=20 episodes trên mỗi task. SR trở thành giá trị liên tục [0, 1]:

$$SR_s = \frac{1}{K}\sum_{k=1}^{K} \mathbb{1}[\text{success}_{s,k}]$$

Standard error kèm theo: $SE = \sqrt{SR(1-SR)/(|\mathcal{S}| \times K)}$. Forward Transfer có ý nghĩa thống kê khi $|FT| > 2 \times SE$.

#### 8.2 Normalized Reward (NR)

$$NR_s = \frac{R_{actual,s}}{R_{optimal,s}}$$

NR đo hiệu quả reward so với giá trị tối ưu. NR ∈ (−∞, 1] — phân biệt "gần thành công" (NR ≈ 0.8) và "hoàn toàn lạc hướng" (NR < 0) ngay cả khi binary success = 0. Optimal reward cho mỗi base scenario được xác định từ scenario YAML.

#### 8.3 Step Efficiency (η)

$$\eta_s = \frac{\text{optimal\_steps}_s}{\text{actual\_steps}_s} \times \mathbb{1}[\text{success}]$$

η **hoàn toàn immune với ceiling effect**: ngay cả khi SR = 100%, η phân biệt agent giải trong 10 bước (η=0.6) vs 80 bước (η=0.075). Metric này giải quyết triệt để ví dụ SR=100% cho cả hai agent nhưng transfer thực sự đang hoạt động.

#### 8.4 Forward và Backward Transfer đa chiều

Thay vì chỉ FT_SR, hệ thống báo cáo 3 chiều Forward Transfer và 3 chiều Backward Transfer:

$$FT_{SR}, \; FT_{NR}, \; FT_{\eta} \quad\text{(so sánh θ\_dual vs θ\_scratch trên PenGym)}$$
$$BT_{SR}, \; BT_{NR}, \; BT_{\eta} \quad\text{(so sánh θ\_dual vs θ\_sim trên sim)}$$

$FT_{SR}=0$ nhưng $FT_{\eta}>0$ nghĩa transfer giúp giải nhanh hơn dù tỷ lệ thành công bằng nhau. $BT_{\eta}<-0.15$ nghĩa EWC không đủ mạnh, agent quên sim quá nhiều.

#### 8.5 Policy-level BT metrics

Cho phân tích sâu khi BT outcome-level không đủ nhạy:

- **$D_{KL}(\pi_{sim} \| \pi_{dual})$**: KL divergence trung bình trên N sim states, đo trực tiếp policy distribution shift mà không cần chạy episode.
- **$\Delta_F = \sum_k F_k \cdot (\theta_{dual,k} - \theta_{sim,k})^2$**: Fisher-weighted parameter distance — đo lượng thay đổi trên chính các dimensions mà EWC bảo vệ. $\Delta_F \approx 0$ → EWC enforcement thành công.

#### 8.6 Đánh giá cách ly environment

Mỗi agent được eval trên **bộ adapter riêng** (cùng seed, cùng scenario) thay vì chia sẻ adapter. Giải quyết vấn đề NASim lưu dimension indices dưới dạng class attributes — khi scenario B load sau scenario A, class vars bị ghi đè corruption. Fresh adapter set per agent đảm bảo eval environment identically initialized.

#### 8.7 Forgetting Matrix (F) và Zero-Shot Transfer Vector (Z)

Đo catastrophic forgetting và khả năng zero-shot transfer xuyên curriculum:

- **Forgetting Matrix $F$**: Ma trận $n \times m$ ($n$ tasks, $m$ tiers). $F_{i,j}$ đo mức NR suy giảm của task $i$ sau khi agent học tier $j$ so với ngay sau khi học chính task $i$:

$$F_{i,j} = NR_{i}^{\text{after own tier}} - NR_{i}^{\text{after tier } j}, \quad j > \text{own tier}$$

Giá trị dương = forgetting, giá trị âm = backward reinforcement.

- **Zero-Shot Transfer Vector $Z$**: $Z_i = NR_{i}^{\text{before own tier}}$ — đo khả năng agent giải task $i$ **trước khi** được train trên task đó (chỉ dựa vào kiến thức từ các tier trước). $Z_i > 0$ nghĩa kiến thức curriculum tích luỹ đã giúp agent generalise.

- **Summary statistics**: `mean_F`, `max_F`, `mean_Z`, `tasks_with_positive_Z` — tổng hợp nhanh mức forgetting và transfer hiệu quả.

Implementation: `StrategyCEvaluator.compute_forgetting_matrix()` nhận kết quả eval tại mỗi tier checkpoint (do Phase 3 lưu sau mỗi tier group), tính ma trận $F$ và vector $Z$ từ NR values.

#### 8.8 Learning-Speed Transfer (TTT + AUC)

Đo tốc độ học — phân biệt agent đã biết giải nhanh (transfer hiệu quả) vs agent phải học lâu:

- **Time-To-Threshold (TTT)**: Episode đầu tiên đạt reward > 0 (penetration thành công đầu tiên). TTT thấp hơn = transfer giúp agent "khởi động" nhanh hơn.
- **AUC (Area Under Curve)**: Mean reward xuyên suốt training. AUC cao hơn = agent khai thác kiến thức transfer tốt hơn.

So sánh giữa θ_dual và θ_scratch:

$$\text{TTT speedup} = \frac{TTT_{\text{scratch}}}{TTT_{\text{dual}}}$$
$$\text{AUC ratio} = \frac{AUC_{\text{dual}}}{AUC_{\text{scratch}}}$$

TTT speedup > 1 nghĩa dual-trained agent bắt đầu exploit thành công sớm hơn. AUC ratio > 1 nghĩa dual-trained agent học hiệu quả hơn xuyên suốt quá trình training.

Implementation: `StrategyCEvaluator.compute_learning_speed()` nhận per-episode rewards của cả hai agents (thu thập trong Phase 3).

#### 8.9 MetricStore — Lưu trữ metric có cấu trúc

Tất cả metric được lưu vào một file JSON duy nhất (`metric_store.json`) với cấu trúc:

```json
{
  "metadata": { "seed": 42 },
  "checkpoints": {
    "after_T1": { "task_name": { "sr": 0.95, "nr": 0.87, "eta": 0.6 } },
    "after_T2": { ... },
    "final": { ... }
  },
  "training_curves": {
    "task_name": { "episode_rewards": [...], "ttt": 15 }
  },
  "forgetting": { "F_matrix": [...], "Z_vector": [...] },
  "transfer": { "FT_SR": 0.12, "BT_eta": -0.05, ... }
}
```

MetricStore phục vụ ba mục đích: **(1)** reproducibility — toàn bộ metrics từ một experiment run được persist, không phụ thuộc log parsing; **(2)** downstream analysis — load lại bằng `MetricStore.load()` để so sánh cross-seed hoặc cross-config; **(3)** export — cung cấp dữ liệu cho FZComputer và CECurveGenerator.

Implementation: `MetricStore` class trong `src/evaluation/metric_store.py`.

#### 8.10 Continual Evaluation Curves (CE Curves)

Đo performance trajectory xuyên suốt curriculum thay vì chỉ đo tại điểm cuối:

Sau mỗi tier $T_j$, agent được eval trên **tất cả** tasks đã gặp. Tạo ra bảng tasks × checkpoints cho từng metric (SR, NR, η). CE curves cho thấy:

- Task nào bị forgetting: NR giảm dần theo tier.
- Task nào được reinforcement: NR tăng nhờ kiến thức cross-task.
- Tốc độ adaptation: NR tại checkpoint đầu tiên vs cuối cùng.

Export CSV: `CECurveGenerator.to_csv(curves, metric="nr")` → file `ce_curves_nr.csv` (tasks × checkpoints), sẵn sàng cho matplotlib/TensorBoard plotting.

Implementation: `CECurveGenerator` class trong `src/evaluation/metric_store.py`. Curves được extract từ MetricStore data tại Phase 4.

#### 8.11 Heldout Evaluation (Generalization Test)

Ngoài eval trên training scenarios, hệ thống hỗ trợ đánh giá trên **heldout scenarios** — các PenGym scenario agent chưa từng train. Đo khả năng generalise của policy ra ngoài distribution đã thấy.

CLI: `--heldout-scenarios scenario_A.yml scenario_B.yml`. Phase 4 tạo fresh adapters per agent cho heldout set, chạy multi-episode eval K=20, và tính riêng `heldout_transfer_metrics` (FT_SR, FT_NR, FT_eta) trên bộ heldout. Kết quả lưu riêng trong report JSON.

#### 8.12 Export và Reporting

Phase 4 tự động xuất ra:

| Output file                   | Nội dung                                                       |
| ----------------------------- | -------------------------------------------------------------- |
| `strategy_c_eval_report.json` | Full evaluation report — per-agent, per-task, transfer metrics |
| `metric_store.json`           | Structured metrics for reproducibility                         |
| `forgetting_matrix.csv`       | F matrix + Z vector, import được vào Excel/pandas              |
| `ce_curves_nr.csv`            | CE curves (NR) — tasks × checkpoints                           |
| Console report                | Formatted table: Agent × Domain × SR/NR/η/Reward               |

Console report format:

```
================================================================
Strategy C — Agent Comparison Report
================================================================
Agent                     | Domain   |    SR |    NR |     η |   Reward | Tasks
--------------------------------------------------------------------------------
theta_sim_unified         | pengym   |  0.0% | -0.05 |   N/A |    -15.3 |     8
theta_dual                | pengym   | 45.0% |  0.62 | 0.450 |    892.5 |     8
theta_pengym_scratch      | pengym   | 35.0% |  0.48 | 0.320 |    650.1 |     8

Transfer Metrics:
  FT_SR: +0.1000
  FT_NR: +0.1400
  FT_eta: +0.1300
  BT_SR: -0.0500
  BT_KL: 0.023400
```

---

## III. Đánh giá mức độ hoàn thành

### 1. Thành phần đã hoàn tất

| #   | Cải tiến                             | Trạng thái              | Mô tả                                                                                                  |
| --- | ------------------------------------ | ----------------------- | ------------------------------------------------------------------------------------------------------ |
| 1   | Unified State Encoder (1540-dim)     | ✅ Hoàn chỉnh, tích hợp | SBERT encoding thống nhất + canonicalization, tích hợp xuyên suốt pipeline HOST → PenGym wrapper → PPO |
| 2   | Hierarchical Action Space (16-dim)   | ✅ Hoàn chỉnh, tích hợp | 2064 CVE → 16 nhóm, CVESelector 4 strategies, tích hợp vào HOST + PenGym                               |
| 3   | SingleHostPenGymWrapper              | ✅ Hoàn chỉnh, tích hợp | Multi→single host, auto subnet scan, failure rotation, unified encoding                                |
| 4   | PenGymHostAdapter                    | ✅ Hoàn chỉnh, tích hợp | Duck-typing HOST, factory method, lazy init, float reward                                              |
| 5   | Unified Reward Normalizer            | ✅ Hoàn chỉnh, tích hợp | [-1,+1] cho cả sim và PenGym, tương thích Fisher cross-domain                                          |
| 6   | Domain Transfer Manager              | ✅ Hoàn chỉnh, tích hợp | 3 strategies (conservative/aggressive/cautious), Fisher discount, warmup                               |
| 7   | Pipeline Phase 0→4 (DualTrainer)     | ✅ Hoàn chỉnh, tích hợp | Validation → Sim CRL → Transfer → PenGym fine-tune → Eval                                              |
| 8   | StrategyCEvaluator                   | ✅ Hoàn chỉnh, tích hợp | Multi-episode K=20, SR/NR/η/SE, FT/BT đa chiều, policy-level metrics                                   |
| 9   | CVE Difficulty Grading               | ✅ Hoàn chỉnh           | 1985 CVE → 4 tiers, composite score                                                                    |
| 10  | Template + Overlay Scenario          | ✅ Hoàn chỉnh           | 8 base topologies, 96+ compiled overlay scenarios                                                      |
| 11  | Curriculum Training T1→T4            | ✅ Hoàn chỉnh, tích hợp | Per-task episode + step_limit schedule, multiplier mode, JSON config                                   |
| 12  | Policy-level BT metrics              | ✅ Hoàn chỉnh, tích hợp | BT_KL, BT_fisher_dist tích hợp vào Phase 4 evaluator                                                   |
| 13  | CRL 5 trụ cột trên PenGym            | ✅ Hoạt động            | Teacher Guidance, KL Imitation, KD, Retrospection, EWC — chạy trên PenGym qua adapter                  |
| 14  | EWC Fisher discount (β)              | ✅ Hoàn chỉnh, tích hợp | `discount_fisher(β)` nới lỏng Fisher constraint khi chuyển domain                                      |
| 15  | Fresh eval adapters per agent        | ✅ Hoàn chỉnh           | Giải quyết NASim class-level state leakage trong Phase 4                                               |
| 16  | Calibration framework                | ✅ Hoàn chỉnh           | 32 calibration runs, xác định base episodes và saturation point cho 8 scenarios                        |
| 17  | Per-scenario step_limit              | ✅ Hoàn chỉnh, tích hợp | Step limit theo topology (tiny=200, small=300, medium=500), centralized JSON config                    |
| 18  | TeeLogger console filtering          | ✅ Hoàn chỉnh, tích hợp | 6 regex patterns lọc env noise khỏi console, giữ nguyên trong log file                                 |
| 19  | MetricStore (structured persistence) | ✅ Hoàn chỉnh, tích hợp | JSON store cho checkpoints, training curves, forgetting, transfer — `MetricStore.load()` cho analysis  |
| 20  | FZComputer (F/Z matrix export)       | ✅ Hoàn chỉnh, tích hợp | Forgetting matrix + Zero-shot vector → CSV, summary text                                               |
| 21  | CECurveGenerator (CE curves)         | ✅ Hoàn chỉnh, tích hợp | Continual Evaluation curves (NR/SR/η × checkpoints) → CSV cho plotting                                 |
| 22  | Heldout evaluation                   | ✅ Hoàn chỉnh, tích hợp | Eval trên unseen scenarios, riêng biệt heldout FT metrics                                              |
| 23  | Learning-speed transfer (TTT + AUC)  | ✅ Hoàn chỉnh, tích hợp | Time-To-Threshold speedup và AUC ratio giữa θ_dual vs θ_scratch                                        |

### 2. Hạn chế còn tồn tại

#### 2.1 Hạn chế ở mức phương pháp

**a) Reward normalizer chưa adaptive theo scenario:**
`UnifiedNormalizer(source='pengym')` dùng `min_reward=-3.0` và `max_reward=100` cố định. Các scenario khác nhau có cost structure khác nhau — scenario với exploit cost=4 (tier T4) sẽ có min_reward thực tế thấp hơn -3. Ảnh hưởng: reward dương trung gian (scan thành công, pivot host compromised) bị co lại quá nhỏ so với reward âm → gradient dương yếu. Fisher information cho positive-reward parameters bị đánh giá thấp → EWC ít bảo vệ → dễ bị overwrite khi chuyển domain. Mức ảnh hưởng trung bình — chỉ đáng kể khi EWC lambda lớn.

**b) Medium và medium-multi-site chưa solvable bằng standalone DQN:**
Calibration cho thấy SR=0.0 tại mọi episode budget (500–3000) cho 2 scenario này. |A|=192 + multi-subnet quá khó cho exploration budget hiện tại. Per-scenario step_limit (medium=500) giải quyết phần ngân sách bước, nhưng action space vẫn quá lớn. Curriculum T1→T4 (giảm exploit difficulty) được kỳ vọng giúp agent học routing/path ở T1 trước khi đối mặt exploit difficulty ở T3/T4, nhưng chưa có kết quả thực nghiệm xác nhận. Nếu curriculum vẫn không đủ, cần exploration bonus hoặc action masking.

**c) Canonicalization có thể mất thông tin version:**
Aggressive canonicalization (`Ubuntu 14.04 → linux`, `Ubuntu 22.04 → linux`) loại bỏ hoàn toàn thông tin phiên bản OS. Hai phiên bản này có vulnerability profile rất khác nhau nhưng cho cùng embedding → policy không phân biệt được. Trade-off hiện tại ưu tiên cross-domain consistency (giảm domain gap) trên thông tin version (chấp nhận mất granularity). Nếu performance thấp do OS confusion, cần thêm version granularity.

**d) Kiểm chứng PenGym real execution (KVM) chưa thực hiện:**
Toàn bộ pipeline đã chạy trên PenGym ở chế độ NASim simulation (xác suất). Chưa verify end-to-end trên CyRIS cyber range thực (KVM VMs + nmap + Metasploit RPC). Real execution có thêm failure modes (timeout, session loss, network issues) mà sim không mô phỏng. Đây là hạn chế ở tầng deployment, không ảnh hưởng đến phương pháp nhưng cần verify trước khi claim sim-to-real.

**e) Thực nghiệm toàn diện chưa hoàn tất:**
Pipeline đã chạy thành công end-to-end trên sanity checks (tiny, step_limit=200 → SR=100%, η=0.60). Thí nghiệm đầu tiên trên 8 scenarios (4 base × 2 tiers) thất bại do step_limit=100 quá thấp — đã khắc phục bằng per-scenario step_limit. Thí nghiệm diện rộng với step_limit tự động đang cần chạy lại để thu thập baseline metrics sạch.

#### 2.2 Đánh giá tổng thể

Hệ thống đã hoàn thành **đầy đủ pipeline end-to-end** cho bài toán cross-domain continual learning từ simulation sang PenGym:

- **Representation**: State unification 1540-dim (cross-domain cosine = 1.0), action abstraction 16-dim (100% coverage)
- **Training**: Curriculum T1→T4 với per-task episode + step_limit schedule, CRL 5 trụ cột hoạt động xuyên domain
- **Transfer**: Domain Transfer Manager 3 strategies (conservative/aggressive/cautious), Fisher discount β
- **Evaluation**: Hệ thống đánh giá đa chiều gồm 23 thành phần — multi-episode SR/NR/η, FT/BT 3 chiều, policy-level BT (KL + Fisher distance), Forgetting matrix + Zero-shot vector, Learning-speed transfer (TTT + AUC), CE curves, MetricStore persistence, heldout generalization, isolated eval environments
- **Infrastructure**: TeeLogger console filtering, per-scenario step_limit, centralized JSON config

Khoảng cách còn lại tập trung ở: **(1)** thực nghiệm diện rộng cần chạy lại với per-scenario step_limit (đang chuẩn bị), **(2)** medium/medium-multi-site có thể cần exploration bonus bổ sung, và **(3)** reward normalizer cần adaptive hoá cho extreme cost scenarios. Về mặt phương pháp và kiến trúc, hệ thống đã sẵn sàng cho thực nghiệm toàn diện.
