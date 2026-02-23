# Strategy C — Phân Tích Khoảng Cách & Tổng Kết Hệ Thống

> **Tác giả:** Senior Engineer / Architect Review  
> **Ngày:** 2025-02-23  
> **Phạm vi:** So sánh đặc tả Strategy C (`docs/strategy_C_shared_state_dual_training.md`, 1211 dòng) với hiện trạng mã nguồn thực tế. Xác định: đã làm đúng những gì, thiếu gì, sai lệch gì, và đề xuất cải tiến.

---

## Mục Lục

1. [Tóm Tắt Tổng Quan](#1-tóm-tắt-tổng-quan)
2. [Những Gì Đã Triển Khai Đúng](#2-những-gì-đã-triển-khai-đúng)
3. [Khoảng Cách Nghiêm Trọng (Bắt Buộc Sửa)](#3-khoảng-cách-nghiêm-trọng-bắt-buộc-sửa)
4. [Khoảng Cách Trung Bình (Nên Sửa)](#4-khoảng-cách-trung-bình-nên-sửa)
5. [Khoảng Cách Nhỏ (Có Thì Tốt)](#5-khoảng-cách-nhỏ-có-thì-tốt)
6. [Phân Tích Theo Từng Thành Phần](#6-phân-tích-theo-từng-thành-phần)
7. [Đề Xuất Cải Tiến (Xếp Hạng Ưu Tiên)](#7-đề-xuất-cải-tiến-xếp-hạng-ưu-tiên)
8. [Lộ Trình Triển Khai](#8-lộ-trình-triển-khai)
9. [Phụ Lục: Ánh Xạ File → Yêu Cầu](#9-phụ-lục-ánh-xạ-file--yêu-cầu)

---

## 1. Tóm Tắt Tổng Quan

### Kết luận: **Nền Tảng Vững Chắc, Thiếu Pipeline Chuyển Giao**

Mã nguồn hiện tại đã xây dựng thành công **lớp hạ tầng** mà Strategy C yêu cầu: cầu nối adapter/wrapper giữa SCRIPT và PenGym, không gian hành động mức service, và huấn luyện CRL trên các kịch bản PenGym. Tuy nhiên, **yếu tố cốt lõi tạo sự khác biệt** của Strategy C — bộ mã hóa trạng thái thống nhất và pipeline huấn luyện kép (Phase 0→1→2→3→4) — **hoàn toàn chưa có**.

| Hạng mục                                | Yêu cầu Strategy C                 | Trạng thái mã nguồn   |
| ---------------------------------------- | ----------------------------------- | ---------------------- |
| Wrapper PenGym đơn-host                  | Option 1 (khuyến nghị)             | ✅ Đã triển khai đầy đủ |
| Không gian hành động mức service         | Semantic grouping (khuyến nghị)     | ✅ Đã triển khai (16 chiều) |
| Mã hóa SBERT cho cả hai môi trường      | Bắt buộc                           | ✅ Đang hoạt động      |
| CRL 5 trụ cột trên PenGym               | Yêu cầu Phase 3                    | ✅ Đang hoạt động      |
| Unified State Encoder (1540 chiều)       | Yêu cầu cốt lõi                    | ❌ Chưa triển khai     |
| Chuẩn hóa OS/Service (canonicalization)  | Bắt buộc cho chuyển giao liên miền | ❌ Chưa triển khai     |
| Pipeline huấn luyện kép                  | Kiến trúc cốt lõi                  | ❌ Chưa triển khai     |
| DomainTransferManager                    | Yêu cầu Phase 2                    | ❌ Chưa triển khai     |
| Reset/khởi động chuẩn hóa trạng thái    | Thiết yếu cho chuyển giao          | ❌ Chưa triển khai     |
| Giảm Fisher cho chuyển giao liên miền    | Yêu cầu Phase 2                    | ❌ Chưa triển khai     |
| Ma trận đánh giá 4 agent                 | Yêu cầu Phase 4                    | ❌ Chưa triển khai     |
| Chuẩn hóa reward về [-1,+1]             | Khuyến nghị                        | ⚠️ Một phần (sai khoảng) |

**Kết luận cuối:** Hệ thống có thể huấn luyện agent SCRIPT CRL trên PenGym (≈ Phase 3 của Strategy C chạy đơn lẻ), nhưng không thể thực hiện **chuyển giao sim→PenGym** — đóng góp chính của Strategy C. Pipeline huấn luyện kép — giá trị cốt lõi — chưa tồn tại.

---

## 2. Những Gì Đã Triển Khai Đúng

### 2.1 SingleHostPenGymWrapper ✅

**Strategy C §3.3 khuyến nghị Option 1:** Bọc PenGym để hoạt động ở chế độ đơn-host, giữ nguyên mô hình của SCRIPT: 1 agent → 1 mục tiêu.

**Triển khai:** `src/envs/wrappers/single_host_wrapper.py` (543 dòng)

| Khía cạnh                  | Đặc tả Strategy C                | Triển khai thực tế                               | Khớp |
| -------------------------- | -------------------------------- | ------------------------------------------------ | ---- |
| Chế độ đơn-host            | Có                               | Có — `step()` thao tác trên `current_target`     | ✅   |
| Xoay vòng mục tiêu        | Duyệt qua các sensitive host     | Tự động tiến + xoay vòng khi thất bại (ngưỡng=5) | ✅+  |
| Đầu ra trạng thái          | 1540 chiều (unified)             | 1538 chiều tương thích SCRIPT (qua StateAdapter)  | ⚠️   |
| Ngữ nghĩa episode         | Chiếm 1 mục tiêu = kết thúc     | `_check_target_compromised()` → done=True         | ✅   |
| Chứa PenGym env            | Có, được bọc                     | `self._env = PenGymEnv(scenario)`                 | ✅   |

**Đánh giá:** Wrapper được thiết kế tốt, có thêm tính năng ngoài đặc tả (xoay vòng khi thất bại, khám phá subnet_scan, ưu tiên mục tiêu). Sai lệch duy nhất là số chiều trạng thái (1538 thay vì 1540).

### 2.2 PenGymStateAdapter ✅ (Về Cấu Trúc)

**Strategy C §2.2:** Mã hóa quan sát PenGym bằng SBERT để khớp định dạng SCRIPT.

**Triển khai:** `src/envs/adapters/state_adapter.py` (361 dòng)

| Khía cạnh                  | Đặc tả Strategy C                | Triển khai thực tế                  | Khớp |
| -------------------------- | -------------------------------- | ----------------------------------- | ---- |
| SBERT cho OS/Service/Port  | Có                               | Có — dùng chung encoder             | ✅   |
| Trích xuất từng host       | Có                               | Có — `_get_host_segment()`          | ✅   |
| Số chiều đầu ra cố định   | 1540                             | 1538                                | ⚠️   |
| Cache SBERT                | Khuyến nghị                      | Có — dict `_sbert_cache`            | ✅   |
| Suy luận port              | Từ service qua CONFIG.yml        | Có — `DEFAULT_SERVICE_PORT_MAP`     | ✅   |
| Web FP → Process thay thế | Option A (slot web_fp)           | Dùng thông tin process thay thế     | ✅   |

**Đánh giá:** Adapter chuyển đổi đúng quan sát PenGym sang vector SBERT. Tuân theo triết lý mã hóa Strategy C nhưng thiếu định dạng thống nhất 3 chiều access + 1 chiều discovery.

### 2.3 Không Gian Hành Động Mức Service (Semantic Grouping) ✅

**Strategy C §3.2 khuyến nghị SEMANTIC GROUPING:** Gom các exploit theo CVE thành exploit theo service.

**Triển khai:** `src/agent/actions/service_action_space.py` (516 dòng)

| Khía cạnh                    | Đặc tả Strategy C                          | Triển khai thực tế                      | Khớp |
| ---------------------------- | ------------------------------------------ | --------------------------------------- | ---- |
| Phương pháp                  | Semantic grouping                          | 16 hành động mức service                | ✅   |
| Hành động quét               | PORT_SCAN, SERVICE_SCAN, OS_SCAN, WEB_SCAN | 4 quét + SUBNET_SCAN                   | ✅+  |
| Hành động khai thác          | Theo service (ssh, ftp, http, ...)         | 9 exploit theo service                  | ✅   |
| Hành động leo thang          | pe_tomcat, pe_proftpd, pe_cron             | 3 hành động privesc                     | ✅   |
| Chọn CVE trong nhóm          | CVESelector (Tier 2)                       | Có — rank/random/round_robin/match      | ✅   |

**Đánh giá:** Hoàn toàn phù hợp với khuyến nghị Strategy C. Thiết kế 16 hành động ánh xạ 1:1 với bảng hành động thống nhất đề xuất.

### 2.4 PenGymHostAdapter (Duck-Typing) ✅

**Strategy C §5.4:** Coi mỗi host mục tiêu PenGym như một task CRL.

**Triển khai:** `src/envs/adapters/pengym_host_adapter.py` (273 dòng)

| Khía cạnh                     | Đặc tả Strategy C                | Triển khai thực tế                          | Khớp |
| ----------------------------- | -------------------------------- | ------------------------------------------- | ---- |
| Duck-typing giao diện HOST    | Có (PenGym tasks = HOST tasks)   | Composition với lazy wrapper                | ✅   |
| Mỗi host = 1 CRL task         | Có                               | Phương thức factory `from_scenario()`       | ✅   |
| Khởi tạo lười                 | Không đặc tả                     | `_ensure_wrapper()` + `_active_scenario`    | ✅+  |

### 2.5 PenGymScriptTrainer (CRL trên PenGym) ✅

**Strategy C §5.4:** Sử dụng framework CL của SCRIPT trên PenGym.

**Triển khai:** `src/training/pengym_script_trainer.py` (376 dòng)

| Khía cạnh                  | Đặc tả Strategy C                | Triển khai thực tế                      | Khớp |
| -------------------------- | -------------------------------- | --------------------------------------- | ---- |
| Đầy đủ 5 trụ cột SCRIPT   | Bắt buộc                         | Đầy đủ qua Agent_CL(method="script")   | ✅   |
| STATE_DIM/ACTION_DIM       | 1540/16                          | 1538/16                                 | ⚠️   |
| Task list = PenGym hosts   | Có                               | Có — danh sách PenGymHostAdapter        | ✅   |
| Train + evaluate + save    | Có                               | API đầy đủ train/evaluate/save/load     | ✅   |

**Đánh giá:** Đây là Phase 3 của Strategy C về mặt khái niệm, nhưng chạy đơn lẻ (không có Phase 1-2 chuyển giao trước đó). Trainer hoạt động, nhưng ở chế độ "huấn luyện từ đầu trên PenGym" chứ không phải "tinh chỉnh mô hình đã transfer từ sim".

### 2.6 Framework CRL của SCRIPT ✅ (Không thay đổi)

Toàn bộ 5 trụ cột CRL gốc của SCRIPT vẫn nguyên vẹn và hoạt động:

| Trụ cột                  | Vị trí triển khai                                           | Trạng thái |
| ------------------------ | ----------------------------------------------------------- | ---------- |
| Teacher Guidance         | `script.py` → `ScriptAgent.get_new_task_learner()` → set guide | ✅       |
| KL Imitation             | `script.py` → `ExplorePolicy._update()` → `imi_loss`       | ✅         |
| Knowledge Distillation   | `script.py` → `Keeper.compress()` → KD loss                | ✅         |
| Retrospection            | `script.py` → `Keeper.compress()` → retro_loss             | ✅         |
| EWC                      | `script.py` → class `OnlineEWC`                            | ✅         |

---

## 3. Khoảng Cách Nghiêm Trọng (Bắt Buộc Sửa)

### Gap C1: Thiếu UnifiedStateEncoder (1540 chiều)

**Strategy C §2.2 đặc tả:**
```
UnifiedStateEncoder.TOTAL_DIM = 1540
= 3 (access) + 1 (discovery) + 384 (OS) + 384 (port) + 384 (service) + 384 (aux)
```

**Hiện trạng:**
- `host.py` → `StateEncoder.state_space = 1538` (access 2 chiều, không có chiều discovery)
- `state_adapter.py` → `STATE_DIM = 1538` (cùng định dạng access 2 chiều)
- Tồn tại **hai đường mã hóa riêng biệt** — `StateEncoder` cho sim, `PenGymStateAdapter` cho PenGym
- Không có lớp duy nhất nào mã hóa cho **cả hai** môi trường

**Tác động:** Không có bộ mã hóa thống nhất, vector trạng thái từ sim và PenGym có bố cục ngữ nghĩa khác nhau. Trọng số policy huấn luyện trên sim (1538 chiều với access 2 chiều tại chỉ mục 0-1) không thể xử lý trực tiếp trạng thái PenGym, dù cả hai đều danh nghĩa 1538 chiều. 2 chiều bổ sung (access 3 chiều + discovery 1 chiều) cần thiết cho việc căn chỉnh liên miền đúng đắn.

**Nguyên nhân gốc:** Strategy A được triển khai trước (phương pháp adapter — giữ riêng hai định dạng), và mã nguồn chưa bao giờ phát triển theo hướng thống nhất của Strategy C.

### Gap C2: Thiếu Pipeline Huấn Luyện Kép (Phase 0→1→2→3→4)

**Strategy C §5.1-5.4 đặc tả pipeline 4 giai đoạn:**

| Phase | Việc cần làm                                                         | Trạng thái hiện tại |
| ----- | -------------------------------------------------------------------- | ------------------- |
| 0     | Kiểm chứng: tính nhất quán SBERT, ổn định PenGym, phân phối trạng thái | ❌ Chưa triển khai |
| 1     | Huấn luyện lại SCRIPT trên sim với mã hóa thống nhất → θ_uni          | ❌ Chưa triển khai |
| 2     | Chuyển θ_uni → PenGym (reset norm, giảm Fisher, giảm LR)             | ❌ Chưa triển khai |
| 3     | Tinh chỉnh trên PenGym với ràng buộc EWC từ Phase 1                  | ⚠️ Một phần — PenGymScriptTrainer huấn luyện CRL trên PenGym nhưng **từ đầu**, không phải từ trọng số đã transfer |
| 4     | Đánh giá 4 agent, tính toán metrics chuyển giao                       | ❌ Chưa triển khai |

**Tác động:** Đây là **toàn bộ đóng góp nghiên cứu** của Strategy C. Không có pipeline, không có thí nghiệm chuyển giao sim→PenGym, không có agent huấn luyện kép, và không có chất lượng chuyển giao đo lường được.

### Gap C3: Thiếu DomainTransferManager

**Strategy C §5.3 đặc tả:**
```python
class DomainTransferManager:
    def transfer(self, strategy='conservative'):
        # 1. Sao chép trọng số từ sim agent
        # 2. Reset chuẩn hóa trạng thái + khởi động nóng
        # 3. Giảm ma trận Fisher theo hệ số β ∈ [0.1, 0.5]
        # 4. Giảm tốc độ học ×0.1
```

**Hiện trạng:** Không có class hay logic tương đương nào tồn tại trong toàn bộ mã nguồn.

**Tác động:** Không có cơ chế chuyển giao có kiểm soát, không có cách di chuyển tri thức từ sim sang PenGym trong khi quản lý sự dịch chuyển phân phối. Ba thao tác thiết yếu (reset norm, giảm Fisher, giảm LR) đều thiếu.

### Gap C4: Thiếu Reset/Khởi Động Chuẩn Hóa Trạng Thái

**Strategy C §5.3 xác định đây là Rủi ro R3 (Tác động nghiêm trọng, Xác suất cao):**

Class `Normalization` trong `common.py` sử dụng `RunningMeanStd` (thuật toán Welford). Sau 500 episode trên sim, `running_mean` và `running_std` đã hội tụ về phân phối trạng thái sim. Khi chuyển sang PenGym:
- Nếu giữ nguyên thống kê: trạng thái PenGym bị chuẩn hóa bằng thống kê sai → đầu vào policy bị méo
- Nếu reset thống kê: trọng số policy kỳ vọng đầu vào chuẩn hóa theo phân phối sim → đầu ra ban đầu vô nghĩa

**Hiện trạng:**
- Class `Normalization` không có phương thức `reset()`
- `RunningMeanStd` không có cơ chế pha trộn (blend) hay khởi động nóng (warmup)
- Sao chép norm từ Explorer→Keeper tồn tại (trong `script.py`), nhưng không xử lý liên miền
- Thống kê chuẩn hóa trạng thái tồn tại xuyên suốt TẤT CẢ các task mà không reset

**Tác động:** Bất kỳ nỗ lực chuyển giao sim→PenGym nào sẽ thất bại tại lớp chuẩn hóa trước khi đến được suy luận policy. Đây là khoảng cách gây hại kỹ thuật nặng nhất.

### Gap C5: Thiếu Giảm Fisher cho Chuyển Giao Liên Miền

**Strategy C §5.3 đặc tả:** Giảm thông tin Fisher theo hệ số β ∈ [0.1, 0.5] khi chuyển từ sim sang PenGym.

**Hiện trạng:**
- `OnlineEWC` trong `script.py` có `ewc_gamma` (mặc định 0.99), nhưng đây là cho tích lũy Fisher online **trong cùng miền** ($F_t = γ F_{t-1} + F_{current}$)
- Không có cơ chế giảm Fisher **liên miền** (nhân tất cả giá trị Fisher với hệ số cố định β < 1)
- `Script_Config` không có tham số `fisher_discount` hay `transfer_beta`

**Tác động:** Không có giảm Fisher, ràng buộc EWC từ huấn luyện sim ở mức tối đa trên PenGym. Do sự dịch chuyển phân phối trạng thái, ma trận Fisher từ sim có thể khóa trọng số ở vị trí tối ưu cho sim nhưng không tối ưu cho PenGym → **agent không thể thích nghi**.

### Gap C6: Thiếu Chuẩn Hóa OS/Service (Canonicalization)

**Strategy C §2.3.1-2.3.4 đặc tả:**
```python
CANONICAL_MAP = {
    'ubuntu': 'linux', 'debian': 'linux', 'centos': 'linux', ...
}
SERVICE_CANONICAL_MAP = {
    'openssh': 'ssh', 'vsftpd': 'ftp', 'apache httpd': 'http', ...
}
```

**Hiện trạng:**
- `PenGymStateAdapter._decode_os()` trả về tên OS thô (ví dụ: `"linux"`) không qua chuẩn hóa
- `PenGymStateAdapter._decode_services()` trả về tên service thô không qua chuẩn hóa
- `StateEncoder` trong `host.py` (phía sim) truyền chuỗi `env_data` thô vào SBERT
- Không có lớp chuẩn hóa nào tồn tại ở bất kỳ đâu trong mã nguồn

**Tác động:** Khi cùng một OS/service có biểu diễn chuỗi khác nhau giữa sim và PenGym, SBERT tạo ra embedding khác nhau → trạng thái host "giống nhau" có vector biểu diễn khác nhau → policy không thể tổng quát hóa liên miền.

---

## 4. Khoảng Cách Trung Bình (Nên Sửa)

### Gap M1: Sai Khoảng Chuẩn Hóa Reward

**Strategy C §4.1:** Chuẩn hóa tất cả reward về khoảng **[-1, +1]**.

**Hiện trạng:** `reward_normalizer.py` ánh xạ reward PenGym về **thang gốc của SCRIPT** [-10, 1000]:
```python
class LinearNormalizer(RewardNormalizer):
    # Ánh xạ pengym [-1, 100] → script [-10, 1000]
    # KHÔNG phải về [-1, +1] như Strategy C đặc tả
```

**Tác động:** Thang reward ảnh hưởng đến độ lớn gradient → ảnh hưởng đến tính toán thông tin Fisher → ảnh hưởng đến cường độ ràng buộc EWC. Không có chuẩn hóa [-1, +1], việc tổng hợp CL liên miền sẽ bị thiên lệch về miền có reward lớn hơn.

### Gap M2: Thiếu Ma Trận Đánh Giá 4 Agent

**Strategy C §6.1 đặc tả so sánh 4 agent:**

| Agent           | Mô tả                                          | Tồn tại? |
| --------------- | ----------------------------------------------- | -------- |
| θ_sim_baseline  | SCRIPT gốc (1538 chiều) trên sim                | ✅ Có thể huấn luyện |
| θ_sim_unified   | SCRIPT huấn luyện lại với mã hóa thống nhất     | ❌ Không tạo được (chưa có unified encoder) |
| θ_dual          | Huấn luyện kép (chuyển giao sim→PenGym)          | ❌ Không tạo được (chưa có pipeline chuyển giao) |
| θ_pengym_scratch | Huấn luyện từ đầu trên PenGym                   | ✅ PenGymScriptTrainer có thể tạo |

**Bộ đánh giá hiện tại:** `sim_to_real_eval.py` triển khai đánh giá Strategy A (chuyển giao zero-shot mô hình đã huấn luyện). Không có bộ đánh giá riêng cho Strategy C.

### Gap M3: Thiếu Framework Kiểm Chứng Phase 0

**Strategy C §8 Phase 0 đặc tả:**

| Bước | Kiểm chứng                                                    | Trạng thái |
| ---- | -------------------------------------------------------------- | ---------- |
| 0.1  | Kiểm tra tính nhất quán SBERT (sim vs PenGym, cosine > 0.9)   | ❌         |
| 0.2  | Kiểm tra ổn định PenGym (< 5% episode lỗi)                    | ❌         |
| 0.3  | Phân tích phân phối trạng thái (histogram, điểm KL)           | ❌         |
| 0.4  | Phân tích trùng lặp kịch bản                                  | ❌         |

**Tác động:** Không có Phase 0, ta không biết các giả định cốt lõi (G1-G7) của Strategy C có đúng không. Triển khai mà không kiểm chứng có nguy cơ lãng phí công sức.

### Gap M4: Thiếu Tham Số Cấu Hình Cho Chuyển Giao

**Strategy C ngầm yêu cầu các tham số cấu hình sau:**

| Tham số                   | Mục đích                                    | Có trong Script_Config? |
| ------------------------- | ------------------------------------------- | ----------------------- |
| `fisher_discount_beta`    | Giảm Fisher liên miền ∈ [0.1, 0.5]         | ❌                      |
| `transfer_lr_factor`      | Giảm LR (×0.1) cho tinh chỉnh              | ❌                      |
| `norm_warmup_episodes`    | Số episode chạy ngẫu nhiên trước tinh chỉnh | ❌                      |
| `norm_reset_on_transfer`  | Có reset thống kê running khi đổi miền không | ❌                      |
| `transfer_strategy`       | 'aggressive'/'conservative'/'cautious'      | ❌                      |

### Gap M5: Chưa Sửa StateEncoder Phía Sim

**Strategy C §5.2 yêu cầu sửa `host.py` StateEncoder:**
```python
# CŨ: state_space = 1538 (2 access + 4×384)
# MỚI: state_space = 1540 (3 access + 1 discovery + 4×384)
```

**Hiện trạng:** `StateEncoder` trong `host.py` **hoàn toàn chưa được sửa** so với bài báo SCRIPT gốc. Vẫn dùng mã hóa access 2 chiều và không có chiều discovery.

**Tác động:** Dù có xây unified encoder cho phía PenGym, phía sim vẫn tạo vector 1538 chiều. Cả hai phía phải tạo ra định dạng giống hệt nhau để chuyển giao hoạt động.

---

## 5. Khoảng Cách Nhỏ (Có Thì Tốt)

### Gap m1: Thiếu Action Masking / ActionCompatibilityLayer

Strategy C §3.2 đề cập `ActionCompatibilityLayer` để xử lý các hành động tồn tại ở một env nhưng không ở env kia. Chưa triển khai, nhưng ưu tiên thấp vì ServiceActionSpace 16 hành động đã là phần giao (intersection).

### Gap m2: Cấu Trúc Thư Mục Đầu Ra

Strategy C §9 đề xuất cấu trúc thư mục cụ thể (`outputs/logs/strategy_c/phase0_validation/`, v.v.). Các thư mục đầu ra hiện tại theo mô hình Strategy A.

### Gap m3: Thiếu Framework Kiểm Định Thống Kê

Strategy C §6.3 đặc tả: Wilcoxon signed-rank test, Friedman test + Nemenyi post-hoc, Cohen's d cho effect size. Không có tiện ích kiểm định thống kê nào trong mã nguồn.

### Gap m4: Thiếu Giám Sát Phân Phối Trạng Thái

Strategy C §7.1 (giảm thiểu R3) khuyến nghị ghi log liên tục KL divergence của phân phối trạng thái giữa sim và PenGym. Không có giám sát như vậy.

### Gap m5: Chưa Ghim Phiên Bản Mô Hình SBERT

Strategy C §10.1 cảnh báo về việc xác minh cùng checkpoint SBERT giữa các môi trường. Module `Encoder.py` tải mô hình theo tên nhưng không xác minh checksum.

---

## 6. Phân Tích Theo Từng Thành Phần

### 6.1 Pipeline Mã Hóa Trạng Thái

```
Tầm nhìn Strategy C:                  Thực tế hiện tại:
─────────────────────                 ──────────────────

┌─────────────────────┐               ┌──────────────────┐
│ UnifiedStateEncoder │               │ StateEncoder      │ (chỉ sim)
│ 1540 chiều          │               │ 1538 chiều        │
│ ┌─────────────────┐ │               │ [2 access         │
│ │ 3d access       │ │               │  + 4×384 SBERT]   │
│ │ 1d discovery    │ │               └──────────────────┘
│ │ canonicalize()  │ │                      ↕ KHÔNG KẾT NỐI
│ │ encode_both()   │ │               ┌──────────────────┐
│ └─────────────────┘ │               │ PenGymStateAdapter│ (chỉ PenGym)
└─────────────────────┘               │ 1538 chiều        │
 Dùng cho CẢ HAI env                  │ [2 access         │
                                      │  + 4×384 SBERT]   │
                                      └──────────────────┘
                                       Tách rời khỏi sim encoder
```

**Vấn đề chính:** Hai đường mã hóa riêng biệt tạo ra vector 1538 chiều danh nghĩa giống nhau nhưng từ các luồng code khác nhau, không đảm bảo tính nhất quán qua chuẩn hóa. Strategy C yêu cầu MỘT bộ mã hóa dùng cho CẢ HAI môi trường.

### 6.2 Pipeline Huấn Luyện

```
Tầm nhìn Strategy C:                  Thực tế hiện tại:
─────────────────────                 ──────────────────

Phase 0: Kiểm chứng giả định         ❌ Chưa triển khai
    ↓
Phase 1: Huấn luyện lại sim + enc     ❌ Chưa triển khai
    ↓ θ_uni, Fisher_sim, norm_sim       thống nhất
Phase 2: DomainTransferManager        ❌ Chưa triển khai
    ↓ Reset norm, giảm Fisher, giảm LR
Phase 3: Tinh chỉnh trên PenGym      ⚠️ PenGymScriptTrainer tồn tại
    ↓ θ_dual                              nhưng chạy ĐƠN LẺ
Phase 4: Đánh giá 4 agent            ❌ Chưa triển khai

                                      Thực tế đang chạy:
                                      PenGymScriptTrainer.train()
                                      → Huấn luyện CRL từ đầu trên PenGym
                                      → Không pre-train trên sim, không transfer
```

### 6.3 Xử Lý Chuẩn Hóa & Chuyển Giao

```
Tầm nhìn Strategy C:                  Thực tế hiện tại:
─────────────────────                 ──────────────────

Class Normalization với:               Class Normalization:
├── reset()                            ├── __call__(x, update) → chuẩn hóa
├── warmup(random_rollouts)            ├── RunningMeanStd(shape)
├── blend(sim_stats, real_stats, α)    └── Không reset, warmup, hay blend
└── save/load thống kê norm

                                       Agent.save/load lưu thống kê norm
DomainTransferManager:                 ❌ Không tồn tại
├── _reset_normalizer()
├── _warmup_normalizer(episodes=10)
├── _discount_fisher(β=0.3)
└── _adjust_lr(factor=0.1)

Script_Config với:                     Script_Config:
├── fisher_discount_beta               ├── ewc_lambda = 2000
├── transfer_lr_factor                 ├── ewc_gamma = 0.99 (tích lũy online)
├── norm_warmup_episodes               └── Không có tham số chuyển giao
└── transfer_strategy
```

### 6.4 Phân Tích Triển Khai EWC

Triển khai EWC hiện tại trong `script.py` được cấu trúc tốt cho học liên tục **trong cùng miền** nhưng thiếu khả năng liên miền:

| Tính năng EWC         | Trong miền (các task sim) | Liên miền (Sim→PenGym) |
| --------------------- | ------------------------ | ----------------------- |
| Tính Fisher           | ✅ `compute_fisher()`    | Cùng code có thể dùng   |
| Tích lũy Fisher       | ✅ Online: $F_t = γF_{t-1} + F_{current}$ | ❌ Không có cơ chế giảm |
| Chuẩn hóa Fisher      | ✅ L2 theo từng tham số  | Cùng code có thể dùng   |
| Lưu tham số cũ        | ✅ dict `saved_params`, xóa cũ | ❌ Không lưu trữ theo miền |
| Phạt EWC              | ✅ $Σ F_i(θ_i - θ^*_i)^2$ | Cùng code có thể dùng  |

**Đánh giá:** Hạ tầng EWC vững chắc. Thêm giảm Fisher liên miền chỉ cần ~20 dòng code trong class `OnlineEWC` + 1 tham số mới trong `Script_Config`.

---

## 7. Đề Xuất Cải Tiến (Xếp Hạng Ưu Tiên)

### Ưu tiên 1: Triển khai UnifiedStateEncoder (Tác động: Nghiêm trọng, Công sức: Trung bình)

**Cần xây:**
```
src/envs/core/unified_state_encoder.py (file mới)
```

**Yêu cầu:**
- Nhận đầu vào từ CẢ HAI `StateEncoder` (sim) và `PenGymStateAdapter` (PenGym)
- Đầu ra vector 1540 chiều: [3 access + 1 discovery + 4×384 SBERT]
- Bao gồm phương thức chuẩn hóa OS/service (canonicalization)
- Dùng chung một instance mô hình SBERT
- Cache SBERT dùng chung giữa các lời gọi

**Cách triển khai:**
1. Tạo class `UnifiedStateEncoder` với phương thức `encode_from_sim()` và `encode_from_pengym()`
2. Cả hai phương thức đều: chuẩn hóa nội bộ → mã hóa SBERT → đóng gói vào vector 1540 chiều
3. Sửa `host.py` để `StateEncoder.state_space` tham chiếu `UnifiedStateEncoder.TOTAL_DIM`
4. Sửa `PenGymStateAdapter.STATE_DIM` thành 1540 và cập nhật mã hóa access/discovery

**Ước tính công sức:** 200-300 dòng code, 2-3 ngày

### Ưu tiên 2: Xây DomainTransferManager (Tác động: Nghiêm trọng, Công sức: Trung bình)

**Cần xây:**
```
src/training/domain_transfer.py (file mới)
```

**Yêu cầu:**
- `transfer(sim_agent, strategy='conservative')` → trả về agent PenGym đã cấu hình
- Ba chiến lược: aggressive (sao chép tất cả), conservative (reset norm, giảm Fisher), cautious (reset tất cả, chỉ chuyển trọng số)
- `_reset_normalizer()`: Reset `RunningMeanStd.n`, `.mean`, `.S`, `.std`
- `_warmup_normalizer()`: Chạy N episode ngẫu nhiên trên PenGym, cập nhật thống kê norm
- `_discount_fisher(β)`: Nhân tất cả giá trị Fisher với β
- `_adjust_lr(factor)`: Giảm actor_lr và critic_lr

**Phụ thuộc:** Cần thêm phương thức `reset()` vào class `Normalization` trong `common.py`.

**Ước tính công sức:** 150-200 dòng, 2 ngày

### Ưu tiên 3: Thêm Tham Số Cấu Hình Chuyển Giao vào Script_Config (Tác động: Cao, Công sức: Thấp)

**Cần sửa:** `src/agent/policy/config.py`

```python
class Script_Config(PolicyDistillation_Config):
    def __init__(self, ...,
                 # Tham số chuyển giao (Strategy C)
                 fisher_discount_beta=0.3,       # β ∈ [0.1, 0.5]
                 transfer_lr_factor=0.1,          # LR × 0.1 cho tinh chỉnh
                 norm_warmup_episodes=10,          # Số episode ngẫu nhiên trước tinh chỉnh
                 norm_reset_on_transfer=True,      # Reset running stats khi đổi miền
                 transfer_strategy='conservative', # aggressive/conservative/cautious
                 **kwargs):
```

**Ước tính công sức:** 20 dòng, 30 phút

### Ưu tiên 4: Thêm Reset/Warmup cho Normalization (Tác động: Nghiêm trọng, Công sức: Thấp)

**Cần sửa:** `src/agent/policy/common.py`

```python
class RunningMeanStd:
    def reset(self):
        """Reset thống kê cho chuyển giao miền."""
        self.n = 0
        self.mean = np.zeros(self.shape)
        self.S = np.zeros(self.shape)
        self.std = np.sqrt(self.S)

class Normalization:
    def reset(self):
        """Reset thống kê running cho chuyển giao miền."""
        self.running_ms.reset()

    def warmup(self, states: np.ndarray):
        """Cập nhật thống kê hàng loạt từ các trạng thái đã thu thập."""
        for s in states:
            self.running_ms.update(s)
```

**Ước tính công sức:** 15 dòng, 15 phút

### Ưu tiên 5: Thêm Giảm Fisher vào OnlineEWC (Tác động: Cao, Công sức: Thấp)

**Cần sửa:** `src/agent/continual/script.py` → class `OnlineEWC`

```python
def discount_fisher(self, beta: float):
    """Giảm Fisher liên miền. Nhân tất cả giá trị Fisher với β."""
    for task_id in self.saved_fisher:
        for name in self.saved_fisher[task_id]:
            self.saved_fisher[task_id][name] *= beta
```

**Ước tính công sức:** 10 dòng, 15 phút

### Ưu tiên 6: Xây Bộ Điều Phối Huấn Luyện Kép (Tác động: Nghiêm trọng, Công sức: Cao)

**Cần xây:**
```
src/training/dual_trainer.py (file mới)
```

**Yêu cầu:**
- Điều phối Phase 0→1→2→3→4 thành một pipeline duy nhất
- Phase 0: Chạy kiểm tra kiểm chứng (tính nhất quán SBERT, ổn định PenGym)
- Phase 1: Huấn luyện trên sim với mã hóa thống nhất → lưu θ_uni
- Phase 2: Gọi DomainTransferManager.transfer() → agent đã cấu hình
- Phase 3: Tinh chỉnh trên PenGym (PenGymScriptTrainer đã sửa đổi)
- Phase 4: Chạy ma trận đánh giá 4 agent

**Phụ thuộc:** Yêu cầu Ưu tiên 1-5 đã triển khai trước.

**Ước tính công sức:** 300-400 dòng, 3-5 ngày

### Ưu tiên 7: Triển khai Chuẩn Hóa Reward về [-1, +1] (Tác động: Trung bình, Công sức: Thấp)

**Cần sửa:** `src/envs/wrappers/reward_normalizer.py`

Thêm class normalizer mới:
```python
class UnifiedNormalizer(RewardNormalizer):
    """Chuẩn hóa reward về [-1, +1] cho CL liên miền."""
    def __init__(self, source: str):
        self.max_reward = 1000.0 if source == 'simulation' else 100.0
        self.min_reward = -10.0 if source == 'simulation' else -3.0

    def normalize(self, reward: float) -> float:
        if reward > 0: return min(reward / self.max_reward, 1.0)
        if reward < 0: return max(reward / abs(self.min_reward), -1.0)
        return 0.0
```

**Ước tính công sức:** 20 dòng, 30 phút

### Ưu tiên 8: Xây Framework Đánh Giá 4 Agent (Tác động: Trung bình, Công sức: Trung bình)

**Cần xây:**
```
src/evaluation/strategy_c_eval.py (file mới)
```

**Yêu cầu:**
- Tải/tạo tất cả 4 agent (θ_baseline, θ_unified, θ_dual, θ_scratch)
- Đánh giá từng agent trên sim và PenGym (nếu áp dụng)
- Tính toán metrics chuyển giao: forward transfer, backward transfer, transfer ratio
- Tính toán metrics CL: mức tuân thủ ràng buộc EWC, giữ lại tri thức/tiếp thu tri thức
- Xuất ma trận so sánh dưới dạng JSON + bảng console

**Ước tính công sức:** 200-300 dòng, 2-3 ngày

---

## 8. Lộ Trình Triển Khai

### Sprint 1: Nền Tảng (3-4 ngày)

| Nhiệm vụ                                   | Ưu tiên | Công sức  | File                                                                          |
| ------------------------------------------- | ------- | --------- | ----------------------------------------------------------------------------- |
| Thêm Normalization.reset() + warmup()       | P4      | 15 phút   | `common.py`                                                                   |
| Thêm giảm Fisher vào OnlineEWC              | P5      | 15 phút   | `script.py`                                                                   |
| Thêm tham số chuyển giao vào Script_Config  | P3      | 30 phút   | `config.py`                                                                   |
| Thêm UnifiedNormalizer                       | P7      | 30 phút   | `reward_normalizer.py`                                                        |
| Triển khai UnifiedStateEncoder               | P1      | 2-3 ngày  | file mới `unified_state_encoder.py`, sửa `host.py`, `state_adapter.py`       |

**Sprint 1 đạt được:** Tất cả khối xây dựng cho chuyển giao, cộng mã hóa thống nhất.

### Sprint 2: Pipeline Chuyển Giao (4-5 ngày)

| Nhiệm vụ                                    | Ưu tiên | Công sức | File                                     |
| -------------------------------------------- | ------- | -------- | ---------------------------------------- |
| Xây DomainTransferManager                    | P2      | 2 ngày   | file mới `domain_transfer.py`            |
| Xây bộ điều phối DualTrainer                 | P6      | 3 ngày   | file mới `dual_trainer.py`               |
| Tích hợp với PenGymScriptTrainer              | —       | 1 ngày   | sửa `pengym_script_trainer.py`           |

**Sprint 2 đạt được:** Pipeline Phase 0→1→2→3 hoàn chỉnh.

### Sprint 3: Đánh Giá & Hoàn Thiện (3-4 ngày)

| Nhiệm vụ                                    | Ưu tiên | Công sức  | File                                     |
| -------------------------------------------- | ------- | --------- | ---------------------------------------- |
| Xây framework đánh giá 4 agent               | P8      | 2-3 ngày  | file mới `strategy_c_eval.py`            |
| Thêm script kiểm chứng Phase 0               | —       | 1 ngày    | script kiểm chứng mới                   |
| Cấu trúc thư mục đầu ra                      | m2      | 30 phút   | thiết lập thư mục                        |

**Sprint 3 đạt được:** Khả năng đánh giá Strategy C hoàn chỉnh.

### Tổng ước tính công sức: 10-13 ngày phát triển tập trung

---

## 9. Phụ Lục: Ánh Xạ File → Yêu Cầu

### File Hiện Có vs Yêu Cầu Strategy C

| File                                            | Vai trò trong Strategy C                         | Trạng thái     | Hành động cần thiết                                       |
| ----------------------------------------------- | ------------------------------------------------ | -------------- | --------------------------------------------------------- |
| `src/agent/host.py`                             | StateEncoder → UnifiedStateEncoder                | ⚠️ Chưa sửa   | Sửa `state_space`, dùng unified encoder                   |
| `src/agent/policy/common.py`                    | Reset/warmup chuẩn hóa                           | ⚠️ Thiếu method | Thêm `reset()`, `warmup()` vào Normalization              |
| `src/agent/policy/config.py`                    | Tham số cấu hình chuyển giao                     | ⚠️ Thiếu tham số | Thêm fisher_discount_beta, transfer_lr_factor, v.v.      |
| `src/agent/policy/PPO.py`                       | Input dim 1540                                    | ⚠️ Dùng StateEncoder.state_space | Sẽ tự cập nhật khi StateEncoder thay đổi |
| `src/agent/continual/script.py`                 | Giảm Fisher, EWC liên miền                       | ⚠️ Thiếu giảm  | Thêm `discount_fisher(β)` vào OnlineEWC                  |
| `src/agent/agent_continual.py`                  | Điều phối CRL                                    | ✅ Hoạt động   | Không cần thay đổi                                        |
| `src/envs/adapters/state_adapter.py`            | PenGym → trạng thái thống nhất                   | ⚠️ 1538 chiều, không canon | Nâng lên 1540 chiều, thêm canonicalization   |
| `src/envs/adapters/service_action_mapper.py`    | Ánh xạ hành động                                 | ✅ Hoạt động   | Không cần thay đổi                                        |
| `src/envs/adapters/pengym_host_adapter.py`      | Duck-typing giao diện HOST                       | ✅ Hoạt động   | Không cần thay đổi                                        |
| `src/envs/wrappers/single_host_wrapper.py`      | PenGym đơn-host                                  | ✅ Hoạt động   | Cập nhật thuộc tính state_dim nếu có unified encoder      |
| `src/envs/wrappers/reward_normalizer.py`        | Chuẩn hóa reward                                 | ⚠️ Sai khoảng  | Thêm UnifiedNormalizer ([-1, +1])                         |
| `src/training/pengym_script_trainer.py`         | Tinh chỉnh Phase 3                               | ⚠️ Chạy đơn lẻ | Sửa để nhận mô hình đã pre-train từ Phase 2              |
| `src/training/pengym_trainer.py`                | Huấn luyện PPO trên PenGym                       | ✅ Hoạt động   | Không cần thay đổi                                        |
| `src/evaluation/sim_to_real_eval.py`            | Bộ đánh giá Strategy A                           | ✅ Hoạt động   | Tách biệt khỏi Strategy C                                |
| `src/agent/actions/service_action_space.py`     | Không gian hành động thống nhất                  | ✅ Hoạt động   | Không cần thay đổi                                        |

### File Mới Cần Tạo Cho Strategy C

| File mới                                    | Mục đích                                     | Ưu tiên            |
| ------------------------------------------- | -------------------------------------------- | ------------------- |
| `src/envs/core/unified_state_encoder.py`    | Bộ mã hóa duy nhất cho cả hai môi trường    | P1 (Nghiêm trọng)  |
| `src/training/domain_transfer.py`           | DomainTransferManager                         | P2 (Nghiêm trọng)  |
| `src/training/dual_trainer.py`              | Bộ điều phối Phase 0-4                       | P6 (Nghiêm trọng)  |
| `src/evaluation/strategy_c_eval.py`         | Ma trận đánh giá 4 agent                     | P8 (Trung bình)    |
| `scripts/phase0_validation.py`              | Thí nghiệm kiểm chứng Phase 0               | P3 (Trung bình)    |

---

## Tổng Kết: Phân Bố Mức Độ Khoảng Cách

```
Nghiêm trọng (chặn Strategy C):    6 gap  (C1-C6)
Trung bình (giảm chất lượng):      5 gap  (M1-M5)
Nhỏ (hoàn thiện):                  5 gap  (m1-m5)
                                   ─────
Tổng cộng:                         16 gap

Đã triển khai đúng:                6 thành phần chính
Khớp một phần:                     3 thành phần (sai chiều hoặc chạy đơn lẻ)
```

**Mã nguồn đã xây ~60% hạ tầng Strategy C (lớp cầu nối), nhưng 0% đổi mới cốt lõi (pipeline chuyển giao).** Sprint 1-2 (7-9 ngày) sẽ đưa hệ thống đến mức prototype Strategy C hoạt động được.
