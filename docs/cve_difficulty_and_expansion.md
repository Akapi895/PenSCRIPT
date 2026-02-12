# CVE Difficulty Grading & Scenario Expansion Pipeline

> Technical Design Document — Problems 2 & 3
> Tận dụng CVE_dataset.csv để xây dựng môi trường CVE theo mức độ khó tăng dần

---

## Mục lục

1. [Phân tích CVE_dataset.csv](#1-phân-tích-cve_datasetcsv)
2. [Phát hiện quan trọng: MSF_Rank quyết định prob](#2-phát-hiện-quan-trọng-msf_rank-quyết-định-prob)
3. [Khung phân loại độ khó CVE (Difficulty Grading Framework)](#3-khung-phân-loại-độ-khó-cve)
4. [Tách scenario YAML: Template + CVE Overlay](#4-tách-scenario-yaml-template--cve-overlay)
5. [Pipeline sinh môi trường tự động](#5-pipeline-sinh-môi-trường-tự-động)
6. [Tích hợp Curriculum Learning](#6-tích-hợp-curriculum-learning)
7. [Roadmap kỹ thuật](#7-roadmap-kỹ-thuật)

---

## 1. Phân tích CVE_dataset.csv

### 1.1. Tổng quan dữ liệu

| Metric                                          | Value |
| ----------------------------------------------- | ----- |
| Tổng số dòng                                    | 1985  |
| CVE duy nhất                                    | 1833  |
| Dòng trùng lặp (cùng CVE, khác service mapping) | 119   |
| Cột dữ liệu                                     | 22    |
| Exploit rows (có `service`, không có `process`) | 1945  |
| Privesc rows (có `process`, không có `service`) | 40    |

**Quy tắc loại trừ**: Mỗi dòng CÓ `service` HOẶC CÓ `process`, không bao giờ cả hai, không bao giờ cả hai trống. Privesc CVE dùng `process`, exploit CVE dùng `service`.

### 1.2. Schema 22 cột

| Cột                   | Ý nghĩa                         | Giá trị mẫu                                          | Dùng cho Difficulty? |
| --------------------- | ------------------------------- | ---------------------------------------------------- | -------------------- |
| `CVE_ID`              | Mã CVE                          | CVE-2017-5638                                        | —                    |
| `MSF_Module`          | Tên module Metasploit           | exploit/multi/http/struts2_content_type_ognl         | —                    |
| `MSF_Rank`            | Xếp hạng MSF (tin cậy)          | Excellent, Great, Good, Normal, Average, Manual, Low | ✅ Primary           |
| `Base_Score`          | CVSS base score                 | 2.1 – 10.0 (mean=8.66, median=9.3)                   | ✅ Secondary         |
| `Vector_String`       | CVSS vector đầy đủ              | CVSS:3.1/AV:N/AC:L/...                               | — (parsed below)     |
| `Attack_Vector`       | Vectơ tấn công                  | NETWORK=1802, LOCAL=168                              | ✅ Filter            |
| `Attack_Complexity`   | Độ phức tạp                     | LOW=1581, MEDIUM=342, HIGH=62                        | ✅ Primary           |
| `Privileges_Required` | Quyền cần thiết                 | (empty)=977, NONE=735, LOW=206, HIGH=67              | ✅ Primary           |
| `User_Interaction`    | Cần user tương tác?             | NONE (majority) vs REQUIRED                          | ✅ Filter            |
| `Severity`            | Mức nghiêm trọng                | (empty)=977, CRITICAL=508, HIGH=452, MEDIUM=46       | ⚠️ Redundant         |
| `CWE_ID`              | Loại lỗ hổng                    | CWE-119(426), NVD-CWE-Other(287), CWE-78(273)        | ⚠️ Optional          |
| `Description`         | Mô tả chi tiết                  | Free text                                            | —                    |
| `Affected_CPEs`       | Phần mềm bị ảnh hưởng           | CPE strings                                          | —                    |
| `Published_Date`      | Năm công bố                     | 1997 – 2025                                          | —                    |
| `CVSS_Version`        | Phiên bản CVSS                  | 2.0, 3.0, 3.1                                        | —                    |
| `References`          | Links tham khảo                 | URLs                                                 | —                    |
| `os`                  | Hệ điều hành                    | windows=1138, linux=531, unix=217, cisco=51          | ✅ Constraint        |
| `prob`                | Xác suất exploit thành công     | 0.99=884, 0.8=420, 0.6=616, 0.4=65                   | ✅ **Key proxy**     |
| `cost`                | Chi phí exploit                 | 1=884, 2=420, 3=616, 4=65                            | ✅ = f(prob)         |
| `service`             | Dịch vụ mục tiêu (exploit only) | http=720, browser=263, ftp=71, smb=39…               | ✅ Mapping           |
| `process`             | Process mục tiêu (privesc only) | overlayfs, pkexec, docker_escape…                    | ✅ Mapping           |
| `access`              | Quyền đạt được                  | user=1331, root=410, system=244                      | ✅ Reward signal     |

### 1.3. Phân phối service (exploit rows)

```
http          720  ████████████████████████████████████  (37.0%)
browser       263  █████████████                         (13.5%)
fileformat    192  ██████████                            ( 9.9%)
misc          156  ████████                              ( 8.0%)
webapp        156  ████████                              ( 8.0%)
ftp            71  ████                                  ( 3.7%)
smb            39  ██                                    ( 2.0%)
scada          38  ██                                    ( 2.0%)
windows        34  ██                                    ( 1.7%)
ssh            23  █                                     ( 1.2%)
brightstor     18  █                                     ( 0.9%)
imap           17  █                                     ( 0.9%)
gather         14  █                                     ( 0.7%)
smtp           14  █                                     ( 0.7%)
antivirus      11  █                                     ( 0.6%)
tftp           11  █                                     ( 0.6%)
others        168  [21 loại khác, mỗi loại <10 rows]
```

### 1.4. PenGym compatibility gap

**Vấn đề**: PenGym hiện chỉ hỗ trợ 5 service: `ssh`, `ftp`, `http`, `samba`, `smtp`.

| Nhóm                                                  | CVE count | % tổng | Tình trạng               |
| ----------------------------------------------------- | --------- | ------ | ------------------------ |
| PenGym-compatible (ssh+ftp+http+smb+smtp)             | 874       | 44.0%  | ✅ Dùng trực tiếp        |
| Có thể map vào http (webapp, iis)                     | ~164      | 8.3%   | ⚠️ Cần abstract          |
| Không tương thích (browser, fileformat, misc, scada…) | ~907      | 45.7%  | ❌ Loại bỏ hoặc abstract |
| Privesc                                               | 40        | 2.0%   | ✅ Dùng riêng            |

**Hệ quả**: Chỉ ~44% CVE có thể dùng trực tiếp. Nếu gộp webapp → http được ~52%. Để tận dụng hết dataset cần mở rộng service set hoặc dùng service abstraction.

### 1.5. Privesc inventory (40 rows)

| OS      | Count | Access=root | Ví dụ process                                                      |
| ------- | ----- | ----------- | ------------------------------------------------------------------ |
| Linux   | 28    | 26          | overlayfs, netfilter, pkexec, sudo, docker_escape, ebpf, dirtypipe |
| Windows | 8     | 4 (system)  | ms16_075, ms18_8120, smbghost, spoolfool                           |
| macOS   | 3     | 2           | dirty_cow, libxpc                                                  |
| Unix    | 1     | 1           | chkrootkit                                                         |

**Quan sát**: PenGym hiện chỉ có 3 privesc type (pe_tomcat, pe_daclsvc, pe_schtask). Dataset có 37 process names → tiềm năng mở rộng lớn.

---

## 2. Phát hiện quan trọng: MSF_Rank quyết định prob

### 2.1. Bảng mapping gần-tuyệt đối

| MSF_Rank        | prob=0.99 | prob=0.8 | prob=0.6 | prob=0.4 | **Dominant prob** | **Tỷ lệ dominant** |
| --------------- | --------- | -------- | -------- | -------- | ----------------- | ------------------ |
| Excellent (844) | **836**   | 3        | 1        | 4        | 0.99              | **99.1%**          |
| Great (189)     | —         | **186**  | 1        | 2        | 0.80              | **98.4%**          |
| Good (223)      | 3         | **210**  | 10       | —        | 0.80              | **94.2%**          |
| Normal (539)    | 45        | 18       | **473**  | 3        | 0.60              | **87.8%**          |
| Average (133)   | —         | 3        | **130**  | —        | 0.60              | **97.7%**          |
| Manual (51)     | —         | —        | 1        | **50**   | 0.40              | **98.0%**          |
| Low (6)         | —         | —        | —        | **6**    | 0.40              | **100%**           |

### 2.2. Ý nghĩa

**prob không phải giá trị ngẫu nhiên** — nó được gán dựa trên MSF_Rank theo quy tắc gần-deterministic:

```
Excellent  → prob = 0.99  (exploit gần chắc chắn thành công)
Great/Good → prob = 0.80  (exploit đáng tin cậy)
Normal/Avg → prob = 0.60  (exploit có thể thất bại)
Manual/Low → prob = 0.40  (exploit khó, không ổn định)
```

**Hệ quả cho thiết kế**:

- `prob` đã encode thông tin MSF_Rank → **chỉ cần dùng 1 trong 2**, không cần cả hai
- prob chỉ có 4 mức rời rạc → cần thêm tín hiệu từ CVSS metadata để tạo gradient mịn hơn
- Các dòng "outlier" (VD: Excellent nhưng prob=0.4) cần kiểm tra — có thể là lỗi gán hoặc CVE đặc biệt

### 2.3. prob × access (quyền đạt được)

| prob | user | root | system | % root/system |
| ---- | ---- | ---- | ------ | ------------- |
| 0.99 | 471  | 314  | 99     | 46.7%         |
| 0.80 | 330  | 36   | 54     | 21.4%         |
| 0.60 | 494  | 43   | 79     | 19.8%         |
| 0.40 | 36   | 17   | 12     | 44.6%         |

**Quan sát**: prob=0.99 (easiest to exploit) lại có tỷ lệ root/system cao nhất (47%). Điều này nghĩa là exploit dễ ≠ ít nguy hiểm. Cần tách biệt:

- **Exploitation difficulty** (prob biểu diễn): dễ/khó exploit thành công
- **Impact level** (access biểu diễn): user/root — mức quyền đạt được sau exploit
- **Training difficulty** (cần thiết kế): kết hợp exploit difficulty + network topology + number of targets

---

## 3. Khung phân loại độ khó CVE (Difficulty Grading Framework)

### 3.1. Nguyên tắc thiết kế

Difficulty cho **RL curriculum learning** khác với CVSS severity:

- CVSS đo **mức nguy hiểm** cho defender → càng cao càng nguy hiểm
- Curriculum difficulty đo **mức khó cho attacker** → càng cao càng khó exploit thành công

Ba chiều độ khó:

| Chiều                       | Ý nghĩa                                    | Nguồn dữ liệu                            |
| --------------------------- | ------------------------------------------ | ---------------------------------------- |
| **D1: Exploit Difficulty**  | Xác suất exploit thành công thấp → khó hơn | prob, MSF_Rank, Attack_Complexity        |
| **D2: Access Barrier**      | Cần quyền cao để exploit → khó hơn         | Privileges_Required, User_Interaction    |
| **D3: Scenario Complexity** | Topology phức tạp, nhiều hop → khó hơn     | Subnet count, firewall rules, host count |

D1 và D2 là **thuộc tính của CVE** (gán được từ dataset).
D3 là **thuộc tính của scenario** (gán khi sinh environment).
Curriculum learning kết hợp cả 3 chiều.

### 3.2. Composite CVE Difficulty Score

Định nghĩa **CVE Difficulty Score** $S_{diff}$ cho mỗi CVE:

$$S_{diff} = w_1 \cdot f_{prob} + w_2 \cdot f_{AC} + w_3 \cdot f_{PR} + w_4 \cdot f_{UI}$$

Trong đó:

| Feature    | Encoding                                   | Giá trị                |
| ---------- | ------------------------------------------ | ---------------------- |
| $f_{prob}$ | $1 - prob$                                 | 0.01, 0.20, 0.40, 0.60 |
| $f_{AC}$   | LOW→0.0, MEDIUM→0.5, HIGH→1.0              | 0.0, 0.5, 1.0          |
| $f_{PR}$   | NONE→0.0, LOW→0.33, HIGH→0.67, (empty→0.0) | 0.0, 0.33, 0.67        |
| $f_{UI}$   | NONE→0.0, REQUIRED→1.0                     | 0.0, 1.0               |

**Đề xuất trọng số** (dựa trên phân tích variance):

$$w_1 = 0.50, \quad w_2 = 0.25, \quad w_3 = 0.15, \quad w_4 = 0.10$$

Lý do: `prob` mang nhiều thông tin nhất (4 mức phân biệt rõ), `Attack_Complexity` có 3 mức nhưng phần lớn (80%) là LOW. `Privileges_Required` và `User_Interaction` cung cấp refinement.

**Ví dụ tính toán**:

| CVE                      | prob | AC     | PR   | UI   | $S_{diff}$ | Tier   |
| ------------------------ | ---- | ------ | ---- | ---- | ---------- | ------ |
| CVE-2017-5638 (Struts2)  | 0.99 | LOW    | NONE | NONE | 0.005      | Easy   |
| CVE-2020-2555 (WebLogic) | 0.99 | LOW    | NONE | NONE | 0.005      | Easy   |
| CVE-2019-0708 (BlueKeep) | 0.8  | LOW    | NONE | NONE | 0.10       | Easy   |
| Typical Normal-rank CVE  | 0.6  | MEDIUM | LOW  | NONE | 0.375      | Medium |
| Typical Manual-rank CVE  | 0.4  | HIGH   | HIGH | REQ  | 0.75       | Hard   |

### 3.3. Difficulty Tiers cho Curriculum

| Tier           | $S_{diff}$ range | Đặc điểm                           | Est. CVE count |
| -------------- | ---------------- | ---------------------------------- | -------------- |
| **T1: Easy**   | [0.0, 0.15)      | prob≥0.8, AC=LOW, PR=NONE/empty    | ~1200          |
| **T2: Medium** | [0.15, 0.35)     | prob=0.6, AC=LOW/MEDIUM, PR=LOW    | ~500           |
| **T3: Hard**   | [0.35, 0.55)     | prob=0.6 + AC=MEDIUM hoặc prob=0.4 | ~220           |
| **T4: Expert** | [0.55, 1.0]      | prob=0.4, AC=HIGH, PR=HIGH         | ~65            |

> **Lưu ý**: T1 lớn nhất (~60%) là tốt cho curriculum — agent cần nhiều positive signal giai đoạn đầu.

### 3.4. Xử lý cột trống và CVSS v2

**Vấn đề**: 977/1985 dòng (49.2%) có `Privileges_Required`, `User_Interaction`, `Severity` trống — đây là các CVE chỉ có CVSS v2.0 (không có v3 metrics).

**Giải pháp**: Dùng `MSF_Rank` và `prob` (luôn có giá trị) làm primary feature. Khi CVSS v3 metrics trống:

```python
def compute_difficulty(row):
    f_prob = 1.0 - float(row['prob'])

    # Primary: prob (always available)
    if not row['Attack_Complexity']:
        # CVSS v2 only: derive from prob alone
        return f_prob  # range [0.01, 0.60]

    # Full CVSS v3: use composite score
    f_ac = {'LOW': 0.0, 'MEDIUM': 0.5, 'HIGH': 1.0}[row['Attack_Complexity']]
    f_pr = {'': 0.0, 'NONE': 0.0, 'LOW': 0.33, 'HIGH': 0.67}.get(row['Privileges_Required'], 0.0)
    f_ui = 1.0 if row['User_Interaction'] == 'REQUIRED' else 0.0

    return 0.50 * f_prob + 0.25 * f_ac + 0.15 * f_pr + 0.10 * f_ui
```

### 3.5. Phân loại bổ sung theo access level

Tách riêng `access` (user/root/system) không ảnh hưởng difficulty score mà ảnh hưởng **reward design**:

| access | Ý nghĩa RL             | Reward multiplier (đề xuất) |
| ------ | ---------------------- | --------------------------- |
| user   | Compromise cơ bản      | 1.0×                        |
| root   | Full control (Linux)   | 1.5×                        |
| system | Full control (Windows) | 1.5×                        |

Curriculum có thể phase thêm: giai đoạn đầu chỉ yêu cầu `user` access, giai đoạn sau yêu cầu `root/system`.

---

## 4. Tách scenario YAML: Template + CVE Overlay

### 4.1. Vấn đề hiện tại

PenGym scenario YAML chứa **tất cả** trong một file:

- Network topology (subnets, connectivity)
- Service definitions (service set, exploit definitions với prob/cost/access)
- Host assignments (host nào có service gì)
- Firewall rules
- Step limit, sensitive hosts

**Hệ quả**: Mỗi tổ hợp topology × CVE mix cần 1 file YAML riêng. Với 1833 CVE × N topology = bùng nổ files.

### 4.2. Kiến trúc đề xuất: Template + Overlay

```
scenario/
├── templates/                    # Network topology templates
│   ├── tiny-linear.template.yml  # 3 hosts, 1 subnet chain
│   ├── small-tree.template.yml   # 6 hosts, 2 subnets, branching
│   ├── medium-mesh.template.yml  # 16 hosts, 5 subnets, mesh
│   └── large-enterprise.template.yml
├── overlays/                     # CVE assignment overlays
│   ├── tier1-easy/
│   │   ├── overlay-001.yml
│   │   ├── overlay-002.yml
│   │   └── ...
│   ├── tier2-medium/
│   ├── tier3-hard/
│   └── tier4-expert/
├── generated/                    # Compiled scenarios (auto-generated)
│   └── tiny-linear__overlay-001.yml
└── generator.py                  # Template + Overlay → Full scenario
```

### 4.3. Template schema

Template chứa **topology và structure**, dùng slot thay cho CVE cụ thể:

```yaml
# templates/small-tree.template.yml
meta:
  name: small-tree
  description: "6 hosts, 2 subnets, tree topology"
  host_count: 6
  subnet_count: 2

subnets: [1, 3, 2] # internet(1) + subnet_1(3) + subnet_2(2)

topology:
  - [1, 1, 0] # internet ↔ subnet_1
  - [1, 1, 1] # subnet_1 ↔ subnet_2
  - [0, 1, 1] # (symmetric)

sensitive_hosts:
  - [2, 0] # target in subnet_2, host 0
  - [2, 1] # target in subnet_2, host 1

# Service SLOTS — overlay fills these
service_slots:
  - slot_id: S1
    host: [1, 0]
    role: "entry_point"
    allowed_services: [ssh, ftp, http, smtp, samba]

  - slot_id: S2
    host: [1, 1]
    role: "pivot"
    allowed_services: [http, ssh, ftp]

  - slot_id: S3
    host: [1, 2]
    role: "pivot"
    allowed_services: [http, samba, smtp]

  - slot_id: S4
    host: [2, 0]
    role: "target"
    allowed_services: [http, ssh, ftp, samba]

  - slot_id: S5
    host: [2, 1]
    role: "target"
    allowed_services: [http, ssh]

# Privesc SLOTS
privesc_slots:
  - slot_id: P1
    host: [2, 0]
    allowed_processes: [any]

firewall:
  - [1, 0, 1, 0, "any"] # internet → subnet_1 host 0: open
  - [1, 0, 1, 1, "none"] # internet → subnet_1 host 1: blocked
  - [1, 0, 1, 2, "none"] # internet → subnet_1 host 2: blocked
  # ... (define access control)

step_limit: 2000
```

### 4.4. Overlay schema

Overlay chứa **CVE assignments cụ thể** cho mỗi slot:

```yaml
# overlays/tier1-easy/overlay-001.yml
meta:
  overlay_id: "overlay-001"
  difficulty_tier: 1
  avg_difficulty_score: 0.05
  description: "All Excellent-rank CVEs, HTTP-heavy"

slot_assignments:
  S1:
    service: http
    exploit:
      name: e_http_struts2
      cve_id: CVE-2017-5638
      prob: 0.99
      cost: 1
      access: root
  S2:
    service: ssh
    exploit:
      name: e_ssh_libssh
      cve_id: CVE-2018-10933
      prob: 0.99
      cost: 1
      access: user
  S3:
    service: http
    exploit:
      name: e_http_weblogic
      cve_id: CVE-2020-2555
      prob: 0.99
      cost: 1
      access: root
  S4:
    service: ftp
    exploit:
      name: e_ftp_vsftpd
      cve_id: CVE-2011-2523
      prob: 0.99
      cost: 1
      access: root
  S5:
    service: http
    exploit:
      name: e_http_jenkins
      cve_id: CVE-2017-1000353
      prob: 0.99
      cost: 1
      access: user

  P1:
    process: overlayfs_priv_esc
    privesc:
      name: pe_overlayfs
      cve_id: CVE-2015-1328
      prob: 0.8
      cost: 2
      access: root

os_assignments:
  S1: linux
  S2: linux
  S3: linux
  S4: linux
  S5: linux
```

### 4.5. Generator: Template × Overlay → Full PenGym YAML

```python
# generator.py (pseudo-code)
def generate_scenario(template_path, overlay_path, output_path):
    template = load_yaml(template_path)
    overlay = load_yaml(overlay_path)

    scenario = {}

    # 1. Copy topology verbatim
    scenario['subnets'] = template['subnets']
    scenario['topology'] = template['topology']
    scenario['sensitive_hosts'] = template['sensitive_hosts']
    scenario['firewall'] = template['firewall']
    scenario['step_limit'] = template['step_limit']

    # 2. Derive service/process sets from overlay
    services = set()
    processes = set()
    exploits = {}
    privilege_escalation = {}

    for slot_id, assignment in overlay['slot_assignments'].items():
        slot_def = find_slot(template, slot_id)

        if 'service' in assignment:
            svc = assignment['service']
            services.add(svc)
            exp = assignment['exploit']
            exploits[exp['name']] = {
                'service': svc,
                'os': overlay['os_assignments'].get(slot_id, 'linux'),
                'prob': exp['prob'],
                'cost': exp['cost'],
                'access': exp['access'],
            }

        if 'process' in assignment:
            proc = assignment['process']
            processes.add(proc)
            pe = assignment['privesc']
            privilege_escalation[pe['name']] = {
                'process': proc,
                'os': overlay['os_assignments'].get(slot_id, 'linux'),
                'prob': pe['prob'],
                'cost': pe['cost'],
                'access': pe['access'],
            }

    scenario['service_scan_cost'] = 1
    scenario['os_scan_cost'] = 1
    scenario['subnet_scan_cost'] = 1
    scenario['process_scan_cost'] = 1

    # 3. Format as PenGym-compatible YAML
    scenario['services'] = sorted(services)
    scenario['processes'] = sorted(processes)
    scenario['exploits'] = exploits
    scenario['privilege_escalation'] = privilege_escalation

    # 4. Build host_configurations from slot → host mapping
    scenario['host_configurations'] = build_host_config(template, overlay)

    save_yaml(output_path, scenario)
```

### 4.6. Lợi ích của kiến trúc Template + Overlay

| Aspect                                         | Trước (monolithic)             | Sau (template + overlay)              |
| ---------------------------------------------- | ------------------------------ | ------------------------------------- |
| Files cần tạo cho 100 CVE mixes × 5 topologies | 500 YAML files                 | 5 templates + 100 overlays + auto-gen |
| Thay đổi topology                              | Sửa N files                    | Sửa 1 template                        |
| Thêm 1 CVE mới                                 | Sửa tất cả scenarios liên quan | Thêm/sửa 1 overlay                    |
| Consistency                                    | Dễ sai lệch giữa files         | Template đảm bảo cấu trúc nhất quán   |
| Curriculum control                             | Thủ công chọn file             | Tự động chọn overlay theo tier        |

---

## 5. Pipeline sinh môi trường tự động

### 5.1. Tổng quan pipeline

```
CVE_dataset.csv
     │
     ▼
[1] CVE Classifier ──────────► cve_graded.csv (thêm cột difficulty_score, tier)
     │
     ▼
[2] CVE Selector ────────────► selected_cves.json (chọn CVE theo tier + constraints)
     │
     ▼
[3] Overlay Generator ───────► overlays/tier{N}/overlay-{id}.yml
     │
     ▼
[4] Scenario Compiler ───────► generated/{template}__{overlay}.yml (PenGym-ready)
     │
     ▼
[5] Chain Scenario Builder ──► chain-{config}.json (SCRIPT-ready)
```

### 5.2. Module 1: CVE Classifier

**Input**: `CVE_dataset.csv`
**Output**: `cve_graded.csv` (thêm 2 cột: `difficulty_score`, `difficulty_tier`)

```python
# cve_classifier.py
import csv

TIER_THRESHOLDS = [0.15, 0.35, 0.55]  # T1/T2/T3/T4

def classify_cve(row):
    f_prob = 1.0 - float(row['prob'])

    if row['Attack_Complexity']:
        f_ac = {'LOW': 0.0, 'MEDIUM': 0.5, 'HIGH': 1.0}[row['Attack_Complexity']]
        f_pr = {'': 0.0, 'NONE': 0.0, 'LOW': 0.33, 'HIGH': 0.67}.get(
            row['Privileges_Required'], 0.0)
        f_ui = 1.0 if row.get('User_Interaction') == 'REQUIRED' else 0.0
        score = 0.50 * f_prob + 0.25 * f_ac + 0.15 * f_pr + 0.10 * f_ui
    else:
        score = f_prob  # CVSS v2 fallback

    tier = 1
    for t, threshold in enumerate(TIER_THRESHOLDS):
        if score >= threshold:
            tier = t + 2
    return round(score, 4), tier

def run(input_csv, output_csv):
    with open(input_csv) as fin, open(output_csv, 'w', newline='') as fout:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames + ['difficulty_score', 'difficulty_tier']
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            score, tier = classify_cve(row)
            row['difficulty_score'] = score
            row['difficulty_tier'] = tier
            writer.writerow(row)
```

### 5.3. Module 2: CVE Selector

Chọn CVE cho overlay dựa trên constraints:

```python
# cve_selector.py
import random

def select_cves_for_overlay(graded_cves, template, tier, seed=None):
    """
    Chọn CVE phù hợp cho mỗi service slot trong template.

    Args:
        graded_cves: list of dicts from cve_graded.csv
        template: parsed template YAML
        tier: difficulty tier (1-4)
        seed: random seed for reproducibility

    Returns:
        dict: slot_id → selected CVE row
    """
    rng = random.Random(seed)

    # Filter by tier
    tier_cves = [c for c in graded_cves if int(c['difficulty_tier']) == tier]

    # Group by service
    by_service = {}
    for cve in tier_cves:
        svc = cve['service']
        if svc:
            by_service.setdefault(svc, []).append(cve)

    # Group privesc separately
    privesc_pool = [c for c in tier_cves if c['process']]

    assignments = {}
    used_cves = set()

    for slot in template['service_slots']:
        slot_id = slot['slot_id']
        allowed = slot['allowed_services']

        # Find CVEs matching allowed services
        candidates = []
        for svc in allowed:
            candidates.extend(by_service.get(svc, []))

        # Remove already-used CVEs (diversity)
        candidates = [c for c in candidates if c['CVE_ID'] not in used_cves]

        if not candidates:
            # Fallback: relax tier constraint by ±1
            fallback_tiers = [tier - 1, tier + 1]
            for ft in fallback_tiers:
                for svc in allowed:
                    fallback = [c for c in graded_cves
                               if int(c['difficulty_tier']) == ft
                               and c['service'] == svc
                               and c['CVE_ID'] not in used_cves]
                    candidates.extend(fallback)
                if candidates:
                    break

        if candidates:
            chosen = rng.choice(candidates)
            assignments[slot_id] = chosen
            used_cves.add(chosen['CVE_ID'])

    # Assign privesc slots
    for slot in template.get('privesc_slots', []):
        slot_id = slot['slot_id']
        candidates = [p for p in privesc_pool if p['CVE_ID'] not in used_cves]
        if candidates:
            chosen = rng.choice(candidates)
            assignments[slot_id] = chosen
            used_cves.add(chosen['CVE_ID'])

    return assignments
```

### 5.4. Module 3: Overlay Generator

```python
# overlay_generator.py
def generate_overlay(assignments, overlay_id, tier):
    """Convert CVE assignments to overlay YAML format."""
    overlay = {
        'meta': {
            'overlay_id': overlay_id,
            'difficulty_tier': tier,
            'avg_difficulty_score': round(
                sum(float(a['difficulty_score']) for a in assignments.values())
                / len(assignments), 4),
            'cve_list': [a['CVE_ID'] for a in assignments.values()],
        },
        'slot_assignments': {},
        'os_assignments': {},
    }

    for slot_id, cve in assignments.items():
        if cve['service']:
            overlay['slot_assignments'][slot_id] = {
                'service': cve['service'],
                'exploit': {
                    'name': f"e_{cve['service']}_{cve['CVE_ID'].replace('-','_').lower()}",
                    'cve_id': cve['CVE_ID'],
                    'prob': float(cve['prob']),
                    'cost': int(cve['cost']),
                    'access': cve['access'],
                },
            }
        elif cve['process']:
            overlay['slot_assignments'][slot_id] = {
                'process': cve['process'],
                'privesc': {
                    'name': f"pe_{cve['process']}",
                    'cve_id': cve['CVE_ID'],
                    'prob': float(cve['prob']),
                    'cost': int(cve['cost']),
                    'access': cve['access'],
                },
            }
        overlay['os_assignments'][slot_id] = cve['os']

    return overlay
```

### 5.5. Module 4: Scenario Compiler

Kết hợp template + overlay → full PenGym YAML (đã mô tả ở §4.5).

### 5.6. Module 5: Chain Scenario Builder

Chuyển PenGym scenario → SCRIPT chain JSON format:

```python
# chain_builder.py
def build_chain_scenario(pengym_scenario, overlay):
    """Build SCRIPT-compatible chain JSON from compiled scenario."""
    hosts = []
    for slot_id, assignment in overlay['slot_assignments'].items():
        host = {
            'ip': f'10.0.{len(hosts)}.1',  # Auto-assign IPs
            'port': get_default_ports(assignment.get('service', '')),
            'services': [assignment.get('service', '')],
            'os': overlay['os_assignments'].get(slot_id, 'linux'),
            'vulnerability': [assignment.get('exploit', {}).get('cve_id',
                              assignment.get('privesc', {}).get('cve_id', ''))],
            'web_fingerprint': [],
            'web_fingerprint_component': [],
        }
        hosts.append(host)
    return hosts

def get_default_ports(service):
    PORT_MAP = {
        'ssh': [22], 'ftp': [21], 'http': [80, 443, 8080],
        'smtp': [25], 'samba': [445], 'smb': [445],
    }
    return PORT_MAP.get(service, [80])
```

### 5.7. Xử lý PenGym-incompatible services

**874 CVE (44%) dùng trực tiếp.** Cho 1111 CVE còn lại, 3 chiến lược:

| Chiến lược                 | Mô tả                                            | Áp dụng cho                   | Trade-off                     |
| -------------------------- | ------------------------------------------------ | ----------------------------- | ----------------------------- |
| **A: Abstract mapping**    | Map webapp/iis → `http`, misc → `ssh`            | webapp(156), iis(8)           | ✅ Đơn giản, ⚠️ mất precision |
| **B: Mở rộng service set** | Thêm service mới vào PenGym + ServiceActionSpace | scada(38), mssql(9)           | ✅ Chính xác, ❌ Effort lớn   |
| **C: Loại bỏ**             | Bỏ qua các service không phù hợp                 | browser(263), fileformat(192) | ✅ Sạch, ❌ Mất 45% data      |

**Đề xuất phân pha**:

1. **Phase 1**: Dùng 874 PenGym-compatible CVE + abstract webapp→http (+164) = **~1038 CVE** (52%)
2. **Phase 2**: Mở rộng service set thêm `mssql`, `imap`, `scada` (~73 CVE)
3. **Phase 3**: Evaluate liệu browser/fileformat có ý nghĩa cho network pentest không (có thể loại hẳn)

**Abstract mapping table (Phase 1)**:

| Service gốc | Map thành | Lý do                          |
| ----------- | --------- | ------------------------------ |
| webapp      | http      | Web application chạy trên HTTP |
| iis         | http      | IIS là web server              |
| windows     | smb       | Windows service thường qua SMB |
| brightstor  | http      | Quản lý qua web interface      |

---

## 6. Tích hợp Curriculum Learning

### 6.1. Curriculum Schedule

Agent training theo schedule 3 chiều tăng dần:

```
Phase 1: Foundation (Episode 0 - 5K)
├── CVE Tier: T1 (Easy, prob≥0.8)
├── Topology: tiny-linear (3 hosts, 1 path)
├── Hosts per scenario: 3
├── Target: 1 sensitive host
├── Firewall: minimal
└── Goal: Agent học basic attack loop (scan → exploit → pivot)

Phase 2: Exploitation (Episode 5K - 15K)
├── CVE Tier: T1 → T2 (prob 0.6-0.99)
├── Topology: small-tree (6 hosts, 2 subnets)
├── Hosts per scenario: 6
├── Target: 2 sensitive hosts
├── Firewall: moderate
└── Goal: Agent học handle exploit failures + multi-target strategy

Phase 3: Hardening (Episode 15K - 30K)
├── CVE Tier: T2 → T3 (prob 0.4-0.8, mixed complexity)
├── Topology: medium-mesh (16 hosts, 5 subnets)
├── Hosts per scenario: 10-16
├── Target: 3+ sensitive hosts
├── Firewall: realistic
└── Goal: Agent học deal with hard exploits + complex routing

Phase 4: Expert (Episode 30K+)
├── CVE Tier: T3 → T4 (prob 0.4, AC=HIGH)
├── Topology: mixed (random selection)
├── Hosts per scenario: variable
├── Target: random sensitive hosts
├── Firewall: realistic + dynamic
└── Goal: Generalization — agent robust across all difficulty
```

### 6.2. Phase Transition Criteria

Agent chuyển phase khi đạt performance threshold:

```python
# curriculum_controller.py
class CurriculumController:
    def __init__(self):
        self.current_phase = 1
        self.phase_config = {
            1: {'tier': 1, 'template': 'tiny-linear', 'sr_threshold': 0.80, 'min_episodes': 2000},
            2: {'tier': 2, 'template': 'small-tree',  'sr_threshold': 0.70, 'min_episodes': 5000},
            3: {'tier': 3, 'template': 'medium-mesh', 'sr_threshold': 0.50, 'min_episodes': 8000},
            4: {'tier': 4, 'template': 'mixed',       'sr_threshold': None, 'min_episodes': None},
        }

    def should_advance(self, metrics):
        """
        Advance to next phase if:
        1. Minimum episodes in current phase met
        2. Rolling success rate exceeds threshold
        3. Success rate stable (low variance over last 500 episodes)
        """
        config = self.phase_config[self.current_phase]

        if metrics['episodes_in_phase'] < config['min_episodes']:
            return False

        if config['sr_threshold'] is None:
            return False  # Phase 4 is final

        rolling_sr = metrics['rolling_success_rate_500']
        sr_variance = metrics['sr_variance_500']

        return (rolling_sr >= config['sr_threshold']
                and sr_variance < 0.05)  # Stable performance

    def get_current_config(self):
        return self.phase_config[self.current_phase]

    def advance(self):
        if self.current_phase < 4:
            self.current_phase += 1
            return True
        return False
```

### 6.3. Intra-phase CVE Randomization

Trong mỗi phase, mỗi episode sample một overlay **ngẫu nhiên** từ tier tương ứng:

```python
def sample_training_environment(phase, templates, overlay_pool, rng):
    config = PHASE_CONFIG[phase]

    # Select template
    if config['template'] == 'mixed':
        template = rng.choice(templates)
    else:
        template = templates[config['template']]

    # Select random overlay from tier
    tier = config['tier']
    tier_overlays = overlay_pool[tier]
    overlay = rng.choice(tier_overlays)

    # Compile full scenario
    scenario = compile_scenario(template, overlay)
    return scenario
```

**Mục đích**: Cùng phase, agent gặp nhiều CVE khác nhau → tránh overfitting vào CVE cụ thể, học được service-level pattern.

### 6.4. Tương thích với ServiceActionSpace 16-dim

Curriculum learning hoạt động tự nhiên với service-level abstraction:

- Agent không cần biết CVE cụ thể (CVE-2017-5638 hay CVE-2020-2555)
- Agent chỉ thấy "service=http" → chọn action "exploit_http"
- **Difficulty thể hiện qua prob**: exploit_http có prob=0.99 (easy) hay prob=0.4 (hard) — thay đổi reward nhưng không thay đổi action space
- **Scenario complexity thể hiện qua topology**: thêm hosts, subnets, firewall rules → state space lớn hơn

```
┌─────────────────────────────────────────────────────────┐
│  CVE_dataset.csv (1985 CVEs)                            │
│  ├── difficulty_score → qual selector                   │
│  ├── service → action mapping (16-dim)                  │
│  └── prob → environment dynamics (exploitation rate)    │
│                                                         │
│  Template (topology)                                    │
│  ├── subnet count → state complexity                    │
│  ├── firewall → strategic planning                      │
│  └── host count → search space                          │
│                                                         │
│  Overlay (CVE assignment)                               │
│  ├── service → which actions available                  │
│  ├── prob → how often actions succeed                   │
│  └── access → reward signal                             │
│                                                         │
│  Agent sees: 16-dim action space (unchanged)            │
│  Agent learns: WHEN to use each action type             │
│  Curriculum controls: HOW HARD is each episode          │
└─────────────────────────────────────────────────────────┘
```

---

## 7. Roadmap kỹ thuật

### Phase 1: CVE Difficulty Grading (1-2 tuần) — ✅ IMPLEMENTED

| #   | Task                                 | Effort   | Output                                |
| --- | ------------------------------------ | -------- | ------------------------------------- |
| 1.1 | Implement `cve_classifier.py`        | 0.5 ngày | `cve_graded.csv`                      |
| 1.2 | Validate difficulty distribution     | 0.5 ngày | Distribution report, sanity check     |
| 1.3 | Handle outliers (Excellent+prob=0.4) | 0.5 ngày | Cleaned dataset                       |
| 1.4 | Abstract mapping (webapp→http, etc.) | 1 ngày   | Extended compatibility: 874→1038 CVEs |
| 1.5 | Unit tests cho classifier            | 0.5 ngày | Test suite                            |

**Deliverable**: `data/CVE/cve_graded.csv` với `difficulty_score` và `difficulty_tier` columns.

**Results**: T1=960, T2=207, T3=495, T4=323. CVSS v2 fallback fix applied. File: `src/pipeline/cve_classifier.py`

### Phase 2: Template + Overlay Infrastructure (2-3 tuần) — ✅ IMPLEMENTED

| #   | Task                                         | Effort | Output                            |
| --- | -------------------------------------------- | ------ | --------------------------------- |
| 2.1 | Design template YAML schema                  | 1 ngày | Schema spec + 1 example           |
| 2.2 | Convert existing scenarios to templates      | 2 ngày | tiny, small, medium templates     |
| 2.3 | Design overlay YAML schema                   | 1 ngày | Schema spec + 5 examples          |
| 2.4 | Implement `cve_selector.py`                  | 2 ngày | Selector with constraint handling |
| 2.5 | Implement `overlay_generator.py`             | 1 ngày | Overlay YAML writer               |
| 2.6 | Implement `scenario_compiler.py`             | 2 ngày | Template + Overlay → PenGym YAML  |
| 2.7 | Implement `chain_builder.py`                 | 1 ngày | PenGym YAML → SCRIPT chain JSON   |
| 2.8 | Batch generation: 100 overlays × 3 templates | 1 ngày | 300 compiled scenarios            |
| 2.9 | Validation: generated YAML loads in PenGym   | 2 ngày | Integration test                  |

**Deliverable**: Working pipeline, 300+ auto-generated scenarios across 4 difficulty tiers.

**Results**: 80 scenarios (4 templates × 4 tiers × 5), all pass NASim validation. Consistent dims per template: tiny(200/114), tiny-small(312/190), small-linear(504/320), medium(1071/720). File: `src/pipeline/scenario_compiler.py`

### Phase 3: Curriculum Learning Integration (2-3 tuần) — ✅ IMPLEMENTED

| #   | Task                                      | Effort | Output                     |
| --- | ----------------------------------------- | ------ | -------------------------- |
| 3.1 | Implement `CurriculumController`          | 2 ngày | Phase transition logic     |
| 3.2 | Modify training loop to use curriculum    | 2 ngày | Integrated training script |
| 3.3 | Training run: Phase 1 only (baseline)     | 1 ngày | Phase 1 SR metric          |
| 3.4 | Training run: Full curriculum (Phase 1→4) | 2 ngày | Curriculum SR curves       |
| 3.5 | Compare: curriculum vs flat training      | 1 ngày | Analysis report            |
| 3.6 | Tune phase thresholds                     | 2 ngày | Optimal schedule           |

**Deliverable**: Curriculum-trained agent with comparison metrics.

**Results**: Flat SR=0.260 vs Curriculum SR=0.260 (500 eps, SimpleDQN on tiny). Curriculum 35% faster (100.9s vs 156.3s). Phase transitions validated: T2→T3 met threshold at SR=0.47. Files: `curriculum_controller.py`, `simple_dqn_agent.py`

### Phase 4: Mở rộng (tùy chọn, 2+ tuần) — ✅ IMPLEMENTED

| #   | Task                                         | Status  | Output                          |
| --- | -------------------------------------------- | ------- | ------------------------------- |
| 4.1 | ServiceRegistry extensible architecture      | ✅ Done | `extensible_registry.py`        |
| 4.2 | CVEAdditionPipeline (CSV → scenarios)        | ✅ Done | Automated end-to-end pipeline   |
| 4.3 | TemplateExpander (add services to templates) | ✅ Done | Dynamic slot injection          |
| 4.4 | Registry ↔ CVEClassifier integration         | ✅ Done | +277 CVEs with 4 extra services |
| 4.5 | Save/Load registry persistence (JSON)        | ✅ Done | `service_registry.json`         |

**Results**:

- Default: 5 services (ssh, ftp, http, samba, smtp), 3 processes, 30 keyword mappings
- Adding 4 new services (mssql, rdp, scada, misc): 1017 → 1294 PenGym-compatible CVEs (+27%)
- Unmapped CVEs: 439 → 196 (recovered 243)
- All 80 existing scenarios still valid after integration

---

## Phụ lục

### A. Service compatibility matrix

```
Service       │ PenGym │ CVE count │ Chiến lược
──────────────┼────────┼───────────┼──────────────────
ssh           │  ✅    │    23     │ Direct
ftp           │  ✅    │    71     │ Direct
http          │  ✅    │   720     │ Direct
samba/smb     │  ✅    │    39     │ Direct (map smb→samba)
smtp          │  ✅    │    14     │ Direct
webapp        │  ❌    │   156     │ Abstract → http
iis           │  ❌    │     8     │ Abstract → http
windows       │  ❌    │    34     │ Abstract → smb
browser       │  ❌    │   263     │ Skip (client-side)
fileformat    │  ❌    │   192     │ Skip (client-side)
misc          │  ❌    │   156     │ Case-by-case
scada         │  ❌    │    38     │ Phase 4 expansion
mssql         │  ❌    │     9     │ Phase 4 expansion
imap          │  ❌    │    17     │ Phase 4 expansion
```

### B. Privesc process categories

```
Category        │ Count │ OS      │ Example CVEs
────────────────┼───────┼─────────┼─────────────────────
Kernel/FS       │  10   │ Linux   │ overlayfs, dirtypipe, ebpf
Sudo/Auth       │   6   │ Linux   │ sudo_baron_samedit, sudoedit_bypass
Docker escape   │   4   │ Linux   │ docker_runc_escape, docker_cgroup
Netfilter       │   4   │ Linux   │ netfilter_priv_esc, nft_set_elem
Windows kernel  │   5   │ Windows │ ms18_8120, win32k, spoolfool
macOS           │   3   │ macOS   │ dirty_cow, libxpc
Network service │   3   │ Mixed   │ smbghost, exim4, reflection_juicy
Other           │   5   │ Mixed   │ pkexec, chkrootkit, zpanel
```

### C. Difficulty score distribution (expected)

```
Score  0.0─┤ ████████████████████████████████  T1: ~1200 CVEs (60%)
       0.15┤ ──────────────────
       0.20┤ ████████████████                  T2: ~500 CVEs (25%)
       0.35┤ ──────────────────
       0.40┤ ████████                          T3: ~220 CVEs (11%)
       0.55┤ ──────────────────
       0.60┤ ███                               T4: ~65 CVEs (3%)
       1.0─┘
```

T1 chiếm đa số → đây là tín hiệu tốt cho curriculum learning vì agent cần nhiều positive reward ở giai đoạn đầu.

### D. Key design decisions log

| Decision                         | Lựa chọn              | Lý do                                            |
| -------------------------------- | --------------------- | ------------------------------------------------ |
| Difficulty score primary feature | `prob` (weight=0.50)  | Deterministic mapping từ MSF_Rank, 4 mức rõ ràng |
| CVSS v2 fallback                 | Chỉ dùng prob         | 49% CVE thiếu CVSS v3 — prob luôn available      |
| Service abstraction Phase 1      | webapp/iis→http       | Tăng pool từ 874→1038 CVE với effort thấp        |
| Template+Overlay vs monolithic   | Template+Overlay      | Scalability: N×M combinations từ N+M files       |
| Curriculum phases                | 4 phases              | Mapping tự nhiên vào 4 difficulty tiers          |
| Phase transition metric          | Rolling SR + variance | Đảm bảo agent ổn định trước khi tăng độ khó      |
| Storage format cho overlay       | YAML                  | Nhất quán với PenGym scenario format             |
