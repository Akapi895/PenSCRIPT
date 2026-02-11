# Roadmap & Kiến trúc mở rộng CVE cho PenGym + SCRIPT

> **Phiên bản:** 1.0 — Ngày tạo: 2026-02-11  
> **Phạm vi:** Toàn bộ hệ thống Fusion (PenGym + SCRIPT Agent + Service Action Space)  
> **Mục đích:** Tài liệu tổng hợp các đầu việc, lập luận thiết kế, và định hướng mở rộng CVE lâu dài.

---

## Mục lục

- [I. Tổng kết các việc cần làm (Roadmap)](#i-tổng-kết-các-việc-cần-làm-roadmap)
- [II. Giải thích lựa chọn 16 dimensions](#ii-giải-thích-lựa-chọn-16-dimensions)
- [III. Phân tích vấn đề hiện tại liên quan đến CVE](#iii-phân-tích-vấn-đề-hiện-tại-liên-quan-đến-cve)
- [IV. Chiến lược mở rộng CVE (chưa chỉnh sửa code)](#iv-chiến-lược-mở-rộng-cve-chưa-chỉnh-sửa-code)
- [V. Cải tiến SCRIPT để mở rộng với CVE lớn](#v-cải-tiến-script-để-mở-rộng-với-cve-lớn)
- [VI. Tổng kết & Định hướng dài hạn](#vi-tổng-kết--định-hướng-dài-hạn)

---

## I. Tổng kết các việc cần làm (Roadmap)

### 1. Data — CVE, Metadata, Scenario

| # | Việc cần làm | Vì sao |
|---|---|---|
| D1 | Chuẩn hóa schema cho `actions.json` | File hiện tại (~2060 CVE) có trường không nhất quán: một số entry có `exp_info` dạng list of dict, một số có dạng dict rỗng, một số hoàn toàn thiếu `setting`. Điều này gây fail khi phân loại CVE vào service group. |
| D2 | Bổ sung trường `target_service` vào mỗi CVE entry | Hiện tại phân loại CVE→service phải dùng keyword matching (~78% coverage). Thêm trường tường minh sẽ đưa coverage lên 100%. |
| D3 | Xây pipeline tự động thu thập CVE metadata từ NVD/Metasploit | Khi CVE tăng từ 2060 → 5000+, việc thu thập thủ công không khả thi. Cần script crawl NVD API + Metasploit module DB để extract service, port, rank, OS target. |
| D4 | Tạo thêm scenario JSON cho các topology mạng khác nhau | Hiện chỉ có `chain-msfexp_vul-sample-6_envs` (6 host) và `sample-40_envs` (40 host). Cần scenario với topology phức tạp hơn (nhiều subnet, firewall rules) để agent học generalize. |
| D5 | Tách PenGym scenario YAML thành template + CVE overlay | Scenario YAML hiện hardcode danh sách `exploits` và `services`. Cần tách thành: network topology template + CVE/service overlay file, để cùng một topology có thể test với nhiều bộ CVE khác nhau. |

### 2. Environment / Simulation

| # | Việc cần làm | Vì sao |
|---|---|---|
| E1 | Hoàn thiện `ServiceActionMapper` cho PenGym evaluation | Đã tạo `ServiceActionSpace` (16-dim) và `match` strategy. Cần tạo file `service_action_mapper.py` để map service-level action + target host → PenGym flat action index. |
| E2 | Xây `SingleHostPenGymWrapper` cho Strategy C | SCRIPT train trên single-target, PenGym là multi-host. Cần wrapper tách PenGym thành từng target riêng biệt — giữ nguyên giao diện `reset()` → state, `step(action)` → (state, reward, done). |
| E3 | Xây reward normalization layer | SCRIPT dùng reward scale 1000 (success) / -10 (cost), PenGym dùng 100 (sensitive host value) / -1 (cost). Cần layer chuẩn hóa về cùng scale để policy transfer không bị bias. |
| E4 | Kiểm chứng PenGym real execution trên KVM | PenGym có vulnerable service resources (vsftpd-2.3.4, httpd-2.4.49, samba-4.5.9, opensmtpd-6.6.1p1, proftpd-1.3.3) nhưng chưa xác nhận chạy end-to-end với agent tự động. Cần test trên môi trường CyRIS thực. |
| E5 | Thêm dynamic difficulty vào NASim scenarios | Tất cả exploit probability hiện = 0.999999 (near-deterministic). Cần scenario với prob thấp hơn (0.3–0.8) để agent học policy robust hơn, tránh overfit. |

### 3. Agent / State / Action / Reward

| # | Việc cần làm | Vì sao |
|---|---|---|
| A1 | Train agent với service-level action space (16-dim) trên nhiều scenario | Đã verify trên `chain-6_envs` (100% SR từ episode 1). Cần train trên `chain-40_envs` và `all_scenario_msf-41` để đánh giá scalability. |
| A2 | Implement Strategy A evaluation end-to-end | `run_eval_service_level.py` đã tạo. Cần: (a) tạo `service_action_mapper.py`, (b) chạy trên PenGym `tiny.yml` / `medium.yml`, (c) so sánh SR giữa CVE-level (0%) và service-level. |
| A3 | Implement Strategy C dual training | Pre-train trên SCRIPT sim (service-level) → Fine-tune trên PenGym (NASim mode) với EWC regularization. Cần: unified state format (1540-dim), Fisher information matrix, dual training loop. |
| A4 | Implement continual learning với service-level actions | Tích hợp `ServiceActionSpace` vào `Agent_CL` (EWC + Knowledge Distillation). Policy distillation đặc biệt đơn giản hơn vì teacher/student cùng 16-dim output. |
| A5 | Benchmark: so sánh học với 16-dim vs 2064-dim | Train cả hai cấu hình trên cùng scenario, so sánh: convergence speed, final SR, sample efficiency. Dùng làm evidence cho lựa chọn 16-dim. |
| A6 | Khảo sát tác động của CVE selector strategy | So sánh `match` (oracle, training only) vs `rank` vs `random` với cùng policy. `match` cho upper bound, `rank` cho realistic performance — gap giữa hai chính là "CVE selection loss". |

### 4. Script & Automation

| # | Việc cần làm | Vì sao |
|---|---|---|
| S1 | Tạo script benchmark tự động (train → eval → report) | Hiện phải chạy từng bước thủ công. Cần `run_benchmark.py` chạy: train service-level → eval trên PenGym → generate comparison table. |
| S2 | CI/CD cho codebase (lint, type check, unit test) | Codebase thiếu test. Đặc biệt cần unit test cho `ServiceActionSpace._classify_exploit()` — nếu keyword matching sai, toàn bộ training sẽ bị ảnh hưởng. |
| S3 | Tạo config file cho hyperparameter tuning | Hyperparameter hiện nằm rải rác: `PPO_Config` (hardcode), `config.ini` (SBERT model path), scenario YAML. Cần unified config (YAML hoặc Hydra) để quản lý experiment. |
| S4 | Dockerize môi trường training | PenGym dependency phức tạp (NASim, python-nmap, pymetasploit3, CyRIS). Docker container giúp reproduce kết quả. |

---

## II. Giải thích lựa chọn 16 dimensions

### 16-dim đại diện cho điều gì

Mỗi dimension tương ứng với **một loại hành động tấn công ở mức service** — cấp độ trừu tượng mà cả pentester thực tế lẫn PenGym đều hoạt động:

```
Idx  Action              Category   PenGym equiv
───  ──────────────────  ─────────  ────────────
 0   port_scan           scan       subnet_scan
 1   service_scan        scan       service_scan
 2   os_scan             scan       os_scan
 3   web_scan            scan       process_scan
 4   exploit_ssh         exploit    e_ssh
 5   exploit_ftp         exploit    e_ftp
 6   exploit_http        exploit    e_http
 7   exploit_smb         exploit    e_samba
 8   exploit_smtp        exploit    e_smtp
 9   exploit_rdp         exploit    —
10   exploit_sql         exploit    —
11   exploit_java_rmi    exploit    —
12   exploit_misc        exploit    —
13   privesc_tomcat      privesc    pe_tomcat
14   privesc_schtask     privesc    pe_schtask
15   privesc_daclsvc     privesc    pe_daclsvc
```

**Cấu trúc: 4 scan + 9 exploit + 3 privesc = 16.**

### Lý do chọn 16 — Trade-off analysis

**Bài toán gốc:** SCRIPT có 2064 actions (4 scan + 2060 CVE exploit). PenGym có ~18 actions (4 scan + 5 exploit + 3 privesc, nhân với số host). Cần chọn action dimension `d` sao cho:

1. **Biểu diễn đủ chiến lược tấn công** — mỗi action phải mang semantics tấn công khác biệt.
2. **Agent có thể học được** — `d` quá lớn → exploration khó, sample inefficient.
3. **Tương thích PenGym** — mapping 1:1 với PenGym actions.

**Tại sao đúng 16:**

| Số dim | Ưu điểm | Nhược điểm | Kết luận |
|--------|---------|------------|----------|
| **8** (chỉ scan + 4 exploit phổ biến) | Rất dễ học. Exploration nhanh. | Mất phân biệt giữa SSH/FTP/SMB → policy không học được "scan rồi exploit đúng service". RDP, SQL exploit bị gộp vào misc → mất thông tin. | ❌ Quá thô. Agent không biết chọn exploit theo service. |
| **12** (4 scan + 5 exploit + 3 privesc) | Khớp hoàn toàn PenGym medium.yml. | RDP/SQL/Java RMI bị gộp → khi mở rộng ra real-world scenario có những service này, phải retrain. Không có bucket cho unclassified CVE. | ⚠️ Đủ cho PenGym hiện tại, thiếu cho tương lai. |
| **16** (4 scan + 9 exploit + 3 privesc) | Khớp PenGym (13/16 mappable). Cover top-7 services + misc. Có chỗ cho RDP/SQL/RMI khi PenGym mở rộng. Misc bucket cho CVE không phân loại. | 3 action (rdp, sql, java_rmi) hiện chưa có PenGym equiv → wasted capacity nhỏ. | ✅ **Cân bằng tốt nhất.** |
| **24+** (thêm telnet, vnc, ldap, dns, imap...) | Cover nhiều service hơn. | Phần lớn service thêm vào chỉ có 2–6 CVE (ví dụ: telnet=6, vnc=3, ldap=2). Agent gần như không bao giờ chọn những action này → sparse gradient, lãng phí capacity. Exploration space tăng 50%, convergence chậm tương ứng. | ❌ Diminishing returns nghiêm trọng. |

**Kết luận:**  
16 là điểm cân bằng nơi mỗi service group chứa đủ CVE để training signal có ý nghĩa (ssh=35, ftp=70, http=1313, smb=40, smtp=46), trong khi action space đủ nhỏ để PPO converge nhanh (episode 1 đã đạt 100% SR trên 6-target scenario). Ba action dự phòng (rdp, sql, java_rmi) không gây hại cho training hiện tại (policy sẽ tự học bỏ qua) nhưng sẵn sàng cho khi PenGym mở rộng.

### Phân phối CVE thực tế qua 16 nhóm

```
exploit_http     │████████████████████████████████████████████████████  1313 CVEs (63.7%)
exploit_misc     │██████████████████                                    456      (22.1%)
exploit_ftp      │███                                                    70      ( 3.4%)
exploit_smtp     │██                                                     46      ( 2.2%)
exploit_smb      │██                                                     40      ( 1.9%)
exploit_ssh      │██                                                     35      ( 1.7%)
privesc_tomcat   │██                                                     31      ( 1.5%)
exploit_java_rmi │█                                                      25      ( 1.2%)
exploit_sql      │█                                                      19      ( 0.9%)
privesc_schtask  │█                                                      17      ( 0.8%)
exploit_rdp      │                                                        4      ( 0.2%)
privesc_daclsvc  │                                                        4      ( 0.2%)
```

HTTP chiếm 63.7% — điều này phản ánh thực tế: phần lớn CVE thực tế là web-based. `exploit_misc` (22.1%) chứa CVE không phân loại được bằng keyword (local privilege escalation, buffer overflow, format string...). Khi bổ sung trường `target_service` tường minh (mục D2), con số misc sẽ giảm đáng kể.

---

## III. Phân tích vấn đề hiện tại liên quan đến CVE

### 3.1. PenGym: Service-level abstraction — đúng đắn nhưng tĩnh

**Thiết kế hiện tại:**  
PenGym định nghĩa exploit trong scenario YAML:

```yaml
exploits:
  e_ssh:
    service: ssh
    os: linux
    prob: 0.999999
    cost: 3
    access: user
```

Exploit `e_ssh` thành công nếu target host có **service ssh**. Không quan tâm CVE cụ thể nào, OpenSSH version nào. Đây là cấp độ trừu tượng phù hợp cho NASim simulation.

**Vấn đề:**

| Hạn chế | Hệ quả |
|---------|--------|
| Danh sách exploit/service cố định trong mỗi YAML | Thêm service mới (RDP, SQL) đòi hỏi sửa YAML + thêm vulnerable VM resource + sửa CyRIS config. Chi phí cao. |
| Không có khái niệm "CVE version" | Không thể đánh giá agent trên cùng service nhưng khác vulnerability (ví dụ: SSH brute-force vs SSH key exchange bug). Mọi SSH exploit đều "cùng một action". |
| Exploit probability = 0.999999 (hardcode) | Agent không học xử lý failure. Trong real-world, exploit thành công phụ thuộc version, patch level, network condition. Policy trở nên overconfident. |
| Chỉ 5 service + 3 process | PenGym chỉ cover 8/16 service action. Khi mở rộng, phải thêm resource (vulnerable software), CyRIS template, firewall rule — tất cả đều manual. |

### 3.2. SCRIPT: CVE-level action space — granular nhưng không scalable

**Thiết kế hiện tại:**  
`Action.py` load toàn bộ 2060 CVE từ `actions.json` thành `legal_actions`. PPO Actor output layer = 2064 neurons.

**Vấn đề:**

| Hạn chế | Hệ quả |
|---------|--------|
| Output layer = `len(CVE_DB)` | Thêm 100 CVE mới → phải retrain toàn bộ policy từ đầu (architecture change). Không thể incremental update. |
| Exploit chỉ thành công khi target host có CHÍNH XÁC CVE đó | Agent phải "đoán" target vulnerable với CVE nào trong 2060 lựa chọn. Với 50 step limit, xác suất random hit = 50/2060 = 2.4%. Cực kỳ sample-inefficient. |
| `action_constraint()` dùng if-elif chain | 5 case cho scan, 1 case cho exploit. Thêm action type mới (ví dụ: lateral movement, data exfil) đòi hỏi sửa cả constraint logic. |
| CVE metadata chất lượng không đồng đều | 437/2060 CVE (21.2%) thiếu hoàn toàn `setting` và `exp_info` → không phân loại được service. Training signal cho `exploit_misc` rất noisy. |
| Không có semantic grouping | Agent xử lý CVE-2017-5638 (Apache Struts, HTTP) và CVE-2017-7494 (Samba, SMB) bằng cùng một cách: hai neuron riêng biệt, không shared representation. Knowledge từ việc exploit HTTP service A không transfer sang HTTP service B. |

### 3.3. Rủi ro khi CVE tăng

```
Số CVE        | Hệ quả với kiến trúc hiện tại
──────────────┼────────────────────────────────────────────────────
2,060 (hiện)  │ Training converge ~200 eps. Manageable.
5,000         │ Actor output layer 5004 neurons. Softmax trên 5004 dim →
              │ gradient rất sparse → convergence chậm 3-5x.
              │ actions.json ~ 50MB. Load time tăng.
10,000        │ Exploration gần như bất khả thi với random policy.
              │ Memory cho replay buffer tăng (mỗi transition lưu
              │ action prob vector 10004-dim).
              │ PPO batch update chậm đáng kể.
20,000+       │ Kiến trúc hiện tại KHÔNG HOẠT ĐỘNG.
              │ Cần hierarchical / option framework bắt buộc.
```

**Với Service-Level Action Space (16-dim), tất cả con số trên không thay đổi bất kể số CVE.**

---

## IV. Chiến lược mở rộng CVE (chưa chỉnh sửa code)

### 4.1. Tách CVE ra khỏi code chính

**Quyết định: CÓ. CVE phải là DATA, không phải CODE.**

Hiện tại CVE nằm ở hai nơi:
- `actions.json` (2060 entries) — đã là data, nhưng schema lỏng
- PenGym scenario YAML (exploit definitions) — embed trong config

**Kiến trúc đề xuất:**

```
data/
  cve_registry/
    registry.json          ← Master CVE list (id, name, target_service, rank, port, os)
    groups/
      ssh.json             ← CVE details cho service SSH
      ftp.json             
      http.json            
      smb.json             
      ...
  scenarios/
    topologies/
      tiny-3host.yml       ← Network topology ONLY (subnets, hosts, firewall)
      medium-16host.yml    
    cve_overlays/
      web-focused.yml      ← CVE mix: 70% HTTP, 15% FTP, 15% SSH
      mixed-enterprise.yml ← CVE mix: equal distribution
```

**`registry.json` schema:**

```json
{
  "version": "1.0",
  "generated": "2026-02-11",
  "source": "MSF 6.4.50 + NVD",
  "entries": [
    {
      "id": "CVE-2017-5638",
      "msf_module": "exploit/multi/http/struts2_content_type_ognl",
      "target_service": "http",
      "target_port": 8080,
      "target_os": ["linux", "windows"],
      "rank": "excellent",
      "access_level": "user",
      "cvss_score": 10.0,
      "tags": ["rce", "web", "struts"]
    }
  ]
}
```

**Lợi ích:**
- `target_service` tường minh → classification 100%, không cần keyword matching
- Thêm CVE mới = thêm entry vào JSON + thêm vào group file
- Không sửa bất kỳ file code nào
- Có thể version control, diff, review CVE changes riêng biệt

### 4.2. Scenario — khi nào tạo mới, khi nào tái sử dụng

**Scenario là gì trong PenGym:**

Scenario YAML định nghĩa **mạng ảo hoàn chỉnh**: network topology + host configurations + firewall rules + available services/exploits + sensitive targets. NASim sử dụng scenario để tạo MDP: state space, action space, transition function, reward function đều derived từ scenario.

**Nguyên tắc quyết định:**

| Tiêu chí | Thêm CVE vào scenario cũ | Tạo scenario mới |
|----------|--------------------------|-------------------|
| Topology mạng | Giữ nguyên | Thay đổi (thêm subnet, firewall) |
| Service types | Service đã có trong YAML | Service mới (RDP, SQL) cần thêm |
| Mục đích | Test CVE mới trên cùng topology | Test generalization trên topology khác |
| Ảnh hưởng đến action space | Không thay đổi (service-level) | Có thể thay đổi nếu service mới |

**Quy trình đề xuất:**

```
Bước 1: CVE mới thuộc service đã có? (ssh/ftp/http/smb/smtp)
         ├─ CÓ → Thêm vào cve_registry, KHÔNG sửa scenario YAML.
         │        ServiceActionSpace auto-group vào service group.
         │        PenGym scenario vẫn dùng e_ssh (service-level).
         │
         └─ KHÔNG → Service mới cần thêm:
              Bước 2: Thêm service vào SERVICE_ACTION_DEFS (tăng dim)
              Bước 3: Tạo PenGym resource (vulnerable VM software)
              Bước 4: Tạo scenario YAML mới HOẶC clone + thêm service
              Bước 5: Retrain policy (dim thay đổi → architecture change)
```

**Quan trọng:** Với kiến trúc service-level, thêm CVE cho service đã có KHÔNG đòi hỏi thay đổi gì ngoài data. Chỉ khi thêm **service type hoàn toàn mới** mới cần retrain.

### 4.3. Mapping CVE → State / Action / Reward

**Nguyên tắc: Agent thấy service, không thấy CVE.**

```
           SCRIPT simulation                PenGym environment
           ┌──────────────┐                ┌──────────────────┐
State:     │ SBERT encode │                │ NASim obs decode │
           │ "ssh,http"   │                │ [0,1,0,1,0]→     │
           │ → 384-dim    │                │ "ssh,http"→384d  │
           └──────────────┘                └──────────────────┘
                   ▼                               ▼
           ┌──────────────┐                ┌──────────────────┐
Action:    │ Policy → 4   │                │ Policy → 4       │
           │ (exploit_ssh)│                │ (exploit_ssh)    │
           └──────────────┘                └──────────────────┘
                   ▼                               ▼
           ┌──────────────┐                ┌──────────────────┐
CVE Select:│ ssh group →  │                │ e_ssh → NASim    │
           │ CVE-2020-... │                │ service check    │
           └──────────────┘                └──────────────────┘
                   ▼                               ▼
Reward:    │ +1000 (hit)  │                │ +100 (sensitive) │
           │ -10  (cost)  │                │ -1   (cost)      │
```

**Chuẩn hóa mapping:**

1. **State:** Agent nhận thông tin service-level (`"ssh, http on ports 22, 80"`) qua SBERT encoding. Không nhận CVE ID hay version string. → Policy học "nếu thấy SSH service, dùng exploit_ssh".

2. **Action:** Policy output = service-level action index (0..15). CVE Selector (heuristic, không liên quan RL) chọn CVE cụ thể. → Policy hoàn toàn CVE-agnostic.

3. **Reward:** Chuẩn hóa về cùng scale giữa SCRIPT sim và PenGym:
   ```
   normalized_reward = (raw_reward - min_reward) / (max_reward - min_reward)
   ```
   Hoặc đơn giản hơn: scale SCRIPT reward ÷10 (1000→100, -10→-1).

---

## V. Cải tiến SCRIPT để mở rộng với CVE lớn

### 5.1. Vấn đề hiện tại

| Vấn đề | Vị trí | Chi tiết |
|--------|--------|----------|
| Actor output = `Action.action_space` (hardcode) | `PPO.py` dòng khởi tạo Actor | `Actor(StateEncoder.state_space, Action.action_space, ...)` — tham chiếu trực tiếp biến class-level, không configurable. |
| `action_constraint()` là if-elif | `Action.py` | 5 case cố định cho 4 scan + 1 exploit type. Thêm action type mới phải sửa function. |
| `HOST.step()` là if-elif | `host.py` | 7 branch: PORT_SCAN, OS_SCAN, SERVICE_SCAN, PORT_SERVICE_SCAN, WEB_SCAN, Exploit. Mỗi branch tạo riêng PortScan/OSScan/Exploit object. |
| `StateEncoder` dimensions hardcode | `host.py` class-level attributes | `access_dim=2`, `os_dim=384`, ... tính ở import time. Không thể thay đổi runtime. |
| `Exploit.simulate_act()` kiểm tra exact CVE match | `Exploit.py` | `if vul in self.env_data['vulnerability']` — exploit chỉ thành công khi target có đúng CVE. Đúng cho CVE-level sim, nhưng block service-level training. |

### 5.2. Kiến trúc đề xuất: Data-Driven + Plugin-Like

```
                    ┌─────────────────────────┐
                    │     Config Registry      │  (YAML/JSON)
                    │  ┌─────────────────────┐ │
                    │  │ cve_registry.json    │ │  ← CVE database
                    │  │ service_groups.json  │ │  ← Service → CVE mapping
                    │  │ action_config.yml    │ │  ← Action space definition
                    │  │ reward_config.yml    │ │  ← Reward scaling
                    │  └─────────────────────┘ │
                    └────────────┬────────────┘
                                 │ load at startup
                    ┌────────────▼────────────┐
                    │   ServiceActionSpace     │
                    │  ┌──────────┐  ┌───────┐ │
                    │  │ Tier 1:  │  │Tier 2:│ │
                    │  │RL Policy │  │CVE Sel│ │
                    │  │ (16-dim) │  │ector  │ │
                    │  └──────────┘  └───────┘ │
                    └────────────┬────────────┘
                                 │
               ┌─────────────────┼────────────────┐
               │                 │                 │
    ┌──────────▼──────┐  ┌──────▼──────┐  ┌──────▼──────────┐
    │  ScanExecutor   │  │ExploitExec  │  │ PrivescExecutor  │
    │  (port, svc,    │  │ (ssh, ftp,  │  │ (tomcat, cron,   │
    │   os, web)      │  │  http, ...) │  │  daclsvc)        │
    └─────────────────┘  └─────────────┘  └──────────────────┘
    Plugin interface: execute(target_info, env_data) → (success, info)
```

**Nguyên tắc thiết kế:**

1. **Data-driven:** Action space, CVE grouping, reward values đều đọc từ config file. Không hardcode trong Python code.

2. **Plugin-like executor:** Mỗi service type là một executor plugin. Thêm RDP exploit = thêm `RDPExploitExecutor` + đăng ký trong config. Không sửa `HOST.step()`.

3. **Strategy pattern cho CVE selection:** `select_cve()` nhận strategy object. Training dùng `MatchStrategy` (oracle), evaluation dùng `RankStrategy` (heuristic). Có thể thêm `MLStrategy` (learned selector) sau.

4. **Immutable action dim across CVE changes:** Policy luôn output 16-dim. Chỉ CVE Selector layer thay đổi khi thêm/xóa CVE.

### 5.3. Lộ trình cải tiến từng bước

**Phase 0 — Đã hoàn thành (hiện tại):**
- [x] `ServiceActionSpace` class với 16-dim action space
- [x] CVE keyword-based classification (~78% coverage)
- [x] `match` strategy cho training (exact CVE match via env_data)
- [x] `PPO_agent` chấp nhận custom `state_dim` / `action_dim`
- [x] Training pipeline `run_train_service_level.py` verified (100% SR)

**Phase 1 — Data consolidation (ước tính: 1–2 tuần):**
- [ ] Tạo `cve_registry.json` với trường `target_service` tường minh cho mỗi CVE
- [ ] Migrate `actions.json` → `cve_registry/` (schema mới)
- [ ] `ServiceActionSpace` đọc từ registry thay vì keyword matching
- [ ] Unit test cho CVE classification: verify 100% CVE có `target_service`

**Phase 2 — ServiceActionMapper + Strategy A eval (1–2 tuần):**
- [ ] Tạo `service_action_mapper.py` (map service action + host → PenGym flat index)
- [ ] Chạy Strategy A evaluation trên `tiny.yml`, `medium.yml`
- [ ] So sánh kết quả với CVE-level evaluation (đã có: 0% SR)
- [ ] Document kết quả, xác nhận service-level > CVE-level transfer

**Phase 3 — Plugin executor (2–3 tuần):**
- [ ] Tách `HOST.step()` if-elif → Executor registry
- [ ] Mỗi action type: class kế thừa `BaseExecutor` với method `execute()`
- [ ] `action_constraint()` → data-driven rules (config file, không if-elif)
- [ ] Backward compatible: `HOST.step()` delegate sang executor registry

**Phase 4 — Strategy C + Continual Learning (3–4 tuần):**
- [ ] Unified state encoder (1540-dim) cho cả SCRIPT sim và PenGym
- [ ] `SingleHostPenGymWrapper` + reward normalization
- [ ] EWC-based fine-tuning trên PenGym
- [ ] Tích hợp CL pipeline (`Agent_CL`) với ServiceActionSpace

**Phase 5 — Scale test (2–3 tuần):**
- [ ] Crawl thêm CVE từ NVD/MSF → 5000+ entries
- [ ] Verify: training performance KHÔNG thay đổi (action dim vẫn = 16)
- [ ] Verify: CVE Selector quality (rank strategy accuracy)
- [ ] Stress test: 40-host scenario, continual learning trên 20+ tasks

---

## VI. Tổng kết & Định hướng dài hạn

### Hướng phát triển

```
HIỆN TẠI                         NGẮN HẠN                    DÀI HẠN
─────────────────────────────────────────────────────────────────────────
Research prototype               Verified framework           Scalable platform
  • CVE-level actions (2064)       • Service-level (16)         • Dynamic service discovery
  • Single scenario                • Multi-scenario             • Auto-generated scenarios
  • Manual CVE curation            • Registry-based             • NVD/MSF auto-sync
  • SCRIPT sim only                • SCRIPT + PenGym NASim      • PenGym real (KVM)
  • Single-task training           • Continual learning         • Multi-agent, adversarial
  • Hardcoded dimensions           • Configurable dims          • Adaptive architecture
```

### Nguyên tắc thiết kế — tránh technical debt

1. **CVE là data, không phải code.**  
   Mọi thông tin CVE-specific (target service, port, rank, module path) nằm trong JSON/YAML. Python code xử lý ở cấp service-level. Thêm 1000 CVE = thêm 1000 dòng JSON, 0 dòng Python.

2. **Action space dimension là architectural decision, không phải data-driven.**  
   Dim = 16 được chọn dựa trên phân tích service taxonomy. Chỉ thay đổi khi thêm **loại service mới** (hiếm), không khi thêm CVE (thường xuyên). Viết rõ trong code: `SERVICE_ACTION_DEFS` là constant; `cve_indices` trong mỗi group là dynamic.

3. **Tách concern: RL policy ≠ CVE selection.**  
   RL policy (Tier 1) học chiến lược: "scan → exploit đúng service → privesc". CVE Selector (Tier 2) là heuristic/rule-based: "trong SSH group, chọn CVE có rank cao nhất match port target". Hai module có lifecycle, evaluation metric, và update frequency hoàn toàn khác nhau.

4. **Backward compatibility giữa các phase.**  
   Mỗi phase phải chạy được independent. Phase 1 (data consolidation) không yêu cầu Phase 3 (plugin executor). Baseline CVE-level training (`run.py`) vẫn hoạt động song song với service-level training (`run_train_service_level.py`).

5. **Test trước khi refactor.**  
   Trước khi sửa `HOST.step()` (Phase 3), phải có benchmark số liệu trên kiến trúc hiện tại. Sau refactor, so sánh: nếu SR / convergence speed thay đổi > 5%, investigate trước khi merge.

6. **Config, không hardcode.**  
   Mọi magic number (reward scale, step limit, exploit probability, SBERT model name, hidden layer size) phải nằm trong config file duy nhất. Developer mới nhìn vào 1 file là thấy toàn bộ hyperparameter.

---

> **Ghi chú cho developer:**  
> File này dùng làm tài liệu tham chiếu chính cho dự án. Khi hoàn thành một đầu mục, đánh dấu `[x]` trong Phase tương ứng (mục V.3) và ghi ngày hoàn thành. Khi thêm đầu mục mới, thêm vào bảng tương ứng (mục I) với lý do rõ ràng.
