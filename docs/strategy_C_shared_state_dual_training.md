# Chiến Lược C: Shared State Format + Dual Training

> Chuẩn hóa State Format → Train CL trên Simulation → Fine-tune trên PenGym

---

## 1. Tổng Quan

### 1.1 Mục Tiêu Nghiên Cứu

Thiết kế và triển khai một **unified state representation** cho phép SCRIPT agent hoạt động nhất quán trên cả hai môi trường (Simulation và PenGym), từ đó thực hiện **dual training**: pre-train trên simulation (nhanh, không giới hạn episodes) rồi fine-tune trên PenGym (chậm, nhưng realistic). Mục tiêu cuối cùng là tạo ra một agent có khả năng:

1. **Generalize across environments:** Policy học được trên simulation vẫn meaningful khi áp dụng trên PenGym
2. **Continual learning across domains:** SCRIPT framework (EWC + Knowledge Distillation) hoạt động across cả sim tasks và real tasks
3. **Measurable transfer quality:** Có metrics rõ ràng để đo lường mức độ transfer thành công

**Khác biệt cốt lõi so với Strategy A:** Strategy A giữ nguyên hai state formats và xây adapter layer; Strategy C **thay đổi state format ở cả hai phía** để chúng nói cùng một ngôn ngữ từ đầu.

### 1.2 Phạm Vi Áp Dụng

| Phạm vi              | Chi tiết                                                                    |
| -------------------- | --------------------------------------------------------------------------- |
| **Thay đổi code**    | State encoding ở cả `pentest/src/agent/host.py` và `pentest/src/envs/core/` |
| **Retraining**       | Cần retrain trên simulation với state format mới                            |
| **Fine-tuning**      | Train tiếp trên PenGym với cùng model                                       |
| **Evaluation**       | So sánh sim-only, pengym-only, và dual-trained agents                       |
| **Phạm vi scenario** | Bắt đầu với scenario có overlap giữa hai môi trường                         |

### 1.3 Giả Định Nền Tảng

> **⚠️ CÁC GIẢ ĐỊNH QUAN TRỌNG — Mỗi giả định sau cần được kiểm chứng trước khi triển khai đầy đủ. Violation bất kỳ giả định nào có thể invalidate toàn bộ approach.**

| #   | Giả định                                                                                                          | Rủi ro sai lệch                                                                                | Cách kiểm chứng                                           |
| --- | ----------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| G1  | SBERT embeddings cho cùng một service string (VD: "ssh") sẽ cho ra vector giống nhau bất kể source (sim hay real) | **Trung bình.** Real scan results có thể chứa version info, noise → SBERT output khác          | Encode cùng string từ cả hai phía, tính cosine similarity |
| G2  | Action semantics tương đồng: "exploit ssh" trong sim và PenGym đều cùng ý nghĩa                                   | **Thấp** cho scan actions, **Trung bình** cho exploits. PenGym sử dụng real Metasploit modules | So sánh action prerequisites, success conditions          |
| G3  | Reward scale có thể chuẩn hóa mà không mất semantics                                                              | **Thấp.** Reward scaling là kỹ thuật standard trong RL                                         | Verify bằng training curves                               |
| G4  | PenGym environment stable đủ để train RL (không crash, deterministic enough)                                      | **Cao.** Real execution có nhiều failure modes: timeout, session loss, network issues          | Chạy 100 episodes random actions, đo failure rate         |
| G5  | State normalization statistics từ sim vẫn hợp lệ khi transfer sang PenGym                                         | **Cao.** Distribution shift gần như chắc chắn xảy ra                                           | So sánh mean/std của states từ hai môi trường             |
| G6  | EWC Fisher information matrix tính trên sim vẫn có ý nghĩa cho PenGym                                             | **Trung bình-Cao.** Fisher matrix phụ thuộc vào state distribution                             | Phân tích Fisher values trên sim vs PenGym states         |
| G7  | SCRIPT CL framework (EWC + KD + Retrospection) hoạt động khi tasks span across domains (sim → real)               | **Chưa xác định.** Đây là contribution nghiên cứu chính                                        | Đây là điều cần chứng minh, không phải giả định           |

---

## 2. Thiết Kế Unified State Representation

### 2.1 Phân Tích Thành Phần State Hiện Tại

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    STATE COMPONENT COMPARISON                           │
├───────────────┬────────────────────────┬────────────────────────────────┤
│ Component     │ SCRIPT (Simulation)    │ PenGym (NASim-based)           │
├───────────────┼────────────────────────┼────────────────────────────────┤
│ Access        │ 2d one-hot             │ 3 scalars (compromised,        │
│               │ [1,0]=comp, [0,1]=reach │  reachable, access_level)     │
├───────────────┼────────────────────────┼────────────────────────────────┤
│ OS Info       │ SBERT("linux") → 384d  │ One-hot [1,0,0] for 'linux'   │
├───────────────┼────────────────────────┼────────────────────────────────┤
│ Port Info     │ SBERT("22,80,443")     │ N/A (implicit in services)     │
│               │ → 384d                 │                                │
├───────────────┼────────────────────────┼────────────────────────────────┤
│ Service Info  │ SBERT("ssh,http")      │ Binary [1,0,1,0,0] per         │
│               │ → 384d                 │ scenario.services              │
├───────────────┼────────────────────────┼────────────────────────────────┤
│ Web FP        │ SBERT avg → 384d       │ N/A                            │
├───────────────┼────────────────────────┼────────────────────────────────┤
│ Processes     │ N/A                    │ Binary [1,0,1] per             │
│               │                        │ scenario.processes             │
├───────────────┼────────────────────────┼────────────────────────────────┤
│ Host Value    │ N/A                    │ Scalar (reward value)           │
├───────────────┼────────────────────────┼────────────────────────────────┤
│ Discovery     │ N/A                    │ Scalar (discovery_value)        │
├───────────────┼────────────────────────┼────────────────────────────────┤
│ Scope         │ Single host            │ All hosts (flat/matrix)         │
├───────────────┼────────────────────────┼────────────────────────────────┤
│ Total Dim     │ 1538                   │ Variable (depends on scenario)  │
└───────────────┴────────────────────────┴────────────────────────────────┘
```

### 2.2 Thiết Kế Shared State Format

**Nguyên tắc thiết kế:**

1. **Semantic preservation:** Giữ nguyên ý nghĩa ngữ nghĩa của mỗi component
2. **Fixed dimensionality:** Dimension cố định bất kể scenario
3. **Source-agnostic:** Cùng một host state phải cho ra cùng vector bất kể nguồn
4. **Information completeness:** Capture đủ thông tin cho cả hai môi trường

**Đề xuất: Hybrid SBERT + Structured Format**

```python
class UnifiedStateEncoder:
    """
    Shared state format cho cả Simulation và PenGym.

    Design rationale:
    - SBERT cho text-based info (OS, services): flexible, handles unseen values
    - Structured vectors cho discrete info (access, discovered): precise, no ambiguity
    - Fixed dimension = 1542 cho mọi scenario

    Format:
    ┌─────────┬──────────┬──────────────┬──────────────┬──────────────┬────────────────┐
    │ Access  │ Discovery│ OS (SBERT)   │ Port (SBERT) │ Service      │ Web/Process    │
    │ 3-dim   │ 1-dim    │ 384-dim      │ 384-dim      │ (SBERT)      │ (SBERT)        │
    │         │          │              │              │ 384-dim      │ 384-dim        │
    └─────────┴──────────┴──────────────┴──────────────┴──────────────┴────────────────┘
                                                                        Total: 1540-dim
    """

    # ═══════════════════════════════════════════════════════════
    # SECTION A: Structured Components (low-dimensional, precise)
    # ═══════════════════════════════════════════════════════════

    ACCESS_DIM = 3      # [unknown, reachable, compromised] one-hot
    DISCOVERY_DIM = 1   # 0.0 = undiscovered, 1.0 = discovered

    # ═══════════════════════════════════════════════════════════
    # SECTION B: SBERT Semantic Components (high-dimensional, flexible)
    # ═══════════════════════════════════════════════════════════

    SBERT_DIM = 384     # all-MiniLM-L6-v2 output dimension
    OS_DIM = SBERT_DIM          # 384
    PORT_DIM = SBERT_DIM        # 384
    SERVICE_DIM = SBERT_DIM     # 384
    AUXILIARY_DIM = SBERT_DIM   # 384 — web_fingerprint OR process info

    TOTAL_DIM = ACCESS_DIM + DISCOVERY_DIM + OS_DIM + PORT_DIM + SERVICE_DIM + AUXILIARY_DIM
    # = 3 + 1 + 384 + 384 + 384 + 384 = 1540

    # Slice indices
    ACCESS_SLICE = slice(0, 3)
    DISCOVERY_SLICE = slice(3, 4)
    OS_SLICE = slice(4, 388)
    PORT_SLICE = slice(388, 772)
    SERVICE_SLICE = slice(772, 1156)
    AUX_SLICE = slice(1156, 1540)
```

### 2.3 Chi Tiết Từng Component

#### 2.3.1 Access Vector (3-dim)

```python
# Thay đổi từ 2-dim (SCRIPT) sang 3-dim (compatible cả hai)
ACCESS_ENCODING = {
    'unknown':     [1, 0, 0],  # Chưa biết gì
    'reachable':   [0, 1, 0],  # Reachable nhưng chưa compromise
    'compromised': [0, 0, 1],  # Đã compromise
}

# Mapping từ PenGym:
# compromised=1 → 'compromised'
# reachable=1, compromised=0 → 'reachable'
# otherwise → 'unknown'

# Mapping từ SCRIPT:
# self.access == "compromised" → 'compromised'
# self.access == "reachable" → 'reachable'
# self.access is None → 'unknown'
```

**Lý do thay đổi:** SCRIPT gốc dùng 2-dim nhưng không encode trạng thái "unknown" (host chưa được discover). PenGym phân biệt rõ 3 trạng thái. Thêm 1 dimension cho compatible.

#### 2.3.2 OS Encoding (384-dim via SBERT)

```python
def encode_os(self, os_info, source: str):
    """
    Encode OS information to SBERT vector.

    CRITICAL: Cần canonicalize OS string trước khi encode.

    Ví dụ:
    - SCRIPT: "Linux 3.2 - 4.9" (từ nmap output)
    - PenGym scan: "Linux 4.15 - 5.6" (từ real nmap)
    - PenGym scenario: one-hot [1,0] với os_names=['linux','windows']

    Canonicalization: → "linux" (lowercase, strip version)
    """
    if source == 'pengym_scenario':
        # From NASim one-hot: reconstruct string
        os_string = self._onehot_to_string(os_info, self.os_names)
    elif source == 'pengym_scan':
        # From real nmap: canonicalize
        os_string = self._canonicalize_os(os_info)
    elif source == 'simulation':
        # From SCRIPT env_data: may already be a string
        os_string = self._canonicalize_os(os_info)

    return self._encode_cached(os_string)

def _canonicalize_os(self, os_string: str) -> str:
    """
    Normalize OS strings cho consistency.

    ⚠️ RISK: Over-canonicalization mất thông tin.
              Under-canonicalization tạo different embeddings cho same OS.

    Rules:
    1. Lowercase
    2. Strip version numbers
    3. Map aliases: 'ubuntu' → 'linux', 'centos' → 'linux'
    """
    os_lower = os_string.lower().strip()

    CANONICAL_MAP = {
        'ubuntu': 'linux',
        'debian': 'linux',
        'centos': 'linux',
        'red hat': 'linux',
        'fedora': 'linux',
        'windows server': 'windows',
        'windows 10': 'windows',
    }

    for pattern, canonical in CANONICAL_MAP.items():
        if pattern in os_lower:
            return canonical

    # Strip version numbers
    import re
    return re.sub(r'[\d\.\-]+', '', os_lower).strip()
```

> **⚠️ RỦI RO TIỀM ẨN #1: OS Canonicalization**
>
> Nếu canonicalization quá aggressive (mọi Linux variant → "linux"), SBERT embedding sẽ giống nhau cho Ubuntu 14.04 và Ubuntu 22.04, trong khi lỗ hổng bảo mật rất khác nhau giữa các phiên bản. Nếu quá conservative, cùng một OS sẽ có embedding khác nhau giữa sim và real.
>
> **Khuyến nghị:** Bắt đầu với aggressive canonicalization, đo impact lên performance. Nếu performance thấp mà root cause là OS confusion, thêm version granularity dần.

#### 2.3.3 Port Encoding (384-dim via SBERT)

```python
def encode_ports(self, port_info, source: str):
    """
    ⚠️ ĐIỂM PHỨC TẠP:

    PenGym KHÔNG tracking port riêng biệt trong NASim observation.
    Port info nằm trong host_map (maintained bởi PenGym utilities).

    SCRIPT tracking port list trực tiếp qua PortScan action.

    Chiến lược:
    - Simulation: Sử dụng port list từ PortScan result, join thành string
    - PenGym: Derive port list từ host_map['services'] + CONFIG.yml service_port mapping

    Output: SBERT("22,80,443,445")
    """
    if source == 'pengym':
        # Derive from host_map services
        port_list = self._services_to_ports(port_info)
    elif source == 'simulation':
        port_list = port_info  # Already a list of port strings

    if port_list:
        port_string = ','.join(sorted(port_list))  # Sort cho consistency
        return self._encode_cached(port_string)
    return np.zeros(self.SBERT_DIM, dtype=np.float32)
```

#### 2.3.4 Service Encoding (384-dim via SBERT)

```python
def encode_services(self, service_info, source: str):
    """
    ⚠️ ĐIỂM PHỨC TẠP: Service naming conventions differ.

    SCRIPT sim: ["ssh", "http", "apache httpd 2.4.49"]  — detailed
    PenGym NASim obs: [1, 0, 1, 0, 0] binary vector cho ['ssh','ftp','http','samba','smtp']
    PenGym host_map: {'ssh': True, 'http': True} — from real nmap
    PenGym real scan: "OpenSSH 7.6p1 Ubuntu 4ubuntu0.3" — từ nmap -sV

    Chiến lược: Canonicalize to base service names, join, SBERT encode.
    """
    if source == 'pengym_obs':
        # From binary vector
        active_services = [name for name, val in
                          zip(self.service_names, service_info) if val > 0]
    elif source == 'pengym_scan':
        # From real nmap output
        active_services = [self._canonicalize_service(s) for s in service_info]
    elif source == 'simulation':
        active_services = [self._canonicalize_service(s) for s in service_info]

    if active_services:
        service_string = ','.join(sorted(active_services))
        return self._encode_cached(service_string)
    return np.zeros(self.SBERT_DIM, dtype=np.float32)
```

> **⚠️ RỦI RO TIỀM ẨN #2: Service Name Mismatch**
>
> Real nmap scan trả về chi tiết: `"Apache httpd 2.4.49 ((Unix))"`. Simulation env_data có thể dùng: `"http"` hoặc `"apache"`. Nếu không canonicalize đúng, SBERT sẽ cho embedding khác nhau dù cùng service.
>
> **Impact:** Policy nhận state vector khác nhau cho cùng 1 tình huống → action selection sai.
>
> **Mitigation:** Xây dựng `service_canonical_map` mapping từ full nmap output → base service name:
>
> ```python
> SERVICE_CANONICAL_MAP = {
>     'openssh': 'ssh',
>     'vsftpd': 'ftp',
>     'proftpd': 'ftp',
>     'apache httpd': 'http',
>     'nginx': 'http',
>     'microsoft-ds': 'samba',
>     'opensmtpd': 'smtp',
>     'postfix': 'smtp',
> }
> ```

#### 2.3.5 Auxiliary Slot (384-dim — Web Fingerprint OR Processes)

```python
def encode_auxiliary(self, aux_info, source: str):
    """
    Slot linh hoạt cho thông tin bổ sung:
    - Simulation: web_fingerprint (từ WebScan)
    - PenGym: processes (từ ProcessScan)

    ⚠️ QUYẾT ĐỊNH THIẾT KẾ QUAN TRỌNG:

    Option A: Luôn dùng cho web_fingerprint (PenGym bỏ qua slot này)
    Option B: Luôn dùng cho processes (SCRIPT bỏ qua slot này)
    Option C: Encode cả hai khi có, average khi cả hai available
    Option D: Split thành 2 slots (tăng dimension lên 1924)

    Khuyến nghị: Option A cho Phase 1 (vì SCRIPT đã dùng web_fingerprint),
    migrate sang Option D nếu process info quan trọng cho PenGym performance.
    """
    ...
```

> **⚠️ RỦI RO TIỀM ẨN #3: Auxiliary Slot Semantics**
>
> Đây là quyết định thiết kế có ảnh hưởng lớn. Nếu chọn Option A (web_fingerprint), thì PenGym sẽ luôn có 384 zeros ở slot này — network phải học cách ignore 384 zeros. Nếu chọn Option C, vector meaning thay đổi tùy context → potential confusion cho policy.
>
> **Khuyến nghị:** Dùng **Option A** ban đầu. PenGym observations sẽ luôn có aux_slot = zeros. Policy sẽ học rằng aux_slot = zeros là normal trên PenGym. Nếu sau này cần process info, tạo slot riêng (Option D).

### 2.4 Validation Của Shared State Format

```python
def validate_state_consistency(sim_encoder, pengym_encoder):
    """
    Test case: Tạo cùng 1 host scenario programmatically,
    encode bằng cả hai encoder, verify output gần nhau.

    Expected: cosine_similarity(sim_state, pengym_state) > 0.9
    cho cùng host info, > 0.95 cho identical strings.
    """
    test_cases = [
        {
            'access': 'reachable',
            'os': 'linux',
            'ports': ['22', '80'],
            'services': ['ssh', 'http'],
        },
        {
            'access': 'compromised',
            'os': 'linux',
            'ports': ['22', '80', '445'],
            'services': ['ssh', 'http', 'samba'],
        },
        # Edge cases
        {
            'access': 'unknown',
            'os': None,
            'ports': [],
            'services': [],
        },
    ]

    for tc in test_cases:
        sim_state = sim_encoder.encode(tc)
        pengym_state = pengym_encoder.encode(tc)
        similarity = cosine_similarity(sim_state, pengym_state)
        assert similarity > 0.9, f"State mismatch: {similarity:.3f} for {tc}"
```

---

## 3. Unified Action Space

### 3.1 Thiết Kế Action Space Chung

Thay vì mapping giữa hai action spaces (Strategy A), Strategy C xây dựng **một action space duy nhất** mà cả hai môi trường cùng sử dụng.

```python
class UnifiedActionSpace:
    """
    Shared action space cho cả Simulation và PenGym.

    ╔═══════════════════════════════════════════════════════════════╗
    ║              UNIFIED ACTION SPACE DESIGN                      ║
    ╠═══════════╦═══════════════════╦════════════════════════════════╣
    ║ Category  ║ Actions           ║ Execution                      ║
    ╠═══════════╬═══════════════════╬════════════════════════════════╣
    ║ Recon     ║ 0: PORT_SCAN      ║ Sim: function call              ║
    ║           ║ 1: SERVICE_SCAN   ║ PenGym: nmap -sS/-sV           ║
    ║           ║ 2: OS_SCAN        ║ PenGym: nmap -O                ║
    ║           ║ 3: WEB_SCAN       ║ Sim: web fingerprint            ║
    ║           ║                   ║ PenGym: process scan fallback   ║
    ╠═══════════╬═══════════════════╬════════════════════════════════╣
    ║ Exploit   ║ 4: e_ssh          ║ Sim: env_data lookup            ║
    ║           ║ 5: e_ftp          ║ PenGym: msf module execution    ║
    ║           ║ 6: e_http         ║                                ║
    ║           ║ 7: e_samba        ║                                ║
    ║           ║ 8: e_smtp         ║                                ║
    ║           ║ 9+: CVE-specific  ║ Sim-only hoặc mapped           ║
    ╠═══════════╬═══════════════════╬════════════════════════════════╣
    ║ PrivEsc   ║ N+0: pe_tomcat    ║ PenGym-specific, sim: no-op    ║
    ║           ║ N+1: pe_proftpd   ║                                ║
    ║           ║ N+2: pe_cron      ║                                ║
    ╚═══════════╩═══════════════════╩════════════════════════════════╝
    """
```

### 3.2 Action Compatibility Layer

```python
class ActionCompatibilityLayer:
    """
    Xử lý trường hợp action tồn tại trong 1 env nhưng không tồn tại trong env kia.

    ⚠️ PHỨC TẠP: SCRIPT hiện tại có ~40+ CVE-specific exploits,
    trong khi PenGym chỉ có 5 service-based exploits + 3 privilege escalation.

    Strategies:

    1. INTERSECTION: Chỉ giữ actions tồn tại ở cả hai
       - Pro: Hoàn toàn compatible
       - Con: Mất nhiều exploits, giảm action space đáng kể

    2. SUPERSET: Giữ tất cả, unmappable → no-op với penalty
       - Pro: Không mất information
       - Con: Agent có thể learn to avoid certain actions chỉ vì env limitation

    3. SEMANTIC GROUPING: Group CVE exploits → service-based exploits
       - Pro: Giữ semantic meaning, reduce redundancy
       - Con: Mất CVE-specific knowledge

    Khuyến nghị: SEMANTIC GROUPING cho Phase 1.
    """

    EXPLOIT_GROUPS = {
        'ssh':   ['CVE-2020-16846', ...],   # Group tất cả SSH-related CVEs
        'ftp':   ['CVE-...', ...],
        'http':  ['CVE-2017-5638', 'CVE-2018-11776', 'CVE-2019-0230', ...],
        'samba': ['CVE-2017-7494', ...],
        'smtp':  ['CVE-...', ...],
    }

    def group_action(self, cve_action_id: int) -> int:
        """Map CVE-specific action → service-group action."""
        ...
```

> **⚠️ RỦI RO TIỀM ẨN #4: Action Space Reduction Loss**
>
> SCRIPT agent hiện tại đã học chọn CVE-specific exploit dựa trên thông tin từ scans. Nếu group thành service-based exploits, agent mất khả năng phân biệt giữa CVE-2017-5638 (Struts2) và CVE-2018-7600 (Drupal) — cả hai đều exploit HTTP service nhưng yêu cầu conditions khác nhau.
>
> **Impact tiềm tàng:**
>
> - **Nếu PenGym chỉ có 1 exploit/service:** Không mất gì, vì PenGym cũng không phân biệt
> - **Nếu PenGym có nhiều exploit/service:** Cần giữ granularity, dùng SUPERSET
>
> **Khuyến nghị:** Bắt đầu với SEMANTIC GROUPING, đo lường sim performance trước/sau grouping. Nếu sim performance drop > 10%, xem xét SUPERSET.

### 3.3 Multi-Host vs Single-Host Scope

```
⚠️ ĐIỂM PHỨC TẠP QUAN TRỌNG NHẤT CỦA STRATEGY C

SCRIPT hiện tại: train agent cho SINGLE HOST tại một thời điểm.
  - Agent nhận state của 1 host, chọn action cho host đó
  - Training loop iterate qua từng target host tuần tự
  - Episode = compromise 1 target

PenGym:         train agent cho ENTIRE NETWORK.
  - Agent nhận observation của toàn bộ network (all hosts)
  - Action = (host_index, action_type) — chọn cả host và hành động
  - Episode = compromise tất cả sensitive hosts

→ Đây KHÔNG CHỈ là state format difference.
  Đây là FUNDAMENTAL TASK STRUCTURE difference.
```

**Các cách tiếp cận:**

#### Option 1: Giữ Single-Host Mode cho PenGym

```python
# Wrap PenGym để hoạt động ở single-host mode
class SingleHostPenGymWrapper:
    """
    Giữ nguyên SCRIPT paradigm: 1 agent → 1 target host tại 1 thời điểm.
    PenGym được dùng như tool execution backend, không phải RL environment.

    Flow:
    1. Chọn target host (given by eval protocol hoặc curriculum)
    2. Agent nhận state của target host (via unified encoder)
    3. Agent chọn action type (scan/exploit)
    4. Action được execute trên target host qua PenGym
    5. Episode ends khi target compromised hoặc step limit

    Pro: Minimal change to SCRIPT, direct comparison with sim baseline
    Con: Không tận dụng multi-host reasoning capability của PenGym
    """

    def __init__(self, pengym_env, target_host_address):
        self.env = pengym_env
        self.target = target_host_address
        self.unified_encoder = UnifiedStateEncoder(pengym_env.scenario)

    def reset(self):
        self.env.reset()
        return self._get_target_state()

    def step(self, action_type_idx):
        # Convert single-host action to PenGym multi-host action
        pengym_action_idx = self._to_pengym_action(action_type_idx, self.target)
        obs, reward, done, truncated, info = self.env.step(pengym_action_idx)
        state = self._get_target_state()
        target_done = self._check_target_compromised()
        return state, reward, target_done, info
```

#### Option 2: Upgrade SCRIPT lên Multi-Host Mode

```python
# Mở rộng SCRIPT state sang multi-host
class MultiHostStateEncoder(UnifiedStateEncoder):
    """
    State = concatenate(per-host state for each host in network)

    Dimension = num_hosts × per_host_dim = num_hosts × 1540
    VD: medium-multi-site với 16 hosts → 16 × 1540 = 24,640 dimensions!

    ⚠️ RISK: State space quá lớn cho MLP [512, 512]
    → Cần redesign network architecture (attention, GNN, etc.)
    → Ngoài scope của Strategy C Phase 1
    """
    pass
```

> **⚠️ RỦI RO TIỀM ẨN #5: Scope Mismatch**
>
> Lựa chọn giữa Option 1 và 2 ảnh hưởng TOÀN BỘ thiết kế.
>
> **Khuyến nghị rõ ràng: Bắt đầu với Option 1 (Single-Host wrapper).** Lý do:
>
> 1. Minimal code change → faster iteration
> 2. Direct comparison with sim baseline (cùng paradigm)
> 3. Strategy C goal là chứng minh CL transfer, không phải multi-host reasoning
> 4. Nếu single-host transfer thành công, mở rộng sang multi-host là next step tự nhiên

---

## 4. Unified Reward Framework

### 4.1 Reward Normalization

```python
class UnifiedRewardNormalizer:
    """
    Chuẩn hóa reward across environments.

    Simulation rewards:     PenGym rewards:
    PORT_SCAN:    0         Scan:       -1 (cost only)
    SERVICE_SCAN: +100      Exploit:    host_value - cost (50-100)
    OS_SCAN:      +100      PrivEsc:    0 - cost
    WEB_SCAN:     +100
    Exploit:      +1000
    Constraint:   -5 to -10

    Normalization strategy: Scale tất cả về [-1, +1] range
    - Max positive (exploit success): +1.0
    - Neutral (no info gain):          0.0
    - Negative (invalid action):      -0.1 to -1.0

    ⚠️ RISK: Reward normalization có thể thay đổi optimal policy!
    Cần verify rằng relative ordering of action values preserved.
    """

    def __init__(self, source: str):
        if source == 'simulation':
            self.max_reward = 1000   # exploit success
            self.min_reward = -10    # constraint violation
        elif source == 'pengym':
            self.max_reward = 100    # sensitive host value
            self.min_reward = -3     # action cost

    def normalize(self, reward: float) -> float:
        if reward > 0:
            return reward / self.max_reward  # [0, 1]
        elif reward < 0:
            return reward / abs(self.min_reward)  # [-1, 0]
        return 0.0
```

> **⚠️ RỦI RO TIỀM ẨN #6: Reward Scale Mismatch Và CL**
>
> EWC Fisher information matrix phụ thuộc vào gradient magnitude, mà gradient magnitude phụ thuộc vào reward scale. Nếu sim rewards (range [-10, 1000]) và PenGym rewards (range [-3, 100]) không được normalize, Fisher matrix từ sim sẽ dominate transitions trên PenGym (weights bị "đóng băng" quá mạnh bởi sim Fisher).
>
> **Mitigation bắt buộc:** Normalize rewards TRƯỚC khi tính Fisher matrix, hoặc scale Fisher matrix khi chuyển domain.

---

## 5. Dual Training Pipeline

### 5.1 Tổng Quan

```
════════════════════════════════════════════════════════════════════════
    DUAL TRAINING PIPELINE
════════════════════════════════════════════════════════════════════════

Phase 1: Pre-training on Simulation (fast)
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Unified State Encoder (sim mode)                                   │
│       ↓                                                             │
│  SCRIPT CL Training:                                                │
│    Task 1 (CVE-1) → Task 2 (CVE-2) → ... → Task N (CVE-N)        │
│    [EWC consolidation between tasks]                                │
│    [Knowledge Distillation from expert to student]                  │
│       ↓                                                             │
│  Output: θ_sim (model weights), Fisher_sim, state_norm_sim          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
Phase 2: Transfer to PenGym (critical transition)
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Load θ_sim                                                         │
│       ↓                                                             │
│  ⚠️ State Normalization Reset/Adaptation:                           │
│     Option A: Reset normalizer, warm-up on PenGym random rollouts   │
│     Option B: Blend sim stats with PenGym stats gradually           │
│     Option C: Use batch normalization instead of running stats      │
│       ↓                                                             │
│  ⚠️ Fisher Information Handling:                                    │
│     Option A: Keep Fisher_sim, apply as constraint on PenGym        │
│     Option B: Discount Fisher_sim by factor β ∈ [0.1, 0.5]         │
│     Option C: Reset Fisher, recompute on PenGym                    │
│       ↓                                                             │
│  Configure PenGym SingleHostWrapper                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
Phase 3: Fine-tuning on PenGym (slow, careful)
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Unified State Encoder (pengym mode)                                │
│       ↓                                                             │
│  Fine-tune with constrained learning:                               │
│    - Lower learning rate (1e-5 vs 1e-4)                            │
│    - EWC constraint từ Phase 1 (giữ sim knowledge)                 │
│    - Optional: KD từ sim-trained policy                            │
│    - Few episodes (PenGym chậm: ~100s/episode)                     │
│       ↓                                                             │
│  Output: θ_dual (dual-trained weights)                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
Phase 4: Evaluation (comprehensive)
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Compare 4 agents:                                                  │
│                                                                     │
│  A. θ_sim_baseline (SCRIPT gốc, SBERT encoding)                    │
│     → Evaluate trên Simulation → Upper bound on sim                 │
│                                                                     │
│  B. θ_sim_unified (SCRIPT retrained, unified encoding)              │
│     → Evaluate trên Simulation → Check encoding change impact       │
│     → Evaluate trên PenGym → Zero-shot transfer                     │
│                                                                     │
│  C. θ_dual (dual-trained)                                           │
│     → Evaluate trên PenGym → Main result                            │
│     → Evaluate trên Simulation → Check forgetting                   │
│                                                                     │
│  D. θ_pengym_scratch (trained from scratch on PenGym)               │
│     → Evaluate trên PenGym → Lower bound reference                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Phase 1 Chi Tiết: Retrain on Simulation

```python
# Thay đổi cần thiết:

# 1. Replace StateEncoder trong host.py
# OLD:
class StateEncoder:
    state_space = access_dim + os_dim + port_dim + service_dim + web_fingerprint_dim
    # = 2 + 384 + 384 + 384 + 384 = 1538

# NEW:
class StateEncoder:
    state_space = UnifiedStateEncoder.TOTAL_DIM  # 1540
    # Thay đổi encoding logic để dùng UnifiedStateEncoder

# 2. Update PPO network dimensions
# Actor: Input(1540) → [512, 512] → Output(action_dim)
# Critic: Input(1540) → [512, 512] → Output(1)
# Thay đổi nhỏ (1538 → 1540), không ảnh hưởng architecture

# 3. Retrain
# Sử dụng cùng scenario chain, cùng hyperparameters
# So sánh convergence speed và final performance với baseline
```

**Kỳ vọng:** Performance trên sim gần bằng baseline (< 5% drop), vì thay đổi state encoding minimal (2 dim extra, canonicalization).

### 5.3 Phase 2 Chi Tiết: Transfer Setup

```python
class DomainTransferManager:
    """
    Quản lý quá trình transfer từ sim → PenGym.

    ⚠️ ĐÂY LÀ PHASE PHỨC TẠP NHẤT VÀ NHIỀU RỦI RO NHẤT
    """

    def __init__(self, sim_agent, pengym_env):
        self.sim_agent = sim_agent
        self.pengym_env = pengym_env

    def transfer(self, strategy='conservative'):
        """
        strategy options:
        - 'aggressive': Keep everything from sim, just change env
        - 'conservative': Reset normalizer, discount Fisher
        - 'cautious': Reset normalizer + Fisher, only transfer weights
        """

        # 1. Copy weights
        pengym_agent = copy.deepcopy(self.sim_agent)

        # 2. Handle state normalization
        if strategy in ['conservative', 'cautious']:
            self._reset_normalizer(pengym_agent)
            self._warmup_normalizer(pengym_agent, num_random_episodes=10)

        # 3. Handle Fisher information
        if strategy == 'cautious':
            self._reset_fisher(pengym_agent)
        elif strategy == 'conservative':
            self._discount_fisher(pengym_agent, discount=0.3)

        # 4. Adjust learning rates
        pengym_agent.config.actor_lr *= 0.1  # 1e-5
        pengym_agent.config.critic_lr *= 0.1  # 5e-6

        return pengym_agent

    def _warmup_normalizer(self, agent, num_random_episodes=10):
        """
        Chạy random policy trên PenGym để collect state statistics.

        ⚠️ RISK: Random policy trên PenGym sẽ generate rất ít
        'compromised' states (vì exploits unlikely to succeed randomly).
        → Normalizer sẽ biased toward non-compromised states.

        Mitigation: Blend sim stats (50%) + PenGym random stats (50%)
        """
        ...

    def _discount_fisher(self, agent, discount=0.3):
        """
        Giảm Fisher information magnitude.

        Lý do: Fisher matrix từ sim encode "importance" of weights
        cho sim tasks. Trên PenGym, importance distribution có thể khác.
        Discount = 0.3 nghĩa là chỉ giữ 30% constraint strength.

        ⚠️ RISK:
        - Discount quá cao (>0.5): Sim knowledge bị lock, PenGym không learn được
        - Discount quá thấp (<0.1): Sim knowledge bị quên nhanh (catastrophic forgetting)
        - Optimal discount: Cần hyperparameter search
        """
        if hasattr(agent, 'cl_agent'):
            for param_name in agent.cl_agent.fisher_dict:
                agent.cl_agent.fisher_dict[param_name] *= discount
```

> **⚠️ RỦI RO TIỀM ẨN #7: State Normalization Distribution Shift**
>
> Đây là rủi ro **rất dễ bị bỏ qua** nhưng **cực kỳ destructive**.
>
> SCRIPT sử dụng running mean/std normalization (`Normalization` class trong `common.py`):
>
> ```python
> # Running normalization
> normalized_state = (state - running_mean) / (running_std + 1e-8)
> ```
>
> Sau 500 episodes × N targets trên simulation, `running_mean` và `running_std` đã converge đến simulation state distribution. Khi chuyển sang PenGym:
>
> - **Nếu giữ nguyên stats:** PenGym states bị normalize bằng sim statistics → distorted input → policy output vô nghĩa
> - **Nếu reset stats:** Policy weights expect normalized inputs theo sim distribution → initial outputs vô nghĩa cho đến khi stats converge lại
>
> **Không có giải pháp hoàn hảo.** Khuyến nghị:
>
> 1. Bắt đầu bằng `cautious` strategy (reset stats)
> 2. Warmup 10-20 episodes random actions (collect PenGym state statistics)
> 3. Sau warmup, bắt đầu fine-tune
> 4. Log state distribution statistics liên tục để monitor

### 5.4 Phase 3 Chi Tiết: Fine-Tuning on PenGym

```python
class PenGymFineTuner:
    """
    Fine-tune SCRIPT agent trên PenGym environment.

    Constraints:
    1. PenGym chậm: ~80-120s/episode → budget = 50-100 episodes max
    2. EWC constraint từ sim training phải được giữ
    3. Knowledge Distillation từ sim policy (teacher = θ_sim)
    4. Learning rate thấp để tránh catastrophic change
    """

    def fine_tune(self, agent, pengym_env, num_episodes=50):
        """
        ⚠️ QUYẾT ĐỊNH QUAN TRỌNG:

        Nên coi PenGym training là "task mới" trong CL framework hay không?

        Option A: YES — PenGym = new task in continual sequence
          - Pro: EWC tự động protect sim knowledge
          - Pro: Consistent với SCRIPT CL framework
          - Con: SCRIPT expect mỗi task = 1 target host. PenGym scenario
                 có thể có nhiều hosts → conflict

        Option B: NO — Fine-tuning đơn thuần (không dùng CL machinery)
          - Pro: Simpler implementation
          - Con: Risk catastrophic forgetting of sim knowledge

        Khuyến nghị: Option A, wrap PenGym scenario thành "task" cho SCRIPT.
        Mỗi PenGym target host = 1 CL task.
        """

        # Treat each PenGym target as a CL task
        target_hosts = self._get_sensitive_hosts(pengym_env)

        for host in target_hosts:
            # Create single-host wrapper
            single_env = SingleHostPenGymWrapper(pengym_env, host)

            # Train on this "task" using SCRIPT CL framework
            agent.cl_agent.train_on_task(
                task_env=single_env,
                task_id=f"pengym_{host}",
                episodes=num_episodes // len(target_hosts),
                # ↑ PenGym time budget split across hosts
            )

            # CL consolidation after each task
            agent.cl_agent.consolidate_knowledge()
```

> **⚠️ RỦI RO TIỀM ẨN #8: Time Budget vs Learning Quality**
>
> Trên simulation, SCRIPT train 500 episodes/task với ~1s/episode → 500 seconds/task.
> Trên PenGym, mỗi episode ~100s. Budget 50 episodes = 5000 seconds ≈ 83 minutes/task.
>
> Nhưng 50 episodes có thể **không đủ để converge** trên PenGym (SCRIPT converge tại episode 39 trên sim, nhưng PenGym stochasticity cao hơn).
>
> **Mitigation:**
>
> 1. Leverage sim-trained weights (không train from scratch)
> 2. Acceptance criterion: SR ≥ 0.5 (không cần 1.0)
> 3. Nếu budget cho phép, tăng lên 100-200 episodes

---

## 6. Evaluation Framework

### 6.1 Ma Trận Đánh Giá

```
                    ┌───────────────────────────────────────────────┐
                    │           EVALUATION MATRIX                    │
                    ├───────────┬──────────┬──────────┬─────────────┤
                    │           │ Sim Eval │ PenGym   │ Interp.     │
                    │           │          │ Eval     │             │
                    ├───────────┼──────────┼──────────┼─────────────┤
                    │ Baseline  │ SR=1.0   │ N/A      │ Upper bound │
                    │ (θ_sim)   │ R=6640   │          │ on sim      │
                    ├───────────┼──────────┼──────────┼─────────────┤
                    │ Unified   │ SR=?     │ SR=?     │ Encoding    │
                    │ (θ_uni)   │ R=?      │ R=?      │ impact      │
                    ├───────────┼──────────┼──────────┼─────────────┤
                    │ Dual      │ SR=?     │ SR=?     │ MAIN RESULT │
                    │ (θ_dual)  │ R=?      │ R=?      │ Transfer +  │
                    │           │          │          │ adaptation  │
                    ├───────────┼──────────┼──────────┼─────────────┤
                    │ Scratch   │ N/A      │ SR=?     │ Lower bound │
                    │ (θ_pg)    │          │ R=?      │ reference   │
                    └───────────┴──────────┴──────────┴─────────────┘
```

### 6.2 Metrics Chi Tiết

```python
EVALUATION_METRICS = {
    # ═══ Performance Metrics ═══
    'success_rate': {
        'description': 'Fraction of episodes where goal reached',
        'target': '>= 0.5 for dual, >= 0.3 for zero-shot',
    },
    'average_return': {
        'description': 'Mean cumulative reward per episode',
        'note': 'Use NORMALIZED reward for cross-env comparison',
    },
    'average_steps': {
        'description': 'Mean steps in successful episodes',
        'note': 'PenGym steps take longer, so compare step COUNT not time',
    },

    # ═══ Transfer Quality Metrics ═══
    'forward_transfer': {
        'formula': 'SR_pengym(θ_dual) - SR_pengym(θ_scratch)',
        'description': 'How much sim pre-training helps PenGym learning',
        'positive_means': 'Sim experience transfers to PenGym',
    },
    'backward_transfer': {
        'formula': 'SR_sim(θ_dual) - SR_sim(θ_sim_baseline)',
        'description': 'How much PenGym fine-tuning hurts sim performance',
        'negative_means': 'Catastrophic forgetting occurring',
    },
    'transfer_ratio': {
        'formula': 'SR_pengym(θ_dual) / SR_sim(θ_sim_baseline)',
        'description': 'Overall transfer efficiency',
    },

    # ═══ CL-Specific Metrics ═══
    'ewc_constraint_satisfaction': {
        'description': 'How much weights deviate from sim-optimal under EWC',
        'formula': 'Σ F_i × (θ_dual_i - θ_sim_i)² / Σ F_i',
    },
    'knowledge_retention_score': {
        'description': 'Average performance on sim tasks after PenGym training',
        'formula': 'mean(SR_sim_task_i(θ_dual)) for i in sim_tasks',
    },
    'knowledge_acquisition_score': {
        'description': 'Performance on PenGym tasks',
        'formula': 'mean(SR_pengym_task_j(θ_dual)) for j in pengym_tasks',
    },

    # ═══ State Encoding Quality ═══
    'state_consistency_score': {
        'description': 'Cosine similarity of states for same host across envs',
        'target': '> 0.85',
    },
    'state_distribution_overlap': {
        'description': 'KL divergence between sim and PenGym state distributions',
        'note': 'Lower = better alignment',
    },
}
```

### 6.3 Statistical Rigor

```python
EXPERIMENT_PROTOCOL = {
    'seeds': [0, 42, 123, 456, 789],  # 5 random seeds minimum
    'confidence_interval': 0.95,
    'min_eval_episodes': 20,  # Per seed, per scenario

    'statistical_tests': {
        'paired_comparison': 'Wilcoxon signed-rank test',
        'multi_comparison': 'Friedman test + Nemenyi post-hoc',
        'effect_size': "Cohen's d",
    },

    'report_format': {
        'tables': 'Mean ± Std, p-value, effect size',
        'plots': 'Learning curves, box plots, gap analysis bars',
    },
}
```

---

## 7. Phân Tích Rủi Ro Toàn Diện

### 7.1 Risk Registry

| ID  | Rủi ro                                                                | Probability | Impact   | Mitigation                                         | Detection                                        |
| --- | --------------------------------------------------------------------- | ----------- | -------- | -------------------------------------------------- | ------------------------------------------------ |
| R1  | OS canonicalization quá aggressive → mất version info → wrong exploit | Medium      | High     | Configurable canonicalization levels               | Monitor exploit success rate per OS variant      |
| R2  | Service name mismatch giữa sim và real nmap                           | High        | High     | Comprehensive `SERVICE_CANONICAL_MAP`              | Unit test cross-environment state encoding       |
| R3  | State normalization distribution shift phá hủy policy                 | High        | Critical | Reset + warmup strategy, monitor KL divergence     | Plot state distribution histograms               |
| R4  | EWC Fisher matrix từ sim quá strong → PenGym không learn              | Medium      | High     | Configurable discount factor, grid search          | Monitor PenGym training loss + SR                |
| R5  | PenGym execution failures → noisy rewards → poor learning             | High        | Medium   | Retry mechanism, filter outlier episodes           | Log PenGym error rate, flag episodes with errors |
| R6  | Action space mismatch → agent chọn invalid actions                    | Medium      | Medium   | Compatibility layer + action masking               | Track unmappable action rate                     |
| R7  | Reward scale difference → biased CL consolidation                     | Medium      | High     | Normalize rewards before all computations          | Compare gradient magnitudes across domains       |
| R8  | Insufficient PenGym training budget (50-100 episodes)                 | High        | Medium   | Leverage sim pre-training, lower SR target         | Learning curve analysis                          |
| R9  | PenGym scenario structure khác sim → OOD situation                    | Medium      | High     | Choose overlapping scenarios carefully             | Compare scenario feature coverage                |
| R10 | SBERT model version/weights khác nhau giữa environments               | Low         | Critical | Pin SBERT model version, load from same checkpoint | Verify model hash at startup                     |

### 7.2 Risk Dependency Graph

```
R3 (State Norm Shift) ─────→ R4 (Fisher Too Strong) ─────→ R8 (Budget)
        ↓                            ↓                         ↓
    Policy fails              Knowledge locked           No convergence
    on PenGym                 Can't adapt
        ↓                            ↓                         ↓
    ════════════════════════════════════════════════════════════════
                    STRATEGY C FAILS ENTIRELY
    ════════════════════════════════════════════════════════════════

R1 (OS Canon.) ──→ R2 (Service Mismatch) ──→ R6 (Action Invalid)
        ↓                   ↓                        ↓
    Wrong SBERT          State differs          Agent confused
    embedding            between envs
        ↓                   ↓                        ↓
    ════════════════════════════════════════════════════════════════
                DEGRADED PERFORMANCE (recoverable)
    ════════════════════════════════════════════════════════════════
```

### 7.3 Contingency Plans

**Scenario 1: State encoding produces very different distributions**

```
IF state_consistency_score < 0.7:
  → Investigate which component causes divergence
  → Option A: Add domain-specific normalization per component
  → Option B: Train a small alignment network: sim_state → pengym_state
  → Option C: Switch to raw binary/structured encoding (abandon SBERT for this)
```

**Scenario 2: EWC prevents any learning on PenGym**

```
IF pengym_SR after 50 episodes ≈ 0 AND pengym_loss not decreasing:
  → Reduce ewc_lambda: 2000 → 200 → 20
  → If still no learning: reset Fisher entirely, use only weight transfer
  → Last resort: fine-tune without EWC (accept sim forgetting)
```

**Scenario 3: PenGym too unstable for RL training**

```
IF PenGym error_rate > 20% OR episode crash_rate > 10%:
  → Add comprehensive try/catch in PenGym wrapper
  → Skip and re-try failed episodes (don't update policy on errors)
  → If fundamental instability: use PenGym for eval-only, train only on sim
```

---

## 8. Kế Hoạch Triển Khai Chi Tiết

### Phase 0: Validation Experiments (1-2 tuần)

| Step | Mục tiêu                    | Output                   | Tiêu chí pass                        |
| ---- | --------------------------- | ------------------------ | ------------------------------------ |
| 0.1  | SBERT consistency test      | Cosine sim scores        | sim vs pengym > 0.9 cho cùng strings |
| 0.2  | PenGym stability test       | Error rate, crash rate   | < 5% error episodes                  |
| 0.3  | State distribution analysis | Histograms, KL scores    | Identify specific divergence sources |
| 0.4  | Scenario overlap analysis   | Feature comparison table | ≥ 1 scenario có significant overlap  |

### Phase 1: Unified Encoder + Sim Retrain (1-2 tuần)

| Step | Mục tiêu                                       | Output                               | Tiêu chí pass                      |
| ---- | ---------------------------------------------- | ------------------------------------ | ---------------------------------- |
| 1.1  | Implement `UnifiedStateEncoder`                | Code + unit tests                    | All tests pass                     |
| 1.2  | Implement `SingleHostPenGymWrapper`            | Code + integration test              | PenGym episode runs                |
| 1.3  | Integrate unified encoder into SCRIPT training | Modified `host.py` + training script | Training starts                    |
| 1.4  | Retrain on sim with unified encoding           | θ_uni model, training curves         | SR ≥ 0.95, < 5% drop from baseline |
| 1.5  | Compare θ_uni vs θ_baseline on sim             | Comparison report                    | Quantified encoding impact         |

### Phase 2: Zero-Shot Transfer (3-5 ngày)

| Step | Mục tiêu                                      | Output                    | Tiêu chí pass                   |
| ---- | --------------------------------------------- | ------------------------- | ------------------------------- |
| 2.1  | Deploy θ_uni trên PenGym (no training)        | Eval results              | Episodes complete without crash |
| 2.2  | Gap analysis                                  | Report                    | Identified failure modes        |
| 2.3  | State distribution comparison (sim vs PenGym) | Visualization + KL scores | Quantified distribution shift   |

### Phase 3: Dual Training (2-3 tuần)

| Step | Mục tiêu                              | Output                 | Tiêu chí pass                        |
| ---- | ------------------------------------- | ---------------------- | ------------------------------------ |
| 3.1  | Implement `DomainTransferManager`     | Code                   | Transfer pipeline works              |
| 3.2  | Implement `PenGymFineTuner`           | Code                   | Fine-tuning loop runs                |
| 3.3  | Hyperparameter search cho transfer    | Best configs           | ≥ 3 settings tested                  |
| 3.4  | Full dual training run                | θ_dual model           | Training completes                   |
| 3.5  | Compare θ_dual vs θ_uni vs θ_baseline | Full evaluation matrix | SR_pengym(θ_dual) > SR_pengym(θ_uni) |

### Phase 4: Analysis (1-2 tuần)

| Step | Mục tiêu                              | Output                       | Tiêu chí pass              |
| ---- | ------------------------------------- | ---------------------------- | -------------------------- |
| 4.1  | Forward/Backward transfer analysis    | Transfer scores              | FT > 0 (positive transfer) |
| 4.2  | CL mechanism analysis (EWC/KD effect) | Ablation study               | Each component contributes |
| 4.3  | Statistical significance tests        | p-values, effect sizes       | p < 0.05 for main claims   |
| 4.4  | Write results documentation           | `docs/strategy_c_results.md` | Complete analysis          |

---

## 9. Thiết Kế Output Structure

```
pentest/outputs/
├── logs_baseline_sim/                    # Baseline SCRIPT (đã rename)
│   └── chain/
│       ├── baseline_cl_script_40tasks_seed42/
│       └── baseline_standard_6targets_seed42/
│
├── models_baseline_sim/                  # Baseline models (đã rename)
│   └── chain/
│       └── chain-msfexp_vul-sample-6_envs-seed_0/
│
├── tensorboard_baseline_sim/             # Baseline tensorboard (đã rename)
│
├── logs/                                 # Kết quả mới
│   ├── strategy_a/                       # Strategy A results
│   │   ├── zero_shot_eval/
│   │   └── finetune_eval/
│   │
│   └── strategy_c/                       # Strategy C results
│       ├── phase0_validation/
│       │   ├── sbert_consistency_test.json
│       │   ├── pengym_stability_test.json
│       │   └── state_distribution_analysis/
│       ├── phase1_sim_retrain/
│       │   ├── unified_encoding_training_curves/
│       │   └── baseline_comparison.json
│       ├── phase2_zero_shot/
│       │   ├── pengym_eval_results.json
│       │   └── gap_analysis.json
│       ├── phase3_dual_training/
│       │   ├── transfer_settings/
│       │   ├── finetune_curves/
│       │   └── hyperparam_search/
│       └── phase4_analysis/
│           ├── transfer_metrics.json
│           ├── ablation_study.json
│           └── statistical_tests.json
│
├── models/
│   ├── strategy_a/
│   └── strategy_c/
│       ├── theta_unified/                # Phase 1 output
│       ├── theta_dual/                   # Phase 3 output
│       └── theta_pengym_scratch/         # Reference model
│
└── tensorboard/
    └── strategy_c/
```

---

## 10. Lưu Ý Quan Trọng Cho Người Triển Khai

### 10.1 Kiểm Tra Trước Khi Bắt Đầu

- [ ] **SBERT model version:** Verify cùng model checkpoint được dùng ở cả `pentest/src/agent/nlp/Encoder.py` và bất kỳ PenGym integration nào. Model khác = embedding khác = transfer thất bại.
- [ ] **PenGym environment stability:** Chạy ≥ 50 random episodes, ghi nhận error rate. Nếu > 10%, debug PenGym trước.
- [ ] **Scenario compatibility:** Verify scenario file PenGym sử dụng có cùng services, exploits, topology structure mà sim scenario expect.
- [ ] **CUDA memory:** SBERT model (~100MB) + PPO networks + PenGym overhead → kiểm tra GPU memory đủ.

### 10.2 Debugging Priorities

Khi gặp vấn đề, kiểm tra theo thứ tự:

1. **State vector giá trị hợp lý?** Print và inspect raw state vectors từ cả sim và PenGym. Không nên có NaN, Inf, hoặc giá trị cực đoan.
2. **State normalization statistics?** Print running_mean và running_std. Nếu chúng diverge mạnh giữa sim warmup và PenGym execution → distribution shift issue.
3. **Action selection distribution?** Print softmax output của policy. Nếu entropy ≈ 0 (policy quá confident) hoặc entropy ≈ log(action_dim) (uniform random) → training problem.
4. **Reward scale?** Print raw rewards từ PenGym. Verify chúng nằm trong expected range sau normalization.
5. **EWC loss magnitude?** Nếu EWC loss >> task loss → EWC quá mạnh → giảm lambda.

### 10.3 Giả Định Rủi Ro Nhất (Cần Kiểm Chứng Đầu Tiên)

**Giả định G4 (PenGym stability) và G5 (Normalization transfer) là hai giả định dễ sai lệch nhất và có impact lớn nhất.** Nên ưu tiên kiểm chứng hai giả định này trong Phase 0 trước khi đầu tư effort vào implementation.

Nếu G4 false (PenGym không stable): Strategy C dùng PenGym cho eval-only, không fine-tune.
Nếu G5 false (Normalization hoàn toàn không transfer): Cần batch normalization hoặc domain-adaptive normalization approach thay vì running stats.

---

## 11. Tham Chiếu File Quan Trọng

| Mục đích            | File                                                                                                             | Ghi chú                                |
| ------------------- | ---------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| State encoding gốc  | `pentest/src/agent/host.py` class `StateEncoder`                                                                 | Cần modify                             |
| SBERT encoder       | `pentest/src/agent/nlp/Encoder.py`                                                                               | Giữ nguyên, dùng chung                 |
| PPO policy          | `pentest/src/agent/policy/PPO.py`                                                                                | Update input_dim: 1538→1540            |
| SCRIPT CL framework | `pentest/src/agent/continual/Script.py`                                                                          | Core CL logic, modify for cross-domain |
| CL config           | `pentest/src/agent/policy/config.py` class `Script_Config`                                                       | Add transfer-specific params           |
| Training agent      | `pentest/src/agent/agent.py` class `Agent`                                                                       | Modify state_norm handling             |
| CL training agent   | `pentest/src/agent/agent_continual.py` class `Agent_CL`                                                          | Entry for CL training                  |
| PenGym environment  | `PenGym/pengym/envs/environment.py` class `PenGymEnv`                                                            | Wrap, don't modify                     |
| PenGym host vector  | `PenGym/pengym/envs/host_vector.py` class `PenGymHostVector`                                                     | Reference for obs format               |
| PenGym config       | `PenGym/pengym/CONFIG.yml`                                                                                       | Service-port mapping                   |
| PenGym scenarios    | `PenGym/database/scenarios/*.yml`                                                                                | Choose matching scenarios              |
| Pentest scenarios   | `pentest/data/scenarios/`                                                                                        | Existing sim scenarios                 |
| Baseline experiment | `pentest/outputs_baseline_sim/logs_baseline_sim/chain/baseline_standard_6targets_seed42/experiment_summary.json` | Reference metrics                      |
| Action definitions  | `pentest/src/agent/actions/`                                                                                     | Unified action space source            |
| State normalization | `pentest/src/agent/policy/common.py` class `Normalization`                                                       | Critical for transfer                  |
| Replay buffer       | `pentest/src/agent/policy/common.py` class `ReplayBuffer_PPO`                                                    | May need domain-aware variant          |
