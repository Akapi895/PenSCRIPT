"""
Action Mapper — Map SCRIPT action indices → PenGym NASim flat action indices.

SCRIPT action space:
  [PORT_SCAN(0), SERVICE_SCAN(1), OS_SCAN(2), WEB_SCAN(3), Exploit_0(4), ..., Exploit_N]
  Total: 4 scans + N CVE-based exploits = action_dim (e.g. 2064)

PenGym NASim action space (flat mode):
  Each action = (action_name, target_host)
  e.g., (service_scan, (1,0)), (e_ssh, (2,0)), (pe_tomcat, (3,0))
  Indexed sequentially: 0, 1, 2, ..., env.action_space.n - 1

Mapping strategy: SEMANTIC SIMILARITY
  PORT_SCAN    → service_scan  (both discover services/ports)
  SERVICE_SCAN → service_scan
  OS_SCAN      → os_scan
  WEB_SCAN     → process_scan  (best-effort - no web_scan in NASim)
  CVE exploit  → scenario-specific exploit (matched by target service)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger as logging


class ActionMapper:
    """Map SCRIPT policy actions to PenGym NASim environment actions."""

    # Mapping from SCRIPT scan names → NASim action name patterns
    SCAN_MAP = {
        'Port Scan': 'service_scan',       # PORT_SCAN → service_scan
        'Service Scan': 'service_scan',     # SERVICE_SCAN → service_scan
        'OS Detect': 'os_scan',             # OS_SCAN → os_scan
        'Web Scan': 'process_scan',         # WEB_SCAN → process_scan (proxy)
    }

    # Mapping from CVE exploit target service → PenGym exploit name
    # These map service keywords found in CVE exp_info to NASim scenario exploits
    SERVICE_TO_EXPLOIT_MAP = {
        'ssh': 'e_ssh',
        'ftp': 'e_ftp',
        'vsftp': 'e_ftp',
        'http': 'e_http',
        'apache': 'e_http',
        'httpd': 'e_http',
        'smtp': 'e_smtp',
        'opensmtp': 'e_smtp',
        'samba': 'e_samba',
        'smb': 'e_samba',
        'struts': 'e_http',
        'tomcat': 'pe_tomcat',
        'proftpd': 'pe_daclsvc',  # PenGym uses pe_daclsvc for proftpd
        'cron': 'pe_schtask',     # PenGym uses pe_schtask for cron
    }

    def __init__(self, script_actions, pengym_env):
        """
        Args:
            script_actions: The Action class (from src.agent.actions.Action)
                            containing legal_actions, Scan_actions, All_EXP, etc.
            pengym_env: PenGym environment instance (has .action_space)
        """
        self.script_actions = script_actions
        self.env = pengym_env
        self.action_space = pengym_env.action_space

        # Build the mapping table
        self._build_pengym_action_index()
        self._build_script_to_pengym_map()

        # Stats
        self.total_mapped = 0
        self.total_unmapped = 0
        self.unmapped_log: List[str] = []

    def _build_pengym_action_index(self):
        """Index all PenGym actions by (action_name, target_host) for fast lookup."""
        self.pengym_actions = {}  # (action_name, target_tuple) → flat_index
        self.pengym_action_names = set()
        self.pengym_exploit_names = set()
        self.pengym_privesc_names = set()
        self.pengym_scan_names = set()

        for i in range(self.action_space.n):
            action = self.action_space.get_action(i)
            name = action.name
            target = tuple(action.target)
            self.pengym_actions[(name, target)] = i
            self.pengym_action_names.add(name)

            # Categorize
            if name.startswith('e_'):
                self.pengym_exploit_names.add(name)
            elif name.startswith('pe_'):
                self.pengym_privesc_names.add(name)
            elif name.endswith('_scan'):
                self.pengym_scan_names.add(name)

        print(f"[ActionMapper] PenGym action space: {self.action_space.n} actions")
        print(f"  Scans: {sorted(self.pengym_scan_names)}")
        print(f"  Exploits: {sorted(self.pengym_exploit_names)}")
        print(f"  PrivEsc: {sorted(self.pengym_privesc_names)}")

    def _build_script_to_pengym_map(self):
        """Build mapping from each SCRIPT action index → PenGym action name."""
        self.action_name_map: Dict[int, str] = {}
        mapped_count = 0
        unmapped_count = 0

        for idx, action in enumerate(self.script_actions.legal_actions):
            pengym_name = None

            # Scan actions
            if action.name in self.SCAN_MAP:
                candidate = self.SCAN_MAP[action.name]
                if candidate in self.pengym_action_names:
                    pengym_name = candidate
                else:
                    # Try alternatives
                    for alt_name in self.pengym_scan_names:
                        if candidate.replace('_', '') in alt_name.replace('_', ''):
                            pengym_name = alt_name
                            break

            # Exploit actions
            elif action.type and action.type != "Scan":
                pengym_name = self._match_exploit(action)

            if pengym_name:
                self.action_name_map[idx] = pengym_name
                mapped_count += 1
            else:
                unmapped_count += 1

        total = len(self.script_actions.legal_actions)
        coverage = mapped_count / total * 100 if total > 0 else 0
        print(f"[ActionMapper] Mapping coverage: {mapped_count}/{total} "
              f"({coverage:.1f}%), unmapped: {unmapped_count}")

    def _match_exploit(self, action) -> Optional[str]:
        """Match a SCRIPT CVE exploit to a PenGym exploit/privesc by target service.

        Uses action.vulnerability list, action.name, and action.exp_info
        to determine which service the exploit targets.
        """
        # Strategy 1: Check action name for service keywords
        name_lower = action.name.lower()
        for keyword, pengym_name in self.SERVICE_TO_EXPLOIT_MAP.items():
            if keyword in name_lower:
                if pengym_name in self.pengym_action_names:
                    return pengym_name

        # Strategy 2: Check exp_info for service/module info
        if action.exp_info:
            exp_str = str(action.exp_info).lower()
            for keyword, pengym_name in self.SERVICE_TO_EXPLOIT_MAP.items():
                if keyword in exp_str:
                    if pengym_name in self.pengym_action_names:
                        return pengym_name

        # Strategy 3: Check setting for target port → service inference
        if action.setting:
            setting_str = str(action.setting).lower()
            for keyword, pengym_name in self.SERVICE_TO_EXPLOIT_MAP.items():
                if keyword in setting_str:
                    if pengym_name in self.pengym_action_names:
                        return pengym_name

            # Check port numbers
            if '22' in setting_str or 'ssh' in setting_str:
                if 'e_ssh' in self.pengym_action_names:
                    return 'e_ssh'
            if '21' in setting_str:
                if 'e_ftp' in self.pengym_action_names:
                    return 'e_ftp'
            if '80' in setting_str or '8080' in setting_str:
                if 'e_http' in self.pengym_action_names:
                    return 'e_http'
            if '445' in setting_str:
                if 'e_samba' in self.pengym_action_names:
                    return 'e_samba'
            if '25' in setting_str:
                if 'e_smtp' in self.pengym_action_names:
                    return 'e_smtp'

        return None

    def map_action(self, script_action_idx: int,
                   target_host: Tuple[int, int]) -> int:
        """Convert (SCRIPT action index, target host) → PenGym flat action index.

        Args:
            script_action_idx: Index in SCRIPT's legal_actions list
            target_host: (subnet_id, host_id) tuple for the target in PenGym

        Returns:
            PenGym flat action index, or -1 if no valid mapping exists
        """
        if script_action_idx not in self.action_name_map:
            self.total_unmapped += 1
            action_name = self.script_actions.legal_actions[script_action_idx].name \
                if script_action_idx < len(self.script_actions.legal_actions) else "UNKNOWN"
            self.unmapped_log.append(
                f"Unmappable: SCRIPT[{script_action_idx}]={action_name} → target={target_host}")
            return -1

        pengym_action_name = self.action_name_map[script_action_idx]
        key = (pengym_action_name, target_host)

        if key in self.pengym_actions:
            self.total_mapped += 1
            return self.pengym_actions[key]

        # Target host might not have this action available
        # Try subnet_scan which doesn't need a specific host target
        if pengym_action_name == 'subnet_scan':
            # subnet_scan targets subnet, not host
            for (name, target), idx in self.pengym_actions.items():
                if name == 'subnet_scan' and target[0] == target_host[0]:
                    self.total_mapped += 1
                    return idx

        self.total_unmapped += 1
        self.unmapped_log.append(
            f"No PenGym action for ({pengym_action_name}, {target_host})")
        return -1

    def get_random_valid_action(self, target_host: Tuple[int, int]) -> int:
        """Get a random valid PenGym action for the given target host.

        Used as fallback when SCRIPT action can't be mapped.
        """
        valid = []
        for (name, target), idx in self.pengym_actions.items():
            if target == target_host:
                valid.append(idx)
        if valid:
            return np.random.choice(valid)
        # If no actions for this specific host, sample from entire space
        return self.action_space.sample()

    def get_all_actions_for_host(self, target_host: Tuple[int, int]) -> Dict[str, int]:
        """Get all available PenGym actions for a specific host.

        Returns:
            Dict mapping action_name → flat_action_idx
        """
        result = {}
        for (name, target), idx in self.pengym_actions.items():
            if target == target_host:
                result[name] = idx
        return result

    def get_mapping_stats(self) -> Dict:
        """Get statistics about action mapping quality."""
        total_script = len(self.script_actions.legal_actions)
        mapped = len(self.action_name_map)
        return {
            'total_script_actions': total_script,
            'mapped_actions': mapped,
            'unmapped_actions': total_script - mapped,
            'coverage_pct': mapped / total_script * 100 if total_script > 0 else 0,
            'total_pengym_actions': self.action_space.n,
            'total_mapped_calls': self.total_mapped,
            'total_unmapped_calls': self.total_unmapped,
            'valid_call_rate': (self.total_mapped / max(self.total_mapped + self.total_unmapped, 1)) * 100,
            'pengym_scans': sorted(self.pengym_scan_names),
            'pengym_exploits': sorted(self.pengym_exploit_names),
            'pengym_privesc': sorted(self.pengym_privesc_names),
        }

    def describe(self) -> str:
        """Human-readable mapping description."""
        lines = [
            "=== ActionMapper Configuration ===",
            f"SCRIPT actions: {len(self.script_actions.legal_actions)}",
            f"PenGym actions: {self.action_space.n}",
            f"Mapped: {len(self.action_name_map)}",
            "",
            "--- Scan Mapping ---",
        ]
        for script_name, pengym_name in self.SCAN_MAP.items():
            status = "✓" if pengym_name in self.pengym_action_names else "✗"
            lines.append(f"  {script_name:20s} → {pengym_name:20s} [{status}]")

        lines.append("")
        lines.append("--- Exploit Mapping (by index) ---")
        for idx, pengym_name in sorted(self.action_name_map.items()):
            if idx >= 4:  # Skip scans (indices 0-3)
                script_name = self.script_actions.legal_actions[idx].name
                lines.append(f"  [{idx:4d}] {script_name:40s} → {pengym_name}")

        return "\n".join(lines)
