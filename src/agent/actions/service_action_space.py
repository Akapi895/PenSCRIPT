"""
Service-Level Action Abstraction for SCRIPT Agent.

Replaces the 2064-dim CVE-level action space with a ~16-dim service-level
action space. This solves three critical problems:

  1. PenGym Compatibility:  100% action mapping (vs. 3.4% with CVE-level)
  2. CVE Scalability:       New CVEs auto-group by service, no retraining
  3. Strategy C Ready:      Unified action space for both environments

Architecture:
  ┌──────────────┐
  │  RL Policy   │  Outputs service-level action (dim ≈ 16)
  │  (PPO Actor) │  e.g., "exploit_ssh", "exploit_http"
  └──────┬───────┘
         │
  ┌──────▼────────┐
  │ CVE Selector  │  Picks best specific CVE from the service group
  │ (Heuristic)   │  e.g., ssh group → CVE-2020-14145 (rank=excellent)
  └──────┬────────┘
         │
  ┌──────▼────────┐
  │ Environment   │  SCRIPT sim: execute specific CVE via MSF
  │               │  PenGym:     execute service-level action directly
  └───────────────┘

Usage:
    from src.agent.actions.service_action_space import ServiceActionSpace

    sas = ServiceActionSpace()
    print(sas.action_dim)        # 16
    print(sas.action_names)      # ['port_scan', 'service_scan', ...]

    # During training (SCRIPT simulation):
    service_action_idx = policy.select_action(state)   # 0..15
    cve_action = sas.select_cve(service_action_idx, host_info)
    next_state, reward, done, info = host.step(cve_action.original_idx)

    # During PenGym evaluation:
    service_action_idx = policy.select_action(state)   # 0..15
    pengym_action_name = sas.to_pengym_action(service_action_idx)
"""

import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass, field


# =============================================================================
# Service Action Definitions
# =============================================================================

@dataclass
class ServiceAction:
    """One entry in the service-level action space."""
    idx: int                    # Index in service action space (0..N-1)
    name: str                   # e.g., 'exploit_ssh'
    category: str               # 'scan' | 'exploit' | 'privesc'
    pengym_name: Optional[str]  # PenGym/NASim action name (e.g., 'e_ssh')
    cve_indices: List[int] = field(default_factory=list)  # indices in Action.legal_actions


# The fixed service-level action space.
# Order: 4 scans + 7 service exploits + 2 catch-all + 3 privesc = 16
SERVICE_ACTION_DEFS = [
    # --- Scans (same for SCRIPT and PenGym) ---
    ServiceAction(0,  'port_scan',       'scan',    'subnet_scan'),
    ServiceAction(1,  'service_scan',    'scan',    'service_scan'),
    ServiceAction(2,  'os_scan',         'scan',    'os_scan'),
    ServiceAction(3,  'web_scan',        'scan',    'process_scan'),

    # --- Service-level exploits ---
    ServiceAction(4,  'exploit_ssh',     'exploit', 'e_ssh'),
    ServiceAction(5,  'exploit_ftp',     'exploit', 'e_ftp'),
    ServiceAction(6,  'exploit_http',    'exploit', 'e_http'),
    ServiceAction(7,  'exploit_smb',     'exploit', 'e_samba'),
    ServiceAction(8,  'exploit_smtp',    'exploit', 'e_smtp'),
    ServiceAction(9,  'exploit_rdp',     'exploit', None),      # No PenGym equiv
    ServiceAction(10, 'exploit_sql',     'exploit', None),      # No PenGym equiv
    ServiceAction(11, 'exploit_java_rmi','exploit', None),      # No PenGym equiv
    ServiceAction(12, 'exploit_misc',    'exploit', None),      # Catch-all

    # --- Privilege Escalation ---
    ServiceAction(13, 'privesc_tomcat',  'privesc', 'pe_tomcat'),
    ServiceAction(14, 'privesc_schtask', 'privesc', 'pe_schtask'),
    ServiceAction(15, 'privesc_daclsvc', 'privesc', 'pe_daclsvc'),
]

# CVE classification keywords (service → keywords to search in CVE metadata)
SERVICE_KEYWORDS = {
    'exploit_ssh':      ['ssh', 'openssh', 'libssh'],
    'exploit_ftp':      ['ftp', 'vsftpd', 'proftpd', 'filezilla'],
    'exploit_http':     ['http', 'apache', 'struts', 'tomcat', 'jetty', 'nginx',
                         'cgi', 'weblogic', 'jboss', 'drupal', 'wordpress',
                         'php', 'rails', 'django', 'flask', 'iis', 'servlet',
                         'asp', 'coldfusion', 'glassfish', 'websphere'],
    'exploit_smb':      ['smb', 'samba', 'cifs', 'netbios', 'ms17_010',
                         'eternalblue'],
    'exploit_smtp':     ['smtp', 'mail', 'postfix', 'sendmail', 'opensmtpd',
                         'exim', 'haraka'],
    'exploit_rdp':      ['rdp', 'remote desktop', 'bluekeep', 'ms12_020'],
    'exploit_sql':      ['sql', 'mysql', 'postgres', 'mssql', 'oracle',
                         'mariadb', 'sqlite'],
    'exploit_java_rmi': ['rmi', 'java rmi', 'jmx', 'jndi'],
}

# Port-based fallback classification
PORT_TO_SERVICE = {
    '22': 'exploit_ssh',
    '21': 'exploit_ftp',
    '80': 'exploit_http',
    '443': 'exploit_http',
    '8080': 'exploit_http',
    '8443': 'exploit_http',
    '445': 'exploit_smb',
    '139': 'exploit_smb',
    '25': 'exploit_smtp',
    '587': 'exploit_smtp',
    '3389': 'exploit_rdp',
    '3306': 'exploit_sql',
    '5432': 'exploit_sql',
    '1433': 'exploit_sql',
    '1521': 'exploit_sql',
    '1099': 'exploit_java_rmi',
}

# Privilege escalation keywords
PRIVESC_KEYWORDS = {
    'privesc_tomcat':  ['tomcat'],
    'privesc_schtask': ['cron', 'schtask', 'scheduled_task', 'at_job'],
    'privesc_daclsvc': ['daclsvc', 'proftpd', 'service_permissions'],
}


class ServiceActionSpace:
    """
    Service-level action space that abstracts CVE-specific actions into
    service categories. Provides a fixed-dimension action space independent
    of the number of CVEs in the database.

    The action space dimension is always len(SERVICE_ACTION_DEFS) = 16,
    regardless of whether there are 2000 or 20000 CVEs.
    """

    # Class-level constant for use without instantiation
    DEFAULT_ACTION_DIM = len(SERVICE_ACTION_DEFS)  # 16

    def __init__(self, action_class=None):
        """
        Args:
            action_class: The Action class from src.agent.actions.Action.
                          If None, only the action definitions are available
                          (useful for PenGym-only evaluation).
        """
        self.actions = [ServiceAction(
            idx=a.idx, name=a.name, category=a.category,
            pengym_name=a.pengym_name, cve_indices=list(a.cve_indices)
        ) for a in SERVICE_ACTION_DEFS]

        self.action_dim = len(self.actions)
        self.action_names = [a.name for a in self.actions]
        self._name_to_idx = {a.name: a.idx for a in self.actions}
        self._pengym_to_idx = {a.pengym_name: a.idx for a in self.actions
                               if a.pengym_name is not None}

        # Build CVE groupings if Action class is available
        self.action_class = action_class
        self._cve_groups: Dict[str, List[int]] = defaultdict(list)
        self._cve_metadata: Dict[int, Dict] = {}  # idx → {rank, port, ...}

        if action_class is not None:
            self._build_cve_groups()

    def _build_cve_groups(self):
        """Classify each CVE exploit into a service group."""
        action_cls = self.action_class
        scan_count = len(action_cls.Scan_actions)

        # Map scan actions (indices 0..3) to service actions (0..3)
        scan_names = ['port_scan', 'service_scan', 'os_scan', 'web_scan']
        for i in range(min(scan_count, 4)):
            self.actions[i].cve_indices = [i]

        # Classify each exploit by service
        classified = 0
        unclassified_indices = []

        for i, exp in enumerate(action_cls.All_EXP):
            original_idx = scan_count + i  # Index in legal_actions
            # Build searchable text from all available metadata
            search_text = self._get_searchable_text(exp)

            service_name = self._classify_exploit(search_text, exp)
            if service_name:
                self._cve_groups[service_name].append(original_idx)
                classified += 1
            else:
                unclassified_indices.append(original_idx)

            # Store metadata for CVE selection
            self._cve_metadata[original_idx] = self._extract_metadata(exp)

        # Unclassified → exploit_misc
        self._cve_groups['exploit_misc'].extend(unclassified_indices)

        # Assign grouped CVE indices to service actions
        for action in self.actions:
            if action.category in ('exploit', 'privesc') and action.name in self._cve_groups:
                action.cve_indices = self._cve_groups[action.name]

        total_exploits = len(action_cls.All_EXP)
        print(f"[ServiceActionSpace] CVE Classification: "
              f"{classified}/{total_exploits} classified "
              f"({classified/total_exploits*100:.1f}%), "
              f"{len(unclassified_indices)} → exploit_misc")
        for a in self.actions:
            if a.cve_indices:
                print(f"  {a.name:20s}: {len(a.cve_indices):4d} CVEs")

    def _get_searchable_text(self, exp) -> str:
        """Build a lowercase search string from exploit metadata."""
        parts = [exp.name.lower()]

        # exp_info can be dict or list of dicts
        if hasattr(exp, 'exp_info') and exp.exp_info:
            if isinstance(exp.exp_info, dict):
                parts.append(str(exp.exp_info).lower())
            elif isinstance(exp.exp_info, list):
                for item in exp.exp_info:
                    parts.append(str(item).lower())

        # setting can be dict
        if hasattr(exp, 'setting') and exp.setting:
            parts.append(str(exp.setting).lower())

        # description
        if hasattr(exp, 'description') and exp.description:
            parts.append(str(exp.description).lower())

        return ' '.join(parts)

    def _classify_exploit(self, search_text: str, exp) -> Optional[str]:
        """Classify an exploit into a service group."""
        # 1. Check privilege escalation keywords first
        for svc_name, keywords in PRIVESC_KEYWORDS.items():
            if any(kw in search_text for kw in keywords):
                return svc_name

        # 2. Check service exploit keywords
        for svc_name, keywords in SERVICE_KEYWORDS.items():
            if any(kw in search_text for kw in keywords):
                return svc_name

        # 3. Port-based fallback
        if hasattr(exp, 'setting') and isinstance(exp.setting, dict):
            rport = str(exp.setting.get('RPORT', exp.setting.get('rport', '')))
            if rport in PORT_TO_SERVICE:
                return PORT_TO_SERVICE[rport]

        return None  # → will go to exploit_misc

    def _extract_metadata(self, exp) -> Dict:
        """Extract useful metadata for CVE selection heuristic."""
        meta = {
            'name': exp.name,
            'rank': 'unknown',
            'port': None,
            'os': None,
        }

        # Try to extract rank
        if hasattr(exp, 'exp_info') and exp.exp_info:
            if isinstance(exp.exp_info, dict):
                meta['rank'] = exp.exp_info.get('Rank', 'unknown')
            elif isinstance(exp.exp_info, list) and exp.exp_info:
                meta['rank'] = exp.exp_info[0].get('rank',
                               exp.exp_info[0].get('Rank', 'unknown'))

        # Try to extract port
        if hasattr(exp, 'setting') and isinstance(exp.setting, dict):
            meta['port'] = exp.setting.get('RPORT', exp.setting.get('rport'))

        return meta

    # =========================================================================
    # CVE Selector (Tier 2): Pick best CVE within a service group
    # =========================================================================

    # Rank priority (higher = better)
    RANK_PRIORITY = {
        'excellent': 5,
        'great': 4,
        'good': 3,
        'normal': 2,
        'average': 1,
        'low': 0,
        'manual': -1,
        'unknown': 1,
    }

    def select_cve(self, service_action_idx: int,
                   host_info: Any = None,
                   strategy: str = 'rank',
                   env_data: dict = None) -> int:
        """Select the best CVE exploit index for a given service-level action.

        This is the Tier-2 selector. Given that the RL policy chose a
        service-level action (e.g., exploit_ssh), pick the specific CVE
        from the SSH group that best matches the target host.

        Args:
            service_action_idx: Index in service action space (0..15)
            host_info: Optional host info for context-aware selection.
                       Can be a Host_info object or dict with keys like
                       'port', 'services', 'os'.
            strategy: 'rank' (best rank), 'random', 'round_robin',
                      or 'match' (use env_data to find exact CVE match)
            env_data: Target host's environment data dict (from scenario JSON).
                      Used in 'match' strategy to find the CVE that matches
                      the target's actual vulnerability. This is the TRAINING
                      mode — equivalent to PenGym's service-level exploit
                      where success depends on having the service, not a
                      specific CVE.

        Returns:
            Index in Action.legal_actions for the selected CVE.
            Returns the scan action index directly for scan actions.
        """
        action = self.actions[service_action_idx]

        # Scan actions: return directly
        if action.category == 'scan':
            return action.cve_indices[0] if action.cve_indices else service_action_idx

        # No CVEs in this group → return random from misc
        if not action.cve_indices:
            misc_action = self.actions[self._name_to_idx['exploit_misc']]
            if misc_action.cve_indices:
                return random.choice(misc_action.cve_indices)
            return 0  # Fallback to scan

        # --- Match strategy: find CVE matching target vulnerability ---
        if strategy == 'match' and env_data and self.action_class:
            matched = self._select_by_match(action.cve_indices, env_data)
            if matched is not None:
                return matched
            # No match in this service group — fall through to rank

        if strategy == 'random':
            return random.choice(action.cve_indices)

        if strategy in ('rank', 'match'):
            return self._select_by_rank(action.cve_indices, host_info)

        if strategy == 'round_robin':
            # Simple round-robin using a counter
            if not hasattr(self, '_rr_counters'):
                self._rr_counters = defaultdict(int)
            idx = self._rr_counters[service_action_idx] % len(action.cve_indices)
            self._rr_counters[service_action_idx] += 1
            return action.cve_indices[idx]

        return random.choice(action.cve_indices)

    def _select_by_rank(self, cve_indices: List[int],
                        host_info: Any = None) -> int:
        """Select CVE with highest rank (and optional port/OS matching)."""
        best_idx = cve_indices[0]
        best_score = -1

        for idx in cve_indices:
            meta = self._cve_metadata.get(idx, {})
            rank = meta.get('rank', 'unknown')
            if isinstance(rank, str):
                rank = rank.lower().strip()
            score = self.RANK_PRIORITY.get(rank, 1)

            # Bonus for port match
            if host_info and meta.get('port'):
                host_ports = []
                if hasattr(host_info, 'port') and host_info.port:
                    host_ports = host_info.port
                elif isinstance(host_info, dict) and host_info.get('port'):
                    host_ports = host_info['port']
                if str(meta['port']) in [str(p) for p in host_ports]:
                    score += 2

            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx

    def _select_by_match(self, cve_indices: List[int],
                         env_data: dict) -> Optional[int]:
        """Select CVE that matches the target's actual vulnerability.

        This implements PenGym-equivalent semantics for SCRIPT's simulation:
        if the target has vulnerability X and X is in this service group,
        return X. This lets the RL policy learn at the service level
        while the CVE selector handles the tactical matching.

        Args:
            cve_indices: CVE indices in this service group
            env_data: Target's environment data with 'vulnerability' key

        Returns:
            Matching CVE index, or None if no match found
        """
        target_vulns = set(env_data.get('vulnerability', []))
        if not target_vulns:
            return None

        for idx in cve_indices:
            exp = self.action_class.legal_actions[idx]
            exp_vulns = set(getattr(exp, 'vulnerability', []))
            if target_vulns & exp_vulns:
                return idx

        return None

    # =========================================================================
    # PenGym Mapping (for Strategy C evaluation)
    # =========================================================================

    def to_pengym_action(self, service_action_idx: int) -> Optional[str]:
        """Convert service-level action index → PenGym/NASim action name.

        Returns:
            PenGym action name (e.g., 'e_ssh', 'service_scan') or None
            if the action has no PenGym equivalent.
        """
        if 0 <= service_action_idx < self.action_dim:
            return self.actions[service_action_idx].pengym_name
        return None

    def from_pengym_action(self, pengym_name: str) -> Optional[int]:
        """Convert PenGym action name → service-level action index."""
        return self._pengym_to_idx.get(pengym_name)

    def get_pengym_mappable_actions(self) -> List[Tuple[int, str]]:
        """Return list of (service_idx, pengym_name) for all mappable actions."""
        return [(a.idx, a.pengym_name) for a in self.actions
                if a.pengym_name is not None]

    def get_pengym_coverage(self, pengym_action_names: set) -> Dict:
        """Compute mapping coverage against a PenGym scenario's actions."""
        mapped = 0
        unmapped_pengym = []
        for pg_name in pengym_action_names:
            if pg_name in self._pengym_to_idx:
                mapped += 1
            else:
                unmapped_pengym.append(pg_name)

        service_mapped = sum(1 for a in self.actions if a.pengym_name in pengym_action_names)
        return {
            'pengym_actions': len(pengym_action_names),
            'mapped_to_service': mapped,
            'unmapped_pengym': unmapped_pengym,
            'service_actions_with_pengym': service_mapped,
            'coverage_pct': mapped / max(len(pengym_action_names), 1) * 100,
        }

    # =========================================================================
    # Utilities
    # =========================================================================

    def get_group_size(self, service_action_idx: int) -> int:
        """Number of CVEs in a service group."""
        if 0 <= service_action_idx < self.action_dim:
            return len(self.actions[service_action_idx].cve_indices)
        return 0

    def describe(self) -> str:
        """Human-readable summary."""
        lines = [
            "╔══════════════════════════════════════════════════════╗",
            "║     Service-Level Action Space                       ║",
            "╠══════════════════════════════════════════════════════╣",
            f"║  Action dimension: {self.action_dim:>4d}                            ║",
            "╠══════════════════════════════════════════════════════╣",
        ]
        for a in self.actions:
            pg = a.pengym_name or '-'
            cve_count = len(a.cve_indices)
            lines.append(f"║  [{a.idx:2d}] {a.name:20s} → {pg:16s} ({cve_count:4d} CVEs) ║")
        lines.append("╚══════════════════════════════════════════════════════╝")
        return '\n'.join(lines)

    def summary_dict(self) -> Dict:
        """Machine-readable summary for logging."""
        return {
            'action_dim': self.action_dim,
            'actions': [
                {
                    'idx': a.idx,
                    'name': a.name,
                    'category': a.category,
                    'pengym_name': a.pengym_name,
                    'num_cves': len(a.cve_indices),
                }
                for a in self.actions
            ],
            'total_cves_grouped': sum(len(a.cve_indices) for a in self.actions
                                      if a.category != 'scan'),
        }
