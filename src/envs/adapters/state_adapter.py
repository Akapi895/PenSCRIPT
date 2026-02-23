"""
State Adapter — Convert PenGym NASim observations → SCRIPT SBERT state vector.

PenGym NASim obs (flat mode, per-host segment):
  [subnet_onehot | host_onehot | compromised | reachable | discovered |
   value | discovery_value | access | OS_flags | service_flags | process_flags]

SCRIPT StateEncoder (state_space-dim, currently 1538):
  [access(2) | os_sbert(384) | port_sbert(384) | service_sbert(384) | web_fp_sbert(384)]

Unified format (Strategy C, 1540-dim):
  [access(3) | discovery(1) | os_sbert(384) | port_sbert(384) |
   service_sbert(384) | aux_sbert(384)]

This adapter extracts per-host info from the NASim flat observation,
reconstructs text strings, then SBERT-encodes them to match SCRIPT's format.

See also: ``src.envs.core.unified_state_encoder.UnifiedStateEncoder`` for
the Strategy C unified encoder (1540-dim).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from src.agent.host import StateEncoder


class PenGymStateAdapter:
    """Convert PenGym NASim observations to SCRIPT-compatible state vectors."""

    STATE_DIM = StateEncoder.state_space  # Canonical: 1538 (2+384+384+384+384)
    ACCESS_DIM = 2
    SBERT_DIM = 384  # all-MiniLM-L12-v2 output dimension

    # Offsets in the output vector (same as StateEncoder)
    OS_OFFSET = 2                    # access_dim
    PORT_OFFSET = 2 + 384            # 386
    SERVICE_OFFSET = 2 + 384 * 2     # 770
    WEB_FP_OFFSET = 2 + 384 * 3     # 1154

    # Default service → port mapping (from PenGym CONFIG.yml)
    DEFAULT_SERVICE_PORT_MAP = {
        'ssh': '22',
        'ftp': '21',
        'http': '80',
        'smtp': '25',
        'samba': '445',
        'proftpd': '2121',
    }

    def __init__(self, scenario, encoder=None, service_port_map: Dict = None):
        """
        Args:
            scenario: NASim scenario object (has .subnets, .os, .services,
                      .processes, .hosts, .address_space_bounds, etc.)
            encoder:  SBERT encoder instance (from src.agent.nlp.Encoder).
                      If None, imports the global singleton.
            service_port_map: Dict mapping service name → port string.
        """
        self.scenario = scenario
        self.service_port_map = service_port_map or self.DEFAULT_SERVICE_PORT_MAP

        # Lazy-load encoder to avoid import overhead at module level
        if encoder is None:
            from src.agent.nlp.Encoder import encoder as _enc
            self.encoder = _enc
        else:
            self.encoder = encoder

        # Scenario metadata
        self.os_names: List[str] = list(scenario.os)
        self.service_names: List[str] = list(scenario.services)
        self.process_names: List[str] = list(scenario.processes)

        # Host layout in NASim observation
        # address_space_bounds = (num_subnets, max_hosts_per_subnet)
        self.num_subnets = scenario.address_space_bounds[0]
        self.max_hosts = scenario.address_space_bounds[1]

        # Per-host vector offsets within NASim observation
        # Layout: [subnet_oh | host_oh | compromised | reachable | discovered |
        #          value | disc_value | access | os_oh | services_bin | processes_bin]
        self._compromised_offset = self.num_subnets + self.max_hosts
        self._reachable_offset = self._compromised_offset + 1
        self._discovered_offset = self._reachable_offset + 1
        self._value_offset = self._discovered_offset + 1
        self._disc_value_offset = self._value_offset + 1
        self._access_offset = self._disc_value_offset + 1
        self._os_offset = self._access_offset + 1
        self._service_offset = self._os_offset + len(self.os_names)
        self._process_offset = self._service_offset + len(self.service_names)
        self._host_vec_size = self._process_offset + len(self.process_names)

        # Cache for SBERT embeddings (text → 384-dim vector)
        self._sbert_cache: Dict[str, np.ndarray] = {}

        # Build host_num_map: (subnet, host) → row index in obs tensor
        # NOTE: scenario.subnets includes internet at index 0.
        #       Address space bounds[0] = total subnets (including internet).
        #       We skip internet (subnet 0) and iterate user subnets 1..N-1.
        #       Use subnets[subnet_id] (NOT subnet_id - 1) to get correct size.
        self.host_num_map: Dict[Tuple[int, int], int] = {}
        host_idx = 0
        for subnet_id in range(1, self.num_subnets):
            for host_id in range(self.scenario.subnets[subnet_id]):
                self.host_num_map[(subnet_id, host_id)] = host_idx
                host_idx += 1
        self.num_hosts = host_idx

        # Total flat obs size = (num_hosts + 1) * host_vec_size
        # The +1 is the auxiliary row for action results
        self.flat_obs_size = (self.num_hosts + 1) * self._host_vec_size

        print(f"[StateAdapter] Initialized: {self.num_hosts} hosts, "
              f"host_vec_size={self._host_vec_size}, "
              f"os={self.os_names}, services={self.service_names}, "
              f"processes={self.process_names}")

    def _encode_sbert(self, text: str) -> np.ndarray:
        """SBERT encode with caching."""
        if text not in self._sbert_cache:
            vec = self.encoder.encode_SBERT(text).flatten()
            # Ensure exactly SBERT_DIM dimensions
            if len(vec) > self.SBERT_DIM:
                vec = vec[:self.SBERT_DIM]
            elif len(vec) < self.SBERT_DIM:
                padded = np.zeros(self.SBERT_DIM, dtype=np.float32)
                padded[:len(vec)] = vec
                vec = padded
            self._sbert_cache[text] = vec.astype(np.float32)
        return self._sbert_cache[text]

    def _get_host_segment(self, flat_obs: np.ndarray,
                          host_addr: Tuple[int, int]) -> Optional[np.ndarray]:
        """Extract per-host segment from flat NASim observation.

        Args:
            flat_obs: Full flat observation vector
            host_addr: (subnet_id, host_id) tuple

        Returns:
            1D array of the host's features, or None if host not in map
        """
        if host_addr not in self.host_num_map:
            return None
        row = self.host_num_map[host_addr]
        start = row * self._host_vec_size
        end = start + self._host_vec_size
        if end > len(flat_obs):
            return None
        return flat_obs[start:end]

    def convert(self, flat_obs: np.ndarray,
                host_addr: Tuple[int, int]) -> np.ndarray:
        """Convert PenGym observation to SCRIPT 1538-dim state vector for a single host.

        Args:
            flat_obs: Flat NASim observation (1D array)
            host_addr: (subnet_id, host_id) of the target host

        Returns:
            1538-dim float32 vector compatible with SCRIPT policy
        """
        state = np.zeros(self.STATE_DIM, dtype=np.float32)
        host_seg = self._get_host_segment(flat_obs, host_addr)

        if host_seg is None:
            return state  # All zeros if host not found

        # --- 1. Access vector (2-dim) ---
        compromised = host_seg[self._compromised_offset]
        reachable = host_seg[self._reachable_offset]
        access_level = host_seg[self._access_offset]  # 0=none, 1=user, 2=root

        if compromised > 0.5 or access_level >= 1:
            state[0] = 1.0  # [1, 0] = compromised in SCRIPT
        elif reachable > 0.5:
            state[1] = 1.0  # [0, 1] = reachable in SCRIPT

        # --- 2. OS embedding (384-dim at offset 2) ---
        os_flags = host_seg[self._os_offset:self._os_offset + len(self.os_names)]
        detected_os = self._decode_os(os_flags)
        if detected_os:
            state[self.OS_OFFSET:self.OS_OFFSET + self.SBERT_DIM] = \
                self._encode_sbert(detected_os)

        # --- 3. Port embedding (384-dim at offset 386) ---
        # Infer ports from detected services using service→port mapping
        service_flags = host_seg[self._service_offset:
                                 self._service_offset + len(self.service_names)]
        port_string = self._decode_ports(service_flags)
        if port_string:
            state[self.PORT_OFFSET:self.PORT_OFFSET + self.SBERT_DIM] = \
                self._encode_sbert(port_string)

        # --- 4. Service embedding (384-dim at offset 770) ---
        service_string = self._decode_services(service_flags)
        if service_string:
            state[self.SERVICE_OFFSET:self.SERVICE_OFFSET + self.SBERT_DIM] = \
                self._encode_sbert(service_string)

        # --- 5. Web fingerprint embedding (384-dim at offset 1154) ---
        # PenGym does not have web fingerprint → leave as zeros (accepted gap)
        # Optionally, encode process info here as a proxy
        process_flags = host_seg[self._process_offset:
                                 self._process_offset + len(self.process_names)]
        process_string = self._decode_processes(process_flags)
        if process_string:
            state[self.WEB_FP_OFFSET:self.WEB_FP_OFFSET + self.SBERT_DIM] = \
                self._encode_sbert(process_string)

        return state

    def convert_all_hosts(self, flat_obs: np.ndarray) -> Dict[Tuple[int, int], np.ndarray]:
        """Convert observation for ALL hosts in the network.

        Returns:
            Dict mapping (subnet, host) → 1538-dim state vector
        """
        result = {}
        for addr in self.host_num_map:
            result[addr] = self.convert(flat_obs, addr)
        return result

    # ---- Strategy C unified encoding (1540-dim) ----

    def convert_unified(self, flat_obs: np.ndarray,
                        host_addr: Tuple[int, int]) -> np.ndarray:
        """Convert PenGym observation to a 1540-dim **unified** state vector.

        Delegates to :class:`~src.envs.core.unified_state_encoder.UnifiedStateEncoder`
        so that the same canonicalisation and layout are used across both
        simulation and PenGym domains (Strategy C §2.2).

        Args:
            flat_obs: Flat NASim observation (1D array).
            host_addr: ``(subnet_id, host_id)`` of the target host.

        Returns:
            1540-dim float32 vector.
        """
        from src.envs.core.unified_state_encoder import UnifiedStateEncoder

        if not hasattr(self, '_unified_encoder'):
            self._unified_encoder = UnifiedStateEncoder(encoder=self.encoder)

        host_data = self.get_host_data(flat_obs, host_addr)
        if host_data is None:
            return np.zeros(UnifiedStateEncoder.TOTAL_DIM, dtype=np.float32)

        return self._unified_encoder.encode_from_pengym(
            compromised=host_data['compromised'],
            reachable=host_data['reachable'],
            discovered=host_data['discovered'],
            access_level=host_data['access_level'],
            os=host_data['os'],
            services=host_data['services'],
            ports=host_data['ports'],
            processes=host_data['processes'],
        )

    def convert_all_hosts_unified(
        self, flat_obs: np.ndarray,
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """Convert observation for ALL hosts using the unified 1540-dim encoding.

        Returns:
            Dict mapping ``(subnet, host)`` → 1540-dim state vector.
        """
        result = {}
        for addr in self.host_num_map:
            result[addr] = self.convert_unified(flat_obs, addr)
        return result

    def get_sensitive_hosts(self) -> List[Tuple[int, int]]:
        """Get list of sensitive (goal) host addresses from scenario."""
        sensitive = []
        if hasattr(self.scenario, 'sensitive_hosts'):
            for addr in self.scenario.sensitive_hosts:
                sensitive.append(tuple(addr))
        return sensitive

    def get_reachable_hosts(self, flat_obs: np.ndarray) -> List[Tuple[int, int]]:
        """Get list of currently reachable host addresses."""
        reachable = []
        for addr in self.host_num_map:
            seg = self._get_host_segment(flat_obs, addr)
            if seg is not None:
                if seg[self._reachable_offset] > 0.5 or \
                   seg[self._discovered_offset] > 0.5:
                    reachable.append(addr)
        return reachable

    def get_host_data(self, flat_obs: np.ndarray,
                      host_addr: Tuple[int, int]) -> Optional[Dict]:
        """Extract structured host information from PenGym observation.

        Built on top of ``_get_host_segment()``.

        Args:
            flat_obs: Flat NASim observation (1D array).
            host_addr: ``(subnet_id, host_id)`` tuple.

        Returns:
            A dict with the following keys, or ``None`` if *host_addr* is not
            present in the scenario's host map:

            - ``address``      – ``Tuple[int, int]``
            - ``reachable``    – ``bool``
            - ``compromised``  – ``bool``
            - ``discovered``   – ``bool``
            - ``access_level`` – ``float`` (0=none, 1=user, 2=root)
            - ``value``        – ``float`` (scenario-defined host value)
            - ``os``           – ``str`` (decoded OS name or ``''``)
            - ``services``     – ``List[str]`` (active service names)
            - ``ports``        – ``List[str]`` (inferred port numbers)
            - ``processes``    – ``List[str]`` (active process names)
        """
        seg = self._get_host_segment(flat_obs, host_addr)
        if seg is None:
            return None

        # Decode binary flags
        os_flags = seg[self._os_offset:self._os_offset + len(self.os_names)]
        service_flags = seg[self._service_offset:
                            self._service_offset + len(self.service_names)]
        process_flags = seg[self._process_offset:
                            self._process_offset + len(self.process_names)]

        active_services: List[str] = [
            self.service_names[i]
            for i in np.where(service_flags > 0.5)[0]
            if i < len(self.service_names)
        ]
        active_processes: List[str] = [
            self.process_names[i]
            for i in np.where(process_flags > 0.5)[0]
            if i < len(self.process_names)
        ]

        # Infer ports from services
        ports: List[str] = [
            self.service_port_map[svc]
            for svc in active_services
            if svc in self.service_port_map
        ]

        return {
            'address': host_addr,
            'reachable': bool(seg[self._reachable_offset] > 0.5),
            'compromised': bool(seg[self._compromised_offset] > 0.5),
            'discovered': bool(seg[self._discovered_offset] > 0.5),
            'access_level': float(seg[self._access_offset]),
            'value': float(seg[self._value_offset]),
            'os': self._decode_os(os_flags),
            'services': active_services,
            'ports': ports,
            'processes': active_processes,
        }

    # ---- Decode helpers ----

    def _decode_os(self, os_flags: np.ndarray) -> str:
        """Decode one-hot OS vector → OS name string."""
        active = np.where(os_flags > 0.5)[0]
        if len(active) == 0:
            return ""
        os_names = [self.os_names[i] for i in active if i < len(self.os_names)]
        if not os_names:
            return ""
        return " or ".join(os_names)  # Matches SCRIPT's StateEncoder behavior

    def _decode_services(self, service_flags: np.ndarray) -> str:
        """Decode binary service vector → comma-separated service string."""
        active = np.where(service_flags > 0.5)[0]
        if len(active) == 0:
            return ""
        names = [self.service_names[i] for i in active
                 if i < len(self.service_names)]
        return ",".join(names)

    def _decode_ports(self, service_flags: np.ndarray) -> str:
        """Infer port string from active services using service→port mapping."""
        active = np.where(service_flags > 0.5)[0]
        if len(active) == 0:
            return ""
        ports = []
        for i in active:
            if i < len(self.service_names):
                svc = self.service_names[i]
                port = self.service_port_map.get(svc, str(10000 + i))
                ports.append(port)
        return ",".join(ports)

    def _decode_processes(self, process_flags: np.ndarray) -> str:
        """Decode binary process vector → comma-separated process string."""
        active = np.where(process_flags > 0.5)[0]
        if len(active) == 0:
            return ""
        names = [self.process_names[i] for i in active
                 if i < len(self.process_names)]
        return ",".join(names)

    def describe(self) -> str:
        """Return human-readable description of the adapter configuration."""
        lines = [
            "=== PenGymStateAdapter Configuration ===",
            f"Output dimension: {self.STATE_DIM}",
            f"SBERT dimension: {self.SBERT_DIM}",
            f"Number of hosts: {self.num_hosts}",
            f"Host vector size (NASim): {self._host_vec_size}",
            f"Flat obs size: {self.flat_obs_size}",
            f"OS names: {self.os_names}",
            f"Service names: {self.service_names}",
            f"Process names: {self.process_names}",
            f"Service→Port map: {self.service_port_map}",
            f"Host map: {self.host_num_map}",
        ]
        return "\n".join(lines)
