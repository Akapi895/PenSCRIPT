"""
UnifiedStateEncoder — Single encoder for BOTH simulation and PenGym domains.

Strategy C §2.2 specifies a unified 1540-dim state vector:

    [access(3) | discovery(1) | os_sbert(384) | port_sbert(384) |
     service_sbert(384) | aux_sbert(384)]

The two extra dimensions vs the original SCRIPT StateEncoder (1538):
  - access expands from 2-dim [compromised, reachable] to 3-dim
    [compromised, user_access, reachable]
  - 1-dim discovery flag (host has been discovered/scanned)

Canonicalisation maps (§2.3) ensure that semantically identical OS/service
strings from different domains produce the same SBERT embedding.

Usage::

    from src.envs.core.unified_state_encoder import UnifiedStateEncoder

    enc = UnifiedStateEncoder()

    # Simulation side (from host.py StateEncoder data)
    vec_sim = enc.encode_from_sim(
        access="compromised", os="linux", ports=["22","80"],
        services=["ssh","http"], web_fps=["Apache/2.4"],
    )

    # PenGym side (from NASim observation segment)
    vec_pg = enc.encode_from_pengym(
        compromised=True, reachable=True, discovered=True,
        access_level=2, os="linux", services=["ssh","ftp"],
        ports=["22","21"], processes=["sshd","proftpd"],
    )

    assert vec_sim.shape == vec_pg.shape == (1540,)
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Canonical maps (Strategy C §2.3)
# ---------------------------------------------------------------------------

OS_CANONICAL_MAP: Dict[str, str] = {
    # Linux variants
    "ubuntu": "linux",
    "debian": "linux",
    "centos": "linux",
    "fedora": "linux",
    "redhat": "linux",
    "rhel": "linux",
    "kali": "linux",
    "arch": "linux",
    "suse": "linux",
    "opensuse": "linux",
    "gentoo": "linux",
    "linux": "linux",
    # Windows variants
    "windows": "windows",
    "win10": "windows",
    "win11": "windows",
    "win7": "windows",
    "winxp": "windows",
    "windows server": "windows",
    "win_server": "windows",
    # BSD variants
    "freebsd": "bsd",
    "openbsd": "bsd",
    "netbsd": "bsd",
    # macOS
    "macos": "macos",
    "darwin": "macos",
    "osx": "macos",
}

SERVICE_CANONICAL_MAP: Dict[str, str] = {
    # SSH
    "openssh": "ssh",
    "ssh": "ssh",
    "sshd": "ssh",
    "dropbear": "ssh",
    # FTP
    "vsftpd": "ftp",
    "proftpd": "ftp",
    "ftp": "ftp",
    "ftpd": "ftp",
    "pure-ftpd": "ftp",
    # HTTP
    "apache": "http",
    "apache httpd": "http",
    "apache2": "http",
    "nginx": "http",
    "lighttpd": "http",
    "http": "http",
    "httpd": "http",
    "iis": "http",
    # SMTP
    "postfix": "smtp",
    "sendmail": "smtp",
    "exim": "smtp",
    "smtp": "smtp",
    # SMB / Samba
    "samba": "smb",
    "smb": "smb",
    "cifs": "smb",
    # RDP
    "rdp": "rdp",
    "xrdp": "rdp",
    "remote desktop": "rdp",
    # SQL
    "mysql": "sql",
    "postgresql": "sql",
    "mssql": "sql",
    "mariadb": "sql",
    "sql": "sql",
    # Java RMI
    "java_rmi": "java_rmi",
    "rmi": "java_rmi",
    "java rmi": "java_rmi",
    # Tomcat
    "tomcat": "tomcat",
    "apache tomcat": "tomcat",
}

# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class UnifiedStateEncoder:
    """Single state encoder for both simulation and PenGym domains.

    Produces a 1540-dim vector with the layout::

        [access(3) | discovery(1) | os_sbert(384) | port_sbert(384) |
         service_sbert(384) | aux_sbert(384)]

    Class Attributes
    ----------------
    TOTAL_DIM : int
        Total output dimension (1540).
    ACCESS_DIM : int
        3 — [compromised, user_access, reachable]
    DISCOVERY_DIM : int
        1 — binary flag for host discovered/scanned
    SBERT_DIM : int
        384 — SBERT model output dimension (all-MiniLM-L12-v2)
    """

    SBERT_DIM = 384
    ACCESS_DIM = 3     # [compromised, user_access, reachable]
    DISCOVERY_DIM = 1  # [discovered]
    NUM_SBERT_SLOTS = 4  # os, port, service, aux(web_fp/process)

    TOTAL_DIM = ACCESS_DIM + DISCOVERY_DIM + SBERT_DIM * NUM_SBERT_SLOTS  # 1540

    # Offsets
    DISCOVERY_OFFSET = ACCESS_DIM                           # 3
    OS_OFFSET = ACCESS_DIM + DISCOVERY_DIM                  # 4
    PORT_OFFSET = OS_OFFSET + SBERT_DIM                     # 388
    SERVICE_OFFSET = PORT_OFFSET + SBERT_DIM                # 772
    AUX_OFFSET = SERVICE_OFFSET + SBERT_DIM                 # 1156

    def __init__(self, encoder=None):
        """
        Args:
            encoder: SBERT ``Encoder`` instance.  If ``None``, the global
                     singleton from ``src.agent.nlp.Encoder`` is imported.
        """
        if encoder is None:
            from src.agent.nlp.Encoder import encoder as _enc
            self._encoder = _enc
        else:
            self._encoder = encoder
        # Verify SBERT dim
        assert self._encoder.SBERT_model_dim == self.SBERT_DIM, (
            f"SBERT model dim {self._encoder.SBERT_model_dim} != "
            f"expected {self.SBERT_DIM}"
        )
        # Cache for SBERT embeddings (canonical_text → 384-dim)
        self._cache: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Canonicalisation
    # ------------------------------------------------------------------

    @staticmethod
    def canonicalize_os(raw: str) -> str:
        """Map raw OS string to canonical form."""
        if not raw:
            return ""
        key = raw.strip().lower()
        return OS_CANONICAL_MAP.get(key, key)

    @staticmethod
    def canonicalize_service(raw: str) -> str:
        """Map raw service string to canonical form."""
        if not raw:
            return ""
        key = raw.strip().lower()
        return SERVICE_CANONICAL_MAP.get(key, key)

    @staticmethod
    def canonicalize_services(raw_list: List[str]) -> List[str]:
        """Canonicalize a list of service names and deduplicate."""
        seen = set()
        result = []
        for s in raw_list:
            c = UnifiedStateEncoder.canonicalize_service(s)
            if c and c not in seen:
                seen.add(c)
                result.append(c)
        return result

    # ------------------------------------------------------------------
    # SBERT helpers
    # ------------------------------------------------------------------

    def _encode_sbert(self, text: str) -> np.ndarray:
        """SBERT encode with caching and dimension safety."""
        if text not in self._cache:
            vec = self._encoder.encode_SBERT(text).flatten()
            if len(vec) > self.SBERT_DIM:
                vec = vec[:self.SBERT_DIM]
            elif len(vec) < self.SBERT_DIM:
                padded = np.zeros(self.SBERT_DIM, dtype=np.float32)
                padded[:len(vec)] = vec
                vec = padded
            self._cache[text] = vec.astype(np.float32)
        return self._cache[text]

    # ------------------------------------------------------------------
    # Encoding: Simulation side
    # ------------------------------------------------------------------

    def encode_from_sim(
        self,
        access: str = "",
        os: str = "",
        ports: Optional[List[str]] = None,
        services: Optional[List[str]] = None,
        web_fps: Optional[List[str]] = None,
        discovered: bool = False,
    ) -> np.ndarray:
        """Encode simulation HOST data into a 1540-dim unified vector.

        This replaces the old ``StateEncoder.update_vector()`` path.

        Args:
            access:     ``"compromised"`` | ``"reachable"`` | ``""``
            os:         OS string (e.g. ``"linux"``, ``"Ubuntu or Debian"``)
            ports:      List of port strings (e.g. ``["22", "80"]``)
            services:   List of service strings (e.g. ``["ssh", "http"]``)
            web_fps:    List of web-fingerprint strings
            discovered: Whether host has been discovered (scan performed)

        Returns:
            1540-dim float32 vector.
        """
        state = np.zeros(self.TOTAL_DIM, dtype=np.float32)

        # --- Access (3-dim) ---
        if access == "compromised":
            state[0] = 1.0  # [1, 0, 0] = root/compromised
        elif access == "user":
            state[1] = 1.0  # [0, 1, 0] = user access only
        elif access == "reachable":
            state[2] = 1.0  # [0, 0, 1] = reachable but not compromised

        # --- Discovery (1-dim) ---
        if discovered or ports or services or os:
            state[self.DISCOVERY_OFFSET] = 1.0

        # --- OS SBERT (384-dim) ---
        if os:
            canonical_os = self.canonicalize_os(os)
            if canonical_os:
                state[self.OS_OFFSET:self.OS_OFFSET + self.SBERT_DIM] = \
                    self._encode_sbert(canonical_os)

        # --- Port SBERT (384-dim) ---
        if ports:
            port_str = ",".join(ports)
            state[self.PORT_OFFSET:self.PORT_OFFSET + self.SBERT_DIM] = \
                self._encode_sbert(port_str)

        # --- Service SBERT (384-dim) ---
        if services:
            canonical_services = self.canonicalize_services(services)
            if canonical_services:
                svc_str = ",".join(canonical_services)
                state[self.SERVICE_OFFSET:self.SERVICE_OFFSET + self.SBERT_DIM] = \
                    self._encode_sbert(svc_str)

        # --- Aux/WebFP SBERT (384-dim) ---
        if web_fps:
            # Average SBERT embeddings of all web fingerprints
            wp_vec = np.zeros(self.SBERT_DIM, dtype=np.float32)
            for wp in web_fps:
                wp_vec += self._encode_sbert(wp)
            wp_vec /= len(web_fps)
            state[self.AUX_OFFSET:self.AUX_OFFSET + self.SBERT_DIM] = wp_vec

        return state

    # ------------------------------------------------------------------
    # Encoding: PenGym side
    # ------------------------------------------------------------------

    def encode_from_pengym(
        self,
        compromised: bool = False,
        reachable: bool = False,
        discovered: bool = False,
        access_level: float = 0.0,
        os: str = "",
        services: Optional[List[str]] = None,
        ports: Optional[List[str]] = None,
        processes: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Encode PenGym NASim host data into a 1540-dim unified vector.

        This replaces the old ``PenGymStateAdapter.convert()`` path.

        Args:
            compromised: NASim ``compromised`` flag.
            reachable:   NASim ``reachable`` flag.
            discovered:  NASim ``discovered`` flag.
            access_level: NASim access level (0=none, 1=user, 2=root).
            os:          Decoded OS name.
            services:    List of active service names.
            ports:       List of inferred port strings.
            processes:   List of active process names (used as aux slot).

        Returns:
            1540-dim float32 vector.
        """
        state = np.zeros(self.TOTAL_DIM, dtype=np.float32)

        # --- Access (3-dim) ---
        if compromised or access_level >= 2:
            state[0] = 1.0  # root / compromised
        elif access_level >= 1:
            state[1] = 1.0  # user access
        elif reachable:
            state[2] = 1.0  # reachable only

        # --- Discovery (1-dim) ---
        if discovered:
            state[self.DISCOVERY_OFFSET] = 1.0

        # --- OS SBERT (384-dim) ---
        if os:
            canonical_os = self.canonicalize_os(os)
            if canonical_os:
                state[self.OS_OFFSET:self.OS_OFFSET + self.SBERT_DIM] = \
                    self._encode_sbert(canonical_os)

        # --- Port SBERT (384-dim) ---
        if ports:
            port_str = ",".join(ports)
            state[self.PORT_OFFSET:self.PORT_OFFSET + self.SBERT_DIM] = \
                self._encode_sbert(port_str)

        # --- Service SBERT (384-dim) ---
        if services:
            canonical_services = self.canonicalize_services(services)
            if canonical_services:
                svc_str = ",".join(canonical_services)
                state[self.SERVICE_OFFSET:self.SERVICE_OFFSET + self.SBERT_DIM] = \
                    self._encode_sbert(svc_str)

        # --- Aux/Process SBERT (384-dim) ---
        if processes:
            proc_str = ",".join(processes)
            state[self.AUX_OFFSET:self.AUX_OFFSET + self.SBERT_DIM] = \
                self._encode_sbert(proc_str)

        return state

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------

    def pad_legacy_state(self, legacy_vec: np.ndarray) -> np.ndarray:
        """Pad a legacy 1538-dim SCRIPT vector to 1540-dim unified format.

        The legacy layout is ``[access(2) | 4×384 SBERT]``.
        This inserts a third access dim (user_access=0) and a discovery
        dim (=1 if any SBERT slot is non-zero, else 0).

        Useful for loading old checkpoints trained with 1538-dim.

        Args:
            legacy_vec: 1538-dim float32 array.

        Returns:
            1540-dim float32 array.
        """
        assert legacy_vec.shape == (1538,), (
            f"Expected 1538-dim, got {legacy_vec.shape}"
        )
        unified = np.zeros(self.TOTAL_DIM, dtype=np.float32)

        # Access: old [compromised, reachable] → new [compromised, user=0, reachable]
        unified[0] = legacy_vec[0]   # compromised
        # unified[1] = 0             # user_access (not in legacy)
        unified[2] = legacy_vec[1]   # reachable

        # Discovery: set to 1 if any SBERT slot is non-zero
        sbert_block = legacy_vec[2:]  # 4×384 = 1536
        if np.any(sbert_block != 0):
            unified[self.DISCOVERY_OFFSET] = 1.0

        # Copy 4×384 SBERT blocks
        unified[self.OS_OFFSET:] = sbert_block

        return unified

    def describe(self) -> str:
        """Human-readable description of the encoder layout."""
        return (
            f"UnifiedStateEncoder(dim={self.TOTAL_DIM}, "
            f"access={self.ACCESS_DIM}, discovery={self.DISCOVERY_DIM}, "
            f"sbert={self.SBERT_DIM}×{self.NUM_SBERT_SLOTS}, "
            f"cache_size={len(self._cache)})"
        )
