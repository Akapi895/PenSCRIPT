"""
Service-Level Action Mapper for PenGym evaluation.

Replaces the old CVE-level ActionMapper with a direct service→PenGym mapping.
Because both the service action space and PenGym use service-level actions,
the mapping is nearly 1:1 and achieves ~100% coverage.

Old (CVE-level):   2064 actions → 3.4% coverage → 0% success
New (service-level): 16 actions → ~100% coverage → meaningful transfer

v2 fix: subnet_scan now routes FROM compromised hosts (not the target),
matching NASim semantics where subnet_scan discovers *adjacent* subnets
from the scanner's position.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from src.agent.actions.service_action_space import ServiceActionSpace


class ServiceActionMapper:
    """Map service-level actions to PenGym NASim environment actions.

    This is a thin wrapper that converts:
        service_action_idx (0..15) + target_host → PenGym flat action idx

    The mapping is straightforward because both spaces use service-level
    abstractions (e.g., exploit_ssh → e_ssh).

    **subnet_scan special handling (v2):**
    NASim's ``subnet_scan target=(X, Y)`` scans FROM subnet X to discover
    hosts in adjacent subnets.  The mapper routes port_scan to scan FROM
    a compromised host rather than the target host, ensuring correct
    discovery of new subnets.
    """

    def __init__(self, service_action_space: ServiceActionSpace, pengym_env):
        """
        Args:
            service_action_space: ServiceActionSpace instance
            pengym_env: PenGym NASim environment with flat action space
        """
        self.sas = service_action_space
        self.env = pengym_env
        self.action_space = pengym_env.action_space

        # Build PenGym action index: (action_name, target_host) → flat_idx
        self.pengym_actions: Dict[Tuple[str, Tuple], int] = {}
        self.pengym_action_names: set = set()

        for i in range(self.action_space.n):
            action = self.action_space.get_action(i)
            name = action.name
            target = tuple(action.target)
            self.pengym_actions[(name, target)] = i
            self.pengym_action_names.add(name)

        # Compute coverage
        coverage = self.sas.get_pengym_coverage(self.pengym_action_names)

        print(f"[ServiceActionMapper] PenGym: {self.action_space.n} flat actions, "
              f"{len(self.pengym_action_names)} unique action types")
        print(f"  Coverage: {coverage['mapped_to_service']}/{coverage['pengym_actions']} "
              f"PenGym action types mapped ({coverage['coverage_pct']:.1f}%)")
        if coverage['unmapped_pengym']:
            print(f"  Unmapped PenGym actions: {coverage['unmapped_pengym']}")

        # Stats
        self.total_mapped = 0
        self.total_unmapped = 0
        self.action_usage: Dict[str, int] = {}

    def map_action(self, service_action_idx: int,
                   target_host: Tuple[int, int],
                   compromised_hosts: Optional[List[Tuple[int, int]]] = None,
                   ) -> int:
        """Convert (service_action_idx, target_host) → PenGym flat action idx.

        Args:
            service_action_idx: Index in service action space (0..15)
            target_host: (subnet_id, host_id) tuple
            compromised_hosts: List of already-compromised host addresses.
                Used by ``subnet_scan`` to scan FROM a compromised position
                instead of the (possibly unreachable) target host.

        Returns:
            PenGym flat action index, or -1 if not mappable
        """
        pengym_name = self.sas.to_pengym_action(service_action_idx)

        if pengym_name is None:
            self.total_unmapped += 1
            return -1

        # subnet_scan: scan FROM compromised hosts, not the target
        if pengym_name == 'subnet_scan':
            return self._map_subnet_scan(target_host, compromised_hosts)

        key = (pengym_name, target_host)
        if key in self.pengym_actions:
            self.total_mapped += 1
            self.action_usage[pengym_name] = self.action_usage.get(pengym_name, 0) + 1
            return self.pengym_actions[key]

        self.total_unmapped += 1
        return -1

    def _map_subnet_scan(
        self,
        target_host: Tuple[int, int],
        compromised_hosts: Optional[List[Tuple[int, int]]],
    ) -> int:
        """Route subnet_scan to scan FROM a compromised host.

        NASim semantics: ``subnet_scan target=(X, Y)`` scans from subnet X
        and discovers hosts in adjacent subnets.  We prioritise scanning
        from compromised hosts because the agent can only scan from
        positions it controls.

        Priority order:
            1. Scan from a compromised host (most recent first — likely
               closest to undiscovered subnets).
            2. Scan from the target host itself (fallback if target IS
               the compromised host or in the same subnet).
        """
        # Priority 1: scan FROM compromised hosts (most recent first)
        if compromised_hosts:
            for comp_host in reversed(compromised_hosts):
                key = ('subnet_scan', comp_host)
                if key in self.pengym_actions:
                    self.total_mapped += 1
                    self.action_usage['subnet_scan'] = (
                        self.action_usage.get('subnet_scan', 0) + 1
                    )
                    return self.pengym_actions[key]

        # Priority 2: scan from target host's subnet (old behaviour)
        for (name, target), idx in self.pengym_actions.items():
            if name == 'subnet_scan' and target[0] == target_host[0]:
                self.total_mapped += 1
                self.action_usage['subnet_scan'] = (
                    self.action_usage.get('subnet_scan', 0) + 1
                )
                return idx

        self.total_unmapped += 1
        return -1

    def get_random_valid_action(
        self,
        target_host: Tuple[int, int],
        compromised_hosts: Optional[List[Tuple[int, int]]] = None,
    ) -> int:
        """Fallback: random valid PenGym action for the target host.

        If no valid actions exist for the target host (unreachable),
        try actions from compromised hosts (useful for subnet_scan).
        """
        valid = [idx for (name, target), idx in self.pengym_actions.items()
                 if target == target_host]
        if valid:
            return int(np.random.choice(valid))

        # Fallback: try actions from compromised hosts
        if compromised_hosts:
            for comp_host in reversed(compromised_hosts):
                comp_valid = [
                    idx for (name, target), idx in self.pengym_actions.items()
                    if target == comp_host
                ]
                if comp_valid:
                    return int(np.random.choice(comp_valid))

        return self.action_space.sample()

    def get_all_actions_for_host(self, target_host: Tuple[int, int]) -> Dict[str, int]:
        """Get all available PenGym actions for a specific host."""
        return {name: idx for (name, target), idx in self.pengym_actions.items()
                if target == target_host}

    def get_mapping_stats(self) -> Dict:
        """Stats about mapping quality."""
        coverage = self.sas.get_pengym_coverage(self.pengym_action_names)
        return {
            'service_action_dim': self.sas.action_dim,
            'total_pengym_actions': self.action_space.n,
            'pengym_action_types': len(self.pengym_action_names),
            'coverage_pct': coverage['coverage_pct'],
            'unmapped_pengym': coverage['unmapped_pengym'],
            'total_mapped_calls': self.total_mapped,
            'total_unmapped_calls': self.total_unmapped,
            'valid_call_rate': (
                self.total_mapped / max(self.total_mapped + self.total_unmapped, 1) * 100
            ),
            'action_usage': dict(self.action_usage),
        }
