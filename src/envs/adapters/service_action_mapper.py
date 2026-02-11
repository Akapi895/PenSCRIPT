"""
Service-Level Action Mapper for PenGym evaluation.

Replaces the old CVE-level ActionMapper with a direct service→PenGym mapping.
Because both the service action space and PenGym use service-level actions,
the mapping is nearly 1:1 and achieves ~100% coverage.

Old (CVE-level):   2064 actions → 3.4% coverage → 0% success
New (service-level): 16 actions → ~100% coverage → meaningful transfer
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
                   target_host: Tuple[int, int]) -> int:
        """Convert (service_action_idx, target_host) → PenGym flat action idx.

        Args:
            service_action_idx: Index in service action space (0..15)
            target_host: (subnet_id, host_id) tuple

        Returns:
            PenGym flat action index, or -1 if not mappable
        """
        pengym_name = self.sas.to_pengym_action(service_action_idx)

        if pengym_name is None:
            self.total_unmapped += 1
            return -1

        key = (pengym_name, target_host)
        if key in self.pengym_actions:
            self.total_mapped += 1
            self.action_usage[pengym_name] = self.action_usage.get(pengym_name, 0) + 1
            return self.pengym_actions[key]

        # For subnet_scan, the target is a subnet, not a host
        if pengym_name == 'subnet_scan':
            for (name, target), idx in self.pengym_actions.items():
                if name == 'subnet_scan' and target[0] == target_host[0]:
                    self.total_mapped += 1
                    return idx

        self.total_unmapped += 1
        return -1

    def get_random_valid_action(self, target_host: Tuple[int, int]) -> int:
        """Fallback: random valid PenGym action for the target host."""
        valid = [idx for (name, target), idx in self.pengym_actions.items()
                 if target == target_host]
        if valid:
            return np.random.choice(valid)
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
