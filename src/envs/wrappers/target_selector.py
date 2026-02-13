"""
Target Selector — Pluggable strategies for choosing the next target host.

In PenGym's multi-host network environment, the SCRIPT agent operates on
one host at a time.  A ``TargetSelector`` decides *which* host to focus on
next, given the current network state.

Usage::

    from src.envs.wrappers.target_selector import PrioritySensitiveSelector

    selector = PrioritySensitiveSelector()
    target = selector.select(
        available=wrapper.get_available_targets(),
        sensitive=wrapper.get_sensitive_hosts(),
        host_info_fn=wrapper.get_host_info,
    )
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np


class TargetSelector(ABC):
    """Abstract base for target-host selection strategies."""

    @abstractmethod
    def select(
        self,
        available: List[Tuple[int, int]],
        sensitive: List[Tuple[int, int]],
        host_info_fn: Callable[[Tuple[int, int]], Optional[Dict]],
        blocked: Optional[Set[Tuple[int, int]]] = None,
    ) -> Optional[Tuple[int, int]]:
        """Choose the next target host.

        Args:
            available: Currently reachable, uncompromised host addresses.
            sensitive: Goal (high-value) host addresses defined by the scenario.
            host_info_fn: ``Callable(host_addr) → dict`` returning structured
                host information (or ``None``).  The dict must contain at
                least ``'reachable'``, ``'compromised'``, and ``'value'`` keys.
            blocked: Optional set of host addresses that should be skipped
                (e.g., hosts where repeated actions always fail due to
                firewall rules).

        Returns:
            The selected ``(subnet_id, host_id)`` tuple, or ``None`` if no
            valid target exists.
        """
        ...

    def reset(self) -> None:
        """Reset any internal state (called on episode reset)."""
        pass

    def __repr__(self) -> str:
        return self.__class__.__name__


class PrioritySensitiveSelector(TargetSelector):
    """Prioritise sensitive hosts, then fall back to any reachable host.

    Selection order:

    1. Reachable sensitive host that is **not yet compromised** and not
       blocked.
    2. Any reachable uncompromised host that is not blocked
       (for pivoting).
    3. Any discovered (but not yet reachable) sensitive host that is
       not blocked — the agent may need to scan/pivot to reach it.
    4. First available host (fallback).
    5. ``None`` — no valid target exists.  The wrapper should stay on
       the current (compromised) host for exploration (subnet_scan).

    **v2 change:** Removed the old priority-3 blind fallback that returned
    ``sensitive[0]`` regardless of reachability.  Added ``blocked``
    parameter so the wrapper can exclude hosts where all actions fail
    (e.g., due to firewall rules).
    """

    def select(
        self,
        available: List[Tuple[int, int]],
        sensitive: List[Tuple[int, int]],
        host_info_fn: Callable[[Tuple[int, int]], Optional[Dict]],
        blocked: Optional[Set[Tuple[int, int]]] = None,
    ) -> Optional[Tuple[int, int]]:
        _blocked = blocked or set()

        # 1. Uncompromised, reachable, sensitive hosts
        for host in sensitive:
            if host in _blocked:
                continue
            info = host_info_fn(host)
            if info and info.get("reachable") and not info.get("compromised"):
                return host

        # 2. Any reachable uncompromised host (for pivoting)
        for host in available:
            if host in _blocked:
                continue
            info = host_info_fn(host)
            if info and not info.get("compromised"):
                return host

        # 3. Discovered (but maybe not yet fully reachable) sensitive hosts
        for host in sensitive:
            if host in _blocked:
                continue
            info = host_info_fn(host)
            if info and info.get("discovered") and not info.get("compromised"):
                return host

        # 4. Any available host at all
        for host in available:
            if host in _blocked:
                continue
            if available:
                return host

        # 5. Nothing available — caller should handle exploration
        return None


class ReachabilityAwareSelector(TargetSelector):
    """Proximity-aware target selection for multi-hop networks.

    Unlike ``PrioritySensitiveSelector`` which jumps directly to distant
    sensitive hosts (often blocked by inter-subnet firewalls), this
    selector considers network topology to guide breadth-first expansion:

    1. **Sensitive hosts in compromised subnets** — intra-subnet access
       bypasses firewalls, so a sensitive host in a subnet where we
       already have a foothold is the highest-value immediate target.
    2. **Pivot hosts near blocked targets** — if a sensitive host was
       blocked (firewall prevented our exploits), try other hosts in
       the *same subnet* as the blocked one; they may provide the
       intra-subnet path we need.
    3. **Breadth-first expansion** — prefer hosts that are in the same
       or adjacent subnet to a compromised host, enabling progressive
       network traversal.
    4. **Any reachable uncompromised host** — general fallback.
    5. **Discovered sensitive hosts** — long-range targets (may need
       more pivoting first).

    This selector naturally handles:

    * **Linear topologies** requiring step-by-step subnet traversal.
    * **Multi-entry-point** networks (two internet connections) by
      exploring from whichever side has available targets.
    * **Firewall-restricted paths** by prioritising same-subnet pivots
      over cross-subnet sensitive hosts.
    """

    def select(
        self,
        available: List[Tuple[int, int]],
        sensitive: List[Tuple[int, int]],
        host_info_fn: Callable[[Tuple[int, int]], Optional[Dict]],
        blocked: Optional[Set[Tuple[int, int]]] = None,
    ) -> Optional[Tuple[int, int]]:
        _blocked = blocked or set()

        # Gather intelligence: which subnets do we control?
        all_hosts = set(available) | set(sensitive)
        compromised_subnets: Set[int] = set()
        for host in all_hosts:
            info = host_info_fn(host)
            if info and info.get("compromised"):
                compromised_subnets.add(host[0])

        # Subnets containing blocked targets (need nearby pivots)
        blocked_subnets: Set[int] = {h[0] for h in _blocked}

        # Helper: is h a valid (uncompromised, not blocked) candidate?
        def _valid(h: Tuple[int, int]) -> bool:
            if h in _blocked:
                return False
            info = host_info_fn(h)
            return bool(info and not info.get("compromised"))

        # 1. Sensitive hosts in a subnet we already control
        #    (intra-subnet = no firewall restrictions)
        for host in sensitive:
            if host[0] in compromised_subnets and _valid(host):
                return host

        # 2. Pivot hosts in same subnet as a blocked target
        #    (unlock the blocked-target's subnet from inside)
        for host in available:
            if host[0] in blocked_subnets and _valid(host):
                return host

        # 3. Hosts in compromised subnets (same-subnet expansion)
        for host in available:
            if host[0] in compromised_subnets and _valid(host):
                return host

        # 4. Any reachable uncompromised host not blocked
        #    (breadth-first: try the next accessible host)
        for host in available:
            if _valid(host):
                return host

        # 5. Discovered-but-not-yet-reachable sensitive hosts
        for host in sensitive:
            if host in _blocked:
                continue
            info = host_info_fn(host)
            if info and info.get("discovered") and not info.get("compromised"):
                return host

        # 6. Nothing available
        return None


class RoundRobinSelector(TargetSelector):
    """Cycle through hosts in order.

    Useful for ensuring every reachable host is visited at least once.
    """

    def __init__(self) -> None:
        self._idx: int = 0

    def select(
        self,
        available: List[Tuple[int, int]],
        sensitive: List[Tuple[int, int]],
        host_info_fn: Callable[[Tuple[int, int]], Optional[Dict]],
        blocked: Optional[Set[Tuple[int, int]]] = None,
    ) -> Optional[Tuple[int, int]]:
        _blocked = blocked or set()
        candidates = [h for h in available if h not in _blocked]
        if not candidates:
            return None
        target = candidates[self._idx % len(candidates)]
        self._idx += 1
        return target

    def reset(self) -> None:
        self._idx = 0


class ValuePrioritySelector(TargetSelector):
    """Select the host with the highest scenario-defined value.

    Ties are broken by address order.
    """

    def select(
        self,
        available: List[Tuple[int, int]],
        sensitive: List[Tuple[int, int]],
        host_info_fn: Callable[[Tuple[int, int]], Optional[Dict]],
        blocked: Optional[Set[Tuple[int, int]]] = None,
    ) -> Optional[Tuple[int, int]]:
        _blocked = blocked or set()
        if not available:
            return None

        best: Optional[Tuple[int, int]] = None
        best_val: float = -float("inf")

        for host in available:
            if host in _blocked:
                continue
            info = host_info_fn(host)
            if info and not info.get("compromised"):
                val = info.get("value", 0.0)
                if val > best_val:
                    best = host
                    best_val = val

        return best if best is not None else (available[0] if available else None)
