"""
SingleHostPenGymWrapper — The SOLE bridge between SCRIPT's RL agent and PenGym.

Wraps PenGym's multi-host network environment (``PenGymEnv``) into the
single-host-per-step interface that SCRIPT's PPO agent expects:

    wrapper.set_target(host_addr)
    state = wrapper.reset()          # 1538-dim
    next_state, reward, done, info = wrapper.step(action_idx)  # action 0..15

Internally composes:

* ``PenGymStateAdapter``   — NASim obs → SCRIPT 1538-dim vector
* ``ServiceActionMapper``  — service action (0..15) → PenGym flat index
* ``RewardNormalizer``     — PenGym reward → SCRIPT-scale reward
* ``TargetSelector``       — choose next host when auto-advancing

**v2 enhancements** (subnet discovery & firewall-aware pivoting):

* Auto ``subnet_scan`` after each compromise to discover adjacent hosts.
* Failure counter per target — after ``FAILURE_ROTATE_THRESHOLD``
  consecutive failures, the target is marked *blocked* and the selector
  picks an alternate host.
* ``ServiceActionMapper`` now routes ``port_scan`` (→ ``subnet_scan``)
  from compromised hosts instead of the (possibly unreachable) target.

Design reference: docs/pengym_integration_architecture.md §5.1
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from src.envs.wrappers.reward_normalizer import (
    IdentityNormalizer,
    LinearNormalizer,
    RewardNormalizer,
)
from src.envs.wrappers.target_selector import (
    PrioritySensitiveSelector,
    ReachabilityAwareSelector,
    TargetSelector,
)


class SingleHostPenGymWrapper:
    """Wrap ``PenGymEnv`` into SCRIPT's single-host-per-step interface.

    This is the **only** integration point between SCRIPT and PenGym.
    All training loops and evaluation scripts should use this wrapper
    rather than accessing ``PenGymEnv``, ``PenGymStateAdapter``, or
    ``ServiceActionMapper`` directly.
    """

    # After this many consecutive failures (reward ≤ 0) on a target,
    # mark it as *blocked* and rotate to the next reachable host.
    FAILURE_ROTATE_THRESHOLD = 5

    # -----------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------

    def __init__(
        self,
        scenario_path: str,
        fully_obs: bool = True,
        flat_actions: bool = True,
        flat_obs: bool = True,
        reward_normalizer: Optional[RewardNormalizer] = None,
        target_selector: Optional[TargetSelector] = None,
        seed: int = 42,
        auto_select_target: bool = True,
    ):
        """
        Args:
            scenario_path: Path to a NASim scenario YAML file.
            fully_obs: PenGym fully-observable mode (default ``True``).
            flat_actions: Use flat action space (must be ``True`` for mapper).
            flat_obs: Use flat observation (must be ``True`` for adapter).
            reward_normalizer: Strategy for reward scaling.  Defaults to
                ``LinearNormalizer(src_min=-1, src_max=100,
                dst_min=-10, dst_max=1000)``.
            target_selector: Strategy for choosing the next target host.
                Defaults to ``PrioritySensitiveSelector()``.
            seed: Random seed for reproducibility.
            auto_select_target: When ``True``, automatically advance to
                the next uncompromised target when the current one is
                compromised.
        """
        self.scenario_path = str(scenario_path)
        self.fully_obs = fully_obs
        self.flat_actions = flat_actions
        self.flat_obs = flat_obs
        self.seed = seed
        self.auto_select_target = auto_select_target

        self.reward_normalizer: RewardNormalizer = (
            reward_normalizer
            if reward_normalizer is not None
            else LinearNormalizer()
        )
        self.target_selector: TargetSelector = (
            target_selector
            if target_selector is not None
            else ReachabilityAwareSelector()
        )

        # Internal components — populated by _build()
        self.env = None
        self.scenario = None
        self.state_adapter = None
        self.action_mapper = None
        self.sas = None

        # Runtime state
        self._current_target: Optional[Tuple[int, int]] = None
        self._flat_obs: Optional[np.ndarray] = None
        self._episode_steps: int = 0
        self._episode_reward: float = 0.0
        self._compromised_hosts: List[Tuple[int, int]] = []

        # Failure-based target rotation (v2)
        self._consecutive_failures: Dict[Tuple[int, int], int] = {}
        self._blocked_targets: Set[Tuple[int, int]] = set()

        # Build
        self._build()

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _build(self) -> None:
        """Create PenGymEnv, StateAdapter, ServiceActionMapper."""
        from src.envs import load as load_pengym_env, utilities
        from src.envs.adapters.state_adapter import PenGymStateAdapter
        from src.envs.adapters.service_action_mapper import ServiceActionMapper
        from src.agent.actions.service_action_space import ServiceActionSpace

        # Ensure NASim simulation mode
        utilities.ENABLE_PENGYM = False
        utilities.ENABLE_NASIM = True

        # Environment
        self.env = load_pengym_env(
            self.scenario_path,
            fully_obs=self.fully_obs,
            flat_actions=self.flat_actions,
            flat_obs=self.flat_obs,
        )
        self.env.action_space.seed(self.seed)
        self.scenario = utilities.scenario

        # Service Action Space (no Action class needed for PenGym-only use)
        self.sas = ServiceActionSpace(action_class=None)

        # Adapters
        self.state_adapter = PenGymStateAdapter(self.scenario)
        self.action_mapper = ServiceActionMapper(self.sas, self.env)

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def state_dim(self) -> int:
        """Always ``1538``."""
        from src.envs.adapters.state_adapter import PenGymStateAdapter as _PSA

        return _PSA.STATE_DIM

    @property
    def action_dim(self) -> int:
        """Always ``16`` (service-level action space)."""
        return self.sas.action_dim

    @property
    def current_target(self) -> Optional[Tuple[int, int]]:
        """Currently targeted host address, or ``None``."""
        return self._current_target

    # -----------------------------------------------------------------
    # Scenario management
    # -----------------------------------------------------------------

    def load_scenario(self, scenario_path: str) -> None:
        """Load a new scenario, resetting **all** internal state.

        Called by ``CurriculumController`` when switching scenarios.

        Args:
            scenario_path: Path to a NASim scenario YAML file.

        Postconditions:
            - ``self.env`` is a fresh ``PenGymEnv``
            - ``self.state_adapter`` matches the new scenario
            - ``self.action_mapper`` matches the new env
            - ``self._current_target`` is ``None`` — call ``reset()`` next
        """
        self.scenario_path = str(scenario_path)
        self._current_target = None
        self._flat_obs = None
        self._episode_steps = 0
        self._episode_reward = 0.0
        self._compromised_hosts = []
        self._consecutive_failures = {}
        self._blocked_targets = set()
        self._build()

    # -----------------------------------------------------------------
    # Target management
    # -----------------------------------------------------------------

    def set_target(self, host_addr: Tuple[int, int]) -> None:
        """Set the current target host explicitly.

        Args:
            host_addr: ``(subnet_id, host_id)`` tuple.

        Raises:
            ValueError: If *host_addr* is not present in the scenario.
        """
        if host_addr not in self.state_adapter.host_num_map:
            raise ValueError(
                f"Host {host_addr} not in scenario host map. "
                f"Valid hosts: {list(self.state_adapter.host_num_map.keys())}"
            )
        self._current_target = host_addr

    def _auto_select_target(self) -> Optional[Tuple[int, int]]:
        """Use the ``TargetSelector`` strategy to pick the next host.

        Passes the ``blocked`` set so the selector skips hosts where
        repeated actions always fail (e.g., firewall-blocked hosts).
        """
        if self._flat_obs is None:
            return None

        available = self.get_available_targets()
        sensitive = self.get_sensitive_hosts()
        return self.target_selector.select(
            available=available,
            sensitive=sensitive,
            host_info_fn=lambda addr: self.get_host_info(addr),
            blocked=self._blocked_targets,
        )

    def _discover_from_compromised(self) -> None:
        """Execute ``subnet_scan`` from every compromised host.

        This is called automatically after a host is compromised so that
        adjacent subnets are discovered *before* the target selector runs.
        Without this, hosts in un-scanned subnets appear as unreachable
        and the selector cannot pick them.

        The subnet_scan steps are **transparent** to the RL agent: they
        don't increment ``_episode_steps`` and their (small negative)
        rewards are absorbed into the environment's internal state.
        """
        for comp_host in list(self._compromised_hosts):
            key = ('subnet_scan', comp_host)
            if key in self.action_mapper.pengym_actions:
                scan_idx = self.action_mapper.pengym_actions[key]
                obs, _, env_done, truncated, _ = self.env.step(scan_idx)
                self._flat_obs = obs if obs.ndim == 1 else obs.flatten()
                if bool(env_done) or bool(truncated):
                    break

    # -----------------------------------------------------------------
    # Core interface: reset / step
    # -----------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset the environment and return the initial state for the current target.

        If no target is set and ``auto_select_target`` is ``True``,
        automatically selects the first available target via the
        ``TargetSelector``.

        Returns:
            1538-dim ``float32`` state vector.

        Raises:
            RuntimeError: If no target is set and auto-select is disabled.
        """
        obs, info = self.env.reset()
        self._flat_obs = obs if obs.ndim == 1 else obs.flatten()

        # Reset episode counters
        self._episode_steps = 0
        self._episode_reward = 0.0
        self._compromised_hosts = []
        self._consecutive_failures = {}
        self._blocked_targets = set()
        self.target_selector.reset()

        # Target selection
        if self._current_target is None or self.auto_select_target:
            self._current_target = self._auto_select_target()

        if self._current_target is None:
            if not self.auto_select_target:
                raise RuntimeError(
                    "No target set and auto_select_target is False. "
                    "Call set_target() before reset()."
                )
            # Fallback: pick first non-internet host
            for addr in self.state_adapter.host_num_map:
                if addr[0] > 0:
                    self._current_target = addr
                    break

        return self.state_adapter.convert(self._flat_obs, self._current_target)

    def step(
        self, service_action_idx: int
    ) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute a service-level action on the current target in PenGym.

        Args:
            service_action_idx: Index ``0..15`` in the service action space.

        Returns:
            A 4-tuple ``(next_state, reward, done, info)`` where:

            - **next_state** — 1538-dim ``float32`` vector of the current target.
            - **reward** — Normalised reward.
            - **done** — ``True`` when all sensitive hosts are compromised.
            - **info** — dict with diagnostic keys:

              ``raw_reward``, ``pengym_action``, ``target_host``,
              ``action_name``, ``mapped``, ``target_compromised``,
              ``network_done``, ``reachable_hosts``, ``compromised_hosts``
        """
        assert self._flat_obs is not None, "Call reset() before step()"
        assert self._current_target is not None, "No target host set"

        # --- Map action (v2: pass compromised hosts for subnet_scan) ---
        pengym_action = self.action_mapper.map_action(
            service_action_idx,
            self._current_target,
            compromised_hosts=list(self._compromised_hosts),
        )
        mapped = pengym_action != -1

        if not mapped:
            pengym_action = self.action_mapper.get_random_valid_action(
                self._current_target,
                compromised_hosts=list(self._compromised_hosts),
            )

        # --- Execute on PenGym ---
        obs, raw_reward, env_done, truncated, env_info = self.env.step(
            int(pengym_action)
        )
        self._flat_obs = obs if obs.ndim == 1 else obs.flatten()

        # --- Normalise reward ---
        reward = self.reward_normalizer.normalize(raw_reward)

        # --- Episode bookkeeping ---
        self._episode_steps += 1
        self._episode_reward += reward

        # --- Check target status ---
        target_info = self.get_host_info(self._current_target)
        target_compromised = (
            target_info is not None and target_info.get("compromised", False)
        )
        if target_compromised and self._current_target not in self._compromised_hosts:
            self._compromised_hosts.append(self._current_target)

        # Determine if target is *fully exploited* for auto-advance.
        # Sensitive hosts need ROOT (access_level ≥ 2); non-sensitive
        # pivot hosts only need USER (access_level ≥ 1).
        sensitive_hosts = set(self.get_sensitive_hosts())
        target_fully_exploited = False
        if target_info is not None:
            access_level = target_info.get("access_level", 0)
            if self._current_target in sensitive_hosts:
                target_fully_exploited = access_level >= 2  # ROOT required
            else:
                target_fully_exploited = access_level >= 1  # USER sufficient

        # done = env says done OR truncated
        done = bool(env_done) or bool(truncated)

        # --- v2: Track consecutive failures for target rotation ---
        if raw_reward <= 0 and not target_compromised:
            self._consecutive_failures[self._current_target] = (
                self._consecutive_failures.get(self._current_target, 0) + 1
            )
        else:
            self._consecutive_failures[self._current_target] = 0

        # If too many consecutive failures, block this target
        if (self._consecutive_failures.get(self._current_target, 0)
                >= self.FAILURE_ROTATE_THRESHOLD):
            self._blocked_targets.add(self._current_target)

        # --- Auto-advance target ---
        should_rotate = (
            self.auto_select_target
            and not done
            and (
                target_fully_exploited
                or self._current_target in self._blocked_targets
            )
        )

        if should_rotate:
            # v2/v3: Always discover from compromised hosts when rotating
            # so the selector has up-to-date visibility of available hosts.
            self._discover_from_compromised()

            if target_fully_exploited:
                # New network position — reset all blocks.  Previously
                # unreachable hosts may now be reachable from the newly
                # compromised subnet.
                self._blocked_targets.clear()
                self._consecutive_failures.clear()

            new_target = self._auto_select_target()
            if new_target is not None:
                self._current_target = new_target
                # Reset failure counter for the new target
                self._consecutive_failures[self._current_target] = 0

        # --- Action name ---
        action_name = (
            self.sas.action_names[service_action_idx]
            if 0 <= service_action_idx < len(self.sas.action_names)
            else "UNKNOWN"
        )

        # --- Build info dict ---
        info = {
            "raw_reward": float(raw_reward),
            "pengym_action": int(pengym_action),
            "target_host": self._current_target,
            "action_name": action_name,
            "mapped": mapped,
            "target_compromised": target_compromised,
            "network_done": done,
            "reachable_hosts": self.state_adapter.get_reachable_hosts(
                self._flat_obs
            ),
            "compromised_hosts": list(self._compromised_hosts),
        }

        # --- Convert state ---
        next_state = self.state_adapter.convert(
            self._flat_obs, self._current_target
        )

        return next_state, reward, done, info

    # -----------------------------------------------------------------
    # Observation helpers
    # -----------------------------------------------------------------

    def get_available_targets(self) -> List[Tuple[int, int]]:
        """Return reachable, uncompromised host addresses."""
        if self._flat_obs is None:
            return []
        targets: List[Tuple[int, int]] = []
        for addr in self.state_adapter.host_num_map:
            info = self.get_host_info(addr)
            if info is not None:
                if (info["reachable"] or info["discovered"]) and not info[
                    "compromised"
                ]:
                    targets.append(addr)
        return targets

    def get_host_info(self, host_addr: Tuple[int, int]) -> Optional[Dict]:
        """Structured info about a host from the current observation.

        Delegates to ``PenGymStateAdapter.get_host_data()``.

        Returns:
            dict with keys ``address``, ``reachable``, ``compromised``,
            ``discovered``, ``access_level``, ``value``, ``os``,
            ``services``, ``ports``, ``processes``, or ``None``.
        """
        if self._flat_obs is None:
            return None
        return self.state_adapter.get_host_data(self._flat_obs, host_addr)

    def get_all_host_states(self) -> Dict[Tuple[int, int], np.ndarray]:
        """Return 1538-dim state vectors for ALL hosts."""
        if self._flat_obs is None:
            return {}
        return self.state_adapter.convert_all_hosts(self._flat_obs)

    def get_sensitive_hosts(self) -> List[Tuple[int, int]]:
        """Goal host addresses defined by the scenario."""
        return self.state_adapter.get_sensitive_hosts()

    def get_episode_stats(self) -> dict:
        """Stats for the current (ongoing or just-finished) episode."""
        return {
            "steps": self._episode_steps,
            "total_reward": self._episode_reward,
            "compromised_hosts": list(self._compromised_hosts),
            "current_target": self._current_target,
        }

    # -----------------------------------------------------------------
    # Misc
    # -----------------------------------------------------------------

    def describe(self) -> str:
        """Human-readable description of the wrapper configuration."""
        lines = [
            "=== SingleHostPenGymWrapper ===",
            f"Scenario:          {self.scenario_path}",
            f"State dim:         {self.state_dim}",
            f"Action dim:        {self.action_dim}",
            f"Reward normalizer: {self.reward_normalizer.describe()}",
            f"Target selector:   {self.target_selector}",
            f"Auto-select:       {self.auto_select_target}",
            f"Seed:              {self.seed}",
        ]
        if self.state_adapter is not None:
            lines.append(f"Hosts:             {self.state_adapter.num_hosts}")
            lines.append(
                f"Sensitive hosts:   {self.get_sensitive_hosts()}"
            )
        if self.action_mapper is not None:
            stats = self.action_mapper.get_mapping_stats()
            lines.append(
                f"Mapping coverage:  {stats.get('coverage_pct', 0):.1f}%"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"SingleHostPenGymWrapper(scenario={Path(self.scenario_path).name}, "
            f"state_dim={self.state_dim}, action_dim={self.action_dim})"
        )
