"""
PenGymHostAdapter — Bridge between SingleHostPenGymWrapper and SCRIPT's HOST interface.

SCRIPT's training loops (``KnowledgeExplorer.run_train_episode``,
``KnowledgeExplorer.get_expert_samples``, ``Agent.Evaluate``) all expect
a **target** object (typed as ``HOST``) with the following interface::

    o = target.reset()                       # → ndarray[1538]
    next_o, r, done, result = target.perform_action(a)  # a ∈ 0..15
    target.ip                                 # str identifier
    target.info                               # Host_info-like object
    target.env_data                           # dict with 'vulnerability' key

This module wraps ``SingleHostPenGymWrapper`` to conform to that contract.

Usage::

    from src.envs.wrappers.single_host_wrapper import SingleHostPenGymWrapper
    from src.envs.adapters.pengym_host_adapter import PenGymHostAdapter

    wrapper = SingleHostPenGymWrapper(scenario_path="scenarios/tiny.yml")
    host = PenGymHostAdapter(wrapper, name="tiny")
    o = host.reset()
    next_o, r, done, result = host.perform_action(3)

For CRL (continual reinforcement learning), create one adapter per scenario::

    tasks = [
        PenGymHostAdapter.from_scenario("scenarios/tiny.yml", name="tiny"),
        PenGymHostAdapter.from_scenario("scenarios/small-linear.yml", name="small"),
    ]
"""

import numpy as np
from typing import Dict, Optional, Set, Tuple

from src.agent.defination import Host_info


class PenGymHostAdapter:
    """Adapt ``SingleHostPenGymWrapper`` to SCRIPT's ``HOST`` interface.

    Attributes:
        ip: Human-readable identifier for this task/scenario.
        info: ``Host_info``-like object with mocked fields.
        env_data: dict with ``'vulnerability'`` key (for ``Agent_CL`` logging).
        action_history: Set of actions taken this episode (for diagnostics).

    Notes:
        NASim's ``HostVector`` uses **class-level** index variables that are
        overwritten each time a new scenario is loaded.  When multiple adapters
        coexist (multi-task CRL), creating all wrappers upfront corrupts the
        shared class state.  To avoid this, the :meth:`from_scenario` factory
        defers wrapper creation until :meth:`reset` is first called.  On each
        ``reset()`` the adapter checks whether the NASim class state still
        belongs to *this* scenario and recreates the wrapper if necessary.
    """

    # Class-level tracker: which scenario's NASim ``HostVector`` class vars
    # are currently active.  Compared against ``self._scenario_path`` on each
    # ``reset()`` to decide whether the wrapper needs to be rebuilt.
    _active_scenario: Optional[str] = None

    def __init__(
        self,
        wrapper=None,
        name: str = "pengym",
        *,
        _scenario_path: Optional[str] = None,
        _seed: int = 42,
        _wrapper_kwargs: Optional[Dict] = None,
    ):
        """
        Args:
            wrapper: A ``SingleHostPenGymWrapper`` instance (already built).
                May be ``None`` for lazy/deferred mode (used by
                :meth:`from_scenario`).
            name: Human-readable identifier used as ``self.ip`` and
                in ``env_data['vulnerability']``.
            _scenario_path: (internal) Scenario YAML path for lazy creation.
            _seed: (internal) Random seed for lazy wrapper creation.
            _wrapper_kwargs: (internal) Extra kwargs for lazy wrapper creation.
        """
        self.wrapper = wrapper
        self._name = name
        self._scenario_path = _scenario_path
        self._seed = _seed
        self._wrapper_kwargs = _wrapper_kwargs or {}

        # --- HOST.ip ---
        self.ip: str = name

        # --- HOST.info (mock) ---
        self.info = Host_info(ip=name)
        self.info.os = "pengym"
        self.info.services = []

        # --- HOST.env_data (for Agent_CL.train_continually logging) ---
        self.env_data: Dict = {"vulnerability": f"pengym_{name}"}

        # --- Diagnostics ---
        self.action_history: Set[int] = set()

        # Unified encoding: return float rewards instead of int
        self._use_float_reward: bool = (
            getattr(wrapper, 'use_unified_encoding', False)
            if wrapper is not None else False
        )

    # -----------------------------------------------------------------
    # Factory
    # -----------------------------------------------------------------

    @classmethod
    def from_scenario(
        cls,
        scenario_path: str,
        name: Optional[str] = None,
        seed: int = 42,
        **wrapper_kwargs,
    ) -> "PenGymHostAdapter":
        """Create an adapter from a scenario file path (lazy / deferred).

        The underlying ``SingleHostPenGymWrapper`` is **not** created here.
        It will be built on the first call to :meth:`reset`, ensuring that
        NASim's class-level ``HostVector`` state is correct at the time the
        environment is actually used.

        This is the recommended way to build task lists for CRL training.

        Args:
            scenario_path: Path to a NASim scenario YAML.
            name: Human-readable name. Defaults to the scenario filename stem.
            seed: Random seed for reproducibility.
            **wrapper_kwargs: Extra kwargs forwarded to ``SingleHostPenGymWrapper``.

        Returns:
            A ``PenGymHostAdapter`` instance (wrapper created on first reset).
        """
        from pathlib import Path

        if name is None:
            name = Path(scenario_path).stem

        # Deferred: store params, create wrapper on reset()
        return cls(
            wrapper=None,
            name=name,
            _scenario_path=str(scenario_path),
            _seed=seed,
            _wrapper_kwargs=wrapper_kwargs,
        )

    # -----------------------------------------------------------------
    # Lazy wrapper management
    # -----------------------------------------------------------------

    def _ensure_wrapper(self):
        """(Re)create the wrapper if NASim class-level state is stale.

        NASim's ``HostVector`` stores dimension indices as **class** attributes.
        When a second scenario is loaded, those indices are overwritten,
        corrupting any environment built from a prior scenario.

        This method checks whether the currently active NASim state belongs to
        *this* adapter's scenario.  If not (or if no wrapper exists yet), it
        (re)builds the ``SingleHostPenGymWrapper``, which triggers
        ``HostVector.vectorize()`` and restores the correct class vars.
        """
        if self._scenario_path is None:
            # Direct wrapper mode — nothing to recreate
            if self.wrapper is None:
                raise RuntimeError(
                    "PenGymHostAdapter has no wrapper and no scenario_path. "
                    "Use from_scenario() or pass a wrapper to __init__."
                )
            return

        need_recreate = (
            self.wrapper is None
            or PenGymHostAdapter._active_scenario != self._scenario_path
        )
        if need_recreate:
            from src.envs.wrappers.single_host_wrapper import SingleHostPenGymWrapper

            self.wrapper = SingleHostPenGymWrapper(
                scenario_path=self._scenario_path,
                seed=self._seed,
                **self._wrapper_kwargs,
            )
            PenGymHostAdapter._active_scenario = self._scenario_path
            self._use_float_reward = getattr(
                self.wrapper, 'use_unified_encoding', False
            )

    # -----------------------------------------------------------------
    # HOST interface: reset / perform_action
    # -----------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset the environment and return initial observation.

        On the first call (or when switching scenarios in multi-task CRL),
        this will (re)create the underlying ``SingleHostPenGymWrapper`` to
        ensure NASim's class-level ``HostVector`` state is correct.

        Returns:
            ``ndarray`` with shape ``(state_dim,)`` and dtype ``float32``
            (1540-dim when unified, 1538-dim legacy).
        """
        self.action_history = set()
        self._ensure_wrapper()
        obs = self.wrapper.reset()
        return obs

    def perform_action(self, action: int) -> Tuple[np.ndarray, float, int, str]:
        """Execute a service-level action.

        Args:
            action: Index ``0..15`` in the service action space.

        Returns:
            4-tuple ``(next_obs, reward, done, result_str)`` where:

            - **next_obs** — state vector (1540-dim unified or 1538-dim legacy).
            - **reward** — Float reward when unified normalizer is active,
              otherwise integer (SCRIPT convention).
            - **done** — ``1`` if episode finished, ``0`` otherwise.
            - **result_str** — Human-readable result string.
        """
        self.action_history.add(action)

        next_obs, reward, done, info = self.wrapper.step(action)

        # Unified encoding: keep float reward for [-1,+1] range.
        # Legacy: integer reward (SCRIPT convention).
        r = float(reward) if self._use_float_reward else int(round(reward))
        d = 1 if done else 0

        # Build result string from wrapper info
        action_name = info.get("action_name", f"action_{action}")
        mapped = info.get("mapped", True)
        target_compromised = info.get("target_compromised", False)

        if target_compromised:
            result_str = f"[SUCCESS] {action_name} → target compromised"
        elif not mapped:
            result_str = f"[UNMAPPED] {action_name} → no valid mapping"
        elif r > 0:
            result_str = f"[HIT] {action_name} → reward={r}"
        else:
            result_str = f"[MISS] {action_name} → reward={r}"

        return next_obs, r, d, result_str

    # -----------------------------------------------------------------
    # Scenario management (for CRL)
    # -----------------------------------------------------------------

    def load_scenario(self, scenario_path: str, name: Optional[str] = None):
        """Switch to a different scenario (for wrapper reuse).

        Args:
            scenario_path: Path to a NASim scenario YAML.
            name: New name. If ``None``, derived from scenario filename.
        """
        from pathlib import Path

        self.wrapper.load_scenario(scenario_path)
        if name is None:
            name = Path(scenario_path).stem
        self._name = name
        self.ip = name
        self.info = Host_info(ip=name)
        self.info.os = "pengym"
        self.env_data = {"vulnerability": f"pengym_{name}"}
        self.action_history = set()

    # -----------------------------------------------------------------
    # String repr
    # -----------------------------------------------------------------

    def __repr__(self) -> str:
        return f"PenGymHostAdapter(name={self._name!r})"
