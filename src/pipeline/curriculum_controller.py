"""
Curriculum Controller — Phase 3 of the CVE Difficulty & Expansion Pipeline.

Manages tiered training phases (T1→T2→T3→T4).
At each phase, the agent trains on scenarios of increasing difficulty.
Phase transition occurs when the agent meets performance thresholds.

Design from docs/cve_difficulty_and_expansion.md §6.
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from collections import deque
from dataclasses import dataclass, field


@dataclass
class PhaseConfig:
    """Configuration for a single curriculum phase."""
    tier: int
    min_episodes: int = 50          # Minimum episodes before transition
    max_episodes: int = 500         # Hard limit per phase
    sr_threshold: float = 0.7       # Success rate to pass
    sr_window: int = 20             # Sliding window for success rate
    variance_threshold: float = 0.15  # Max allowed variance in SR window
    warmup_episodes: int = 10       # Episodes before SR tracking starts


@dataclass
class CurriculumConfig:
    """Overall curriculum configuration."""
    phases: List[PhaseConfig] = field(default_factory=lambda: [
        PhaseConfig(tier=1, sr_threshold=0.70, min_episodes=50, max_episodes=300),
        PhaseConfig(tier=2, sr_threshold=0.60, min_episodes=80, max_episodes=500),
        PhaseConfig(tier=3, sr_threshold=0.50, min_episodes=100, max_episodes=600),
        PhaseConfig(tier=4, sr_threshold=0.40, min_episodes=100, max_episodes=800),
    ])
    seed: int = 42
    scenarios_per_tier: int = 5    # How many scenarios to use per tier
    recycle_scenarios: bool = True  # Re-use scenarios within a phase


class CurriculumController:
    """Manages curriculum phase transitions during RL training.

    Usage:
        controller = CurriculumController(config, scenario_pipeline)
        while not controller.is_complete():
            scenario_path = controller.get_next_scenario()
            env = nasim_load(scenario_path)
            # ... train one episode ...
            controller.record_episode(success=True, reward=100.0, steps=15)
    """

    def __init__(self, config: CurriculumConfig,
                 scenario_dir: str,
                 log_dir: Optional[str] = None):
        """
        Args:
            config: Curriculum configuration
            scenario_dir: Path to generated/compiled scenarios directory
            log_dir: Optional directory for logging phase transitions
        """
        self.config = config
        self.scenario_dir = Path(scenario_dir)
        self.log_dir = Path(log_dir) if log_dir else None

        self.current_phase_idx = 0
        self.total_episodes = 0

        # Per-phase tracking
        self._phase_episodes = 0
        self._phase_successes = deque(maxlen=100)
        self._phase_rewards = deque(maxlen=100)
        self._sr_window = deque(maxlen=self.current_phase.sr_window)

        # Scenario pool per phase
        self._scenario_pools: Dict[int, List[str]] = {}
        self._scenario_idx: Dict[int, int] = {}
        self._build_scenario_pools()

        # History
        self.phase_history: List[Dict] = []
        self._transition_log: List[Dict] = []

        # Random
        self._rng = random.Random(config.seed)

    @property
    def current_phase(self) -> PhaseConfig:
        """Current phase configuration."""
        return self.config.phases[self.current_phase_idx]

    @property
    def current_tier(self) -> int:
        """Current difficulty tier."""
        return self.current_phase.tier

    @property
    def is_final_phase(self) -> bool:
        return self.current_phase_idx >= len(self.config.phases) - 1

    def _build_scenario_pools(self):
        """Build pools of scenario paths for each tier."""
        for phase in self.config.phases:
            tier = phase.tier
            pattern = f"*_T{tier}_*.yml"
            scenarios = sorted(
                str(p) for p in self.scenario_dir.glob(pattern))
            if not scenarios:
                print(f"  WARNING: No scenarios found for T{tier} in "
                      f"{self.scenario_dir}")
            else:
                # Limit to scenarios_per_tier
                if len(scenarios) > self.config.scenarios_per_tier:
                    rng = random.Random(self.config.seed + tier)
                    scenarios = rng.sample(scenarios,
                                           self.config.scenarios_per_tier)
            self._scenario_pools[tier] = scenarios
            self._scenario_idx[tier] = 0

    def get_next_scenario(self) -> str:
        """Get the next scenario path for the current phase.

        Cycles through the scenario pool for the current tier.
        """
        tier = self.current_tier
        pool = self._scenario_pools.get(tier, [])
        if not pool:
            raise RuntimeError(f"No scenarios available for tier {tier}")

        idx = self._scenario_idx[tier]
        path = pool[idx % len(pool)]
        self._scenario_idx[tier] = (idx + 1) % len(pool)
        return path

    def record_episode(self, success: bool, reward: float = 0.0,
                        steps: int = 0) -> Dict:
        """Record the result of a training episode.

        Args:
            success: Whether the episode was successful
            reward: Total episode reward
            steps: Number of steps taken

        Returns:
            Dict with current status and any transition info
        """
        self.total_episodes += 1
        self._phase_episodes += 1
        self._phase_successes.append(1 if success else 0)
        self._phase_rewards.append(reward)
        self._sr_window.append(1 if success else 0)

        status = {
            'phase': self.current_phase_idx + 1,
            'tier': self.current_tier,
            'phase_episode': self._phase_episodes,
            'total_episodes': self.total_episodes,
            'success_rate': self.get_success_rate(),
            'transition': None,
        }

        # Check for phase transition
        if self._should_transition():
            transition_info = self._do_transition()
            status['transition'] = transition_info

        return status

    def get_success_rate(self) -> float:
        """Current success rate over the sliding window."""
        if len(self._sr_window) == 0:
            return 0.0
        return np.mean(list(self._sr_window))

    def get_sr_variance(self) -> float:
        """Variance of success rate in the current window."""
        if len(self._sr_window) < 5:
            return 1.0  # High variance = not stable yet
        return np.var(list(self._sr_window))

    def _should_transition(self) -> bool:
        """Check if conditions for phase transition are met."""
        phase = self.current_phase

        # Hard limit
        if self._phase_episodes >= phase.max_episodes:
            return True

        # Minimum episodes
        if self._phase_episodes < phase.min_episodes:
            return False

        # Warmup
        if self._phase_episodes < phase.warmup_episodes:
            return False

        # Success rate threshold
        sr = self.get_success_rate()
        if sr < phase.sr_threshold:
            return False

        # Variance check (stability)
        if self.get_sr_variance() > phase.variance_threshold:
            return False

        return True

    def _do_transition(self) -> Dict:
        """Execute phase transition."""
        old_phase = self.current_phase_idx
        old_tier = self.current_tier
        sr = self.get_success_rate()
        reason = 'threshold_met' if sr >= self.current_phase.sr_threshold \
            else 'max_episodes'

        transition = {
            'from_phase': old_phase + 1,
            'from_tier': old_tier,
            'episodes_in_phase': self._phase_episodes,
            'final_sr': sr,
            'final_avg_reward': np.mean(list(self._phase_rewards)) if self._phase_rewards else 0,
            'reason': reason,
        }

        self.phase_history.append(transition)
        self._transition_log.append(transition)

        # Advance to next phase (if available)
        if not self.is_final_phase:
            self.current_phase_idx += 1
            self._phase_episodes = 0
            self._phase_successes.clear()
            self._phase_rewards.clear()
            self._sr_window = deque(maxlen=self.current_phase.sr_window)

            transition['to_phase'] = self.current_phase_idx + 1
            transition['to_tier'] = self.current_tier
        else:
            transition['to_phase'] = None
            transition['to_tier'] = None
            transition['curriculum_complete'] = True

        return transition

    def is_complete(self) -> bool:
        """Check if all curriculum phases are done."""
        if self.is_final_phase and self._should_transition():
            return True
        return False

    def get_status(self) -> Dict:
        """Get current curriculum status."""
        return {
            'current_phase': self.current_phase_idx + 1,
            'total_phases': len(self.config.phases),
            'current_tier': self.current_tier,
            'phase_episodes': self._phase_episodes,
            'total_episodes': self.total_episodes,
            'success_rate': self.get_success_rate(),
            'sr_variance': self.get_sr_variance(),
            'threshold': self.current_phase.sr_threshold,
            'min_episodes_remaining': max(
                0, self.current_phase.min_episodes - self._phase_episodes),
            'max_episodes_remaining': max(
                0, self.current_phase.max_episodes - self._phase_episodes),
            'scenarios_in_pool': len(
                self._scenario_pools.get(self.current_tier, [])),
            'phase_history': self.phase_history,
        }

    def save_log(self, output_path: str):
        """Save curriculum log to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        log = {
            'config': {
                'seed': self.config.seed,
                'phases': [
                    {
                        'tier': p.tier,
                        'sr_threshold': p.sr_threshold,
                        'min_episodes': p.min_episodes,
                        'max_episodes': p.max_episodes,
                    }
                    for p in self.config.phases
                ],
            },
            'transitions': self.phase_history,
            'final_status': self.get_status(),
        }

        with open(output_path, 'w') as f:
            json.dump(log, f, indent=2, default=str)
        return str(output_path)


class FlatController:
    """A 'flat' controller that randomly samples from all tiers.

    Used as the baseline comparison against CurriculumController.
    Same interface so the training loop works with either.
    """

    def __init__(self, scenario_dir: str, max_episodes: int = 2000,
                 seed: int = 42):
        self.scenario_dir = Path(scenario_dir)
        self.max_episodes = max_episodes
        self.total_episodes = 0
        self._rng = random.Random(seed)

        # Collect all scenarios
        self._all_scenarios = sorted(
            str(p) for p in self.scenario_dir.glob("*.yml"))
        if not self._all_scenarios:
            raise RuntimeError(f"No scenarios in {self.scenario_dir}")

        self._sr_window = deque(maxlen=50)
        self._rewards = deque(maxlen=100)
        self.phase_history = []

    @property
    def current_tier(self) -> int:
        return 0  # "all tiers"

    def get_next_scenario(self) -> str:
        return self._rng.choice(self._all_scenarios)

    def record_episode(self, success: bool, reward: float = 0.0,
                        steps: int = 0) -> Dict:
        self.total_episodes += 1
        self._sr_window.append(1 if success else 0)
        self._rewards.append(reward)
        return {
            'phase': 0,
            'tier': 0,
            'total_episodes': self.total_episodes,
            'success_rate': self.get_success_rate(),
            'transition': None,
        }

    def get_success_rate(self) -> float:
        if not self._sr_window:
            return 0.0
        return np.mean(list(self._sr_window))

    def is_complete(self) -> bool:
        return self.total_episodes >= self.max_episodes

    def get_status(self) -> Dict:
        return {
            'mode': 'flat',
            'total_episodes': self.total_episodes,
            'max_episodes': self.max_episodes,
            'success_rate': self.get_success_rate(),
            'avg_reward': np.mean(list(self._rewards)) if self._rewards else 0,
        }

    def save_log(self, output_path: str):
        log = {
            'mode': 'flat',
            'total_episodes': self.total_episodes,
            'final_sr': self.get_success_rate(),
            'avg_reward': np.mean(list(self._rewards)) if self._rewards else 0,
        }
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(log, f, indent=2, default=str)
        return str(output_path)
