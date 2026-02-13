"""
Reward Normalizer — Configurable strategies for bridging PenGym ↔ SCRIPT reward scales.

PenGym (NASim) rewards are scenario-defined (typically [-cost, +value], range ~[-1, 100]).
SCRIPT simulation rewards are much larger ([-10, +1000]).

These normalizers transform PenGym rewards into a range compatible with
SCRIPT's PPO critic so that transferred or jointly-trained policies
receive value estimates of an appropriate magnitude.

Usage::

    from src.envs.wrappers.reward_normalizer import LinearNormalizer

    norm = LinearNormalizer(src_min=-1, src_max=100, dst_min=-10, dst_max=1000)
    scaled = norm.normalize(raw_reward)
"""

from abc import ABC, abstractmethod


class RewardNormalizer(ABC):
    """Abstract base for reward normalization strategies."""

    @abstractmethod
    def normalize(self, raw_reward: float) -> float:
        """Transform a raw PenGym reward to SCRIPT-compatible scale.

        Args:
            raw_reward: The reward returned by ``PenGymEnv.step()``.

        Returns:
            Scaled reward suitable for the PPO update.
        """
        ...

    @abstractmethod
    def describe(self) -> str:
        """Return a human-readable description of this normalizer."""
        ...

    def __repr__(self) -> str:
        return self.describe()


class LinearNormalizer(RewardNormalizer):
    """Linear scaling from PenGym range to SCRIPT range.

    Default maps ``[-1, 100] → [-10, 1000]``.

    Any raw reward that falls outside ``[src_min, src_max]`` is clamped
    before mapping.
    """

    def __init__(
        self,
        src_min: float = -1.0,
        src_max: float = 100.0,
        dst_min: float = -10.0,
        dst_max: float = 1000.0,
    ):
        """
        Args:
            src_min: Minimum expected PenGym reward (clamp floor).
            src_max: Maximum expected PenGym reward (clamp ceiling).
            dst_min: Target minimum (maps from *src_min*).
            dst_max: Target maximum (maps from *src_max*).
        """
        self.src_min = src_min
        self.src_max = src_max
        self.dst_min = dst_min
        self.dst_max = dst_max

    def normalize(self, raw_reward: float) -> float:
        clamped = max(self.src_min, min(self.src_max, raw_reward))
        src_range = self.src_max - self.src_min
        ratio = (clamped - self.src_min) / max(src_range, 1e-8)
        return self.dst_min + ratio * (self.dst_max - self.dst_min)

    def describe(self) -> str:
        return (
            f"LinearNormalizer([{self.src_min}, {self.src_max}] "
            f"→ [{self.dst_min}, {self.dst_max}])"
        )


class ClipNormalizer(RewardNormalizer):
    """Clip-and-scale: divide by *scale*, then clip to ``[-clip, +clip]``.

    Commonly used in RL transfer learning to avoid extreme advantage values.
    """

    def __init__(self, scale: float = 100.0, clip: float = 10.0):
        """
        Args:
            scale: Divisor applied before clipping.
            clip:  Symmetric absolute limit after scaling.
        """
        self.scale = scale
        self.clip = clip

    def normalize(self, raw_reward: float) -> float:
        return max(-self.clip, min(self.clip, raw_reward / self.scale))

    def describe(self) -> str:
        return f"ClipNormalizer(scale={self.scale}, clip={self.clip})"


class IdentityNormalizer(RewardNormalizer):
    """No-op normalizer — passes through the raw reward unchanged.

    Use this during **evaluation** so that reported metrics reflect the
    true PenGym reward scale.
    """

    def normalize(self, raw_reward: float) -> float:
        return raw_reward

    def describe(self) -> str:
        return "IdentityNormalizer()"
