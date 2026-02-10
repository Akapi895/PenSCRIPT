"""
PenGym Environment Package - Gymnasium-compliant RL environment for pentesting
"""
from nasim.scenarios import make_benchmark_scenario
from nasim.scenarios import load_scenario
from .core.environment import PenGymEnv
from . import utilities


def create_environment(scenario_name, fully_obs=False, flat_actions=True,
                       flat_obs=True):
    """Create a PenGym environment from a benchmark scenario.

    Args:
        scenario_name (str): name of the benchmark scenario to use
        fully_obs (bool): whether the environment uses fully observable mode
            (default=False)
        flat_actions (bool): whether to use a flat action space (default=True)
        flat_obs (bool): whether to use a 1D observation space (default=True)

    Returns:
        PenGymEnv: the created PenGym environment
    """
    scenario = make_benchmark_scenario(scenario_name)
    utilities.scenario = scenario
    return PenGymEnv(scenario, fully_obs, flat_actions, flat_obs)


def load(path, fully_obs=False, flat_actions=True, flat_obs=True):
    """Load a PenGym environment from a scenario file.

    Args:
        path (str): path to scenario file
        fully_obs (bool): whether the environment uses fully observable mode
            (default=False)
        flat_actions (bool): whether to use a flat action space (default=True)
        flat_obs (bool): whether to use a 1D observation space (default=True)

    Returns:
        PenGymEnv: the created PenGym environment
    """
    scenario = load_scenario(path)
    utilities.scenario = scenario
    return PenGymEnv(scenario, fully_obs, flat_actions, flat_obs)


__all__ = [
    'create_environment',
    'load',
    'PenGymEnv',
    'utilities',
]
