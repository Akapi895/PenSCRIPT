"""
Adapters for bridging SCRIPT agent ↔ PenGym environment.

Exports:
  - PenGymStateAdapter   : PenGym obs → SCRIPT 1538-dim state
  - ActionMapper          : CVE-level action mapping (legacy/research)
  - ServiceActionMapper   : Service-level action mapping (default)
"""
from .state_adapter import PenGymStateAdapter
from .action_mapper import ActionMapper
from .service_action_mapper import ServiceActionMapper

__all__ = [
    'PenGymStateAdapter',
    'ActionMapper',           # CVE-level (legacy/research)
    'ServiceActionMapper',    # Service-level (default for all new code)
]
