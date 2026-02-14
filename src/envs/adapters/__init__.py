"""
Adapters for bridging SCRIPT agent ↔ PenGym environment.

Exports:
  - PenGymStateAdapter   : PenGym obs → SCRIPT 1538-dim state
  - ActionMapper          : CVE-level action mapping (legacy/research)
  - ServiceActionMapper   : Service-level action mapping (default)
  - PenGymHostAdapter     : Wrapper → SCRIPT HOST interface (for CRL)
"""
from .state_adapter import PenGymStateAdapter
from .action_mapper import ActionMapper
from .service_action_mapper import ServiceActionMapper
from .pengym_host_adapter import PenGymHostAdapter

__all__ = [
    'PenGymStateAdapter',
    'ActionMapper',           # CVE-level (legacy/research)
    'ServiceActionMapper',    # Service-level (default for all new code)
    'PenGymHostAdapter',      # HOST bridge for SCRIPT CRL
]
