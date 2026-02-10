"""
Execution Mode Utility - Switch between Simulation and Real-world execution.

Usage:
    from src.envs.mode import set_simulation_mode, set_real_mode, get_current_mode
    
    set_simulation_mode()  # Use NASim simulation
    set_real_mode()        # Use real nmap/metasploit
"""
from src.envs import utilities


def set_simulation_mode():
    """Enable NASim simulation mode (no real network access)."""
    utilities.ENABLE_PENGYM = False
    utilities.ENABLE_NASIM = True
    print("[MODE] Switched to SIMULATION (NASim)")


def set_real_mode():
    """Enable PenGym real-world mode (requires nmap/metasploit)."""
    utilities.ENABLE_PENGYM = True
    utilities.ENABLE_NASIM = False
    print("[MODE] Switched to REAL (PenGym - nmap/metasploit)")


def set_dual_mode():
    """Enable both modes for comparison (logs both results)."""
    utilities.ENABLE_PENGYM = True
    utilities.ENABLE_NASIM = True
    print("[MODE] Switched to DUAL (both PenGym + NASim)")


def get_current_mode() -> str:
    """Get current execution mode."""
    if utilities.ENABLE_PENGYM and utilities.ENABLE_NASIM:
        return "DUAL"
    elif utilities.ENABLE_PENGYM:
        return "REAL"
    elif utilities.ENABLE_NASIM:
        return "SIMULATION"
    else:
        return "NONE (disabled)"


def print_mode_status():
    """Print current mode status."""
    print(f"[MODE STATUS]")
    print(f"  ENABLE_PENGYM: {utilities.ENABLE_PENGYM}")
    print(f"  ENABLE_NASIM:  {utilities.ENABLE_NASIM}")
    print(f"  Current Mode:  {get_current_mode()}")
