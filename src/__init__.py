"""
Fusion Standalone - RL Pentest Training Framework
Root package with path utilities
"""
import os
from pathlib import Path

# Project root detection
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Standard paths
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SCENARIOS_DIR = DATA_DIR / "scenarios"
CONFIG_DIR = DATA_DIR / "config"
LOGS_DIR = OUTPUT_DIR / "logs"
MODELS_DIR = OUTPUT_DIR / "models"
TENSORBOARD_DIR = OUTPUT_DIR / "tensorboard"

# Ensure output directories exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)


def get_scenario_path(name: str) -> Path:
    """Get full path to scenario file"""
    # Try with extension first
    path = SCENARIOS_DIR / name
    if path.exists():
        return path
    # Try adding common extensions
    for ext in ['.json', '.yml', '.yaml']:
        path = SCENARIOS_DIR / f"{name}{ext}"
        if path.exists():
            return path
    return SCENARIOS_DIR / name


def get_model_path(name: str) -> Path:
    """Get full path to model file"""
    if not name.endswith('.pt'):
        name = f"{name}.pt"
    return MODELS_DIR / name


def get_config_path(name: str) -> Path:
    """Get full path to config file"""
    return CONFIG_DIR / name


__all__ = [
    'PROJECT_ROOT',
    'DATA_DIR',
    'OUTPUT_DIR',
    'SCENARIOS_DIR',
    'CONFIG_DIR',
    'LOGS_DIR',
    'MODELS_DIR',
    'TENSORBOARD_DIR',
    'get_scenario_path',
    'get_model_path',
    'get_config_path',
]
