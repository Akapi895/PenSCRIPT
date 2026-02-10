#!/usr/bin/env python3
"""
Fusion Standalone - Automated Setup and Run Script
Single command to setup environment and start training.

Usage:
    python setup_and_run.py                     # Default: setup + train with chain_1.json
    python setup_and_run.py --skip-setup        # Skip venv creation, just run
    python setup_and_run.py --scenario tiny.yml # Use different scenario
    python setup_and_run.py --mode eval         # Evaluation mode
"""
import sys
import os
import subprocess
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()
VENV_DIR = PROJECT_ROOT / "venv"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"


def run_command(cmd: list, cwd: Path = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"  > {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd or PROJECT_ROOT, check=check, capture_output=False)


def check_python():
    """Check Python version."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"Error: Python 3.9+ required, found {version.major}.{version.minor}")
        sys.exit(1)
    print(f"[OK] Python {version.major}.{version.minor}.{version.micro}")


def check_venv_exists() -> bool:
    """Check if virtual environment exists."""
    if sys.platform == "win32":
        python_path = VENV_DIR / "Scripts" / "python.exe"
    else:
        python_path = VENV_DIR / "bin" / "python"
    return python_path.exists()


def create_venv():
    """Create virtual environment if not exists."""
    if check_venv_exists():
        print("[OK] Virtual environment already exists")
        return
    
    print("[...] Creating virtual environment...")
    run_command([sys.executable, "-m", "venv", str(VENV_DIR)])
    print("[OK] Virtual environment created")


def get_venv_python() -> str:
    """Get path to venv Python executable."""
    if sys.platform == "win32":
        return str(VENV_DIR / "Scripts" / "python.exe")
    else:
        return str(VENV_DIR / "bin" / "python")


def get_venv_pip() -> str:
    """Get path to venv pip executable."""
    if sys.platform == "win32":
        return str(VENV_DIR / "Scripts" / "pip.exe")
    else:
        return str(VENV_DIR / "bin" / "pip")


def install_dependencies():
    """Install dependencies from requirements.txt."""
    print("[...] Installing dependencies...")
    
    pip_path = get_venv_pip()
    
    # Upgrade pip first
    run_command([pip_path, "install", "--upgrade", "pip"], check=False)
    
    # Install requirements
    run_command([pip_path, "install", "-r", str(REQUIREMENTS_FILE)])
    
    print("[OK] Dependencies installed")


def run_training(args):
    """Run the training script with given arguments."""
    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60 + "\n")
    
    python_path = get_venv_python()
    run_script = PROJECT_ROOT / "run.py"
    
    cmd = [
        python_path, str(run_script),
        "--mode", args.mode,
        "--scenario", args.scenario,
        "--episodes", str(args.episodes),
        "--max-steps", str(args.max_steps),
        "--device", args.device,
        "--env-type", args.env_type,
    ]
    
    if args.model_path:
        cmd.extend(["--model-path", args.model_path])
    
    if args.execution_mode:
        cmd.extend(["--execution-mode", args.execution_mode])
    
    run_command(cmd)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Automated setup and training for Fusion Standalone"
    )
    
    # Setup options
    parser.add_argument("--skip-setup", action="store_true", help="Skip venv creation and pip install")
    parser.add_argument("--setup-only", action="store_true", help="Only setup, don't run training")
    
    # Training options (passed to run.py)
    parser.add_argument("--mode", choices=["train", "eval", "demo"], default="train")
    parser.add_argument("--scenario", type=str, default="tiny.yml")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--env-type", choices=["simulation", "pengym"], default="simulation")
    parser.add_argument("--execution-mode", choices=["sim", "real", "dual"], default="sim",
                        help="Execution mode: sim (NASim), real (nmap/metasploit), dual (both)")
    parser.add_argument("--model-path", type=str, default=None)
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    print("\n" + "="*60)
    print("Fusion Standalone - Automated Setup")
    print("="*60 + "\n")
    
    # Check Python version
    check_python()
    
    # Setup phase
    if not args.skip_setup:
        create_venv()
        install_dependencies()
    else:
        if not check_venv_exists():
            print("Error: Virtual environment not found. Run without --skip-setup first.")
            sys.exit(1)
        print("[SKIP] Using existing virtual environment")
    
    print("\n[OK] Setup complete!")
    
    # Run training if not setup-only
    if not args.setup_only:
        run_training(args)
    else:
        print("\nSetup-only mode. Run training with:")
        print(f"  python run.py --mode train --scenario {args.scenario}")


if __name__ == "__main__":
    main()
