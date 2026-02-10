#!/usr/bin/env python3
"""
Fusion Standalone - RL Pentest Training Entry Point
Single command to setup and run training without external dependencies.
"""
import sys
import os
from pathlib import Path

# ============================================================
# PYTHONPATH Setup - ensures imports work from project root
# ============================================================
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# Now safe to import from src
# ============================================================
import argparse
import json

# Core imports
from src import PROJECT_ROOT, SCENARIOS_DIR, MODELS_DIR, LOGS_DIR, get_scenario_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fusion Standalone - RL Pentest Agent Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --mode train --scenario chain_1.json --episodes 1000
  python run.py --mode eval --scenario chain_1.json --model-path outputs/models/chain_1.pt
  python run.py --mode demo --env-type pengym
        """
    )
    
    # Mode selection
    parser.add_argument(
        "--mode", 
        choices=["train", "eval", "demo"], 
        default="train",
        help="Execution mode: train, eval, or demo (default: train)"
    )
    
    # Environment configuration
    parser.add_argument(
        "--scenario", 
        type=str, 
        default="tiny.yml",
        help="Scenario file name (default: tiny.yml)"
    )
    parser.add_argument(
        "--env-type", 
        choices=["simulation", "pengym"], 
        default="simulation",
        help="Environment type: simulation or pengym (default: simulation)"
    )
    
    # Training parameters
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    
    # Paths
    parser.add_argument("--log-dir", type=str, default=None, help="Log directory")
    parser.add_argument("--model-path", type=str, default=None, help="Model path for eval mode")
    
    # PenGym specific
    parser.add_argument("--pengym-config", type=str, default=None, help="PenGym CONFIG.yml path")
    parser.add_argument("--disable-pengym-real", action="store_true", help="Disable real PenGym execution")
    parser.add_argument("--execution-mode", choices=["sim", "real", "dual"], default="sim",
                        help="Execution mode: sim (NASim only), real (PenGym only), dual (both)")
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_scenario(scenario_path: Path) -> dict:
    """Load scenario configuration."""
    with open(scenario_path, 'r') as f:
        if scenario_path.suffix == '.json':
            return json.load(f)
        else:
            import yaml
            return yaml.safe_load(f)


def create_simulation_env(scenario_path: Path):
    """Create simulation environment from JSON scenario file (Script's HOST class).
    
    Returns a simple wrapper that contains the list of HOST targets.
    """
    from src.agent.host import HOST
    from src.agent.actions import Action
    
    # Load scenario data
    with open(scenario_path, 'r', encoding='utf-8') as f:
        environment_data = json.load(f)
    
    # Create HOST objects for each target in the scenario
    target_list = []
    for host in environment_data:
        ip = host["ip"]
        t = HOST(ip, env_data=host, env_file=scenario_path)
        target_list.append(t)
    
    print(f"[ENV] Loaded {len(target_list)} targets from scenario")
    
    # Return simple environment wrapper
    class SimulationEnv:
        def __init__(self, targets):
            self.targets = targets
            self.current_target_idx = 0
            self.current_target = targets[0] if targets else None
            
            # Environment interface
            from src.agent.host import StateEncoder
            self.state_dim = StateEncoder.state_space
            self.action_dim = Action.action_space
            
        def reset(self):
            self.current_target_idx = 0
            self.current_target = self.targets[0] if self.targets else None
            if self.current_target:
                return self.current_target.reset()
            return None
            
        def step(self, action):
            if self.current_target:
                return self.current_target.perform_action(action)
            return None, 0, True, "No target"
            
        @property
        def action_space(self):
            return type('ActionSpace', (), {'sample': lambda: 0, 'n': self.action_dim})()
    
    return SimulationEnv(target_list)


def create_pengym_env(scenario_path: Path, config_path: str = None, disable_real: bool = False):
    """Create PenGym environment."""
    try:
        from src.envs import load as load_pengym_env, utilities
        
        # Set execution mode
        if disable_real:
            utilities.ENABLE_PENGYM = False
            utilities.ENABLE_NASIM = True
        
        return load_pengym_env(str(scenario_path))
    except ImportError as e:
        print(f"Error: PenGym not available - {e}")
        print("Make sure nasim, python-nmap, and pymetasploit3 are installed.")
        sys.exit(1)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup
    set_seed(args.seed)
    
    # Paths
    scenario_path = get_scenario_path(args.scenario)
    log_dir = Path(args.log_dir) if args.log_dir else LOGS_DIR / args.scenario.replace('.json', '').replace('.yml', '')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Fusion Standalone - RL Pentest Training Framework")
    print(f"{'='*60}")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Mode: {args.mode}")
    print(f"Scenario: {scenario_path}")
    print(f"Environment: {args.env_type}")
    print(f"Device: {args.device}")
    print(f"Log Directory: {log_dir}")
    print(f"{'='*60}\n")
    
    # Check scenario exists
    if not scenario_path.exists():
        print(f"Error: Scenario not found: {scenario_path}")
        print(f"Available scenarios in {SCENARIOS_DIR}:")
        for f in SCENARIOS_DIR.glob('**/*'):
            if f.is_file():
                print(f"  - {f.relative_to(SCENARIOS_DIR)}")
        sys.exit(1)
    
    # Apply execution mode BEFORE creating environment
    if args.env_type == "pengym":
        from src.envs.mode import set_simulation_mode, set_real_mode, set_dual_mode, print_mode_status
        
        if args.execution_mode == "sim":
            set_simulation_mode()
        elif args.execution_mode == "real":
            set_real_mode()
        elif args.execution_mode == "dual":
            set_dual_mode()
        
        print_mode_status()
    
    # ============================================================
    # Use Bot class for proper training/eval pipeline
    # ============================================================
    if args.env_type == "simulation":
        from src.agent.policy.config import PPO_Config
        
        config = PPO_Config(
            train_eps=args.episodes,
            step_limit=args.max_steps,
        )
        
        # Import Bot - the orchestrator class
        # We need to add project root to path for Bot imports
        sys.path.insert(0, str(PROJECT_ROOT / "src" / "agent"))
        
        from src.agent.host import HOST, StateEncoder
        from src.agent.actions import Action
        from src.agent.agent import Agent
        
        # Load scenario and create HOST targets (same as Bot.make_env)
        with open(scenario_path, 'r', encoding='utf-8') as f:
            environment_data = json.load(f)
        
        target_list = []
        for host_data in environment_data:
            ip = host_data["ip"]
            vul = host_data["vulnerability"][0]
            if vul not in Action.Vul_cve_set:
                print(f"[WARN] host vul {vul} is not in Vul_cve_set, skipping")
                continue
            t = HOST(ip, env_data=host_data, env_file=scenario_path)
            target_list.append(t)
        
        print(f"[ENV] Loaded {len(target_list)} targets from scenario")
        print(f"[ENV] State dim: {StateEncoder.state_space}, Action dim: {Action.action_space}")
        
        # Create TensorBoard logger
        from torch.utils.tensorboard import SummaryWriter
        tb_logger = SummaryWriter(log_dir=str(log_dir))
        print(f"[LOG] TensorBoard logs: {log_dir}")
        
        # Create agent
        agent = Agent(
            policy_name="PPO",
            config=config,
            seed=args.seed,
            use_wandb=False,
            logger=tb_logger,
        )
        
        if args.mode == "train":
            print(f"\n{'='*60}")
            print(f"Training: {args.episodes} episodes, {args.max_steps} step limit")
            print(f"{'='*60}\n")
            
            train_matrix = agent.train_with_tqdm(task_list=target_list)
            
            # Save model
            model_dir = MODELS_DIR / args.scenario.replace('.json', '').replace('.yml', '')
            agent.save(model_dir)
            print(f"\nModel saved to: {model_dir}")
            
        elif args.mode == "eval":
            model_dir = Path(args.model_path) if args.model_path else MODELS_DIR / args.scenario.replace('.json', '').replace('.yml', '')
            if not model_dir.exists():
                print(f"Error: Model not found: {model_dir}")
                sys.exit(1)
            
            agent.load(model_dir)
            print(f"Loaded model from: {model_dir}")
            
            attack_path, total_rewards, success_rate = agent.Evaluate(
                target_list=target_list,
                interactive=True,
                verbose=True,
                step_limit=args.max_steps
            )
            print(f"\nEvaluation: Reward={total_rewards}, Success Rate={success_rate*100:.1f}%")
            
        elif args.mode == "demo":
            print("Demo mode: Testing environment targets...")
            for i, target in enumerate(target_list):
                state = target.reset()
                print(f"Target {i}: IP={target.ip}, State dim={len(state)}")
    
    else:
        # PenGym environment mode
        env = create_pengym_env(scenario_path, args.pengym_config, args.disable_pengym_real)
        print(f"[ENV] PenGym environment created")
        print("PenGym training not yet implemented in standalone mode.")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

