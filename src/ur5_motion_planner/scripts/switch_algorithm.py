#!/usr/bin/env python3
"""
Algorithm Switcher Script for UR5 Motion Planner

This script helps users easily switch between different RL algorithms
by updating the rl_config.yaml file.

Usage:
    python switch_algorithm.py [algorithm_name]
    
Available algorithms: SAC, TD3, PPO, DDPG, A2C

Examples:
    python switch_algorithm.py SAC
    python switch_algorithm.py TD3
    python switch_algorithm.py PPO
"""

import sys
import os
import yaml
from pathlib import Path

# Available algorithm configurations
ALGORITHM_CONFIGS = {
    'SAC': {
        'algorithm': 'SAC',
        'learning_rate': 0.0003,
        'buffer_size': 1000000,
        'batch_size': 256,
        'tau': 0.005,
        'ent_coef': 'auto',
        'target_update_interval': 1,
        'total_timesteps': 100000,
        'description': 'Soft Actor-Critic - Best overall performance'
    },
    'TD3': {
        'algorithm': 'TD3',
        'learning_rate': 0.001,
        'buffer_size': 1000000,
        'batch_size': 100,
        'tau': 0.005,
        'policy_delay': 2,
        'target_policy_noise': 0.2,
        'target_noise_clip': 0.5,
        'total_timesteps': 150000,
        'description': 'Twin Delayed DDPG - Most stable'
    },
    'PPO': {
        'algorithm': 'PPO',
        'learning_rate': 0.0003,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'clip_range': 0.2,
        'vf_coef': 0.5,
        'ent_coef_ppo': 0.0,
        'max_grad_norm': 0.5,
        'total_timesteps': 200000,
        'description': 'Proximal Policy Optimization - Most reliable'
    },
    'DDPG': {
        'algorithm': 'DDPG',
        'learning_rate': 0.001,
        'buffer_size': 1000000,
        'batch_size': 100,
        'tau': 0.005,
        'total_timesteps': 100000,
        'description': 'Deep Deterministic Policy Gradient - Simple baseline'
    },
    'A2C': {
        'algorithm': 'A2C',
        'learning_rate': 0.0007,
        'n_steps': 5,
        'vf_coef': 0.25,
        'ent_coef_ppo': 0.01,
        'max_grad_norm': 0.5,
        'total_timesteps': 150000,
        'description': 'Advantage Actor-Critic - Fast training'
    }
}

def load_config(config_path):
    """Load existing configuration."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None

def save_config(config, config_path):
    """Save configuration to file."""
    try:
        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False, indent=2)
        return True
    except Exception as e:
        print(f"Error saving config file: {e}")
        return False

def update_algorithm_config(config, algorithm_name):
    """Update configuration with algorithm-specific parameters."""
    if algorithm_name not in ALGORITHM_CONFIGS:
        return False
    
    alg_config = ALGORITHM_CONFIGS[algorithm_name]
    
    # Update model section
    if 'model' not in config:
        config['model'] = {}
    
    config['model']['algorithm'] = alg_config['algorithm']
    
    # Update algorithm-specific parameters
    for key, value in alg_config.items():
        if key not in ['algorithm', 'total_timesteps', 'description']:
            config['model'][key] = value
    
    # Update training timesteps
    if 'training' not in config:
        config['training'] = {}
    config['training']['total_timesteps'] = alg_config['total_timesteps']
    
    # Update paths to be algorithm-specific
    alg_lower = algorithm_name.lower()
    config['training']['tensorboard_log'] = f"./ur5_{alg_lower}_tensorboard/"
    config['training']['model_save_path'] = f"models/{alg_lower}_ur5_reacher_model.zip"
    
    return True

def print_algorithm_info():
    """Print information about available algorithms."""
    print("\n" + "="*60)
    print("AVAILABLE RL ALGORITHMS FOR UR5 REACHING")
    print("="*60)
    
    for alg_name, alg_config in ALGORITHM_CONFIGS.items():
        print(f"\n{alg_name}:")
        print(f"  Description: {alg_config['description']}")
        print(f"  Training timesteps: {alg_config['total_timesteps']:,}")
        print(f"  Learning rate: {alg_config['learning_rate']}")
    
    print("\n" + "="*60)
    print("USAGE: python switch_algorithm.py <ALGORITHM_NAME>")
    print("Example: python switch_algorithm.py SAC")
    print("="*60)

def main():
    # Find config file
    script_dir = Path(__file__).parent
    config_path = script_dir / "config" / "rl_config.yaml"
    
    # If not found, try parent directory structure
    if not config_path.exists():
        config_path = script_dir / ".." / "config" / "rl_config.yaml"
    
    if not config_path.exists():
        print(f"Error: Could not find rl_config.yaml")
        print(f"Searched in: {config_path}")
        return 1
    
    # Check command line arguments
    if len(sys.argv) != 2:
        print_algorithm_info()
        return 1
    
    algorithm_name = sys.argv[1].upper()
    
    # Validate algorithm
    if algorithm_name not in ALGORITHM_CONFIGS:
        print(f"Error: Unknown algorithm '{algorithm_name}'")
        print(f"Available algorithms: {list(ALGORITHM_CONFIGS.keys())}")
        return 1
    
    # Load current config
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    if config is None:
        return 1
    
    # Update configuration
    print(f"Switching to algorithm: {algorithm_name}")
    if not update_algorithm_config(config, algorithm_name):
        print(f"Error: Failed to update configuration for {algorithm_name}")
        return 1
    
    # Save updated configuration
    if not save_config(config, config_path):
        print("Error: Failed to save updated configuration")
        return 1
    
    alg_info = ALGORITHM_CONFIGS[algorithm_name]
    print(f"\nSuccessfully switched to {algorithm_name}!")
    print(f"Description: {alg_info['description']}")
    print(f"Training timesteps: {alg_info['total_timesteps']:,}")
    print(f"Config saved to: {config_path}")
    print(f"\nYou can now run your training with the {algorithm_name} algorithm.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
