#!/usr/bin/env python3
"""
Advanced RL Trainer for UR5 Motion Planner

This module provides a flexible training framework that supports:
- Custom neural network architectures
- Automatic model checkpointing
- Training visualization and plotting
- Integration with Stable Baselines3 algorithms
- Comprehensive training monitoring
"""

import os
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

import torch
import torch.nn as nn
from stable_baselines3 import SAC, TD3, DDPG, PPO, A2C
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym

from .config_loader import RLConfig


class CustomMLP(BaseFeaturesExtractor):
    """
    Custom Multi-Layer Perceptron feature extractor.
    Allows for more flexible network architectures than standard MlpPolicy.
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 256, 
                 hidden_layers: List[int] = None, activation_fn: nn.Module = nn.ReLU):
        """
        Initialize custom MLP network.
        
        Args:
            observation_space: The observation space of the environment
            features_dim: Number of features extracted by the network
            hidden_layers: List of hidden layer sizes (default: [512, 256, 128])
            activation_fn: Activation function to use (default: ReLU)
        """
        super().__init__(observation_space, features_dim)
        
        if hidden_layers is None:
            hidden_layers = [512, 256, 128]
        
        # Build the network layers
        layers = []
        input_dim = observation_space.shape[0]
        
        # Add hidden layers
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation_fn(),
                nn.Dropout(0.1),  # Add dropout for regularization
            ])
            prev_dim = hidden_dim
        
        # Final layer
        layers.append(nn.Linear(prev_dim, features_dim))
        layers.append(activation_fn())
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.mlp(observations)


class CustomActorCriticPolicy(ActorCriticPolicy):
    """
    Custom Actor-Critic policy with configurable network architecture.
    """
    
    def __init__(self, *args, **kwargs):
        # Extract custom network parameters
        self.custom_arch = kwargs.pop('custom_arch', None)
        self.activation_fn = kwargs.pop('activation_fn', nn.ReLU)
        
        super().__init__(*args, **kwargs)
    
    def _build_mlp_extractor(self) -> None:
        """Build the MLP feature extractor."""
        if self.custom_arch:
            self.mlp_extractor = CustomMLP(
                self.observation_space,
                features_dim=self.custom_arch.get('features_dim', 256),
                hidden_layers=self.custom_arch.get('hidden_layers', [512, 256, 128]),
                activation_fn=self.activation_fn
            )
        else:
            super()._build_mlp_extractor()


class TrainingCallback(BaseCallback):
    """
    Custom callback for monitoring training progress and creating plots.
    """
    
    def __init__(self, save_path: str, plot_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.plot_freq = plot_freq
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps = []
        self.losses = []
        
        # Current episode tracking
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        # Track episode progress
        self.current_episode_reward += self.locals.get('rewards', [0])[0]
        self.current_episode_length += 1
        
        # Check if episode is done
        if self.locals.get('dones', [False])[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.timesteps.append(self.num_timesteps)
            
            # Reset for next episode
            self.current_episode_reward = 0
            self.current_episode_length = 0
            
            # Create plots periodically
            if len(self.episode_rewards) % self.plot_freq == 0:
                self._create_plots()
        
        return True
    
    def _create_plots(self):
        """Create and save training progress plots."""
        if len(self.episode_rewards) < 2:
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        ax1.plot(self.episode_rewards)
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.grid(True)
        
        # Moving average of rewards
        if len(self.episode_rewards) >= 10:
            window = min(50, len(self.episode_rewards) // 4)
            moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            ax2.plot(moving_avg)
            ax2.set_title(f'Moving Average Rewards (window={window})')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Average Reward')
            ax2.grid(True)
        
        # Episode lengths
        ax3.plot(self.episode_lengths)
        ax3.set_title('Episode Lengths')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Steps')
        ax3.grid(True)
        
        # Reward distribution
        ax4.hist(self.episode_rewards, bins=30, alpha=0.7)
        ax4.set_title('Reward Distribution')
        ax4.set_xlabel('Total Reward')
        ax4.set_ylabel('Frequency')
        ax4.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.save_path, f'training_progress_{len(self.episode_rewards)}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save metrics as CSV
        metrics_path = os.path.join(self.save_path, 'training_metrics.csv')
        with open(metrics_path, 'w') as f:
            f.write('episode,timestep,reward,length\n')
            for i, (ts, reward, length) in enumerate(zip(self.timesteps, self.episode_rewards, self.episode_lengths)):
                f.write(f'{i+1},{ts},{reward},{length}\n')


class UR5Trainer:
    """
    Advanced trainer for UR5 reinforcement learning with custom networks and monitoring.
    """
    
    def __init__(self, env, config: RLConfig, custom_network_config: Optional[Dict] = None):
        """
        Initialize the trainer.
        
        Args:
            env: The UR5 environment
            config: Configuration object
            custom_network_config: Custom network architecture configuration
        """
        self.env = env
        self.config = config
        self.custom_network_config = custom_network_config or {}
        
        # Setup paths
        self.setup_paths()
        
        # Training state
        self.model = None
        self.training_thread = None
        self.is_training = False
        self.training_start_time = None
        
        # Callbacks
        self.callbacks = []
        
    def setup_paths(self):
        """Setup all necessary directories and paths."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        algorithm_name = self.config.get('model.algorithm', 'SAC')
        
        # Base directories
        self.base_dir = Path("training_runs") / f"{algorithm_name}_{timestamp}"
        self.models_dir = self.base_dir / "models"
        self.plots_dir = self.base_dir / "plots"
        self.logs_dir = self.base_dir / "logs"
        
        # Create directories
        for dir_path in [self.models_dir, self.plots_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.model_path = self.models_dir / "latest_model.zip"
        self.best_model_path = self.models_dir / "best_model.zip"
        self.tensorboard_log = str(self.logs_dir)
        
    def create_custom_policy(self, algorithm_class):
        """
        Create custom policy with user-defined network architecture.
        
        Args:
            algorithm_class: The SB3 algorithm class
            
        Returns:
            Policy class configured with custom network
        """
        if not self.custom_network_config:
            return "MlpPolicy"  # Use default policy
        
        # Store reference to custom network config for the policy classes
        custom_config = self.custom_network_config
        
        # Define custom policy based on algorithm type
        if algorithm_class in [SAC]:
            from stable_baselines3.sac.policies import SACPolicy
            
            class CustomSACPolicy(SACPolicy):
                def __init__(self, *args, **kwargs):
                    kwargs['net_arch'] = custom_config.get('net_arch', [256, 256])
                    kwargs['activation_fn'] = getattr(nn, custom_config.get('activation_fn', 'ReLU'))
                    super().__init__(*args, **kwargs)
                    
            return CustomSACPolicy
            
        elif algorithm_class in [TD3, DDPG]:
            from stable_baselines3.td3.policies import TD3Policy
            
            class CustomTD3Policy(TD3Policy):
                def __init__(self, *args, **kwargs):
                    kwargs['net_arch'] = custom_config.get('net_arch', [256, 256])
                    kwargs['activation_fn'] = getattr(nn, custom_config.get('activation_fn', 'ReLU'))
                    super().__init__(*args, **kwargs)
                    
            return CustomTD3Policy
            
        elif algorithm_class in [PPO, A2C]:
            from stable_baselines3.common.policies import ActorCriticPolicy
            
            class CustomActorCriticPolicy(ActorCriticPolicy):
                def __init__(self, *args, **kwargs):
                    kwargs['net_arch'] = custom_config.get('net_arch', [256, 256])
                    kwargs['activation_fn'] = getattr(nn, custom_config.get('activation_fn', 'ReLU'))
                    super().__init__(*args, **kwargs)
                    
            return CustomActorCriticPolicy
        
        return "MlpPolicy"  # Fallback to default
    
    def setup_callbacks(self):
        """Setup training callbacks for monitoring and checkpointing."""
        # Training progress callback
        progress_callback = TrainingCallback(
            save_path=str(self.plots_dir),
            plot_freq=self.config.get('training.plot_frequency', 50)
        )
        
        # Checkpoint callback - save model every N timesteps
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.get('training.checkpoint_frequency', 10000),
            save_path=str(self.models_dir),
            name_prefix='checkpoint'
        )
        
        self.callbacks = [progress_callback, checkpoint_callback]
        
        return self.callbacks
    
    def create_model(self, algorithm_name: str = None):
        """
        Create the RL model with custom network architecture.
        
        Args:
            algorithm_name: Name of the algorithm to use
        """
        if algorithm_name is None:
            algorithm_name = self.config.get('model.algorithm', 'SAC')
        
        # Get algorithm class
        algorithms = {
            'SAC': SAC, 'TD3': TD3, 'DDPG': DDPG, 'PPO': PPO, 'A2C': A2C
        }
        
        if algorithm_name not in algorithms:
            raise ValueError(f"Algorithm {algorithm_name} not supported")
        
        AlgorithmClass = algorithms[algorithm_name]
        
        # Get model parameters
        model_kwargs = self.config.get_model_kwargs()
        
        # Create custom policy
        policy = self.create_custom_policy(AlgorithmClass)
        
        # Add tensorboard logging
        model_kwargs['tensorboard_log'] = self.tensorboard_log
        
        # Create model
        self.model = AlgorithmClass(
            policy,
            self.env,
            **model_kwargs
        )
        
        print(f"Created {algorithm_name} model with custom architecture")
        if self.custom_network_config:
            print(f"Network config: {self.custom_network_config}")
        
        return self.model
    
    def load_model(self, model_path: str):
        """Load a pre-trained model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        algorithm_name = self.config.get('model.algorithm', 'SAC')
        algorithms = {
            'SAC': SAC, 'TD3': TD3, 'DDPG': DDPG, 'PPO': PPO, 'A2C': A2C
        }
        
        AlgorithmClass = algorithms[algorithm_name]
        self.model = AlgorithmClass.load(model_path, env=self.env)
        
        print(f"Loaded model from: {model_path}")
        return self.model
    
    def save_model(self, path: str = None):
        """Save the current model."""
        if self.model is None:
            print("No model to save!")
            return
        
        if path is None:
            path = self.model_path
        
        self.model.save(path)
        print(f"Model saved to: {path}")
    
    def train(self, total_timesteps: int = None, save_frequency: int = 10000):
        """
        Train the model with automatic saving and monitoring.
        
        Args:
            total_timesteps: Total number of timesteps to train
            save_frequency: How often to save the model (in timesteps)
        """
        if self.model is None:
            self.create_model()
        
        if total_timesteps is None:
            total_timesteps = self.config.get('training.total_timesteps', 100000)
        
        # Setup callbacks
        self.setup_callbacks()
        
        # Start training
        self.is_training = True
        self.training_start_time = time.time()
        
        print(f"Starting training for {total_timesteps:,} timesteps...")
        print(f"Models will be saved to: {self.models_dir}")
        print(f"Plots will be saved to: {self.plots_dir}")
        print(f"Tensorboard logs: {self.tensorboard_log}")
        
        try:
            # Create auto-save callback
            class AutoSaveCallback(BaseCallback):
                def __init__(self, trainer, save_freq):
                    super().__init__()
                    self.trainer = trainer
                    self.save_freq = save_freq
                    self.last_save = 0
                
                def _on_step(self):
                    if self.num_timesteps - self.last_save >= self.save_freq:
                        save_path = self.trainer.models_dir / f"model_{self.num_timesteps}.zip"
                        self.trainer.model.save(save_path)
                        self.trainer.save_model()  # Also save as latest
                        self.last_save = self.num_timesteps
                        print(f"Auto-saved model at timestep {self.num_timesteps}")
                    return True
            
            auto_save_callback = AutoSaveCallback(self, save_frequency)
            all_callbacks = self.callbacks + [auto_save_callback]
            
            # Train the model
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=all_callbacks,
                log_interval=self.config.get('training.log_interval', 4)
            )
            
            # Final save
            self.save_model()
            final_model_path = self.models_dir / f"final_model_{total_timesteps}.zip"
            self.save_model(final_model_path)
            
            training_time = time.time() - self.training_start_time
            print(f"Training completed in {training_time:.2f} seconds")
            
        except KeyboardInterrupt:
            print("Training interrupted by user")
            self.save_model()
            
        except Exception as e:
            print(f"Training error: {e}")
            self.save_model()  # Save what we have
            raise
        
        finally:
            self.is_training = False
    
    def evaluate(self, n_episodes: int = 10, deterministic: bool = True):
        """
        Evaluate the trained model.
        
        Args:
            n_episodes: Number of episodes to evaluate
            deterministic: Whether to use deterministic actions
            
        Returns:
            List of episode rewards
        """
        if self.model is None:
            raise ValueError("No model to evaluate! Train or load a model first.")
        
        print(f"Evaluating model for {n_episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Length = {steps}")
        
        # Save evaluation results
        eval_results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'episodes': episode_rewards
        }
        
        print(f"Evaluation Results:")
        print(f"  Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        print(f"  Mean Length: {eval_results['mean_length']:.1f}")
        
        return eval_results
    
    def create_final_plots(self):
        """Create comprehensive final training plots."""
        if not self.callbacks:
            print("No training data available for plotting")
            return
        
        training_callback = None
        for callback in self.callbacks:
            if isinstance(callback, TrainingCallback):
                training_callback = callback
                break
        
        if training_callback and len(training_callback.episode_rewards) > 0:
            training_callback._create_plots()
            print(f"Final plots saved to: {self.plots_dir}")
    
    def get_training_summary(self):
        """Get a summary of the training session."""
        if not self.is_training and self.training_start_time:
            training_time = time.time() - self.training_start_time
        else:
            training_time = 0
        
        summary = {
            'algorithm': self.config.get('model.algorithm', 'Unknown'),
            'total_timesteps': self.config.get('training.total_timesteps', 0),
            'training_time': training_time,
            'models_dir': str(self.models_dir),
            'plots_dir': str(self.plots_dir),
            'tensorboard_log': self.tensorboard_log,
            'custom_network': bool(self.custom_network_config)
        }
        
        return summary


def create_custom_network_config(
    hidden_layers: List[int] = None,
    activation_fn: str = 'ReLU',
    net_arch: List = None
) -> Dict:
    """
    Helper function to create custom network configuration.
    
    Args:
        hidden_layers: List of hidden layer sizes for feature extractor
        activation_fn: Activation function name ('ReLU', 'Tanh', 'ELU', etc.)
        net_arch: Network architecture for policy/value networks
        
    Returns:
        Network configuration dictionary
    """
    if hidden_layers is None:
        hidden_layers = [512, 256, 128]
    
    if net_arch is None:
        net_arch = [256, 256]
    
    return {
        'hidden_layers': hidden_layers,
        'activation_fn': activation_fn,
        'net_arch': net_arch,
        'features_dim': hidden_layers[-1] if hidden_layers else 256
    }


# Example usage configurations
NETWORK_CONFIGS = {
    'small': create_custom_network_config([128, 64], 'ReLU', [64, 64]),
    'medium': create_custom_network_config([256, 128], 'ReLU', [128, 128]),
    'large': create_custom_network_config([512, 256, 128], 'ReLU', [256, 256]),
    'deep': create_custom_network_config([512, 512, 256, 256, 128], 'ReLU', [256, 256]),
    'wide': create_custom_network_config([1024, 512, 256], 'ReLU', [512, 512])
}
