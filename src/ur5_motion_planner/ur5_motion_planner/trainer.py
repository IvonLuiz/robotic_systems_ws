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
import seaborn as sns
import pandas as pd
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
                nn.Dropout(0.1),
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
    Professional-grade callback for monitoring training progress and creating publication-quality plots.
    """
    
    def __init__(self, save_path: str, plot_freq: int = 50, verbose: int = 0, 
                 algorithm_name: str = 'RL', smooth_window: int = 10):
        super().__init__(verbose)
        self.save_path = save_path
        self.plot_freq = plot_freq
        self.algorithm_name = algorithm_name
        self.smooth_window = smooth_window
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps = []
        self.episode_numbers = []
        self.training_losses = []
        
        # Current episode tracking
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.episode_count = 0
        
        # DataFrame for professional plotting
        self.training_data = pd.DataFrame()
        
    def _on_step(self) -> bool:
        # Track episode progress
        reward = self.locals.get('rewards', [0])[0] if self.locals.get('rewards') is not None else 0
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        # Check if episode is done
        done = self.locals.get('dones', [False])[0] if self.locals.get('dones') is not None else False
        if done:
            self.episode_count += 1
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.timesteps.append(self.num_timesteps)
            self.episode_numbers.append(self.episode_count)
            
            # Update DataFrame
            new_row = pd.DataFrame({
                'Episode': [self.episode_count],
                'Timestep': [self.num_timesteps],
                'EpisodeReward': [self.current_episode_reward],
                'EpisodeLength': [self.current_episode_length],
                'Algorithm': [self.algorithm_name],
                'Unit': [0],  # For compatibility with multi-run experiments
                'Condition': [self.algorithm_name]
            })
            self.training_data = pd.concat([self.training_data, new_row], ignore_index=True)
            
            # Reset for next episode
            self.current_episode_reward = 0
            self.current_episode_length = 0
            
            # Create plots periodically
            if self.episode_count % self.plot_freq == 0 and self.episode_count > 0:
                self._create_professional_plots()
        
        return True
    
    def _smooth_data(self, data, window_size):
        """Apply moving average smoothing to data."""
        if len(data) < window_size:
            return data
        
        smoothed = []
        for i in range(len(data)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(data), i + window_size // 2 + 1)
            smoothed.append(np.mean(data[start_idx:end_idx]))
        return smoothed
    
    def _create_professional_plots(self):
        """Create publication-quality training plots using seaborn."""
        if len(self.episode_rewards) < 2:
            return
        
        # Set the aesthetic style
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.2)
        
        fig = plt.figure(figsize=(16, 12))
        colors = sns.color_palette("husl", 3)
        
        # Plot 1: Episode Rewards with Smoothed Trend
        ax1 = plt.subplot(2, 3, 1)
        episodes = np.arange(1, len(self.episode_rewards) + 1)
        
        # Raw rewards (lighter)
        ax1.plot(episodes, self.episode_rewards, alpha=0.3, color=colors[0], linewidth=0.8, label='Raw')
        
        # Smoothed rewards (darker)
        smoothed_rewards = self._smooth_data(self.episode_rewards, self.smooth_window)
        ax1.plot(episodes, smoothed_rewards, color=colors[0], linewidth=2.5, 
                label=f'Smoothed (window={self.smooth_window})')
        
        ax1.set_xlabel('Episode', fontweight='bold')
        ax1.set_ylabel('Episode Reward', fontweight='bold')
        ax1.set_title(f'{self.algorithm_name} Training Progress', fontweight='bold', fontsize=14)
        ax1.legend(frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Reward Distribution
        ax2 = plt.subplot(2, 3, 2)
        sns.histplot(self.episode_rewards, bins=30, kde=True, color=colors[0], alpha=0.7, ax=ax2)
        ax2.axvline(np.mean(self.episode_rewards), color=colors[1], linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(self.episode_rewards):.2f}')
        ax2.set_xlabel('Episode Reward', fontweight='bold')
        ax2.set_ylabel('Frequency', fontweight='bold')
        ax2.set_title('Reward Distribution', fontweight='bold', fontsize=14)
        ax2.legend(frameon=True, fancybox=True, shadow=True)
        
        # Plot 3: Episode Lengths
        ax3 = plt.subplot(2, 3, 3)
        smoothed_lengths = self._smooth_data(self.episode_lengths, self.smooth_window)
        ax3.plot(episodes, self.episode_lengths, alpha=0.3, color=colors[2], linewidth=0.8)
        ax3.plot(episodes, smoothed_lengths, color=colors[2], linewidth=2.5)
        ax3.set_xlabel('Episode', fontweight='bold')
        ax3.set_ylabel('Episode Length', fontweight='bold')
        ax3.set_title('Episode Length Progress', fontweight='bold', fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Cumulative Reward vs Timesteps
        ax4 = plt.subplot(2, 3, 4)
        cumulative_rewards = np.cumsum(self.episode_rewards)
        ax4.plot(self.timesteps, cumulative_rewards, color=colors[1], linewidth=2.5)
        ax4.set_xlabel('Timesteps', fontweight='bold')
        ax4.set_ylabel('Cumulative Reward', fontweight='bold')
        ax4.set_title('Cumulative Performance', fontweight='bold', fontsize=14)
        ax4.grid(True, alpha=0.3)
        
        # Format x-axis in scientific notation if needed
        if max(self.timesteps) > 5000:
            ax4.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        
        # Plot 5: Rolling Statistics
        ax5 = plt.subplot(2, 3, 5)
        window = min(50, len(self.episode_rewards) // 4) if len(self.episode_rewards) > 4 else len(self.episode_rewards)
        if window > 1:
            rolling_mean = pd.Series(self.episode_rewards).rolling(window=window, min_periods=1).mean()
            rolling_std = pd.Series(self.episode_rewards).rolling(window=window, min_periods=1).std()
            
            ax5.plot(episodes, rolling_mean, color=colors[0], linewidth=2.5, label=f'Rolling Mean (w={window})')
            ax5.fill_between(episodes, rolling_mean - rolling_std, rolling_mean + rolling_std, 
                           alpha=0.2, color=colors[0], label='±1 Std')
            
            ax5.set_xlabel('Episode', fontweight='bold')
            ax5.set_ylabel('Reward', fontweight='bold')
            ax5.set_title('Rolling Statistics', fontweight='bold', fontsize=14)
            ax5.legend(frameon=True, fancybox=True, shadow=True)
            ax5.grid(True, alpha=0.3)
        
        # Plot 6: Performance Metrics Summary
        ax6 = plt.subplot(2, 3, 6)
        
        # Calculate key metrics
        recent_rewards = self.episode_rewards[-min(50, len(self.episode_rewards)):]
        metrics = {
            'Mean Reward': np.mean(self.episode_rewards),
            'Recent Mean\n(last 50)': np.mean(recent_rewards),
            'Best Reward': np.max(self.episode_rewards),
            'Std Deviation': np.std(self.episode_rewards),
        }
        
        # Create bar plot
        bars = ax6.bar(range(len(metrics)), list(metrics.values()), 
                      color=colors[:len(metrics)], alpha=0.8)
        ax6.set_xticks(range(len(metrics)))
        ax6.set_xticklabels(list(metrics.keys()), rotation=45, ha='right')
        ax6.set_ylabel('Value', fontweight='bold')
        ax6.set_title('Performance Summary', fontweight='bold', fontsize=14)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics.values()):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout(pad=2.0)
        
        # Save the plot
        plot_path = os.path.join(self.save_path, f'training_analysis_ep{self.episode_count}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Save data for external plotting tools
        self._save_training_data()
        
        print(f"📊 Training plots saved: {plot_path}")
    
    def _save_training_data(self):
        """Save training data in multiple formats for analysis."""
        # Save as CSV (compatible with the professional plotting tools)
        csv_path = os.path.join(self.save_path, 'training_progress.csv')
        self.training_data.to_csv(csv_path, index=False)
        
        # Save detailed metrics
        metrics_path = os.path.join(self.save_path, 'detailed_metrics.csv')
        detailed_df = pd.DataFrame({
            'Episode': self.episode_numbers,
            'Timestep': self.timesteps,
            'EpisodeReward': self.episode_rewards,
            'EpisodeLength': self.episode_lengths,
            'CumulativeReward': np.cumsum(self.episode_rewards),
            'AverageReward': np.cumsum(self.episode_rewards) / np.arange(1, len(self.episode_rewards) + 1),
            'Algorithm': [self.algorithm_name] * len(self.episode_rewards)
        })
        detailed_df.to_csv(metrics_path, index=False)
        
        # Save summary statistics
        summary_path = os.path.join(self.save_path, 'training_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Training Summary for {self.algorithm_name}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total Episodes: {len(self.episode_rewards)}\n")
            f.write(f"Total Timesteps: {self.timesteps[-1] if self.timesteps else 0}\n")
            f.write(f"Mean Episode Reward: {np.mean(self.episode_rewards):.3f} ± {np.std(self.episode_rewards):.3f}\n")
            f.write(f"Best Episode Reward: {np.max(self.episode_rewards):.3f}\n")
            f.write(f"Worst Episode Reward: {np.min(self.episode_rewards):.3f}\n")
            f.write(f"Mean Episode Length: {np.mean(self.episode_lengths):.1f} ± {np.std(self.episode_lengths):.1f}\n")
            
            # Recent performance (last 25% of episodes)
            recent_start = max(0, len(self.episode_rewards) - len(self.episode_rewards) // 4)
            recent_rewards = self.episode_rewards[recent_start:]
            if recent_rewards:
                f.write(f"\nRecent Performance (last {len(recent_rewards)} episodes):\n")
                f.write(f"Mean Reward: {np.mean(recent_rewards):.3f} ± {np.std(recent_rewards):.3f}\n")
                f.write(f"Improvement: {np.mean(recent_rewards) - np.mean(self.episode_rewards[:len(self.episode_rewards)//4]) if len(self.episode_rewards) > 4 else 0:.3f}\n")
    
    def create_final_report(self):
        """Create a comprehensive final training report."""
        if not self.episode_rewards:
            return
            
        # Create final comprehensive plot
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.4)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        colors = sns.color_palette("Set2", 4)
        
        # Main performance curve
        episodes = np.arange(1, len(self.episode_rewards) + 1)
        smoothed_rewards = self._smooth_data(self.episode_rewards, self.smooth_window)
        
        axes[0, 0].plot(episodes, self.episode_rewards, alpha=0.3, color=colors[0], linewidth=1)
        axes[0, 0].plot(episodes, smoothed_rewards, color=colors[0], linewidth=3, 
                       label=f'{self.algorithm_name} Performance')
        axes[0, 0].set_xlabel('Training Episode', fontweight='bold')
        axes[0, 0].set_ylabel('Episode Reward', fontweight='bold')
        axes[0, 0].set_title('Learning Curve', fontweight='bold', fontsize=16)
        axes[0, 0].legend(frameon=True, fancybox=True, shadow=True)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Performance vs Timesteps (for sample efficiency analysis)
        axes[0, 1].plot(self.timesteps, smoothed_rewards, color=colors[1], linewidth=3)
        axes[0, 1].set_xlabel('Environment Timesteps', fontweight='bold')
        axes[0, 1].set_ylabel('Episode Reward', fontweight='bold')
        axes[0, 1].set_title('Sample Efficiency', fontweight='bold', fontsize=16)
        axes[0, 1].grid(True, alpha=0.3)
        if max(self.timesteps) > 5000:
            axes[0, 1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        
        # Episode length analysis
        smoothed_lengths = self._smooth_data(self.episode_lengths, self.smooth_window)
        axes[1, 0].plot(episodes, self.episode_lengths, alpha=0.3, color=colors[2], linewidth=1)
        axes[1, 0].plot(episodes, smoothed_lengths, color=colors[2], linewidth=3)
        axes[1, 0].set_xlabel('Training Episode', fontweight='bold')
        axes[1, 0].set_ylabel('Episode Length', fontweight='bold')
        axes[1, 0].set_title('Episode Duration', fontweight='bold', fontsize=16)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Final performance distribution
        sns.histplot(self.episode_rewards, bins=30, kde=True, color=colors[3], alpha=0.7, ax=axes[1, 1])
        axes[1, 1].axvline(np.mean(self.episode_rewards), color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {np.mean(self.episode_rewards):.2f}')
        axes[1, 1].set_xlabel('Episode Reward', fontweight='bold')
        axes[1, 1].set_ylabel('Frequency', fontweight='bold')
        axes[1, 1].set_title('Final Performance Distribution', fontweight='bold', fontsize=16)
        axes[1, 1].legend(frameon=True, fancybox=True, shadow=True)
        
        plt.suptitle(f'{self.algorithm_name} Training Report', fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout(pad=3.0)
        
        # Save final report
        final_path = os.path.join(self.save_path, 'final_training_report.png')
        plt.savefig(final_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📈 Final training report saved: {final_path}")
        
        return final_path


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
        
        # Setup GPU/CPU device
        self.setup_device()
        
        # Setup paths
        self.setup_paths()
        
        # Training state
        self.model = None
        self.training_thread = None
        self.is_training = False
        self.training_start_time = None
        
        # Callbacks
        self.callbacks = []
        
    def setup_device(self):
        """Setup GPU/CPU device for training."""
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🚀 GPU detected: {gpu_name}")
            print(f"💾 GPU memory: {gpu_memory:.1f} GB")
            print(f"🔥 Using GPU for training: {self.device}")
        else:
            self.device = torch.device("cpu")
            print("⚠️ GPU not available, using CPU for training")
            print("💡 To use GPU, ensure CUDA-compatible PyTorch is installed")
        
        # Set device for stable-baselines3
        # Note: SB3 will automatically use GPU if torch.cuda.is_available()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
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
    
    def setup_callbacks(self, algorithm_name: str = 'RL'):
        """Setup training callbacks for monitoring and checkpointing."""
        # Training progress callback with professional plotting
        progress_callback = TrainingCallback(
            save_path=str(self.plots_dir),
            plot_freq=self.config.get('training.plot_frequency', 50),
            algorithm_name=algorithm_name,
            smooth_window=self.config.get('training.smooth_window', 10)
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
        print(f"Using custom policy: {policy.__name__ if isinstance(policy, type) else policy}")
        
        # Add tensorboard logging
        model_kwargs['tensorboard_log'] = self.tensorboard_log
        
        # Add device configuration for GPU usage
        model_kwargs['device'] = self.device
        
        # Create model
        self.model = AlgorithmClass(
            policy,
            self.env,
            **model_kwargs
        )
        
        print(f"Created {algorithm_name} model with custom architecture")
        print(f"Model device: {self.device}")
        if self.custom_network_config:
            print(f"Network config: {self.custom_network_config}")
        
        return self.model
    
    def check_gpu_usage(self):
        """Check and display current GPU usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            cached = torch.cuda.memory_reserved(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"🔥 GPU Memory Usage:")
            print(f"   Allocated: {allocated:.2f} GB")
            print(f"   Cached: {cached:.2f} GB")
            print(f"   Total: {total:.2f} GB")
            print(f"   Utilization: {(allocated/total)*100:.1f}%")
            return {
                'allocated_gb': allocated,
                'cached_gb': cached,
                'total_gb': total,
                'utilization_percent': (allocated/total)*100
            }
        else:
            print("❌ GPU not available")
            return None
    
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
        
        algorithm_name = self.config.get('model.algorithm', 'SAC')
        self.setup_callbacks(algorithm_name)
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
                    self.last_gpu_check = 0
                    self.gpu_check_freq = save_freq  # Check GPU usage when saving
                
                def _on_step(self):
                    if self.num_timesteps - self.last_save >= self.save_freq:
                        save_path = self.trainer.models_dir / f"model_{self.num_timesteps}.zip"
                        self.trainer.model.save(save_path)
                        self.trainer.save_model()  # Also save as latest
                        self.last_save = self.num_timesteps
                        print(f"💾 Auto-saved model at timestep {self.num_timesteps}")
                        
                        # Check GPU usage when saving
                        if torch.cuda.is_available():
                            gpu_info = self.trainer.check_gpu_usage()
                            if gpu_info and gpu_info['utilization_percent'] > 90:
                                print("⚠️ High GPU memory usage detected!")
                    
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
            print(f"✅ Training completed in {training_time:.2f} seconds")
            
        except KeyboardInterrupt:
            print("⚠️ Training interrupted by user")
            self.save_model()
            
        except Exception as e:
            print(f"❌ Training error: {e}")
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
            # Create final comprehensive report
            final_report_path = training_callback.create_final_report()
            print(f"📈 Final comprehensive plots saved to: {self.plots_dir}")
            return final_report_path
        else:
            print("No training callback found or no episode data available")
            return None
    
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
            'device': str(self.device),
            'gpu_available': torch.cuda.is_available(),
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


def check_gpu_setup():
    """
    Utility function to check GPU setup for RL training.
    Call this before training to verify everything is configured correctly.
    """
    print("="*60)
    print("🔍 GPU SETUP VERIFICATION")
    print("="*60)
    
    # Check PyTorch installation
    print(f"🐍 PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print("✅ CUDA is available!")
        print(f"🔥 CUDA version: {torch.version.cuda}")
        
        # GPU details
        gpu_count = torch.cuda.device_count()
        print(f"🖥️ Number of GPUs: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Test GPU access
        try:
            device = torch.device("cuda:0")
            test_tensor = torch.randn(1000, 1000).to(device)
            result = torch.mm(test_tensor, test_tensor.T)
            print("✅ GPU computation test passed!")
            
            # Memory info
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            cached = torch.cuda.memory_reserved(0) / 1024**3
            print(f"📊 Current GPU memory: {allocated:.2f} GB allocated, {cached:.2f} GB cached")
            
        except Exception as e:
            print(f"❌ GPU test failed: {e}")
            
    else:
        print("❌ CUDA is not available")
        print("💡 Possible solutions:")
        print("   - Install CUDA-enabled PyTorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("   - Check NVIDIA drivers: nvidia-smi")
        print("   - Verify CUDA installation")
    
    # Check if Stable Baselines3 will use GPU
    print("\n🤖 Stable Baselines3 GPU Support:")
    if torch.cuda.is_available():
        print("✅ SB3 will automatically use GPU for neural network training")
        print("🚀 Models will be trained on GPU for faster performance")
    else:
        print("⚠️ SB3 will use CPU (slower training)")
    
    print("="*60)
    
    return torch.cuda.is_available()


if __name__ == "__main__":
    # Quick GPU check when running this file directly
    check_gpu_setup()
