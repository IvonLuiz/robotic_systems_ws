import yaml
import os
from ament_index_python.packages import get_package_share_directory


class RLConfig:
    """Configuration loader for RL training parameters."""
    
    def __init__(self, config_file="rl_config.yaml"):
        """
        Load configuration from YAML file.
        
        Args:
            config_file: Name of the config file in the config directory
        """
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            # Try to get the installed package config first
            package_share_directory = get_package_share_directory('ur5_motion_planner')
            config_path = os.path.join(package_share_directory, 'config', self.config_file)
            
            if not os.path.exists(config_path):
                # Fallback to local config directory (for development)
                current_dir = os.path.dirname(os.path.abspath(__file__))
                package_root = os.path.dirname(current_dir)
                config_path = os.path.join(package_root, 'config', self.config_file)
            
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
                
        except Exception as e:
            print(f"Error loading config file: {e}")
            return self._get_default_config()
    
    def _get_default_config(self):
        """Return default configuration if file loading fails."""
        return {
            'environment': {
                'max_steps_per_episode': 200,
                'workspace_bounds': {'x_min': -0.21, 'x_max': 0.2, 'y_min': -0.21, 'y_max': 0.2, 'z_min': 0.9, 'z_max': 0.91},
                'goal_tolerance': 0.05,
                'action_scale': 1.0,
                'dt': 0.1
            },
            'reward': {
                'max_reward': 100.0,
                'distance_penalty_scale': 500.0,
                'goal_bonus': 100.0
            },
            'model': {
                'algorithm': 'SAC',
                'policy': 'MlpPolicy',
                'learning_rate': 0.0003,
                'verbose': 1
            },
            'training': {
                'total_timesteps': 50000,
                'log_interval': 4,
                'eval_episodes': 10,
                'tensorboard_log': './ur5_sac_tensorboard/',
                'model_save_path': 'models/sac_ur5_reacher_model.zip'
            },
            'joint_limits': {
                'lower': [-3.14159] * 6,
                'upper': [3.14159] * 6
            },
            'home_position': [0.0, -1.5708, 0.01, -1.5708, 0.01, 0.01],
            'ros': {
                'action_server_timeout': 20.0,
                'joint_state_timeout': 10.0,
                'trajectory_execution_time': 1.0
            }
        }
    
    def get(self, key_path, default=None):
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: String path like 'environment.max_steps_per_episode'
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_model_kwargs(self):
        """Get model parameters as kwargs for stable-baselines3."""
        model_config = self.config.get('model', {})
        algorithm = model_config.get('algorithm', 'SAC')
        
        # Convert config to stable-baselines3 format
        kwargs = {}
        
        # Basic parameters (common to most algorithms)
        if 'learning_rate' in model_config:
            kwargs['learning_rate'] = model_config['learning_rate']
        if 'verbose' in model_config:
            kwargs['verbose'] = model_config['verbose']
        if 'gamma' in model_config:
            kwargs['gamma'] = model_config['gamma']
        
        # Off-policy algorithms (SAC, TD3, DDPG)
        if algorithm in ['SAC', 'TD3', 'DDPG']:
            if 'buffer_size' in model_config:
                kwargs['buffer_size'] = model_config['buffer_size']
            if 'learning_starts' in model_config:
                kwargs['learning_starts'] = model_config['learning_starts']
            if 'batch_size' in model_config:
                kwargs['batch_size'] = model_config['batch_size']
            if 'tau' in model_config:
                kwargs['tau'] = model_config['tau']
            if 'gradient_steps' in model_config:
                kwargs['gradient_steps'] = model_config['gradient_steps']
        
        # SAC specific parameters
        if algorithm == 'SAC':
            if 'ent_coef' in model_config:
                kwargs['ent_coef'] = model_config['ent_coef']
            if 'target_update_interval' in model_config:
                kwargs['target_update_interval'] = model_config['target_update_interval']
        
        # TD3 specific parameters
        elif algorithm == 'TD3':
            if 'policy_delay' in model_config:
                kwargs['policy_delay'] = model_config['policy_delay']
            if 'target_policy_noise' in model_config:
                kwargs['target_policy_noise'] = model_config['target_policy_noise']
            if 'target_noise_clip' in model_config:
                kwargs['target_noise_clip'] = model_config['target_noise_clip']
        
        # PPO/A2C specific parameters (on-policy)
        elif algorithm in ['PPO', 'A2C']:
            if 'n_steps' in model_config:
                kwargs['n_steps'] = model_config['n_steps']
            if 'ent_coef_ppo' in model_config:
                kwargs['ent_coef'] = model_config['ent_coef_ppo']
            if 'vf_coef' in model_config:
                kwargs['vf_coef'] = model_config['vf_coef']
            if 'max_grad_norm' in model_config:
                kwargs['max_grad_norm'] = model_config['max_grad_norm']
            
            # PPO specific
            if algorithm == 'PPO':
                if 'n_epochs' in model_config:
                    kwargs['n_epochs'] = model_config['n_epochs']
                if 'clip_range' in model_config:
                    kwargs['clip_range'] = model_config['clip_range']
            
        return kwargs
    
    def get_training_kwargs(self):
        """Get training parameters as kwargs."""
        training_config = self.config.get('training', {})
        return {
            'total_timesteps': training_config.get('total_timesteps', 50000),
            'log_interval': training_config.get('log_interval', 4)
        }
