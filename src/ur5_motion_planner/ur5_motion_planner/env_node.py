import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import gymnasium as gym
from gymnasium import spaces

import numpy as np
import threading
import time
import os

from geometry_msgs.msg import Pose
from visualization_msgs.msg import Marker

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from sensor_msgs.msg import JointState

from stable_baselines3 import SAC, TD3, DDPG, PPO, A2C
from stable_baselines3.common.env_checker import check_env

from .config_loader import RLConfig


class UR5Env(gym.Env, Node):
    """
    Reinforcement Learning Environment for a UR5 robot in Gazebo.
    Inherits from both gymnasium.Env and rclpy.node.Node.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, config_file="rl_config.yaml"):
        Node.__init__(self, 'ur5_rl_environment')
        self.callback_group = ReentrantCallbackGroup()
        gym.Env.__init__(self)

        # Load configuration
        self.config = RLConfig(config_file)

        # --- Gym Environment Setup ---
        self.joint_limits_lower = np.array(self.config.get('joint_limits.lower', [-np.pi] * 6))
        self.joint_limits_upper = np.array(self.config.get('joint_limits.upper', [np.pi] * 6))
        target_vector_low = np.array([-2.0, -2.0, -2.0])
        target_vector_high = np.array([2.0, 2.0, 2.0])
        target_position_low = np.array([-1.0, -1.0, -1.0])
        target_position_high = np.array([1.0, 1.0, 1.0])

        # Action: 6 joint velocities [-1.0, 1.0] rad/s scaled by action_scale
        action_scale = self.config.get('environment.action_scale', 1.0)
        action_low = np.array([-action_scale] * 6, dtype=np.float32)
        action_high = np.array([action_scale] * 6, dtype=np.float32)
        self.action_space = spaces.Box(low=action_low, high=action_high)

        # Observation: 6 (joint angles) + 3 (vector to target) + 3 (target position)
        obs_low = np.concatenate([self.joint_limits_lower, target_vector_low, target_position_low])
        obs_high = np.concatenate([self.joint_limits_upper, target_vector_high, target_position_high])
        self.observation_space = spaces.Box(low=obs_low, high=obs_high)

        # --- ROS 2 Communication Setup ---
        self._action_client = ActionClient(
            self, FollowJointTrajectory, 
            '/scaled_joint_trajectory_controller/follow_joint_trajectory',
            callback_group=self.callback_group
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Joint state subscriber - will be created/destroyed on demand
        self.joint_state_sub = None
        self.current_joint_state = None
        self._joint_state_update_needed = False
        self.robot_joints_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]

        # RVIZ visualization setup
        self.target_marker_pub = self.create_publisher(
            Marker, 
            '/target_marker', 
            10,
            callback_group=self.callback_group
        )

        self.target_marker = Marker()
        self.target_marker.header.frame_id = "base_link"
        self.target_marker.type = Marker.SPHERE
        self.target_marker.action = Marker.ADD
        self.target_marker.scale.x = 0.1
        self.target_marker.scale.y = 0.1
        self.target_marker.scale.z = 0.1
        self.target_marker.color.r = 1.0
        self.target_marker.color.g = 0.0
        self.target_marker.color.b = 0.0
        self.target_marker.color.a = 0.8
        self.target_marker.id = 0

        self._wait_services_ready()

        # RL State
        self.episode_step_count = 0
        self.max_steps_per_episode = self.config.get('environment.max_steps_per_episode', 200)
        self.last_target_dist = None
        
        # Get configured parameters
        self.dt = self.config.get('environment.dt', 0.1)
        self.goal_tolerance = self.config.get('environment.goal_tolerance', 0.05)
        self.home_position = self.config.get('home_position', [0, -np.pi/2, 0.01, -np.pi/2, 0.01, 0.01])
        
        # Reward parameters
        self.reward_threshold = self.config.get('reward.reward_threshold', 0.35)
        self.distance_penalty_scale = self.config.get('reward.distance_penalty_scale', 2.0)
        self.distance_reward_steepness = self.config.get('reward.distance_reward_steepness', 0.5)
        self.closer_reward_scale = self.config.get('reward.closer_reward_scale', 20.0)
        self.goal_bonus = self.config.get('reward.goal_bonus', 100.0)

    def _wait_services_ready(self):
        # Wait for essential connections
        self.get_logger().info("Waiting for Action Server...")
        self._action_client.wait_for_server()
        self.get_logger().info("Action Server is ready.")
        
        # Wait for initial joint states with timeout
        self.get_logger().info("Waiting for initial joint states...")
        timeout = self.config.get('ros.joint_state_timeout', 10.0)
        start_time = time.time()
        
        # Enable joint state updates temporarily to get initial state
        self._enable_joint_state_updates()
        
        while self.current_joint_state is None and rclpy.ok():
            # Allow ROS callbacks to be processed
            rclpy.spin_once(self, timeout_sec=0.1)
            
            # Check timeout
            if time.time() - start_time > timeout:
                self.get_logger().warn("Timeout waiting for initial joint states. Continuing anyway...")
                break
                
        if self.current_joint_state is not None:
            self.get_logger().info("Received initial joint states.")
        else:
            self.get_logger().warn("Joint states not received during initialization.")
        
        # Disable continuous updates after getting initial state
        self._disable_joint_state_updates()
        
    def _publish_target_marker(self):
        """Publishes the target marker in RViz."""
        self.target_marker.header.stamp = self.get_clock().now().to_msg()
        self.target_marker.pose.position.x = float(self.target_position[0])
        self.target_marker.pose.position.y = float(self.target_position[1])
        self.target_marker.pose.position.z = float(self.target_position[2])
        self.target_marker.pose.orientation.w = 1.0
        self.target_marker.pose.orientation.x = 0.0
        self.target_marker.pose.orientation.y = 0.0
        self.target_marker.pose.orientation.z = 0.0
        
        self.target_marker_pub.publish(self.target_marker)
        self.get_logger().debug(f"Published target marker at position: {self.target_position}")

    def _update_target_marker(self):
        """Updates the target marker position in RViz."""
        self._publish_target_marker()

    def joint_state_callback(self, msg):
        # Only update if we're expecting joint state updates
        if not self._joint_state_update_needed:
            return
            
        # Reorder the joints to match the order we expect
        joint_positions = [0.0] * len(self.robot_joints_names)
        for i, name in enumerate(self.robot_joints_names):
            if name in msg.name:
                idx = msg.name.index(name)
                joint_positions[i] = msg.position[idx]
        self.current_joint_state = np.array(joint_positions)
        self._joint_state_update_needed = False  # We got what we needed
    
    def _enable_joint_state_updates(self):
        """Enable joint state updates by creating subscriber if needed."""
        if self.joint_state_sub is None:
            self.joint_state_sub = self.create_subscription(
                JointState,
                '/joint_states',
                self.joint_state_callback,
                10,
                callback_group=self.callback_group)
        self._joint_state_update_needed = True
    
    def _disable_joint_state_updates(self):
        """Disable joint state updates by destroying subscriber."""
        if self.joint_state_sub is not None:
            self.destroy_subscription(self.joint_state_sub)
            self.joint_state_sub = None
        self._joint_state_update_needed = False

    def step(self, action):
        self.episode_step_count += 1
        
        # 1. Get current joint state on demand
        current_angles = self._get_current_joint_state()
        
        if current_angles is None:
            return self.observation_space.sample(), 0, False, True, {"error": "Joint states not available"}

        # Integrate velocity over a small time step dt
        dt = self.dt
        new_target_angles = current_angles + action * dt

        # Clip to joint limits
        if np.any(new_target_angles < self.joint_limits_lower) or np.any(new_target_angles > self.joint_limits_upper):
            self.get_logger().warn(f"New target angles {new_target_angles} exceeded joint limits {self.joint_limits_lower} - {self.joint_limits_upper}", throttle_duration_sec=2.0)
        new_target_angles = np.clip(new_target_angles, self.joint_limits_lower, self.joint_limits_upper)

        self.get_logger().debug(f"Step {self.episode_step_count}")
        self.get_logger().debug(f"Action: {action}")
        self.get_logger().debug(f"Current angles: {current_angles}")
        self.get_logger().debug(f"New target angles: {new_target_angles}")

        # Send the trajectory to the robot and wait for completion
        result = self.send_trajectory_goal(new_target_angles.tolist(), dt)

        # 2. Get New Observation after trajectory completion
        observation = self._get_observation()
        if observation is None:
            # very unlikely to happen, but checking just in case
            reward = 0
            terminated = False
            truncated = True
            return self.observation_space.sample(), reward, terminated, truncated, {"error": "Failed to get observation"}
        if not result:
            # trajectory execution usually fail due to hitting the ground or itself
            reward = -200
            terminated = False
            truncated = True
            self.get_logger().error(f"Penalizing step with {reward} reward and resetting.")
            return observation, reward, terminated, truncated, {"error": f"Failed to send trajectory goal. Penalizing step with {reward} reward and resetting."}

        # 3. Calculate Reward
        reward, terminated = self.calculate_reward()
        
        # 4. Check for Termination
        if terminated:
            self.get_logger().info("Episode terminated: Goal Reached!")
            self.episode_step_count = 0

        # Check for timeout
        truncated = False
        if self.episode_step_count >= self.max_steps_per_episode:
            truncated = True
            self.get_logger().info("Episode timed out.")

        return observation, reward, terminated, truncated, {}

    def calculate_reward(self):
        '''
        Calculates a reward using a smooth exponential function.
        
        The total reward is shaped by:
        1. Exponential Distance Reward: A continuous function that provides a
        strong negative penalty for being far from the target and a strong
        positive reward for being closer than a defined threshold.
        2. Improvement Reward: Rewards the agent for reducing the distance to
        the target compared to the previous step.
        3. Step Penalty: A small penalty for each step taken to encourage
        efficiency.
        4. Goal Bonus: A large reward for reaching the target.
        '''
        reward = 0
        improvement = 0
        terminated = False
        curr_target_dist = self._calculate_target_dist()

        # 1)
        # reward = scale * (e^(steepness * (threshold - distance)) - 1)
        # `self.reward_threshold`: negative when farther, and positive when closer.
        #distance_reward = self.distance_penalty_scale * \
        #    (np.exp(self.distance_reward_steepness * (self.reward_threshold - curr_target_dist)) - 1)
        if curr_target_dist is not None:
            distance_reward = -self.distance_penalty_scale * np.power(curr_target_dist, 2)  # Using squared distance for a smoother penalty
        else:
            self.get_logger().warn("Current target distance is None, setting distance_reward to 0.")
            distance_reward = 0
        reward += distance_reward

        # 2)
        #if self.last_target_dist is not None:
        #    # this value is positive when getting closer, negative otherwise
        #    improvement = self.last_target_dist - curr_target_dist
        #    reward += self.closer_reward_scale * improvement
        #self.last_target_dist = curr_target_dist

        # 3)
        #if self.episode_step_count > 0:
        #    # penalize for taking too many steps to encourage efficiency
        #    reward -= 0.001 * self.episode_step_count

        # 4)
        if curr_target_dist < self.goal_tolerance:
            reward += self.goal_bonus
            terminated = True

        self.get_logger().info(
            f"Dist: {curr_target_dist:.3f}, "
            f"TotalReward: {reward:.3f}",
            throttle_duration_sec=1
        )

        return reward, terminated

    def _calculate_target_dist(self):
        '''
        Calculate the Euclidean distance between two points.
        '''
        end_effector_pos = self.get_end_effector_pose()[:3]
        return np.linalg.norm(end_effector_pos - self.target_position)

    def reset(self, seed=None, options=None):
        # Check if ROS is still running
        if not rclpy.ok():
            self.get_logger().warn("ROS is shutting down, returning default observation")
            default_obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            return default_obs, {}
            
        super().reset(seed=seed)
        self.episode_step_count = 0
        self.last_target_dist = None

        # Reset robot to a home position
        home_joint_angles = self.home_position
        result = self.send_trajectory_goal(home_joint_angles) # returns True if successful

        if not result:
            self.get_logger().error("Failed to send home joint angles on reset. Trying again...")
            return self.reset()

        # Generate a new random target
        self.target_position = self._generate_random_target()
        self._update_target_marker()
        self.get_logger().info(f"New target generated at: {self.target_position}")

        initial_observation = self._get_observation()
        if initial_observation is None:
            self.get_logger().error("Failed to get initial observation during reset. Retrying...")
            time.sleep(0.5)
            return self.reset(seed=seed, options=options)

        return initial_observation, {}

    def _get_observation(self):
        joint_angles = self._get_current_joint_state()
        ee_transform = self.get_end_effector_pose()
        target_position = self.target_position
        self.get_logger().debug("Getting observation...")

        if joint_angles is None or ee_transform is None or target_position is None:
            # this is very unlikely, but just in case
            self.get_logger().warn("Could not get complete observation.")
            return None

        self.get_logger().debug(f"Joint Angles: {joint_angles}, End Effector Pose: {ee_transform}")

        ee_pos = ee_transform[:3]
        vector_to_target = target_position - ee_pos

        return np.concatenate([joint_angles, vector_to_target, target_position]).astype(np.float32)

    def _generate_random_target(self):
        """
        Generate a reachable (x, y, z) position within the workspace.
        
        Supports two sampling methods:
        1. Cartesian: Direct sampling within x,y,z bounds (default)
        2. Spherical: Sampling in spherical coordinates (useful for even distribution around robot)
        """
        bounds = self.config.get('environment.workspace_bounds', {})
        sampling_method = self.config.get('environment.target_sampling_method', 'cartesian')
        
        if sampling_method == 'spherical':
            # Spherical coordinate sampling - useful for even distribution around robot base
            radius_min = bounds.get('radius_min', 0.3)  # minimum reach distance
            radius_max = bounds.get('radius_max', 0.8)   # maximum reach distance
            theta_min = bounds.get('theta_min', -np.pi)   # azimuthal angle range
            theta_max = bounds.get('theta_max', np.pi)
            phi_min = bounds.get('phi_min', 0)            # polar angle range (0 = up)
            phi_max = bounds.get('phi_max', np.pi/2)      # pi/2 = horizontal plane
            
            radius = np.random.uniform(radius_min, radius_max)
            theta = np.random.uniform(theta_min, theta_max)  # azimuthal angle
            phi = np.random.uniform(phi_min, phi_max)        # polar angle
            
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(phi)
            
            # limit the z coordinate to avoid unreachable poses
            z_offset = bounds.get('z_offset', 0.0)
            z = min(z, z_offset)

        else:
            # Default: Cartesian coordinate sampling within defined workspace bounds
            x = np.random.uniform(bounds.get('x_min', -0.21), bounds.get('x_max', 0.2))
            y = np.random.uniform(bounds.get('y_min', -0.21), bounds.get('y_max', 0.2))
            z = np.random.uniform(bounds.get('z_min', 0.9), bounds.get('z_max', 0.91))
        
        target = np.array([x, y, z])
        
        # Optional: Validate that target is within UR5 reachable workspace
        # UR5 has approximately 0.85m reach from base
        distance_from_base = np.linalg.norm(target[:2])  # distance in x-y plane
        max_reach = bounds.get('max_reach_validation', 0.85)
        
        if distance_from_base > max_reach:
            self.get_logger().debug(f"Generated target at distance {distance_from_base:.3f}m may be unreachable (max: {max_reach}m)")
        
        return target

    def get_joint_angles(self):
        if self.current_joint_state is None:
            self.get_logger().warn("Joint states have not been received yet.")
            return None
        return self.current_joint_state
    
    def _get_current_joint_state(self):
        """Get current joint state by temporarily enabling updates."""
        self._enable_joint_state_updates()
        
        # Wait for a fresh joint state update
        timeout = 2.0  # Short timeout for getting joint states
        start_time = time.time()
        old_state = self.current_joint_state.copy() if self.current_joint_state is not None else None
        
        while rclpy.ok() and time.time() - start_time < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            
            # Check if we got a new state (or first state)
            if self.current_joint_state is not None:
                if old_state is None or not np.array_equal(self.current_joint_state, old_state):
                    break
        
        self._disable_joint_state_updates()
        return self.current_joint_state

    def get_end_effector_pose(self):
        try:
            # this transformation assumes the end effector is named 'tool0', 
            # and gets the position and orientation relative to the base_link
            transform = self.tf_buffer.lookup_transform('base_link', 'tool0', rclpy.time.Time())
            pos = transform.transform.translation
            rot = transform.transform.rotation
            return np.array([pos.x, pos.y, pos.z, rot.x, rot.y, rot.z, rot.w])
        except TransformException as ex:
            self.get_logger().warn(f'Could not get transform: {ex}')
            return None

    def send_trajectory_goal(self, joint_angles, dt=None):
        # Check if ROS is still running
        if not rclpy.ok():
            self.get_logger().warn("ROS is shutting down, cannot send trajectory goal")
            return False
            
        if dt is None:
            dt = self.config.get('ros.trajectory_execution_time', 1.0)
            
        goal_msg = FollowJointTrajectory.Goal()
        trajectory_point = JointTrajectoryPoint()
        trajectory_point.positions = joint_angles
        trajectory_point.time_from_start = Duration(sec=int(dt), nanosec=int((dt % 1) * 1e9))
        
        goal_msg.trajectory.joint_names = self.robot_joints_names
        goal_msg.trajectory.points.append(trajectory_point)
        
        timeout = self.config.get('ros.action_server_timeout', 20.0)
        
        # Send goal
        send_goal_future = self._action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future, timeout_sec=timeout)
        
        if not send_goal_future.done():
            self.get_logger().error("Failed to send goal - timeout")
            return False

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal was rejected")
            return False

        # Wait for trajectory completion
        self.get_logger().debug("Waiting for trajectory execution")
        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, get_result_future, timeout_sec=timeout)

        if not get_result_future.done():
            self.get_logger().error("Failed to get result - timeout")
            return False

        result = get_result_future.result().result
        success = result.error_code == FollowJointTrajectory.Result.SUCCESSFUL

        return success

    def close(self):
        """Cleanup ROS 2 resources."""
        self.get_logger().info("Cleaning up environment.")
        self.destroy_node()


def main():
    rclpy.init()
    config = RLConfig()
    env = UR5Env()
    
    executor = MultiThreadedExecutor()
    executor.add_node(env)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    run_evaluation = False

    # --- 1. Check the environment ---
    # Uncomment the following lines to enable environment checks
    # It will display warnings if something is wrong
    #print("Checking environment...")
    #check_env(env)
    #print("Environment check passed!")

    # --- 2. Import and setup the trainer ---
    from .trainer import UR5Trainer, NETWORK_CONFIGS
    
    # Choose network configuration
    network_type = config.get('model.network_type', 'medium')  # small, medium, large, deep, wide
    algorithm_name = config.get('model.algorithm', 'SAC')
    custom_network_config = NETWORK_CONFIGS.get(network_type, NETWORK_CONFIGS['medium'])
    
    # Create trainer
    trainer = UR5Trainer(env, config, custom_network_config)
    
    # Check if we should load a pre-trained model
    load_pre_trained = config.get('training.load_pretrained_model', False)
    if load_pre_trained:
        pre_trained_model_path = config.get('training.pretrained_model_path', None)
        if pre_trained_model_path and os.path.exists(pre_trained_model_path):
            print(f"Loading pre-trained model from: {pre_trained_model_path}")
            trainer.load_model(pre_trained_model_path)
        else:
            print("Pre-trained model path not found, creating new model")
            trainer.create_model(algorithm_name)
    else:
        print("Creating new model")
        trainer.create_model(algorithm_name)

    # --- 3. Start Training ---
    print("="*60)
    print("STARTING UR5 RL TRAINING")
    print("="*60)
    
    algorithm_name = config.get('model.algorithm', 'SAC')
    total_timesteps = config.get('training.total_timesteps', 100000)
    save_frequency = config.get('training.auto_save_frequency', 10000)
    
    print(f"Algorithm: {algorithm_name}")
    print(f"Network Type: {network_type}")
    print(f"Network Config: {custom_network_config}")
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"Auto-save Frequency: {save_frequency:,}")
    print("-" * 60)
    
    try:
        # Train the model with automatic saving and monitoring
        trainer.train(total_timesteps=total_timesteps, save_frequency=save_frequency)
        trainer.create_final_plots()
        run_evaluation = True

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        trainer.save_model()
        print("Model saved before exit")
        run_evaluation = False  # Explicitly disable evaluation
        
        # Immediately shutdown ROS to prevent further operations
        print("Shutting down ROS...")
        rclpy.shutdown()
        thread.join()
        print("Cleanup complete.")
        return  # Exit immediately
        
    except Exception as e:
        print(f"\nTraining error: {e}")
        trainer.save_model()
        print("Model saved despite error")
        run_evaluation = False  # Disable evaluation on error
        raise

    if run_evaluation:
        # --- 4. Evaluation ---
        eval_episodes = config.get('training.eval_episodes', 10)
        print(f"\nStarting evaluation with {eval_episodes} episodes...")
        
        eval_results = trainer.evaluate(n_episodes=eval_episodes)
        
        # --- 5. Training Summary ---
        summary = trainer.get_training_summary()
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Algorithm: {summary['algorithm']}")
        print(f"Total Timesteps: {summary['total_timesteps']:,}")
        print(f"Training Time: {summary['training_time']:.2f} seconds")
        print(f"Custom Network: {summary['custom_network']}")
        print(f"Models saved to: {summary['models_dir']}")
        print(f"Plots saved to: {summary['plots_dir']}")
        print(f"Tensorboard logs: {summary['tensorboard_log']}")
        print(f"Final evaluation reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        print("="*60)

    rclpy.shutdown()
    thread.join()
    return

if __name__ == '__main__':
    main()
