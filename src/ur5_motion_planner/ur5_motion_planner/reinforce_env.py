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

from stable_baselines3 import SAC
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
        # Action: 6 joint velocities [-1.0, 1.0] rad/s scaled by action_scale
        action_scale = self.config.get('environment.action_scale', 1.0)
        action_low = np.array([-action_scale] * 6, dtype=np.float32)
        action_high = np.array([action_scale] * 6, dtype=np.float32)
        self.action_space = spaces.Box(low=action_low, high=action_high)

        # Observation: 6 joint angles + 3 (vector to target)
        obs_low = np.array([-np.inf] * 9, dtype=np.float32)
        obs_high = np.array([np.inf] * 9, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high)

        # --- ROS 2 Communication Setup ---
        self._action_client = ActionClient(
            self, FollowJointTrajectory, 
            '/scaled_joint_trajectory_controller/follow_joint_trajectory',
            callback_group=self.callback_group
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Subscriber to get current joint states
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10,
            callback_group=self.callback_group)
            
        self.current_joint_state = None
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
        
        # Get configured parameters
        self.dt = self.config.get('environment.dt', 0.1)
        self.goal_tolerance = self.config.get('environment.goal_tolerance', 0.05)
        self.joint_limits_lower = np.array(self.config.get('joint_limits.lower', [-np.pi] * 6))
        self.joint_limits_upper = np.array(self.config.get('joint_limits.upper', [np.pi] * 6))
        self.home_position = self.config.get('home_position', [0, -np.pi/2, 0.01, -np.pi/2, 0.01, 0.01])
        
        # Reward parameters
        self.reward_max = self.config.get('reward.max_reward', 100.0)
        self.distance_penalty_scale = self.config.get('reward.distance_penalty_scale', 500.0)
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
        self.get_logger().info(f"Published target marker at position: {self.target_position}")

    def _update_target_marker(self):
        """Updates the target marker position in RViz."""
        self._publish_target_marker()

    def joint_state_callback(self, msg):
        # Reorder the joints to match the order we expect
        joint_positions = [0.0] * len(self.robot_joints_names)
        for i, name in enumerate(self.robot_joints_names):
            if name in msg.name:
                idx = msg.name.index(name)
                joint_positions[i] = msg.position[idx]
        self.current_joint_state = np.array(joint_positions)

    def step(self, action):
        self.episode_step_count += 1
        
        # 1. Apply Action: Convert velocity action to position target
        current_angles = self.get_joint_angles()
        if current_angles is None:
            return self.observation_space.sample(), 0, False, True, {"error": "Joint states not available"}

        # Integrate velocity over a small time step dt
        dt = self.dt
        new_target_angles = current_angles + action * dt
        
        # Clip to joint limits
        new_target_angles = np.clip(new_target_angles, self.joint_limits_lower, self.joint_limits_upper)

        # Send the trajectory to the robot
        result = self.send_trajectory_goal(new_target_angles.tolist(), dt)
        if not result:
            self.get_logger().error("Failed to send trajectory goal")
            return self.observation_space.sample(), 0, False, True, {"error": "Failed to send trajectory goal"}
        #time.sleep(dt) # Wait for the action to have an effect


        # 2. Get New Observation
        observation = self._get_observation()
        if observation is None:
            return self.observation_space.sample(), 0, False, True, {"error": "Failed to get observation"}

        # 3. Calculate Reward
        ee_pos = self.get_end_effector_pose()[:3]
        distance_to_target = np.linalg.norm(ee_pos - self.target_position)
        
        reward = self.reward_max - self.distance_penalty_scale * distance_to_target
        self.get_logger().info(f"Step {self.episode_step_count}: Distance to target: {distance_to_target:.4f}, Reward: {reward:.4f}")

        # 4. Check for Termination
        terminated = False
        if distance_to_target < self.goal_tolerance: # Goal reached
            reward += self.goal_bonus
            terminated = True
            self.get_logger().info("Goal Reached!")

        # Check for timeout
        truncated = False
        if self.episode_step_count >= self.max_steps_per_episode:
            truncated = True
            self.get_logger().info("Episode timed out.")

        return observation, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_step_count = 0

        # Reset robot to a home position
        home_joint_angles = self.home_position
        result = self.send_trajectory_goal(home_joint_angles) # returns True if successful

        if not result:
            self.get_logger().error("Failed to send home joint angles")
            return None

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
        joint_angles = self.get_joint_angles()
        ee_transform = self.get_end_effector_pose()
        self.get_logger().info("Getting observation...")

        if joint_angles is None or ee_transform is None:
            self.get_logger().warn("Could not get complete observation.")
            return None

        self.get_logger().info(f"Joint Angles: {joint_angles}, End Effector Pose: {ee_transform}")

        ee_pos = ee_transform[:3]
        vector_to_target = self.target_position - ee_pos

        return np.concatenate([joint_angles, vector_to_target]).astype(np.float32)

    def _generate_random_target(self):
        # Generate a reachable (x, y, z) position within the workspace
        bounds = self.config.get('environment.workspace_bounds', {})
        x = np.random.uniform(bounds.get('x_min', -0.21), bounds.get('x_max', 0.2))
        y = np.random.uniform(bounds.get('y_min', -0.21), bounds.get('y_max', 0.2))
        z = np.random.uniform(bounds.get('z_min', 0.9), bounds.get('z_max', 0.91))
        return np.array([x, y, z])

    def get_joint_angles(self):
        if self.current_joint_state is None:
            self.get_logger().warn("Joint states have not been received yet.")
            return None
        return self.current_joint_state

    def get_end_effector_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform('base_link', 'tool0', rclpy.time.Time())
            pos = transform.transform.translation
            rot = transform.transform.rotation
            return np.array([pos.x, pos.y, pos.z, rot.x, rot.y, rot.z, rot.w])
        except TransformException as ex:
            self.get_logger().warn(f'Could not get transform: {ex}')
            return None

    def send_trajectory_goal(self, joint_angles, dt=2.0):
        goal_msg = FollowJointTrajectory.Goal()
        trajectory_point = JointTrajectoryPoint()
        trajectory_point.positions = joint_angles
        trajectory_point.time_from_start = Duration(sec=1, nanosec=0)
        
        goal_msg.trajectory.joint_names = self.robot_joints_names
        goal_msg.trajectory.points.append(trajectory_point)
        
        send_goal_future = self._action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(
            self, send_goal_future, timeout_sec=20
        )
        
        if not send_goal_future.done():
            self.get_logger().error("Failed to send goal")
            return False

        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().error("Goal was rejected")
            return False

        # Wait for result
        self.get_logger().info("Waiting for trajectory execution")
        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(
            self, get_result_future, timeout_sec=20
        )

        if not get_result_future.done():
            self.get_logger().error("Failed to get result")
            return False

        result = get_result_future.result().result
        return result.error_code == FollowJointTrajectory.Result.SUCCESSFUL
        

    def close(self):
        """Cleanup ROS 2 resources."""
        self.get_logger().info("Cleaning up environment.")
        self.destroy_node()

def main():
    rclpy.init()
    
    env = UR5Env()
    
    executor = MultiThreadedExecutor()
    executor.add_node(env)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()

    try:
        # --- 1. Check the environment ---
        # It will display warnings if something is wrong
        check_env(env)
        print("\nEnvironment check passed!")

        # --- 2. Instantiate and Train the Agent ---
        # SAC (Soft Actor-Critic) is a good choice for continuous control tasks
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./ur5_sac_tensorboard/"
        )
        
        print("\n--- Starting Training ---")
        # Train for a specified number of steps
        model.learn(total_timesteps=20000, log_interval=4)
        
        # --- 3. Save the Trained Model ---
        model_path = "models/sac_ur5_reacher_model.zip"
        print(f"--- Model saved to {model_path} ---")

        # --- 4. Load and Evaluate the Trained Model ---
        del model # remove to demonstrate saving and loading
        
        print("\n--- Loading and Evaluating Trained Model ---")
        loaded_model = SAC.load(model_path, env=env)
        episodes = 100

        for episode in range(episodes):
            obs, info = env.reset()
            done = False
            total_reward = 0
            while not done:
                # Use the model to predict the best action
                action, _states = loaded_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
            model.save(model_path)
            print(f"Evaluation Episode {episode + 1} finished with total reward: {total_reward:.2f}")

    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        print("Closing environment and shutting down ROS.")
        env.close()
        rclpy.shutdown()
        thread.join()

if __name__ == '__main__':
    main()
