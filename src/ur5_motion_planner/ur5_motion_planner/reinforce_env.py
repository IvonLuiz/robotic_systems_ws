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

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from sensor_msgs.msg import JointState

from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

class UR5Env(gym.Env, Node):
    """
    Reinforcement Learning Environment for a UR5 robot in Gazebo.
    Inherits from both gymnasium.Env and rclpy.node.Node.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self):
        # Initialize the ROS 2 Node part in a ReentrantCallbackGroup
        # to allow for parallel callbacks and service calls
        Node.__init__(self, 'ur5_rl_environment')
        self.callback_group = ReentrantCallbackGroup()
        
        # Initialize the Gym Environment part
        gym.Env.__init__(self)

        # --- Gym Environment Setup ---
        # Action: 6 joint velocities [-1.0, 1.0] rad/s
        action_low = np.array([-1.0] * 6, dtype=np.float32)
        action_high = np.array([1.0] * 6, dtype=np.float32)
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
        
        # Waiting for action server and for the first joint state message to initialize current_joint_state
        self.get_logger().info("Waiting for Action Server...")
        self._action_client.wait_for_server()
        self.get_logger().info("Action Server is ready.")
        self.get_logger().info("Waiting for Joint States...")
        while self.current_joint_state is None:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.current_joint_state is not None:
                self.get_logger().info("Received initial joint states.")
            else:
                self.get_logger().warn("Waiting for joint states...")

        # --- RL State ---
        self.target_position = self._generate_random_target()
        self.episode_step_count = 0
        self.max_steps_per_episode = 200 # Timeout after 200 steps

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
        if current_angles is None: # Handle case where joint states are not yet received
            # Return a dummy response or wait
            return self.observation_space.sample(), 0, False, True, {"error": "Joint states not available"}

        # Integrate velocity over a small time step dt
        dt = 0.1 
        new_target_angles = current_angles + action * dt
        
        # Clip to joint limits to be safe (TODO: Define proper limits)
        lower_limits = np.array([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi])
        upper_limits = np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])
        new_target_angles = np.clip(new_target_angles, lower_limits, upper_limits)

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
        
        reward = -distance_to_target  # Dense reward

        # 4. Check for Termination
        terminated = False
        if distance_to_target < 0.05: # Goal reached
            reward += 100.0
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
        home_joint_angles = [
            0,
            -np.pi / 2,
            0.01,
            -np.pi / 2,
            0.01,
            0.01
        ]
        result = self.send_trajectory_goal(home_joint_angles) # returns True if successful

        if not result:
            self.get_logger().error("Failed to send home joint angles")
            return None

        # Generate a new random target
        self.target_position = self._generate_random_target()
        self.get_logger().info(f"New target generated at: {self.target_position}")

        initial_observation = self._get_observation()
        if initial_observation is None:
            # Handle the error case, maybe by retrying
            self.get_logger().error("Failed to get initial observation during reset. Trying again.")
            time.sleep(0.5)
            return self.reset(seed=seed, options=options)

        return initial_observation, {}

    def _get_observation(self):
        joint_angles = self.get_joint_angles()
        ee_transform = self.get_end_effector_pose()

        if joint_angles is None or ee_transform is None:
            self.get_logger().warn("Could not get complete observation.")
            return None
        
        ee_pos = ee_transform[:3]
        vector_to_target = self.target_position - ee_pos
        
        return np.concatenate([joint_angles, vector_to_target]).astype(np.float32)

    def _generate_random_target(self):
        # Generate a reachable (x, y, z) position within the workspace
        # These are example bounds, you should define a proper workspace
        x = np.random.uniform(0.3, 0.6)
        y = np.random.uniform(-0.4, 0.4)
        z = np.random.uniform(0.2, 0.5)
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
        model.save(model_path)
        print(f"--- Model saved to {model_path} ---")

        # --- 4. Load and Evaluate the Trained Model ---
        del model # remove to demonstrate saving and loading
        
        print("\n--- Loading and Evaluating Trained Model ---")
        loaded_model = SAC.load(model_path, env=env)

        for episode in range(10): # Evaluate for 10 episodes
            obs, info = env.reset()
            done = False
            total_reward = 0
            while not done:
                # Use the model to predict the best action
                action, _states = loaded_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
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
