from rclpy.node import Node
import numpy as np
import rclpy
import time
import os
import argparse
import csv
from concurrent.futures import Future

from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_srvs.srv import Trigger
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from builtin_interfaces.msg import Duration
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from control_msgs.msg import JointTolerance


# Timeout constants
TIMEOUT_WAIT_ACTION = 20


class DatasetGenerator(Node):
    def __init__(self, num_points: int):
        """
        This class generates datasets for UR5 robot to be used in machine learning algorithms.
        The generation is done by applying forward kinematics to a set of joint angles and
        saving the resulting end-effector positions and orientations. The angles will be the
        ground truth for the dataset, while the end-effector positions and orientations will be
        the inputs for the machine learning algorithms.
        
        :param num_points: Number of data points to generate.
        """
        super().__init__('dataset_generator_node')  # Initialize the Node class
        self.num_points = num_points
        self.completion_future = Future()  # Future to signal completion of dataset generation
        self.initial_angles = [0, -np.pi/2, 0.01, -np.pi/2,  0.01, 0.01]  # UR5 initial joint angles pointing straight down
        self.angle_step = np.pi/2  # new angle will be updated between previous angle + [-angle_step, angle_step] (radians)
        self.reach_position_duration = 1  # seconds to reach position (integer). 
                                          # the shorter the duration, the faster the robot will move.
        self.start_time = time.time()  # initialize start time for data collection
        self.robot_joints_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self.declare_parameter(
            "controller_name",
            "/scaled_joint_trajectory_controller/follow_joint_trajectory",
        )

        # Setup callback group for concurrent service calls
        self.callback_group = ReentrantCallbackGroup()
        # Initialize controller action client
        controller_name = self.get_parameter("controller_name").value
        self._action_client = ActionClient(
            self, 
            FollowJointTrajectory, 
            controller_name,
            callback_group=self.callback_group
        )
        self.get_logger().info("Waiting for gazebo simulation to initialize...")
        self.wait_for_simulation()

        # Add TF2 listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Send a single random joint angle configuration and print end-effector pose
        self.send_random_joint_angles()

    def get_timer(self) -> None:
        """
        Get the timer for elapsed time since the start of data collection.

        :return: None
        """
        elapsed_time = time.time() - self.start_time
        return f"{elapsed_time:.2f}"  # Return elapsed time in seconds with 2 decimal places

    def get_end_effector_pose(self) -> list[float]:
        """
        Get end effector pose using TF2.

        :return: list[float] - A list containing the end-effector pose [x, y, z, qx, qy, qz, qw] or None if unavailable.
        """
        try:
            # Lookup transform from base_link to tool0 (or your end effector frame)
            transform = self.tf_buffer.lookup_transform(
                'base_link',  # source frame
                'tool0',      # target frame
                rclpy.time.Time())
            
            pose = [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ]
            return pose
        except TransformException as ex:
            self.get_logger().warn(f'Could not get transform: {ex}')
            return None
    
    def send_trajectory(self, waypts: list[list[float]], time_vec: list[Duration], action_client: ActionClient) -> bool:
        """
        Send robot trajectory.

        :param waypts: list[list[float]] - List of waypoints for the trajectory.
        :param time_vec: list[Duration] - List of time durations for each waypoint.
        :param action_client: ActionClient - Action client to send the trajectory.
        :return: bool - True if the trajectory execution was successful, False otherwise.
        """
        if len(waypts) != len(time_vec):
            raise Exception("waypoints vector and time vec should be same length")

        # Construct test trajectory
        joint_trajectory = JointTrajectory()
        joint_trajectory.joint_names = self.robot_joints_names
        for i in range(len(waypts)):
            point = JointTrajectoryPoint()
            point.positions = waypts[i]
            point.time_from_start = time_vec[i]
            joint_trajectory.points.append(point)

        # Create goal message
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = joint_trajectory
        goal_msg.goal_tolerance = [
            JointTolerance(name=joint_name, position=0.0001)  # 0.0001 rad ≈ 0.0057 degrees
            for joint_name in self.robot_joints_names
        ]
        goal_msg.goal_time_tolerance = Duration(sec=1)  # 1-second margin for goal completion
        
        # Send goal and wait for result
        self.get_logger().info("Sending trajectory goal")
        send_goal_future = action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future, timeout_sec=TIMEOUT_WAIT_ACTION)
        
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
        rclpy.spin_until_future_complete(self, get_result_future, timeout_sec=TIMEOUT_WAIT_ACTION)
        
        if not get_result_future.done():
            self.get_logger().error("Failed to get result")
            return False
            
        result = get_result_future.result().result
        return result.error_code == FollowJointTrajectory.Result.SUCCESSFUL

    def sample_joint_angles(self, previous_angles: list[float] = None) -> list[float]:
        """
        Generate random joint angles within valid and safe limits.

        :param previous_angles: list[float] - List of previous joint angles to base the new angles on.
        :return: list[float] - A list of new random joint angles.
        """        
        # Generate random angles within the specified limits
        random_angles = np.random.uniform(
            low=-self.angle_step, high=self.angle_step, size=(len(self.robot_joints_names))
        )
        new_random_angles = np.array(random_angles) + np.array(previous_angles)

        return new_random_angles.tolist()

    def wait_for_simulation(self) -> None:
        """
        Wait for the simulation to be ready.
        """
        self.get_logger().info("Waiting for action server to be ready...")
        while True:
            if self._action_client.server_is_ready():
                self.get_logger().info("Action server is available, dataset generation in 3 seconds.")
                time.sleep(3)
                break

    def send_random_joint_angles(self) -> None:
        """
        Send random joint angles to the robot.
        Since we have constraints on the UR5 robot, we will generate angles
        within a range that avoids collisions and ensures safe operation.
        The angle generation will be medium size steps from the previous angles,
        starting from the initial pose. If we get a collision or failure, we will
        generate new angles from previous correct angles.
        """
        previous_angles = self.initial_angles  # previous angles will be our initial angles
        angles = self.initial_angles  # next angles will be our initial angles

        n = 0
        while n < self.num_points:
            self.get_logger().info(f"------Generating dataset point {n+1}/{self.num_points}------")
            self.get_logger().info(f"Current time running: {self.get_timer()} s")
            self.get_logger().info(f"Generated joint angles: {angles}")
            
            # time vector
            time_from_start = []
            duration_msg = Duration()
            duration_msg.sec = self.reach_position_duration
            time_from_start.append(duration_msg)

            # sending tracjectory and checking result
            result = self.send_trajectory([angles], time_from_start, self._action_client)

            if result:
                self.get_logger().info("Trajectory executed successfully.")
                # Get the end-effector pose after movement
                end_effector_pose = self.get_end_effector_pose()
                if end_effector_pose:
                    self.get_logger().info(f"End effector pose after movement: {end_effector_pose}")
                    self.save_dataset(angles, end_effector_pose, 'data/dataset')
                    self.save_dataset_reachability(angles, True, 'data/reachability')
                    # update states
                    previous_angles = angles
                    n += 1
                else:
                    self.get_logger().warn("Could not get end effector pose after movement, resetting to previous angles.")
                    angles = previous_angles  # reset to previous angles if pose is not available
                    self.save_dataset_reachability(angles, False, 'data/reachability')
            else:
                self.get_logger().error("Failed to execute trajectory, resetting to previous angles.")
                angles = previous_angles  # reset to previous angles if failed
                self.save_dataset_reachability(angles, False, 'data/reachability')

            angles = self.sample_joint_angles(previous_angles)  # sample new angles for the next iteration
        
        self.get_logger().info("Dataset generation completed.")
        self.completion_future.set_result(True)

    def save_dataset(self, angles: list[float], end_effector_pose: list[float], filename: str = 'data/dataset') -> None:
        """
        Add a new row to the dataset and overwrite the previous one.
        This dataset tracks end-effector pose from joint angles.

        :param angles: list[float] - List of joint angles.
        :param end_effector_pose: list[float] - List of end-effector pose elements (x, y, z, qx, qy, qz, qw).
        :param filename: str - Base filename for saving the dataset.
        """
        os.makedirs('data', exist_ok=True)
        
        npz_filename = filename + '.npz'
        csv_filename = filename + '.csv'
        
        # Initialize empty arrays if file doesn't exist
        if not os.path.exists(npz_filename):
            joint_angles = np.empty((0, 6))  # Empty array for 6 joint angles
            ee_poses = np.empty((0, 7))     # Empty array for 7 pose elements (x,y,z + quaternion)
        else:
            # Load existing data
            with np.load(npz_filename) as data:
                joint_angles = data['joint_angles']
                ee_poses = data['end_effector_pose']
        if not os.path.exists(csv_filename):
            # Create CSV file with header if it doesn't exist
            with open(csv_filename, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(self.robot_joints_names + ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
        
        new_angles = np.array([angles])
        new_pose = np.array([end_effector_pose])
        
        # vertically stack the new data
        joint_angles = np.vstack((joint_angles, new_angles))
        ee_poses = np.vstack((ee_poses, new_pose))

        # .npz and .csv saving
        np.savez_compressed(npz_filename, 
                        joint_angles=joint_angles, 
                        end_effector_pose=ee_poses)
        with open(csv_filename, mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(angles + end_effector_pose)

    def save_dataset_reachability(self, angles: list[float], reachable: bool, filename: str = 'data/reachability') -> None:
        """
        Add a new row to the dataset and overwrite the previous one.
        This dataset tracks whether the given joint angles are reachable.

        :param angles: list[float] - List of joint angles.
        :param reachable: bool - True if the joint angles are reachable, False otherwise.
        :param filename: str - Base filename for saving the reachability data.
        """
        os.makedirs('data', exist_ok=True)

        csv_filename = filename + '.csv'
        npz_filename = filename + '.npz'

        # Initialize empty arrays if file doesn't exist
        if not os.path.exists(npz_filename):
            joint_angles = np.empty((0, 6))  # Empty array for 6 joint angles
            reachability = np.empty((0, 1))  # Empty array for reachability
        else:
            # Load existing data
            with np.load(npz_filename) as data:
                joint_angles = data['joint_angles']
                reachability = data['reachability']

        # Create CSV file with header if it doesn't exist
        if not os.path.exists(csv_filename):
            with open(csv_filename, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(self.robot_joints_names + ['reachable'])

        # Append the data to CSV
        with open(csv_filename, mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(angles + [reachable])

        # Append the data to NPZ
        new_angles = np.array([angles])
        new_reachability = np.array([[reachable]])
        joint_angles = np.vstack((joint_angles, new_angles))
        reachability = np.vstack((reachability, new_reachability))

        np.savez_compressed(npz_filename, joint_angles=joint_angles, reachability=reachability)
def main():
    rclpy.init()
    
    parser = argparse.ArgumentParser(description='UR5 Dataset Generator')
    parser.add_argument('--num-points', type=int, default=100,
                      help='Number of data points to generate (default: 100)')
    args = parser.parse_args()

    dataset_generator_node = DatasetGenerator(num_points=args.num_points)
    executor = MultiThreadedExecutor()
    executor.add_node(dataset_generator_node)

    try:
        # Spin until completion or interrupt
        while rclpy.ok() and not dataset_generator_node.completion_future.done():
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        dataset_generator_node.get_logger().info("Keyboard interrupt received")
    finally:
        # Clean shutdown sequence
        dataset_generator_node.get_logger().info("Shutting down...")
        executor.shutdown()
        dataset_generator_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()