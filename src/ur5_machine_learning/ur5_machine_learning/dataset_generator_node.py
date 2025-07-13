from rclpy.node import Node
import numpy as np
import rclpy
import time
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
    def __init__(self, num_points=None):
        '''!
        This class generates datasets for UR5 robot to be used in machine learning algorithms.
        The generation is done by applying forward kinematics to a set of joint angles and
        saving the resulting end-effector positions and orientations. The angles will be the
        ground truth for the dataset, while the end-effector positions and orientations will be
        the inputs for the machine learning algorithms.
        '''
        super().__init__('dataset_generator_node')  # Initialize the Node class
        self.num_points = num_points

        # Setup callback group for concurrent service calls
        self.callback_group = ReentrantCallbackGroup()

        # Example waypoints for testing (subistitute for random later)
        self.waypts = [
            [-1.6006, -1.7272, -2.2030, -0.8079, 1.5951, -0.0311],
            [-1.2, -1.4, -1.9, -1.2, 1.5951, -0.0311],
            [-1.6006, -1.7272, -2.2030, -0.8079, 1.5951, -0.0311],
        ]
        self.reach_position_duration = 1  # seconds to reach position (integer)

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

        # Initialize controller action client
        controller_name = self.get_parameter("controller_name").value
        self._action_client = ActionClient(
            self, 
            FollowJointTrajectory, 
            controller_name,
            callback_group=self.callback_group
        )
        # Wait for the simulation to initialize
        self.get_logger().info("Waiting for gazebo simulation to initialize...")
        self.wait_for_simulation()

        # Add TF2 listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Send a single random joint angle configuration and print end-effector pose
        self.get_logger().info("Testing single movement...")
        self.send_random_joint_angles()

    def get_end_effector_pose(self):
        """Get end effector pose using TF2."""
        try:
            # Lookup transform from base_link to tool0 (or your end effector frame)
            transform = self.tf_buffer.lookup_transform(
                'base_link',  # source frame
                'tool0',      # target frame
                rclpy.time.Time())
            
            # Extract position and orientation
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

    def generate_dataset(self):
        """Generate a dataset of joint angles and corresponding end-effector poses."""
        pass
    
    def send_trajectory(self, waypts, time_vec, action_client):
        """Send robot trajectory."""
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

    def sample_joint_angles(self):
        """Generate random joint angles within valid limits."""
        #return np.random.uniform(low=-np.pi, high=np.pi, size=len(self.robot_joints_names))
        return self.waypts[0]  # For testing, use the first waypoint

    def wait_for_simulation(self):
        """Wait for the simulation to be ready."""        
        self.get_logger().info("Waiting for action server to be ready...")
        while True:
            if self._action_client.server_is_ready():
                self.get_logger().info("Action server is available")
                break
        self.get_logger().info("Simulation is ready.")

    def send_random_joint_angles(self):
        """Send random joint angles to the robot."""
        joint_angles = self.sample_joint_angles()
        
        # Create time vector with builtin_interfaces.msg.Duration
        time_from_start = []
        duration_msg = Duration()
        duration_msg.sec = self.reach_position_duration
        time_from_start.append(duration_msg)
            
        self.get_logger().info(f"Sending joint angles: {joint_angles}")
        result = self.send_trajectory([joint_angles], time_from_start, self._action_client)
        if result:
            self.get_logger().info("Trajectory executed successfully.")
            # Get the end-effector pose after movement
            end_effector_pose = self.get_end_effector_pose()
            if end_effector_pose:
                self.get_logger().info(f"End effector pose after movement: {end_effector_pose}")
            else:
                self.get_logger().warn("Could not get end effector pose after movement")
        else:
            self.get_logger().error("Failed to execute trajectory.")

    def save_dataset(self, dataset, filename="ur5_dataset.npz"):
        """Save the dataset to a file."""
        pass


def main():
    rclpy.init()
    num_points = 10
    dataset_generator_node = DatasetGenerator(num_points=num_points)
    rclpy.spin(dataset_generator_node)

    # Cleanup
    dataset_generator_node.shutdown()
    dataset_generator_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()