from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.task import Future
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance
from trajectory_msgs.msg import JointTrajectoryPoint
from scipy.spatial.transform import Rotation as R
from ur5_interfaces.msg import PoseList
from tf_transformations import quaternion_matrix
from builtin_interfaces.msg import Duration
import numpy as np
import rclpy
import math
import time


class IKMotionPlanner(Node):
    # Denavit-Hartenberg parameters for UR5 manipulator
    # The parameters are in the form of [theta, d, a, alpha]
    # theta, d, a, alpha
    # 1: theta_1, 0.089159, 0, pi/2
    # 2: theta_2, 0, -0.425, 0
    # 3: theta_3, 0, -0.39225, 0
    # 4: theta_4, 0.10915, 0, pi/2
    # 5: theta_5, 0.09465, 0, -pi/2
    # 6: theta_6, 0.0823, 0, 0
    dh_table: np.ndarray = np.array(
        [
            [0, 0.089159, 0, np.pi / 2],
            [0, 0, -0.425, 0],
            [0, 0, -0.39225, 0],
            [0, 0.10915, 0, np.pi / 2],
            [0, 0.09465, 0, -np.pi / 2],
            [0, 0.0823, 0, 0],
        ]
    )
    pose_list: list[Pose] = []

    def __init__(self):
        super().__init__("ik_motion_planner")
        self.get_logger().info("IKMotionPlanner node has been started.")

        self.declare_parameter("is_left_shoulder", True)
        self.declare_parameter("is_elbow_up", True)
        self.declare_parameter(
            "controller_name",
            "/scaled_joint_trajectory_controller/follow_joint_trajectory",
        )
        self.declare_parameter("pose_list_topic", "/pose_list")

        controller_name = self.get_parameter("controller_name").value
        self._action_client = ActionClient(self, FollowJointTrajectory, controller_name)

        self.get_logger().info(f"Waiting for action server on {controller_name}")
        self._action_client.wait_for_server()
        self.get_logger().info("Action server is available.")

        # Subscriber to the PoseList topic
        self.get_logger().info(
            f"Subscribing to pose list topic: {self.get_parameter('pose_list_topic').value}"
        )
        self.pose_subscriber = self.create_subscription(
            PoseList,
            self.get_parameter("pose_list_topic").value,
            self._pose_list_callback,
            10,
        )

        self.get_logger().info("IKMotionPlanner node is ready to process poses.")

    def _pose_list_callback(self, msg: PoseList):
        """
        Callback function for the PoseList subscriber.
        It processes the received poses and calculates inverse kinematics for each pose.

        :param msg: PoseList - List of poses to process
        """
        self.get_logger().info(f"Received {len(msg.poses)} poses.")
        self.pose_list = msg.poses
        joint_angles = self.calculate_inverse_kinematics(self.pose_list[0])
        self._execute_trajectory(joint_angles)

    def _execute_trajectory(self, joint_angles: list[float]):
        """
        Execute the trajectory with the given joint angles.

        :param joint_angles: list[float] - List of joint angles to execute
        """
        self.get_logger().info(
            f"Executing trajectory with joint angles: {joint_angles}"
        )

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        point = JointTrajectoryPoint()
        self.get_logger().info(
            f"Setting joint angles for trajectory point: {joint_angles}"
        )
        self.get_logger().info(f"Joint angles type: {type(joint_angles)}")
        self.get_logger().info(f"Joint angles length: {len(joint_angles)}")
        point.positions = joint_angles
        point.velocities = [0.0] * len(joint_angles)  # No velocity control
        point.time_from_start = Duration(sec=10, nanosec=0)

        goal_msg.trajectory.points.append(point)
        self.get_logger().info(
            f"Sending trajectory:\nJoint names: {goal_msg.trajectory.joint_names}\nPositions: {point.positions}"
        )
        goal_msg.goal_time_tolerance = Duration(sec=0, nanosec=0)
        goal_msg.goal_tolerance = [
            JointTolerance(
                position=0.01, velocity=0.01, name=goal_msg.trajectory.joint_names[i]
            )
            for i in range(6)
        ]

        send_goal_future = self._action_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self._goal_response_callback)

        self.pose_list.pop(0)

    def _goal_response_callback(self, future: Future):
        """
        Callback function for the goal response.

        :param future: Future - The future object containing the result of the action
        """
        response = future.result()
        if not response.accepted:
            self.get_logger().error("Goal was rejected by the action server.")
            raise RuntimeError("Goal was rejected by the action server.")
        self.get_logger().info("Goal accepted by the action server.")
        result_future = response.get_result_async()
        result_future.add_done_callback(self._result_callback)

    def _result_callback(self, future: Future):
        """
        Callback function for the result of the action.

        :param future: Future - The future object containing the result of the action
        """
        result = future.result().result
        status = future.result().status
        self.get_logger().info(
            f"Action completed with status: {status}, result: {result}"
        )
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Action succeeded.")
            time.sleep(1)  # Wait for a second before processing the next pose
            self.get_logger().info("Processing next pose in the list if available.")
            if self.pose_list:
                next_pose = self.pose_list[0].position
                self.get_logger().info(f"Next pose to process: {next_pose}")
                joint_angles = self.calculate_inverse_kinematics(next_pose)
                self._execute_trajectory(joint_angles)
        else:
            if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
                self.get_logger().error(
                    f"Action failed with error code: {result.error_code}"
                )
                raise RuntimeError(
                    f"Action failed with error code: {result.error_code}"
                )

    def _calculate_transformation_matrix(
        self, theta: float, d: float, a: float, alpha: float
    ) -> np.ndarray:
        return np.array(
            [
                [
                    np.cos(theta),
                    -np.sin(theta) * np.cos(alpha),
                    np.sin(theta) * np.sin(alpha),
                    a * np.cos(theta),
                ],
                [
                    np.sin(theta),
                    np.cos(theta) * np.cos(alpha),
                    -np.cos(theta) * np.sin(alpha),
                    a * np.sin(theta),
                ],
                [0, np.sin(alpha), np.cos(alpha), d],
                [0, 0, 0, 1],
            ]
        )

    def _calculate_rigid_transformation(
        self, P: np.ndarray, Q: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the rigid transformation matrix from point P to point Q.

        :param P: np.ndarray - Point P in homogeneous coordinates
        :param Q: np.ndarray - Point Q in homogeneous coordinates
        :return: np.ndarray - Rigid transformation matrix
        """
        rot, rmsd = R.align_vectors(P, Q)
        P_rotated = rot.apply(P)
        translation = np.mean(Q, axis=0) - np.mean(P_rotated, axis=0)
        T = np.eye(4)
        T[:3, :3] = rot.as_matrix()
        T[:3, 3] = translation
        return T

    def pose_to_matrix(self, pose: Pose) -> np.ndarray:
        """Convert ROS Pose to 4x4 transformation matrix."""
        q = pose.orientation
        t = pose.position
        T = quaternion_matrix([q.x, q.y, q.z, q.w])
        T[0, 3] = t.x
        T[1, 3] = t.y
        T[2, 3] = t.z
        return T

    def clamp(self, value: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
        """Clamp a value to avoid math domain errors."""
        return max(min(value, max_val), min_val)

    def calculate_inverse_kinematics(
        self, target_pose: Pose
    ) -> tuple[float, float, float, float, float, float]:
        self.get_logger().info(
            f"Calculating inverse kinematics for target pose: {target_pose}"
        )

        is_left_shoulder = 1 if self.get_parameter("is_left_shoulder").value else -1
        is_elbow_up = 1 if self.get_parameter("is_elbow_up").value else -1

        T_0_6 = self.pose_to_matrix(target_pose)
        R_0_6 = T_0_6[:3, :3]
        P_0_6 = T_0_6[:3, 3]

        d_4 = self.dh_table[3, 1]
        d_6 = self.dh_table[5, 1]
        a_2 = self.dh_table[1, 2]
        a_3 = self.dh_table[2, 2]

        # Position of wrist (joint 5)
        P_0_5 = P_0_6 - d_6 * R_0_6[:, 2]

        # θ1
        psi = math.atan2(P_0_5[1], P_0_5[0])
        phi = is_left_shoulder * math.acos(
            self.clamp(d_4 / math.hypot(P_0_5[0], P_0_5[1]))
        )
        theta_1 = psi + phi + math.pi / 2

        # Rotation from base to wrist
        sin1 = math.sin(theta_1)
        cos1 = math.cos(theta_1)
        P_1_6_Z = P_0_6[0] * sin1 - P_0_6[1] * cos1
        raw_theta_5_cos = (P_1_6_Z - d_4) / d_6
        self.get_logger().info(f"Raw acos input (theta_5): {raw_theta_5_cos}")
        theta_5 = is_left_shoulder * math.acos(self.clamp(raw_theta_5_cos))

        # θ6
        R_0_1 = self._calculate_transformation_matrix(
            self.dh_table[0, 0],
            self.dh_table[0, 1],
            self.dh_table[0, 2],
            self.dh_table[0, 3],
        )[:3, :3]
        R_1_6 = np.linalg.inv(R_0_1) @ R_0_6
        if abs(math.sin(theta_5)) < 1e-6:
            theta_6 = 0.0  # avoid division by zero
        else:
            theta_6 = math.atan2(
                -R_1_6[1, 2] / math.sin(theta_5), R_1_6[0, 2] / math.sin(theta_5)
            )

        # θ3 and θ2 via cosine law
        # Compute T_1_4 based on known DH params
        T_0_1 = self._calculate_transformation_matrix(
            theta_1, self.dh_table[0, 1], self.dh_table[0, 2], self.dh_table[0, 3]
        )
        T_1_0 = np.linalg.inv(T_0_1)
        T_1_6 = T_1_0 @ T_0_6
        P_1_6 = T_1_6[:3, 3]

        # P_1_3: Vector to wrist joint (joint 4) from frame 1
        P_1_3 = P_1_6 - d_6 * T_1_6[:3, 2]
        P_1_3_norm = np.linalg.norm(P_1_3)

        cos_theta_3 = self.clamp((P_1_3_norm**2 - a_2**2 - a_3**2) / (2 * a_2 * a_3))
        theta_3 = is_elbow_up * math.acos(cos_theta_3)

        # θ2 from triangle geometry
        k1 = a_2 + a_3 * math.cos(theta_3)
        k2 = a_3 * math.sin(theta_3)
        gamma = math.atan2(k2, k1)
        theta_2 = math.atan2(P_1_3[2], -P_1_3[0]) - gamma

        # θ4 from orientation
        R_1_2 = self._calculate_transformation_matrix(
            theta_2, self.dh_table[1, 1], self.dh_table[1, 2], self.dh_table[1, 3]
        )[:3, :3]
        R_2_3 = self._calculate_transformation_matrix(
            theta_3, self.dh_table[2, 1], self.dh_table[2, 2], self.dh_table[2, 3]
        )[:3, :3]
        R_1_3 = R_1_2 @ R_2_3
        R_3_6 = np.linalg.inv(R_1_3) @ R_1_6
        theta_4 = math.atan2(R_3_6[0, 1], R_3_6[0, 0])

        return theta_1, theta_2, theta_3, theta_4, theta_5, theta_6


def main():
    rclpy.init()
    node = IKMotionPlanner()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
