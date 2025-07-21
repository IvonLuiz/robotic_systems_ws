from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.task import Future
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance
from trajectory_msgs.msg import JointTrajectoryPoint
from scipy.spatial.transform import Rotation as R
from tf_transformations import quaternion_matrix
from builtin_interfaces.msg import Duration
import numpy as np
import rclpy
import math
import cmath
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
    d1: float = 0.0892
    a2: float = -0.425
    a3: float = -0.392
    d4: float = 0.1093
    d5: float = 0.09475
    d6: float = 0.0825
    d = np.matrix([d1, 0, 0, d4, d5, d6])
    a = np.matrix([0, a2, a3, 0, 0, 0])
    alph = np.matrix([math.pi / 2, 0, 0, math.pi / 2, -math.pi / 2, 0])

    def __init__(self):
        super().__init__("ik_motion_planner")
        self.get_logger().info("IKMotionPlanner node has been started.")

        self.declare_parameter("is_left_shoulder", True)
        self.declare_parameter("is_elbow_up", False)
        self.declare_parameter("is_wrist_up", True)
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
            Pose,
            self.get_parameter("pose_list_topic").value,
            self._pose_list_callback,
            10,
        )

        self.get_logger().info("IKMotionPlanner node is ready to process poses.")

    def _pose_list_callback(self, msg: Pose):
        """
        Callback function for the subscriber.
        It processes the received poses and calculates inverse kinematics for each pose.

        :param msg: Pose
        """
        self.get_logger().info("Received pose.")
        joint_angles = self.calculate_inverse_kinematics(msg)
        self.get_logger().info(
            f"Calculated joint angles for first pose: {np.degrees(joint_angles)}"
        )
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
        elif result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().error(
                f"Action failed with error code: {result.error_code}"
            )
            raise RuntimeError(f"Action failed with error code: {result.error_code}")

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

    def pose_to_matrix(self, pose: Pose) -> np.ndarray:
        # Extract position and orientation
        position = pose.position
        orientation = pose.orientation

        # Convert quaternion to rotation matrix
        rotation = R.from_quat(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        )
        rotation_matrix = rotation.as_matrix()  # 3x3 rotation matrix

        # Construct the 4x4 transformation matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = [position.x, position.y, position.z]
        self.get_logger().info(
            f"Position: {position.x}, {position.y}, {position.z}, "
            f"Orientation: {orientation.x}, {orientation.y}, {orientation.z}, {orientation.w}"
        )

        self.get_logger().info(f"Transformation matrix: {transform_matrix}")

        return transform_matrix

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

    def clamp(self, value: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
        """Clamp a value to avoid math domain errors."""
        return max(min(value, max_val), min_val)

    def AH(self, n, th, c):
        T_a = np.matrix(np.identity(4))
        T_a[0, 3] = self.a[0, n - 1]

        T_d = np.matrix(np.identity(4))
        T_d[2, 3] = self.d[0, n - 1]

        Rzt = np.matrix(
            [
                [math.cos(th[n - 1, c]), -math.sin(th[n - 1, c]), 0, 0],
                [math.sin(th[n - 1, c]), math.cos(th[n - 1, c]), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        Rxa = np.matrix(
            [
                [1, 0, 0, 0],
                [0, math.cos(self.alph[0, n - 1]), -math.sin(self.alph[0, n - 1]), 0],
                [0, math.sin(self.alph[0, n - 1]), math.cos(self.alph[0, n - 1]), 0],
                [0, 0, 0, 1],
            ]
        )

        A_i = T_d @ Rzt @ T_a @ Rxa
        return A_i

    def calculate_inverse_kinematics(
        self, target_pose: Pose
    ) -> tuple[float, float, float, float, float, float]:
        desired_pos = self.pose_to_matrix(target_pose)
        self.get_logger().info(f"Desired position: {desired_pos}")
        th = np.matrix(np.zeros((6, 8)))

        P_05 = (
            desired_pos @ np.matrix([0, 0, -self.d6, 1]).T - np.matrix([0, 0, 0, 1]).T
        )

        # theta1
        psi = math.atan2(P_05[1, 0], P_05[0, 0])
        phi = math.acos(self.d4 / math.sqrt(P_05[1, 0] ** 2 + P_05[0, 0] ** 2))

        th[0, 0:4] = math.pi / 2 + psi + phi
        th[0, 4:8] = math.pi / 2 + psi - phi
        th = th.real

        # theta5
        for c in [0, 4]:
            T_10 = np.linalg.inv(self.AH(1, th, c))
            T_16 = T_10 @ desired_pos
            th[4, c : c + 2] = +math.acos((T_16[2, 3] - self.d4) / self.d6)
            th[4, c + 2 : c + 4] = -math.acos((T_16[2, 3] - self.d4) / self.d6)

        th = th.real

        # theta6
        for c in [0, 2, 4, 6]:
            T_10 = np.linalg.inv(self.AH(1, th, c))
            T_16 = np.linalg.inv(T_10 @ desired_pos)
            th[5, c : c + 2] = math.atan2(
                -T_16[1, 2] / math.sin(th[4, c]), T_16[0, 2] / math.sin(th[4, c])
            )

        th = th.real

        # theta3
        for c in [0, 2, 4, 6]:
            T_10 = np.linalg.inv(self.AH(1, th, c))
            T_65 = self.AH(6, th, c)
            T_54 = self.AH(5, th, c)
            T_14 = (T_10 @ desired_pos) @ np.linalg.inv(T_54 @ T_65)
            P_13 = T_14 @ np.matrix([0, -self.d4, 0, 1]).T - np.matrix([0, 0, 0, 1]).T

            t3 = cmath.acos(
                (np.linalg.norm(P_13) ** 2 - self.a2**2 - self.a3**2)
                / (2 * self.a2 * self.a3)
            )
            th[2, c] = t3.real
            th[2, c + 1] = -t3.real

        # theta2 and theta4
        for c in range(8):
            T_10 = np.linalg.inv(self.AH(1, th, c))
            T_65 = np.linalg.inv(self.AH(6, th, c))
            T_54 = np.linalg.inv(self.AH(5, th, c))
            T_14 = (T_10 @ desired_pos) @ T_65 @ T_54
            P_13 = T_14 @ np.matrix([0, -self.d4, 0, 1]).T - np.matrix([0, 0, 0, 1]).T

            # theta2
            p13_y = float(P_13[1, 0])
            p13_x = float(P_13[0, 0])
            theta3 = float(th[2, c])
            p13_norm = float(np.linalg.norm(P_13))

            th[1, c] = -math.atan2(p13_y, -p13_x) + math.asin(
                self.a3 * math.sin(theta3) / p13_norm
            )

            # theta4
            T_32 = np.linalg.inv(self.AH(3, th, c))
            T_21 = np.linalg.inv(self.AH(2, th, c))
            T_34 = T_32 @ T_21 @ T_14
            th[3, c] = math.atan2(T_34[1, 0], T_34[0, 0])

        return tuple(th.real[:, 0].flatten().tolist()[0])


def main():
    rclpy.init()
    node = IKMotionPlanner()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
