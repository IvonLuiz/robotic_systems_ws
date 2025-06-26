import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import numpy as np
from geometry_msgs.msg import Pose
from control_msgs.action import FollowJointTrajectory
from scipy.spatial.transform import Rotation as R
from ur5_interfaces.msg import PoseList
import math


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
        for pose in msg.poses:
            self.get_logger().info(f"Processing pose: {pose}")
            # Here you can add code to send the result to the action server or use it as needed.

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

    def calculate_inverse_kinematics(
        self, target_pose: Pose
    ) -> tuple[float, float, float, float, float, float]:
        """
        Calculate the inverse kinematics for a UR5 Manipulator

        :target_pose: Pose - Desired pose to be obtainede
        """
        self.get_logger().info(
            f"Calculating inverse kinematics for target pose: {target_pose}"
        )

        is_left_shoulder = 1 if self.get_parameter("is_left_shoulder").value else -1
        is_elbow_up = 1 if self.get_parameter("is_elbow_up").value else -1

        T_0_6: np.ndarray = np.eye(4)
        transformation_matrixs = []
        for i in range(len(self.dh_table.shape[0])):
            theta, d, a, alpha = self.dh_table[i]
            T_i = self._calculate_transformation_matrix(theta, d, a, alpha)
            transformation_matrixs.append(T_i)
            T_0_6 = T_0_6 @ T_i

        d_4 = self.dh_table[3, 1]
        d_6 = self.dh_table[5, 1]
        P_0_5 = T_0_6 @ np.array([0, 0, -d_6, 1]) - np.array([0, 0, 0, 1])

        psi = math.atan2(P_0_5[1], P_0_5[0])

        phi = is_left_shoulder * math.acos(
            d_4 / math.sqrt(P_0_5[0] ** 2 + P_0_5[1] ** 2)
        )

        theta_1 = psi + phi + math.pi / 2

        P_0_6_X = T_0_6[:3, 0]
        P_0_6_Y = T_0_6[:3, 1]

        P_1_6_Z = P_0_6_X * math.sin(theta_1) + P_0_6_Y * math.cos(theta_1)
        theta_5 = is_left_shoulder * math.acos((P_1_6_Z - d_4) / d_6)

        T_0_1 = transformation_matrixs[0]

        T_6_1 = np.linalg.inv(np.linalg.inv(T_0_1) @ T_0_6)
        T_6_1_Z_x = T_6_1[:2, 0]
        T_6_1_Z_y = T_6_1[:2, 1]

        theta_6 = math.atan2(
            -T_6_1_Z_y / math.sin(theta_5), T_6_1_Z_x / math.sin(theta_5)
        )

        T_1_4 = np.eye(4)
        for i in range(1, 4):
            T_1_4 = T_1_4 @ transformation_matrixs[i]

        P_1_3 = T_1_4 @ np.array([0, -d_4, 0, 1]) - np.array([0, 0, 0, 1])

        a_2 = self.dh_table[1, 2]
        a_3 = self.dh_table[2, 2]
        P_1_3_norm = np.linalg.norm(P_1_3)
        theta_3 = is_elbow_up * math.acos(
            (P_1_3_norm**2 - a_2**2 - a_3**3) / (2 * a_2 * a_3)
        )

        theta_2 = math.atan2(
            P_1_3[1],
            -P_1_3[0],
        ) - math.asin(a_3 * math.sin(theta_3) / P_1_3_norm)

        T_3_4 = transformation_matrixs[3]
        theta_4 = math.atan2(T_3_4[0, 1], T_3_4[0, 0])

        return theta_1, theta_2, theta_3, theta_4, theta_5, theta_6


def main():
    rclpy.init()
    node = IKMotionPlanner()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
