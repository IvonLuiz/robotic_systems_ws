#!/usr/bin/env python3
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node


class InitPositionClient(Node):
    def __init__(self):
        super().__init__("init_position_client")
        self._client = ActionClient(
            self,
            FollowJointTrajectory,
            "joint_trajectory_controller/follow_joint_trajectory",
        )

    def send_goal(self):
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
        point.positions = [
            0.0,
            -1.57,
            0.0,
            -1.57,
            0.0,
            0.0,
        ]  # sua pose "braço para cima"
        point.time_from_start.sec = 2
        goal_msg.trajectory.points = [point]

        self._client.wait_for_server()
        self._client.send_goal_async(goal_msg)


def main(args=None):
    rclpy.init(args=args)
    node = InitPositionClient()
    node.send_goal()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
