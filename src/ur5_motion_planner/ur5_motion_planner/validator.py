from rclpy.node import Node
from ur5_interfaces.msg import PoseList
from geometry_msgs.msg import Pose, TransformStamped
from tf2_ros import TransformListener, Buffer
from builtin_interfaces.msg import Duration
import rclpy
import random


class Validator(Node):

    def __init__(self):
        super().__init__("motion_planner_validator")
        self.get_logger().info("Motion Planner Validator Node has been started.")

        self.declare_parameter("random_seed", 42)
        self.declare_parameter("pose_list_topic", "/pose_list_ik")
        self.declare_parameter("amount_of_poses", 10)
        self.declare_parameter("pose_threshold", 0.1)
        self.declare_parameter("sleep_duration", 5)
        random.seed(self.get_parameter("random_seed").value)

        random_pose_list = self.generate_random_pose_list()

        test_topic = self.create_publisher(
            PoseList, self.get_parameter("pose_list_topic").value, 10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        test_topic.publish(random_pose_list)

        self.get_logger().info(
            f"Published {len(random_pose_list.poses)} random poses to topic: {self.get_parameter('pose_list_topic').value}"
        )
        self.get_logger().info("Validating...")

        sleep_duration = self.get_parameter("sleep_duration").value

        self.get_logger().info("Motion Planner Validator Node has been started.")
        for i, pose in enumerate(random_pose_list.poses):
            self.get_logger().info(
                f"Pose {i}: Position ({pose.position.x}, {pose.position.y}, {pose.position.z}), "
                f"Orientation ({pose.orientation.x}, {pose.orientation.y}, {pose.orientation.z}, {pose.orientation.w})"
            )

            if self.wait_until_pose_reached(
                desired_pose=pose, timeout_sec=sleep_duration
            ):
                self.get_logger().info(f"Pose {i} reached successfully.")

    def generate_random_pose_list(self) -> PoseList:
        """Generate a random pose for testing purposes."""
        pose_list = PoseList()
        amount_of_poses = self.get_parameter("amount_of_poses").value

        for _ in range(amount_of_poses):
            pose = Pose()
            pose.position.x = random.uniform(-1.0, 1.0)
            pose.position.y = random.uniform(-1.0, 1.0)
            pose.position.z = random.uniform(0.0, 1.0)

            # Random orientation as quaternion
            pose.orientation.x = random.uniform(-1.0, 1.0)
            pose.orientation.y = random.uniform(-1.0, 1.0)
            pose.orientation.z = random.uniform(-1.0, 1.0)
            pose.orientation.w = random.uniform(0.0, 1.0)

            pose_list.poses.append(pose)

        return pose_list

    def validate_pose(self, current_pose: Pose, suggested_pose: Pose) -> bool:
        threshold = self.get_parameter("pose_threshold").value

        position_diff = (
            (current_pose.position.x - suggested_pose.position.x) ** 2
            + (current_pose.position.y - suggested_pose.position.y) ** 2
            + (current_pose.position.z - suggested_pose.position.z) ** 2
        ) ** 0.5

        orientation_diff = (
            (current_pose.orientation.x - suggested_pose.orientation.x) ** 2
            + (current_pose.orientation.y - suggested_pose.orientation.y) ** 2
            + (current_pose.orientation.z - suggested_pose.orientation.z) ** 2
            + (current_pose.orientation.w - suggested_pose.orientation.w) ** 2
        ) ** 0.5

        if position_diff < threshold and orientation_diff < threshold:
            self.get_logger().info(
                f"Pose is valid: Position diff {position_diff}, Orientation diff {orientation_diff}"
            )
            return True

        self.get_logger().warn(
            f"Pose is invalid: Position diff {position_diff}, Orientation diff {orientation_diff}"
        )

        return False

    def get_actual_pose(self):
        try:
            trans: TransformStamped = self.tf_buffer.lookup_transform(
                "base_link", "ee_link", rclpy.time.Time()
            )
            self.get_logger().info(f"Actual pose: {trans.transform}")
        except Exception as e:
            self.get_logger().warn(f"TF error: {str(e)}")

    def wait_until_pose_reached(self, desired_pose, timeout_sec=5.0):
        start_time = self.get_clock().now()
        while (self.get_clock().now() - start_time).nanoseconds / 1e9 < timeout_sec:
            actual_pose = self.get_actual_pose()

            if self.validate_pose(desired_pose, actual_pose):
                self.get_logger().info("Pose reached!")
                return True

            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().warn("Timeout: pose not reached.")
        return False


def main():
    rclpy.init()
    node = Validator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
