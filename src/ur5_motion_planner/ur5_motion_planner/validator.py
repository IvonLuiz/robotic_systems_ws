from rclpy.node import Node
from ur5_interfaces.msg import PoseList
from geometry_msgs.msg import Pose, TransformStamped
from tf2_ros import TransformListener, Buffer
from rclpy.duration import Duration
from pathlib import Path
import rclpy
import random


class Validator(Node):
    random_pose_list: PoseList
    test_topic: rclpy.publisher.Publisher

    def __init__(self):
        super().__init__("motion_planner_validator")
        self.get_logger().info("Motion Planner Validator Node has been started.")

        self.declare_parameter("random_seed", 42)
        self.declare_parameter("pose_list_topic", "/pose_list_ik")
        self.declare_parameter("amount_of_poses", 10)
        self.declare_parameter("pose_threshold", 0.1)
        self.declare_parameter("sleep_duration", 5)
        self.declare_parameter(
            "filename", f"./results/validation_results_{self.get_clock().now().nanoseconds}.txt"
        )
        random.seed(self.get_parameter("random_seed").value)

        self.random_pose_list = self.generate_random_pose_list()

        self.test_topic = self.create_publisher(
            PoseList, self.get_parameter("pose_list_topic").value, 10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer_validator = self.create_timer(2.0, self.validate)

    def validate(self):
        if not self.wait_for_tf(timeout_sec=5.0):
            self.get_logger().warn("TF not available, cannot validate poses.")
            return

        self.timer_validator.cancel()
        self.test_topic.publish(self.random_pose_list)

        self.get_logger().info(
            f"Published {len(self.random_pose_list.poses)} random poses to topic: {self.get_parameter('pose_list_topic').value}"
        )
        self.get_logger().info("Validating...")

        sleep_duration = self.get_parameter("sleep_duration").value

        self.get_logger().info("Motion Planner Validator Node has been started.")
        for i, pose in enumerate(self.random_pose_list.poses):
            self.get_logger().info(
                f"Pose {i}: Position ({pose.position.x}, {pose.position.y}, {pose.position.z}), "
                f"Orientation ({pose.orientation.x}, {pose.orientation.y}, {pose.orientation.z}, {pose.orientation.w})"
            )

            if self.wait_until_pose_reached(
                desired_pose=pose, timeout_sec=sleep_duration
            ):
                self.get_logger().info(f"Pose {i} reached successfully.")
            else:
                self.get_logger().warn(f"Pose {i} not reached within timeout.")

        # End node
        self.get_logger().info("Validation completed.")
        self.get_logger().info("Motion Planner Validator Node has been stopped.")
        self.destroy_node()
        rclpy.shutdown()

    def wait_for_tf(self, timeout_sec=5.0):
        start = self.get_clock().now()
        duration = Duration(seconds=timeout_sec)

        while (self.get_clock().now() - start) < duration:
            if self.tf_buffer.can_transform(
                "base_link", "wrist_3_link", rclpy.time.Time()
            ):
                self.get_logger().info("TF available. Proceeding.")
                return True
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().warn("TF not available after timeout.")
        return False

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

        return False

    def get_actual_pose(self):
        target_frame = "base_link"
        source_frame = "wrist_3_link"
        timeout = Duration(seconds=3.0)

        try:
            if self.tf_buffer.can_transform(
                target_frame, source_frame, rclpy.time.Time(), timeout
            ):
                trans: TransformStamped = self.tf_buffer.lookup_transform(
                    target_frame, source_frame, rclpy.time.Time()
                )

                # Convert Transform to Pose
                pose = Pose()
                pose.position = trans.transform.translation
                pose.orientation = trans.transform.rotation
                return pose
            else:
                self.get_logger().warn(
                    f"Transform from {source_frame} to {target_frame} not available yet."
                )
                return None
        except Exception as e:
            self.get_logger().warn(f"TF error: {str(e)}")
            return None

    def wait_until_pose_reached(self, desired_pose, timeout_sec=5.0):
        start_time = self.get_clock().now()
        last_check_time = self.get_clock().now()
        filename = self.get_parameter("filename").value

        while (self.get_clock().now() - start_time).nanoseconds / 1e9 < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)  # Non-blocking wait

            now = self.get_clock().now()
            elapsed_since_last_check = (now - last_check_time).nanoseconds / 1e9

            if elapsed_since_last_check < 0.5:
                continue  # Wait until 1 second has passed before checking again

            last_check_time = now  # Update check time

            actual_pose = self.get_actual_pose()
            if actual_pose is None:
                self.get_logger().warn("Failed to get actual pose.")
                continue

            self.get_logger().info(f"Actual pose: {actual_pose}")
            self.get_logger().info(f"Desired pose: {desired_pose}")

            if self.validate_pose(desired_pose, actual_pose):
                self.get_logger().info("Pose reached!")
                self.write_results_to_file(
                    filename,
                    f"Pose reached: Position ({actual_pose.position.x}, {actual_pose.position.y}, {actual_pose.position.z}), ",
                )
                return True

        self.get_logger().warn("Timeout: pose not reached.")
        self.write_results_to_file(
            filename,
            f"Pose not reached within timeout: Position ({desired_pose.position.x}, {desired_pose.position.y}, {desired_pose.position.z}), Actual Position ({actual_pose.position.x if actual_pose else 'N/A'}, {actual_pose.position.y if actual_pose else 'N/A'}, {actual_pose.position.z if actual_pose else 'N/A'})",
        )
        return False

    def write_results_to_file(self, filename: str, results: str):
        output_file = Path(filename)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        with open(output_file, "a") as file:
            file.write(results + "\n")


def main():
    rclpy.init()
    node = Validator()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
