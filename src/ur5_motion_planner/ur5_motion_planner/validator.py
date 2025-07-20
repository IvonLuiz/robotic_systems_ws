from rclpy.node import Node
from ur5_interfaces.msg import PoseList
from geometry_msgs.msg import Pose, TransformStamped
from tf2_ros import TransformListener, Buffer
from rclpy.duration import Duration
from pathlib import Path
import rclpy
import random
import numpy as np


class Validator(Node):
    test_topic: rclpy.publisher.Publisher

    def __init__(self):
        super().__init__("motion_planner_validator")
        self.get_logger().info("Motion Planner Validator Node has been started.")

        self.declare_parameter("random_seed", 42)
        self.declare_parameter("pose_list_topic", "/pose_list")
        self.declare_parameter("pose_threshold", 0.05)
        self.declare_parameter("num_poses", 10)
        self.declare_parameter("sleep_duration", 30)
        self.declare_parameter(
            "filename",
            f"./results/validation_results_{self.get_clock().now().nanoseconds}.txt",
        )
        random.seed(self.get_parameter("random_seed").value)

        self.test_topic = self.create_publisher(
            Pose, self.get_parameter("pose_list_topic").value, 10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer_validator = self.create_timer(2.0, self.validate)

    def validate(self):
        if not self.wait_for_tf(timeout_sec=5.0):
            self.get_logger().warn("TF not available, cannot validate poses.")
            return

        self.timer_validator.cancel()

        self.get_logger().info("Validating...")

        sleep_duration = self.get_parameter("sleep_duration").value

        self.get_logger().info("Motion Planner Validator Node has been started.")
        passed = 0
        for i in range(self.get_parameter("num_poses").value):
            pose = self.generate_random_pose()
            self.test_topic.publish(pose)
            self.get_logger().info(
                f"Pose {i}: Position ({pose.position.x}, {pose.position.y}, {pose.position.z}), "
                f"Orientation ({pose.orientation.x}, {pose.orientation.y}, {pose.orientation.z}, {pose.orientation.w})"
            )

            if self.wait_until_pose_reached(
                desired_pose=pose, timeout_sec=sleep_duration
            ):
                self.get_logger().info(f"Pose {i} reached successfully.")
                passed += 1
            else:
                self.get_logger().warn(f"Pose {i} not reached within timeout.")

        # End node
        self.get_logger().info("Validation completed.")
        self.get_logger().info(
            f"Total poses passed: {passed}/{self.get_parameter('num_poses').value}"
        )
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

    def generate_random_pose(self) -> Pose:
        pose = Pose()

        # Generate a random position inside a sphere shell
        min_radius = 0.3  # avoid unreachable tiny poses
        max_radius = 0.85  # UR5 max reach

        # Uniform sampling in spherical coordinates
        radius = random.uniform(min_radius, max_radius)
        theta = random.uniform(0, 2 * np.pi)  # azimuthal angle
        phi = random.uniform(
            0, np.pi / 2
        )  # polar angle (0 = up, pi/2 = XY plane), limit to upper hemisphere

        # Convert to Cartesian
        pose.position.x = radius * np.sin(phi) * np.cos(theta)
        pose.position.y = radius * np.sin(phi) * np.sin(theta)
        pose.position.z = radius * np.cos(phi)

        # Normalize quaternion
        qx, qy, qz, qw = np.random.uniform(-1.0, 1.0, 4)
        norm = (qx**2 + qy**2 + qz**2 + qw**2) ** 0.5
        pose.orientation.x = qx / norm
        pose.orientation.y = qy / norm
        pose.orientation.z = qz / norm
        pose.orientation.w = qw / norm

        return pose

    def validate_pose(self, current_pose: Pose, suggested_pose: Pose) -> bool:
        threshold = self.get_parameter("pose_threshold").value

        position_diff = (
            (current_pose.position.x - suggested_pose.position.x) ** 2
            + (current_pose.position.y - suggested_pose.position.y) ** 2
            + (current_pose.position.z - suggested_pose.position.z) ** 2
        ) ** 0.5

        self.get_logger().info(
            f"Validating pose: Position diff {position_diff}"
        )

        if position_diff < threshold:
            self.get_logger().info(
                f"Pose is valid: Position diff {position_diff}"
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
                pose.position.x = (
                    -pose.position.x
                )  # Invert X for RViz positive quadrant
                pose.position.y = (
                    -pose.position.y
                )  # Invert Y for RViz positive quadrant
                pose.orientation = trans.transform.rotation
                pose.orientation.z = (
                    pose.orientation.z - 1
                )  # Invert Z for RViz positive quadrant
                return pose
            else:
                self.get_logger().warn(
                    f"Transform from {source_frame} to {target_frame} not available yet."
                )
                return None
        except Exception as e:
            self.get_logger().warn(f"TF error: {str(e)}")
            return None

    def wait_until_pose_reached(self, desired_pose, timeout_sec=20.0):
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
