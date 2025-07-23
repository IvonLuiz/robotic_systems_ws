import rclpy
import rclpy.duration
from rclpy.node import Node
from geometry_msgs.msg import Pose
import rclpy.time
from tf2_ros import TransformListener, Buffer
from rclpy.time import Time
from rclpy.duration import Duration
import pandas as pd
import numpy as np
import time
import threading
from rclpy.executors import MultiThreadedExecutor


class IKEvaluationNode(Node):
    def __init__(self, input_csv="evaluation_results.csv", output_csv="ik_evaluation_results.csv"):
        super().__init__('ik_evaluator')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pose_publisher = self.create_publisher(Pose, '/pose_list', 10)

        self.targets_df = pd.read_csv(input_csv)
        self.output_csv = output_csv
        self.results = []

        self.get_logger().info(f"📂 Loaded {len(self.targets_df)} episodes from {input_csv}.")

    def evaluate(self):
        for index, row in self.targets_df.iterrows():
            episode = int(row["episode"])
            target_pose = Pose()
            target_pose.position.x = row["target_x"]
            target_pose.position.y = row["target_y"]
            target_pose.position.z = row["target_z"]
            target_pose.orientation.w = 1.0 

            self.send_pose_and_record(episode, target_pose)
            time.sleep(1.0)

        df = pd.DataFrame(self.results)
        df.to_csv(self.output_csv, index=False)
        self.get_logger().info(f"✅ IK evaluation results saved to '{self.output_csv}'.")

    def send_pose_and_record(self, episode: int, pose: Pose, max_wait_sec: float = 30.0, convergence_threshold: float = 0.001):
        self.pose_publisher.publish(pose)
        self.get_logger().info(f"📤 Episode {episode}: Target pose published.")

        target_frame = "base_link"
        source_frame = "tool0"
        timeout_duration = Duration(seconds=1)

        tx, ty, tz = -pose.position.x, -pose.position.y, pose.position.z
        target_point = np.array([tx, ty, tz])

        final_position = None
        start_time = time.time()
        last_distances = []

        while time.time() - start_time < max_wait_sec:
            now = rclpy.time.Time()

            if self.tf_buffer.can_transform(target_frame, source_frame, now, timeout_duration):
                try:
                    transform = self.tf_buffer.lookup_transform(
                        target_frame,
                        source_frame,
                        now
                    )
                    pos = transform.transform.translation
                    ee_point = np.array([pos.x, pos.y, pos.z])

                    dist = np.linalg.norm(ee_point - target_point)
                    self.get_logger().info(
                        f"📏 Current distance: {dist:.4f} m | "
                        f"Target: ({tx:.3f}, {ty:.3f}, {tz:.3f}) | "
                        f"EE Pose: ({ee_point[0]:.3f}, {ee_point[1]:.3f}, {ee_point[2]:.3f})",
                        throttle_duration_sec=0.5
                    )

                    # Check normal convergence
                    if dist < convergence_threshold:
                        final_position = pos
                        break

                    # Track the last 3 distances
                    last_distances.append(dist)
                    if len(last_distances) > 3:
                        last_distances.pop(0)

                    # Check stagnation: low variance over 3 samples
                    if len(last_distances) == 3 and np.std(last_distances) < 1e-4:
                        self.get_logger().warn("⚠️ Distance stagnated over last 3 readings. Assuming motion stopped.")
                        final_position = pos
                        break

                except Exception as e:
                    self.get_logger().warn(f"❗ lookup_transform failed: {str(e)}")
            else:
                self.get_logger().warn(f"⏳ Waiting for transform: {target_frame} → {source_frame}...")

            time.sleep(0.5)

        if final_position is None:
            self.get_logger().error(f"⚠️ Episode {episode}: TF did not converge after {max_wait_sec:.1f}s.")
            return

        fx, fy, fz = final_position.x, final_position.y, final_position.z
        final_dist = np.linalg.norm([fx - tx, fy - ty, fz - tz])

        self.get_logger().info(f"🎯 Episode {episode}: Final distance = {final_dist:.4f} m")

        self.results.append({
            "episode": episode,
            "target_x": tx,
            "target_y": ty,
            "target_z": tz,
            "final_x": fx,
            "final_y": fy,
            "final_z": fz,
            "final_distance": final_dist,
            "steps": 1
        })


def main():
    rclpy.init()
    executor = MultiThreadedExecutor()
    node = IKEvaluationNode()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        node.evaluate()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
