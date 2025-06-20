from rclpy.node import Node


class IKMotionPlanner(Node):

    def __init__(self):
        super().__init__("ik_motion_planner")
        self.get_logger().info("IKMotionPlanner node has been started.")

    def plan_motion(self, target_pose):
        # Placeholder for motion planning logic
        self.get_logger().info(f"Planning motion to target pose: {target_pose}")
        return True  # Indicating success


def main():
    print("Hi from ur5_motion_planner.")


if __name__ == "__main__":
    main()
