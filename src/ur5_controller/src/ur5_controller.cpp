#include "ur5_controller/ur5_controller.hpp"

rclcpp_action::GoalResponse UR5Controller::pose_callback(
    const rclcpp_action::GoalUUID &uuid,
    std::shared_ptr<const ur5_interface::action::MoveToPose::Goal> goal) {
  RCLCPP_INFO(this->get_logger(), "Received goal request %s",
              std::string(uuid.begin(), uuid.end()).c_str());

  {
    std::lock_guard<std::mutex> guard(this->_mutex);
    if (this->_current_goal && this->_current_goal->is_active()) {
      return rclcpp_action::GoalResponse::REJECT;
    }
  }

  const geometry_msgs::msg::Pose &target_pose = goal->target_pose.pose;

  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse UR5Controller::pose_cancel(
    const std::shared_ptr<
        rclcpp_action::ServerGoalHandle<ur5_interface::action::MoveToPose>>
        goal_handle) {
  RCLCPP_INFO(this->get_logger(), "Received request to cancel goal");

  if (!goal_handle) {
    RCLCPP_ERROR(this->get_logger(), "Goal handle is null");
    return rclcpp_action::CancelResponse::REJECT;
  }

  if (goal_handle->is_canceling()) {
    RCLCPP_INFO(this->get_logger(), "Goal is already being canceled");
    return rclcpp_action::CancelResponse::REJECT;
  }

  {
    std::lock_guard<std::mutex> guard(this->_mutex);
    if (this->_current_goal && this->_current_goal->is_active()) {
      this->_current_goal.reset();
    }
  }

  return rclcpp_action::CancelResponse::ACCEPT;
}

void UR5Controller::handle_pose_callback(
    const std::shared_ptr<
        rclcpp_action::ServerGoalHandle<ur5_interface::action::MoveToPose>>
        goal_handle) {
  {
    std::lock_guard<std::mutex> guard(this->_mutex);
    this->_current_goal = goal_handle;
  }

  auto result = std::make_shared<ur5_interface::action::MoveToPose::Result>();
  auto feedback =
      std::make_shared<ur5_interface::action::MoveToPose::Feedback>();

  if (!goal_handle) {
    RCLCPP_ERROR(this->get_logger(), "Goal handle is null");
    result->success = false;
    goal_handle->abort(result);
    return;
  }

  RCLCPP_INFO(this->get_logger(), "Executing goal...");

  result->success = true;
  goal_handle->succeed(result);
  {
    std::lock_guard<std::mutex> guard(this->_mutex);
    this->_current_goal.reset();
  }
}

UR5Controller::UR5Controller() : rclcpp::Node("UR5_CONTROLLER") {
  this->declare_parameter("pose_action_server_name", "pose_server");
  this->declare_parameter("pose_action_timeout", 0.5);
  this->declare_parameter("joint_trajectory_topic",
                          "/joint_trajectory_controller/joint_trajectory");
  this->declare_parameter("robot_sleep_duration_in_seconds", 2);
  this->declare_parameter("robot_time_from_start", 2.0);

  RCLCPP_INFO(this->get_logger(), "Iniciando UR5Controller Node...");

  this->_cb_group =
      this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

  RCLCPP_INFO(this->get_logger(), "Creating action server...");
  this->_server =
      rclcpp_action::create_server<ur5_interface::action::MoveToPose>(
          this, this->get_parameter("pose_action_server_name").as_string(),
          std::bind(&UR5Controller::pose_callback, this, std::placeholders::_1,
                    std::placeholders::_2),
          std::bind(&UR5Controller::pose_cancel, this, std::placeholders::_1),
          std::bind(&UR5Controller::handle_pose_callback, this,
                    std::placeholders::_1),
          rcl_action_server_get_default_options(), this->_cb_group);

  RCLCPP_INFO(this->get_logger(), "Action server created successfully.");
  RCLCPP_INFO(this->get_logger(), "Initializing joint trajectory topics...");
  this->_trajectory_pub =
      this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
          this->get_parameter("joint_trajectory_topic").as_string(), 10);
}

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);

  // rclcpp::spin(std::make_shared<UR5Controller>());
  rclcpp::executors::MultiThreadedExecutor executor;
  auto node = std::make_shared<UR5Controller>();
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
