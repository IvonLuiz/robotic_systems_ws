#include "ur5_controller/ur5_controller.hpp"

rclcpp_action::GoalResponse UR5Controller::pose_callback(
    const rclcpp_action::GoalUUID &uuid,
    std::shared_ptr<const ur5_interface::action::MoveToPose::Goal> goal) {
  // TODO: ADD THIS IMPLEMENTATION
  RCLCPP_INFO(this->get_logger(), "Received goal request");
  (void)uuid;
  (void)goal;

  {
    std::lock_guard<std::mutex> guard(this->_mutex);
    if (this->_current_goal && this->_current_goal->is_active())
      return rclcpp_action::GoalResponse::REJECT;
  }
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse UR5Controller::pose_cancel(
    const std::shared_ptr<
        rclcpp_action::ServerGoalHandle<ur5_interface::action::MoveToPose>>
        goal_handle) {
  // TODO: ADD THIS IMPLEMENTATION
  RCLCPP_INFO(this->get_logger(), "Received request to cancel goal");
  (void)goal_handle;
  return rclcpp_action::CancelResponse::ACCEPT;
}

void UR5Controller::handle_pose_callback(
    const std::shared_ptr<
        rclcpp_action::ServerGoalHandle<ur5_interface::action::MoveToPose>>
        goal_handle) {
  // TODO: ADD THIS IMPLEMENTATION
  {
    std::lock_guard<std::mutex> guard(this->_mutex);
    this->_current_goal = goal_handle;
  }
  RCLCPP_INFO(this->get_logger(), "Executing goal...");
}

UR5Controller::UR5Controller() : rclcpp::Node("UR5_CONTROLLER") {
  this->declare_parameter("pose_action_server_name", "pose_server");

  RCLCPP_INFO(this->get_logger(), "Iniciando UR5Controller Node...");

  this->_cb_group =
      this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

  this->_server =
      rclcpp_action::create_server<ur5_interface::action::MoveToPose>(
          this, this->get_parameter("pose_action_server_name").as_string(),
          std::bind(&UR5Controller::pose_callback, this, std::placeholders::_1,
                    std::placeholders::_2),
          std::bind(&UR5Controller::pose_cancel, this, std::placeholders::_1),
          std::bind(&UR5Controller::handle_pose_callback, this,
                    std::placeholders::_1),
          rcl_action_server_get_default_options(), this->_cb_group);
}

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);

  // rclcpp::spin(std::make_shared<UR5Controller>());
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(std::make_shared<UR5Controller>());
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
