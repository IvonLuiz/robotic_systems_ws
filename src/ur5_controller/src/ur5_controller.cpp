#include "ur5_controller/ur5_controller.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include <memory>
#include <moveit/robot_state/robot_state.hpp>
#include <rclcpp/logging.hpp>

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

  if (this->_robot_state->setFromIK(
          this->_robot_model_group, target_pose,
          this->get_parameter("pose_action_timeout").as_double())) {
    RCLCPP_INFO(this->get_logger(), "IK solution found for target pose");
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  RCLCPP_ERROR(this->get_logger(), "No IK solution found for target pose");
  return rclcpp_action::GoalResponse::REJECT;
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
  this->declare_parameter("pose_action_timeout", 0.5);

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

void UR5Controller::init_robot_model() {
  auto shared_node = shared_from_this();

  RCLCPP_INFO(shared_node->get_logger(), "Initing robot model...");

  auto _robot_model_loader =
      std::make_shared<robot_model_loader::RobotModelLoader>(
          shared_node, "robot_description");

  _robot_model = _robot_model_loader->getModel();

  if (_robot_model) {
    RCLCPP_INFO(shared_node->get_logger(), "Robot model successfully loaded.");
  } else {
    RCLCPP_ERROR(shared_node->get_logger(), "Failed to load robot model.");
    return;
  }

  _robot_state = std::make_shared<moveit::core::RobotState>(_robot_model);

  _robot_state->setToDefaultValues();
  _robot_model_group = _robot_model->getJointModelGroup("manipulator");

  if (_robot_model_group) {
    RCLCPP_INFO(shared_node->get_logger(), "Joint model group loaded.");
  } else {
    RCLCPP_ERROR(shared_node->get_logger(),
                 "Failed to load joint model group.");
  }
}

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);

  // rclcpp::spin(std::make_shared<UR5Controller>());
  rclcpp::executors::MultiThreadedExecutor executor;
  auto node = std::make_shared<UR5Controller>();
  node->init_robot_model();
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
