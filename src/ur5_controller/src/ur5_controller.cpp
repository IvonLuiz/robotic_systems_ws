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

  if (this->_robot_state->setFromIK(
          this->_robot_model_group, target_pose,
          this->get_parameter("pose_action_timeout").as_double())) {
    RCLCPP_INFO(this->get_logger(), "IK solution found for target pose");

    this->_robot_state->copyJointGroupPositions(this->_robot_model_group,
                                                this->_joint_values);

    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  RCLCPP_ERROR(this->get_logger(), "No IK solution found for target pose");
  return rclcpp_action::GoalResponse::REJECT;
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

  planning_scene::PlanningScene planning_scene(this->_robot_model);
  collision_detection::CollisionRequest collision_request;
  collision_detection::CollisionResult collision_result;
  planning_scene.checkCollision(collision_request, collision_result);

  if (collision_result.collision) {
    RCLCPP_ERROR(this->get_logger(), "Collision detected in planning scene");
    result->success = false;
    goal_handle->abort(result);
    return;
  }

  trajectory_msgs::msg::JointTrajectory trajectory;
  trajectory.joint_names = this->_robot_model_group->getActiveJointModelNames();

  trajectory_msgs::msg::JointTrajectoryPoint point;
  point.positions = this->_joint_values;
  point.time_from_start = rclcpp::Duration::from_seconds(
      this->get_parameter("robot_time_from_start").as_double());
  trajectory.points.push_back(point);

  this->_trajectory_pub->publish(trajectory);
  RCLCPP_INFO(this->get_logger(), "Trajectory published successfully");

  const auto start_time = std::chrono::steady_clock::now();
  const auto duration = std::chrono::seconds(2);
  rclcpp::Rate rate(10);

  while (rclcpp::ok() &&
         (std::chrono::steady_clock::now() - start_time) < duration) {
    if (goal_handle->is_canceling()) {
      RCLCPP_WARN(this->get_logger(), "Goal canceled by client");

      result->success = false;
      goal_handle->canceled(result);
      return;
    }

    feedback->progress_percentage =
        (std::chrono::steady_clock::now() - start_time).count() /
        static_cast<float>(duration.count());
    goal_handle->publish_feedback(feedback);

    rate.sleep();
  }

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
