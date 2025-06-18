#pragma once

#include "ur5_interface/action/move_to_pose.hpp"
#include <functional>
#include <memory>
#include <moveit/kinematics_base/kinematics_base.hpp>
#include <moveit/robot_model/joint_model_group.hpp>
#include <moveit/robot_model/robot_model.hpp>
#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <mutex>
#include <rcl_action/action_server.h>
#include <rclcpp/callback_group.hpp>
#include <rclcpp/executor.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/utilities.hpp>
#include <rclcpp_action/create_server.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <rclcpp_action/server.hpp>
#include <rclcpp_action/server_goal_handle.hpp>
#include <rclcpp_action/types.hpp>
#include <string>

static const std::string PLANNING_GROUP = "ur_manipulator";

class UR5Controller : public rclcpp::Node {
private:
  rclcpp_action::Server<ur5_interface::action::MoveToPose>::SharedPtr _server;
  rclcpp::CallbackGroup::SharedPtr _cb_group;
  std::mutex _mutex;
  std::shared_ptr<
      rclcpp_action::ServerGoalHandle<ur5_interface::action::MoveToPose>>
      _current_goal;

  rclcpp_action::GoalResponse pose_callback(
      const rclcpp_action::GoalUUID &uuid,
      std::shared_ptr<const ur5_interface::action::MoveToPose::Goal> goal);

  rclcpp_action::CancelResponse pose_cancel(
      const std::shared_ptr<
          rclcpp_action::ServerGoalHandle<ur5_interface::action::MoveToPose>>
          goal_handle);

  void handle_pose_callback(
      const std::shared_ptr<
          rclcpp_action::ServerGoalHandle<ur5_interface::action::MoveToPose>>
          goal_handle);

  moveit::core::RobotModelPtr _robot_model;
  std::shared_ptr<moveit::core::RobotState> _robot_state;
  moveit::core::JointModelGroup *_robot_model_group;

public:
  UR5Controller();
  void init_robot_model();
};
