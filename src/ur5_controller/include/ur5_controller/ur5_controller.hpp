#pragma once

#include "geometry_msgs/msg/pose.hpp"
#include "ur5_interface/action/move_to_pose.hpp"
#include <functional>
#include <memory>
#include <mutex>
#include <pluginlib/class_loader.hpp>
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
#include <trajectory_msgs/msg/joint_trajectory.hpp>

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

  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr
      _trajectory_pub;

public:
  UR5Controller();
};
