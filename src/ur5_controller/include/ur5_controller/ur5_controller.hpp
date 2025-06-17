#pragma once

#include <memory>
#include <moveit/robot_model/robot_model.hpp>
#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/utilities.hpp>

class UR5Controller : public rclcpp::Node {
public:
  UR5Controller();
};
