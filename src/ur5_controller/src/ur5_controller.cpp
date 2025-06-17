#include "ur5_controller/ur5_controller.hpp"

UR5Controller::UR5Controller() : rclcpp::Node("UR5_CONTROLLER") {}

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);

  rclcpp::spin(std::make_shared<UR5Controller>());

  rclcpp::shutdown();
  return 0;
}
