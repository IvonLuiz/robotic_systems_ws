cmake_minimum_required(VERSION 3.8)
project(ur5_controller)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(PACKAGES
  ament_cmake
  rclcpp
  rclcpp_action
  ur5_interface
  pluginlib
  trajectory_msgs
  geometry_msgs
)

foreach(pkg IN LISTS PACKAGES)
  find_package(${pkg} REQUIRED)
endforeach()

set(COMPONENTS
  Boost
)

foreach(pkg IN LISTS COMPONENTS)
  find_package(${pkg} REQUIRED COMPONENTS system)
endforeach()

include_directories(
  include
)

# List of nodes: alternating name and source
set(NODES
  ur5_gazebo_controller src/ur5_gazebo_controller.cpp
)

# Macro to define node executables
macro(add_node name)
  add_executable(${name} ${ARGN})
  ament_target_dependencies(${name} ${PACKAGES} ${COMPONENTS})
  install(TARGETS ${name} DESTINATION lib/${PROJECT_NAME})
endmacro()

# Add each node using the macro
list(LENGTH NODES NODES_LENGTH)
math(EXPR LAST_INDEX "${NODES_LENGTH} - 2")

foreach(i RANGE 0 ${LAST_INDEX} 2)
  list(GET NODES ${i} node_name)
  math(EXPR src_index "${i} + 1")
  list(GET NODES ${src_index} node_src)
  add_node(${node_name} ${node_src})
endforeach()

# Install include directory
install(
  DIRECTORY include/
  DESTINATION include
)

# Linting and testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

