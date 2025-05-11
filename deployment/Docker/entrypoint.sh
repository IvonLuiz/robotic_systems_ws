#!/bin/bash

# docker_entrypoint.sh
set -e

# move

# Source ROS setup
if [ -f "/opt/ros/${ROS_DISTRO}/setup.sh" ]; then
    source "/opt/ros/${ROS_DISTRO}/setup.sh"
else
    echo "Error: ROS setup.sh not found for ${ROS_DISTRO}"
    exit 1
fi

# Source workspace setup if exists
if [ -f "${WORKSPACE}/install/setup.sh" ]; then
    source "${WORKSPACE}/install/setup.sh"
fi

# Clone and build Universal Robots Client Library if not already present
if [ ! -d "/root/ur_ws/src/ur_client_library" ]; then
    echo "Cloning Universal Robots Client Library..."
    cd /root/ur_ws/src
    git clone https://github.com/UniversalRobots/Universal_Robots_Client_Library ur_client_library
    cd ur_client_library
    mkdir build && cd build
    cmake ..
    make
    make install
    sudo apt install ros-${ROS_DISTRO}-ur-client-library
fi

cd ${WORKSPACE}/src

# Execute the command passed to docker run
exec "$@"