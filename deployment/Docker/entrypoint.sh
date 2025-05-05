#!/bin/bash

# docker_entrypoint.sh
set -e

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

# Execute the command passed to docker run
exec "$@"