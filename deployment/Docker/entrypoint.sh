#!/bin/bash

# docker_entrypoint.sh
set -e

echo "Starting ROS 2 ${ROS_DISTRO} container..."

# Source ROS setup
if [ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then
    source "/opt/ros/${ROS_DISTRO}/setup.bash"
    echo "✓ Sourced ROS ${ROS_DISTRO} setup"
else
    echo "Error: ROS setup.bash not found for ${ROS_DISTRO}"
    exit 1
fi

# Source workspace setup if exists
if [ -f "${WORKSPACE}/install/setup.bash" ]; then
    source "${WORKSPACE}/install/setup.bash"
    echo "✓ Sourced workspace setup"
fi

echo "Installing workspace dependencies..."
cd ${WORKSPACE}
rosdep install --from-paths src --rosdistro ${ROS_DISTRO}

# Change to workspace directory
cd ${WORKSPACE}

echo "Container setup complete. Ready to use!"
echo "================================================="

# Execute the command passed to docker run
exec "$@"