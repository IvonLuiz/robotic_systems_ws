#!/bin/bash

# Check if ROS distribution is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <ros2_distro>"
    echo "Supported distributions: foxy, galactic, humble, iron, jazzy, rolling"
    exit 1
fi

ROS_DISTRO=$1
IMAGE_NAME="ur_ros2_${ROS_DISTRO}"
CONTAINER_NAME="ur_ros2_dev_${ROS_DISTRO}"
WORKSPACE_DIR=$(pwd)

# X11 configuration
XSOCK="/tmp/.X11-unix"
XAUTH="/tmp/.docker.xauth"

# Create Xauthority file
touch "$XAUTH"
xauth nlist "$DISPLAY" | sed -e 's/^..../ffff/' | xauth -f "$XAUTH" nmerge -

# Allow local connections
xhost +local:docker > /dev/null

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_FLAGS="--gpus all -e NVIDIA_DRIVER_CAPABILITIES=all"
    echo "NVIDIA GPU detected. Enabling GPU acceleration."
else
    GPU_FLAGS="-e LIBGL_ALWAYS_SOFTWARE=1"
    echo "No NVIDIA GPU detected. Using software rendering."
fi

echo "Starting ROS2 ${ROS_DISTRO} container..."
echo "Press Ctrl+D or type 'exit' to stop the container."

# Run Docker container
docker run -it --rm \
    --name "${CONTAINER_NAME}" \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --env="XAUTHORITY=${XAUTH}" \
    --env="ROS_DISTRO=${ROS_DISTRO}" \
    --env="WORKSPACE=/root/ur_ws" \
    --volume="${XSOCK}:${XSOCK}:rw" \
    --volume="${XAUTH}:${XAUTH}:rw" \
    --volume="${WORKSPACE_DIR}:/root/ur_ws/src/ur_pick_and_place:rw" \
    --net=host \
    --privileged \
    ${GPU_FLAGS} \
    "${IMAGE_NAME}"

# Revoke access after run
xhost -local:docker > /dev/null