#!/bin/bash

ROS_DISTRO="jazzy"
IMAGE_NAME="ur_ros2_${ROS_DISTRO}"
CONTAINER_NAME="ur_ros2_dev_${ROS_DISTRO}"

# Get absolute paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

# Debug output
echo "--------------------------------------------------"
echo "Mounting configuration:"
echo "Host workspace: $WORKSPACE_DIR"
echo "Mounting host workspace → container /root/ur_ws/"
echo "--------------------------------------------------"

# X11 configuration
XSOCK="/tmp/.X11-unix"
XAUTH="/tmp/.docker.xauth"

# Create Xauthority
touch "$XAUTH"
xauth nlist "$DISPLAY" | sed -e 's/^..../ffff/' | xauth -f "$XAUTH" nmerge -

# Allow local connections
xhost +local:docker > /dev/null

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Enabling GPU acceleration."
    echo "Starting ROS2 ${ROS_DISTRO} container..."
    
    docker run -it --rm \
        --name "${CONTAINER_NAME}" \
        --env="DISPLAY" \
        --env="QT_X11_NO_MITSHM=1" \
        --env="XAUTHORITY=${XAUTH}" \
        --env="ROS_DISTRO=${ROS_DISTRO}" \
        --env="WORKSPACE=/root/ur_ws" \
        --volume="${XSOCK}:${XSOCK}:rw" \
        --volume="${XAUTH}:${XAUTH}:rw" \
        --volume="${WORKSPACE_DIR}:/root/ur_ws:rw" \
        --env="ROS_DOMAIN_ID=42" \
        --env="UROBOT_IP=192.168.56.101" \
        --privileged \
        --gpus all \
        -e NVIDIA_DRIVER_CAPABILITIES=all \
        "${IMAGE_NAME}"
else
    echo "No NVIDIA GPU detected. Using software rendering."
    echo "Starting ROS2 ${ROS_DISTRO} container..."
    
    docker run -it --rm \
        --name "${CONTAINER_NAME}" \
        --env="DISPLAY" \
        --env="QT_X11_NO_MITSHM=1" \
        --env="XAUTHORITY=${XAUTH}" \
        --env="ROS_DISTRO=${ROS_DISTRO}" \
        --env="WORKSPACE=/root/ur_ws" \
        --volume="${XSOCK}:${XSOCK}:rw" \
        --volume="${XAUTH}:${XAUTH}:rw" \
        --volume="${WORKSPACE_DIR}:/root/ur_ws:rw" \
        --env="ROS_DOMAIN_ID=42" \
        --env="UROBOT_IP=192.168.56.101" \
        --env="LIBGL_ALWAYS_SOFTWARE=1" \
        --privileged \
        "${IMAGE_NAME}"
fi

echo "Press Ctrl+D or type 'exit' to stop the container."

# Revoke access after run
xhost -local:docker > /dev/null