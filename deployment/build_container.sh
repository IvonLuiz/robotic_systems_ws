#!/bin/bash

sudo chown -R $USER:$USER . 

ROS_DISTRO="jazzy"
IMAGE_NAME="ur_ros2_${ROS_DISTRO}"
WORKSPACE_DIR=$(pwd)

if ! command -v docker &> /dev/null; then
    echo "Docker is not installed."
    exit 1
fi

cd Docker/
if [ ! -f "Dockerfile" ]; then
    echo "Dockerfile not found in current directory."
    exit 1
fi

echo "Building Docker image for ROS 2 ${ROS_DISTRO}..."

docker build -t ${IMAGE_NAME} . \
    --build-arg UID=$(id -u) \
    --build-arg GID=$(id -g) \
    --build-arg USERNAME=$(whoami) \
    --build-arg ROS_DISTRO=${ROS_DISTRO}

if [ $? -eq 0 ]; then
    echo ""
    echo "Docker image built successfully!"
    echo "Image name: $IMAGE_NAME"
    echo ""
    echo "To run the container, execute:"
    echo "bash run_container.sh"
    echo "To open in other terminals:"
    echo "docker exec -it ur_ros2_dev_jazzy bash"
else
    echo "Docker build failed."
    exit 1
fi
