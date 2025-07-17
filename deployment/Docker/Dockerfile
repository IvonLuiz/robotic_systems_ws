ARG ROS_DISTRO=jazzy
FROM ros:${ROS_DISTRO}-ros-base

ENV ROS_DISTRO=${ROS_DISTRO}
ENV DEBIAN_FRONTEND=noninteractive
ENV WORKSPACE=/root/ur_ws

SHELL ["/bin/bash", "-c"]

# Install dependencies and tools
RUN apt-get update && apt-get install -y \
    git wget nano python3-pip python3-rosdep cmake make \
    python3-colcon-common-extensions python3-vcstool lsb-release bash \
    && rm -rf /var/lib/apt/lists/*

# Install ROS 2 packages
RUN apt-get update && apt-get install -y \
    ros-${ROS_DISTRO}-moveit \
    ros-${ROS_DISTRO}-ur-description \
    ros-${ROS_DISTRO}-ur-msgs \
    ros-${ROS_DISTRO}-ur-client-library \
    ros-${ROS_DISTRO}-ur-robot-driver \
    ros-${ROS_DISTRO}-ur-moveit-config \
    ros-${ROS_DISTRO}-rviz2 \
    ros-${ROS_DISTRO}-xacro \
    ros-${ROS_DISTRO}-ur-* \
    && rm -rf /var/lib/apt/lists/*

# Setup workspace
WORKDIR ${WORKSPACE}

COPY . .

RUN rosdep init && \
    rosdep update || echo "rosdep update completed with warnings"

# Entry script
RUN echo '#!/bin/bash\n\
. /opt/ros/${ROS_DISTRO}/setup.sh\n\
. ${WORKSPACE}/install/setup.sh\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

RUN . /opt/ros/jazzy/setup.sh && \
    colcon build

RUN echo "source /opt/ros/jazzy/setup.sh" >> ~/.bashrc
RUN echo "source ${WORKSPACE}/install/setup.sh" >> ~/.bashrc

RUN pip3 install numpy scipy --break-system-packages

# Set the entrypoint
ENTRYPOINT ["bash"]
