ARG ROS_DISTRO=jazzy
FROM ros:${ROS_DISTRO}-ros-base

ARG ROS_DISTRO
ENV ROS_DISTRO=${ROS_DISTRO}
ENV DEBIAN_FRONTEND=noninteractive
ENV WORKSPACE=/root/ur_ws
ENV GZ_VERSION=harmonic

SHELL ["/bin/bash", "-c"]

# Install dependencies and tools
RUN apt-get update && apt-get install -y \
    git wget nano python3-pip python3-rosdep \
    python3-colcon-common-extensions python3-vcstool lsb-release bash \
    && if [ "${ROS_DISTRO}" = "foxy" ]; then \
        apt-get install -y libyaml-cpp-dev; \
    fi \
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
    && rm -rf /var/lib/apt/lists/*

# Setup workspace
RUN mkdir -p ${WORKSPACE}/src
WORKDIR ${WORKSPACE}/src

# Special setup for Foxy
RUN if [ "${ROS_DISTRO}" = "foxy" ]; then \
      wget https://github.com/Kitware/CMake/releases/download/v3.22.6/cmake-3.22.6-linux-x86_64.sh && \
      chmod +x cmake-3.22.6-linux-x86_64.sh && \
      ./cmake-3.22.6-linux-x86_64.sh --skip-license --prefix=/usr/local && \
      rm cmake-3.22.6-linux-x86_64.sh; \
    fi

# Clone appropriate branch of UR ROS2 Driver
RUN git clone https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver.git && \
    cd Universal_Robots_ROS2_Driver && \
    if [ "${ROS_DISTRO}" = "foxy" ]; then \
        git checkout foxy; \
    elif [ "${ROS_DISTRO}" = "galactic" ]; then \
        git checkout galactic; \
    elif [ "${ROS_DISTRO}" = "humble" ]; then \
        git checkout humble; \
    elif [ "${ROS_DISTRO}" = "iron" ]; then \
        git checkout iron; \
    elif [ "${ROS_DISTRO}" = "jazzy" ] || [ "${ROS_DISTRO}" = "rolling" ]; then \
        git checkout main; \
    else \
        echo "Distro ${ROS_DISTRO} not officially supported. Using 'main' as default." && git checkout main; \
    fi

WORKDIR ${WORKSPACE}
RUN rosdep init && \
    rosdep update || echo "rosdep update completed with warnings"

# Install workspace dependencies with special handling for Foxy
RUN if [ "${ROS_DISTRO}" = "foxy" ]; then \
        rosdep install --from-paths src --ignore-src -y --rosdistro ${ROS_DISTRO} || \
        apt-get update && apt-get install -y libyaml-cpp-dev && rm -rf /var/lib/apt/lists/*; \
    else \
        rosdep install --from-paths src --ignore-src -y --rosdistro ${ROS_DISTRO} || true; \
    fi

# Build workspace with retry for Foxy
RUN . /opt/ros/${ROS_DISTRO}/setup.sh && \
    if [ "${ROS_DISTRO}" = "foxy" ]; then \
        colcon build --symlink-install --packages-skip ur_calibration || \
        (apt-get update && apt-get install -y libyaml-cpp-dev && colcon build --symlink-install); \
    else \
        colcon build --symlink-install; \
    fi

# Entry script
RUN echo '#!/bin/bash\n\
. /opt/ros/${ROS_DISTRO}/setup.sh\n\
. ${WORKSPACE}/install/setup.sh\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

# Copy the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]

WORKDIR ${WORKSPACE}
RUN . /opt/ros/${ROS_DISTRO}/setup.sh
