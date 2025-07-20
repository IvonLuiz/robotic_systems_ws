FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV WORKSPACE=/root/ur_ws

SHELL ["/bin/bash", "-c"]

# Install dependencies and tools
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install required tools and ROS 2 dependencies
RUN apt-get update && apt-get install -y \
    locales \
    curl \
    gnupg2 \
    lsb-release \
    build-essential \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add - && \
    echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2.list


RUN apt-get update && apt-get install -y \
    ros-dev-tools \
    ros-jazzy-desktop \
    ros-jazzy-moveit \
    ros-jazzy-ur-* \
    ros-jazzy-xacro \
    ros-jazzy-tf-transformations \
    && rm -rf /var/lib/apt/lists/*

WORKDIR ${WORKSPACE}

COPY . .


# Entry script
RUN echo '#!/bin/bash\n\
. /opt/ros/jazzy/setup.sh\n\
. ${WORKSPACE}/install/setup.sh\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

RUN echo "source /opt/ros/jazzy/setup.sh" >> ~/.bashrc
RUN echo "source ${WORKSPACE}/install/setup.sh" >> ~/.bashrc

SHELL ["/bin/bash", "-c"]

RUN rosdep init && \
    rosdep update || echo "rosdep update completed with warnings"

RUN . /opt/ros/jazzy/setup.sh && \
    colcon build

RUN pip3 install numpy scipy --break-system-packages

# Set the entrypoint
ENTRYPOINT ["bash"]
