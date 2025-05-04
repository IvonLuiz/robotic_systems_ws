# UR Arm Pick and Place Project

This project provides a Dockerized environment for working with a Universal Robots (UR) arm in a ROS 2-based pick-and-place application. The container can be built for any supported ROS 2 distribution, as long as OSRF provides a base image and compatible packages for it.

## Prerequisites

- **Docker**: Ensure Docker is installed and running on your system. You can install Docker by following the instructions at [docker.com](https://www.docker.com/).
- **Supported ROS 2 Distributions**: This container has been tested with the following ROS 2 distributions:
  - `foxy`
  - `jazzy`
  
  Other distributions may work but are not guaranteed.

## Building the Docker Container

To build the Docker container, use the `build_container.sh` script. Replace `<ros2_distro>` with the desired ROS 2 distribution (e.g., `foxy` or `jazzy`).

```bash
cd Docker
bash build_container.sh <ros2_distro>
```

Once the container is built, you can run it using the run_container.sh script. Replace <ros2_distro> with the same ROS 2 distribution used during the build step.

```bash
bash run_container.sh <ros2_distro>
```