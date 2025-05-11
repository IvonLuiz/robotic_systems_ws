# UR Arm Pick and Place Project

This project provides a Dockerized environment for working with a Universal Robots (UR) arm in a ROS 2-based pick-and-place application. The container can be built for any supported ROS 2 distribution, as long as OSRF provides a base image and compatible packages for it.

## Prerequisites

- **Docker**: Ensure Docker is installed and running on your system. You can install Docker by following the instructions at [docker.com](https://www.docker.com/).
- **Supported ROS 2 Distributions**: This container has been tested with the following ROS 2 distributions:
  - `foxy`
  - `jazzy`
  
  Other distributions may work but are not guaranteed.

## Development
### Setup URSim for simulation with docker

The documentation for this steps are on: [reference](https://docs.universal-robots.com/Universal_Robots_ROS2_Documentation/doc/ur_client_library/doc/setup/ursim_docker.html).

In an separated secondary terminal, we will create a dedicated docker network.
```bash
docker network create --subnet=192.168.56.0/24 --driver bridge ursim-net
```

Start URSim Container
```bash
docker run --rm -it \
  --name ursim \
  --net ursim-net \
  --ip 192.168.56.101 \
  -v ${HOME}/.ursim/programs:/ursim/programs \
  -v ${HOME}/.ursim/urcaps:/urcaps \
  universalrobots/ursim_e-series
```

### Building the Docker Container for development

To build the Docker container, use the `build_container.sh` script. Replace `<ros2_distro>` with the desired ROS 2 distribution (e.g., `foxy` or `jazzy`).

```bash
cd deployment
bash build_container.sh <ros2_distro>
```

Once the container is built, you can run it using the run_container.sh script. Replace <ros2_distro> with the same ROS 2 distribution used during the build step.

```bash
bash run_container.sh <ros2_distro>
```

This script will start docker and also clone and install the [ur_client_library](https://docs.universal-robots.com/Universal_Robots_ROS2_Documentation/doc/ur_client_library/doc/installation.html) located in this [repo](https://github.com/UniversalRobots/Universal_Robots_Client_Library) if it is not already in the src/ directory.

