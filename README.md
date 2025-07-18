# UR Arm Pick and Place Project

This project provides a Dockerized environment for working with a Universal Robots (UR) arm in a ROS 2-based pick-and-place application. The container can be built for any supported ROS 2 distribution, as long as OSRF provides a base image and compatible packages for it.

## Prerequisites

- **Docker**: Ensure Docker is installed and running on your system. You can install Docker by following the instructions at [docker.com](https://www.docker.com/).
- **Supported ROS 2 Distributions**: This container has been tested with the following ROS 2 distributions:
  - `jazzy`

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

to open other terminal:
```bash
docker exec -it ur_ros2_dev_jazzy bash
```

### Building the Docker Container for development

To build the Docker container, use the `build_container.sh` script.

```bash
cd deployment
bash build_container.sh
```

Once the container is built, you can run it using the run_container.sh script.

```bash
bash run_container.sh
```

to open other terminal:
```bash
docker exec -it ur_ros2_dev_jazzy bash
```

## How to build and run code:

### With docker container:

Inside the container:

```bash
colcon build --base-path src
source install/setup.bash 
```

### Local Setup

```bash
rosdep init
rosdep update
rosdep install --from-paths src/
```

### Building the Workspace

```bash
python3 -m venv env
source env/bin/activate
colcon build --base-path src
```

## Working

## New terminal setup
Remember to always source when using a new terminal (on container or not):

```bash
source install/setup.bash 
```

### Inverse kinematics
Our IK motion planner implementation can be run by launching the simulation and the script:

In one terminal:
```bash
ros2 launch gazebo_control ur_sim_control.launch.py
```

In a second separate terminal
```bash
ros2 run ur5_motion_planner ik_motion_planner
```

to send a pose modify the example:
```bash
ros2 topic pub /pose_list ur5_interfaces/msg/PoseList "{
  poses: [
    position: {x: 0.4, y: 0.2, z: 0.3}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
  ]
}"
```

### Machine Learning
#### Dataset generation

To generate a certain dataset size, we will also start the simulation and the script. The dataset_generator_node will make random movinments to the robot and save the end effector position and the angles that made the moviments.

In one terminal:
```bash
ros2 launch gazebo_control ur_sim_control.launch.py
```

In a second separate terminal:
```bash
ros2 run ur5_machine_learning dataset_generator_node --num-points 10000 --angle-step 45
```
The parameters can be changed accordingly, or left as default (1000 steps and 45 degrees).
Note: the random angle step used in the code will be from [-angle_step, +angle_step].