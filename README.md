# VitVS: Vision Transformer-based Visual Servoing

## Overview
A repository implementing Image Based Visual Servoing (IBVS) that combines:
- Dino V2 Vision Transformer for feature extraction
- Feature correspondence matching
- ROS and Gazebo simulation for evaluation

## System Requirements
- Ubuntu/Linux
- Docker
- NVIDIA Container Toolkit

## Prerequisites

1. **Install Docker**
   - Follow the official guide: [Docker Installation](https://docs.docker.com/engine/install/)

2. **Install NVIDIA Container Toolkit (for GPU support)**
   - Follow the official guide: [NVIDIA Container Toolkit Installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/AlessandroScherl/VitVS.git
   
2. **Configure GPU support (optional)**
  - To disable GPU support, comment out the GPU version and enable the non-GPU in buildandrun.sh

3. **Build and run Docker container**
   ```bash
   ./buildandrun.sh
4. **Inside the Docker container:**
   ```bash
   catkin_make
   cd catkin_ws/src/ibvs/src
   ./run_ibvs.sh
At this point:

- Gazebo and RViz should start
- Visual Servoing code will begin running
- Simulation runs according to the config file in catkin_ws/ibvs/config

## Additional Configuration
To enable Gazebo GUI:
- Open catkin_ws/ibvs/launch/ibvs.launch
- Set the argument "gui" to true

Connecting another shell to the docker container:
```
docker exec -ti viso bash
