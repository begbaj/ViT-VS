# VitVS: Zero-shot ViT features for Visual Servoing

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
   ```
   
2. **Configure GPU support (optional)**
  - To disable GPU support, comment out the GPU version and enable the non-GPU in buildandrun.sh

3. **Build and run Docker container**
   ```bash
   ./buildandrun.sh
   ```
4. **Inside the Docker container:**
   ```bash
   catkin_make
   cd catkin_ws/src/ibvs/src
   ```

5. Then the visual servoing system can be run with different feature detection methods using a single unified script. Here are the available options:

```bash
./run_ibvs.sh --method  [--config ] [--perturbation]
```

#### Arguments:

- `--method`: Specify the feature detection method (required)
  - Options: `sift`, `orb`, `akaze`, `dino`
- `--config`: Specify a custom configuration file (optional)
  - Default: `config.yaml`
- `--perturbation`: Enable perturbation mode (optional)
  - Adds image perturbation during visual servoing

#### Examples:

Run with SIFT feature detection:
```bash
./run_ibvs.sh --method sift
```

Run with ORB and perturbation:
```bash
./run_ibvs.sh --method orb --perturbation
```

Run DINOv2 with a custom config:
```bash
./run_ibvs.sh --method dino --config custom_config.yaml
```

Run AKAZE with all options:
```bash
./run_ibvs.sh --method akaze --config custom_config.yaml --perturbation
```

   
At this point:

- Gazebo and RViz should start
- Visual Servoing code will begin running
- Simulation runs according to the config file in catkin_ws/ibvs/config

## Additional Configuration
To run the Visual Servoing with SIFT+BF corresponding matching:
   ```bash
   catkin_make
   cd catkin_ws/src/ibvs/src
   ./run_sift.sh
```
To enable Gazebo GUI:
- Open catkin_ws/ibvs/launch/ibvs.launch
- Set the argument "gui" to true

Connecting another shell to the docker container:
```bash
   docker exec -ti viso bash
```
Create the perturbed models:
- inside the container go to "~/catkin_ws/src/ibvs"
- run:
```bash
   python3 generate_perturbed_models.py
```
To inspect a .npz file run:
```bash
   python3 npzviewer.py results_config_sift_standard.npz 
```
