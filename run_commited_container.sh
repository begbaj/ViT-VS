#!/bin/bash
####################################################################
# Visual Servoing Docker Container Startup Script                     #
#                                                                    #
# This code is available under a GPL v3.0 license and comes without  #
# any explicit or implicit warranty.                                 #
#                                                                    #
# (C) Alessandro Scherl 2024 <alessandro.scherl@technikum-wien.at>  #
####################################################################

# Setup for RealSense camera and X server access
echo "Setting up X server access and USB devices..."
xhost +local:docker
udevadm control --reload-rules && udevadm trigger

# Run Docker container with GPU support
docker run -it --rm -t -d \
    --name viso_sim \
    --network="host" \
    -e DISPLAY=$DISPLAY \
    --privileged \
    --runtime=nvidia \
    --gpus '"device=0",capabilities=utility' \
    -p 8888:8888 \
    --volume="/tmp/.X11-unix-cv2425g26:/tmp/.X11-unix:rw" \
    -v $HOME/.Xauthority:/root/.Xauthority:ro \
    -v ./log/:/root/.ros/:rw \
    --mount src="$(pwd)/catkin_ws",target=/root/catkin_ws/src/,type=bind \
    viso_sim

# Connect to container
echo "Connecting to container..."
docker exec -it viso_sim bash
