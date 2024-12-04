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
echo "Starting Docker container..."
docker run -it --rm -t -d \
    --name viso_sim \
    --network="host" \
    -e DISPLAY=$DISPLAY \
    --privileged \
    --runtime=nvidia \
    -p 8888:8888 \
    --gpus all \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --mount src="$(pwd)/catkin_ws",target=/root/catkin_ws/src/,type=bind \
    --device=/dev/bus/usb/001/003:/dev/bus/usb/001/003 \
    -v /dev:/dev \
    viso_sim:latest

# Connect to container
echo "Connecting to container..."
docker exec -it viso_sim bash
