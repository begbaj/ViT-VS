#!/bin/bash
####################################################################
# Visual Servoing Docker Container Startup Script                     #
#                                                                    #
# This code is available under a GPL v3.0 license and comes without  #
# any explicit or implicit warranty.                                 #
#                                                                    #
# (C) Alessandro Scherl 2024 <alessandro.scherl@technikum-wien.at>  #
####################################################################

# Setup for RealSense camera
xhost +local:docker
udevadm control --reload-rules && udevadm trigger

# Build Docker image
echo "Building Docker image..."
docker build . -t viso

# Choose one of the following run commands:

# 1. Run with GPU support:
#docker run -it --rm -t -d \
#    --name viso \
#    --network="host" \
#    -e DISPLAY=$DISPLAY \
#    --privileged \
#    --runtime=nvidia \
#    --gpus all \
#    -p 8888:8888 \
#    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
#    --mount src="$(pwd)/catkin_ws",target=/root/catkin_ws/src/,type=bind \
#    viso

# 2. Run without GPU support:
 docker run -it --rm -t -d \
     --name viso \
     --network="host" \
     -e DISPLAY=$DISPLAY \
     --privileged \
     -p 8888:8888 \
     --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
     --mount src="$(pwd)/catkin_ws",target=/root/catkin_ws/src/,type=bind \
     viso

# Connect to container
echo "Connecting to container..."
docker exec -it viso bash
