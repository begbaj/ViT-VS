####################################################################
# Visual Servoing Docker Container                                    #
#                                                                    #
# This code is available under a GPL v3.0 license and comes without  #
# any explicit or implicit warranty.                                 #
#                                                                    #
# (C) Alessandro Scherl 2024 <alessandro.scherl@technikum-wien.at>  #
####################################################################

# Base image
FROM ubuntu:20.04

# Metadata
LABEL maintainer="Alessandro Scherl <alessandro.scherl@technikum-wien.at>"

# Environment setup
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install base dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    gnupg2 \
    git \
    nano \
    gedit \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python requirements
RUN pip3 install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install ROS Noetic
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    apt-get update && \
    apt-get install -y \
        ros-noetic-desktop-full \
        ros-noetic-realsense2-camera \
        ros-noetic-moveit \
        ros-noetic-realsense2-description \
        ros-noetic-rgbd-launch \
    && rm -rf /var/lib/apt/lists/*

# ROS workspace setup
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
    mkdir -p ~/catkin_ws/src && \
    /bin/bash -c "source /opt/ros/noetic/setup.bash && cd ~/catkin_ws && catkin_make"

# Initialize rosdep
RUN rosdep init && \
    rosdep update

# Set up catkin workspace
WORKDIR /root/catkin_ws
RUN echo "source /root/catkin_ws/devel/setup.bash" >> ~/.bashrc && \
    rosdep install --rosdistro $ROS_DISTRO --ignore-src --from-paths src -y && \
    /bin/bash -c '. /opt/ros/noetic/setup.bash; cd /root/catkin_ws; catkin_make' && \
    echo "source /home/catkin_ws/devel/setup.bash" >> ~/.bashrc

# Install additional dependencies
RUN pip install git+https://github.com/lucasb-eyer/pydensecrf.git
RUN ln -s /usr/bin/python3 /usr/bin/python