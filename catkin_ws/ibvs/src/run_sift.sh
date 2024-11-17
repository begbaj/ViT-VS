#!/bin/bash

# Function to kill ROS and Gazebo processes
kill_ros_and_gazebo() {
    echo "Shutting down ROS and Gazebo processes..."
    pkill -f gzclient
    pkill -f gzserver
    pkill -f roscore
    pkill -f rosmaster
    sleep 5  # Wait for processes to fully terminate
}

# Function to run a single configuration
run_configuration() {
    local config=$1
    
    # Ensure clean slate before starting
    kill_ros_and_gazebo
    
    # Start roslaunch
    roslaunch ibvs ibvs.launch &
    ROSLAUNCH_PID=$!
    
    # Wait for 5 seconds to ensure everything is up
    sleep 5
    
    # Run Python script
    python3 vs_sift.py --config $config
    
    # Wait for 3 seconds
    sleep 3
    
    # Kill roslaunch and its child processes
    echo "Terminating roslaunch and child processes..."
    pkill -P $ROSLAUNCH_PID
    kill $ROSLAUNCH_PID
    
    # Ensure all ROS and Gazebo processes are terminated
    kill_ros_and_gazebo
}


# Run with following configuration
run_configuration "config.yaml"



echo "All configurations completed."
