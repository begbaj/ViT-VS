#!/bin/bash

CONFIG="config_standard.yaml"

# Function to kill ROS and Gazebo processes
kill_ros_and_gazebo() {
    echo "Shutting down ROS and Gazebo processes..."
    pkill -f gzclient
    pkill -f gzserver
    pkill -f roscore
    pkill -f rosmaster
    sleep 5  # Wait for processes to fully terminate
}

# Function to run with specified perturbation setting
run_configuration() {
    local perturbation=$1

    echo "========================================="
    echo "Running DINO visual servoing WITHOUT rotation compensation"
    if [ "$perturbation" = true ]; then
        echo "Mode: Perturbed"
    else
        echo "Mode: Standard"
    fi
    echo "========================================="

    # Ensure clean slate before starting
    kill_ros_and_gazebo

    # Start roslaunch
    roslaunch ibvs ibvs.launch &
    ROSLAUNCH_PID=$!

    # Wait for 5 seconds to ensure everything is up
    echo "Waiting for ROS and Gazebo to initialize..."
    sleep 5

    # Run Python script with appropriate flags
    echo "Starting visual servoing..."
    if [ "$perturbation" = true ]; then
        python3 vitvs_v2_rotation_off.py --config $CONFIG --perturbation
    else
        python3 vitvs_v2_rotation_off.py --config $CONFIG
    fi

    # Wait for 3 seconds
    sleep 3

    # Kill roslaunch and its child processes
    echo "Terminating roslaunch and child processes..."
    pkill -P $ROSLAUNCH_PID
    kill $ROSLAUNCH_PID

    # Ensure all ROS and Gazebo processes are terminated
    kill_ros_and_gazebo

    echo "Completed run (Perturbation: ${perturbation})"
    echo "----------------------------------------"
}

# Main execution
echo "Starting DINO evaluation without rotation compensation"
echo "Using configuration: ${CONFIG}"

# Run without perturbation
run_configuration false

# Run with perturbation
run_configuration true

echo "All tests completed!"