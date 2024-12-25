#!/bin/bash

# Array of all methods to test (including DINO)
METHODS=("sift" "orb" "akaze" "dino")
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

# Function to run with specified method and perturbation setting
run_configuration() {
    local method=$1
    local perturbation=$2
    
    echo "========================================="
    echo "Running visual servoing with ${method}"
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
    if [ "$method" = "dino" ]; then
        if [ "$perturbation" = true ]; then
            python3 vitvs_v2.py --config $CONFIG --perturbation
        else
            python3 vitvs_v2.py --config $CONFIG
        fi
    else
        if [ "$perturbation" = true ]; then
            python3 ibvs_standard.py --method $method --config $CONFIG --perturbation
        else
            python3 ibvs_standard.py --method $method --config $CONFIG
        fi
    fi
    
    # Wait for 3 seconds
    sleep 3
    
    # Kill roslaunch and its child processes
    echo "Terminating roslaunch and child processes..."
    pkill -P $ROSLAUNCH_PID
    kill $ROSLAUNCH_PID
    
    # Ensure all ROS and Gazebo processes are terminated
    kill_ros_and_gazebo
    
    echo "Completed run with ${method} (Perturbation: ${perturbation})"
    echo "----------------------------------------"
}

# Main execution
echo "Starting visual servoing test suite"
echo "Using configuration: ${CONFIG}"

# Loop through each method
for method in "${METHODS[@]}"; do
    # Run without perturbation
    run_configuration $method false
    
    # Run with perturbation
    run_configuration $method true
done

echo "All tests completed!"
