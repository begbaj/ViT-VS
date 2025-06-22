#!/bin/bash

# Print usage if no arguments or help flag is provided
usage() {
    echo "Usage: $0 --method <sift|orb|akaze|dino> [--config <config_file>] [--perturbation]"
    echo "Options:"
    echo "  --method        Specify the feature detection method (sift, orb, akaze, or dino)"
    echo "  --launch        Specify the .launch to run (check launch/ directory for available options, default: ibvs.launch)"
    echo "  --config        Optional: Specify the configuration file (default: config.yaml)"
    echo "  --perturbation  Optional: Enable perturbation mode"
    exit 1
}

# Function to kill ROS and Gazebo processes
kill_ros_and_gazebo() {
    echo "Shutting down ROS and Gazebo processes..."
    pkill -f gzclient
    pkill -f gzserver
    pkill -f roscore
    pkill -f rosmaster
    sleep 5  # Wait for processes to fully terminate
}

# Function to run with specified method and configuration
run_configuration() {
    local method=$1
    local config=$2
    local perturbation=$3
    
    echo "Running visual servoing with ${method} feature detection..."
    if [ "$perturbation" = true ]; then
        echo "Perturbation mode enabled"
    fi
    
    # Ensure clean slate before starting
    kill_ros_and_gazebo

    # Start roslaunch
    DISABLE_ROS1_EOL_WARNINGS=1 roslaunch ibvs $LAUNCH &
    ROSLAUNCH_PID=$!
    # ROSLAUNCH_PID=$!

    # Wait for 30 seconds to ensure everything is up
    echo "Waiting for ROS and Gazebo to initialize..."
    sleep 30

    echo "-----ROSTOPIC LIST-----"
    echo ""

    rostopic list

    echo ""
    echo "-----ROSTOPIC LIST-----"
    
    # Run Python script based on method
    echo "Starting visual servoing..."
    if [ "$method" = "dino" ]; then
        if [ "$perturbation" = true ]; then
            python3 vitvs_v2.py --config $config --perturbation
        else
            python3 vitvs_v2.py --config $config
        fi
    else
        if [ "$perturbation" = true ]; then
            python3 ibvs_standard.py --method $method --config $config --perturbation
        else
            python3 ibvs_standard.py --method $method --config $config
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
}

# Parse command line arguments
METHOD=""
CONFIG="config.yaml"  # Default configuration file
PERTURBATION=false    # Default perturbation setting
LAUNCH="ibvs.launch"  # Default launch file

while [[ $# -gt 0 ]]; do
    case $1 in
        --method)
            METHOD="$2"
            shift 2
            ;;
        --launch)
            LAUNCH="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --perturbation)
            PERTURBATION=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate method
if [[ ! $METHOD =~ ^(sift|orb|akaze|dino)$ ]]; then
    echo "Error: Invalid method. Must be one of: sift, orb, akaze, dino"
    usage
fi

# Run with specified method and configuration
run_configuration $METHOD $CONFIG $PERTURBATION

echo "Visual servoing completed."
