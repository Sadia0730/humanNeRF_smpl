#!/bin/bash

# Configuration
CONFIG_PATH="./configs/human_nerf/zju_mocap/387/adventure.yaml"
START_CAM=2
END_CAM=22

echo "Starting camera rendering for cameras $START_CAM to $END_CAM"

for cam_id in $(seq $START_CAM $END_CAM)
do
    echo "Processing camera $cam_id..."
    python run.py --type movement \
                 --cfg $CONFIG_PATH \
                 --cam_id $cam_id \
                 load_net latest

    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Camera $cam_id completed successfully"
    else
        echo "Error processing camera $cam_id"
    fi

    echo "----------------------------------------"
done

echo "All cameras processed"
