#!/bin/bash

SENSOR_FUSION_DIR=/Volumes/Data_EXT/data/workspaces/sensor_fusion

cd $SENSOR_FUSION_DIR

python -m src.visual_odometry.test_scripts.vo_2d2d_param_experiment \
    --dataset_dir "$SENSOR_FUSION_DIR/data/KITTI" \
    --output_dir "$SENSOR_FUSION_DIR/outputs/vo_experiments/2d2d_param_optimization" \
    --log_dir "$SENSOR_FUSION_DIR/logs/vo_experiments/2d2d_param_optimization" \
    --num_trials 1
