#!/bin/bash

module load rocky8 micromamba

SENSOR_FUSION_DIR=/gpfs/mariana/home/taishi/workspace/researches/sensor_fusion

cd $SENSOR_FUSION_DIR

python -m src._experiments.run_visual_odometry_2d3d \
    --dataset_path "$SENSOR_FUSION_DIR/data/KITTI" \
    --output_path "$SENSOR_FUSION_DIR/outputs/vo_estimates/pose_estimation_2d3d" \
    --config_file "$SENSOR_FUSION_DIR/configs/kitti_config_vo_experiment.yaml"