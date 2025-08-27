#!/bin/bash
#SBATCH -p common
#SBATCH -t 2-00:00:00

module load rocky8 micromamba

SENSOR_FUSION_DIR=/gpfs/mariana/home/taishi/workspace/researches/sensor_fusion

cd $SENSOR_FUSION_DIR

micromamba run -n sensor_fusion python -m src._experiments.run_visual_odometry_2d3d \
    --dataset_path "$SENSOR_FUSION_DIR/data/KITTI" \
    --output_path "$SENSOR_FUSION_DIR/outputs/vo_experiments/pose_estimation_2d3d" \
    --config_file "$SENSOR_FUSION_DIR/configs/kitti_config_vo_experiment.yaml"