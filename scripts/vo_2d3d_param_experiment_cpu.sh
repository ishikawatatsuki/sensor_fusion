#!/bin/bash
#SBATCH -p common
#SBATCH -t 2-00:00:00
module load rocky8 micromamba

SENSOR_FUSION_DIR=/gpfs/mariana/home/taishi/workspace/researches/sensor_fusion

cd $SENSOR_FUSION_DIR

micromamba run -n sensor_fusion python -m src.visual_odometry.test_scripts.vo_2d3d_param_experiment\
    --dataset_dir "$SENSOR_FUSION_DIR/data/KITTI" \
    --output_dir "$SENSOR_FUSION_DIR/outputs/vo_experiments/2d3d_param_optimization_final" \
    --log_dir "$SENSOR_FUSION_DIR/logs/vo_experiments/2d3d_param_optimization_final" \
    --num_trials 150