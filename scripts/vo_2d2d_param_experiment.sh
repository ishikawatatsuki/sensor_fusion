#!/bin/bash
#SBATCH --gpus 1
#--gres=gpu:A100:1
#--constraint=A100-80
#SBATCH --cpus-per-gpu 4
#SBATCH --mem-per-gpu 8GB
#SBATCH -p common
#SBATCH -t 12:00:00
module load rocky8 micromamba

SENSOR_FUSION_DIR=/gpfs/mariana/home/taishi/workspace/researches/sensor_fusion

cd $SENSOR_FUSION_DIR

micromamba run -n sensor_fusion python -m src.visual_odometry.test_scripts.vo_2d2d_param_experiment\
    --dataset_dir "$SENSOR_FUSION_DIR/data/KITTI" \
    --output_dir "$SENSOR_FUSION_DIR/outputs/vo_experiments/2d2d_param_optimization" \
    --log_dir "$SENSOR_FUSION_DIR/logs/vo_experiments/2d2d_param_optimization" \
    --n_trials 10
