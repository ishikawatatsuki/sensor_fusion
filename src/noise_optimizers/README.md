## Description

`noise_optimizers` directory contains python scripts designed to optimize process noise and measurement noise for each filter.

Each optimizer script is called by corresponding jupyter notebook located under `notebooks/experiments/noise_optimizations/` and result is stored under `exports/_noise_optimizations/noise_vectors/{filter name}/` in numpy file format.

Currently, all the optimizer python scripts refer to the KITTI example dataset stored in `example_data/` so that no data download is needed. The result is stored in temporary folder labeled by `{filter name}_example` under the same export path `exports/_noise_optimizations/`.

## MEMO

The optimization script for Particle Filter and Ensemble Kalman Filter might take more than hours. If you want to quit executing while the script is running, you can quit the process by stopping the running container.