## Description

The `exports` directory consists of results exported by each experiment.
Description for each directory is as follows:

- `_inference_time`: 

  _inference_time consists of the result of time take to run each filter on different machines such as m1 macbook pro and raspberry pi 4. The results are exported as numpy format.

- `_noise_optimizations`:

  As the name suggests, this directory contains noise vector stored in numpy file format and the error json files that compare errors between before and after noise optimization. The noise optimization is applied to each filter.

- `ensemble_kalaman_filter`:

  This directory stores estimated error and inference time given dropout rate resulted from experiments that find optimal parameters for each filter conducted in the notebooks stored under `notebooks/KITTI/experiments/tuning/` directory. The error and time report is computed for both KITTI sequences and stored in pandas dataframe format.

- `particle_filter`:

  Likewise `ensemble_kalman_filter`, the directory consists of the results computed by the notebooks, runs an experiment to find optimal parameters for Particle Filter, stored in the directory `notebooks/KITTI/experiments/tuning/`. The file path under the `particle_filter` represents, `{setup number}/{KITTI dataset sequence number}/{VO dropout rate}_{GPS dropout rate}_{either time or error}_df.json`.

- `unscented_kalman_filter`:

  Likewise those two directories, `unscented_kalman_filter` directory also stores the result obtained from the notebooks, which find optimal parameter used in Unscented Kalaman Filter, stored in the directory `notebooks/KITTI/experiments/tuning/`. Same as the file path in the `particle_filter`, file path under the `unscented_kalman_filter` represents, `{setup number}/{KITTI dataset sequence number}/{VO dropout rate}_{GPS dropout rate}_{either time or error}_df.json`.

These error and time report json files stored in `ensemble_kalaman_filter`, `particle_filter`, and `unscented_kalman_filter` can be loaded in the notebooks located in `notebooks/KITTI/experiments/tuning/`.