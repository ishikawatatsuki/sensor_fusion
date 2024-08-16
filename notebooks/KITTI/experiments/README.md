## Description

In the KITTI directory, there are several directories:

- `filters`: 

  filters directory consists of several notebooks where each filter test different sensor setups using KITTI dataset sequence number 09.

- `sequences`:

  sequences directory consists of two notebooks. Each notebook test all filters by different setups on KITTI dataset sequence number 04 and 09 suggested by the name of the notebook.

- `tests`:

  tests directory consists of different kind of tests, such as KITTI grounded vehicle trajectory estimation in 3d space, IMU-only filter estimation, and comparison between un-smoothed IMU data and smoothed IMU sensor data pose estimation.

- `tuning`:

  tuning directory contains multiple notebooks that apply manual search for three parameter-required-filters including Unscented Kalman Filter, Particle Filter, and Ensemble Kalman Filter to find optimal parameters.

- `noise_optimizations`:

  noise_optimizations directory consists of some notebooks that apply optimization to find process noise and measurement noise involved in filters. The optimization process is applied to each filter and results are exported under `exports/_noise_optimizations/` directory.