## Description
The src directory contains custom libraries that are used to load and fuse sensor data and optimize the noise and parameters of filters.

The child directories under src are:
* `data_loader`: 
  - contains python scripts to load KITTI dataset and UAV dataset.
* `kalman_filters`: 
  - contains various filters used in the research, including EKF, UKF, PF, EnKF and CKF. 
* `noise_optimizers`: 
  - contains noise optimization scripts for each filter, where by the process noise vector and measurement noise vector is optimized.
* `parameter_tuners`: 
  - contains parameter tuning scripts for the filter which consist of parameters such as alpha, beta and kappa values in UKF and number of particles in PF.
