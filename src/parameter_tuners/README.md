## Description

The parameter_tuners directory consists of three python scripts that apply manual search to find optimal combination of parameters for each filter.

The parameters to be found for each filter are:
* Unscented Kalman Filter: 
  - Alpha
  - Beta
  - Kappa
  
  Those are known as scaling factor/parameters to define how to spread sigma points around the mean vector(current state at time t).

* Particle Filter:
  - Particle/Sample size

  Finding optimal particle size provides good result of an estimation and fast computation. Setting low number of particle size may cause filter divergence where all the particles do not represent the state properly.

* Ensemble Kalman Filter:
  - Ensemble size

  Finding an optimal size of ensemble properly represents the state uncertainty and provides a good estimation.

## MEMO
Those parameter tuning scripts are called in Jupyter notebooks under `src/notebooks/KITTI/experiments/tuning/` directory.
All jupyter notebooks require KITTI datasets to be downloaded before running the notebook cells.
Please refer to the download dataset section.
