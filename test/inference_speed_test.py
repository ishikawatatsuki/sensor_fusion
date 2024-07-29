import sys
if __name__ == "__main__":
    sys.path.append('../src')
import numpy as np
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from configs import SetupEnum, MeasurementDataEnum, SamplingEnum, NoiseTypeEnum, FilterEnum, Configs
from kalman_filters import (
  ExtendedKalmanFilter,
  UnscentedKalmanFilter,
  ParticleFilter, ResamplingAlgorithms,
  EnsembleKalmanFilter,
  CubatureKalmanFilter
)

FILE_NAME = "inference_time.npy"
HOST_NAME = "m1_macbook_pro"
NUM_ITERATIONS = 10
MULTIPLIER = 1000

alpha_setup1_0 = 0.0001
beta_setup1_0 = 2.
kappa_setup1_0 = 0.

alpha_setup2_0 = 0.6
beta_setup2_0 = 6.
kappa_setup2_0 = -7.

alpha_setup3_0 = 0.0001
beta_setup3_0 = 4.
kappa_setup3_0 = 0.

n_samples_setup1_0 = 2048
resampling_algorithm_setup1_0 = ResamplingAlgorithms.STRATIFIED
n_samples_setup2_0 = 2048
resampling_algorithm_setup2_0 = ResamplingAlgorithms.STRATIFIED
n_samples_setup3_0 = 2048
resampling_algorithm_setup3_0 = ResamplingAlgorithms.RESIDUAL

n_ensemble_setup1_0 = 256
n_ensemble_setup2_0 = 256
n_ensemble_setup3_0 = 256

def run(data):
  
  setups = [SetupEnum.SETUP_1, SetupEnum.SETUP_2, SetupEnum.SETUP_3]
  
  inference_times = []
  print("Inference Speed Test Start.")
  for setup in setups:
    
    ekf_inf = 0
    ukf_inf = 0
    pf_inf = 0
    enkf_inf = 0
    ckf_inf = 0
    for i in tqdm(range(NUM_ITERATIONS)):
      x, P, H, q, r_vo, r_gps = data.get_initial_data(setup=setup,
                                                    filter_type=FilterEnum.EKF, noise_type=NoiseTypeEnum.CURRENT)
      ekf = ExtendedKalmanFilter(x=x.copy(), 
                                    P=P.copy(), 
                                    H=H.copy(),
                                    q=q,
                                    r_vo=r_vo,
                                    r_gps=r_gps,
                                    setup=setup
                                    )
      
      start = datetime.now()
      error_ekf = ekf.run(data=data)
      end = datetime.now()
      processing_time = (end - start).total_seconds()
      ekf_inf += np.round(processing_time / data.N_original, Configs.processing_time_decimal_place)
    
      x, P, H, q, r_vo, r_gps = data.get_initial_data(
        setup=setup,
        filter_type=FilterEnum.UKF, 
        noise_type=NoiseTypeEnum.CURRENT)
      ukf = UnscentedKalmanFilter(
          x=x.copy(), 
          P=P.copy(), 
          H=H.copy(), 
          q=q,
          r_vo=r_vo,
          r_gps=r_gps,
          alpha=alpha_setup1_0, 
          beta=beta_setup1_0, 
          kappa=kappa_setup1_0,
          setup=setup
      )
      start = datetime.now()
      error_ukf = ukf.run(data=data)
      end = datetime.now()
      processing_time = (end - start).total_seconds()
      ukf_inf += np.round(processing_time / data.N_original, Configs.processing_time_decimal_place)
      
      x, P, H, q, r_vo, r_gps = data.get_initial_data(
        setup=setup,
        filter_type=FilterEnum.PF, 
        noise_type=NoiseTypeEnum.CURRENT)
      pf = ParticleFilter(
        N=n_samples_setup1_0, 
        x_dim=x.shape[0], 
        H=H.copy(), 
        q=q,
        r_vo=r_vo,
        r_gps=r_gps,
        setup=setup,
        resampling_algorithm=resampling_algorithm_setup1_0)
      pf.create_gaussian_particles(mean=x.copy(), var=P.copy())
      
      start = datetime.now()
      error_pf = pf.run(data=data)
      end = datetime.now()
      processing_time = (end - start).total_seconds()
      pf_inf += np.round(processing_time / data.N_original, Configs.processing_time_decimal_place)
      
      x, P, H, q, r_vo, r_gps = data.get_initial_data(
        setup=setup,
        filter_type=FilterEnum.EnKF, 
        noise_type=NoiseTypeEnum.CURRENT)
      
      enkf = EnsembleKalmanFilter(
          N=n_ensemble_setup1_0, 
          x=x.copy(), 
          P=P.copy(), 
          H=H.copy(),
          q=q,
          r_vo=r_vo,
          r_gps=r_gps,
          setup=setup)
      start = datetime.now()
      error_enkf = enkf.run(data=data)
      end = datetime.now()
      processing_time = (end - start).total_seconds()
      enkf_inf += np.round(processing_time / data.N_original, Configs.processing_time_decimal_place)

      x, P, H, q, r_vo, r_gps = data.get_initial_data(
        setup=setup,
        filter_type=FilterEnum.CKF, 
        noise_type=NoiseTypeEnum.CURRENT)
      ckf = CubatureKalmanFilter(
          x=x.copy(), 
          P=P.copy(), 
          H=H.copy(),
          q=q,
          r_vo=r_vo,
          r_gps=r_gps,
          setup=setup,
      )
      start = datetime.now()
      error_ckf = ckf.run(data=data)
      end = datetime.now()
      processing_time = (end - start).total_seconds()
      ckf_inf += np.round(processing_time / data.N_original, Configs.processing_time_decimal_place)

    # Store inference time in milliseconds
    inference_times.append(MULTIPLIER * ekf_inf / NUM_ITERATIONS)
    inference_times.append(MULTIPLIER * ukf_inf / NUM_ITERATIONS)
    inference_times.append(MULTIPLIER * pf_inf / NUM_ITERATIONS)
    inference_times.append(MULTIPLIER * enkf_inf / NUM_ITERATIONS)
    inference_times.append(MULTIPLIER * ckf_inf / NUM_ITERATIONS)
  
  print("Inference Speed Test Completed.")
  
  return inference_times
  
  

if __name__ == "__main__":
  import os
  from data_loader import DataLoader
  

  root_path = "../"
  file_export_path = os.path.join(root_path, "exports/_sequences/04")
  kitti_root_dir = os.path.join(root_path, "data")
  vo_root_dir = os.path.join(root_path, "vo_estimates")
  noise_vector_dir = os.path.join(root_path, "exports/_noise_optimizations/noise_vectors")
  kitti_date = '2011_09_30'
  kitti_drive = '0033'
  
  results_header = pd.MultiIndex.from_product([['Setup1 (IMU, VO)','Setup2(IMU, VO+GPS)', 'Setup3(INS, VO+INS)'],
                                            ['EKF', 'UKF', 'PF', 'EnKF', 'CKF']],
                                            names=['Setups','Filter types'])
  index = [HOST_NAME]
  
  report_export_path = os.path.join(root_path, f"exports/inference_time/{HOST_NAME}")
  if not os.path.exists(report_export_path):
    os.mkdir(report_export_path)

  data = DataLoader(sequence_nr=kitti_drive, 
                  kitti_root_dir=kitti_root_dir, 
                  vo_root_dir=vo_root_dir,
                  noise_vector_dir=noise_vector_dir,
                  vo_dropout_ratio=0.0, 
                  gps_dropout_ratio=0.0)

  inference_times = run(data=data)
  print(f"Result: {inference_times}")
  
  filename = os.path.join(report_export_path, FILE_NAME)
  np.array(inference_times).dump(filename)
