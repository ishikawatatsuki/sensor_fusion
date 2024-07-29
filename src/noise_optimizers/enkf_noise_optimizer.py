import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
sys.path.append('../../src')
from data_loader import DataLoader
from configs.configs import SetupEnum, FilterEnum, ErrorEnum, NoiseTypeEnum
from scipy.optimize import minimize
from kalman_filters.ensemble_kalman_filter import EnsembleKalmanFilter
from utils.error_report import get_error_from_list

np.random.seed(777)

class EnKF_NoiseOptimizer:

  x = None
  P = None
  H = None

  result_1 = None
  result_2 = None
  result_3 = None

  n_samples = 2048
  
  setup = SetupEnum.SETUP_1

  header = pd.MultiIndex.from_product([['Setup1(IMU+VO)','Setup2(IMU+VO,GPS)', 'Setup3(INS)'],
                                   ["MAE", "RMSE", "MAX"]],
                                   names=['Setups', 'Error types'])
  index = ["Non-optimized", "Optimized", "∆"]
  error_df = None

  is_compared = False

  enkf_1 = None
  enkf_1_optimized = None

  enkf_2 = None
  enkf_2_optimized = None

  enkf_3 = None
  enkf_3_optimized = None

  error_df_export_path = None
  noise_vector_export_path = None

  maximum_error = 1e+8

  max_iter = 500

  def __init__(self, data, error_df_export_path, noise_vector_export_path):

    self.data = data

    assert os.path.exists(error_df_export_path) and os.path.exists(noise_vector_export_path), "Please set proper export path"

    self.error_df_export_path = error_df_export_path
    self.noise_vector_export_path = noise_vector_export_path

  def J(self, noise_vector):
    q = None
    r_vo = None
    r_gps = None
    if self.setup is SetupEnum.SETUP_1 or self.setup is SetupEnum.SETUP_2:
      q = noise_vector[:-4]
      r_vo = noise_vector[-4:-2]
      r_gps = noise_vector[-2:]
    else: # Setup3
      q = noise_vector[:-4]
      r_vo = noise_vector[-4:-2]
      r_gps = noise_vector[-2:]
    
    try:
      enkf = EnsembleKalmanFilter(
        N=self.n_samples, 
        x=self.x.copy(), 
        P=self.P.copy(), 
        H=self.H.copy(),
        q=q,
        r_vo=r_vo,
        r_gps=r_gps,
        setup=SetupEnum.SETUP_1)
      enkf.run(data=self.data, debug_mode=True)
      estimated = enkf.get_estimated_trajectory()
      actual = self.data.GPS_measurements_in_meter[:, :2]
      return np.sum((actual - estimated) ** 2)
    except:
      return self.maximum_error
  
  def run(self):
    print("Finding optimal noise vector")

    print("For setup 1")
    self.setup = SetupEnum.SETUP_1
    self.x, self.P, self.H, q1, r_vo1, r_gps1 = self.data.get_initial_data(setup=self.setup, filter_type=FilterEnum.EnKF)
    x0_1 = np.concatenate([q1, r_vo1, r_gps1]).tolist()
    bounds1 = [(0., 1e+5) for i in range(len(x0_1))]
    self.result_1 = minimize(self.J, x0_1, bounds=bounds1, method="nelder-mead", options={ 'maxiter': self.max_iter, 'disp': True })
    print(self.result_1)

    print("For setup 2")
    self.setup = SetupEnum.SETUP_2
    self.x, self.P, self.H, q2, r_vo2, r_gps2 = self.data.get_initial_data(setup=self.setup, filter_type=FilterEnum.EnKF)
    x0_2 = np.concatenate([q2, r_vo2, r_gps2]).tolist()
    bounds2 = [(0., 1e+5) for i in range(len(x0_2))]
    self.result_2 = minimize(self.J, x0_2, bounds=bounds2, method="nelder-mead", options={ 'maxiter': self.max_iter, 'disp': True })
    print(self.result_2)
    
    print("For setup 3")
    self.setup = SetupEnum.SETUP_3
    self.x, self.P, self.H, q3, r_vo3, r_gps3 = self.data.get_initial_data(setup=self.setup, filter_type=FilterEnum.EnKF)
    x0_3 = np.concatenate([q3, r_vo3, r_gps3]).tolist()
    bounds3 = [(0., 1e+5) for i in range(len(x0_3))]
    self.result_3 = minimize(self.J, x0_3, bounds=bounds3, method="nelder-mead", options={ 'maxiter': self.max_iter, 'disp': True })
    print(self.result_3)
    
    with open(f'{self.noise_vector_export_path}/setup_1_optimized.npy', 'wb') as file:
      np.save(file, self.result_1['x'])

    with open(f'{self.noise_vector_export_path}/setup_2_optimized.npy', 'wb') as file:
      np.save(file, self.result_2['x'])

    with open(f'{self.noise_vector_export_path}/setup_3_optimized.npy', 'wb') as file:
      np.save(file, self.result_3['x'])
  
  def compare(self, load_exported=False):
    x_1, P_1, H_1, q1, r_vo1, r_gps1 = self.data.get_initial_data(setup=SetupEnum.SETUP_1, filter_type=FilterEnum.EnKF, noise_type=NoiseTypeEnum.CURRENT)

    self.enkf_1 = EnsembleKalmanFilter(
      N=self.n_samples, 
      x=x_1.copy(), 
      P=P_1.copy(), 
      H=H_1.copy(),
      q=q1,
      r_vo=r_vo1,
      r_gps=r_gps1,
      setup=SetupEnum.SETUP_1)
    error_1 = self.enkf_1.run(data=self.data)

    optimized_noise_1 = None
    if load_exported:
      optimized_noise_1 = np.load(f'{self.noise_vector_export_path}/{str(SetupEnum.get_names()[SetupEnum.SETUP_1.value - 1]).lower()}_optimized.npy')
    else:
      optimized_noise_1 = self.result_1['x']

    q1_optimal = optimized_noise_1[:-4]
    r_vo1_optimal = optimized_noise_1[-4:-2]
    r_gps1_optimal = optimized_noise_1[-2:]

    self.enkf_1_optimized = EnsembleKalmanFilter(
      N=self.n_samples, 
      x=x_1.copy(), 
      P=P_1.copy(), 
      H=H_1.copy(),
      q=q1_optimal,
      r_vo=r_vo1_optimal,
      r_gps=r_gps1_optimal,
      setup=SetupEnum.SETUP_1)
    error_1_optimized = self.enkf_1_optimized.run(data=self.data)




    x_2, P_2, H_2, q2, r_vo2, r_gps2 = self.data.get_initial_data(setup=SetupEnum.SETUP_2, filter_type=FilterEnum.EnKF, noise_type=NoiseTypeEnum.CURRENT)
    
    self.enkf_2 = EnsembleKalmanFilter(
      N=self.n_samples, 
      x=x_2.copy(), 
      P=P_2.copy(), 
      H=H_2.copy(),
      q=q2,
      r_vo=r_vo2,
      r_gps=r_gps2,
      setup=SetupEnum.SETUP_2)
    error_2 = self.enkf_2.run(data=self.data)

    optimized_noise_2 = None
    if load_exported:
      optimized_noise_2 = np.load(f'{self.noise_vector_export_path}/{str(SetupEnum.get_names()[SetupEnum.SETUP_2.value - 1]).lower()}_optimized.npy')
    else:
      optimized_noise_2 = self.result_2['x']

    q2_optimal = optimized_noise_2[:-4]
    r_vo2_optimal = optimized_noise_2[-4:-2]
    r_gps2_optimal = optimized_noise_2[-2:]

    self.enkf_2_optimized = EnsembleKalmanFilter(
      N=self.n_samples, 
      x=x_2.copy(), 
      P=P_2.copy(), 
      H=H_2.copy(),
      q=q2_optimal,
      r_vo=r_vo2_optimal,
      r_gps=r_gps2_optimal,
      setup=SetupEnum.SETUP_2)
    error_2_optimized = self.enkf_2_optimized.run(data=self.data)




    x_3, P_3, H_3, q3, r_vo3, r_gps3 = self.data.get_initial_data(setup=SetupEnum.SETUP_3, filter_type=FilterEnum.EnKF, noise_type=NoiseTypeEnum.CURRENT)

    self.enkf_3 = EnsembleKalmanFilter(
      N=self.n_samples, 
      x=x_3.copy(), 
      P=P_3.copy(), 
      H=H_3.copy(),
      q=q3,
      r_vo=r_vo3,
      r_gps=r_gps3,
      setup=SetupEnum.SETUP_3)
    error_3 = self.enkf_3.run(data=self.data)

    optimized_noise_3 = None
    if load_exported:
      optimized_noise_3 = np.load(f'{self.noise_vector_export_path}/{str(SetupEnum.get_names()[SetupEnum.SETUP_3.value - 1]).lower()}_optimized.npy')
    else:
      optimized_noise_3 = self.result_3['x']

    q3_optimal = optimized_noise_3[:-4]
    r_vo3_optimal = optimized_noise_3[-4:-2]
    r_gps3_optimal = optimized_noise_3[-2:]
    self.enkf_3_optimized = EnsembleKalmanFilter(
      N=self.n_samples, 
      x=x_3.copy(), 
      P=P_3.copy(), 
      H=H_3.copy(),
      q=q3_optimal,
      r_vo=r_vo3_optimal,
      r_gps=r_gps3_optimal,
      setup=SetupEnum.SETUP_3)
    error_3_optimized = self.enkf_3_optimized.run(data=self.data)



    error_setup1 = [error_1, error_1_optimized]
    error_setup2 = [error_2, error_2_optimized]
    error_setup3 = [error_3, error_3_optimized]

    mae_1 = np.array(get_error_from_list(error_setup1, e_type=ErrorEnum.MAE))
    rmse_1 = np.array(get_error_from_list(error_setup1, e_type=ErrorEnum.RMSE))
    max_1 = np.array(get_error_from_list(error_setup1, e_type=ErrorEnum.MAX))
    mae_1 = np.append(mae_1, mae_1[1] - mae_1[0]).reshape(-1, 1)
    rmse_1 = np.append(rmse_1, rmse_1[1] - rmse_1[0]).reshape(-1, 1)
    max_1 = np.append(max_1, max_1[1] - max_1[0]).reshape(-1, 1)

    mae_2 = np.array(get_error_from_list(error_setup2, e_type=ErrorEnum.MAE))
    rmse_2 = np.array(get_error_from_list(error_setup2, e_type=ErrorEnum.RMSE))
    max_2 = np.array(get_error_from_list(error_setup2, e_type=ErrorEnum.MAX))
    mae_2 = np.append(mae_2, mae_2[1] - mae_2[0]).reshape(-1, 1)
    rmse_2 = np.append(rmse_2, rmse_2[1] - rmse_2[0]).reshape(-1, 1)
    max_2 = np.append(max_2, max_2[1] - max_2[0]).reshape(-1, 1)

    mae_3 = np.array(get_error_from_list(error_setup3, e_type=ErrorEnum.MAE))
    rmse_3 = np.array(get_error_from_list(error_setup3, e_type=ErrorEnum.RMSE))
    max_3 = np.array(get_error_from_list(error_setup3, e_type=ErrorEnum.MAX))
    mae_3 = np.append(mae_3, mae_3[1] - mae_3[0]).reshape(-1, 1)
    rmse_3 = np.append(rmse_3, rmse_3[1] - rmse_3[0]).reshape(-1, 1)
    max_3 = np.append(max_3, max_3[1] - max_3[0]).reshape(-1, 1)

    errors = np.concatenate([mae_1, rmse_1, max_1, mae_2, rmse_2, max_2, mae_3, rmse_3, max_3], axis=1)
    self.error_df = pd.DataFrame(errors, index=self.index, columns=self.header)

    self.error_df.to_json(f"{self.error_df_export_path}/error_comparison.json")
    self.is_compared = True

    self.error_df

  def visualize_results(self):
    assert self.is_compared, "Please run compare method first."

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(8, 16))
    xs, ys, _ = self.data.GPS_measurements_in_meter.T

    ax1.title.set_text('Current estimation vs noise optimized estimation (Setup1)')
    ax1.plot(xs, ys, lw=2, label='Ground-truth trajectory', color='black')
    ax1.plot(self.enkf_1.mu_x, self.enkf_1.mu_y, lw=2, label='Estimated result (non-optimal)', color='b')
    ax1.plot(self.enkf_1_optimized.mu_x, self.enkf_1_optimized.mu_y, lw=2, label='Estimated result (optimal)', color='r')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')

    ax2.title.set_text('Current estimation vs noise optimized estimation (Setup2)')
    ax2.plot(xs, ys, lw=2, label='Ground-truth trajectory', color='black')
    ax2.plot(self.enkf_2.mu_x, self.enkf_2.mu_y, lw=2, label='Estimated result (non-optimal)', color='b')
    ax2.plot(self.enkf_2_optimized.mu_x, self.enkf_2_optimized.mu_y, lw=2, label='Estimated result (optimal)', color='r')
    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Y [m]')

    ax3.title.set_text('Current estimation vs noise optimized estimation (Setup3)')
    ax3.plot(xs, ys, lw=2, label='Ground-truth trajectory', color='black')
    ax3.plot(self.enkf_3.mu_x, self.enkf_3.mu_y, lw=2, label='Estimated result (non-optimal)', color='b')
    ax3.plot(self.enkf_3_optimized.mu_x, self.enkf_3_optimized.mu_y, lw=2, label='Estimated result (optimal)', color='r')
    ax3.set_xlabel('X [m]')
    ax3.set_ylabel('Y [m]')

    ax1.legend()
    ax1.grid()

    ax2.legend()
    ax2.grid()

    ax3.legend()
    ax3.grid()

    plt.plot()


if __name__ == "__main__":
    kitti_root_dir = '../../data'
    vo_root_dir = '../../vo_estimates'
    noise_vector_dir = '../../exports/_noise_optimizations/noise_vectors'
    kitti_date = '2011_09_30'
    kitti_drive = '0033'
    
    data = DataLoader(sequence_nr=kitti_drive, 
                    kitti_root_dir=kitti_root_dir, 
                    vo_root_dir=vo_root_dir,
                    noise_vector_dir=noise_vector_dir,
                    vo_dropout_ratio=0.0, 
                    gps_dropout_ratio=0.0)
    
    error_df_export_path = '../../exports/_noise_optimizations/errors/enkf'
    noise_vector_export_path = '../../exports/_noise_optimizations/noise_vectors/enkf'

    optimizer = EnKF_NoiseOptimizer(
      data=data, 
      error_df_export_path=error_df_export_path, 
      noise_vector_export_path=noise_vector_export_path)
    optimizer.run()
    optimizer.compare()
    optimizer.visualize_results()
    