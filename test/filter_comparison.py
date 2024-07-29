import sys
if __name__ == "__main__":
    sys.path.append('../src')
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from configs import SetupEnum, MeasurementDataEnum, SamplingEnum, NoiseTypeEnum, FilterEnum, Configs
from kalman_filters import (
  ExtendedKalmanFilter,
  UnscentedKalmanFilter,
  ParticleFilter, ResamplingAlgorithms,
  EnsembleKalmanFilter,
  CubatureKalmanFilter
)


setup = SetupEnum.SETUP_1

n_ensemble_setup1_0 = 256
n_samples_setup1_0 = 2048
resampling_algorithm_setup1_0 = ResamplingAlgorithms.STRATIFIED

def run(data):
  x_ekf, P_ekf, H_ekf, q_ekf, r_vo_ekf, r_gps_ekf = data.get_initial_data(
    setup=setup,
    filter_type=FilterEnum.EKF, noise_type=NoiseTypeEnum.CURRENT)
  ekf = ExtendedKalmanFilter(
    x=x_ekf.copy(), 
    P=P_ekf.copy(), 
    H=H_ekf.copy(),
    q=q_ekf,
    r_vo=r_vo_ekf,
    r_gps=r_gps_ekf,
    setup=setup)
  
  x_pf, P_pf, H_pf, q_pf, r_vo_pf, r_gps_pf = data.get_initial_data(
    setup=setup,
    filter_type=FilterEnum.PF, 
    noise_type=NoiseTypeEnum.CURRENT)
  pf = ParticleFilter(
    N=n_samples_setup1_0, 
    x_dim=x_pf.shape[0], 
    H=H_pf.copy(), 
    q=q_pf,
    r_vo=r_vo_pf,
    r_gps=r_gps_pf,
    setup=setup,
    resampling_algorithm=resampling_algorithm_setup1_0)
  pf.create_gaussian_particles(mean=x_pf.copy(), var=P_pf.copy())
  pf_x, _ = pf.estimate()

  x_enkf, P_enkf, H_enkf, q_enkf, r_vo_enkf, r_gps_enkf = data.get_initial_data(
    setup=setup,
    filter_type=FilterEnum.EnKF, 
    noise_type=NoiseTypeEnum.CURRENT)
  enkf = EnsembleKalmanFilter(
      N=n_ensemble_setup1_0, 
      x=x_enkf.copy(), 
      P=P_enkf.copy(), 
      H=H_enkf.copy(),
      q=q_enkf,
      r_vo=r_vo_enkf,
      r_gps=r_gps_enkf,
      setup=setup)
  
  x_ckf, P_ckf, H_ckf, q_ckf, r_vo_ckf, r_gps_ckf = data.get_initial_data(
    setup=setup,
    filter_type=FilterEnum.CKF, 
    noise_type=NoiseTypeEnum.CURRENT)
  ckf = CubatureKalmanFilter(
      x=x_ckf.copy(), 
      P=P_ckf.copy(), 
      H=H_ckf.copy(),
      q=q_ckf,
      r_vo=r_vo_ckf,
      r_gps=r_gps_ckf,
      setup=setup,
  )
  
  ekf_mean_ = [[ekf.x[0, 0], ekf.x[1, 0]]]
  ckf_mean_ = [[ckf.x[0, 0], ckf.x[1, 0]]]
  pf_mean_ = [[pf_x[0], pf_x[0]]]
  enkf_mean_ = [[enkf.x[0, 0], enkf.x[1, 0]]]
  
  ekf_mean = np.array(ekf_mean_)
  ckf_mean = np.array(ckf_mean_)
  pf_mean = np.array(pf_mean_)
  enkf_mean = np.array(enkf_mean_)
  
  fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(12, 6))
  ekf_ax, ckf_ax = ax1
  pf_ax, enkf_ax = ax2
  
  x_gps, y_gps, _ = data.GPS_measurements_in_meter.T
  x_vo, y_vo, _ = data.VO_measurements.T
  for ax in (ekf_ax, ckf_ax, pf_ax, enkf_ax):
    ax.plot(x_gps, y_gps, lw=2, label="Ground-truth trajectory", color='black')
    ax.plot(x_vo, y_vo, lw=2, label="Visual Odometry trajectory", color='green')
    
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.grid()

  
  ekf_ax.plot(ekf_mean[:, 0], ekf_mean[:, 1], label="Estimated trajectory by EKF", lw=2, color='r')
  ckf_ax.plot(ckf_mean[:, 0], ckf_mean[:, 1], label="Estimated trajectory by CKF",  lw=2, color='r')
  pf_ax.plot(pf_mean[:, 0], pf_mean[:, 1], lw=2, label="Estimated trajectory by PF",  color='r')
  enkf_ax.plot(enkf_mean[:, 0], enkf_mean[:, 1], label="Estimated trajectory EnKF",  lw=2, color='r')
  
  ekf_ax.legend()
  ckf_ax.legend()
  pf_ax.legend()
  enkf_ax.legend()
    
  ekf_ax.title.set_text('Eestimated trajectory using Extended Kalman Filter (EKF)')
  ckf_ax.title.set_text('Estimated trajectory using Cubature Kalman Filter (CKF)')
  pf_ax.title.set_text('Estimated trajectory using Particle Filter (PF)')
  enkf_ax.title.set_text('Estimated trajectory using Ensemble Kalman Filter (EnKF)')
  
  fig.suptitle('Visual-Inertial Odometry based trajectory estimation')
  fig.set_figwidth(11)
  fig.set_figheight(9)
  fig.tight_layout()
  
  plt.pause(3)
    
  t_last = 0.
  for t_idx in tqdm(range(1, data.N)):
    t = data.ts[t_idx]
    dt = t - t_last

    ax, ay, az = data.IMU_acc_with_noise[t_idx]
    wx, wy, wz = data.IMU_angular_velocity_with_noise[t_idx]
    u = np.array([
        ax,
        ay,
        az,
        wx,
        wy,
        wz
    ])
    
    z_vo = data.VO_measurements_with_noise[t_idx, :2].reshape(-1, 1)
    
    ekf.predict_setup1_2(u=u.copy(), dt=dt, Q=ekf.Q)
    ckf.predict_setup1_2(u=u.copy(), dt=dt, Q=ckf.Q)
    pf.predict_setup1_2(u=u.copy(), dt=dt, Q=pf.Q)
    enkf.predict_setup1_2(u=u.copy(), dt=dt, Q=enkf.Q)
    
    ekf.update(z=z_vo.copy(), R=ekf.R_vo)
    ckf.update(z=z_vo.copy(), R=ckf.R_vo)
    pf.update(z=z_vo.copy(), R=pf.R_vo)
    enkf.update(z=z_vo.copy().reshape(-1), R=enkf.R_vo)
    
    pf.resample()
    
    x_hat_pf, _ = pf.estimate()

    ekf_mean_.append([ekf.x[0, 0], ekf.x[1, 0]])
    ckf_mean_.append([ckf.x[0, 0], ckf.x[1, 0]])
    pf_mean_.append([x_hat_pf[0], x_hat_pf[1]])
    enkf_mean_.append([enkf.x[0], enkf.x[1]])
    
    ekf_mean = np.array(ekf_mean_)
    ckf_mean = np.array(ckf_mean_)
    pf_mean = np.array(pf_mean_)
    enkf_mean = np.array(enkf_mean_)
    
    ekf_ax.plot(ekf_mean[:, 0], ekf_mean[:, 1], lw=2, label='estimated trajectory', color='r')
    ckf_ax.plot(ckf_mean[:, 0], ckf_mean[:, 1], lw=2, label='estimated trajectory', color='r')
    pf_ax.plot(pf_mean[:, 0], pf_mean[:, 1], lw=2, label='estimated trajectory', color='r')
    enkf_ax.plot(enkf_mean[:, 0], enkf_mean[:, 1], lw=2, label='estimated trajectory', color='r')

    particle_indices = np.linspace(0, pf.N-1, 100, dtype=int).tolist()
    samples_indices = np.linspace(0, enkf.N-1, 100, dtype=int).tolist()

    pf_ax.scatter(
      pf.particles[particle_indices, 0], 
      pf.particles[particle_indices, 1], marker='o', alpha=.01, s=[50], color="pink")
    
    enkf_ax.scatter(
      enkf.samples[samples_indices, 0], 
      enkf.samples[samples_indices, 1], marker='o', alpha=.01, s=[50], color="pink")
    
    plt.pause(interval=0.0001)
    
    ekf_mean_.pop(0)
    ckf_mean_.pop(0)
    pf_mean_.pop(0)
    enkf_mean_.pop(0)
    
    t_last = t
    
  plt.show()
    

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
  
  data = DataLoader(sequence_nr=kitti_drive, 
                  kitti_root_dir=kitti_root_dir, 
                  vo_root_dir=vo_root_dir,
                  noise_vector_dir=noise_vector_dir,
                  vo_dropout_ratio=0.0, 
                  gps_dropout_ratio=0.0,
                  visualize_data=False)
  
  run(data=data)