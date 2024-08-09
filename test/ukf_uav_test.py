import os
import sys
if __name__ == "__main__":
    sys.path.append('../src')
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from decimal import Decimal, getcontext
from utils import lla_to_enu, lla_to_ned
from utils.error_report import get_error_from_list, get_error_report
from configs.configs import MeasurementDataEnum, SetupEnum, SamplingEnum, ErrorEnum
from kalman_filters import (
  ExtendedKalmanFilter,
  UnscentedKalmanFilter,
  ParticleFilter, ResamplingAlgorithms,
  EnsembleKalmanFilter,
  CubatureKalmanFilter
)

root_path = "../"

file_export_path = os.path.join(root_path, "exports/UAV")

uav_root_path = os.path.join(root_path, "data/UAV/log0000")
uav_reference_path = os.path.join(root_path, "data/UAV/v3/combined_log0000.csv")

px4_root_path = os.path.join(uav_root_path, "px4/13_07_41")
px4_imu0_path = os.path.join(px4_root_path, "imu_combined/log0000_px4_imu0_combined.csv")
px4_imu1_path = os.path.join(px4_root_path, "imu_combined/log0000_px4_imu1_combined.csv")
px4_gps_path = os.path.join(px4_root_path, "13_07_41_sensor_gps_0.csv")
px4_vo_path = os.path.join(px4_root_path, "13_07_41_vehicle_visual_odometry_0.csv")
imu0_bias_path = os.path.join(px4_root_path, "13_07_41_estimator_sensor_bias_0.csv")
imu1_bias_path = os.path.join(px4_root_path, "13_07_41_estimator_sensor_bias_1.csv")

voxl_root_path = os.path.join(uav_root_path, "run/mpa")
voxl_imu0_path = os.path.join(voxl_root_path, "imu0/data.csv")
voxl_imu1_path = os.path.join(voxl_root_path, "imu1/data.csv")
voxl_qvio_path = os.path.join(voxl_root_path, "qvio/data.csv")

ref_df = pd.read_csv(uav_reference_path)

px4_imu0_df = pd.read_csv(px4_imu0_path)
px4_imu1_df = pd.read_csv(px4_imu1_path)
px4_gps_df = pd.read_csv(px4_gps_path)
px4_vo_df = pd.read_csv(px4_vo_path)
imu0_bias_df = pd.read_csv(imu0_bias_path)
imu1_bias_df = pd.read_csv(imu1_bias_path)

voxl_imu0_df = pd.read_csv(voxl_imu0_path)
voxl_imu1_df = pd.read_csv(voxl_imu1_path)
voxl_qvio_df = pd.read_csv(voxl_qvio_path)
# Convert lla(Int) into lla(float)

gps_pose_columns = ['lon', 'lat', 'alt']
gps_ned_pose_columns = ['north', 'east', 'down']
vo_pos_columns = ['position[0]', 'position[1]', 'position[2]']
vo_pos_var_columns = ['position_variance[0]', 'position_variance[1]', 'position_variance[2]']

time_update = ['voxl_imu0', 'voxl_imu0', 'px4_imu0', 'px4_imu1', 'px4_imu0_bias', 'px4_imu1_bias']
measurement_update = ['voxl_vo', 'px4_vo', 'px4_gps']

SENSOR_SLIP_THRESHOLD = 2 # 2 seconds

getcontext().prec = 10
def int2float_lla(x):
    return float(Decimal(x / Decimal(10**(len(str(x)) - 2))))

def int_lla_to_float_lla(df):
    df['lat'] = df['lat'].apply(lambda x: int2float_lla(x))
    df['lon'] = df['lon'].apply(lambda x: int2float_lla(x))
    df['alt'] = df['alt'].apply(lambda x: int2float_lla(x))
    
# Convert lla into North-East-Down coordinate in meter

def get_ned_coord(df):    
    origin = df[['lon', 'lat', 'alt']].iloc[0].values
    ned_pose = lla_to_ned(df[['lon', 'lat', 'alt']].values.T, origin).T
    df = pd.concat([
        df,
        pd.DataFrame(ned_pose, columns=['north', 'east', 'down'])
    ], axis=1)
    return df
  

def print_error_report(report, title):
    print(f"----- {title} -----")
    print(f"Mean Absolute Error: {report[ErrorEnum.MAE]}")
    print(f"Root Mean Squared Error: {report[ErrorEnum.RMSE]}")
    print(f"Maximum Error: {report[ErrorEnum.MAX]}")
    print("")



int_lla_to_float_lla(px4_gps_df)
px4_gps_df = get_ned_coord(px4_gps_df)
    
def get_config(timestamp):

    return {
        'voxl_imu0': {
            'index': 0,
            'last_timestamp': timestamp,
            'columns': ['AX(m/s2)', 'AY(m/s2)', 'AZ(m/s2)', 'GX(rad/s)', 'GY(rad/s)', 'GZ(rad/s)'],
            'df': voxl_imu0_df,
        },
        'voxl_imu1': {
            'index': 0,
            'last_timestamp': timestamp,
            'columns': ['AX(m/s2)', 'AY(m/s2)', 'AZ(m/s2)', 'GX(rad/s)', 'GY(rad/s)', 'GZ(rad/s)'],
            'df': voxl_imu1_df,
        },
        'px4_imu0': {
            'index': 0,
            'last_timestamp': timestamp,
            'columns': ['AX(m/s2)', 'AY(m/s2)', 'AZ(m/s2)', 'GX(rad/s)', 'GY(rad/s)', 'GZ(rad/s)'],
            'df': px4_imu0_df,
        },
        'px4_imu1': {
            'index': 0,
            'last_timestamp': timestamp,
            'columns': ['AX(m/s2)', 'AY(m/s2)', 'AZ(m/s2)', 'GX(rad/s)', 'GY(rad/s)', 'GZ(rad/s)'],
            'df': px4_imu1_df,
        },
        'px4_gps': {
            'index': 0,
            'last_timestamp': timestamp,
            'columns': ['lat', 'lon', 'alt', 'north', 'east', 'down', 'vel_m_s', 'vel_n_m_s', 'vel_e_m_s', 'vel_d_m_s'],
            'df': px4_gps_df,
        },
        'px4_vo': {
            'index': 0,
            'last_timestamp': timestamp,
            'columns': ['position[0]', 'position[1]', 'position[2]', 
                        'q[0]', 'q[1]', 'q[2]', 'q[3]', 
                        'velocity[0]','velocity[1]', 'velocity[2]', 
                        'angular_velocity[0]', 'angular_velocity[1]', 'angular_velocity[2]', 
                        'position_variance[0]', 'position_variance[1]', 'position_variance[2]',
                        'orientation_variance[0]', 'orientation_variance[1]', 'orientation_variance[2]', 
                        'velocity_variance[0]', 'velocity_variance[1]', 'velocity_variance[2]'],
            'df': px4_vo_df,
        },
        'px4_imu0_bias': {
            'index': 0,
            'last_timestamp': timestamp,
            'columns': ['gyro_bias[0]', 'gyro_bias[1]', 'gyro_bias[2]', 'accel_bias[0]', 'accel_bias[1]', 'accel_bias[2]'],
            'df': imu0_bias_df,
        },
        'px4_imu1_bias': {
            'index': 0,
            'last_timestamp': timestamp,
            'columns': ['gyro_bias[0]', 'gyro_bias[1]', 'gyro_bias[2]', 'accel_bias[0]', 'accel_bias[1]', 'accel_bias[2]'],
            'df': imu1_bias_df,
        },
        'voxl_vo': {
            'index': 0,
            'last_timestamp': timestamp,
            'columns': voxl_qvio_df.columns.tolist(),
            'df': voxl_qvio_df,
        },
    }

def main():
  # df1 = ref_df.loc[
  #           (ref_df["device"] == "voxl_imu0") | 
  #           (ref_df["device"] == "px4_vo") |
  #           (ref_df["device"] == "px4_gps")]
  
  df1 = ref_df.loc[
            (ref_df["device"] == "voxl_imu0") | 
            (ref_df["device"] == "px4_gps")]
  
  x = np.array([
      [px4_gps_df.iloc[0]['north']],
      [px4_gps_df.iloc[0]['east']],
      [px4_gps_df.iloc[0]['down']],
      [0.],
      [0.],
      [0.],
      [1.],
      [0.],
      [0.],
      [0.]
  ])
  P = np.eye(x.shape[0]) * 0.1
  H = np.array([
      [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]
  ])
  q = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01])
  r_gps = np.array([0.1, 0.1, 0.1])
  r_vo_default = np.array([0.1, 0.1, 0.1])

  alpha_setup = 10.0
  beta_setup = 2.0
  kappa_setup = 0.

  ukf = UnscentedKalmanFilter(
      x=x.copy(), 
      P=P.copy(),
      H=H.copy(), 
      q=q,
      r_vo=r_vo_default,
      r_gps=r_gps,
      alpha=alpha_setup, 
      beta=beta_setup, 
      kappa=kappa_setup
  )
  
  t_last = df1.iloc[0]['timestamp']
  df_reference = get_config(timestamp=t_last)
  original_position = px4_gps_df[['lon', 'lat', 'alt']].iloc[0].values

  ukf_mu_x = [x[0][0]]
  ukf_mu_y = [x[0][0]]
  ukf_mu_z = [x[0][0]]

  N = len(df1.values)
  for idx in tqdm(range(0, N)):
      data_ref = df1.iloc[idx]
      sensor = df_reference[data_ref['device']]
      data = sensor['df'].iloc[sensor['index']][sensor['columns']]
      dt = (data_ref['timestamp'] - sensor['last_timestamp']) / 1_000_000
      # dt = (data_ref['timestamp'] - t_last) / 1_000_000 # delta time in second

      if dt < SENSOR_SLIP_THRESHOLD:
          # Call Kalman Filter step
          if data_ref['device'] in time_update:
              # Call Time update step
              ukf.predict_setup1_2(u=data.values, dt=dt, Q=ukf.Q)
              
          elif data_ref['device'] in measurement_update:
              # Call Measurement update step
              if data_ref['device'] == 'px4_gps':
                  x_hat = ukf.x.copy()
                  ukf_mu_x.append(x_hat[0, 0])
                  ukf_mu_y.append(x_hat[1, 0])
                  ukf_mu_z.append(x_hat[2, 0])
                  
                  gps_ned = lla_to_ned(
                      sensor["df"].iloc[sensor['index']:sensor['index'] + 1][gps_pose_columns].values.T,
                      original_position
                  ).T[0]
                  z = np.array([
                      gps_ned[0],
                      gps_ned[1],
                      gps_ned[2],
                  ])
                  ukf.update(z=z, R=ukf.R_gps)
                  
              elif data_ref['device'] == 'px4_vo':
                  z = sensor["df"].iloc[sensor["index"]][vo_pos_columns].values
                  r_vo = sensor["df"].iloc[sensor["index"]][vo_pos_var_columns].values
                  if np.isnan(r_vo.sum()):
                      R_vo = ukf.R_vo
                  else:
                      R_vo = np.eye(r_vo.shape[0]) * r_vo
                  
                  ukf.update(z=z, R=R_vo)
                  

      # t_last = data_ref['timestamp']
      sensor['last_timestamp'] = data_ref['timestamp']
      sensor['index'] += 1
      
  pf_imu_vo_gps_error = get_error_report(px4_gps_df[gps_ned_pose_columns].values, 
                              np.array([ukf_mu_x, ukf_mu_y, ukf_mu_z]).T)

  print_error_report(
      report=pf_imu_vo_gps_error,
      title="Deviation between PF(IMU, VO+GPS) estimated pose and GPS"
  )

      
  fig = plt.figure()
  ax1 = fig.add_subplot(111, projection='3d')
  ax1.plot(px4_gps_df['north'].values, 
          px4_gps_df['east'].values, 
          px4_gps_df['down'].values, 
          label='UAV trajectory (ground truth)', 
          color='black')
  ax1.plot(ukf_mu_x, ukf_mu_y, ukf_mu_z, label='Estimated trajectory (IMU, VO+GPS)', color='blue')

  ax1.legend()
  ax1.set_xlabel('$X$', fontsize=14)
  ax1.set_ylabel('$Y$', fontsize=14)
  ax1.set_zlabel('$Z$', fontsize=14)
  fig.tight_layout()
  plt.pause(5)

if __name__ == "__main__":
  main()