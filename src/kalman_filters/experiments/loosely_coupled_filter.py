# references: 
# [1] https://docs.ufpr.br/~danielsantos/ProbabilisticRobotics.pdf


import sys
if __name__ == "__main__":
    sys.path.append('../../src')

import numpy as np
from numpy import cos, sin, arcsin, arctan2
from tqdm import tqdm
from configs import MeasurementDataEnum, SetupEnum, FilterEnum, NoiseTypeEnum, Configs
from utils.error_report import get_error_report
import matplotlib.pyplot as plt

if __name__ == "__main__":
    from base_filter import BaseFilter
else:
    from ..base_filter import BaseFilter

# Since symbolic python is too slow, pure numpy implementation is defined.
class LooselyCoupledFilter():
    """Extended Kalman Filter
    for vehicle whose motion is modeled as eq. (5.9) in [1]
    and with observation of its 2d location (x, y)
    """
    x = None
    P = None
    Q = None
    R_vo = None
    R_gps = None
    g = np.array([[0],[0],[9.81]])
    
    def __init__(self, x, P, H, q, r_vo, r_gps):
        self.P = P
        self.H = H
        self.x = x
        self.Q = self.get_diagonal_matrix(q)
        self.R_vo = self.get_diagonal_matrix(r_vo)
        self.R_gps = self.get_diagonal_matrix(r_gps)
        
    def _get_rotation_matrix(self, w):
      roll, pitch, yaw = w
      return np.array([
        [cos(yaw) * cos(pitch), cos(yaw)*sin(pitch)*sin(roll)-sin(yaw)*cos(roll), cos(yaw)*sin(pitch)*cos(roll)+sin(yaw)*sin(roll)],
        [sin(yaw)*cos(pitch), sin(yaw)*sin(pitch)*sin(roll)+cos(pitch)*cos(roll), sin(yaw)*sin(pitch)*cos(roll)-cos(yaw)*sin(roll)],
        [-sin(pitch), cos(pitch)*sin(roll), cos(pitch)*cos(roll)]
      ]) # 3x3 matrix

    # https://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf Figure 1
    def predict(self, u, dt, Q):
      
      a = u[:3]
      w = u[3:]
      R = self._get_rotation_matrix(w)
      if R[3, 1] == 1 or R[3, 1] == -1:
        d_pitch = - arcsin(R[3,1])
        d_roll = arctan2(R[3, 2]/cos(d_pitch), R[3, 3]/cos(d_pitch)) * dt
        d_yaw = arctan2(R[2, 1]/cos(d_pitch), R[1, 1]/cos(d_pitch)) * dt
        d_pitch *= dt
      else:
        d_yaw = 0
        if R[3, 1] == -1:
          d_pitch = np.pi / 2 * dt
          d_roll = d_yaw + arctan2(R[1, 2], R[1, 3]) * dt
        else:
          d_pitch = - np.pi / 2 * dt
          d_roll = - d_yaw + arctan2(-R[1, 2], -R[1, 3])
      
      d_v = (R @ a - self.g) * dt
      
      d_w = np.array([d_roll, d_pitch, d_yaw]).reshape(-1, 1)
      d_x = np.concatenate([d_v, d_w], axis=1)
      self.x += d_x
      
      
        
    def update(self, z, R):
      pass

    def run(self, data):

        # measurement noise
        R_vo = self.R_vo
        R_gps = self.R_gps
        # process noise
        Q = self.Q

        mu_fv = [self.x[0, 0], ]
        mu_pitch = [self.x[1, 0], ]
        mu_roll = [self.x[2, 0], ]
        mu_yaw = [self.x[3, 0], ]
        
        t_last = 0.

        for t_idx in range(1, 2):
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

            
            
            t_last = t

        print("Done")


if __name__ == "__main__":
    import os
    from datetime import datetime
    from data_loader import DataLoader

    root_path = "../../"
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
                    gps_dropout_ratio=0.0)
    _, _, _, q1, r_vo1, r_gps1 = data.get_initial_data(setup=SetupEnum.SETUP_1,filter_type=FilterEnum.EKF, noise_type=NoiseTypeEnum.CURRENT)
    
    x = np.array([
      0.0, #forward velocity (x axis)
      0.0, #leftward velocity (y axis)
      0.0, #upward velocity (z axis)
      0.0, #roll
      0.0, #pitch
      0.0  #yaw
    ])
    P = np.array([
      [0.1, 0., 0., 0., 0., 0.],
      [0., 0.1, 0., 0., 0., 0.],
      [0., 0., 0.1, 0., 0., 0.],
      [0., 0., 0., 0.1, 0., 0.],
      [0., 0., 0., 0., 0.1, 0.],
      [0., 0., 0., 0., 0., 0.1],
    ])
    H = np.array([])

    kf = LooselyCoupledFilter(
        x=x.copy(), 
        P=P.copy(), 
        H=H.copy(),
        q=q1,
        r_vo=r_vo1,
        r_gps=r_gps1
    )