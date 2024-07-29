import sys
if __name__ == "__main__":
    sys.path.append('../../src')

import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from utils.error_report import get_error_report
from configs import MeasurementDataEnum, SetupEnum, FilterEnum, NoiseTypeEnum
from filterpy.kalman import MerweScaledSigmaPoints
from ahrs import Quaternion
from datetime import datetime

if __name__ == "__main__":
    from base_filter import BaseFilter
else:
    from ..base_filter import BaseFilter


np.random.seed(777)

class LocalUnscentedKalmanFilter(BaseFilter):
  
  sigma_points = None
  
  def __init__(
    self,
    q,
    r
  ):
    """ 
    Args:
        q process noise vector: 
        r measurement noise vector: 
    """
    p = np.array([0., 0., 0.]).reshape(-1, 1)
    v = np.array([0., 0., 0.]).reshape(-1, 1)
    a = np.array([0., 0., 0.]).reshape(-1, 1)
    theta = np.array([0., 0., 0.]).reshape(-1, 1)
    omega = np.array([0., 0., 0.]).reshape(-1, 1)
    self.x = np.concatenate([p, v, a, theta, omega], axis=0)
    self.x_hat = np.concatenate([self.x[:3], self.x[:3]], axis=1)
    
    self.N = self.x.shape[0]
    self.P = np.eye(self.N) * 0.01
    self.H = np.array([
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.], #a x
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.], #a y
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.], #a z
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.], #omega x
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.], #omega y
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.], #omega z
    ])
    self.points = MerweScaledSigmaPoints(
      n=self.N, 
      alpha=1e-3, 
      beta=2, 
      kappa=0)
    
    self.set_noise_matrix(q, r)
    
  def set_noise_matrix(self, q, r):
    self.Q = self.get_diagonal_matrix(q)
    self.R = self.get_diagonal_matrix(r)
  
  def compute_sigma_points(self):
    return self.points.sigma_points(self.x.reshape(-1,), self.P)
      
  def prediction_step(self, dt, z):
    # time update
    _sigma_points = self.compute_sigma_points() # 2N+1 x N
    _p = _sigma_points[:, :3]
    _v = _sigma_points[:, 3:6]
    _a = _sigma_points[:, 6:9]
    _theta = _sigma_points[:, 9:12]
    _omega = _sigma_points[:, 12:]
    
    p = _p + _v * dt + 0.5 * dt**2 * _a
    v = _v + dt * _a
    a = _a
    
    theta = _theta + dt * _omega
    omega = _omega

    self.sigma_points = np.concatenate([p, v, a, theta, omega], axis=1)
    self.x = (self.points.Wm @ self.sigma_points).reshape(-1, 1)
    P = np.zeros((self.N, self.N))
    for i, sigma_point in enumerate(self.sigma_points):
        P += self.points.Wc[i] * (sigma_point.reshape(-1, 1) - self.x) @ (sigma_point.reshape(-1, 1) - self.x).T
    self.P = P + self.Q # 15x15
    
    # measurement update
    _sigma_points = self.compute_sigma_points()
    y_sigma_points = _sigma_points @ self.H.T # 2N+1xN @ Nx6
    y_hat = (self.points.Wm @ y_sigma_points).reshape(-1, 1)

    x_dim = _sigma_points.shape[1]
    z_dim = y_sigma_points.shape[1]
    P_y = np.zeros((z_dim, z_dim)) # 6x6
    for i, y_sigma_point in enumerate(y_sigma_points):
        P_y += self.points.Wc[i] * (y_sigma_point.reshape(-1, 1) - y_hat) @ (y_sigma_point.reshape(-1, 1) - y_hat).T
    P_y += self.R # additive measurement noise
    
    P_xy = np.zeros((x_dim, z_dim)) # 10x2
    for idx in range(self.N):
        P_xy += self.points.Wc[idx] * ((_sigma_points[idx].reshape(-1, 1) - self.x) @ (y_sigma_points[idx].reshape(-1, 1) - y_hat).T)
        
    K = P_xy @ np.linalg.inv(P_y)
    residual = z.reshape(-1, 1) - y_hat
    self.x = self.x + K @ residual
    self.P = self.P - K @ P_y @ K.T
    
  
  def correction_step(self, x_master, P_master, beta):
    self.x = x_master.copy()
    self.P = P_master.copy() / beta
    
  def get_position(self):
    self.x_hat = np.concatenate([self.x_hat[:3, 1:], self.x.copy()[:3]], axis=1)
    return self.x_hat

class IMU:
  t_last = None
  
  def __init__(self, label, q, r):
    self.filter = LocalUnscentedKalmanFilter(
      q=q,
      r=r
    )
    self.label=label

class FederatedUnscentedKalmanFilter(BaseFilter):
  
  t_last = None
  sigma_points = None
  
  n_iter = 0
  
  def __init__(
    self,
    imu_list,
    r,
    q
  ):
    self.imus = {}
    self.beta = {}
    self.imu_labels = []
    
    initial_beta = 1 / len(imu_list)

    for imu in imu_list:
      self.imus[imu.label] = imu
      self.beta[imu.label] = initial_beta
      self.imu_labels.append(imu.label)
    
    p = np.array([0., 0., 0.]).reshape(-1, 1)
    v = np.array([0., 0., 0.]).reshape(-1, 1)
    a = np.array([0., 0., 0.]).reshape(-1, 1)
    theta = np.array([0., 0., 0.]).reshape(-1, 1)
    omega = np.array([0., 0., 0.]).reshape(-1, 1)
    self.x = np.concatenate([p, v, a, theta, omega], axis=0)
    self.x_hat = np.concatenate([self.x[:3], self.x[:3]], axis=1)
    
    self.N = self.x.shape[0]
    self.P = np.eye(self.N) * 0.01
    
    self.points = MerweScaledSigmaPoints(
      n=self.N, 
      alpha=1e-3, 
      beta=2, 
      kappa=0)

    self.set_noise_matrix(q, r)
    
  def set_noise_matrix(self, q, r):
    self.Q = self.get_diagonal_matrix(q)
    self.R = self.get_diagonal_matrix(r)
    
  def compute_sigma_points(self):
    return self.points.sigma_points(self.x.reshape(-1,), self.P)
      
  def local_prediction_step(self, imu_label, z):
    try:
      imu = self.imus[imu_label]
      now = datetime.now()
      if imu.t_last:
        dt = (now - imu.t_last).total_seconds()
        if dt < 2.:
          imu.filter.prediction_step(dt=dt, z=z)
      
      imu.t_last = now
    except KeyError:
      pass
  
  def correction_step(self):
    # propagate master state
    self.master_prediction_step()
    # combine all imu states and compute weight, denoted as beta
    self.master_fusion()
    # update each imu's state and error state covariance matrix with the weight
    self.local_correction_step()
    # self.n_iter += 1
    # print(self.n_iter)
    self.x_hat = np.concatenate([self.x_hat[:3, 1:], self.x.copy()[:3]], axis=1)

  def master_prediction_step(self):
    now = datetime.now()
    if self.t_last:
      dt = (now - self.t_last).total_seconds()
      _sigma_points = self.compute_sigma_points() # 2N+1 x N
      _p = _sigma_points[:, :3]
      _v = _sigma_points[:, 3:6]
      _a = _sigma_points[:, 6:9]
      _theta = _sigma_points[:, 9:12]
      _omega = _sigma_points[:, 12:]
      
      p = _p + _v * dt + 0.5 * dt**2 * _a
      v = _v + dt * _a
      a = _a
      theta = _theta + dt * _omega
      omega = _omega

      self.sigma_points = np.concatenate([p, v, a, theta, omega], axis=1)
      self.x = (self.points.Wm @ self.sigma_points).reshape(-1, 1)
      P = np.zeros((self.N, self.N))
      for i, sigma_point in enumerate(self.sigma_points):
          P += self.points.Wc[i] * (sigma_point.reshape(-1, 1) - self.x) @ (sigma_point.reshape(-1, 1) - self.x).T
      self.P = P + self.Q # 15x15
    
    self.t_last = now
      
  def master_fusion(self):
    inv_P_local = np.zeros(self.P.shape)
    x_local = np.zeros(self.x.shape)
    eig_trace_sum = 0.
    for label in self.imu_labels:
      _inv_P_local = np.linalg.inv(self.imus[label].filter.P)
      inv_P_local += _inv_P_local
      x_local += _inv_P_local @ self.imus[label].filter.x
      
      eigenvalues = np.linalg.eigvals(self.imus[label].filter.P)
      trace_eigenvalues = np.sum(eigenvalues)
      self.beta[label] = trace_eigenvalues
      eig_trace_sum += trace_eigenvalues
      
    eig_master = np.linalg.eigvals(self.P)
    eig_trace_sum += np.sum(eig_master)
    
    # eig_trace_sum /= len(self.imu_labels)
    # x_local /= len(self.imu_labels)
    
    self.x = np.linalg.inv(self.P) @ self.x + x_local
    inv_P = np.linalg.inv(self.P) + inv_P_local
    self.P = np.linalg.inv(inv_P)
    
    for label in self.imu_labels:
      self.beta[label] = self.beta[label] / eig_trace_sum
  
  def local_correction_step(self):
    for label in self.imu_labels:
      self.imus[label].filter.correction_step(
        x_master=self.x, 
        P_master=self.P, 
        beta=self.beta[label]
      )
  
  def get_master_position(self):
    return self.x_hat
  
if __name__ == "__main__":
  
  
  q = np.ones(15) * 0.1
  r = np.ones(6) * 0.1
  imu1 = IMU("imu1", q=q, r=r)
  imu2 = IMU("imu2", q=q, r=r)
  imus = [imu1, imu2]
  fukf = FederatedUnscentedKalmanFilter(imu_list=imus, r=r, q=q)
  
  fukf.local_prediction_step("imu1", z=np.array([0.0, 0.0, -9.81, 0.2, 0.1, 0.1]))
  fukf.correction_step()
  
  x, y, z = fukf.get_master_position()
  
  ax = plt.figure().add_subplot(projection='3d')

  ax.set_xlabel('$X$', fontsize=14)
  ax.set_ylabel('$Y$', fontsize=14)
  ax.set_zlabel('$Z$', fontsize=14)
  
  num_points = 1000
  theta_max = 8 * np.pi
  zs = np.linspace(0, 10, num_points)
  theta = np.linspace(0, theta_max, num_points)
  r = zs

  # Helix coordinates
  xs = r * np.sin(theta)
  ys = r * np.cos(theta)
  for i in range(1, num_points):
    x1 = [xs[i-1], xs[i]]
    y1 = [ys[i-1], ys[i]]
    z1 = [zs[i-1], zs[i]]
    ax.plot(x1, y1, z1, label='3D Helix', color='black')
    plt.pause(interval=0.001)
    
  ax.legend()  
  
  # local_ukf = LocalUnscentedKalmanFilter(
  #   q=q,
  #   r=r
  # )
  # z = np.array([0.0, 0.0, -9.81, 0.2, 0.1, 0.1])
  # dt = 0.1
  # local_ukf.prediction_step(dt=dt, z=z)
  
  
  # local_ukf.correction_step()