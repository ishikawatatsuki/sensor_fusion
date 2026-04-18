import os
import sys
import logging
import numpy as np

from .base_filter import BaseFilter
from ..common import (
    State, MotionModel, Pose,
    MeasurementUpdateField,
)
from filterpy.kalman import MerweScaledSigmaPoints

np.random.seed(777)

class UnscentedKalmanFilter(BaseFilter):

    sigma_points = None
    sigma_points = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        alpha = self._get_params(params=self.config.params, key="alpha", default_value=1.0)
        beta = self._get_params(params=self.config.params, key="beta", default_value=2.0) 
        kappa = self._get_params(params=self.config.params, key="kappa", default_value=0.0)
        
        x = self.x.get_state_vector()
        self.N = x.shape[0]
        self.points = MerweScaledSigmaPoints(n=self.N, alpha=alpha, beta=beta, kappa=kappa)

    def _compute_sigma_points(self):
        x = self.x.get_state_vector()
        return self.points.sigma_points(x.reshape(-1,), self.P)

    def kinematics_motion_model(self, u: np.ndarray, dt: float, Q: np.ndarray):
        sigma_points = self._compute_sigma_points()
        p = sigma_points[:, :3]
        v = sigma_points[:, 3:6]
        q = sigma_points[:, 6:10]
        b_w = sigma_points[:, 10:13]
        b_a = sigma_points[:, 13:16]

        a = u[:3]
        w = u[3:]
        a = a.reshape(-1, 1)
        w = w.reshape(-1, 1)

        # Take into account the IMU sensor error
        imu_sensor_error = self.get_imu_sensor_error()

        a -=  imu_sensor_error.acc_bias + self.x.b_a + imu_sensor_error.acc_noise
        w -= imu_sensor_error.gyro_bias + self.x.b_w + imu_sensor_error.gyro_noise

        Omega = State.get_quaternion_update_matrix(w)
        norm_w = self.compute_norm_w(w)

        A = np.cos(norm_w*dt/2) * np.eye(4)
        B = (1/norm_w)*np.sin(norm_w*dt/2) * Omega
        q_k = (np.array(A + B) @ q.T).T # 21x4
        q_k = np.array([q_ / np.linalg.norm(q_) if np.linalg.norm(q_) > 0 else q_  for q_ in q_k])

        R = np.array([self.x.get_rotation_matrix(q_) for q_ in q_k]) #21x3x3
        a_world = (R @ a + self.g)
        acc_val_reshaped = a_world.reshape(a_world.shape[0], a_world.shape[1])
        # v = np.array([Ri @ vi for Ri, vi in zip(R, v)])
        v_k = v + acc_val_reshaped * dt # 21x3

        p_k = p + v_k * dt # + acc_val_reshaped*dt**2 / 2 # 21x3
        
        b_w_k = b_w + imu_sensor_error.gyro_bias.flatten()
        b_a_k = b_a + imu_sensor_error.acc_bias.flatten()

        self.sigma_points = np.concatenate([
            p_k,
            v_k,
            q_k,
            b_w_k,
            b_a_k
        ], axis=1) # 21x10
        
        x = (self.points.Wm @ self.sigma_points).reshape(-1, 1) # 10x1
        self.x = State.get_new_state_from_array(x)
        
        P = np.zeros((self.N, self.N)) # 10x10
        for i, sigma_point in enumerate(self.sigma_points):
            var = sigma_point.reshape(-1, 1) - x
            P += self.points.Wc[i] * (var @ var.T)
        self.P = P + Q # 10x10 additive process noise

    def velocity_motion_model(self, u: np.ndarray, dt: float, Q: np.ndarray):
        sigma_points = self._compute_sigma_points()

        p = sigma_points[:, :3]
        v = sigma_points[:, 3:6]
        q = sigma_points[:, 6:10]
        b_w = sigma_points[:, 10:13]
        b_a = sigma_points[:, 13:16]

        a = u[:3]
        w = u[3:]
        wx, wy, wz = w + 1e-17
        a = a.reshape(-1, 1)
        w = w.reshape(-1, 1)
        
        # Take into account the IMU sensor error
        imu_sensor_error = self.get_imu_sensor_error()

        a -= imu_sensor_error.acc_bias + self.x.b_a + imu_sensor_error.acc_noise
        w -= imu_sensor_error.gyro_bias + self.x.b_w + imu_sensor_error.gyro_noise
        
        omega = State.get_quaternion_update_matrix(w)
        norm_w = self.compute_norm_w(w)
        
        A = np.cos(norm_w*dt/2) * np.eye(4)
        B = (1/norm_w)*np.sin(norm_w*dt/2) * omega
        q_k = (np.array(A + B) @ q.T).T # 21x4
        q_k = np.array([q_ / np.linalg.norm(q_) if np.linalg.norm(q_) > 0 else q_  for q_ in q_k])
        
        R = np.array([self.x.get_rotation_matrix(q_) for q_ in q])
        phi, _, psi = np.array([State.get_euler_angle_from_quaternion_vector(q_row.reshape(-1, 1)) for q_row in q]).T
        
        
        a_world = (R @ a + self.g)
        acc_val_reshaped = a_world.reshape(a_world.shape[0], a_world.shape[1])
        v_k = v + acc_val_reshaped * dt

        vf = self.get_forward_velocity(v_k)
        rx = vf / wx  # turning radius for x axis
        rz = vf / wz  # turning radius for z axis
        dphi = wx * dt
        dpsi = wz * dt
        dpx = - rz * np.sin(psi) + rz * np.sin(psi + dpsi)
        dpy = + rz * np.cos(psi) - rz * np.cos(psi + dpsi)
        dpz = + rx * np.cos(phi) - rx * np.cos(phi + dphi)
        
        dp = np.vstack([dpx, dpy, dpz]).T

        p_k = p + dp

        b_w_k = b_w + imu_sensor_error.gyro_bias.flatten()
        b_a_k = b_a + imu_sensor_error.acc_bias.flatten()

        self.sigma_points = np.concatenate([
            p_k,
            v_k,
            q_k,
            b_w_k,
            b_a_k
        ], axis=1)
        
        x = (self.points.Wm @ self.sigma_points).reshape(-1, 1)
        self.x = State.get_new_state_from_array(x)
        
        P = np.zeros((self.N, self.N)) # 3x3
        for i, sigma_point in enumerate(self.sigma_points):
            var = sigma_point.reshape(-1, 1) - x
            P += self.points.Wc[i] * (var @ var.T)
        self.P = P + Q # 10x10 additive process noise
    
    def measurement_update(self, data: MeasurementUpdateField):
        z = data.z
        R = data.R
        sensor_type = data.sensor_type
        
        z_dim = z.shape[0]
        x = self.x.get_state_vector()
        self.H = self.transition_matrix_helper.get_transition_matrix(state=self.x, data=data)
        mask = self.get_innovation_mask(sensor_type=sensor_type, z_dim=z_dim).reshape(-1, 1)
        
        sigma_points = self._compute_sigma_points()
        y_sigma_points = sigma_points @ self.H.T
        y_hat = (self.points.Wm @ y_sigma_points).reshape(-1, 1)

        x_dim = sigma_points.shape[1]
        z_dim = y_sigma_points.shape[1]

        # compute covariance matrix for residuals
        P_y = np.zeros((z_dim, z_dim)) # 2x2
        for i, y_sigma_point in enumerate(y_sigma_points):
            var_y = y_sigma_point.reshape(-1, 1) - y_hat
            P_y += self.points.Wc[i] * (var_y @ var_y.T)
        P_y += R # additive measurement noise
        
        # compute cross-covariance matrix 
        P_xy = np.zeros((x_dim, z_dim)) # 10x2
        for idx in range(self.N):
            var_x = sigma_points[idx].reshape(-1, 1) - x
            var_y = y_sigma_points[idx].reshape(-1, 1) - y_hat
            P_xy += self.points.Wc[idx] * (var_x @ var_y.T)
            
        # compute kalman gain
        self.K = P_xy @ np.linalg.inv(P_y)
        
        # compute residual
        self.innovation = z - y_hat
        _innovation = self.K @ self.innovation
        _innovation *= mask
        # update state vector and error covariance matrix
        x = x + _innovation
        self.x = State.get_new_state_from_array(x)
        
        z_ = np.dot(self.H, x)
        self.residual = z - z_
        
        self.P = self.P - self.K @ P_y @ self.K.T

        # self.innovations.append(np.sum(self.innovation))

    def set_ensembles(self):
        return None