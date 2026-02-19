import os
import sys
import logging
import numpy as np

from common.constants import MAX_NUM_PARTICLES_TO_VISUALIZE

from .base_filter import BaseFilter

from ..common import (
    State, Pose, MotionModel, TimeUpdateField, MeasurementUpdateField,
    MeasurementUpdateField,
)

class EnsembleKalmanFilter(BaseFilter):

    def __init__(
            self,
            *args,
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        
        self.ensemble_size = self._get_params(params=self.config.params, key="ensemble_size", default_value=1024)
        
        x = self.x.get_state_vector()
        self.x_dim = x.shape[0]
        self.samples = self._generate_ensembles(mean=x)
        
    def _generate_ensembles(self, mean: np.ndarray):
        
        ensembles = np.random.multivariate_normal(
            mean=mean.reshape(-1), 
            cov=self.P,
            size=self.ensemble_size
            )
        ensembles[:, 6:10] = np.repeat(mean[6:10].reshape(1, -1), self.ensemble_size, axis=0) # Initialize quaternions without noise
        return ensembles

    def kinematics_motion_model(self, u: np.ndarray, dt: float, Q: np.ndarray):
        
        p = self.samples[:, :3]
        v = self.samples[:, 3:6]
        q = self.samples[:, 6:10]
        b_w = self.samples[:, 10:13]
        b_a = self.samples[:, 13:16]

        a = u[:3]
        w = u[3:]
        a = a.reshape(-1, 1)
        w = w.reshape(-1, 1)

        # Take into account the IMU sensor error
        imu_sensor_error = self.get_imu_sensor_error()

        a -=  imu_sensor_error.acc_bias + self.x.b_a + imu_sensor_error.acc_noise
        w -= imu_sensor_error.gyro_bias + self.x.b_w + imu_sensor_error.gyro_noise

        R = np.array([self.x.get_rotation_matrix(q_) for q_ in q]) #Nx3x3
        omega = self.get_quaternion_update_matrix(w) 
        norm_w = self.compute_norm_w(w)

        A = np.cos(norm_w*dt/2) * np.eye(4)
        B = (1/norm_w)*np.sin(norm_w*dt/2) * omega

        a_world = (R @ a + self.g)
        acc_val_reshaped = a_world.reshape(a_world.shape[0], a_world.shape[1])
        # v = np.array([Ri @ vi for Ri, vi in zip(R, v)])
        p_k = p + v * dt # + acc_val_reshaped*dt**2 / 2 # Nx3
        v_k = v + acc_val_reshaped * dt # Nx3
        q_k = (np.array(A + B) @ q.T).T # Nx4
        q_k = np.array([q_ / np.linalg.norm(q_) if np.linalg.norm(q_) > 0 else q_  for q_ in q_k])

        b_w_k = b_w + imu_sensor_error.gyro_bias.flatten()
        b_a_k = b_a + imu_sensor_error.acc_bias.flatten()

        process_noise_cov = np.random.multivariate_normal(
                                mean=np.zeros(self.x_dim), 
                                cov=Q, 
                                size=self.ensemble_size)
        
        self.samples = np.concatenate([
            p_k,
            v_k,
            q_k,
            b_w_k,
            b_a_k
        ], axis=1) + process_noise_cov # Nx10
        
        x = np.mean(self.samples, axis=0)
        self.x = State.get_new_state_from_array(x)
        
    def velocity_motion_model(self, u: np.ndarray, dt: float, Q: np.ndarray):
        """ 
            move according to control input u (heading change, velocity) with noise std
            u: control input vector
            dt: delta time
            Q: process noise matrix
        """
        p = self.samples[:, :3]
        v = self.samples[:, 3:6]
        q = self.samples[:, 6:10]
        b_w = self.samples[:, 10:13]
        b_a = self.samples[:, 13:16]
        
        a = u[:3]
        w = u[3:]
        wx, _, wz = w + 1e-17
        a = a.reshape(-1, 1)
        w = w.reshape(-1, 1)
        
        # Take into account the IMU sensor error
        imu_sensor_error = self.get_imu_sensor_error()

        a -= imu_sensor_error.acc_bias + self.x.b_a + imu_sensor_error.acc_noise
        w -= imu_sensor_error.gyro_bias + self.x.b_w + imu_sensor_error.gyro_noise

        R = np.array([self.x.get_rotation_matrix(q_) for q_ in q])
        omega = self.get_quaternion_update_matrix(w)
        norm_w = self.compute_norm_w(w)

        A = np.cos(norm_w*dt/2) * np.eye(4)
        B = (1/norm_w)*np.sin(norm_w*dt/2) * omega

        phi, _, psi = np.array([self.get_euler_angle_from_quaternion(q_row.reshape(-1, 1)) for q_row in q]).T

        vf = self.get_forward_velocity(v)
        
        a_world = (R @ a + self.g)
        acc_val_reshaped = a_world.reshape(a_world.shape[0], a_world.shape[1])

        rx = vf / wx  # turning radius for x axis
        rz = vf / wz  # turning radius for z axis
        
        dphi = wx * dt
        dpsi = wz * dt
        dpx = - rz * np.sin(psi) + rz * np.sin(psi + dpsi)
        dpy = + rz * np.cos(psi) - rz * np.cos(psi + dpsi)
        dpz = + rx * np.cos(phi) - rx * np.cos(phi + dphi)
        
        dp = np.vstack([dpx, dpy, dpz]).T
        
        p_k = p + dp
        v_k = v + acc_val_reshaped * dt
        q_k = (np.array(A + B) @ q.T).T # Nx4
        q_k = np.array([q_ / np.linalg.norm(q_) if np.linalg.norm(q_) > 0 else q_  for q_ in q_k])
        
        b_w_k = b_w + imu_sensor_error.gyro_bias.flatten()
        b_a_k = b_a + imu_sensor_error.acc_bias.flatten()

        process_noise = np.random.multivariate_normal(
            mean=np.zeros(self.x_dim), 
            cov=Q, 
            size=self.ensemble_size
        )
        
        self.samples = np.concatenate([
            p_k,
            v_k,
            q_k,
            b_w_k,
            b_a_k
        ], axis=1) + process_noise

        x = np.mean(self.samples, axis=0)
        self.x = State.get_new_state_from_array(x)
    
    
    def measurement_update(self, data: MeasurementUpdateField):
        """Measurement update step of Kalman filter
        Args:
            - MeasurementUpdateField containing:
                z: np.ndarray           -> measurement input
                R: np.ndarray           -> measurement noise covariance matrix
                sensor_type: SensorType -> state transformation matrix, which transform state vector x to measurement space
        """
        z = data.z
        R = data.R
        sensor_type = data.sensor_type
        
        z_dim = z.shape[0]
        x = self.x.get_state_vector()
        H = self.get_transition_matrix(sensor_type, z_dim=z_dim)
        mask = self.get_innovation_mask(sensor_type=sensor_type, z_dim=z_dim)
        
        mean = np.mean(self.samples, axis=0)
        P = np.zeros((self.x_dim, self.x_dim))
        for sample in self.samples:
            x_var = (sample - mean).reshape(-1, 1)
            P += x_var @ x_var.T
        
        P /= (self.ensemble_size - 1)
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        
        samples = (H @ self.samples.T).T  # NxM
        measurement_noise = np.random.multivariate_normal(
                                mean=np.zeros(z_dim), 
                                cov=R, size=self.ensemble_size)
        z = z.reshape(1, -1) + measurement_noise # N x m
        residuals = (z - samples) 
        innovation = residuals @ K.T
        innovation *= mask
        self.samples += innovation
        
        x = np.mean(self.samples, axis=0)
        self.x = State.get_new_state_from_array(x)
        
        self.innovations.append(np.sum(np.average(residuals, axis=1)))
    
        
    def get_current_estimate(self) -> Pose:
        # NOTE: Overwrite parent's method 
        x = np.mean(self.samples, axis=0)
        state = State.get_new_state_from_array(x)
        return Pose.from_state(state=state)
    
    def set_ensembles(self):
        sample_size = min(self.particle_size, MAX_NUM_PARTICLES_TO_VISUALIZE)
        random_indices = np.random.choice(self.ensemble_size, size=sample_size, replace=False)
        return self.samples[random_indices, :3] # return particles' position for visualization