import os
import sys
import logging
import numpy as np
from enum import Enum, auto
from scipy.stats import multivariate_normal

from .base_filter import BaseFilter
from ..common import (
    State, MotionModel, Pose,
    MeasurementUpdateField,
)

from filterpy.monte_carlo import (
    multinomial_resample, residual_resample, systematic_resample, stratified_resample
)

class ResamplingAlgorithms(Enum):
    MULTINOMIAL = auto()
    RESIDUAL = auto()
    STRATIFIED = auto()
    SYSTEMATIC = auto()
    
    @staticmethod
    def get_enum_name_list():
        return [s.lower() for s in list(ResamplingAlgorithms.__members__.keys())]
    
    @classmethod
    def get_resampling_algorithm_from_str(cls, sensor_str: str):
        s = sensor_str.lower()
        try: 
            index = ResamplingAlgorithms.get_enum_name_list().index(s)
            return cls(index + 1)
        except:
            return None

class ParticleFilter(BaseFilter):
    def __init__(
            self,
            *args,
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        
        self.particle_size = self._get_params(params=self.config.params, key="particle_size", default_value=1024)
        resampling_algorithm = self._get_params(params=self.config.params, key="resampling_algorithm", default_value="multinomial")
        self.scale_for_ess_threshold = self._get_params(params=self.config.params, key="scale_for_ess_threshold", default_value=1.)

        x = self.x.get_state_vector()
        
        self.particles = self._create_gaussian_particles(mean=x, var=self.P)

        self.weights = np.ones(self.particle_size) / self.particle_size
        self.resampling_algorithm = ResamplingAlgorithms.get_resampling_algorithm_from_str(resampling_algorithm)
    
    def _create_gaussian_particles(self, mean, var):
        return mean.reshape(-1) + np.array([np.random.randn(self.particle_size) for _ in range(var.shape[0])]).T @ var

    def kinematics_motion_model(self, u: np.ndarray, dt: float, Q: np.ndarray):
        """ 
            move according to control input u (heading change, velocity) with noise std
            u: control input vector
            dt: delta time
            Q: process noise matrix
        """
        p = self.particles[:, :3]
        v = self.particles[:, 3:6]
        q = self.particles[:, 6:10]
        b_w = self.particles[:, 10:13]
        b_a = self.particles[:, 13:16]
        
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
        
        acc_val = (R @ a - self.g)
        acc_val = self.correct_acceleration(acc_val=acc_val, q=q)
        acc_val_reshaped = acc_val.reshape(acc_val.shape[0], acc_val.shape[1])
        p_k = p + v * dt + acc_val_reshaped*dt**2 / 2 # Nx3
        v_k = v + acc_val_reshaped * dt # Nx3
        q_k = (np.array(A + B) @ q.T).T # Nx4
        q_k = np.array([q_ / np.linalg.norm(q_) if np.linalg.norm(q_) > 0 else q_  for q_ in q_k])

        b_w_k = np.array([ bw + imu_sensor_error.gyro_bias for bw in b_w])
        b_a_k = np.array([ ba + imu_sensor_error.acc_bias for ba in b_a])
        
        process_noise = np.random.multivariate_normal(np.zeros(Q.shape[0]), Q, self.particle_size)
        self.particles = np.concatenate([
            p_k,
            v_k,
            q_k,
            b_w_k,
            b_a_k
        ], axis=1) + process_noise #Nx10
        
        x, _ = self.estimate()
        self.x = State.get_new_state_from_array(x)
        
    def velocity_motion_model(self, u: np.ndarray, dt: float, Q: np.ndarray):
        """ 
            move according to control input u (heading change, velocity) with noise std
            u: control input vector
            dt: delta time
            Q: process noise matrix
        """
        p = self.particles[:, :3]
        v = self.particles[:, 3:6]
        q = self.particles[:, 6:10]
        b_w = self.particles[:, 10:13]
        b_a = self.particles[:, 13:16]
        
        a = u[:3]
        w = u[3:]
        wx, _, wz = w
        a = a.reshape(-1, 1)
        w = w.reshape(-1, 1)
        
        # Take into account the IMU sensor error
        imu_sensor_error = self.get_imu_sensor_error()

        a -=  imu_sensor_error.acc_bias + self.x.b_a + imu_sensor_error.acc_noise
        w -= imu_sensor_error.gyro_bias + self.x.b_w + imu_sensor_error.gyro_noise

        omega = self.get_quaternion_update_matrix(w)
        norm_w = self.compute_norm_w(w)
        phi, _, psi = np.array([self.get_euler_angle_from_quaternion(q_row.reshape(-1, 1)) for q_row in q]).T
        R = np.array([self.x.get_rotation_matrix(q_) for q_ in q])
        
        A = np.cos(norm_w*dt/2) * np.eye(4)
        B = (1/norm_w)*np.sin(norm_w*dt/2) * omega
        
        vf = self.get_forward_velocity(v)
        
        acc_val = (R @ a - self.g)
        # acc_val = self.correct_acceleration(acc_val=acc_val, q=q)
        acc_val_reshaped = acc_val.reshape(acc_val.shape[0], acc_val.shape[1])
        
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
        
        b_w_k = np.array([ bw + imu_sensor_error.gyro_bias for bw in b_w])
        b_a_k = np.array([ ba + imu_sensor_error.acc_bias for ba in b_a])
        
        process_noise = np.random.multivariate_normal(
            np.zeros(Q.shape[0]), 
            Q, 
            self.particle_size
        )
        
        self.particles = np.concatenate([
            p_k,
            v_k,
            q_k,
            b_w_k,
            b_a_k
        ], axis=1) + process_noise
        
        x, _ = self.estimate()
        self.x = State.get_new_state_from_array(x)
    
    def measurement_update(self, data: MeasurementUpdateField):
        """ 
            calculate the likelihood p(zk|xk)
            z: measurement
            R: measurement noise covariance
        """
        z = data.z
        R = data.R
        sensor_type = data.sensor_type
        H = self.get_transition_matrix(sensor_type, z_dim=z.shape[0])
        
        target_distribution = multivariate_normal(mean=z.flatten(), cov=R) 
        measurement_noise = np.random.multivariate_normal(
            np.zeros(R.shape[0]), 
            R, 
            self.particle_size
        )
        
        # residual calculation
        x_, _ = self.estimate()
        z_ = H @ x_
        residual = z - z_
        self.innovations.append(np.sum(residual))
        
        for i, particle in enumerate(self.particles):
            y_hat = H @ particle + measurement_noise[i]
            self.weights[i] = target_distribution.pdf(y_hat)

        self.weights += 1.e-300 # avoiding dividing by zero
        self.weights /= sum(self.weights) # normalize
        
        # Resample when a sensor data is given and is allowed by importance resampling
        if self._allow_resampling():
            self._resample()
            
        x, _ = self.estimate()
        self.x = State.get_new_state_from_array(x)
        
    def _allow_resampling(self):
        '''
            Allow resampling when ESS < particle size:
                Effective sample size (ESS)
                When the ESS gets close to zero resulted from many particles having small weight, it indicates particle degeneracy meaning that many particles with small weight estimate the measurement poorly.
                To prevent particle degeneracy, resampling comes into play.
        '''
        def _calculate_ess():
            return 1. / np.sum(np.square(self.weights))
        
        N_eff = _calculate_ess()
        return N_eff < self.particle_size * self.scale_for_ess_threshold

    def estimate(self):
        """ 
            computer posterior of the system by calcurating the weighted average
        """
        pos = self.particles
        mu = np.average(pos, weights=self.weights, axis=0)
        var = np.average((pos - mu)**2, weights=self.weights, axis=0)

        return mu, var

    def _resample_from_index(self, indexes):
        assert len(indexes) == self.particle_size
        
        self.particles = self.particles[indexes]
        self.weights = self.weights[indexes]
        self.weights /= np.sum(self.weights)

    def _resample(self):
        if self.resampling_algorithm is ResamplingAlgorithms.RESIDUAL:
            indexes = residual_resample(self.weights)
        elif self.resampling_algorithm is ResamplingAlgorithms.STRATIFIED:
            indexes = stratified_resample(self.weights)
        elif self.resampling_algorithm is ResamplingAlgorithms.SYSTEMATIC:
            indexes = systematic_resample(self.weights)
        else:
            # ResamplingAlgorithms.MULTINOMIAL
            indexes = multinomial_resample(self.weights)
        
        self._resample_from_index(indexes)

    def get_current_estimate(self) -> Pose:
        # NOTE: Overwrite parent's method 
        x, _ = self.estimate()
        state = State.get_new_state_from_array(x)
        return Pose.from_state(state=state)
