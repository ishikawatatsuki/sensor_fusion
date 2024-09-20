import sys
if __name__ == "__main__":
    sys.path.append('../../src')

import numpy as np
from enum import Enum
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from configs import SetupEnum, MeasurementDataEnum, FilterEnum, NoiseTypeEnum
from utils.time_reporter import time_measurer
from utils.error_report import get_error_report, print_error_report
from filterpy.monte_carlo import (
    multinomial_resample, residual_resample, systematic_resample, stratified_resample
)

if __name__ == "__main__":
    from base_filter import BaseFilter
else:
    from .base_filter import BaseFilter


class ResamplingAlgorithms(Enum):
    MULTINOMIAL = 1
    RESIDUAL = 2
    STRATIFIED = 3
    SYSTEMATIC = 4

class ParticleFilter(BaseFilter):

    x = None
    def __init__(
        self, 
        N, 
        x_dim, 
        H,
        q, 
        r_vo, 
        r_gps, 
        n_threshold=0.3,
        setup=SetupEnum.SETUP_1,
        resampling_algorithm=ResamplingAlgorithms.MULTINOMIAL):
        """ 
        Args:
            N (int): number of particles
            x_dim (int): length of the state vector
            H (numpy.array): transition matrix from predicted state vector to measurement space
            q (numpy.array): process noise vector
            r_vo (numpy.array): measurement noise vector for VO
            r_gps (numpy.array): measurement noise vector for GPS
            setup (SetupEnum): filter setup
        """
        self.particles = np.empty((N, x_dim))
        self.N_original = N
        self.N = N
        self.H = H

        self.setup = setup
        self.dimension=self.H.shape[0]
        self.Q = self.get_diagonal_matrix(q)
        self.R_vo = self.get_diagonal_matrix(r_vo)
        self.R_gps = self.get_diagonal_matrix(r_gps)

        self.weights = np.ones(self.N) / self.N
        self.n_threshold = n_threshold
        self.resampling_algorithm = resampling_algorithm
        

    def create_uniform_particles(self, x_range):
        for i, x in enumerate(x_range):
            self.particles[:, i] = np.random.uniform(x[0], x[1], size=self.N)

    def create_gaussian_particles(self, mean, var):
        self.x = mean
        self.particles = mean.reshape(-1) + np.array([np.random.randn(self.N) for _ in range(var.shape[0])]).T @ var

    def predict_setup1_2(self, u, dt, Q):
        """ 
            move according to control input u (heading change, velocity) with noise std
            u: control input vector
            dt: delta time
            Q: process noise matrix
        """
        p = self.particles[:, :3]
        v = self.particles[:, 3:6]
        q = self.particles[:, 6:]
        
        a = u[:3]
        w = u[3:]
        a = a.reshape(-1, 1)
        w = w.reshape(-1, 1)
        g = np.array([[0],[0],[9.81]])
        R = np.array([self.get_rotation_matrix(q_.reshape(-1, 1)) for q_ in q]) #Nx3x3
        omega = self.get_quaternion_update_matrix(w)
        norm_w = self.compute_norm_w(w)

        A = np.cos(norm_w*dt/2) * np.eye(4)
        B = (1/norm_w)*np.sin(norm_w*dt/2) * omega
        
        acc_val = (R @ a - g)
        acc_val_reshaped = acc_val.reshape(acc_val.shape[0], acc_val.shape[1])
        p_k = p + v * dt + acc_val_reshaped*dt**2 / 2 # Nx3
        v_k = v + acc_val_reshaped * dt # Nx3
        q_k = q @ np.array(A + B) # Nx4
        q_k = np.array([q_ / np.linalg.norm(q_) if np.linalg.norm(q_) > 0 else q_  for q_ in q_k])

        process_noise = np.random.multivariate_normal(np.zeros(Q.shape[0]), Q, self.N)
        self.particles = np.concatenate([
            p_k,
            v_k,
            q_k,
        ], axis=1) + process_noise #Nx10
        
    def predict_setup3(self, u, dt, Q):
        """ 
            move according to control input u (heading change, velocity) with noise std
            u: control input vector
            dt: delta time
            Q: process noise matrix
        """
        if self.dimension == 2:
            v, omega = u
            r = v / omega  # turning radius
            theta = self.particles[:, 2]
            dtheta = omega * dt
            dx = - r * np.sin(theta) + r * np.sin(theta + dtheta)
            dy = + r * np.cos(theta) - r * np.cos(theta + dtheta)
            
            delta_x = np.concatenate([
                dx.reshape(-1, 1),
                dy.reshape(-1, 1), 
                (np.ones(self.N) * dtheta).reshape(-1, 1)
            ], axis=1)
            process_noise = np.random.multivariate_normal(np.zeros(Q.shape[0]), Q, self.N)
            self.particles += delta_x + process_noise
        else:
            v, wx, wz = u
            rx = v / wx  # turning radius for x axis
            rz = v / wz  # turning radius for z axis
            phi, psi = self.particles[:, 3:].T
            
            dphi = wx * dt
            dpsi = wz * dt
            dx = - rz * np.sin(psi) + rz * np.sin(psi + dpsi)
            dy = + rz * np.cos(psi) - rz * np.cos(psi + dpsi)
            dz = + rx * np.cos(phi) - rx * np.cos(phi + dphi)
            
            delta_x = np.concatenate([
                dx.reshape(-1, 1),
                dy.reshape(-1, 1), 
                dz.reshape(-1, 1), 
                (np.ones(self.N) * dphi).reshape(-1, 1),
                (np.ones(self.N) * dpsi).reshape(-1, 1),
            ], axis=1)
            process_noise = np.random.multivariate_normal(np.zeros(Q.shape[0]), Q, self.N)
            self.particles += delta_x + process_noise
            
    def update(self, z, R):
        """ 
            calculate the likelihood p(zk|xk)
            z: measurement
            R: measurement noise covariance
        """
        # measurement_noise = np.random.normal(0, 
        #                  measurement_noise_vector.reshape(1, -1), 
        #                  (self.N, len(measurement_noise_vector)))
        
        target_distribution = multivariate_normal(mean=z.reshape(-1), cov=R) 
        measurement_noise = np.random.multivariate_normal(np.zeros(R.shape[0]), R, self.N) 
        for i, particle in enumerate(self.particles):
            y_hat = self.H @ particle + measurement_noise[i]
            self.weights[i] = target_distribution.pdf(y_hat)

        self.weights += 1.e-300 # avoiding dividing by zero
        self.weights /= sum(self.weights) # normalize

    def calculate_ess(self):
        '''
            Effective sample size (ESS)
            When the ESS gets close to zero resulted from many particles having small weight, it indicates particle degeneracy meaning that many particles with small weight estimate the measurement poorly.
            To prevent particle degeneracy, resampling comes into play.
        '''
        return 1. / np.sum(np.square(self.weights))
        
    def estimate(self):
        """ 
            computer posterior of the system by calcurating the weighted average
        """
        pos = self.particles
        mu = np.average(pos, weights=self.weights, axis=0)
        var = np.average((pos - mu)**2, weights=self.weights, axis=0)

        return mu, var

    def resample_from_index(self, indexes):
        assert len(indexes) == self.N
        
        self.particles = self.particles[indexes]
        self.weights = self.weights[indexes]
        self.weights /= np.sum(self.weights)

    def resample(self):
        if self.resampling_algorithm is ResamplingAlgorithms.RESIDUAL:
            indexes = residual_resample(self.weights)
        elif self.resampling_algorithm is ResamplingAlgorithms.STRATIFIED:
            indexes = stratified_resample(self.weights)
        elif self.resampling_algorithm is ResamplingAlgorithms.SYSTEMATIC:
            indexes = systematic_resample(self.weights)
        else:
            # ResamplingAlgorithms.MULTINOMIAL
            indexes = multinomial_resample(self.weights)
        
        self.resample_from_index(indexes)

    def _allow_resampling(self, importance_resampling=True):
        '''
            Allow resampling either:
                - when importance resampling is False -> Always resample after measurement update step
                - when importance resampling is True and the effective sample size is less than 
        '''
        return not importance_resampling or (importance_resampling and self.calculate_ess() < self.N * self.n_threshold)
    
    def _time_update_step(self, data, t_idx, dt, Q):
        u = data.get_control_input_by_index(index=t_idx, setup=self.setup)
        predict = self.predict_setup1_2 if self.setup is SetupEnum.SETUP_1 or\
                                                self.setup is SetupEnum.SETUP_2 else self.predict_setup3
        
        predict(u=u, dt=dt, Q=Q)
            
    def _measurement_update_step(self, data, t_idx, R_vo, R_gps, measurement_type, importance_resampling):
        z_vo, _R_vo = data.get_vo_measurement_by_index(
            index=t_idx, 
            measurement_type=measurement_type)
        z_gps, _R_gps = data.get_gps_measurement_by_index(
            index=t_idx, 
            setup=self.setup, 
            measurement_type=measurement_type)
        if measurement_type is MeasurementDataEnum.COVARIANCE:
            R_vo = _R_vo
            R_gps = _R_gps
            
        if z_vo is not None:
            self.update(z=z_vo, R=R_vo)
        
        if z_gps is not None:
            self.update(z=z_gps, R=R_gps)
        
        # check if all sensor data is available
        sensor_data_available = z_vo is not None if self.setup is SetupEnum.SETUP_1 else z_vo is not None and z_gps is not None
        
        # Resample when all sensor data is available and allowed by importance resampling
        if sensor_data_available and self._allow_resampling(importance_resampling=importance_resampling):
            self.resample()

    def run(self, 
            data, 
            measurement_type=MeasurementDataEnum.ALL_DATA, 
            importance_resampling=False,
            show_graph=False, 
            debug_mode=False):
        
        if show_graph is True:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            xs, ys, _ = data.GPS_measurements_in_meter.T
            ax1.plot(xs, ys, lw=2, label='ground-truth trajectory', color='black')
            ax2.plot(xs, ys, lw=2, label='ground-truth trajectory', color='black')
            ax1.set_xlabel('X [m]')
            ax1.set_ylabel('Y [m]')
            ax1.legend()
            ax1.grid()

        # measurement noise
        R_vo = self.R_vo
        R_gps = self.R_gps
        # process noise
        Q = self.Q

        gt = data.get_trajectory_to_compare()
        mu_x = [gt[0, 0],]
        mu_y = [gt[1, 0],]
        mu_z = [gt[2, 0],]
        
        t_last = 0.

        if debug_mode is True:
            print("[PF] start.")
        for t_idx in tqdm(range(1, data.N), disable=not debug_mode):
            t = data.ts[t_idx]
            dt = t - t_last

            # prediction step(time update)
            self._time_update_step(data, t_idx, dt, Q)
            
            x_hat, _ = self.estimate()
            mu_x.append(x_hat[0])
            mu_y.append(x_hat[1])
            mu_z.append(x_hat[2])
            
            # correction step(measurement update)
            self._measurement_update_step(data, t_idx, R_vo, R_gps, measurement_type, importance_resampling)
            

            if show_graph is True:
                if self.N > 100:
                    particle_indices = np.linspace(0, self.N-1, 100, dtype=int).tolist()
                    ax1.scatter(
                        self.particles[particle_indices, 0], 
                        self.particles[particle_indices, 1], 
                        alpha=.2, s=[10])
                else:
                    ax1.scatter(
                        self.particles[:, 0], 
                        self.particles[:, 1], 
                        alpha=.2, s=[10])
            
            t_last = t

        error = \
            get_error_report(
                    gt[:2, :len(mu_x)], 
                    np.array([mu_x, mu_y]))\
            if self.H.shape[0] == 2 else\
            get_error_report(
                gt[:3, :len(mu_x)], 
                np.array([mu_x, mu_y, mu_z])) 
        
        if debug_mode is True:
            print_error_report(error, f"[PF] Error report for {SetupEnum.get_name(self.setup)}")

        if show_graph is True:
            xs, ys, _ = data.VO_measurements.T
            ax2.plot(xs, ys, lw=2, label='VO trajectory', color='b')
            ax2.plot(
                mu_x, mu_y, lw=2, 
                label='estimated trajectory', color='r')
            ax2.set_xlabel('X [m]')
            ax2.set_ylabel('Y [m]')
            ax2.legend()
            ax2.grid()
            
        self.mu_x = mu_x
        self.mu_y = mu_y
        self.mu_z = mu_z
        
        return error
    
    @time_measurer
    def run_with_time(self, *args, **kwargs):
        return self.run(*args, **kwargs)
    
    
if __name__ == "__main__":
    import os
    from data_loader import DataLoader

    root_path = "../../"
    kitti_drive = 'example'
    kitti_data_root_dir = os.path.join(root_path, "example_data")
    noise_vector_dir = os.path.join(root_path, "exports/_noise_optimizations/noise_vectors")
    dimension=2

    # Undo comment out this to change example data to entire sequence data
    # root_path = "../../"
    # kitti_drive = '0033'
    # kitti_data_root_dir = os.path.join(root_path, "data")
    # noise_vector_dir = os.path.join(root_path, "exports/_noise_optimizations/noise_vectors")
    # dimension=2

    data = DataLoader(
        sequence_nr=kitti_drive, 
        kitti_root_dir=kitti_data_root_dir, 
        noise_vector_dir=noise_vector_dir,
        vo_dropout_ratio=0., 
        gps_dropout_ratio=0.,
        visualize_data=False,
        dimension=dimension
    )
    
    filter_type=FilterEnum.PF
    noise_type=NoiseTypeEnum.CURRENT
    
    x_setup1, P_setup1, H_setup1, q1, r_vo1, r_gps1 = data.get_initial_data(setup=SetupEnum.SETUP_1, filter_type=filter_type, noise_type=noise_type)
    x_setup2, P_setup2, H_setup2, q2, r_vo2, r_gps2 = data.get_initial_data(setup=SetupEnum.SETUP_2, filter_type=filter_type, noise_type=noise_type)
    x_setup3, P_setup3, H_setup3, q3, r_vo3, r_gps3 = data.get_initial_data(setup=SetupEnum.SETUP_3, filter_type=filter_type, noise_type=noise_type)

    n_samples_setup1_0 = 512
    resampling_algorithm_setup1_0 = ResamplingAlgorithms.STRATIFIED
    n_samples_setup2_0 = 512
    resampling_algorithm_setup2_0 = ResamplingAlgorithms.MULTINOMIAL
    n_samples_setup3_0 = 64
    resampling_algorithm_setup3_0 = ResamplingAlgorithms.RESIDUAL
    
    measurement_type = MeasurementDataEnum.ALL_DATA
    importance_resampling = True
    debug_mode=True
    interval=5

    pf1_0 = ParticleFilter(N=n_samples_setup1_0, 
                            x_dim=x_setup1.shape[0], 
                            H=H_setup1.copy(), 
                            q=q1,
                            r_vo=r_vo1,
                            r_gps=r_gps1,
                            setup=SetupEnum.SETUP_1,
                            resampling_algorithm=resampling_algorithm_setup1_0)
    pf1_0.create_gaussian_particles(mean=x_setup1.copy(), var=P_setup1.copy())
    error_pf1_0 = pf1_0.run(
        data=data, 
        debug_mode=debug_mode,
        measurement_type=measurement_type, 
        importance_resampling=importance_resampling)
    
    pf1_0.visualize_trajectory(
        data=data, 
        dimension=dimension, 
        interval=interval, 
        title="PF Setup1 trajectories")

    pf2_0 = ParticleFilter(N=n_samples_setup2_0, 
                            x_dim=x_setup2.shape[0], 
                            H=H_setup2.copy(), 
                            q=q2,
                            r_vo=r_vo2,
                            r_gps=r_gps2,
                            setup=SetupEnum.SETUP_2,
                            resampling_algorithm=resampling_algorithm_setup2_0)
    pf2_0.create_gaussian_particles(mean=x_setup2.copy(), var=P_setup2.copy())
    error_pf2_0 = pf2_0.run(
        data=data, 
        debug_mode=debug_mode, 
        measurement_type=measurement_type, 
        importance_resampling=importance_resampling)
    
    pf2_0.visualize_trajectory(
        data=data, 
        dimension=dimension, 
        interval=interval, 
        title="PF Setup2 trajectories")
    
    pf3_0 = ParticleFilter(N=n_samples_setup3_0, 
                            x_dim=x_setup3.shape[0], 
                            H=H_setup3.copy(), 
                            q=q3,
                            r_vo=r_vo3,
                            r_gps=r_gps3,
                            setup=SetupEnum.SETUP_3,
                            resampling_algorithm=resampling_algorithm_setup3_0)
    pf3_0.create_gaussian_particles(mean=x_setup3.copy(), var=P_setup3.copy())
    error_pf3_0 = pf3_0.run(
        data=data,
        debug_mode=debug_mode,
        measurement_type=measurement_type, 
        importance_resampling=importance_resampling)
    
    pf3_0.visualize_trajectory(
        data=data, 
        dimension=dimension, 
        interval=interval, 
        title="PF Setup3 trajectories")