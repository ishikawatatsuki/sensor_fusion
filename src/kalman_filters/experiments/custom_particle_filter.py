import sys
if __name__ == "__main__":
    sys.path.append('../../src')

import random
import numpy as np
from enum import Enum
from tqdm import tqdm
from scipy.stats import multivariate_normal
from configs import MeasurementDataEnum, SetupEnum, FilterEnum, NoiseTypeEnum
import matplotlib.pyplot as plt
from filterpy.monte_carlo import (
    multinomial_resample, residual_resample, systematic_resample, stratified_resample
)
from utils.error_report import get_error_report

if __name__ == "__main__":
    from base_filter import BaseFilter
else:
    from ..base_filter import BaseFilter


class ResamplingAlgorithms(Enum):
    MULTINOMIAL = 1
    RESIDUAL = 2
    STRATIFIED = 3
    SYSTEMATIC = 4

"""
    This particle filter is designed based on the hypothesis that some amount of particles are propagated with, say dt/2 and (u2 - u1)/2, to have more divergence at measurement update step to mitigate weight degeneracy problem.
    The "some amount of particles" are chosen by ratio, which is set to be 0.2 by default, and the minority of the particles are propagated earlier than the rest of the particles.
"""
class CustomParticleFilter(BaseFilter):

    weights = None

    mu_x = None
    mu_y = None
    
    def __init__(
        self, 
        N, 
        x_dim, 
        H,
        q, 
        r_vo, 
        r_gps, 
        n_threshold=0.3,
        early_propagation_ratio=0.2,
        setup=SetupEnum.SETUP_1,
        resampling_algorithm=ResamplingAlgorithms.MULTINOMIAL):
        
        self.particles = np.empty((N, x_dim))
        self.N = N
        self.H = H

        self.setup = setup
        self.Q = self.get_diagonal_matrix(q)
        self.R_vo = self.get_diagonal_matrix(r_vo)
        self.R_gps = self.get_diagonal_matrix(r_gps)

        self.weights = np.ones(self.N) / self.N
        self.n_threshold = n_threshold
        self.resampling_algorithm = resampling_algorithm

        self.early_propagation_ratio = early_propagation_ratio
    
    def create_uniform_particles(self, x_range):
        for i, x in enumerate(x_range):
            self.particles[:, i] = np.random.uniform(x[0], x[1], size=self.N)

    def create_gaussian_particles(self, mean, var):
        self.particles = mean.reshape(-1) + np.array([np.random.randn(self.N) for _ in range(var.shape[0])]).T @ var
    
    def predict_setup1_2(self, particles, u, dt, Q):
        """ 
            move according to control input u (heading change, velocity) with noise std
            u: control input vector
            process_noise_vector: process noise vector
            dt: delta time
        """
        p = particles[:, :3]
        v = particles[:, 3:6]
        q = particles[:, 6:]
        
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
        process_noise = np.random.multivariate_normal(np.zeros(Q.shape[0]), Q, particles.shape[0])
        return np.concatenate([
            p_k,
            v_k,
            q_k,
        ], axis=1) + process_noise #Nx10
        
    def predict_setup3(self, particles, u, dt, Q):
        """ 
            move according to control input u (heading change, velocity) with noise std
            u: control input vector
            process_noise_vector: process noise vector
            dt: delta time
        """
        v, omega = u
        r = v / omega  # turning radius
        theta = particles[:, 2]
        dtheta = omega * dt
        dx = - r * np.sin(theta) + r * np.sin(theta + dtheta)
        dy = + r * np.cos(theta) - r * np.cos(theta + dtheta)
        delta_x = np.concatenate([
            dx.reshape(-1, 1), 
            dy.reshape(-1, 1), 
            (np.ones(particles.shape[0]) * dtheta).reshape(-1, 1)], axis=1)
        process_noise = np.random.multivariate_normal(np.zeros(Q.shape[0]), Q, particles.shape[0])
        particles += delta_x + process_noise

        return particles

    def update(self, z, R):
        """ 
            calculate the likelihood p(zk|xk)
            z: measurement
            R: measurement noise covariance
        """
        
        target_distribution = multivariate_normal(mean=z.reshape(-1), cov=R) 
        measurement_noise = np.random.multivariate_normal(np.zeros(R.shape[0]), R, self.N) 
        for i, particle in enumerate(self.particles):
            y_hat = self.H @ particle + measurement_noise[i]
            self.weights[i] = target_distribution.pdf(y_hat)

        self.weights += 1.e-300 # avoiding dividing by zero
        self.weights /= sum(self.weights) # normalize

    def calculate_ess(self):
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

    def resample_from_index_with_artificial_noise(self, indexes):
        assert len(indexes) == self.N

        self.particles = self.particles[indexes]
        self.particles += np.random.multivariate_normal(
            np.zeros(self.particles.shape[1]), 
            self.artificial_noise_cov, 
            self.N) * self.epsilon
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

    def allow_resampling(self, importance_resampling=True):
        return not importance_resampling or (importance_resampling and self.calculate_ess() < self.N * self.n_threshold)
    

    def split_particles(self):
        # N = int(np.ceil(self.N * self.early_propagation_ratio))
        indices = [i for i in range(0, self.N)]
        group1_indices = np.sort(random.sample(indices, int(len(indices)*self.early_propagation_ratio))).tolist()
        group2_indices = [i for i in indices if i not in group1_indices]

        # rearrange the weight order
        w1 = self.weights[group1_indices]
        w2 = self.weights[group2_indices]
        self.weights = np.concatenate([w1, w2], axis=0)

        return self.particles[group1_indices, :], self.particles[group2_indices, :]
        
    def get_early_propagation_data(self, data, t_idx):
        t1 = data.ts[t_idx - 1]
        t2 = data.ts[t_idx]
        dt = (t2 - t1) / 2

        if self.setup is SetupEnum.SETUP_1 or self.setup is SetupEnum.SETUP_2:
            a  = data.IMU_angular_velocity_with_noise[t_idx-1:t_idx+1]
            w  = data.IMU_angular_velocity_with_noise[t_idx-1:t_idx+1]
            
            ax, ay, az = a[:, 0], a[:, 1], a[:, 2]
            wx, wy, wz = w[:, 0], w[:, 1], w[:, 2]
            u = np.array([
                (ax[1] - ax[0]) / 2,
                (ay[1] - ay[0]) / 2,
                (az[1] - az[0]) / 2,
                (wx[1] - wx[0]) / 2,
                (wy[1] - wy[0]) / 2,
                (wz[1] - wz[0]) / 2,
            ])
            return u, dt

        v = data.INS_velocities_with_noise[t_idx-1:t_idx+1, 0]
        w = data.IMU_angular_velocity_with_noise[t_idx-1:t_idx+1, 2]
        
        u = np.array([
            (v[1] - v[0]) / 2,
            (w[1] - w[0]) / 2,
        ])
        return u, dt
    
    def _time_update_step(self, data, t_idx, dt, Q):
        particle_group1, particle_group2 = self.split_particles()
        u_half, dt_half = self.get_early_propagation_data(t_idx=t_idx, data=data)

        if self.setup is SetupEnum.SETUP_1 or self.setup is SetupEnum.SETUP_2:
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
            particle_group1 = self.predict_setup1_2(particles=particle_group1, u=u_half, dt=dt_half, Q=Q)
            particle_group2 = self.predict_setup1_2(particles=particle_group2, u=u, dt=dt, Q=Q)
            
        else: #SetupEnum.SETUP_3
            u = np.array([
                data.INS_velocities_with_noise[t_idx, 0],
                data.IMU_angular_velocity_with_noise[t_idx, 2]
            ])
            particle_group1 = self.predict_setup3(particles=particle_group1, u=u_half, dt=dt_half, Q=Q)
            particle_group2 = self.predict_setup3(particles=particle_group2, u=u, dt=dt, Q=Q)

        
        self.particles = np.concatenate([particle_group1, particle_group2], axis=0)
        
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
        
        if  z_vo is not None and z_gps is not None and \
            self.allow_resampling(importance_resampling=importance_resampling):
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
        R_vo = np.array([
            [data.VO_noise_std ** 2., 0.],
            [0., data.VO_noise_std ** 2.]
        ])
        R_gps = np.array([
            [data.GPS_measurement_noise_std ** 2., 0.],
            [0., data.GPS_measurement_noise_std ** 2.]
        ])
        # process noise
        if self.setup is SetupEnum.SETUP_1 or self.setup is SetupEnum.SETUP_2:
            q0_noise, q1_noise, q2_noise, q3_noise = data.quaternion_process_noise
            q_noise =  [
                1., 
                1., 
                1.,
                1., 
                1., 
                1.,
                q0_noise, 
                q1_noise, 
                q2_noise, 
                q3_noise]
            i = np.eye(len(q_noise))
            Q = np.array([[val * num for num in i[ind]] for ind, val in enumerate(q_noise)])
        else:
            Q = np.array([
                [data.velocity_noise_std ** 2., 0., 0.],
                [0., data.velocity_noise_std ** 2., 0.],
                [0., 0., data.IMU_angular_velocity_noise_std ** 2.],
            ])

        mu_x = [data.VO_measurements[0, 0],]
        mu_y = [data.VO_measurements[0, 1],]
        t_last = 0.

        if debug_mode is True:
            print("[PF] start.")

        for t_idx in tqdm(range(1, data.N), disable=not debug_mode):
            t = data.ts[t_idx]
            dt = t - t_last

            # prediction step(time update)
            self._time_update_step(data, t_idx, dt, Q)
            
            # correction step(measurement update)
            self._measurement_update_step(data, t_idx, R_vo, R_gps, measurement_type, importance_resampling)

            x_hat, _ = self.estimate()
            mu_x.append(x_hat[0])
            mu_y.append(x_hat[1])

            if show_graph is True:
                if self.N > 100:
                    particle_indices = np.linspace(0, self.N-1, 100, dtype=int).tolist()
                    ax1.scatter(self.particles[particle_indices, 0], self.particles[particle_indices, 1], alpha=.2, s=[10])
                else:
                    ax1.scatter(self.particles[:, 0], self.particles[:, 1], alpha=.2, s=[10])
            
            t_last = t

        error = get_error_report(
            data.GPS_measurements_in_meter.T[:2, :len(mu_x)], 
            np.array([mu_x, mu_y])) 
        
        if debug_mode is True:
            print(f"[PF] errors: {error}")

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
        
        return error

if __name__ == "__main__":
    import os
    from data_loader import DataLoader

    root_path = "../../"
    file_export_path = os.path.join(root_path, "exports/_sequences/04")
    kitti_root_dir = os.path.join(root_path, "data")
    vo_root_dir = os.path.join(root_path, "vo_estimates")
    noise_vector_dir = os.path.join(root_path, "exports/_noise_optimizations/noise_vectors")
    kitti_date = '2011_09_30'
    kitti_drive = '0033'

    data = DataLoader(
        sequence_nr=kitti_drive, 
        kitti_root_dir=kitti_root_dir, 
        vo_root_dir=vo_root_dir,
        noise_vector_dir=noise_vector_dir,
        vo_dropout_ratio=0.0, 
        gps_dropout_ratio=0.0,
        visualize_data=False)
    x_setup1, P_setup1, H_setup1, q1, r_vo1, r_gps1 = data.get_initial_data(setup=SetupEnum.SETUP_1, filter_type=FilterEnum.PF, noise_type=NoiseTypeEnum.CURRENT)
    x_setup2, P_setup2, H_setup2, q2, r_vo2, r_gps2 = data.get_initial_data(setup=SetupEnum.SETUP_2, filter_type=FilterEnum.PF,  noise_type=NoiseTypeEnum.CURRENT)
    x_setup3, P_setup3, H_setup3, q3, r_vo3, r_gps3 = data.get_initial_data(setup=SetupEnum.SETUP_3, filter_type=FilterEnum.PF,  noise_type=NoiseTypeEnum.CURRENT)

    n_samples_setup1_0 = 256
    resampling_algorithm_setup1_0 = ResamplingAlgorithms.STRATIFIED
    n_samples_setup2_0 = 256
    resampling_algorithm_setup2_0 = ResamplingAlgorithms.STRATIFIED
    n_samples_setup3_0 = 256
    resampling_algorithm_setup3_0 = ResamplingAlgorithms.RESIDUAL

    EARLY_PROPAGATION_RATIO = 0.2 # N * ratio amount of samples are propagated earlier than the majority of the particles to get more divergence of the predicted state.

    # pf1_0 = CustomParticleFilter(N=n_samples_setup1_0, 
    #                              x_dim=x_setup1.shape[0], 
    #                              H=H_setup1.copy(), 
    #                              q=q1,
    #                              r_vo=r_vo1,
    #                              r_gps=r_gps1,
    #                              setup=SetupEnum.SETUP_1,
    #                              early_propagation_ratio=EARLY_PROPAGATION_RATIO,
    #                              resampling_algorithm=resampling_algorithm_setup1_0)
    # pf1_0.create_gaussian_particles(mean=x_setup1.copy(), var=P_setup1.copy())
    # error_pf1_0 = pf1_0.run(data=data, 
    #                         debug_mode=True,
    #                         measurement_type=MeasurementDataEnum.ALL_DATA)

    # pf2_0 = CustomParticleFilter(N=n_samples_setup2_0, 
    #                             x_dim=x_setup2.shape[0], 
    #                             H=H_setup2.copy(), 
    #                             q=q2,
    #                             r_vo=r_vo2,
    #                             r_gps=r_gps2,
    #                             setup=SetupEnum.SETUP_2,
    #                              early_propagation_ratio=EARLY_PROPAGATION_RATIO,
    #                             resampling_algorithm=resampling_algorithm_setup2_0)
    # pf2_0.create_gaussian_particles(mean=x_setup2.copy(), var=P_setup2.copy())
    # error_pf2_0 = pf2_0.run(data=data, 
    #                         debug_mode=True,
    #                         measurement_type=MeasurementDataEnum.ALL_DATA)
    
    # pf3_0 = CustomParticleFilter(N=n_samples_setup3_0, 
    #                              x_dim=x_setup3.shape[0], 
    #                              H=H_setup3.copy(), 
    #                              q=q3,
    #                              r_vo=r_vo3,
    #                              r_gps=r_gps3,
    #                              setup=SetupEnum.SETUP_3,
    #                              early_propagation_ratio=EARLY_PROPAGATION_RATIO,
    #                              resampling_algorithm=resampling_algorithm_setup3_0)
    # pf3_0.create_gaussian_particles(mean=x_setup3.copy(), var=P_setup3.copy())
    # error_pf3_0 = pf3_0.run(data=data, 
    #                         measurement_type=MeasurementDataEnum.ALL_DATA,
    #                         show_graph=True,
    #                         debug_mode=True)
