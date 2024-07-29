import sys
if __name__ == "__main__":
    sys.path.append('../../src')

import numpy as np
from filterpy.monte_carlo import (
    multinomial_resample, residual_resample, systematic_resample, stratified_resample
)
from data_loader import DataLoader
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import multivariate_normal
from configs import MeasurementDataEnum, SetupEnum, FilterEnum
from utils.error_report import get_error_report
from kalman_filters.particle_filter import ResamplingAlgorithms
from base_filter import BaseFilter



class KLD_ParticleFilter:

    weights = None
    def __init__(
        self, 
        N, 
        x_dim, 
        H,
        delta,
        epsilon,
        resolutions,
        min_n_particles=32,
        max_n_particles=2**14,
        resampling_algorithm=ResamplingAlgorithms.MULTINOMIAL,
        debug_mode=False):
        
        self.particles = np.empty((N, x_dim))
        self.N_original = N
        self.N = N
        self.H = H

        self.epsilon = epsilon
        self.delta = delta
        self.min_n_particles = min_n_particles
        self.max_n_particles = max_n_particles
        self.resolutions = resolutions

        self.weights = np.ones(self.N) / self.N
        self.resampling_algorithm = resampling_algorithm

        self.debug_mode = debug_mode
    
    def create_uniform_particles(self, x_range):
        for i, x in enumerate(x_range):
            self.particles[:, i] = np.random.uniform(x[0], x[1], size=self.N)

    def create_gaussian_particles(self, mean, var):
        self.particles = mean.reshape(-1) + np.array([np.random.randn(self.N) for _ in range(var.shape[0])]).T @ var

    def compute_norm_w(self, w):
        return np.sqrt(np.sum(w**2))

    def get_rotation_matrix(self, q):
        q1, q2, q3, q4 = q[:, 0]
        return np.array([
            [q1**2 + q2**2 - q3**2 - q4**2, 2*(q2*q3 - q1*q4), 2*(q1*q3 + q2*q4)],
            [2*(q2*q3 + q1*q4), q1**2 - q2**2 + q3**2 - q4**2, 2*(q3*q4 - q1*q2)],
            [2*(q2*q4 - q1*q3), 2*(q1*q2 + q3*q4), q1**2 - q2**2 - q3**2 + q4**2]
           ])
    def get_quaternion_update_matrix(self, w):
        wx, wy, wz = w[:, 0]
        return np.array([
            [0, wz, -wy, wx],
            [-wz, 0, wx, wy],
            [wy, -wx, 0, wz],
            [-wx, -wy, -wz, 0]
        ])

    def draw_random_particle(self):
        Q = np.cumsum(self.weights).tolist()
        u = np.random.uniform(1e-6, Q[-1], 1)[0]
        index = 0
        while Q[index] < u:
            index += 1
        return self.particles[index]

    def propagate_particle(self, particle, u, dt, Q):
        """ 
            move according to control input u (heading change, velocity) with noise std
            u: control input vector
            process_noise_vector: process noise vector
            dt: delta time
        """
        p = particle[:3]
        v = particle[3:6]
        q = particle[6:]
        
        a = u[:3]
        w = u[3:]
        a = a.reshape(-1, 1)
        w = w.reshape(-1, 1)
        g = np.array([[0],[0],[9.81]])
        R = self.get_rotation_matrix(q) #3x3
        omega = self.get_quaternion_update_matrix(w)
        norm_w = self.compute_norm_w(w)

        A = np.cos(norm_w*dt/2) * np.eye(4)
        B = (1/norm_w)*np.sin(norm_w*dt/2) * omega
        
        # predict state vector x
        p_k = p + v * dt + (R @ a - g)*dt**2 / 2
        v_k = v + (R @ a - g) * dt
        q_k = np.array(A + B) @ q
        propagated_particle = np.concatenate([
            p_k,
            v_k,
            q_k,
        ]).reshape(-1)
        return np.random.multivariate_normal(propagated_particle, Q, 1)

    def update_particle(self, particle, z, R):
        """ 
            calculate the likelihood p(zk|xk)
            z: measurement
            R: measurement noise covariance
        """
        target_distribution = multivariate_normal(mean=z, cov=R)
        y_hat = self.H @ particle + np.random.multivariate_normal(np.zeros(R.shape[0]), R, 1).reshape(-1) 
        return target_distribution.pdf(y_hat)
        
    def run(self, u, dt, Q, z, R):
        new_particles = []
        new_weights = []

        k = 0
        bins_with_support = []
        number_of_new_particles = 0
        number_of_required_particles = self.min_n_particles
        while number_of_new_particles < number_of_required_particles:
            # Get sample from discrete distribution given by particle weights
            particle = self.draw_random_particle()

            # propagate a selected particle
            propagated_particle = self.propagate_particle(
                particle=particle.reshape(-1, 1),
                u=u,
                dt=dt,
                Q=Q)
            # compute likelihood (weight of the propagated particle in the measurement space)
            importance_weight = self.update_particle(particle=propagated_particle.reshape(-1), z=z, R=R)
            
            new_particles.append(propagated_particle.reshape(-1).tolist())
            new_weights.append(importance_weight)
            number_of_new_particles += 1
            
            indices = np.floor(propagated_particle.reshape(-1) / self.resolutions).tolist()[:2]
            if indices not in bins_with_support:
                bins_with_support.append(indices)
                k += 1
                
            if k > 1:
                # upper_quantile = np.sqrt(chi2.pdf(1-self.delta, df=k))
                upper_quantile = stats.norm.ppf(1-self.delta)
                number_of_required_particles = self.compute_required_number_of_particles_kld(
                                                k=k, 
                                                upper_quantile=upper_quantile)
                                                                                        
            # Make sure number of particles constraints are not violated
            number_of_required_particles = max(number_of_required_particles, self.min_n_particles)
            number_of_required_particles = min(number_of_required_particles, self.max_n_particles)


        self.particles = np.array(new_particles)
        self.weights = np.array(new_weights)
        self.weights += 1.e-300 # avoiding dividing by zero
        self.weights /= sum(self.weights) # normalize
        self.N = self.particles.shape[0]
        
    def run_filter(self, data, show_graph=False):
        
        if show_graph is True:
            fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
            xs, ys, _ = data.GPS_measurements_in_meter.T
            ax1.plot(xs, ys, lw=2, label='ground-truth trajectory', color='black')
            ax1.set_xlabel('X [m]')
            ax1.set_ylabel('Y [m]')
            ax1.legend()
            ax1.grid()

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

        px, py, _ = data.VO_measurements[0, :]
        mu_x = [px,]
        mu_y = [py,]
        t_last = 0.
        
        n = self.N
        for t_idx in tqdm(range(1, data.N), disable=not self.debug_mode):
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
            vo_z, vo_noise = data.get_vo_measurement_with_noise_cov(t_idx)
            gps_z, gps_noise = data.get_gps_measurement_with_noise_cov(t_idx)   
            R = np.array([
                [vo_noise ** 2., 0., 0., 0.],
                [0., vo_noise ** 2., 0., 0.],
                [0., 0., gps_noise ** 2., 0.],
                [0., 0., 0., gps_noise ** 2.]
            ])
            z = np.concatenate([vo_z, gps_z]).reshape(-1)
            self.run(u=u, dt=dt, Q=Q, z=z, R=R)
              
            if self.N > n and self.debug_mode:
                print(f"at index: {t_idx}, N: {self.N}")
                n = self.N

            if show_graph:
              if self.N > 100:
                  #show only 100 particles in the graph
                  particle_indices = np.linspace(0, self.N-1, 100, dtype=int).tolist()
                  ax1.scatter(self.particles[particle_indices, 0], 
                            self.particles[particle_indices, 1], 
                            alpha=.2, 
                            s=[10])
              else:
                  ax1.scatter(self.particles[:, 0], self.particles[:, 1], alpha=.2, s=[10])
            
            t_last = t
            mean, _ = self.estimate()
            mu_x.append(mean[0])
            mu_y.append(mean[1])

            if np.sum(np.sqrt((mean[:2] - gps_z.reshape(-1))**2)) > 300:
                if show_graph:
                  ax1.scatter(gps_z[0], gps_z[1], alpha=1, s=[100], color='black')
                break
          
        if len(mu_x) != data.N:
            print("The filter diverged.")
            return None
        
        error = get_error_report(data.GPS_measurements_in_meter.T[:2, :len(mu_x)], 
                                 np.array([mu_x, mu_y])) 
        return error

    def calculate_ess(self):
        return 1. / np.sum(np.square(self.weights))
        
    def estimate(self):
        """ returns mean and variance """
        pos = self.particles
        mu = np.average(pos, weights=self.weights, axis=0)
        var = np.average((pos - mu)**2, weights=self.weights, axis=0)

        return mu, var
        
    def compute_required_number_of_particles_kld(self, k, upper_quantile):
        """
        Compute the number of samples needed within a particle filter when k bins in the multidimensional histogram contain
        samples. Use Wilson-Hilferty transformation to approximate the quantiles of the chi-squared distribution as proposed
        by Fox (2003).
    
        :param epsilon: Maxmimum allowed distance (error) between true and estimated distribution.
        :param upper_quantile: Upper standard normal distribution quantile for (1-delta) where delta is the probability that
        the error on the estimated distribution will be less than epsilon.
        :param k: Number of bins containing samples.
        :return: Number of required particles.
        """
        # Helper variable (part between curly brackets in (14) in Fox paper
        x = 1.0 - 2.0 / (9.0*(k-1)) + np.sqrt(2.0 / (9.0*(k-1))) * upper_quantile
        return np.ceil((k-1) / (2.0*self.epsilon) * x * x * x)
        
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

    data = DataLoader(sequence_nr=kitti_drive, 
                    kitti_root_dir=kitti_root_dir, 
                    vo_root_dir=vo_root_dir,
                    noise_vector_dir=noise_vector_dir,
                    vo_dropout_ratio=0.0, 
                    gps_dropout_ratio=0.0)
    x_setup1, P_setup1, H_setup1, q1, r_vo1, r_gps1 = data.get_initial_data(setup=SetupEnum.SETUP_1, filter_type=FilterEnum.PF)
    x_setup2, P_setup2, H_setup2, q2, r_vo2, r_gps2 = data.get_initial_data(setup=SetupEnum.SETUP_2, filter_type=FilterEnum.PF)
    x_setup3, P_setup3, H_setup3, q3, r_vo3, r_gps3 = data.get_initial_data(setup=SetupEnum.SETUP_3, filter_type=FilterEnum.PF)
  
    H = np.array([
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]
    ])
    n_samples_setup1_0 = 256
    resampling_algorithm_setup1_0 = ResamplingAlgorithms.STRATIFIED
    n_samples_setup2_0 = 256
    resampling_algorithm_setup2_0 = ResamplingAlgorithms.STRATIFIED
    n_samples_setup3_0 = 256
    resampling_algorithm_setup3_0 = ResamplingAlgorithms.RESIDUAL

    epsilon = 0.15
    delta=0.1
    resolutions = np.array([
        5, 5, 5,
        10, 10, 10,
        0.1, 0.2, 0.2, 0.1
    ])
    # kld_pf = KLD_ParticleFilter(N=n_samples_setup2_0, 
    #                             x_dim=x_setup2.shape[0], 
    #                             H=H.copy(), 
    #                             debug_mode=True,
    #                             min_n_particles=512,
    #                             delta=delta,
    #                             epsilon=epsilon,
    #                             resolutions=resolutions,
    #                             resampling_algorithm=resampling_algorithm_setup2_0)
    # kld_pf.create_gaussian_particles(mean=x_setup1.copy(), var=P_setup1.copy())

    # error_pf1_0 = kld_pf.run_filter(data=data, show_graph=True)


