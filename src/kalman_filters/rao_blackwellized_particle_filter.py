import sys
if __name__ == "__main__":
    sys.path.append('../../src')

import numpy as np
from enum import Enum
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from configs import SetupEnum, MeasurementDataEnum, FilterEnum, NoiseTypeEnum
from kalman_filters.particle_filter import ResamplingAlgorithms
from filterpy.monte_carlo import (
    multinomial_resample, residual_resample, systematic_resample, stratified_resample
)
from sklearn.metrics import mean_squared_error

# https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=583afd895f94249898716e92c3693dfe8cb8eaac -p 45
# https://www.diva-portal.org/smash/get/diva2:412640/fulltext01.pdf

class RaoBlackwellizedParticleFilter:

    weights = None
    likelihoods = None

    mu_x = None
    mu_y = None

    def __init__(
        self,
        N,
        x,
        P,
        H,
        resampling_algorithm=ResamplingAlgorithms.MULTINOMIAL):

        self.N = N
        self.H = H
        self.x_dim = x.shape[0]
        self.x_p = self.create_gaussian_particles(x, P)
        self.x_k = np.ones((N, self.x_dim)) * x.reshape(-1)
        self.weights = np.ones(N) / N
        self.likelihoods = np.zeros(N)
        self.P = np.array([P for _ in range(0, N)])

        self.resampling_algorithm = resampling_algorithm


    def create_gaussian_particles(self, mean, var):
        return mean.reshape(-1) + np.array([np.random.randn(self.N) for _ in range(var.shape[0])]).T @ var
    
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
    
    def get_jacobian_setup1_2(self, q, u, dt):
        a = u[:3]
        w = u[3:]
        a = a.reshape(-1, 1)
        w = w.reshape(-1, 1)
        norm_w = self.compute_norm_w(w)
        ax, ay, az, wx, wy, wz = u
        q1, q2, q3, q4 = q
        q1_2, q2_2, q3_2, q4_2 = q**2 
        dt2 = dt**2
        cos_w = np.cos(norm_w*dt/2)
        sin_w = np.sin(norm_w*dt/2)/norm_w
        wz_sin = wz*sin_w
        wy_sin = wy*sin_w
        wx_sin = wx*sin_w
        # Jacobian matrix of function f(x,u) with respect to the state variables.
        F = np.array([
            [1., 0., 0., dt, 0., 0., dt2*(2*ax*q1-2*ay*q4+2*az*q3)/2, dt2*(2*ax*q2+2*ay*q3+2*az*q4)/2, dt2*(-2*ax*q3+2*ay*q2+2*az*q1)/2, dt2*(-2*ax*q4-2*ay*q1+2*az*q2)/2],
            [0., 1., 0., 0., dt, 0., dt2*(2*ax*q4+2*ay*q1-2*az*q2)/2, dt2*(2*ax*q3-2*ay*q2-2*az*q1)/2, dt2*(2*ax*q2+2*ay*q3+2*az*q4)/2, dt2*(2*az*q1-2*ay*q4+2*az*q3)/2],
            [0., 0., 1., 0., 0., dt, dt2*(-2*ax*q3+2*ay*q2+2*az*q1)/2, dt2*(2*ax*q4+2*ay*q1-2*az*q2)/2, dt2*(-2*ax*q1+2*ay*q4-2*az*q3)/2, dt2*(2*ax*q2+2*ay*q3+2*az*q4)/2],

            [0., 0., 0., 1., 0., 0., dt*(2*ax*q1-2*ay*q4+2*az*q3), dt*(2*ax*q2+2*ay*q3+2*az*q4), dt*(-2*ax*q3+2*ay*q2+2*az*q1), dt*(-2*ax*q4-2*ay*q1+2*az*q2)],
            [0., 0., 0., 0., 1., 0., dt*(2*ax*q4+2*ay*q1-2*az*q2), dt*(2*ax*q3-2*ay*q2-2*az*q1), dt*(2*ax*q2+2*ay*q3+2*az*q4), dt*(2*ax*q1-2*ay*q4+2*az*q3)],
            [0., 0., 0., 0., 0., 1., dt*(-2*ax*q3+2*ay*q2+2*az*q1), dt*(2*ax*q4+2*ay*q1-2*ay*q2), dt*(-2*ax*q1+2*ay*q4-2*az*q3), dt*(2*ax*q2+2*ay*q3+2*az*q4)],

            [0., 0., 0., 0., 0., 0., cos_w, wz_sin, -wy_sin, wx_sin],
            [0., 0., 0., 0., 0., 0., -wz_sin, cos_w, wx_sin, wy_sin],
            [0., 0., 0., 0., 0., 0., wy_sin, -wx_sin, cos_w, wz_sin],
            [0., 0., 0., 0., 0., 0., -wx_sin, -wy_sin, -wz_sin, cos_w]
        ])
        
        # Jacobian matrix of function f(x,u) with respect to the control input variables.
        G = np.array([
            [dt2*(q1_2+q2_2-q3_2-q4_2)/2, dt2*(-2*q1*q4+2*q2*q3)/2, dt2*(2*q1*q3+2*q2*q4)/2, 0., 0., 0.],
            [dt2*(2*q1*q4+2*q2*q3)/2, dt2*(q1_2-q2_2+q3_2-q4_2)/2, dt2*(-2*q1*q2+2*q3*q4)/2, 0., 0., 0.],
            [dt2*(-2*q1*q3+2*q2*q4)/2, dt2*(2*q1*q2+2*q3*q4)/2, dt2*(q1_2-q2_2-q3_2+q4_2)/2, 0., 0., 0.],
            
            [dt*(q1_2+q2_2-q3_2-q4_2), dt*(-2*q1*q4+2*q2*q3), dt*(2*q1*q3+2*q2*q4), 0., 0., 0.],
            [dt*(2*q1*q4+2*q2*q3), dt*(q1_2-q2_2+q3_2-q4_2), dt*(-2*q1*q2+2*q3*q4), 0., 0., 0.],
            [dt*(-2*q1*q3+2*q2*q4), dt*(2*q1*q2+2*q3*q4), dt*(q1_2-q2_2-q3_2+q4_2), 0., 0., 0.],

            [0., 0., 0., q4*sin_w, -q3*sin_w, q2*sin_w],
            [0., 0., 0., q3*sin_w, q4*sin_w, -q1*sin_w],
            [0., 0., 0., -q2*sin_w, q1*sin_w, q4*sin_w],
            [0., 0., 0., -q1*sin_w, -q2*sin_w, -q3*sin_w]
        ])
            
        return F, G
        
    def motion_model_1_2(self, x, u, dt):
        # propagate state x
        p = x[:, :3]
        v = x[:, 3:6]
        q = x[:, 6:]
        a = u[:3]
        w = u[3:]
        a = a.reshape(-1, 1)
        w = w.reshape(-1, 1)
        g = np.array([[0],[0],[9.81]])
        R = np.array([self.get_rotation_matrix(q_.reshape(-1, 1)) for q_ in q]) #Nx3x3
        Omega = self.get_quaternion_update_matrix(w)
        norm_w = self.compute_norm_w(w)

        A = np.cos(norm_w*dt/2) * np.eye(4)
        B = (1/norm_w)*np.sin(norm_w*dt/2) * Omega
        
        # predict state vector x
        acc_val = (R @ a - g)
        acc_val_reshaped = acc_val.reshape(acc_val.shape[0], acc_val.shape[1])
        p_k = p + v * dt + acc_val_reshaped*dt**2 / 2 # Nx3
        v_k = v + acc_val_reshaped * dt # Nx3
        q_k = q @ np.array(A + B) # Nx4
        
        return np.concatenate([
            p_k,
            v_k,
            q_k,
        ], axis=1)
    
    def ekf_predict_setup1_2(self, u, dt, Q_k, Q_p):
        q_k = self.x_k[:, 6:]
        q_p = self.x_p[:, 6:]
        a = u[:3]
        w = u[3:]
        a = a.reshape(-1, 1)
        w = w.reshape(-1, 1)
        G_p = np.eye(Q_p.shape[0])

        x_k = self.motion_model_1_2(x=self.x_k, u=u, dt=dt)

        for i in range(self.N):
            F_k, G_k = self.get_jacobian_setup1_2(q=q_k[i], u=u, dt=dt)
            F_p, _ = self.get_jacobian_setup1_2(q=q_p[i], u=u, dt=dt)
            F_k_hat = F_k
            Q_k_hat = Q_k
            S = G_p @ Q_p @ G_p.T + F_p @ self.P[i] @ F_p.T
            K = F_k_hat @ self.P[i] @ F_p.T @ np.linalg.inv(S)

            self.x_k[i] = (F_k_hat - K @ F_p) @ self.x_k[i] + K @ self.x_p[i] + x_k[i]
            self.P[i] = F_k_hat @ self.P[i] @ F_k_hat.T + G_k @ Q_k_hat @ G_k.T - K @ S @ K.T
            

    def pf_predict_setup1_2(self, u, dt, Q):
        x_p = self.motion_model_1_2(x=self.x_p, u=u, dt=dt)

        q = self.x_p[:, 6:]
        a = u[:3]
        w = u[3:]
        a = a.reshape(-1, 1)
        w = w.reshape(-1, 1)
        G = np.eye(Q.shape[0])
        for i in range(0, self.N):
            F_p, _ = self.get_jacobian_setup1_2(q=q[i], u=u, dt=dt)
            x_mean = x_p[i] + F_p @ self.x_k[i]
            cov = F_p @ self.P[i] @ F_p.T + G @ Q @ G.T

            self.x_p[i] = np.random.multivariate_normal(mean=x_mean, cov=cov)

    def ekf_measurement_update_1_2(self, z, R):
        y_x_p = self.H @ self.x_p.T
        y_x_k = self.H @ self.x_k.T
        y_hat = y_x_p + y_x_k
        for i in range(self.N):
            S = self.H @ self.P[i] @ self.H.T + R
            K = self.P[i] @ self.H.T @ np.linalg.inv(S)
            residual = z - y_hat.T[i].reshape(-1, 1)

            self.P[i] = self.P[i] - K @ S @ K.T
            self.x_k[i] = self.x_k[i] + (K @ residual).reshape(-1)

    def pf_measurement_update_1_2(self, z, R, is_weight_update=True):
        y_x_p = self.H @ self.x_p.T
        y_x_k = self.H @ self.x_k.T
        y_hat = y_x_p + y_x_k
        for i in range(self.N):
            cov = self.H @ self.P[i] @ self.H.T + R
            if is_weight_update:
                self.weights[i] = multivariate_normal(mean=y_hat.T[i], cov=cov) \
                                    .pdf(z.reshape(-1)) * (self.weights[i] / self.likelihoods[i]) 
            else:
                self.likelihoods[i] = multivariate_normal(mean=y_hat.T[i], cov=cov) \
                                        .pdf(z.reshape(-1)) * self.weights[i]

        if is_weight_update:
            self.weights += 1e-300
            self.weights /= np.sum(self.weights)
        else:
            self.likelihoods += 1e-300
            self.likelihoods /= np.sum(self.likelihoods)

    def predict_setup3(self, u, dt, Q):
        """ 
            move according to control input u (heading change, velocity) with noise std
            u: control input vector
            process_noise_vector: process noise vector
            dt: delta time
        """
        v, omega = u
        r = v / omega  # turning radius
        theta = self.particles[:, 2]
        dtheta = omega * dt
        dx = - r * np.sin(theta) + r * np.sin(theta + dtheta)
        dy = + r * np.cos(theta) - r * np.cos(theta + dtheta)
        delta_x = np.concatenate([
            dx.reshape(-1, 1),
            dy.reshape(-1, 1), 
            (np.ones(self.N) * dtheta).reshape(-1, 1)], axis=1)
        process_noise = np.random.multivariate_normal(np.zeros(Q.shape[0]), Q, self.N)
        self.particles += delta_x + process_noise

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

    def resample_from_index(self, indexes):
        assert len(indexes) == self.N
        
        self.x_p = self.x_p[indexes]
        self.x_k = self.x_k[indexes]
        self.weights = self.weights[indexes]
        self.likelihoods = self.likelihoods[indexes]
        
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

    def estimate(self):
        """ 
            computer posterior of the system by calcurating the weighted average
        """
        pos = self.particles
        mu = np.average(pos, weights=self.weights, axis=0)
        var = np.average((pos - mu)**2, weights=self.weights, axis=0)

        return mu, var

    def visualize_trajectory(self, data, xlim, ylim):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        xs, ys, _ = data.GPS_measurements_in_meter.T
        ax1.plot(xs, ys, lw=2, label='ground-truth trajectory', color='black')
        xs, ys, _ = data.VO_measurements.T
        ax1.plot(xs, ys, lw=2, label='VO trajectory', color='b')
        ax1.plot(
            self.mu_x, self.mu_y, lw=2, 
            label='estimated trajectory', color='r')
        ax1.set_xlabel('X [m]')
        ax1.set_ylabel('Y [m]')
        ax1.legend()
        ax1.grid()
        
        xs, ys, _ = data.GPS_measurements_in_meter.T
        ax2.plot(xs, ys, lw=2, label='ground-truth trajectory', color='black')
        xs, ys, _ = data.VO_measurements.T
        ax2.plot(xs, ys, lw=2, label='VO trajectory', color='b')
        ax2.plot(
            self.mu_x, self.mu_y, lw=2, 
            label='estimated trajectory', color='r')
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)
        ax2.set_xlabel('X [m]')
        ax2.set_ylabel('Y [m]')
        ax2.legend()
        ax2.grid()

    def run(self,
            data, 
            setup=SetupEnum.SETUP_1, 
            measurement_type=MeasurementDataEnum.ALL_DATA, 
            importance_resampling=True,
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
        if setup is SetupEnum.SETUP_1 or setup is SetupEnum.SETUP_2:
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
            print("[RBPF] start.")
        
        for t_idx in tqdm(range(1, data.N), disable=not debug_mode):
            t = data.ts[t_idx]
            dt = t - t_last

            # prediction step(time update)

            # correction step(measurement update)

            t_last = t

        error = mean_squared_error(
            data.GPS_measurements_in_meter.T[:2, :len(mu_x)], 
            np.array([mu_x, mu_y]))
        
        if debug_mode is True:
            print(f"[PF] MSE: {error}")

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





if __name__ == '__main__':
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
    # x_setup1, P_setup1, H_setup1, q1, r_vo1, r_gps1 = data.get_initial_data(setup=SetupEnum.SETUP_1,filter_type=FilterEnum.EKF, noise_type=NoiseTypeEnum.CURRENT)
    x, P, H, q2, r_vo2, r_gps2 = data.get_initial_data(setup=SetupEnum.SETUP_2, filter_type=FilterEnum.EKF, noise_type=NoiseTypeEnum.CURRENT)
    # x_setup3, P_setup3, H_setup3, q3, r_vo3, r_gps3 = data.get_initial_data(setup=SetupEnum.SETUP_3, filter_type=FilterEnum.EKF, noise_type=NoiseTypeEnum.CURRENT)

    # compute p(y | Xp, Y)
    R_vo = np.array([
        [data.VO_noise_std ** 2., 0.],
        [0., data.VO_noise_std ** 2.]
    ])
    R_gps = np.array([
        [data.GPS_measurement_noise_std ** 2., 0.],
        [0., data.GPS_measurement_noise_std ** 2.]
    ])
    q0_noise, q1_noise, q2_noise, q3_noise = data.quaternion_process_noise
    q_noise_p =  [
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
    i = np.eye(len(q_noise_p))
    Q_pf = np.array([[val * num for num in i[ind]] for ind, val in enumerate(q_noise_p)])

    q_noise_k = [data.IMU_acc_noise_std, 
                            data.IMU_acc_noise_std, 
                            data.IMU_acc_noise_std, 
                            data.IMU_angular_velocity_noise_std, 
                            data.IMU_angular_velocity_noise_std, 
                            data.IMU_angular_velocity_noise_std]
    i = np.eye(len(q_noise_k))
    Q_ekf = np.array([[val * num for num in i[ind]] for ind, val in enumerate(q_noise_k)])

    M = np.cov(np.stack((q_noise_k, q_noise_k), axis=1))
    Q = [
        [Q_ekf, M],
        [M.T, Q_ekf]
    ]

    # 1. initialization
    # 2. normalize weights
    rbpf = RaoBlackwellizedParticleFilter(
        N=16,
        x=x,
        P=P,
        H=H
    )
    # 3. - (3.60)
    z_vo = data.VO_measurements_with_noise[0, :2].reshape(-1, 1)
    rbpf.ekf_measurement_update_1_2(z=z_vo, R=R_vo)

    t_last = 0.
    for t_idx in tqdm(range(1, data.N), disable=False):
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
        z_gps = data.GPS_mesurement_in_meter_with_noise[t_idx, :2].reshape(-1, 1)

        # 1. - (3.64)
        rbpf.pf_measurement_update_1_2(z=z_vo, R=R_vo, is_weight_update=False)

        # 2. 
        rbpf.resample()

        # 3. - (3.65)
        rbpf.pf_predict_setup1_2(u=u, dt=dt, Q=Q_pf)
        
        # 4. - (3.61)
        rbpf.ekf_predict_setup1_2(u=u, dt=dt, Q_k=Q_ekf, Q_p=Q_pf)

        # 5. - (3.64), # 9.
        rbpf.pf_measurement_update_1_2(z=z_vo, R=R_vo, is_weight_update=True)

        # 6. - (3.60)
        rbpf.ekf_measurement_update_1_2(z=z_vo, R=R_vo)


        t_last = t