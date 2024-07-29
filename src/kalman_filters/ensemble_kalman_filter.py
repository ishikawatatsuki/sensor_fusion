import sys
if __name__ == "__main__":
    sys.path.append('../../src')

import numpy as np
from configs import MeasurementDataEnum, SetupEnum, FilterEnum, NoiseTypeEnum
import matplotlib.pyplot as plt
from utils.error_report import get_error_report
from tqdm import tqdm

if __name__ == "__main__":
    from base_filter import BaseFilter
else:
    from .base_filter import BaseFilter


class EnsembleKalmanFilter(BaseFilter):

    def __init__(self, N, x, P, H, q, r_vo, r_gps, setup=SetupEnum.SETUP_1):

        self.N = N
        self.x_dim = x.shape[0]
        self.z_dim = H.shape[0]
        self.x = x
        self.H = H
        self.samples = self.generate_ensembles(initial_cov=P)

        self.setup = setup
        self.dimension=self.H.shape[0]
        self.Q = self.get_diagonal_matrix(q)
        self.R_vo = self.get_diagonal_matrix(r_vo)
        self.R_gps = self.get_diagonal_matrix(r_gps)

    def generate_ensembles(self, initial_cov):
        return np.random.multivariate_normal(mean=self.x.reshape(-1), cov=initial_cov, size=self.N)

    def predict_setup1_2(self, u, dt, Q):
        
        p = self.samples[:, :3]
        v = self.samples[:, 3:6]
        q = self.samples[:, 6:]
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

        process_noise_cov = np.random.multivariate_normal(
                                mean=np.zeros(self.x.shape[0]), 
                                cov=Q, 
                                size=self.N)
        
        self.samples = np.concatenate([
            p_k,
            v_k,
            q_k,
        ], axis=1) + process_noise_cov # Nx10
        
        self.x = np.mean(self.samples, axis=0)
        
    def predict_setup3(self, u, dt, Q):
        v, omega = u
        r = v / omega  # turning radius
        theta = self.samples[:, 2]
        dtheta = omega * dt
        dx = - r * np.sin(theta) + r * np.sin(theta + dtheta)
        dy = + r * np.cos(theta) - r * np.cos(theta + dtheta)
        delta_x = np.concatenate([
            dx.reshape(-1, 1), 
            dy.reshape(-1, 1), 
            (np.ones(self.N) * dtheta).reshape(-1, 1)], axis=1)
        process_noise = np.random.multivariate_normal(
                                                mean=np.zeros(self.x.shape[0]), 
                                                cov=Q, 
                                                size=self.N)
        self.samples += delta_x + process_noise
        self.x = np.mean(self.samples, axis=0)
        
    def update(self, z, R):
        samples = self.samples @ self.H.T # Nx2
        y_hat = np.mean(samples, axis=0)
        P_yy = np.zeros((self.z_dim, self.z_dim)) # 2x2
        for sample in samples:
            P_yy += (sample - y_hat).reshape(-1, 1) @ (sample - y_hat).reshape(-1, 1).T
        P_yy /= (self.N-1)
        
        P_xy = np.zeros((self.x_dim, self.z_dim))
        x = self.x.flatten()
        for i, sample in enumerate(self.samples):
            # P_xy += (sample - self.x).reshape(-1, 1) @ (samples[i] - y_hat).reshape(-1, 1).T
            P_xy += np.outer(sample - x, samples[i] - y_hat)
        P_xy /= (self.N-1)

        K = P_xy @ np.linalg.inv(P_yy)
        
        measurement_noise = np.random.multivariate_normal(
                                mean=np.zeros(self.z_dim), 
                                cov=R, size=self.N)
        residuals = z - samples + measurement_noise # Nx2
        self.samples = self.samples + residuals @ K.T
        self.x = np.mean(self.samples, axis=0)
        
    def _time_update_step(self, data, t_idx, dt, Q):
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
            self.predict_setup1_2(u=u, dt=dt, Q=Q)
        else: #SetupEnum.SETUP_3
            u = np.array([
                data.INS_velocities_with_noise[t_idx, 0],
                data.IMU_angular_velocity_with_noise[t_idx, 2]
            ])
            self.predict_setup3(u=u, dt=dt, Q=Q)
    
    def _measurement_update_step(self, data, t_idx, R_vo, R_gps, measurement_type):
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
            self.update(z=z_vo.reshape(-1), R=R_vo)
        
        if z_gps is not None:
            self.update(z=z_gps.reshape(-1), R=R_gps)

    def run(self, 
            data, 
            measurement_type=MeasurementDataEnum.ALL_DATA, 
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

        mu_x = [self.x[0, 0],]
        mu_y = [self.x[1, 0],]
        mu_z = [self.x[2, 0],]
        
        t_last = 0.

        if debug_mode is True:
            print("[EnKF] start.")
        for t_idx in tqdm(range(1, data.N), disable=not debug_mode):
            t = data.ts[t_idx]
            dt = t - t_last

            # prediction step(time update)
            self._time_update_step(data, t_idx, dt, Q)
            
            x_hat = self.x.copy()
            mu_x.append(x_hat[0])
            mu_y.append(x_hat[1])
            mu_z.append(x_hat[2])

            # correction step(measurement update)
            self._measurement_update_step(data, t_idx, R_vo, R_gps, measurement_type)
            
            if show_graph is True:
                if self.N > 100:
                    samples_indices = np.linspace(0, self.N-1, 100, dtype=int).tolist()
                    ax1.scatter(
                        self.samples[samples_indices, 0], 
                        self.samples[samples_indices, 1], alpha=.2, s=[10])
                else:
                    ax1.scatter(self.samples[:, 0], self.samples[:, 1], alpha=.2, s=[10])
            
            t_last = t
            
        error = \
            get_error_report(
                    data.GPS_measurements_in_meter.T[:2, :len(mu_x)], 
                    np.array([mu_x, mu_y]))\
            if self.H.shape[0] == 2 else\
            get_error_report(
                data.GPS_measurements_in_meter.T[:3, :len(mu_x)], 
                np.array([mu_x, mu_y, mu_z])) 

        if debug_mode is True:
            print(f"[EnKF] errors: {error}")
            
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
    dimension=3

    data = DataLoader(sequence_nr=kitti_drive, 
                    kitti_root_dir=kitti_root_dir, 
                    vo_root_dir=vo_root_dir,
                    noise_vector_dir=noise_vector_dir,
                    vo_dropout_ratio=0.7, 
                    gps_dropout_ratio=0.,
                    dimension=dimension)

    x_setup1, P_setup1, H_setup1, q1, r_vo1, r_gps1 = data.get_initial_data(setup=SetupEnum.SETUP_1, filter_type=FilterEnum.EnKF, noise_type=NoiseTypeEnum.CURRENT)
    x_setup2, P_setup2, H_setup2, q2, r_vo2, r_gps2 = data.get_initial_data(setup=SetupEnum.SETUP_2, filter_type=FilterEnum.EnKF, noise_type=NoiseTypeEnum.CURRENT)
    x_setup3, P_setup3, H_setup3, q3, r_vo3, r_gps3 = data.get_initial_data(setup=SetupEnum.SETUP_3, filter_type=FilterEnum.EnKF, noise_type=NoiseTypeEnum.CURRENT)
    n_ensemble_setup1_0 = 64
    n_ensemble_setup2_0 = 64
    n_ensemble_setup3_0 = 64
    measurement_type = MeasurementDataEnum.ALL_DATA

    # enkf1_0 = EnsembleKalmanFilter(
    #     N=n_ensemble_setup1_0, 
    #     x=x_setup1.copy(), 
    #     P=P_setup1.copy(), 
    #     H=H_setup1.copy(),
    #     q=q1,
    #     r_vo=r_vo1,
    #     r_gps=r_gps1,
    #     setup=SetupEnum.SETUP_1)
    # error_enkf1_0 = enkf1_0.run(data=data, debug_mode=True, measurement_type=measurement_type)
    
    # estimated = enkf1_0.get_estimated_trajectory()[:, :dimension]
    # actual = data.GPS_measurements_in_meter[:, :dimension]
    # print(np.sum((actual - estimated) ** 2))

    # enkf1_0.visualize_trajectory(data=data, dimension=dimension, interval=5)

    enkf2_0 = EnsembleKalmanFilter(
        N=n_ensemble_setup2_0, 
        x=x_setup2.copy(), 
        P=P_setup2.copy(), 
        H=H_setup2.copy(),
        q=q2,
        r_vo=r_vo2,
        r_gps=r_gps2,
        setup=SetupEnum.SETUP_2)
    error_enkf2_0 = enkf2_0.run(data=data, debug_mode=True, measurement_type=measurement_type)

    estimated = enkf2_0.get_estimated_trajectory()[:, :dimension]
    actual = data.GPS_measurements_in_meter[:, :dimension]
    print(np.sum((actual - estimated) ** 2))

    enkf2_0.visualize_trajectory(data=data, dimension=dimension, interval=5)
    
    # enkf3_0 = EnsembleKalmanFilter(
    #     N=n_ensemble_setup3_0, 
    #     x=x_setup3.copy(), 
    #     P=P_setup3.copy(), 
    #     H=H_setup3.copy(),
    #     q=q3,
    #     r_vo=r_vo3,
    #     r_gps=r_gps3,
    #     setup=SetupEnum.SETUP_3)
    # error_enkf3_0 = enkf3_0.run(data=data, show_graph=True, debug_mode=True, measurement_type=measurement_type)

    # estimated = enkf3_0.get_estimated_trajectory()[:, :dimension]
    # actual = data.GPS_measurements_in_meter[:, :dimension]
    # print(np.sum((actual - estimated) ** 2))

    # enkf3_0.visualize_trajectory(data=data, dimension=dimension, interval=5)