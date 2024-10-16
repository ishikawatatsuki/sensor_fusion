import os
import sys
if __name__ == "__main__":
    sys.path.append('../../src')

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from ahrs import Quaternion
from scipy.linalg import cholesky
from utils.time_reporter import time_measurer
from utils.error_report import get_error_report, print_error_report
from configs import MeasurementDataEnum, SetupEnum, FilterEnum, NoiseTypeEnum

if __name__ == "__main__":
    from base_filter import BaseFilter
else:
    from .base_filter import BaseFilter


np.random.seed(777)

# https://kalman-filter.com/cubature-kalman-filter/
class CubatureKalmanFilter(BaseFilter):

    sigma_points = None
    
    def __init__(self, x, P, H, q, r_vo, r_gps, setup=SetupEnum.SETUP_1):
        """ 
        Args:
            x (numpy.array): state to estimate: 
                setup1 or setup2: [px, py, pz, vx, vy, vz, q1, q2, q3, q4] 
                setup3          : [px, py, theta]
            P (numpy.array): state error covariance matrix
            H (numpy.array): transition matrix from predicted state vector to measurement space
            q (numpy.array): process noise vector
            r_vo (numpy.array): measurement noise vector for VO
            r_gps (numpy.array): measurement noise vector for GPS
            setup (SetupEnum): filter setup
        """
        self.N = len(x)
        self.x = x
        self.P = P
        self.H = H
        self.m = self.N * 2
        self.W = 1 / self.m
        self.setup = setup
        self.dimension=self.H.shape[0]
        self.Q = self.get_diagonal_matrix(q)
        self.R_vo = self.get_diagonal_matrix(r_vo)
        self.R_gps = self.get_diagonal_matrix(r_gps)

    def compute_sigma_points(self):
        L = cholesky(self.P)
        coef = L * np.sqrt(self.N)
        mean = self.x.flatten()
        sigma_plus = [mean + coef[i] for i in range(self.N)]
        sigma_minus = [mean - coef[i] for i in range(self.N)]
        return np.concatenate([sigma_plus, sigma_minus])

    def predict_setup1_2(self, u, dt, Q):

        sigma_points = self.compute_sigma_points()
        p = sigma_points[:, :3]
        v = sigma_points[:, 3:6]
        q = sigma_points[:, 6:]
        a = u[:3]
        w = u[3:]
        a = a.reshape(-1, 1)
        w = w.reshape(-1, 1)
        g = np.array([[0],[0],[9.81]])
        R = np.array([self.get_rotation_matrix(q_.reshape(-1, 1)) for q_ in q]) #20x3x3
        Omega = self.get_quaternion_update_matrix(w)
        norm_w = self.compute_norm_w(w)

        A = np.cos(norm_w*dt/2) * np.eye(4)
        B = (1/norm_w)*np.sin(norm_w*dt/2) * Omega

        acc_val = (R @ a - g)
        acc_val_reshaped = acc_val.reshape(acc_val.shape[0], acc_val.shape[1])
        p_k = p + v * dt + acc_val_reshaped*dt**2 / 2 # 20x3
        v_k = v + acc_val_reshaped * dt # 20x3
        q_k = q @ np.array(A + B) # 20x4
        q_k = np.array([q_ / np.linalg.norm(q_) if np.linalg.norm(q_) > 0 else q_  for q_ in q_k])
        
        self.sigma_points = np.concatenate([
            p_k,
            v_k,
            q_k,
        ], axis=1) # 20x10
        
        # compute mean value of sigma points
        self.x = np.sum(self.W * self.sigma_points, axis=0).reshape(-1, 1) # 10x1

        # compute covariance matrix for sigma points
        P = np.zeros((self.N, self.N)) # 10x10
        for i, sigma_point in enumerate(self.sigma_points):
            # x = sigma_point.reshape(-1, 1)
            # P += self.W * (np.outer(x, x.T) - np.outer(self.x, self.x.T))
            var = sigma_point.reshape(-1, 1) - self.x
            P += self.W * (var @ var.T)
        
        self.P = P + Q # 10x10 additive process noise

    def predict_setup3(self, u, dt, Q):
        if self.dimension == 2:
            sigma_points = self.compute_sigma_points() # 7x3
            x, y, theta = sigma_points.T
            x = x.reshape(-1, 1)
            y = y.reshape(-1, 1)
            theta = theta.reshape(-1, 1)
            v, omega = u
            r = v / omega  # turning radius

            dtheta = omega * dt
            dx = - r * np.sin(theta) + r * np.sin(theta + dtheta)
            dy = + r * np.cos(theta) - r * np.cos(theta + dtheta)
            x += dx
            y += dy
            theta += dtheta
            
            self.sigma_points = np.concatenate([x, y, theta], axis=1)

            self.x = np.sum(self.W * self.sigma_points, axis=0).reshape(-1, 1) # 3x1

            P = np.zeros((self.N, self.N)) # 3x3
            for i, sigma_point in enumerate(self.sigma_points):
                # x = sigma_point.reshape(-1, 1)
                # P += self.W * (np.outer(x, x.T) - np.outer(self.x, self.x.T))
                var = sigma_point.reshape(-1, 1) - self.x
                P += self.W * (var @ var.T)
            self.P = P + Q
        else:
            chi = self.compute_sigma_points()
            x, y, z, phi, psi = chi.T
            x = x.reshape(-1, 1)
            y = y.reshape(-1, 1)
            z = z.reshape(-1, 1)
            phi = phi.reshape(-1, 1)
            psi = psi.reshape(-1, 1)
        
            v, wx, wz = u
            rx = v / wx
            rz = v / wz
            dphi = wx * dt
            dpsi = wz * dt
            dx = - rz * np.sin(psi) + rz * np.sin(psi + dpsi)
            dy = + rz * np.cos(psi) - rz * np.cos(psi + dpsi)
            dz = + rx * np.cos(phi) - rx * np.cos(phi + dphi)
            
            x += dx
            y += dy
            z += dz
            phi += dphi
            psi += dpsi

            self.sigma_points = np.concatenate([x, y, z, phi, psi], axis=1)
            
            self.x = np.sum(self.W * self.sigma_points, axis=0).reshape(-1, 1) # 3x1
            
            P = np.zeros((self.N, self.N)) # 5x5
            for sigma_point in self.sigma_points:
                var = sigma_point.reshape(-1, 1) - self.x
                P += self.W * (var @ var.T)
            self.P = P + Q

    def update(self, z, R):
        sigma_points = self.compute_sigma_points()
        # traject sigma points into the measurement space
        y_sigma_points = sigma_points @ self.H.T # 20x2
        # compute expected measurement value
        y_hat = np.sum(self.W * y_sigma_points, axis=0).reshape(-1, 1) # 2x1

        x_dim = sigma_points.shape[1]
        z_dim = y_sigma_points.shape[1]

        # compute covariance matrix for residuals
        P_y = np.zeros((z_dim, z_dim)) # 2x2
        for i, y_sigma_point in enumerate(y_sigma_points):
            # y = y_sigma_point.reshape(-1, 1)
            # P_y += self.W * (np.outer(y, y.T) - np.outer(y_hat, y_hat.T))
            var_y = y_sigma_point.reshape(-1, 1) - y_hat
            P_y += self.W * (var_y @ var_y.T)

        P_y += R # additive measurement noise
        
        # compute cross-covariance matrix 
        P_xy = np.zeros((x_dim, z_dim)) # 10x2
        for idx in range(self.N):
            var_x = sigma_points[idx].reshape(-1, 1) - self.x
            var_y = y_sigma_points[idx].reshape(-1, 1) - y_hat
            # P_xy += self.W * (np.outer(x, y.T) - np.outer(self.x, y_hat.T))
            P_xy += self.W * (var_x @ var_y.T)
        
        # compute kalman gain
        K = P_xy @ np.linalg.inv(P_y)

        # compute residual
        residual = z.reshape(-1, 1) - y_hat
        # update state vector and error covariance matrix
        self.x = self.x + K @ residual
        self.P = self.P - K @ P_y @ K.T
        
        self.errors.append(np.sqrt(np.sum(residual**2)))
        
    def _time_update_step(self, data, t_idx, dt, Q):
        u = data.get_control_input_by_index(index=t_idx, setup=self.setup)
        predict = self.predict_setup1_2 if self.setup is SetupEnum.SETUP_1 or\
                                                self.setup is SetupEnum.SETUP_2 else self.predict_setup3
        
        predict(u=u, dt=dt, Q=Q)
    
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
            self.update(z=z_vo, R=R_vo)
        
        if z_gps is not None:
            self.update(z=z_gps, R=R_gps)
            
    def run(self, 
            data, 
            measurement_type=MeasurementDataEnum.ALL_DATA, 
            debug_mode=False,
            show_graph=False):

        # measurement noise
        R_vo = self.R_vo
        R_gps = self.R_gps
        # process noise
        Q = self.Q
        
        gt = data.get_trajectory_to_compare()

        mu_x = [self.x[0, 0],]
        mu_y = [self.x[1, 0],]
        mu_z = [self.x[2, 0],]
        
        t_last = 0.
        if debug_mode is True:
            print("[CKF] start.")
        for t_idx in tqdm(range(1, data.N), disable=not debug_mode):
            t = data.ts[t_idx]
            dt = t - t_last

            # prediction step(time update)
            self._time_update_step(data, t_idx, dt, Q)
            
            x_hat = self.x.copy()
            mu_x.append(x_hat[0, 0])
            mu_y.append(x_hat[1, 0])
            mu_z.append(x_hat[2, 0])

            # correction step(measurement update)
            self._measurement_update_step(data, t_idx, R_vo, R_gps, measurement_type)

            t_last = t
            
        error = \
            get_error_report(
                    gt[:2, :len(mu_x)], 
                    np.array([mu_x, mu_y]))\
            if self.H.shape[0] == 2 else\
            get_error_report(
                gt[:3, :len(mu_x)], 
                np.array([mu_x, mu_y, mu_z])) 

        if debug_mode:
            print_error_report(error, f"[CKF] Error report for {SetupEnum.get_name(self.setup)}")

        if show_graph is True:
            fig, ax1 = plt.subplots(1, 1, figsize=(12, 9))
            xs, ys, _ = gt
            ax1.plot(xs, ys, lw=2, label='ground-truth trajectory', color='black')
            xs, ys, _ = data.VO_measurements.T
            ax1.plot(xs, ys, lw=2, label='VO trajectory', color='b')
            ax1.plot(
                mu_x, mu_y, lw=2,
                label='estimated trajectory', color='r')
            ax1.set_xlabel('X [m]')
            ax1.set_ylabel('Y [m]')
            ax1.legend()
            ax1.grid()
            
        self.mu_x = mu_x
        self.mu_y = mu_y
        self.mu_z = mu_z
        
        return error
    
    @time_measurer
    def run_with_time(self, *args, **kwargs):
        return self.run(*args, **kwargs)

if __name__ == "__main__":
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
        dimension=dimension)
    
    filter_type=FilterEnum.CKF
    noise_type=NoiseTypeEnum.CURRENT

    x_setup1, P_setup1, H_setup1, q1, r_vo1, r_gps1 = data.get_initial_data(setup=SetupEnum.SETUP_1, filter_type=filter_type, noise_type=noise_type)
    x_setup2, P_setup2, H_setup2, q2, r_vo2, r_gps2 = data.get_initial_data(setup=SetupEnum.SETUP_2, filter_type=filter_type, noise_type=noise_type)
    x_setup3, P_setup3, H_setup3, q3, r_vo3, r_gps3 = data.get_initial_data(setup=SetupEnum.SETUP_3, filter_type=filter_type, noise_type=noise_type)
    
    measurement_type = MeasurementDataEnum.ALL_DATA
    debug_mode=True
    interval = 5

    ckf1_0 = CubatureKalmanFilter(
        x=x_setup1.copy(), 
        P=P_setup1.copy(), 
        H=H_setup1.copy(),
        q=q1,
        r_vo=r_vo1,
        r_gps=r_gps1,
        setup=SetupEnum.SETUP_1,
    )
    error_ckf1_0 = ckf1_0.run(
        data=data, 
        debug_mode=debug_mode,
        measurement_type=measurement_type
    )
    
    ckf1_0.visualize_trajectory(
        data=data, 
        dimension=dimension, 
        interval=interval, 
        title="CKF Setup1 trajectories")
    
    ckf2_0 = CubatureKalmanFilter(
        x=x_setup2.copy(), 
        P=P_setup2.copy(), 
        H=H_setup2.copy(),
        q=q2,
        r_vo=r_vo2,
        r_gps=r_gps2,
        setup=SetupEnum.SETUP_2,
    )
    error_ckf2_0 = ckf2_0.run(
        data=data, 
        debug_mode=debug_mode,
        measurement_type=measurement_type
    )

    ckf2_0.visualize_trajectory(
        data=data, 
        dimension=dimension, 
        interval=interval, 
        title="CKF Setup2 trajectories")
    
    ckf3_0 = CubatureKalmanFilter(
        x=x_setup3.copy(), 
        P=P_setup3.copy(), 
        H=H_setup3.copy(),
        q=q3,
        r_vo=r_vo3,
        r_gps=r_gps3,
        setup=SetupEnum.SETUP_3,
    )
    error_ckf3_0 = ckf3_0.run(
        data=data, 
        debug_mode=debug_mode,
        measurement_type=measurement_type
    )

    ckf3_0.visualize_trajectory(
        data=data, 
        dimension=dimension, 
        interval=interval, 
        title="CKF Setup3 trajectories")