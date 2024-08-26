# references: 
# [1] https://docs.ufpr.br/~danielsantos/ProbabilisticRobotics.pdf

import sys
if __name__ == "__main__":
    sys.path.append('../../src')

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.error_report import get_error_report, print_error_report
from configs import MeasurementDataEnum, SetupEnum, FilterEnum, NoiseTypeEnum

if __name__ == "__main__":
    from base_filter import BaseFilter
else:
    from .base_filter import BaseFilter

class ExtendedKalmanFilter(BaseFilter):
    """Extended Kalman Filter
    for vehicle whose motion is modeled as eq. (5.9) in [1]
    and with observation of its 2d location (x, y)
    """
    x = None
    P = None
    Q = None
    R_vo = None
    R_gps = None
    
    def __init__(
        self, 
        x,
        P, 
        H, 
        q, 
        r_vo, 
        r_gps, 
        setup=SetupEnum.SETUP_1,
        ):
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
        self.P = P
        self.H = H
        self.x = x
        self.setup = setup
        self.dimension=self.H.shape[0]
        self.Q = self.get_diagonal_matrix(q)
        self.R_vo = self.get_diagonal_matrix(r_vo)
        self.R_gps = self.get_diagonal_matrix(r_gps)
        
    def predict_setup1_2(self, u, dt, Q):
        """estimate x and P based on previous stete of x and control input u
        Args:
            u  (numpy.array): control input u
            dt (numpy.array): difference of current time and previous time
            Q  (numpy.array): process noise 
        """
        # propagate state x
        p = self.x[:3]
        v = self.x[3:6]
        q = self.x[6:]
        a = u[:3]
        w = u[3:]
        a = a.reshape(-1, 1)
        w = w.reshape(-1, 1)
        g = np.array([[0],[0],[9.81]])
        R = self.get_rotation_matrix(q)
        Omega = self.get_quaternion_update_matrix(w)
        norm_w = self.compute_norm_w(w)

        A = np.cos(norm_w*dt/2) * np.eye(4)
        B = (1/norm_w)*np.sin(norm_w*dt/2) * Omega
        
        p_k = p + v * dt + (R @ a - g)*dt**2 / 2
        v_k = v + (R @ a - g) * dt
        q_k = np.array(A + B) @ q
        q_k /= np.linalg.norm(q_k)
        
        self.x = np.concatenate([
            p_k,
            v_k,
            q_k,
        ])
        
        ax, ay, az, wx, wy, wz = u
        q1, q2, q3, q4 = q[:, 0]
        q1_2, q2_2, q3_2, q4_2 = q[:, 0]**2 
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
        # predict state covariance matrix P
        self.P = F @ self.P @ F.T + G @ Q @ G.T
        
    def predict_setup3(self, u, dt, Q):
        """estimate x and P based on previous stete of x and control input u
        Args:
            u  (numpy.array): control input u
            dt (numpy.array): difference of current time and previous time
            Q  (numpy.array): process noise 
        """
        # propagate state x
        x, y, theta = self.x[:, 0]
        v, omega = u
        r = v / omega  # turning radius

        dtheta = omega * dt
        dx = - r * np.sin(theta) + r * np.sin(theta + dtheta)
        dy = + r * np.cos(theta) - r * np.cos(theta + dtheta)
        self.x += np.array([dx, dy, dtheta]).reshape(-1, 1)

        # propagate covariance P
        # Jacobian of state transition function
        F = np.array([
            [1., 0., - r * np.cos(theta) + r * np.cos(theta + dtheta)],
            [0., 1., - r * np.sin(theta) + r * np.sin(theta + dtheta)],
            [0., 0., 1.]
        ]) 

        # Jacobian of state transition function
        G = np.array([
            [-np.sin(theta)/omega + np.sin(theta + dtheta)/omega, dt*v*np.cos(theta+dtheta)/omega + v*np.sin(theta)/omega**2 - v*np.sin(theta+dtheta)/omega**2],
            [np.cos(theta)/omega - np.cos(theta+dtheta)/omega, dt*v*np.sin(theta+dtheta)/omega - v*np.cos(theta)/omega**2 + v*np.cos(theta+dtheta)/omega**2],
            [0., dt]
        ]) 
        self.P = F @ self.P @ F.T + G @ Q @ G.T

    def update(self, z, R):
        """update x and P based on observation of (x_, y_)
        Args:
            z (numpy.array): measurement for [x_, y_]^T
            R (numpy.array): measurement noise covariance
        """
        # compute Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + R)
        # update state x
        z_ = np.dot(self.H, self.x)  # expected observation from the estimated state
        self.x = self.x + K @ (z - z_)

        self.errors.append(np.sqrt(np.sum((z-z_)**2)))
        
        # update covariance P
        self.P = self.P - K @ self.H @ self.P

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

        mu_x = [self.x[0, 0],]
        mu_y = [self.x[1, 0],]
        mu_z = [self.x[2, 0],]
        t_last = 0.

        if debug_mode is True:
            print("[EKF] start.")
            
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
                    data.GPS_measurements_in_meter.T[:2, :len(mu_x)], 
                    np.array([mu_x, mu_y]))\
            if self.H.shape[0] == 2 else\
            get_error_report(
                data.GPS_measurements_in_meter.T[:3, :len(mu_x)], 
                np.array([mu_x, mu_y, mu_z])) 
        
        if debug_mode is True:
            print_error_report(error, f"[EKF] Error report for {SetupEnum.get_name(self.setup)}")

        if show_graph is True:
            fig, ax1 = plt.subplots(1, 1, figsize=(12, 9))
            xs, ys, _ = data.GPS_measurements_in_meter.T
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

class InternalExtendedKalmanFilter(ExtendedKalmanFilter):
    """
    Extended Kalman Filter declared internally in other filters.
    The filter is used to propagate current state based on Kinematic equation and estimate forward velocity.

    Args:
        Arguments are completely same as the EKF.
    """

    
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.t_last = 0
        self.forward_velocity = 0

    def _time_update_step(self, data, t_idx, dt, Q):
        ax, ay, az = data.IMU_acc_with_noise_original[t_idx]
        wx, wy, wz = data.IMU_angular_velocity_with_noise_original[t_idx]
        u = np.array([
            ax,
            ay,
            az,
            wx,
            wy,
            wz
        ])
        prev_p = self.x[:3].copy()
        self.predict_setup1_2(u=u, dt=dt, Q=Q)
        self.forward_velocity = np.linalg.norm(self.x[:3] - prev_p) / dt
        

    def _measurement_update_step(self, data, t_idx, t, R_vo, measurement_type):
        # z_vo, _R_vo = data.get_vo_measurement_by_index_custom(index=t_idx)
        # z_vo_prev, _ = data.get_vo_measurement_by_index_custom(index=t_idx-1)
        
        z_vo, _R_vo = data.get_vo_measurement_by_index_custom(index=t_idx)
        if measurement_type is MeasurementDataEnum.COVARIANCE:
            R_vo = _R_vo

        if z_vo is not None:
            dt = t - self.t_last
            z_vo_prev = data.get_prev_vo_measurement_from_current_index(index=t_idx)
            z = np.concatenate([
                z_vo,
                (z_vo-z_vo_prev) / dt,
            ]) # px, py, pz, vx, vy, vz
            
            self.update(z=z, R=R_vo*10)
            self.t_last = t

    def get_forward_velocity(self):
        return self.forward_velocity
    

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

    data = DataLoader(sequence_nr=kitti_drive, 
                    kitti_root_dir=kitti_data_root_dir, 
                    noise_vector_dir=noise_vector_dir,
                    vo_dropout_ratio=0., 
                    gps_dropout_ratio=0.,
                    dimension=dimension)
    
    filter_type=FilterEnum.EKF
    noise_type=NoiseTypeEnum.CURRENT
    
    x_setup1, P_setup1, H_setup1, q1, r_vo1, r_gps1 = data.get_initial_data(setup=SetupEnum.SETUP_1, filter_type=filter_type, noise_type=noise_type)
    x_setup2, P_setup2, H_setup2, q2, r_vo2, r_gps2 = data.get_initial_data(setup=SetupEnum.SETUP_2, filter_type=filter_type, noise_type=noise_type)
    x_setup3, P_setup3, H_setup3, q3, r_vo3, r_gps3 = data.get_initial_data(setup=SetupEnum.SETUP_3, filter_type=filter_type, noise_type=noise_type)
    
    measurement_type=MeasurementDataEnum.ALL_DATA
    debug_mode=True
    interval = 5

    ekf1_0 = ExtendedKalmanFilter(
        x=x_setup1.copy(), 
        P=P_setup1.copy(), 
        H=H_setup1.copy(),
        q=q1,
        r_vo=r_vo1,
        r_gps=r_gps1,
        setup=SetupEnum.SETUP_1
    )
    error_ekf1_0 = ekf1_0.run(
        data=data, 
        debug_mode=debug_mode,
        measurement_type=measurement_type)

    ekf1_0.visualize_trajectory(
        data=data, 
        dimension=dimension, 
        interval=interval, 
        title="EKF Setup1 trajectories")

    ekf2_0 = ExtendedKalmanFilter(
        x=x_setup2.copy(), 
        P=P_setup2.copy(), 
        H=H_setup2.copy(),
        q=q2,
        r_vo=r_vo2,
        r_gps=r_gps2,
        setup=SetupEnum.SETUP_2
        )
    error_ekf2_0 = ekf2_0.run(
        data=data, 
        debug_mode=debug_mode, 
        measurement_type=measurement_type)
    
    ekf2_0.visualize_trajectory(
        data=data, 
        dimension=dimension, 
        interval=interval, 
        title="EKF Setup2 trajectories")

    ekf3_0 = ExtendedKalmanFilter(
        x=x_setup3.copy(), 
        P=P_setup3.copy(), 
        H=H_setup3.copy(),
        q=q3,
        r_vo=r_vo3,
        r_gps=r_gps3,
        setup=SetupEnum.SETUP_3
        )
    error_ekf3_0 = ekf3_0.run(
        data=data, 
        debug_mode=debug_mode, 
        measurement_type=measurement_type)
    
    ekf3_0.visualize_trajectory(
        data=data, 
        dimension=dimension, 
        interval=interval, 
        title="EKF Setup3 trajectories")