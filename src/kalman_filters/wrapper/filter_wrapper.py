import sys
if __name__ == "__main__":
    sys.path.append('../../../src')

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from configs import SetupEnum, MeasurementDataEnum, FilterEnum, NoiseTypeEnum
from utils.error_report import get_error_report, print_error_report

if __name__ == "__main__":
    from kalman_filters.extended_kalman_filter import InternalExtendedKalmanFilter
else:
    from ..extended_kalman_filter import InternalExtendedKalmanFilter


class FilterWrapper:



    def __init__(
        self, 
        main_filter, 
        filter_type=FilterEnum, 
        omit_gps=True
    ):
        assert main_filter.setup is SetupEnum.SETUP_3, "Please configure the filter with setup3"
        self.main_filter = main_filter
        self.filter_type = filter_type
        self.omit_gps = omit_gps
    
        self.ekf = None

        self.fig = None
        self.ax1 = None
        self.ax2 = None
    
        self.fv_history = []

    def set_omit_gps(self, omit_gps=True):
        self.omit_gps = omit_gps

    def _init_ekf(self, data):
        x, P, _, q, _, _ = data.get_initial_data(
            setup=SetupEnum.SETUP_1, 
            filter_type=FilterEnum.EKF,
            noise_type=NoiseTypeEnum.DEFAULT
        )
        H = np.eye(10)[:6] # for px, py, pz, vx, vy, vz
        r_vo = np.ones(6)
        self.ekf = InternalExtendedKalmanFilter(
            x=x.copy(), 
            P=P.copy(), 
            H=H.copy(),
            q=q,
            r_vo=r_vo,
            r_gps=np.zeros(1), # not used
            setup=SetupEnum.SETUP_1
        ) 

    def _time_update_step(self, data, t_idx, dt, Q):
        vf = self.ekf.get_forward_velocity()
        u = np.array([
            data.INS_velocities_with_noise[t_idx, 0],
            data.IMU_angular_velocity_with_noise[t_idx, 2]
        ])
        self.main_filter.predict_setup3(u=u, dt=dt, Q=Q)
        
        self.fv_history.append(vf)
        
    def _measurement_update_step(self, data, t_idx, R_vo, R_gps, measurement_type):
        z_vo, _R_vo = data.get_vo_measurement_by_index(
            index=t_idx, 
            measurement_type=measurement_type)
        z_gps, _R_gps = data.get_gps_measurement_by_index(
            index=t_idx, 
            setup=self.main_filter.setup, 
            measurement_type=measurement_type)
        
        if measurement_type is MeasurementDataEnum.COVARIANCE:
            R_vo = _R_vo
            R_gps = _R_gps
        
        if z_vo is not None:
            self.main_filter.update(z=z_vo, R=R_vo)
        
        if z_gps is not None and not self.omit_gps:
            self.main_filter.update(z=z_gps, R=R_gps)

        if self.filter_type is FilterEnum.PF:
            # If the given filter is Particle Filter, consider resampling
            
            # check if all sensor data is available
            sensor_data_available = z_vo is not None if self.omit_gps else z_vo is not None and z_gps is not None
            
            # Resample when all sensor data is available and allowed by importance resampling
            if sensor_data_available and self.main_filter._allow_resampling(importance_resampling=True):
                self.main_filter.resample()

    def run(self, 
            data,
            measurement_type=MeasurementDataEnum.ALL_DATA, 
            debug_mode=False,
            show_graph=False):

        filter_name = FilterEnum.get_names()[self.filter_type.value - 1]
        setup_name = SetupEnum.get_name(self.main_filter.setup)
        
        # initialize EKF
        self._init_ekf(data)
        
        # measurement noise
        R_vo = self.main_filter.R_vo
        R_gps = self.main_filter.R_gps
        # process noise
        Q = self.main_filter.Q

        # initialize estimation log
        self._init_estimate_log()

        if debug_mode:
            print(f"[{filter_name}] start.")

        if show_graph:
            self._init_graph(data)
        
        t_last = 0.
        for t_idx in tqdm(range(1, data.N), disable=not debug_mode):
            t = data.ts[t_idx]
            dt = t - t_last

            # prediction step(time update)
            self._time_update_step(data, t_idx, dt, Q)
            self.ekf._time_update_step(data, t_idx, dt, self.ekf.Q)

            # append current estimate
            self._add_estimate_log()

            # correction step(measurement update)
            self._measurement_update_step(data, t_idx, R_vo, R_gps, measurement_type)
            self.ekf._measurement_update_step(data, t_idx, t, self.ekf.R_vo, measurement_type)

            if show_graph:
                self._show_progress_graph()

            t_last = t
            
        error = \
            get_error_report(
                    data.GPS_measurements_in_meter.T[:2, :len(self.main_filter.mu_x)], 
                    np.array([self.main_filter.mu_x, self.main_filter.mu_y]))\
            if self.main_filter.H.shape[0] == 2 else\
            get_error_report(
                data.GPS_measurements_in_meter.T[:3, :len(self.mu_x)], 
                np.array([self.main_filter.mu_x, self.main_filter.mu_y, self.main_filter.mu_z])) 
            
        if debug_mode:
            print_error_report(error, f"[{filter_name}] Error report for {setup_name}")

        if show_graph:
            self._show_graph(data)
            
        return error 

    def _init_estimate_log(self):
        self.main_filter.mu_x = [self.main_filter.x[0, 0],]
        self.main_filter.mu_y = [self.main_filter.x[1, 0],]
        self.main_filter.mu_z = [self.main_filter.x[2, 0],]
        
    def _add_estimate_log(self):
        if self.filter_type is FilterEnum.PF:
            x_hat, _ = self.main_filter.estimate()
            _x, _y, _z = x_hat
        elif self.filter_type is FilterEnum.EnKF:
            x_hat = self.main_filter.x.copy()
            _x, _y, _z = x_hat
        else:
            x_hat = self.main_filter.x.copy()
            _x, _y, _z = x_hat[:, 0]

        self.main_filter.mu_x.append(_x)
        self.main_filter.mu_y.append(_y)
        self.main_filter.mu_z.append(_z)

    def _init_graph(self, data):
        if self.filter_type in FilterEnum.list_approximation_based_filters():
            self.fig, self.ax2 = plt.subplots(1, 1, figsize=(12, 9))
            xs, ys, _ = data.GPS_measurements_in_meter.T
            self.ax2.plot(xs, ys, lw=2, label='ground-truth trajectory', color='black')
        else:
            self.fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            self.ax1, self.ax2 = axs
            xs, ys, _ = data.GPS_measurements_in_meter.T
            self.ax1.plot(xs, ys, lw=2, label='ground-truth trajectory', color='black')
            self.ax2.plot(xs, ys, lw=2, label='ground-truth trajectory', color='black')
            self.ax1.set_xlabel('X [m]')
            self.ax1.set_ylabel('Y [m]')
            self.ax1.legend()
            self.ax1.grid()

    def _show_progress_graph(self):
        if self.filter_type in FilterEnum.list_sampling_based_filters():
            samples = self.main_filter.particles if self.filter_type is FilterEnum.PF else self.main_filter.samples
            
            if self.main_filter.N > 100:
                particle_indices = np.linspace(0, self.main_filter.N-1, 100, dtype=int).tolist()
                self.ax1.scatter(
                    samples[particle_indices, 0], 
                    samples[particle_indices, 1], 
                    alpha=.2, s=[10]
                )
            else:
                self.ax1.scatter(
                    samples[:, 0], 
                    samples[:, 1], 
                    alpha=.2, s=[10]
                )
        
    def _show_graph(self, data):
        xs, ys, _ = data.VO_measurements.T
        self.ax2.plot(xs, ys, lw=2, label='VO trajectory', color='b')
        self.ax2.plot(
            self.main_filter.mu_x, self.main_filter.mu_y, lw=2, 
            label='estimated trajectory', color='r')
        self.ax2.set_xlabel('X [m]')
        self.ax2.set_ylabel('Y [m]')
        self.ax2.legend()
        self.ax2.grid()
        
    def visualize_trajectory(self, data, title, dimension=2, interval=5):
        self.main_filter.visualize_trajectory(
            data=data, 
            dimension=dimension, 
            interval=interval, 
            title=title
        )
        
if __name__ == "__main__":
    import os
    from data_loader import CustomDataLoader
    from kalman_filters import (
        UnscentedKalmanFilter,
        ParticleFilter, ResamplingAlgorithms
    )

    # root_path = "../../../"
    # kitti_drive = 'example'
    # kitti_data_root_dir = os.path.join(root_path, "example_data")
    # noise_vector_dir = os.path.join(root_path, "exports/_noise_optimizations/noise_vectors")
    # dimension=2

    # Undo comment out this to change example data to entire sequence data
    root_path = "../../../"
    kitti_drive = '0033'
    kitti_data_root_dir = os.path.join(root_path, "data")
    noise_vector_dir = os.path.join(root_path, "exports/_noise_optimizations/noise_vectors")
    dimension=2

    data = CustomDataLoader(
        sequence_nr=kitti_drive, 
        kitti_root_dir=kitti_data_root_dir, 
        noise_vector_dir=noise_vector_dir,
        vo_dropout_ratio=0., 
        gps_dropout_ratio=0.,
        visualize_data=False,
        dimension=dimension
    )
    
    debug_mode=True
    
    x, P, H, q, r_vo, r_gps = data.get_initial_data(
        setup=SetupEnum.SETUP_3, 
        filter_type=FilterEnum.PF,
        noise_type=NoiseTypeEnum.CURRENT
    )

    pf = ParticleFilter(
        N=512, 
        x_dim=x.shape[0], 
        H=H.copy(), 
        q=q,
        r_vo=r_vo,
        r_gps=r_gps,
        setup=SetupEnum.SETUP_3,
        resampling_algorithm=ResamplingAlgorithms.STRATIFIED
    )
    pf.create_gaussian_particles(mean=x.copy(), var=P.copy())
    f = FilterWrapper(
        main_filter=pf,
        filter_type=FilterEnum.PF,
        omit_gps=False
    )
    f.run(
        data=data, 
        debug_mode=True, 
        show_graph=True,
        measurement_type=MeasurementDataEnum.DROPOUT
    )
    f.visualize_trajectory(
        data=data, 
        dimension=dimension, 
        interval=5, 
        title="PF Setup3 trajectories"
    )
    
    # x, P, H, q, r_vo, r_gps = data.get_initial_data(
    #     setup=SetupEnum.SETUP_3, 
    #     filter_type=FilterEnum.UKF,
    #     noise_type=NoiseTypeEnum.CURRENT
    # )
    # ukf = UnscentedKalmanFilter(
    #     x=x.copy(), 
    #     P=P.copy(), 
    #     H=H.copy(), 
    #     q=q,
    #     r_vo=r_vo,
    #     r_gps=r_gps,
    #     alpha=1.0, 
    #     beta=2.0, 
    #     kappa=0.0,
    #     setup=SetupEnum.SETUP_3
    # )
    # f = FilterWrapper(
    #     main_filter=ukf,
    #     filter_type=FilterEnum.UKF,
    # )
    # f.run(
    #     data=data, 
    #     debug_mode=True, 
    #     show_graph=True,
    #     measurement_type=MeasurementDataEnum.DROPOUT
    # )