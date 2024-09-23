import os
import sys
if __name__ == "__main__":
    sys.path.append('../../src')

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pykitti
from ahrs.filters import AngularRate

from utils import lla_to_enu, get_rigid_transformation
from configs import SetupEnum, SamplingEnum, FilterEnum, NoiseTypeEnum, MeasurementDataEnum

sequence_data_map = {
    '0016': { #04
        'kitti_date': '2011_09_30',
        'calib_velo_to_cam': '2011_09_30/calib_velo_to_cam.txt',
        'calib_imu_to_velo': '2011_09_30/calib_imu_to_velo.txt',
        'vo_path': 'trajectory_estimated_04.npy',
        'gt_path': 'trajectory_gt_04.npy',
    },
    '0033': { #09
        'kitti_date': '2011_09_30',
        'calib_velo_to_cam': '2011_09_30/calib_velo_to_cam.txt',
        'calib_imu_to_velo': '2011_09_30/calib_imu_to_velo.txt',
        'vo_path': 'trajectory_estimated_09.npy',
        'gt_path': 'trajectory_gt_09.npy',
    },
    'example': { #example data
        'kitti_date': '2011_09_30',
        'calib_velo_to_cam': '2011_09_30/calib_velo_to_cam.txt',
        'calib_imu_to_velo': '2011_09_30/calib_imu_to_velo.txt',
        'vo_path': 'trajectory_estimated_09_example.npy',
        'gt_path': 'trajectory_gt_09_example.npy',
    }
}

noise_configs = {
    '0016': {
        'GPS_measurement_noise_std': 1.0,# 0.01,
        'VO_noise_std': 1.0, #0.1,
        'IMU_acc_noise_std': 0.02,
        'IMU_angular_velocity_noise_std': 0.01,
        'velocity_noise_std': 0.3,
        'GPS_measurement_noise_std_uncertain': 2.0, #0.1,
        'VO_noise_std_uncertain': 2.0, #1.0,
        'quaternion_process_noise': [0.01, 0.002, 0.002, 0.05]
    },
    '0033': {
        'GPS_measurement_noise_std': 1.0,
        'VO_noise_std': 1.0,
        'IMU_acc_noise_std': 0.02,
        'IMU_angular_velocity_noise_std': 0.01,
        'velocity_noise_std': 0.3,
        'GPS_measurement_noise_std_uncertain': 2.0,
        'VO_noise_std_uncertain': 2.0,
        'quaternion_process_noise': [0.01, 0.002, 0.002, 0.05]
    },
    'example': {
        'GPS_measurement_noise_std': 1.0,
        'VO_noise_std': 1.0,
        'IMU_acc_noise_std': 0.02,
        'IMU_angular_velocity_noise_std': 0.01,
        'velocity_noise_std': 0.3,
        'GPS_measurement_noise_std_uncertain': 2.0,
        'VO_noise_std_uncertain': 2.0,
        'quaternion_process_noise': [0.01, 0.002, 0.002, 0.05]
    },
}

filter_noise_vector_size = {
    FilterEnum.EKF: {
        SetupEnum.SETUP_1: 10,
        SetupEnum.SETUP_2: 10,
        SetupEnum.SETUP_3: 6,
    },
    FilterEnum.UKF: {
        SetupEnum.SETUP_1: 14,
        SetupEnum.SETUP_2: 14,
        SetupEnum.SETUP_3: 7,
    },
    FilterEnum.PF: {
        SetupEnum.SETUP_1: 14,
        SetupEnum.SETUP_2: 14,
        SetupEnum.SETUP_3: 7,
    },
    FilterEnum.EnKF: {
        SetupEnum.SETUP_1: 14,
        SetupEnum.SETUP_2: 14,
        SetupEnum.SETUP_3: 7,
    },
    FilterEnum.CKF: {
        SetupEnum.SETUP_1: 14,
        SetupEnum.SETUP_2: 14,
        SetupEnum.SETUP_3: 7,
    },
}

class DataLoader:

    dataset = None
    T_velo_ref0 = None
    T_imu_velo = None
    T_from_imu_to_cam = None
    T_from_cam_to_imu = None

    GPS_measurement_noise_std = 1.0
    VO_noise_std = 1.0
    IMU_acc_noise_std = 0.02
    IMU_angular_velocity_noise_std = 0.01
    velocity_noise_std = 0.3
    angle_noise_std = 0.01

    GPS_measurement_noise_std_uncertain = None
    VO_noise_std_uncertain = None
    GPS_default_measurement_noise_std_uncertain = 2.0
    VO_default_noise_std_uncertain = 2.0

    quaternion_process_noise = [0.01, 0.002, 0.002, 0.05]

    current_sampling = None
    upsampling_factor = None
    downsampleing_ratio = None
    
    gt = None
    kitti = None

    GPS_from_raw_data_in_meter = None
    GPS_measurements_in_meter = None  # [longitude(deg), latitude(deg), altitude(meter)] x N from GPS
    VO_measurements = None # [longitude(deg), latitude(deg)] x N from Visual Odometry
    IMU_outputs = None # [acc_x, acc_y, acc_z, ang_vel_x, ang_vel_y, ang_vel_z] x N from IMU
    INS_angles = None # [roll(rad), pitch(rad), yaw(rad)] x N
    INS_velocities = None # [forward velocity, leftward velocity, upward velocity] x N from INS

    GPS_measurements_in_meter_original = None  # [longitude(deg), latitude(deg), altitude(meter)] x N from GPS
    VO_measurements_original = None # [longitude(deg), latitude(deg)] x N from Visual Odometry

    N = None
    ts = None
    GPS_mesurement_in_meter_with_noise = None
    VO_measurements_with_noise = None
    IMU_acc_with_noise = None
    IMU_angular_velocity_with_noise = None
    INS_velocities_with_noise = None
    INS_angle_with_noise = None
    IMU_quaternion = None

    N_original = None
    ts_original = None
    GPS_mesurement_in_meter_with_noise_original = None
    VO_measurements_with_noise_original = None
    IMU_acc_with_noise_original = None
    IMU_angular_velocity_with_noise_original = None
    INS_velocities_with_noise_original = None
    INS_angle_with_noise_original = None
    IMU_quaternion_original = None

    vo_dropout_ratio = None
    gps_dropout_ratio = None
    vo_indices = []
    gps_indices = []
    
    pose = []

    noise_vector_dir = None

    debug_mode = None
    def __init__(
        self, 
        sequence_nr='0033', 
        kitti_root_dir='../data',
        noise_vector_dir='../exports/_noise_optimizations/noise_vectors',
        vo_dropout_ratio=0., 
        gps_dropout_ratio=0., 
        sampling=SamplingEnum.DEFAULT_DATA,
        downsampling_ratio=0.1,
        upsampling_factor=10,
        visualize_data=True,
        dimension=2,
        ):
        
        self.dimension = dimension
        self.sequence_nr = sequence_nr
        self.kitti_root_dir = kitti_root_dir
        self.noise_vector_dir = noise_vector_dir
        
        # Setting paths
        self.config = sequence_data_map[sequence_nr].copy()
        
        kitti_root_dir = os.path.join(kitti_root_dir, 'KITTI')
        vo_root_dir = os.path.join(kitti_root_dir, "vo_estimates")
        
        self.config['calib_velo_to_cam'] = os.path.join(kitti_root_dir, self.config['calib_velo_to_cam'])
        self.config['calib_imu_to_velo'] = os.path.join(kitti_root_dir, self.config['calib_imu_to_velo'])
        self.config['vo_path'] = os.path.join(vo_root_dir, self.config['vo_path'])
        self.config['gt_path'] = os.path.join(vo_root_dir, self.config['gt_path'])

        noise = noise_configs[sequence_nr].copy()
        
        self.GPS_measurement_noise_std = noise['GPS_measurement_noise_std']
        self.VO_noise_std = noise['VO_noise_std']
        self.IMU_acc_noise_std = noise['IMU_acc_noise_std']
        self.IMU_angular_velocity_noise_std = noise['IMU_angular_velocity_noise_std']
        self.velocity_noise_std = noise['velocity_noise_std']

        self.GPS_measurement_noise_std_uncertain = noise['GPS_measurement_noise_std_uncertain']
        self.VO_noise_std_uncertain = noise['VO_noise_std_uncertain']
    
        self.quaternion_process_noise = noise['quaternion_process_noise']
        
        self.dataset = pykitti.raw(kitti_root_dir, self.config['kitti_date'], sequence_nr)
        
        self.T_velo_ref0 = get_rigid_transformation(self.config['calib_velo_to_cam'])
        self.T_imu_velo = get_rigid_transformation(self.config['calib_imu_to_velo'])
        self.T_from_imu_to_cam = self.T_imu_velo @ self.T_velo_ref0
        self.T_from_cam_to_imu = np.linalg.inv(self.T_from_imu_to_cam)
        
        self.load_data()
        self.debug_mode = visualize_data

        if self.debug_mode:
            self.report()
            self.show_trajectory()
            
        self.add_noise()

        self.set_upsampling_factor(upsampling_factor)
        self.set_downsampling_ratio(downsampling_ratio)

        self.set_data_sampling(sampling=sampling)

        self.change_dropout_ratio(
            vo_dropout_ratio=vo_dropout_ratio, 
            gps_dropout_ratio=gps_dropout_ratio
        )

    def load_data(self):
        gps = []
        imus = []
        ins_w = []
        ins_v = []
        vo = np.load(self.config['vo_path'])
        gt = np.load(self.config['gt_path'])
        self.N_original = gt.shape[1]

        pose = []
        for index, oxts_data in enumerate(self.dataset.oxts):
            if index < self.N_original:
                packet = oxts_data.packet
                gps.append([
                    packet.lon,
                    packet.lat,
                    packet.alt
                ])
                imus.append([
                    packet.ax,
                    packet.ay,
                    packet.az,
                    packet.wx,
                    packet.wy,
                    packet.wz
                ])
                ins_w.append([
                    packet.roll,
                    packet.pitch,
                    packet.yaw
                ])
                ins_v.append([
                    packet.vf,
                    packet.vl,
                    packet.vu
                ])
                pose.append(oxts_data.T_w_imu)
            
        
        self.GPS_measurements_in_meter_original = self.transform_gps_data_into_imu_coord(np.array(gt).T)
        self.VO_measurements_original = self.transform_vo_data_into_imu_coord(np.array(vo)[:self.N_original])
        
        gps = np.array(gps).T
        origin = gps[:, 0]
        self.GPS_from_raw_data_in_meter = lla_to_enu(gps, origin).T # convert gps coord to xyz global coord
        # self.VO_measurements_original = self.transform_gps_data_into_imu_coord(np.array(vo))
        self.kitti = gps.copy()
        self.gt = gt.copy()
        
        self.IMU_outputs = np.array(imus)
        self.INS_angles = np.array(ins_w)
        self.INS_velocities = np.array(ins_v)
        
        timestamps = np.array(self.dataset.timestamps[:self.N_original])
        elapsed = np.array(timestamps) - timestamps[0]
        self.ts_original = [t.total_seconds() for t in elapsed] # dt
        
        self.pose = np.array(pose)

    def transform_imu_into_gps(self, data):
        if self.debug_mode:
            print("Transform data in IMU space into GPS coordinate.")
        transformed_data = []
        for point in data:
            values = np.array([point[0], point[1], point[2], 1])
            transformed_values = self.T_from_imu_to_cam @ values
            transformed_data.append(
                [transformed_values[0], 
                transformed_values[1], 
                transformed_values[2]])
        
        return np.array(transformed_data)
    
    def transform_gps_data_into_imu_coord(self, gps_data):
        if self.debug_mode:
            print("Transform GPS data into imu coordinate.")
        GPS_measurements_in_meter = []
        for gt_est in gps_data:
            lla_values = np.array([gt_est[0], gt_est[1], gt_est[2], 1])
            transformed_lla_values = self.T_from_cam_to_imu @ lla_values
            GPS_measurements_in_meter.append(
                [transformed_lla_values[0], 
                transformed_lla_values[1], 
                transformed_lla_values[2]])
        
        return np.array(GPS_measurements_in_meter)

    def transform_vo_data_into_imu_coord(self, vo_data):
        if self.debug_mode:
            print("Transform VO data into imu coordinate.")
        vo_array = []
        for vo_est in vo_data:
            VO = np.array([vo_est[0], vo_est[1], vo_est[2], 1])
            transformed = self.T_from_cam_to_imu @ VO
            vo_array.append([transformed[0], transformed[1], transformed[2]])
        
        return np.array(vo_array)

    def report(self):
        print(f"Data size: {self.N_original}")
        print("Shape:")
        print(f'GPS: {self.GPS_measurements_in_meter_original.shape}')
        print(f'VO: {self.VO_measurements_original.shape}')
        print(f'IMU: {self.IMU_outputs.shape}')
        print(f'INS angle: {self.INS_angles.shape}')
        print(f'INS velocity: {self.INS_velocities.shape}')

    @staticmethod
    def get_quaternion_from_euler_angle(angle):
        roll, pitch, yaw = angle
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return np.array([w, x, y, z])
    
    @staticmethod
    def get_euler_angle_from_quaternion(q):
        w, x, y, z = q
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
        pitch = -np.pi/2 + 2*np.arctan2(np.sqrt(1 + 2*(w*y - x*z)), np.sqrt(1-2*(w*y - x*z)))
        yaw = np.arctan2(2*(w*z + x*y), 1-2*(y**2 + z**2))
        
        return np.array([roll, pitch, yaw])

    #### NOTE: CHANGE SAMPLING METHODS
    def _apply_downsampling(self, arr, ratio):
        indices = [int(i) for i in np.linspace(0, len(arr)-1, int(len(arr) * ratio))]
        return np.array(arr[indices])
    
    def _downsample(self, visualize=False):
        
        ratio = np.round(1 - self.downsampleing_ratio, 1)
        
        if ratio == 1.:
            print(f"Data is sampled and synchronized at original sampling frequency ({str(int(10 * ratio))}Hz).")
        else:
            print(f"Data is sampled and synchronized at {str(int(10 * ratio))}Hz.")

        downsampled_ts = self._apply_downsampling(np.array(self.ts_original), ratio).tolist()
        downsampled_gps = self._apply_downsampling(
            self.GPS_mesurement_in_meter_with_noise_original, ratio)
        downsampled_vo = self._apply_downsampling(
            self.VO_measurements_with_noise_original, 
            ratio)
        downsampled_ins_velocity = self._apply_downsampling(
            self.INS_velocities_with_noise_original, 
            ratio)
        downsampled_ins_angle = self._apply_downsampling(
            self.INS_angle_with_noise_original,
            ratio)
        downsampled_imu_angular_velocity = self._apply_downsampling(
            self.IMU_angular_velocity_with_noise_original, 
            ratio)
        downsampled_imu_acceleration = self._apply_downsampling(
            self.IMU_acc_with_noise_original, 
            ratio)

        gps_measurement_in_meter = self._apply_downsampling(self.GPS_measurements_in_meter_original, ratio)
        vo_measurement = self._apply_downsampling(self.VO_measurements_original, ratio)

        if visualize:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            xs, ys, _ = self.GPS_measurements_in_meter_original.T
            ax1.plot(xs, ys, lw=2, label='ground-truth trajectory', color='black')
            xs, ys, _ = self.VO_measurements_original.T
            ax1.plot(xs, ys, lw=2, label='VO trajectory', color='b')
            ax1.set_xlabel('X [m]')
            ax1.set_ylabel('Y [m]')
            ax1.title.set_text('Original trajectory')
            ax1.legend()
            ax1.grid()
            
            xs, ys, _ = downsampled_gps.T
            ax2.plot(xs, ys, lw=2, label='ground-truth trajectory', color='black')
            xs, ys, _ = downsampled_vo.T
            ax2.plot(xs, ys, lw=2, label='VO trajectory', color='b')
            ax2.set_xlabel('X [m]')
            ax2.set_ylabel('Y [m]')
            ax2.title.set_text('Trajectory after downsampling (noised data)')
            ax2.legend()
            ax2.grid()

        print("Shapes after downsampling")
        print(f"time: {np.array(downsampled_ts).shape}")
        print(f"GPS: {downsampled_gps.shape}")
        print(f"INS: {downsampled_ins_velocity.shape}")
        print(f"IMU (angular vel): {downsampled_imu_angular_velocity.shape}")
        print(f"IMU (linear acc): {downsampled_imu_acceleration.shape}")

        self.N = len(downsampled_ts)
        self.ts = downsampled_ts
        self.GPS_mesurement_in_meter_with_noise = downsampled_gps
        self.VO_measurements_with_noise = downsampled_vo
        self.INS_velocities_with_noise = downsampled_ins_velocity
        self.INS_angle_with_noise = downsampled_ins_angle
        self.IMU_angular_velocity_with_noise = downsampled_imu_angular_velocity
        self.IMU_acc_with_noise = downsampled_imu_acceleration
        
        # angular_rate = AngularRate(gyr=self.IMU_angular_velocity_with_noise)
        # self.IMU_quaternion = angular_rate.Q
        quaternions = [DataLoader.get_quaternion_from_euler_angle(angle) for angle in self.INS_angle_with_noise]
        self.IMU_quaternion = np.array(quaternions)
        
        self.GPS_measurements_in_meter = gps_measurement_in_meter
        self.VO_measurements = vo_measurement
        
    def _apply_upsampling(self, a, factor=10):
        b = np.empty((0, a.shape[1]))
        for i in range(0, len(a)-1):
            b = np.concatenate([b, np.linspace(a[i], a[i+1], factor+1)[:factor]], axis=0)
        
        b = np.concatenate([b, a[len(a)-1].reshape(1, -1)], axis=0)
        return b

    def _upsample(self, visualize=False):

        print(f"Data is sampled and synchronized at {str(int(10 * self.upsampling_factor))}Hz")
        upsampled_ts = self._apply_upsampling(np.array(self.ts_original).reshape(-1, 1), factor=self.upsampling_factor).reshape(-1).tolist()
        upsampled_gps = self._apply_upsampling(
            self.GPS_mesurement_in_meter_with_noise_original,
            factor=self.upsampling_factor)
        upsampled_vo = self._apply_upsampling(
            self.VO_measurements_with_noise_original,
            factor=self.upsampling_factor)
        upsampled_ins_velocity = self._apply_upsampling(
            self.INS_velocities_with_noise_original,
            factor=self.upsampling_factor)
        upsampled_ins_angule = self._apply_upsampling(
            self.INS_angle_with_noise_original,
            factor=self.upsampling_factor)
        upsampled_imu_angular_velocity = self._apply_upsampling(
            self.IMU_angular_velocity_with_noise_original, 
            factor=self.upsampling_factor)
        upsampled_imu_acceleration = self._apply_upsampling(
            self.IMU_acc_with_noise_original, 
            factor=self.upsampling_factor)
        
        gps_measurement_in_meter = self._apply_upsampling(
            self.GPS_measurements_in_meter_original, 
            factor=self.upsampling_factor)
        vo_measurement = self._apply_upsampling(
            self.VO_measurements_original, 
            factor=self.upsampling_factor)

        if visualize:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            xs, ys, _ = self.GPS_measurements_in_meter_original.T
            ax1.plot(xs, ys, lw=2, label='ground-truth trajectory', color='black')
            xs, ys, _ = self.VO_measurements_original.T
            ax1.plot(xs, ys, lw=2, label='VO trajectory', color='b')
            ax1.set_xlabel('X [m]')
            ax1.set_ylabel('Y [m]')
            ax1.title.set_text('Original trajectory')
            ax1.legend()
            ax1.grid()
            
            xs, ys, _ = upsampled_gps.T
            ax2.plot(xs, ys, lw=2, label='ground-truth trajectory', color='black')
            xs, ys, _ = upsampled_vo.T
            ax2.plot(xs, ys, lw=2, label='VO trajectory', color='b')
            ax2.set_xlabel('X [m]')
            ax2.set_ylabel('Y [m]')
            ax2.title.set_text('Trajectory after upsampling (noised data)')
            ax2.legend()
            ax2.grid()

        print("Shapes after upsampling")
        print(f"time: {np.array(upsampled_ts).shape}")
        print(f"GPS: {upsampled_gps.shape}")
        print(f"INS: {upsampled_ins_velocity.shape}")
        print(f"IMU (angular vel): {upsampled_imu_angular_velocity.shape}")
        print(f"IMU (linear acc): {upsampled_imu_acceleration.shape}")

        self.N = len(upsampled_ts)
        self.ts = upsampled_ts
        self.GPS_mesurement_in_meter_with_noise = upsampled_gps
        self.VO_measurements_with_noise = upsampled_vo
        self.INS_velocities_with_noise = upsampled_ins_velocity
        self.INS_angle_with_noise = upsampled_ins_angule
        self.IMU_angular_velocity_with_noise = upsampled_imu_angular_velocity
        self.IMU_acc_with_noise = upsampled_imu_acceleration

        # angular_rate = AngularRate(gyr=self.IMU_angular_velocity_with_noise)
        # self.IMU_quaternion = angular_rate.Q
        quaternions = [DataLoader.get_quaternion_from_euler_angle(angle) for angle in self.INS_angle_with_noise]
        self.IMU_quaternion = np.array(quaternions)

        self.GPS_measurements_in_meter = gps_measurement_in_meter
        self.VO_measurements = vo_measurement

    def set_upsampling_factor(self, factor):
        assert 1 <= factor and factor < 1000, "Factor must be larger than or equal to 1."
        self.upsampling_factor = factor
    
    def set_downsampling_ratio(self, ratio):
        assert 0 <= ratio and ratio < 1, "Ratio must be less than or equal to 1."
        self.downsampleing_ratio = ratio

    def _set_default_data(self):
        self.N = self.N_original
        self.ts = self.ts_original
        self.GPS_mesurement_in_meter_with_noise = self.GPS_mesurement_in_meter_with_noise_original
        self.VO_measurements_with_noise = self.VO_measurements_with_noise_original
        self.IMU_acc_with_noise = self.IMU_acc_with_noise_original
        self.IMU_angular_velocity_with_noise = self.IMU_angular_velocity_with_noise_original
        self.INS_velocities_with_noise = self.INS_velocities_with_noise_original
        self.IMU_quaternion = self.IMU_quaternion_original

        self.GPS_measurements_in_meter = self.GPS_measurements_in_meter_original
        self.VO_measurements = self.VO_measurements_original
    
    def set_data_sampling(self, sampling=SamplingEnum.DEFAULT_DATA):
        match sampling:
            case SamplingEnum.DEFAULT_DATA:
                self._set_default_data()
                print("Data sampling is set to normal mode.")
            case SamplingEnum.DOWNSAMPLED_DATA:
                self._downsample()

                print("Data sampling is set to downsampling mode.")
            case SamplingEnum.UPSAMPLED_DATA:
                self._upsample()

                print("Data sampling is set to upsampling mode.")
            
            case _: #SamplingEnum.DEFAULT_DATA
                self._set_default_data()
                print("Data sampling is set to normal mode.")

        self.current_sampling = sampling

    def show_trajectory(self):
        if self.dimension == 2:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            xs, ys, _ = self.GPS_measurements_in_meter_original.T
            ax.plot(xs, ys, label='ground-truth trajectory (GPS)')
            xs, ys, _ = self.VO_measurements_original.T
            ax.plot(xs, ys, label='Visual odometry')
            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
            ax.legend()
            ax.grid()
        else:
            fig = plt.figure()
            ax1 = fig.add_subplot(111, projection='3d')
            ax1.set_title("ground-truth trajectory (GPS)")
            
            xs, ys, zs = self.GPS_measurements_in_meter_original.T
            ax1.plot(xs, ys, zs, label='ground-truth trajectory (GPS)', color='black')
            
            xs, ys, zs = self.VO_measurements_original.T
            ax1.plot(xs, ys, zs, label='Visual odometry', color='red')

            ax1.set_xlabel('$X$', fontsize=14)
            ax1.set_ylabel('$Y$', fontsize=14)
            ax1.set_zlabel('$Z$', fontsize=14)

            fig.tight_layout()
            ax1.legend(loc='best', bbox_to_anchor=(1.1, 0., 0.2, 0.9))

    def add_noise(self):
        if self.debug_mode:
            print("Add noise to GPS data")
        
        _gps_noise = np.random.normal(0.0, self.GPS_measurement_noise_std, (self.N_original, 3))  # gen gaussian noise
        self.GPS_mesurement_in_meter_with_noise_original = self.GPS_measurements_in_meter_original.copy()
        self.GPS_mesurement_in_meter_with_noise_original[:, :3] += _gps_noise  # add the noise to ground-truth x and y positions

        if self.debug_mode:
            print("Adding noise to VO data")
        _vo_noise = np.random.normal(0.0, self.VO_noise_std, (self.N_original, 3))  # gen gaussian noise
        self.VO_measurements_with_noise_original = self.VO_measurements_original.copy()
        self.VO_measurements_with_noise_original[:, :3] += _vo_noise  # add the noise to ground-truth x and y positions

        if self.debug_mode:
            print("Adding noise to IMU sensor data")
            print("Adding noise to linear acceleration")
        IMU_acc_noise = np.random.normal(0.0, self.IMU_acc_noise_std,(self.N_original, 3))  # gaussian noise
        self.IMU_acc_with_noise_original = self.IMU_outputs[:, :3].copy()
        self.IMU_acc_with_noise_original += IMU_acc_noise
        
        if self.debug_mode:
            print("Adding noise to angular velocity")
        IMU_angular_velocity_noise = np.random.normal(0.0, self.IMU_angular_velocity_noise_std, (self.N_original,3))  # gen gaussian noise
        self.IMU_angular_velocity_with_noise_original = self.IMU_outputs[:, 3:].copy()
        self.IMU_angular_velocity_with_noise_original += IMU_angular_velocity_noise  # add the noise to angular velocity as measurement noise


        # angular_rate = AngularRate(gyr=self.IMU_angular_velocity_with_noise_original)
        # self.IMU_quaternion_original = angular_rate.Q
        
        angle_noise = np.random.normal(0.0, self.angle_noise_std, (self.N_original, 3))
        self.INS_angle_with_noise_original = self.INS_angles.copy()
        self.INS_angle_with_noise_original += angle_noise
        
        quaternions = [DataLoader.get_quaternion_from_euler_angle(angle) for angle in self.INS_angle_with_noise_original]
        self.IMU_quaternion_original = np.array(quaternions)

        if self.debug_mode:
            print("Adding noise to INS sensor data")
            print("Adding noise to linear velocity data")
            print("Adding noise to angle data")
        velocity_noise = np.random.normal(0.0, self.velocity_noise_std, (self.N_original, 3))
        self.INS_velocities_with_noise_original = self.INS_velocities.copy()
        self.INS_velocities_with_noise_original += velocity_noise
        
    # NOTE: Visualization methods
    def show_vo_with_noise(self):
        if self.dimension == 2:
            fig, ax = plt.subplots(1, 1, figsize=(12, 9))
            xs, ys, _ = self.VO_measurements.T
            ax.plot(xs, ys, lw=2, label='VO trajectory')
            
            xs, ys, _ = self.VO_measurements_with_noise.T
            ax.plot(xs, ys, lw=0, marker='.', markersize=5, alpha=0.4, label='noised VO trajectory')
            
            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
            ax.legend()
            ax.grid()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            xs, ys, zs = self.VO_measurements.T
            ax.plot(xs, ys, zs, lw=2, label='VO trajectory')

            xs, ys, zs = self.VO_measurements_with_noise.T
            ax.plot(xs, ys, lw=0, marker='.', markersize=5, alpha=0.4, label='noised VO trajectory')
            
            ax.set_xlabel('$X$', fontsize=14)
            ax.set_ylabel('$Y$', fontsize=14)
            ax.set_zlabel('$Z$', fontsize=14)

            fig.tight_layout()
            ax.legend(loc='best', bbox_to_anchor=(1.1, 0., 0.2, 0.9))
            plt.show()

    def show_linear_acceleration_with_noise(self):
        fig, ax = plt.subplots(3, 1, figsize=(6, 9))
        acc_y_labels = ['acceleration along x[m/s^2]', 'acceleration along y[m/s^2]', 'acceleration along z[m/s^2]']
        
        for idx in range(1, 4):  
            i = idx - 1
            ax[i].plot(self.ts, self.IMU_outputs[:, idx-1:idx], lw=1, label='ground-truth')
            ax[i].plot(self.ts, self.IMU_acc_with_noise[:, idx-1:idx], lw=0, marker='.', alpha=0.4, label='observed')
            ax[i].set_xlabel('time elapsed [sec]')
            ax[i].set_ylabel(acc_y_labels[i])
            ax[i].legend()
        fig.tight_layout()

    def show_angular_velocity_with_noise(self):
        fig, ax = plt.subplots(3, 1, figsize=(6, 9))
        angualr_vel_y_labels = ['angualr velocity about x[rad/s]', 'angualr velocity about y[rad/s]', 'angualr velocity about z[rad/s]']
        
        for idx in range(3):  
            i = idx + 4
            ax[idx].plot(self.ts, self.IMU_outputs[:, i-1:i], lw=1, label='ground-truth')
            ax[idx].plot(self.ts, self.IMU_angular_velocity_with_noise[:, idx:idx+1], lw=0, marker='.', alpha=0.4, label='observed')
            ax[idx].set_xlabel('time elapsed [sec]')
            ax[idx].set_ylabel(angualr_vel_y_labels[idx])
            ax[idx].legend()
        fig.tight_layout()
    
    def show_linear_velocity_with_noise(self):
        fig, ax = plt.subplots(3, 1, figsize=(6, 9))
        linear_velocity_y_labels = ['linear velocity along x[m/s]', 'linear velocity along y[m/s]', 'alinear velocity along z[m/s]']
        
        for idx in range(1, 4):  
            i = idx - 1
            ax[i].plot(self.ts, self.INS_velocities[:, idx-1:idx], lw=1, label='ground-truth')
            ax[i].plot(self.ts, self.INS_velocities_with_noise[:, idx-1:idx], lw=0, marker='.', alpha=0.4, label='observed')
            ax[i].set_xlabel('time elapsed [sec]')
            ax[i].set_ylabel(linear_velocity_y_labels[i])
            ax[i].legend()
        fig.tight_layout()

    def show_quaternion(self):
        fig, ax = plt.subplots(2, 2, figsize=(12, 9))
        ax = ax.ravel()
        ax[0].hist(self.IMU_quaternion[:, 0], bins=100)
        ax[1].hist(self.IMU_quaternion[:, 1], bins=100)
        ax[2].hist(self.IMU_quaternion[:, 2], bins=100)
        ax[3].hist(self.IMU_quaternion[:, 3], bins=100)
        plt.plot()
    
    # NOTE: Change attributes
    
    def change_dropout_ratio(self, vo_dropout_ratio, gps_dropout_ratio):
        self.vo_dropout_ratio = vo_dropout_ratio
        self.gps_dropout_ratio = gps_dropout_ratio
        indices = [i for i in range(self.N)]
        
        self.vo_indices = np.sort(
            random.sample(indices, int(len(indices)*(1 - self.vo_dropout_ratio)))).tolist()
        self.gps_indices = np.sort(
            random.sample(indices, int(len(indices)*(1 - self.gps_dropout_ratio)))).tolist()
        
    def change_sensor_uncertainty(self, vo_uncertainty_std=None, gps_uncertainty_std=None):
        '''
            Change the uncertainty of sensor values, which is used to compute measurement error covariance matrix.
        '''
        self.VO_noise_std_uncertain = vo_uncertainty_std if vo_uncertainty_std is not None else self.VO_default_noise_std_uncertain
        self.GPS_measurement_noise_std_uncertain = gps_uncertainty_std if gps_uncertainty_std is not None else self.GPS_default_measurement_noise_std_uncertain
    
    def get_control_input_by_index(self, index, setup):
        if setup is SetupEnum.SETUP_1 or setup is SetupEnum.SETUP_2:
            ax, ay, az = self.IMU_acc_with_noise[index]
            wx, wy, wz = self.IMU_angular_velocity_with_noise[index]
            u = np.array([
                ax,
                ay,
                az,
                wx,
                wy,
                wz
            ])
        else: #SetupEnum.SETUP_3
            if self.dimension == 2:
                u = np.array([
                    self.INS_velocities_with_noise[index, 0],
                    self.IMU_angular_velocity_with_noise[index, 2]
                ])
            else:
                u = np.array([
                    self.INS_velocities_with_noise[index, 0],
                    self.IMU_angular_velocity_with_noise[index, 0], #wx
                    self.IMU_angular_velocity_with_noise[index, 2] #wz
                ])
        return u
    
    def get_gps_measurement_by_index(
        self, 
        index, 
        setup=SetupEnum.SETUP_1,
        measurement_type=MeasurementDataEnum.ALL_DATA
        ):
        """returns GPS data

        Args:
            measurement_type (_type_, optional): _description_. Defaults to MeasurementDataEnum.ALL_DATA.

        Returns:
            gps_data: gps data numpy array
            gps_error_covariance_matrix: gps error covariance matrix when measurement_type=COVARIANCE
        """
        if setup is SetupEnum.SETUP_1:
            return (None, None)
        
        gps_data = self.GPS_mesurement_in_meter_with_noise[index, :self.dimension].reshape(-1, 1)
                        
        if measurement_type is MeasurementDataEnum.ALL_DATA:
            return (gps_data, None)
        elif measurement_type is MeasurementDataEnum.DROPOUT:
            if index not in self.gps_indices:
                return (None, None)
            return (gps_data, None)
        elif measurement_type is MeasurementDataEnum.COVARIANCE:
            error = self.GPS_measurement_noise_std if index in self.vo_indices else self.GPS_measurement_noise_std_uncertain
            q = np.repeat(error ** 2, self.dimension)
            return (gps_data, np.eye(self.dimension) * q)
        
        return (None, None)
        
    def get_vo_measurement_by_index(
        self, 
        index, 
        measurement_type=MeasurementDataEnum.ALL_DATA):
        """return VO data

        Args:
            measurement_type (_type_, optional): _description_. Defaults to MeasurementDataEnum.ALL_DATA.

        Returns:
            vo_data: vo data numpy array
            vo_error_cov_matrix: vo error covariance matrix when measurement_type=COVARIANCE
        """
        vo_data = self.VO_measurements_with_noise[index, :self.dimension].reshape(-1, 1)
        
        if measurement_type is MeasurementDataEnum.ALL_DATA:
            return (vo_data, None)
        elif measurement_type is MeasurementDataEnum.DROPOUT:
            if index not in self.vo_indices:
                return (None, None)
            return (vo_data, None)
        elif measurement_type is MeasurementDataEnum.COVARIANCE:
            error = self.VO_noise_std if index in self.vo_indices else self.VO_noise_std_uncertain
            q = np.repeat(error ** 2, self.dimension)
            return (vo_data, np.eye(self.dimension) * q)
        
        return (None, None)
    
    def get_trajectory_to_compare(self) -> np.ndarray:
        return self.GPS_measurements_in_meter.T
    
    def _get_noise_vectors(self, setup, filter_type, noise_type):
        noise_type_suffix = NoiseTypeEnum.get_suffix()[noise_type.value - 1]
        file_name = str(SetupEnum.get_names()[setup.value - 1]).lower() + noise_type_suffix + '.npy'
        filter_name = str(FilterEnum.get_names()[filter_type.value - 1]).lower()
        noise_path = os.path.join(self.noise_vector_dir, filter_name, file_name)

        if os.path.isfile(noise_path):
            noise_vector = np.load(noise_path, allow_pickle=True)
        else:
            noise_vector = np.ones(filter_noise_vector_size[filter_type][setup])
            noise_vector.dump(noise_path)
        
        return noise_vector[:-4], noise_vector[-4:-2], noise_vector[-2:]
    
    def get_initial_data(
        self, 
        setup=SetupEnum.SETUP_1, 
        filter_type=FilterEnum.EKF, 
        noise_type=NoiseTypeEnum.DEFAULT):

        # process noise, measurement noise for vo, measurement noise for gps
        q, r_vo, r_gps = self._get_noise_vectors(setup=setup, filter_type=filter_type, noise_type=noise_type)

        if setup is SetupEnum.SETUP_1 or setup is SetupEnum.SETUP_2:
            px, py, pz = self.GPS_measurements_in_meter[0, :]
            q1, q2, q3, q4 = self.IMU_quaternion[0]
            x = np.array([
                [px], #Px
                [py], #Py
                [pz], #Pz
                [0], #Vx
                [0], #Vy
                [0], #Vz
                [q1], #q1
                [q2], #q2
                [q3], #q3
                [q4]  #q4
            ]) # 10x1
            
            P = np.eye(x.shape[0]) * 0.1
            # transition matrix from predicted state vector to measurement space
            H = np.eye(x.shape[0])[:self.dimension, :]
            if self.dimension == 3:
                # transition matrix from predicted state vector to measurement space
                r_vo = np.array([1., 1., 10.])
                r_gps = np.array([0.1, 0.1, 0.1])

            return x.copy(), P.copy(), H.copy(), q.copy(), r_vo.copy(), r_gps.copy()
        else:
            px, py, pz = self.GPS_measurements_in_meter[0, :]
            if self.dimension == 2:
                x = np.array([
                    px,
                    py,
                    self.IMU_outputs[0, 5]
                ])
                x = x.reshape(-1, 1)
                
                # covariance for state vector x
                P = np.eye(x.shape[0]) * 0.1
                
                # transition matrix H
                H = np.eye(x.shape[0])[:2, :]
            else:
                x = np.array([
                    px,
                    py,
                    pz,
                    self.IMU_outputs[0, 3], #pitch
                    self.IMU_outputs[0, 5], #yaw
                ])
                x = x.reshape(-1, 1)
                # transition matrix from predicted state vector to measurement space
                # covariance for state vector x
                P = np.eye(x.shape[0]) * 0.1
                
                H = np.eye(x.shape[0])[:3, :]
                
                # TODO: optimize process error for setup3 in 3D
                # EKF: forward velocity, angular rate along x and z
                # Others: px, py, pz, phi, psi
                q = np.array([1., 0.1, 0.1]) if filter_type is FilterEnum.EKF else np.array([1., 1., 1., 0.1, 0.1])
                r_vo = np.array([1., 1., 100.])
                r_gps = np.array([0.1, 0.1, 0.1])

            return x.copy(), P.copy(), H.copy(), q.copy(), r_vo.copy(), r_gps.copy()

class CustomDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_vo_measurement_by_index_custom(self, index, measurement_type=MeasurementDataEnum.ALL_DATA):
        """return VO data
        Args:
            measurement_type (_type_, optional): _description_. Defaults to MeasurementDataEnum.ALL_DATA.

        Returns:
            vo_data: vo data numpy array
            vo_error_cov_matrix: vo error covariance matrix when measurement_type=COVARIANCE
        """
        vo_data = self.VO_measurements_with_noise[index].reshape(-1, 1)
        
        if measurement_type is MeasurementDataEnum.ALL_DATA:
            return (vo_data, None)
        elif measurement_type is MeasurementDataEnum.DROPOUT:
            if index not in self.vo_indices:
                return (None, None)
            return (vo_data, None)
        elif measurement_type is MeasurementDataEnum.COVARIANCE:
            error = self.VO_noise_std if index in self.vo_indices else self.VO_noise_std_uncertain
            q = np.repeat(error ** 2, self.dimension)
            return (vo_data, np.eye(self.dimension) * q)
        
        return (None, None)

    def get_prev_vo_measurement_from_current_index(self, index, measurement_type=MeasurementDataEnum.ALL_DATA):
        """ return previous VO data considering dropout 
        """
        index = index - 1
        if measurement_type is not MeasurementDataEnum.DROPOUT:
            return self.VO_measurements_with_noise[index].reshape(-1, 1)
            
        while index not in self.vo_indices and index >= 0:
            index -= 1
            
        return self.VO_measurements_with_noise[index].reshape(-1, 1)
    
if __name__ == "__main__":
    kitti_drive = 'example'
    kitti_example_data_root_dir = '../../example_data'
    noise_vector_dir = '../../exports/_noise_optimizations/noise_vectors'
    
    # kitti_drive = '0033'
    # kitti_example_data_root_dir = '../../data'
    # noise_vector_dir = '../../exports/_noise_optimizations/noise_vectors'
    
    data = DataLoader(
        sequence_nr=kitti_drive, 
        kitti_root_dir=kitti_example_data_root_dir, 
        noise_vector_dir=noise_vector_dir,
        vo_dropout_ratio=0.2, 
        gps_dropout_ratio=0.2,
        upsampling_factor=10,
        downsampling_ratio=0.8,
        visualize_data=True,
        dimension=2)
    
    data.set_data_sampling(sampling=SamplingEnum.UPSAMPLED_DATA)
    data.set_data_sampling(sampling=SamplingEnum.DOWNSAMPLED_DATA)
    x, P, H, q, r_vo, r_gps = data.get_initial_data(setup=SetupEnum.SETUP_1, filter_type=FilterEnum.EKF)
    
    data.set_data_sampling(sampling=SamplingEnum.DEFAULT_DATA)
    x, P, H, q, r_vo, r_gps = data.get_initial_data(setup=SetupEnum.SETUP_1, filter_type=FilterEnum.EKF)
    
    x, P, H, q, r_vo, r_gps = data.get_initial_data(setup=SetupEnum.SETUP_1, filter_type=FilterEnum.EKF)