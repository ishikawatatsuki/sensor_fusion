import os
import sys
if __name__ == "__main__":
    sys.path.append('../../src')
import yaml
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext
from utils import lla_to_ned
from configs.configs import IMU_Type, SetupEnum, MeasurementDataEnum

logger = logging.getLogger(__name__)
class UAV_DataLoader:
    
    time_update = ['voxl_imu0', 'voxl_imu1', 'px4_imu0', 'px4_imu1', 'px4_imu0_bias', 'px4_imu1_bias', 'actuator_motors', 'actuator_outputs']
    measurement_update = ['voxl_vo', 'px4_vo', 'px4_vehicle_odom', 'px4_gps', 'uwb_position']
    
    gps_pose_columns = ['lon', 'lat', 'alt']
    gps_ned_pose_columns = ['north', 'east', 'down']
    gps_pos_var_columns = ['eph', 'eph', 'epv'] # https://docs.px4.io/main/en/msg_docs/SensorGps.html
    vo_pos_columns = ['position[0]', 'position[1]', 'position[2]']
    vo_pos_var_columns = ['position_variance[0]', 'position_variance[1]', 'position_variance[2]']
    
    voxl_vo_T_columns = ['voxl_vo_T_imu_wrt_vio_x(m)', 'voxl_vo_T_imu_wrt_vio_y(m)', 'voxl_vo_T_imu_wrt_vio_z(m)']
    px4_synched_gps_columns = ['px4_gps_north', 'px4_gps_east', 'px4_gps_down']
    
    imu_config_map = {
        'voxl_imu0': 'icm_42688',
        'voxl_imu1': 'icm_20948',
        'px4_imu0': 'icm_20602',
        'px4_imu1': 'icm_42688'
    }
    
    SENSOR_SLIP_THRESHOLD = 2 # 2 seconds
    
    vo_indices = []
    gps_indices = []

    vo_indices = None
    gps_indices = None


    def __init__(
        self,
        root_path,
        sequence_nr="log0001",
        version="v1",
        config_filename="config.yaml",
        imu_config_filepath="./imu_config.yaml",
        is_debugging=False,
        ):
        
        self.is_debugging = is_debugging
        
        uav_root_path = os.path.join(root_path, "UAV")
        uav_sequence_root_path = os.path.join(root_path, "UAV", sequence_nr)
        
        config_path = os.path.join(root_path, "UAV", config_filename)
        
        config = None
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            f.close()
        
        assert config != None, "Please specify proper config file name"
        
        imu_config = None
        with open(imu_config_filepath, "r") as f:
            imu_config = yaml.safe_load(f)
            f.close()
        assert imu_config != None, "Failed to load IMU configuration"
        
        self.imu_config = imu_config
        
        sensor_data = config["sensor_data"][sequence_nr]
        
        self.data_type = sensor_data["type"]
        
        px4_root_path = os.path.join(uav_sequence_root_path, "px4")
        
        # PX4 data
        px4 = sensor_data["px4"]
        px4_imu0_path = os.path.join(px4_root_path, "imu_combined", px4["imu0"])
        px4_imu1_path = os.path.join(px4_root_path, "imu_combined", px4["imu1"])
        px4_gps_path = os.path.join(px4_root_path,  px4["gps"])
        px4_vo_path = os.path.join(px4_root_path,  px4["vo"])
        px4_vehicle_odom_path = os.path.join(px4_root_path,  px4["vehicle_odometry"])
        imu0_bias_path = os.path.join(px4_root_path,  px4["imu0_bias"])
        imu1_bias_path = os.path.join(px4_root_path,  px4["imu1_bias"])
        actuator_motors_path = os.path.join(px4_root_path, px4["actuator_motors"])
        actuator_outputs_path = os.path.join(px4_root_path, px4["actuator_outputs"])

        # VOXL data
        voxl_root_path = os.path.join(uav_sequence_root_path, "run/mpa")
        voxl_imu0_path = os.path.join(voxl_root_path, "imu0/data.csv")
        voxl_imu1_path = os.path.join(voxl_root_path, "imu1/data.csv")
        voxl_qvio_path = os.path.join(voxl_root_path, "qvio/data.csv")
        voxl_stereo_path = os.path.join(voxl_root_path, "stereo/data.csv")
        
        self.px4_imu0_df = self.load_px4_data(px4_imu0_path, "PX4 IMU0")
        self.px4_imu1_df = self.load_px4_data(px4_imu1_path, "PX4 IMU1")
        self.px4_gps_df = self._init_gps_data(px4_gps_path=px4_gps_path)
        self.px4_vo_df = self.load_px4_data(px4_vo_path, "PX4 VO")
        self.px4_vehicle_odom_df = self.load_px4_data(px4_vehicle_odom_path, "PX4 Vehicle Odom")
        self.px4_imu0_bias_df = self.load_px4_data(imu0_bias_path, "PX4 IMU0 bias")
        self.px4_imu1_bias_df = self.load_px4_data(imu1_bias_path, "PX4 IMU1 bias")
        
        self.px4_actuator_motor_df = self.load_px4_data(actuator_motors_path)
        self.px4_actuator_output_df = self.load_px4_data(actuator_outputs_path)
        if self.data_type == "outdoor":
            self.px4_actuator_motor_df = self.px4_actuator_motor_df[self.px4_actuator_motor_df.columns[:6]]
            self.px4_actuator_output_df = self.px4_actuator_output_df[self.px4_actuator_output_df.columns[:6]]
        

        self.voxl_imu0_df = self.load_voxl_data(voxl_imu0_path, "VOXL IMU0")
        self.voxl_imu1_df = self.load_voxl_data(voxl_imu1_path, "VOXL IMU1")
        self.voxl_qvio_df = self.load_voxl_data(voxl_qvio_path, "VOXL QVIO")
        self.voxl_stereo_df = self.load_voxl_data(voxl_stereo_path, "VOXL stereo")

        
        # custom data
        custom_data = sensor_data["custom"]
        
        self.custom_vo_df = None
        if custom_data["vo"] is not None:
            self.custom_vo_df = pd.read_csv(os.path.join(uav_root_path, custom_data["vo"]))
        else:
            logger.warning("VO estimate data not found.")
            
        self.uwb_df = None
        if custom_data["uwb"] is not None and self.data_type == "indoor":
            self.uwb_df = pd.read_csv(os.path.join(uav_root_path, custom_data["uwb"]))
        else:
            logger.warning("UWB position data not found.")

        
        aggregated_file_path = os.path.join(uav_root_path, version)

        self.ref_df = None
        if custom_data["timestamp"] is not None:
            self.ref_df = pd.read_csv(os.path.join(aggregated_file_path, custom_data["timestamp"]))
        else:
            logger.warning('Timestamp combined data not found.')
        
        self.synced_df = None
        if custom_data["synched"] is not None:
            self.synced_df = pd.read_csv(os.path.join(aggregated_file_path, custom_data["synched"]))
        else:
            logger.warning('Synched data not found.')


        self.N = self.synced_df.shape[0]
        self.ts = np.cumsum(np.concatenate([np.array([0.]), np.diff(self.synced_df["timestamp"].values / 1e6)]))
        self.dimension = 3
        
    def load_voxl_data(self, path, label=""):
        df = pd.read_csv(path)
        df = df[df.columns[1:]]
        if self.is_debugging:
            freq = df.shape[0] / ((df["timestamp(ns)"].iloc[-1] - df["timestamp(ns)"].iloc[0]) / 1e9)
            print(f"[{label}] Sampling rate: {round(freq, 2)}Hz")
        return df

    def load_px4_data(self, csv_path, label=""):
        if self.data_type == "outdoor":
            df = pd.read_csv(csv_path)
            if df.columns[0] == "1":
                df = df[df.columns[1:]]
            if self.is_debugging:
                freq = df.shape[0] / ((df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]) / 1e9)
                print(f"[{label}] Sampling rate: {round(freq, 2)}Hz")
            return df
        
        logger.info(f"Csv file does not exists: {csv_path}")
        return None
    
    def _init_gps_data(self, px4_gps_path):
        if self.data_type == "indoor":
            logger.info(f"Csv file does not exists: {px4_gps_path}")
            return None
            
        # Load csv data
        df = pd.read_csv(px4_gps_path)
        # Convert integer lla into float lla
        df = UAV_DataLoader.int_lla_to_float_lla(df)
        # Compute North-East-Down coordinate
        if self.is_debugging:
            freq = df.shape[0] / ((df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]) / 1e9)
            print(f"[PX4 GPS] Sampling rate: {round(freq, 2)}Hz")
            
        return UAV_DataLoader.get_ned_coord(df)
    
    @staticmethod
    def init_gps_data(px4_gps_path):
        df = pd.read_csv(px4_gps_path)
        df = UAV_DataLoader.int_lla_to_float_lla(df)
        return UAV_DataLoader.get_ned_coord(df)

    @staticmethod
    def int2float_lla(x):
        return float(Decimal(x / Decimal(10**(len(str(x)) - 2))))

    @staticmethod
    def int_lla_to_float_lla(df):
        getcontext().prec = 10
        df['lat'] = df['lat'].apply(lambda x: UAV_DataLoader.int2float_lla(x))
        df['lon'] = df['lon'].apply(lambda x: UAV_DataLoader.int2float_lla(x))
        df['alt'] = df['alt'].apply(lambda x: UAV_DataLoader.int2float_lla(x))
        return df
    
    @staticmethod
    def get_ned_coord(df):    
        origin = df[['lon', 'lat', 'alt']].iloc[0].values
        ned_pose = lla_to_ned(df[['lon', 'lat', 'alt']].values.T, origin).T
        df = pd.concat([
            df,
            pd.DataFrame(ned_pose, columns=['north', 'east', 'down'])
        ], axis=1)
        
        return df
    
    def visualize_imu_data(self, title="IMU data"):
        fig, (ax1, ax2) = plt.subplots(2, 3, figsize=(16, 8))
        acc_titles = ["AX(m/s2)", "AY(m/s2)", "AZ(m/s2)"]
        gyr_titles = ["GX(rad/s)", "GY(rad/s)", "GZ(rad/s)"]
        fig.suptitle(title, fontsize=24)
        
        for i in range(0, 3):
            ax1[i].title.set_text(acc_titles[i])
            ax2[i].title.set_text(gyr_titles[i])
            acc_columns = self.voxl_imu0_df.columns[1:4]
            gyr_columns = self.voxl_imu0_df.columns[4:7]
            
            voxl_imu0_val = (
                    self.voxl_imu0_df[acc_columns].values[:, i] - self.imu_config[IMU_Type.IMU0.value]['accelerometer']['offset'][i]
                ) * self.imu_config[IMU_Type.IMU0.value]['accelerometer']['scale'][i]
            ax1[i].plot(voxl_imu0_val, color="red", lw=0.5, label=f"VOXL IMU0 ({IMU_Type.IMU0.value})")
            
            voxl_imu1_val = (
                    self.voxl_imu1_df[acc_columns].values[:, i] - self.imu_config[IMU_Type.IMU1.value]['accelerometer']['offset'][i]
                ) * self.imu_config[IMU_Type.IMU1.value]['accelerometer']['scale'][i]
            ax1[i].plot(voxl_imu1_val, color="blue", lw=0.5, label=f"VOXL IMU1 ({IMU_Type.IMU1.value})")
            
            if self.px4_imu0_df is not None and self.px4_imu1_df is not None:
                px4_imu0_val = (
                        self.px4_imu0_df[acc_columns].values[:, i] - self.imu_config[IMU_Type.IMU2.value]['accelerometer']['offset'][i]
                    ) * self.imu_config[IMU_Type.IMU0.value]['accelerometer']['scale'][i]
                ax1[i].plot(px4_imu0_val, color="red", lw=0.5, label=f"PX4 IMU0 ({IMU_Type.IMU2.value})")
                
                px4_imu1_val = (
                        self.px4_imu1_df[acc_columns].values[:, i] - self.imu_config[IMU_Type.IMU3.value]['accelerometer']['offset'][i]
                    ) * self.imu_config[IMU_Type.IMU1.value]['accelerometer']['scale'][i]
                ax1[i].plot(px4_imu1_val, color="blue", lw=0.5, label=f"PX4 IMU2 ({IMU_Type.IMU3.value})")
        
            
            voxl_imu0_val = (
                    self.voxl_imu0_df[gyr_columns].values[:, i] - self.imu_config[IMU_Type.IMU0.value]['gyroscope']['offset'][i]
                )
            ax2[i].plot(voxl_imu0_val, color="red", lw=0.5, label=f"VOXL IMU0 ({IMU_Type.IMU0.value})")

            voxl_imu1_val = (
                    self.voxl_imu1_df[gyr_columns].values[:, i] - self.imu_config[IMU_Type.IMU1.value]['gyroscope']['offset'][i]
                )
            ax2[i].plot(voxl_imu1_val, color="blue", lw=0.5, label=f"VOXL IMU1 ({IMU_Type.IMU1.value})")
        
            if self.px4_imu0_df is not None and self.px4_imu1_df is not None:
                px4_imu0_val = (
                        self.px4_imu0_df[gyr_columns].values[:, i] - self.imu_config[IMU_Type.IMU2.value]['gyroscope']['offset'][i]
                    )
                ax2[i].plot(px4_imu0_val, color="red", lw=0.5, label=f"PX4 IMU0 ({IMU_Type.IMU2.value})")

                px4_imu1_val = (
                        self.px4_imu1_df[gyr_columns].values[:, i] - self.imu_config[IMU_Type.IMU3.value]['gyroscope']['offset'][i]
                    )
                ax2[i].plot(px4_imu1_val, color="blue", lw=0.5, label=f"PX4 IMU1 ({IMU_Type.IMU3.value})")
                
            ax1[i].grid()
            ax1[i].legend()
            ax2[i].grid()
            ax2[i].legend()

    
    def visualize_qvio_trajectory(
        self, 
        dimension=2, 
        detail=False, 
        x_lim=None, 
        y_lim=None, 
        z_lim=None
        ):
        try:
            if dimension == 2:
                _, ax1 = plt.subplots(1, 1, figsize=(8, 6))
                xs, ys = self.voxl_qvio_df[['T_imu_wrt_vio_x(m)', 'T_imu_wrt_vio_y(m)']].values.T
                ax1.plot(xs, ys, label='QVIO estimated trajectory')
                ax1.set_title("QVIO estimated trajectory in 2D space")
                ax1.set_xlabel('X [m]')
                ax1.set_ylabel('Y [m]')
                ax1.legend()
                ax1.grid()
            else:
                fig = plt.figure()
                ax1 = fig.add_subplot(111, projection='3d')
                ax1.set_title("QVIO estimated trajectory in 3D space")

                xs, ys, zs = self.voxl_qvio_df[['T_imu_wrt_vio_x(m)', 'T_imu_wrt_vio_y(m)', 'T_imu_wrt_vio_z(m)']].values.T
                ax1.plot(xs, ys, zs, label='QVIO estimated trajectory', color='black')

                ax1.set_xlabel('X', fontsize=14)
                ax1.set_ylabel('Y', fontsize=14)
                ax1.set_zlabel('Z', fontsize=14)
                
                fig.tight_layout()
                ax1.legend(loc='best', bbox_to_anchor=(1.1, 0., 0.2, 0.9))
                
            if x_lim:
                ax1.set_xlim(x_lim)
            
            if y_lim:
                ax1.set_ylim(y_lim)
            
            if z_lim:
                ax1.set_zlim(z_lim)
            
            if detail:
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))

                xs, ys = self.voxl_qvio_df[['T_imu_wrt_vio_x(m)', 'T_imu_wrt_vio_y(m)']].values.T
                axs[0].plot(xs, -ys, label='QVIO estimated trajectory')
                axs[0].set_title("QVIO estimated trajectory X-Y")
                axs[0].set_xlabel('X [m]')
                axs[0].set_ylabel('Y [m]')
                axs[0].legend()
                axs[0].grid()

                xs, ys = self.voxl_qvio_df[['T_imu_wrt_vio_y(m)', 'T_imu_wrt_vio_z(m)']].values.T
                axs[1].plot(-xs, ys, label='QVIO estimated trajectory')
                axs[1].set_title("QVIO estimated trajectory Y-Z")
                axs[1].set_xlabel('Y [m]')
                axs[1].set_ylabel('Z [m]')
                axs[1].legend()
                axs[1].grid()

                xs, ys = self.voxl_qvio_df[['T_imu_wrt_vio_z(m)', 'T_imu_wrt_vio_x(m)']].values.T
                axs[2].plot(xs, ys, label='QVIO estimated trajectory')
                axs[2].set_title("QVIO estimated trajectory Z-X")

                axs[2].set_xlabel('Z [m]')
                axs[2].set_ylabel('X [m]')
                axs[2].legend()
                axs[2].grid()

                fig.tight_layout()
        except Exception as e:
            logger.info(f"An exception occurs: {e}")
            
    
    def get_config(self, timestamp):
        """

        """
        return {
            'voxl_imu0': {
                'index': 0,
                'last_timestamp': timestamp,
                'columns': ['AX(m/s2)', 'AY(m/s2)', 'AZ(m/s2)', 'GX(rad/s)', 'GY(rad/s)', 'GZ(rad/s)'],
                'synced_columns': ['voxl_imu0_AX(m/s2)', 'voxl_imu0_AY(m/s2)', 'voxl_imu0_AZ(m/s2)', 'voxl_imu0_GX(rad/s)', 'voxl_imu0_GY(rad/s)', 'voxl_imu0_GZ(rad/s)'],
                'df': self.voxl_imu0_df,
            },
            'voxl_imu1': {
                'index': 0,
                'last_timestamp': timestamp,
                'columns': ['AX(m/s2)', 'AY(m/s2)', 'AZ(m/s2)', 'GX(rad/s)', 'GY(rad/s)', 'GZ(rad/s)'],
                'synced_columns': ['voxl_imu1_AX(m/s2)', 'voxl_imu1_AY(m/s2)', 'voxl_imu1_AZ(m/s2)', 'voxl_imu1_GX(rad/s)', 'voxl_imu1_GY(rad/s)', 'voxl_imu1_GZ(rad/s)'],
                'df': self.voxl_imu1_df,
            },
            'voxl_vo': {
                'index': 0,
                'last_timestamp': timestamp,
                'columns': self.voxl_qvio_df.columns.tolist(),
                'synced_columns': ['voxl_vo_' + col for col in self.voxl_qvio_df.columns.tolist()],
                'df': self.voxl_qvio_df,
            },
            'voxl_stereo': {
                'index': 0,
                'last_timestamp': timestamp,
                'columns': self.voxl_stereo_df.columns.to_list(),
                'synced_columns':  ['voxl_stereo_' + col for col in self.voxl_stereo_df.columns.tolist()],
                'df': self.voxl_stereo_df,
            },
            'px4_imu0': {
                'index': 0,
                'last_timestamp': timestamp,
                'columns': ['AX(m/s2)', 'AY(m/s2)', 'AZ(m/s2)', 'GX(rad/s)', 'GY(rad/s)', 'GZ(rad/s)'],
                'synced_columns': ['px4_imu0_AX(m/s2)', 'px4_imu0_AY(m/s2)', 'px4_imu0_AZ(m/s2)', 'px4_imu0_GX(rad/s)', 'px4_imu0_GY(rad/s)', 'px4_imu0_GZ(rad/s)'],
                'df': self.px4_imu0_df,
            },
            'px4_imu1': {
                'index': 0,
                'last_timestamp': timestamp,
                'columns': ['AX(m/s2)', 'AY(m/s2)', 'AZ(m/s2)', 'GX(rad/s)', 'GY(rad/s)', 'GZ(rad/s)'],
                'synced_columns': ['px4_imu1_AX(m/s2)', 'px4_imu1_AY(m/s2)', 'px4_imu1_AZ(m/s2)', 'px4_imu1_GX(rad/s)', 'px4_imu1_GY(rad/s)', 'px4_imu1_GZ(rad/s)'],
                'df': self.px4_imu1_df,
            },
            'px4_gps': {
                'index': 0,
                'last_timestamp': timestamp,
                'columns': ['lat', 'lon', 'alt', 'north', 'east', 'down', 'vel_m_s', 'vel_n_m_s', 'vel_e_m_s', 'vel_d_m_s', 'eph', 'epv'],
                'synced_columns': ['px4_gps_lat', 'px4_gps_lon', 'px4_gps_alt', 'px4_gps_north', 'px4_gps_east', 'px4_gps_down', 'px4_gps_vel_m_s', 'px4_gps_vel_n_m_s', 'px4_gps_vel_e_m_s', 'px4_gps_vel_d_m_s', 'px4_gps_eph', 'px4_gps_epv'],
                'df': self.px4_gps_df,
            },
            'px4_vo': {
                'index': 0,
                'last_timestamp': timestamp,
                'columns': ['position[0]', 'position[1]', 'position[2]', 
                            'q[0]', 'q[1]', 'q[2]', 'q[3]', 
                            'velocity[0]','velocity[1]', 'velocity[2]', 
                            'angular_velocity[0]', 'angular_velocity[1]', 'angular_velocity[2]', 
                            'position_variance[0]', 'position_variance[1]', 'position_variance[2]',
                            'orientation_variance[0]', 'orientation_variance[1]', 'orientation_variance[2]', 
                            'velocity_variance[0]', 'velocity_variance[1]', 'velocity_variance[2]'],
                'synced_columns': ['px4_vo_position[0]', 'px4_vo_position[1]', 'px4_vo_position[2]', 
                            'px4_vo_q[0]', 'px4_vo_q[1]', 'px4_vo_q[2]', 'px4_vo_q[3]', 
                            'px4_vo_velocity[0]','px4_vo_velocity[1]', 'px4_vo_velocity[2]', 
                            'px4_vo_angular_velocity[0]', 'px4_vo_angular_velocity[1]', 'px4_vo_angular_velocity[2]', 
                            'px4_vo_position_variance[0]', 'px4_vo_position_variance[1]', 'px4_vo_position_variance[2]',
                            'px4_vo_orientation_variance[0]', 'px4_vo_orientation_variance[1]', 'px4_vo_orientation_variance[2]', 
                            'px4_vo_velocity_variance[0]', 'px4_vo_velocity_variance[1]', 'px4_vo_velocity_variance[2]'],
                'df': self.px4_vo_df,
            },
            'px4_vehicle_odom': {
                'index': 0,
                'last_timestamp': timestamp,
                'columns': ['position[0]', 'position[1]', 'position[2]', 
                            'q[0]', 'q[1]', 'q[2]', 'q[3]', 
                            'velocity[0]','velocity[1]', 'velocity[2]', 
                            'angular_velocity[0]', 'angular_velocity[1]', 'angular_velocity[2]', 
                            'position_variance[0]', 'position_variance[1]', 'position_variance[2]',
                            'orientation_variance[0]', 'orientation_variance[1]', 'orientation_variance[2]', 
                            'velocity_variance[0]', 'velocity_variance[1]', 'velocity_variance[2]'],
                'synced_columns': ['px4_vehicle_odom_position[0]', 'px4_vehicle_odom_position[1]', 'px4_vehicle_odom_position[2]', 
                            'px4_vehicle_odom_q[0]', 'px4_vehicle_odom_q[1]', 'px4_vehicle_odom_q[2]', 'px4_vehicle_odom_q[3]', 
                            'px4_vehicle_odom_velocity[0]','px4_vehicle_odom_velocity[1]', 'px4_vehicle_odom_velocity[2]', 
                            'px4_vehicle_odom_angular_velocity[0]', 'px4_vehicle_odom_angular_velocity[1]', 'px4_vehicle_odom_angular_velocity[2]', 
                            'px4_vehicle_odom_position_variance[0]', 'px4_vehicle_odom_position_variance[1]', 'px4_vehicle_odom_position_variance[2]',
                            'px4_vehicle_odom_orientation_variance[0]', 'px4_vehicle_odom_orientation_variance[1]', 'px4_vehicle_odom_orientation_variance[2]', 
                            'px4_vehicle_odom_velocity_variance[0]', 'px4_vehicle_odom_velocity_variance[1]', 'px4_vehicle_odom_velocity_variance[2]'],
                'df': self.px4_vehicle_odom_df,
            },
            'px4_imu0_bias': {
                'index': 0,
                'last_timestamp': timestamp,
                'columns': ['gyro_bias[0]', 'gyro_bias[1]', 'gyro_bias[2]', 'accel_bias[0]', 'accel_bias[1]', 'accel_bias[2]'],
                'synced_columns': ['px4_imu0_bias_gyro_bias[0]', 'px4_imu0_bias_gyro_bias[1]', 'px4_imu0_bias_gyro_bias[2]', 'px4_imu0_bias_accel_bias[0]', 'px4_imu0_bias_accel_bias[1]', 'px4_imu0_bias_accel_bias[2]'],
                'df': self.px4_imu0_bias_df,
            },
            'px4_imu1_bias': {
                'index': 0,
                'last_timestamp': timestamp,
                'columns': ['gyro_bias[0]', 'gyro_bias[1]', 'gyro_bias[2]', 'accel_bias[0]', 'accel_bias[1]', 'accel_bias[2]'],
                'synced_columns': ['px4_imu1_bias_gyro_bias[0]', 'px4_imu1_bias_gyro_bias[1]', 'px4_imu1_bias_gyro_bias[2]', 'px4_imu1_bias_accel_bias[0]', 'px4_imu1_bias_accel_bias[1]', 'px4_imu1_bias_accel_bias[2]'],
                'df': self.px4_imu1_bias_df,
            },
            'px4_actuator_motors': {
                'index': 0,
                'last_timestamp': timestamp,
                'columns': ['control[0]', 'control[1]', 'control[2]', 'control[3]'],
                'synced_columns': ['control[0]', 'control[1]', 'control[2]', 'control[3]'],
                'df': self.px4_actuator_motor_df,
            },
            'px4_actuator_outputs': {
                'index': 0,
                'last_timestamp': timestamp,
                'columns': ['output[0]', 'output[1]', 'output[2]', 'output[3]'],
                'synced_columns':['output[0]', 'output[1]', 'output[2]', 'output[3]'],
                'df': self.px4_actuator_output_df,
            },
        }
        
    def set_dropout_rate(self, vo_dropout_ratio, gps_dropout_ratio):
        self.vo_dropout_ratio = vo_dropout_ratio
        self.gps_dropout_ratio = gps_dropout_ratio
        indices = [i for i in range(self.N)]
        
        self.vo_indices = np.sort(
            random.sample(indices, int(len(indices)*(1 - self.vo_dropout_ratio)))).tolist()
        self.gps_indices = np.sort(
            random.sample(indices, int(len(indices)*(1 - self.gps_dropout_ratio)))).tolist()

    def get_control_input_by_index(self, index, setup):
        ax, ay, az = self.synced_df[['voxl_imu0_AX(m/s2)', 'voxl_imu0_AY(m/s2)', 'voxl_imu0_AZ(m/s2)']].iloc[index].values
        wx, wy, wz = self.synced_df[['voxl_imu0_GX(rad/s)', 'voxl_imu0_GY(rad/s)', 'voxl_imu0_GZ(rad/s)']].iloc[index].values
        if setup is SetupEnum.SETUP_1 or setup is SetupEnum.SETUP_2:
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
                    0, # not provided
                    wz
                ])
            else:
                u = np.array([
                    0, # not provided
                    wx, #wx
                    wz #wz
                ])
        return u
    
    def get_gps_measurement_by_index(
        self, 
        index, 
        setup=SetupEnum.SETUP_1,
        measurement_type=MeasurementDataEnum.ALL_DATA):

        if self.data_type == "indoor":
            return (None, None)
            
        if setup is SetupEnum.SETUP_1:
            return (None, None)
        
        gps_data = self.synced_df[self.px4_synched_gps_columns].iloc[index].values.reshape(-1, 1)
                        
        if measurement_type is MeasurementDataEnum.ALL_DATA:
            return (gps_data, None)
        elif measurement_type is MeasurementDataEnum.DROPOUT:
            if index not in self.gps_indices:
                return (None, None)
            return (gps_data, None)
        elif measurement_type is MeasurementDataEnum.COVARIANCE:
            error = 10. if index in self.vo_indices else 100.
            q = np.repeat(error ** 2, self.dimension)
            return (gps_data, np.eye(self.dimension) * q)
        
        return (None, None)
        
    def get_vo_measurement_by_index(
        self, 
        index, 
        measurement_type=MeasurementDataEnum.ALL_DATA):
        
        vo_data = self.synced_df[self.voxl_vo_T_columns].iloc[index].values.reshape(-1, 1)
        
        if measurement_type is MeasurementDataEnum.ALL_DATA:
            return (vo_data, None)
        elif measurement_type is MeasurementDataEnum.DROPOUT:
            if index not in self.vo_indices:
                return (None, None)
            return (vo_data, None)
        elif measurement_type is MeasurementDataEnum.COVARIANCE:
            error = 10.0 if index in self.vo_indices else 100.0
            q = np.repeat(error ** 2, self.dimension)
            return (vo_data, np.eye(self.dimension) * q)
        
        return (None, None)
    
    def get_trajectory_to_compare(self) -> np.ndarray:
        if self.data_type == "indoor":
            return self.synced_df[self.voxl_vo_T_columns].values.T
        return self.synced_df[['px4_gps_north', 'px4_gps_east', 'px4_gps_down']].values.T
    
if __name__ == "__main__":
    filename = 'uav_data_loader_debug.log'
    logging.basicConfig(filename=filename, level=logging.INFO)
    logging.FileHandler(filename, mode='w')
    
    
    # root_data_path = "../../example_data"
    # sequence_nr="log0001"
    
    # uncomment this line to load full data
    root_data_path = "../../data"
    sequence_nr="log0003"
    
    loader = UAV_DataLoader(root_path=root_data_path, sequence_nr=sequence_nr)
    config = loader.get_config(timestamp=0)
    print(config["voxl_vo"]["index"])
    print(loader.imu_config)
    loader.visualize_imu_data(title="Test")