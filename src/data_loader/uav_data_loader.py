import os
import sys
if __name__ == "__main__":
    sys.path.append('../../src')

import pandas as pd
from decimal import Decimal, getcontext
from utils import lla_to_ned

file_path = {
    "log0000": {
        "voxl": {
            "imu0": "imu0/data.csv",
            "imu1": "imu1/data.csv",
            "qvio": "qvio/data.csv",
        },
        "px4": {
            "imu0": "imu_combined/log0000_px4_imu0_combined.csv",
            "imu1": "imu_combined/log0000_px4_imu1_combined.csv",
            "gps": "13_07_41_sensor_gps_0.csv",
            "vo": "13_07_41_vehicle_visual_odometry_0.csv",
            "vehicle_odometry": "13_07_41_vehicle_odometry_0.csv",
            "imu0_bias": "13_07_41_estimator_sensor_bias_0.csv",
            "imu1_bias": "13_07_41_estimator_sensor_bias_1.csv"
        },
    },
    "log0001": {
        "voxl": {
            "imu0": "imu0/data.csv",
            "imu1": "imu1/data.csv",
            "qvio": "qvio/data.csv",
        },
        "px4": {
            "imu0": "imu_combined/log0001_px4_imu0_combined.csv",
            "imu1": "imu_combined/log0001_px4_imu1_combined.csv",
            "gps": "13_13_57_sensor_gps_0.csv",
            "vo": "13_13_57_vehicle_visual_odometry_0.csv",
            "vehicle_odometry": "13_13_57_vehicle_odometry_0.csv",
            "imu0_bias":"13_13_57_estimator_sensor_bias_0.csv",
            "imu1_bias": "13_13_57_estimator_sensor_bias_1.csv",
        },
    },
    "log0003": {
        "voxl": {
            "imu0": "imu0/data.csv",
            "imu1": "imu1/data.csv",
            "qvio": "qvio/data.csv",
        },
        "px4": {
            "imu0":  "imu_combined/log0003_px4_imu0_combined.csv",
            "imu1":  "imu_combined/log0003_px4_imu1_combined.csv",
            "gps": "13_16_27_sensor_gps_0.csv",
            "vo":  "13_16_27_vehicle_visual_odometry_0.csv",
            "vehicle_odometry": "13_16_27_vehicle_odometry_0.csv",
            "imu0_bias": "13_16_27_estimator_sensor_bias_0.csv",
            "imu1_bias": "13_16_27_estimator_sensor_bias_1.csv",
        },
    },
    "log0005": {
        "voxl": {
            "imu0": "imu0/data.csv",
            "imu1": "imu1/data.csv",
            "qvio": "qvio/data.csv",
        },
        "px4": {
            "imu0":  "imu_combined/log0005_px4_imu0_combined.csv",
            "imu1":  "imu_combined/log0005_px4_imu1_combined.csv",
            "gps": "log_145_UnknownDate_sensor_gps_0.csv",
            "vo":  "log_145_UnknownDate_vehicle_visual_odometry_0.csv",
            "vehicle_odometry": "log_145_UnknownDate_vehicle_odometry_0.csv",
            "imu0_bias": "log_145_UnknownDate_estimator_sensor_bias_0.csv",
            "imu1_bias": "log_145_UnknownDate_estimator_sensor_bias_1.csv",
        },
    }
}

class UAV_DataLoader:
    
    time_update = ['voxl_imu0', 'voxl_imu1', 'px4_imu0', 'px4_imu1', 'px4_imu0_bias', 'px4_imu1_bias']
    measurement_update = ['voxl_vo', 'px4_vo', 'px4_vehicle_odom', 'px4_gps']
    
    gps_pose_columns = ['lon', 'lat', 'alt']
    gps_ned_pose_columns = ['north', 'east', 'down']
    gps_pos_var_columns = ['eph', 'eph', 'epv'] # https://docs.px4.io/main/en/msg_docs/SensorGps.html
    vo_pos_columns = ['position[0]', 'position[1]', 'position[2]']
    vo_pos_var_columns = ['position_variance[0]', 'position_variance[1]', 'position_variance[2]']
    
    SENSOR_SLIP_THRESHOLD = 2 # 2 seconds
    
    def __init__(
        self,
        root_path,
        sequence_nr="log0000",
        version="v1"
        ):

        uav_root_path = os.path.join(root_path, "UAV", sequence_nr)
        
        uav_reference_path = os.path.join(root_path, f"UAV/{version}/timestamp_combined_{sequence_nr}.csv")
        
        uav_synced_data_path = os.path.join(root_path, f"UAV/{version}/{sequence_nr}_sync.csv")
        
        sensor_path = file_path[sequence_nr]

        if sequence_nr == "log0000":
            px4_root_path = os.path.join(uav_root_path, "px4/13_07_41")
        else:
            px4_root_path = os.path.join(uav_root_path, "px4")
        
        px4 = sensor_path["px4"]
        px4_imu0_path = os.path.join(px4_root_path, px4["imu0"])
        px4_imu1_path = os.path.join(px4_root_path, px4["imu1"])
        px4_gps_path = os.path.join(px4_root_path,  px4["gps"])
        px4_vo_path = os.path.join(px4_root_path,  px4["vo"])
        px4_vehicle_odom_path = os.path.join(px4_root_path,  px4["vehicle_odometry"])
        imu0_bias_path = os.path.join(px4_root_path,  px4["imu0_bias"])
        imu1_bias_path = os.path.join(px4_root_path,  px4["imu1_bias"])

        voxl_root_path = os.path.join(uav_root_path, "run/mpa")
        voxl = sensor_path["voxl"]
        voxl_imu0_path = os.path.join(voxl_root_path, voxl["imu0"])
        voxl_imu1_path = os.path.join(voxl_root_path, voxl["imu1"])
        voxl_qvio_path = os.path.join(voxl_root_path, voxl["qvio"])

        self.ref_df = pd.read_csv(uav_reference_path)
        self.px4_imu0_df = pd.read_csv(px4_imu0_path)
        self.px4_imu1_df = pd.read_csv(px4_imu1_path)
        self.px4_gps_df = self.init_gps_data(px4_gps_path=px4_gps_path)
        self.px4_vo_df = pd.read_csv(px4_vo_path)
        self.px4_vehicle_odom_df = pd.read_csv(px4_vehicle_odom_path)
        self.imu0_bias_df = pd.read_csv(imu0_bias_path)
        self.imu1_bias_df = pd.read_csv(imu1_bias_path)

        self.voxl_imu0_df = pd.read_csv(voxl_imu0_path)
        self.voxl_imu1_df = pd.read_csv(voxl_imu1_path)
        self.voxl_qvio_df = pd.read_csv(voxl_qvio_path)
        
        self.synced_df = pd.read_csv(uav_synced_data_path)

    def init_gps_data(self, px4_gps_path):
        # Load csv data
        df = pd.read_csv(px4_gps_path)
        # Convert integer lla into float lla
        df = self.int_lla_to_float_lla(df)
        # Compute North-East-Down coordinate
        return self.get_ned_coord(df)
    
    
    def int2float_lla(self, x):
        return float(Decimal(x / Decimal(10**(len(str(x)) - 2))))

    def int_lla_to_float_lla(self, df):
        getcontext().prec = 10
        df['lat'] = df['lat'].apply(lambda x: self.int2float_lla(x))
        df['lon'] = df['lon'].apply(lambda x: self.int2float_lla(x))
        df['alt'] = df['alt'].apply(lambda x: self.int2float_lla(x))
        return df
    
    def get_ned_coord(self, df):    
        origin = df[['lon', 'lat', 'alt']].iloc[0].values
        ned_pose = lla_to_ned(df[['lon', 'lat', 'alt']].values.T, origin).T
        df = pd.concat([
            df,
            pd.DataFrame(ned_pose, columns=['north', 'east', 'down'])
        ], axis=1)
        return df
    
    def get_config(self, timestamp):
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
            'df': self.imu0_bias_df,
        },
        'px4_imu1_bias': {
            'index': 0,
            'last_timestamp': timestamp,
            'columns': ['gyro_bias[0]', 'gyro_bias[1]', 'gyro_bias[2]', 'accel_bias[0]', 'accel_bias[1]', 'accel_bias[2]'],
            'synced_columns': ['px4_imu1_bias_gyro_bias[0]', 'px4_imu1_bias_gyro_bias[1]', 'px4_imu1_bias_gyro_bias[2]', 'px4_imu1_bias_accel_bias[0]', 'px4_imu1_bias_accel_bias[1]', 'px4_imu1_bias_accel_bias[2]'],
            'df': self.imu1_bias_df,
        },
        'voxl_vo': {
            'index': 0,
            'last_timestamp': timestamp,
            'columns': self.voxl_qvio_df.columns.tolist(),
            'synced_columns': ['voxl_vo_' + col for col in self.voxl_qvio_df.columns.tolist()],
            'df': self.voxl_qvio_df,
        },
        }
    
if __name__ == "__main__":
    
    root_example_data_path = "../../example_data"
    example_sequence_nr="log0000"
    
    loader = UAV_DataLoader(root_path=root_example_data_path, sequence_nr=example_sequence_nr)
    config = loader.get_config(timestamp=0)
    print(config["voxl_vo"]["index"])
    print(loader.synced_df.columns)